"""Discover: кластеризация материала узла и предложение новых подтем."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from llm_sorter.application.clustering import cluster, cosine, nearest_to_centroid
from llm_sorter.application.rules import RulesService
from llm_sorter.domain.discover import (
    TopicProposal,
    dedup_verdict,
    discover_blocked,
    proposal_is_valid,
    should_discover,
    slots_left,
)
from llm_sorter.domain.traversal import Candidate, candidates_for
from llm_sorter.domain.tree import OTHER_ID, RuleTree
from llm_sorter.domain.validation import TreeValidationError
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.models import Embedder, InvalidModelOutput, TextLLM
from llm_sorter.ports.storage import (
    EmbeddingCache,
    MaterialRepository,
    RulesRepository,
    TranscriptStore,
)

EMBED_BATCH = 16

Progress = Callable[[], None]


@dataclass(frozen=True)
class DiscoverResult:
    version: int | None
    created: list[str]
    rejected: int


def _topic_text(title: str, description: str) -> str:
    return f"{title}. {description}"


def _as_candidate(p: TopicProposal) -> Candidate:
    return Candidate(id=p.id, title=p.title, description=p.description, examples=list(p.examples))


def _free_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


class DiscoverService:
    def __init__(
        self,
        rules: RulesRepository,
        rules_service: RulesService,
        material: MaterialRepository,
        transcripts: TranscriptStore,
        cache: EmbeddingCache,
        embedder: Embedder,
        text_llm: TextLLM,
        metrics: Metrics,
    ) -> None:
        self._rules = rules
        self._rules_service = rules_service
        self._material = material
        self._transcripts = transcripts
        self._cache = cache
        self._embedder = embedder
        self._text = text_llm
        self._metrics = metrics

    async def _eligible(self, tree: RuleTree, uid: str) -> bool:
        try:
            tree.find(uid)
        except KeyError:  # узел архивный
            return False
        return should_discover(tree, uid, await self._material.count_new(uid))

    async def due_nodes(self) -> list[str]:
        """Узлы, набравшие порог нового материала и пригодные для discover."""
        tree = await self._rules.active()
        due = await self._material.due_nodes(tree.settings.discover.buffer_threshold)
        result: list[str] = []
        for u in due:
            if await self._eligible(tree, u):
                result.append(u)
            try:
                await self._report_blocked(tree, u)
            except KeyError:  # узел архивный
                pass
        return result

    async def _report_blocked(self, tree: RuleTree, uid: str) -> None:
        blocked = discover_blocked(tree, uid, await self._material.other_size(uid))
        self._metrics.discover_blocked(tree.path_of(uid) or "root", blocked)

    async def _embed(self, texts: list[str], progress: Progress) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            out.extend(await self._embedder.embed(texts[i : i + EMBED_BATCH]))
            progress()
        return out

    async def _file_vectors(
        self, file_ids: list[str], texts: dict[str, str], progress: Progress
    ) -> np.ndarray:
        cached = await self._cache.get_many(file_ids)
        missing = [f for f in file_ids if f not in cached]
        if missing:
            vecs = await self._embed([texts[f] for f in missing], progress)
            fresh = dict(zip(missing, vecs, strict=True))
            await self._cache.put_many(fresh)
            cached.update(fresh)
        return np.array([cached[f] for f in file_ids], dtype=np.float64)

    async def run(self, node_uid: str, progress: Progress | None = None) -> DiscoverResult:
        """progress вызывается между батчами эмбеддингов и кластерами (heartbeat)."""
        tick: Progress = progress or (lambda: None)
        tree = await self._rules.active()
        if not await self._eligible(tree, node_uid):
            return DiscoverResult(None, [], 0)
        s = tree.settings.discover
        parent_path = tree.path_of(node_uid)

        # отсечка до чтения материала: добавленное во время прогона останется новым
        cutoff = datetime.now(UTC)
        all_ids = list(dict.fromkeys(item.file_id for item in await self._material.list(node_uid)))
        texts = {f: (await self._transcripts.get(f)).text for f in all_ids}
        file_ids = [f for f in all_ids if texts[f].strip()]  # пустые не кластеризуются
        vectors = await self._file_vectors(file_ids, texts, tick)

        siblings = [c for c in candidates_for(tree, node_uid) if c.id != OTHER_ID]
        neighbours: list[tuple[Candidate, list[float]]] | None = None  # лениво
        taken = {c.id for c in siblings} | {OTHER_ID}

        accepted: list[TopicProposal] = []
        rejected = 0
        for group in cluster(vectors, s.min_cluster_size):
            if slots_left(tree, node_uid) <= len(accepted):
                break
            tick()
            samples = [
                texts[file_ids[i]][: s.sample_chars]
                for i in nearest_to_centroid(vectors, group, s.samples_per_cluster)
            ]
            try:
                p = await self._text.propose_topic(
                    samples, parent_path, siblings, tree.settings.language
                )
            except InvalidModelOutput:
                self._metrics.model_error("discover")
                continue
            if not proposal_is_valid(p):  # иначе validate_tree отклонит всю пачку
                self._metrics.model_error("discover")
                continue
            if neighbours is None:
                # живые соседи (кандидаты без other) и их эмбеддинги для дедупа
                vecs = await self._embed(
                    [_topic_text(c.title, c.description) for c in siblings], tick
                )
                neighbours = list(zip(siblings, vecs, strict=True))
            [vec] = await self._embed([_topic_text(p.title, p.description)], tick)
            if neighbours:
                best, sim = max(
                    ((cand, cosine(vec, v)) for cand, v in neighbours), key=lambda cs: cs[1]
                )
                verdict = dedup_verdict(sim, s)
                if verdict == "dup" or (
                    verdict == "ask" and await self._text.same_topic(_as_candidate(p), best)
                ):
                    rejected += 1
                    continue
            p = p.model_copy(update={"id": _free_id(p.id, taken)})
            taken.add(p.id)
            accepted.append(p)
            neighbours.append((_as_candidate(p), vec))

        version: int | None = None
        if accepted:
            try:
                version = (await self._rules_service.add_discovered(node_uid, accepted)).version
            except TreeValidationError:
                # не зацикливаться на узле: прогон считается неудачей модели
                self._metrics.model_error("discover")
                accepted = []
        await self._material.mark_discovered(node_uid, at=cutoff)
        self._metrics.buffer_size(
            tree.other_path(node_uid), await self._material.other_size(node_uid)
        )
        self._metrics.discover_run(parent_path or "root", len(accepted), rejected)
        await self._report_blocked(await self._rules.active(), node_uid)
        return DiscoverResult(version, [p.id for p in accepted], rejected)
