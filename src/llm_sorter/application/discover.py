"""Discover: кластеризация текстов узла и предложение новых подтем (без хранилищ)."""

import asyncio
from dataclasses import dataclass

import numpy as np

from llm_sorter.application.clustering import cluster, cosine, diverse_samples
from llm_sorter.domain.discover import (
    TopicProposal,
    dedup_verdict,
    fix_guard,
    proposal_is_valid,
    slots_left,
)
from llm_sorter.domain.traversal import Candidate, candidates_for
from llm_sorter.domain.tree import OTHER_ID, ROOT_ID, Topic, Tree
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.models import Embedder, InvalidModelOutput, TextLLM

EMBED_BATCH = 16
# буфер узла кластеризуется заново на каждом discover: эмбеддинги текстов кешируются
EMBED_CACHE_SIZE = 100_000
# лист делится, только если получилось столько подтем: одна подтема ничего не даёт
MIN_LEAF_SPLIT = 2
# группировка кластеров (промежуточные уровни): образцов на кластер и символов на образец -
# модель видит все кластеры узла в одном промпте
GROUP_SAMPLES = 5
GROUP_SAMPLE_CHARS = 500
GROUP_BATCH_SIZE = 12


@dataclass(frozen=True)
class Accepted:
    topic: Topic
    members: list[int]  # индексы в texts


@dataclass(frozen=True)
class DiscoverResult:
    accepted: list[Accepted]
    considered: list[int]  # все индексы, не попавшие в принятые темы
    rejected: int


def _topic_text(title: str, description: str) -> str:
    return f"{title}. {description}"


def _as_candidate(p: TopicProposal) -> Candidate:
    return Candidate(id=p.id, title=p.title, description=p.description, examples=list(p.examples))


def _as_candidate_topic(t: Topic) -> Candidate:
    return Candidate(id=t.id, title=t.title, description=t.description, examples=list(t.examples))


def _free_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


class _Run:
    """Один прогон discover узла: общие правила приёма предложенной темы для обоих режимов."""

    def __init__(
        self,
        svc: "DiscoverService",
        tree: Tree,
        node_id: str,
        siblings: list[Candidate],
        parents: list[Candidate],
        leaf: Candidate,
    ) -> None:
        self._svc = svc
        self._tree = tree
        self._node_id = node_id
        self.siblings = siblings
        self.parents = parents
        self._dedup_against = siblings or [leaf]
        self._neighbours: list[tuple[Candidate, list[float]]] | None = None  # лениво
        # id - сегмент routing key: уникален во всём дереве
        self._taken = set(tree.topics) | {OTHER_ID, ROOT_ID}
        self.accepted: list[Accepted] = []
        self.rejected = 0

    def full(self) -> bool:
        return slots_left(self._tree, self._node_id) <= len(self.accepted)

    def shown(self) -> list[Candidate]:
        """Соседи и уже принятые в этом прогоне: иначе модель предлагает их снова."""
        return [*self.siblings, *(_as_candidate_topic(a.topic) for a in self.accepted)]

    async def offer(self, p: TopicProposal, members: list[int]) -> None:
        svc, s = self._svc, self._tree.settings.discover
        if not proposal_is_valid(p):
            svc._metrics.model_error("discover")
            return
        if p.covered_by in {c.id for c in self.shown()}:
            self.rejected += 1  # образцы уже покрывает соседняя тема - перекрытие не нужно
            return
        p = fix_guard(p)
        if self._neighbours is None:
            vecs = await svc._embed(
                [_topic_text(c.title, c.description) for c in self._dedup_against]
            )
            self._neighbours = list(zip(self._dedup_against, vecs, strict=True))
        [vec] = await svc._embed([_topic_text(p.title, p.description)])
        best, sim = max(((c, cosine(vec, v)) for c, v in self._neighbours), key=lambda cs: cs[1])
        verdict = dedup_verdict(sim, s)
        if verdict == "dup" or (
            verdict == "ask" and await svc._text.same_topic(_as_candidate(p), best)
        ):
            self.rejected += 1
            return
        topic_id = _free_id(p.id, self._taken)
        self._taken.add(topic_id)
        topic = Topic(
            id=topic_id,
            parent_id=self._node_id,
            title=p.title,
            description=p.description,
            examples=list(p.examples),
            guard=p.guard,
            origin="discovered",
        )
        self.accepted.append(Accepted(topic, members))
        self._neighbours.append((_as_candidate(p), vec))


class DiscoverService:
    def __init__(self, embedder: Embedder, text_llm: TextLLM, metrics: Metrics) -> None:
        self._embedder = embedder
        self._text = text_llm
        self._metrics = metrics
        self._cache: dict[str, list[float]] = {}  # порядок вставки = порядок вытеснения

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for i in range(0, len(missing), EMBED_BATCH):
            batch = missing[i : i + EMBED_BATCH]
            self._cache.update(zip(batch, await self._embedder.embed(batch), strict=True))
        out = [self._cache[t] for t in texts]
        while len(self._cache) > EMBED_CACHE_SIZE:
            del self._cache[next(iter(self._cache))]
        return out

    async def propose(self, tree: Tree, node_id: str, texts: list[str]) -> DiscoverResult:
        everything = list(range(len(texts)))
        if tree.depth(node_id) >= tree.max_depth_for(node_id) or slots_left(tree, node_id) <= 0:
            return DiscoverResult([], everything, 0)
        s = tree.settings.discover
        valid = [i for i in everything if texts[i].strip()]  # пустые не кластеризуются
        vectors = np.array(await self._embed([texts[i] for i in valid]), dtype=np.float64)
        is_leaf = tree.is_leaf(node_id)
        node = tree.get(node_id)
        run = _Run(
            self,
            tree,
            node_id,
            siblings=[c for c in candidates_for(tree, node_id) if c.id != OTHER_ID],
            parents=[_as_candidate_topic(tree.get(i)) for i in tree.path(node_id)],
            # для листа подтемы сравниваются с самим листом: подтема-копия листа бесполезна
            leaf=Candidate(id=node.id, title=node.title, description=node.description),
        )
        groups = await asyncio.to_thread(cluster, vectors, s.min_cluster_size) if len(valid) else []
        members = [[valid[i] for i in g] for g in groups]
        level = tree.depth(node_id) + 1
        if groups and level < tree.max_depth_for(node_id):
            # промежуточный уровень: модель видит все кластеры сразу и собирает из них
            # широкие непересекающиеся темы, а не тему на каждый узкий кластер
            chars = min(s.sample_chars, GROUP_SAMPLE_CHARS)
            samples = [
                [texts[valid[i]][:chars] for i in diverse_samples(vectors, g, GROUP_SAMPLES)]
                for g in groups
            ]
            for start in range(0, len(samples), GROUP_BATCH_SIZE):
                if run.full():
                    break
                batch = samples[start : start + GROUP_BATCH_SIZE]
                try:
                    proposed = await self._text.group_clusters(
                        batch,
                        run.parents,
                        run.shown(),
                        tree.settings.language,
                        level=level,
                        max_depth=tree.max_depth_for(node_id),
                    )
                except InvalidModelOutput:
                    self._metrics.model_error("discover")
                    continue
                used: set[int] = set()
                picked = []
                for p in proposed:
                    own = [
                        c
                        for c in dict.fromkeys(p.clusters)
                        if 0 <= c < len(batch) and c not in used
                    ]
                    used.update(own)
                    if own:
                        picked.append((p, [i for c in own for i in members[start + c]]))
                for p, m in sorted(picked, key=lambda pm: -len(pm[1])):
                    if run.full():
                        break
                    await run.offer(p, m)
        else:
            for g, m in zip(groups, members, strict=True):
                if run.full():
                    break
                samples_one = [
                    texts[valid[i]][: s.sample_chars]
                    for i in diverse_samples(vectors, g, s.samples_per_cluster)
                ]
                try:
                    one = await self._text.propose_topic(
                        samples_one,
                        run.parents,
                        run.shown(),
                        tree.settings.language,
                        level=level,
                        max_depth=tree.max_depth_for(node_id),
                    )
                except InvalidModelOutput:
                    self._metrics.model_error("discover")
                    continue
                await run.offer(one, m)
        accepted, rejected = run.accepted, run.rejected

        if is_leaf and len(accepted) < MIN_LEAF_SPLIT:
            accepted = []
        used = {i for a in accepted for i in a.members}
        return DiscoverResult(accepted, [i for i in everything if i not in used], rejected)
