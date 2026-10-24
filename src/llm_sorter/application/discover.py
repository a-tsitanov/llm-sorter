"""Discover: кластеризация текстов узла и предложение новых подтем (без хранилищ)."""

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

        siblings = [c for c in candidates_for(tree, node_id) if c.id != OTHER_ID]
        parents = [_as_candidate_topic(tree.get(i)) for i in tree.path(node_id)]
        # для листа подтемы сравниваются с самим листом: подтема-копия листа бесполезна
        dedup_against = siblings or [
            Candidate(id=node.id, title=node.title, description=node.description)
        ]
        neighbours: list[tuple[Candidate, list[float]]] | None = None  # лениво
        # id - сегмент routing key: уникален во всём дереве
        taken = set(tree.topics) | {OTHER_ID, ROOT_ID}

        accepted: list[Accepted] = []
        rejected = 0
        for group in cluster(vectors, s.min_cluster_size) if len(valid) else []:
            if slots_left(tree, node_id) <= len(accepted):
                break
            samples = [
                texts[valid[i]][: s.sample_chars]
                for i in diverse_samples(vectors, group, s.samples_per_cluster)
            ]
            # и уже принятые в этом прогоне: иначе модель предлагает их снова
            shown = [*siblings, *(_as_candidate_topic(a.topic) for a in accepted)]
            try:
                p = await self._text.propose_topic(
                    samples,
                    parents,
                    shown,
                    tree.settings.language,
                    level=tree.depth(node_id) + 1,
                    max_depth=tree.max_depth_for(node_id),
                )
            except InvalidModelOutput:
                self._metrics.model_error("discover")
                continue
            if not proposal_is_valid(p):
                self._metrics.model_error("discover")
                continue
            if p.covered_by in {c.id for c in shown}:
                rejected += 1  # образцы уже покрывает соседняя тема - перекрытие не нужно
                continue
            p = fix_guard(p)
            if neighbours is None:
                vecs = await self._embed(
                    [_topic_text(c.title, c.description) for c in dedup_against]
                )
                neighbours = list(zip(dedup_against, vecs, strict=True))
            [vec] = await self._embed([_topic_text(p.title, p.description)])
            best, sim = max(((c, cosine(vec, v)) for c, v in neighbours), key=lambda cs: cs[1])
            verdict = dedup_verdict(sim, s)
            if verdict == "dup" or (
                verdict == "ask" and await self._text.same_topic(_as_candidate(p), best)
            ):
                rejected += 1
                continue
            p = p.model_copy(update={"id": _free_id(p.id, taken)})
            taken.add(p.id)
            topic = Topic(
                id=p.id,
                parent_id=node_id,
                title=p.title,
                description=p.description,
                examples=list(p.examples),
                guard=p.guard,
                origin="discovered",
            )
            accepted.append(Accepted(topic, [valid[i] for i in group]))
            neighbours.append((_as_candidate(p), vec))

        if is_leaf and len(accepted) < MIN_LEAF_SPLIT:
            accepted = []
        used = {i for a in accepted for i in a.members}
        return DiscoverResult(accepted, [i for i in everything if i not in used], rejected)
