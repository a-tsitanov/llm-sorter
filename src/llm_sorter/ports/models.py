from typing import Protocol

from llm_sorter.domain.discover import TopicGroup, TopicProposal
from llm_sorter.domain.traversal import Candidate


class InvalidModelOutput(Exception):
    """Модель вернула id вне списка кандидатов или невалидный JSON."""


class DecisionModel(Protocol):
    async def decide(self, text: str, candidates: list[Candidate]) -> str: ...


class GuardModel(Protocol):
    async def check(self, text: str, question: str) -> bool: ...


class TextLLM(Protocol):
    async def propose_topic(
        self,
        samples: list[str],
        parents: list[Candidate],
        siblings: list[Candidate],
        language: str,
        level: int,
        max_depth: int,
    ) -> TopicProposal:
        """parents - цепочка от темы уровня 1 до родителя ([] - корень); level - глубина новой
        темы (дети root - 1), max_depth - предел глубины ветки."""
        ...

    async def group_clusters(
        self,
        clusters: list[list[str]],
        parents: list[Candidate],
        siblings: list[Candidate],
        language: str,
        level: int,
        max_depth: int,
    ) -> list[TopicGroup]:
        """Темы уровня level по образцам всех кластеров узла сразу; кластер - не более чем
        в одной теме, неподходящие кластеры не входят ни в одну."""
        ...

    async def same_topic(self, a: Candidate, b: Candidate) -> bool: ...


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
