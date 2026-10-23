from typing import Protocol

from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.traversal import Candidate


class InvalidModelOutput(Exception):
    """Модель вернула id вне списка кандидатов или невалидный JSON."""


class DecisionModel(Protocol):
    async def decide(self, text: str, candidates: list[Candidate]) -> str: ...


class GuardModel(Protocol):
    async def check(self, text: str, question: str) -> bool: ...


class TextLLM(Protocol):
    async def translate(self, text: str, to: str) -> str: ...

    async def summarize(self, text: str, language: str) -> str: ...

    async def propose_topic(
        self,
        samples: list[str],
        parent_path: str,
        siblings: list[Candidate],
        language: str,
        level: int,
        max_depth: int,
    ) -> TopicProposal:
        """level - глубина новой темы (дети root - 1), max_depth - предел глубины ветки."""
        ...

    async def same_topic(self, a: Candidate, b: Candidate) -> bool: ...


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
