"""Фейки моделей и метрик для unit-тестов."""

import hashlib
from collections import Counter
from typing import Any

from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.traversal import Candidate


class RecordingMetrics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.processed_by: Counter[str] = Counter()
        self.buffer_sizes: list[tuple[str, int]] = []
        self.discover_blocked_by_node: dict[str, bool] = {}
        self.discover_runs: list[tuple[str, int, int]] = []

    def processed(self, outcome: str) -> None:
        self.processed_by[outcome] += 1

    def model_call(self, role: str, latency_s: float, ok: bool) -> None:
        self.counts[f"model_call:{role}:{ok}"] += 1

    def model_error(self, role: str) -> None:
        self.counts[f"model_error:{role}"] += 1

    def guard_result(self, path: str, passed: bool) -> None:
        self.counts[f"guard:{path}:{passed}"] += 1

    def buffer_size(self, node: str, size: int) -> None:
        self.buffer_sizes.append((node, size))

    def discover_run(self, node: str, created: int, rejected: int) -> None:
        self.discover_runs.append((node, created, rejected))

    def discover_blocked(self, node: str, blocked: bool) -> None:
        self.discover_blocked_by_node[node] = blocked

    def tree_rev(self, rev: int) -> None:
        self.counts["tree_rev"] = rev


Answer = str | bool | BaseException


class _Queue:
    """Ответы по очереди; исключение в очереди бросается. Пустая очередь - default."""

    def __init__(self, answers: list[Any], default: Any) -> None:
        self.answers = list(answers)
        self.default = default
        self.calls: list[tuple[str, Any]] = []

    def _next(self, text: str, arg: Any) -> Any:
        self.calls.append((text, arg))
        value = self.answers.pop(0) if self.answers else self.default
        if isinstance(value, BaseException):
            raise value
        return value


class FakeDecision(_Queue):
    def __init__(self, answers: list[str | BaseException], default: str = "other") -> None:
        super().__init__(answers, default)

    async def decide(self, text: str, candidates: list[Candidate]) -> str:
        return str(self._next(text, [c.id for c in candidates]))


class FakeGuard(_Queue):
    def __init__(self, answers: list[bool | BaseException], default: bool = True) -> None:
        super().__init__(answers, default)

    async def check(self, text: str, question: str) -> bool:
        return bool(self._next(text, question))


class FakeText:
    def __init__(self) -> None:
        self.next_proposal: TopicProposal | None = None
        self.same: bool = False
        self.calls: Counter[str] = Counter()

    async def propose_topic(
        self,
        samples: list[str],
        parent_path: str,
        siblings: list[Candidate],
        language: str,
        level: int,
        max_depth: int,
    ) -> TopicProposal:
        from llm_sorter.ports.models import InvalidModelOutput

        self.calls["propose_topic"] += 1
        if self.next_proposal is None:
            raise InvalidModelOutput("FakeText.next_proposal is not set")
        return self.next_proposal

    async def same_topic(self, a: Candidate, b: Candidate) -> bool:
        self.calls["same_topic"] += 1
        return self.same


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 8) -> None:
        self.mapping = mapping or {}
        self.dim = dim
        self.calls = 0

    def _fallback(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[: self.dim]]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.mapping.get(t) or self._fallback(t) for t in texts]
