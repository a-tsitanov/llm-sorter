"""Фейки моделей и метрик для unit-тестов."""

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
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


DecisionAnswers = Mapping[Any, str | list[str]] | Callable[[str, list[Candidate]], str]


class ScriptedDecision:
    """Ответ по ключу: tuple id кандидатов (или строка "a,b,other"). Список — очередь,
    последний элемент повторяется. Нет ключа — "other"."""

    def __init__(self, answers: DecisionAnswers | None = None) -> None:
        self._answers = answers if answers is not None else {}
        self.calls = 0
        self.seen: list[tuple[str, ...]] = []

    async def decide(self, text: str, candidates: list[Candidate]) -> str:
        self.calls += 1
        ids = tuple(c.id for c in candidates)
        self.seen.append(ids)
        if callable(self._answers):
            return self._answers(text, candidates)
        for key in (ids, ",".join(ids)):
            if key in self._answers:
                value = self._answers[key]
                if isinstance(value, list):
                    return value.pop(0) if len(value) > 1 else value[0]
                return value
        return "other"


class ScriptedGuard:
    def __init__(self, default: bool = True, answers: Mapping[str, bool] | None = None) -> None:
        self.default = default
        self.answers = dict(answers or {})
        self.calls = 0

    async def check(self, text: str, question: str) -> bool:
        self.calls += 1
        return self.answers.get(question, self.default)


class FakeText:
    def __init__(self) -> None:
        self.next_proposal: TopicProposal | None = None
        self.same: bool = False
        self.calls: Counter[str] = Counter()

    async def translate(self, text: str, to: str) -> str:
        self.calls["translate"] += 1
        return f"[{to}] {text}"

    async def summarize(self, text: str, language: str) -> str:
        self.calls["summarize"] += 1
        return "summary: " + text[:50]

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
