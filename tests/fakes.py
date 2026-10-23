"""Фейки моделей и метрик для unit-тестов."""

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from llm_sorter.domain.classification import Status
from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.traversal import Candidate


class RecordingMetrics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.buffer_sizes: list[tuple[str, int]] = []
        self.discover_blocked_by_path: dict[str, bool] = {}

    def file_classified(self, path: str, status: Status, version: int) -> None:
        self.counts["file_classified"] += 1

    def model_call(self, role: str, latency_s: float, ok: bool) -> None:
        self.counts["model_call"] += 1

    def model_error(self, role: str) -> None:
        self.counts[f"model_error:{role}"] += 1

    def guard_result(self, path: str, passed: bool) -> None:
        self.counts["guard_result"] += 1

    def action_done(self, action_type: str, ok: bool, latency_s: float) -> None:
        self.counts["action_done"] += 1

    def buffer_size(self, path: str, size: int) -> None:
        self.counts["buffer_size"] += 1
        self.buffer_sizes.append((path, size))

    def discover_run(self, path: str, created: int, rejected: int) -> None:
        self.counts["discover_run"] += 1

    def reprocess_moved(self, from_path: str, to_path: str) -> None:
        self.counts["reprocess_moved"] += 1

    def discover_blocked(self, path: str, blocked: bool) -> None:
        self.discover_blocked_by_path[path] = blocked

    def topic_move_attempt(self) -> None:
        self.counts["topic_move_attempt"] += 1

    def active_version(self, v: int) -> None:
        self.counts["active_version"] += 1

    def rules_published(self, source: str) -> None:
        self.counts["rules_published"] += 1


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
