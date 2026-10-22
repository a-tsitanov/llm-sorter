"""In-memory реализации портов для unit-тестов (семантика совпадает с реальными адаптерами)."""

import builtins
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from llm_sorter.domain.classification import (
    Classification,
    FinalClassificationExists,
    Status,
    Transcript,
)
from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.traversal import Candidate
from llm_sorter.domain.tree import RuleTree
from llm_sorter.ports.storage import MaterialItem, VersionConflict


class FakeRules:
    def __init__(self, initial: RuleTree | None = None) -> None:
        self.versions: dict[int, RuleTree] = {}
        self.sources: dict[int, str] = {}
        self._active: int | None = None
        self.conflict_once = False  # одна имитация гонки публикаций
        if initial is not None:
            self.versions[1] = initial.model_copy(update={"version": 1})
            self._active = 1

    @property
    def active_version(self) -> int | None:
        return self._active

    async def active(self) -> RuleTree:
        if self._active is None:
            raise LookupError("no active rules")
        return self.versions[self._active]

    async def get(self, version: int) -> RuleTree:
        try:
            return self.versions[version]
        except KeyError:
            raise LookupError(f"unknown version {version}") from None

    async def publish(
        self,
        tree: RuleTree,
        base_version: int | None,
        source: Literal["manual", "discover", "rollback"],
    ) -> int:
        if self.conflict_once:
            self.conflict_once = False
            raise VersionConflict("simulated race")
        if base_version != self._active:
            raise VersionConflict(f"base={base_version}, active={self._active}")
        n = max(self.versions, default=0) + 1
        self.versions[n] = tree.model_copy(update={"version": n})
        self.sources[n] = source
        self._active = n
        return n


class FakeTranscripts:
    def __init__(self) -> None:
        self.items: dict[str, Transcript] = {}

    async def put(self, t: Transcript) -> bool:
        if t.id in self.items:
            return False
        self.items[t.id] = t
        return True

    async def get(self, file_id: str) -> Transcript:
        try:
            return self.items[file_id]
        except KeyError:
            raise LookupError(file_id) from None


class FakeClassifications:
    def __init__(self) -> None:
        self.items: dict[str, Classification] = {}
        self.saves = 0

    async def get(self, file_id: str) -> Classification | None:
        return self.items.get(file_id)

    async def save(self, c: Classification) -> None:
        existing = self.items.get(c.file_id)
        if existing is not None and existing.final:
            raise FinalClassificationExists(c.file_id)
        self.items[c.file_id] = c
        self.saves += 1


@dataclass
class _Entry:
    added_at: datetime
    node_uid: str
    file_id: str
    kind: Literal["other", "leaf"]


class FakeMaterial:
    """count_new/due_nodes считают оба вида (other и leaf); other_size — только other.
    Время добавления строго монотонно, mark_discovered(at) учитывает отсечку."""

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._last = datetime.min.replace(tzinfo=UTC)
        self._marked: dict[str, datetime] = {}

    def _next(self) -> datetime:
        self._last = max(datetime.now(UTC), self._last + timedelta(microseconds=1))
        return self._last

    async def add(self, node_uid: str, file_id: str, kind: Literal["other", "leaf"]) -> None:
        for e in self._entries:
            if e.node_uid == node_uid and e.file_id == file_id:
                e.kind = kind
                return
        self._entries.append(_Entry(self._next(), node_uid, file_id, kind))

    async def list(self, node_uid: str) -> list[MaterialItem]:
        return [MaterialItem(e.file_id, e.kind) for e in self._entries if e.node_uid == node_uid]

    async def remove(self, node_uid: str, file_id: str) -> None:
        self._entries = [
            e for e in self._entries if not (e.node_uid == node_uid and e.file_id == file_id)
        ]

    async def move_other(self, from_uid: str, to_uid: str) -> None:
        present = {e.file_id for e in self._entries if e.node_uid == to_uid}
        moved = [e for e in self._entries if e.node_uid == from_uid and e.kind == "other"]
        for e in moved:
            self._entries.remove(e)
            if e.file_id not in present:
                self._entries.append(_Entry(self._next(), to_uid, e.file_id, "other"))

    async def count_new(self, node_uid: str) -> int:
        mark = self._marked.get(node_uid)
        return sum(
            1
            for e in self._entries
            if e.node_uid == node_uid and (mark is None or e.added_at > mark)
        )

    async def mark_discovered(self, node_uid: str, at: datetime | None = None) -> None:
        self._marked[node_uid] = at if at is not None else self._next()

    async def due_nodes(self, threshold: int) -> builtins.list[str]:
        uids = list(dict.fromkeys(e.node_uid for e in self._entries))
        return [u for u in uids if await self.count_new(u) >= threshold]

    async def other_size(self, node_uid: str) -> int:
        return sum(1 for e in self._entries if e.node_uid == node_uid and e.kind == "other")


class FakeEmbeddings:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    async def get_many(self, file_ids: list[str]) -> dict[str, list[float]]:
        return {f: self.vectors[f] for f in file_ids if f in self.vectors}

    async def put_many(self, vectors: dict[str, list[float]]) -> None:
        self.vectors.update(vectors)


class FakeStorage:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def write(self, rel_path: str, content: str) -> str:
        self.files[rel_path] = content
        return rel_path


@dataclass(frozen=True)
class NotifierCall:
    channel: str
    text: str
    attachment_name: str
    attachment: str


class FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[NotifierCall] = []

    async def send(self, channel: str, text: str, attachment_name: str, attachment: str) -> None:
        self.calls.append(NotifierCall(channel, text, attachment_name, attachment))


class FakeStarter:
    def __init__(self) -> None:
        self.classify: list[str] = []
        self.discover: list[str] = []
        self.reprocess: list[tuple[str, str, int]] = []

    async def start_classify(self, file_id: str) -> bool:
        if file_id in self.classify:
            return False
        self.classify.append(file_id)
        return True

    async def start_discover(self, node_uid: str) -> bool:
        if node_uid in self.discover:
            return False
        self.discover.append(node_uid)
        return True

    async def start_reprocess(self, file_id: str, node_uid: str, version: int) -> bool:
        key = (file_id, node_uid, version)
        if key in self.reprocess:
            return False
        self.reprocess.append(key)
        return True


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
        self, samples: list[str], parent_path: str, siblings: list[Candidate], language: str
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
