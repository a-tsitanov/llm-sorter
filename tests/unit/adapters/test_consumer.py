import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_sorter.adapters.inbound import rabbit
from llm_sorter.adapters.inbound.poller import run_poller
from llm_sorter.adapters.inbound.rabbit import handle_message
from llm_sorter.adapters.inbound.stub_source import LocalDirSource
from llm_sorter.domain.classification import Transcript
from tests.fakes import FakeStarter, FakeTranscripts

now = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def transcripts() -> FakeTranscripts:
    return FakeTranscripts()


@pytest.fixture
def starter() -> FakeStarter:
    return FakeStarter()


async def test_handle_message_twice_starts_once(
    transcripts: FakeTranscripts, starter: FakeStarter
) -> None:
    body = Transcript(id="x", text="t", received_at=now).model_dump_json().encode()
    await handle_message(body, transcripts, starter)
    await handle_message(body, transcripts, starter)  # FakeStarter вернёт False, исключения нет
    assert starter.classify == ["x"]
    assert transcripts.items["x"].text == "t"


async def test_handle_message_invalid_body_raises_validation(
    transcripts: FakeTranscripts, starter: FakeStarter
) -> None:
    with pytest.raises(ValidationError):
        await handle_message(b"{broken", transcripts, starter)
    with pytest.raises(ValidationError):
        await handle_message(b'{"id": "x"}', transcripts, starter)
    assert starter.classify == [] and transcripts.items == {}


class ListSource:
    def __init__(self, batches: list[list[Transcript]]) -> None:
        self.batches = batches

    async def fetch_new(self) -> list[Transcript]:
        if not self.batches:
            return []
        batch = self.batches.pop(0)
        if not batch:
            raise RuntimeError("source down")
        return batch

    async def ack(self, ids: list[str]) -> None:
        pass


class ListPublisher:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def publish(self, t: Transcript) -> None:
        self.sent.append(t.id)


async def test_poller_publishes_and_survives_errors() -> None:
    a = Transcript(id="a", text="a", received_at=now)
    b = Transcript(id="b", text="b", received_at=now)
    src = ListSource([[a], [], [b]])  # пустой батч имитирует сбой источника
    pub = ListPublisher()
    stop = asyncio.Event()
    task = asyncio.create_task(run_poller(src, pub, 0.01, stop))
    for _ in range(200):
        if pub.sent == ["a", "b"]:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, 1)
    assert pub.sent == ["a", "b"]


class FlakyPublisher(ListPublisher):
    """Падает на заданных по счёту вызовах publish (с 1)."""

    def __init__(self, fail_on: set[int]) -> None:
        super().__init__()
        self.fail_on = fail_on
        self.attempts = 0

    async def publish(self, t: Transcript) -> None:
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise RuntimeError("broker down")
        await super().publish(t)


async def _run_ticks(src: LocalDirSource, pub: ListPublisher, until: int) -> None:
    stop = asyncio.Event()
    task = asyncio.create_task(run_poller(src, pub, 0.01, stop))
    for _ in range(200):
        if len(pub.sent) >= until or getattr(pub, "attempts", 0) >= until:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, 1)


def _inbox(tmp_path: Path, names: list[str]) -> LocalDirSource:
    inbox = tmp_path / "in"
    inbox.mkdir()
    for n in names:
        (inbox / f"{n}.txt").write_text(n)
    return LocalDirSource(inbox, tmp_path / "seen.json")


async def test_poller_failed_publish_marks_only_published(tmp_path: Path) -> None:
    src = _inbox(tmp_path, ["a", "b", "c"])
    pub = FlakyPublisher(fail_on=set(range(2, 400)))
    await _run_ticks(src, pub, until=3)
    assert pub.sent == ["a"]
    assert [t.id for t in await src.fetch_new()] == ["b", "c"]


async def test_poller_retries_rest_of_batch_next_tick(tmp_path: Path) -> None:
    src = _inbox(tmp_path, ["a", "b", "c"])
    pub = FlakyPublisher(fail_on={2})
    await _run_ticks(src, pub, until=3)
    assert pub.sent == ["a", "b", "c"]
    assert await src.fetch_new() == []


async def test_poller_cold_start_broker_down_marks_nothing(tmp_path: Path) -> None:
    src = _inbox(tmp_path, ["a", "b"])
    pub = FlakyPublisher(fail_on=set(range(1, 400)))
    await _run_ticks(src, pub, until=3)
    assert pub.sent == []
    assert [t.id for t in await src.fetch_new()] == ["a", "b"]


class FakeMessage:
    def __init__(self, body: bytes, headers: dict | None = None) -> None:
        self.body = body
        self.headers = headers or {}
        self.message_id = "m1"
        self.calls: list[tuple[str, bool | None]] = []

    async def ack(self) -> None:
        self.calls.append(("ack", None))

    async def reject(self, requeue: bool = False) -> None:
        self.calls.append(("reject", requeue))

    async def nack(self, requeue: bool = True) -> None:
        self.calls.append(("nack", requeue))


class BrokenStarter(FakeStarter):
    async def start_classify(self, file_id: str) -> bool:
        raise RuntimeError("temporal down")


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rabbit, "NACK_BACKOFF_S", 0)


class Republisher:
    def __init__(self, broken: bool = False) -> None:
        self.sent: list[tuple[bytes, int]] = []
        self.broken = broken

    async def __call__(self, body: bytes, retries: int) -> None:
        if self.broken:
            raise RuntimeError("broker down")
        self.sent.append((body, retries))


def _body(file_id: str = "x") -> bytes:
    return Transcript(id=file_id, text="t", received_at=now).model_dump_json().encode()


async def test_process_success_acks(transcripts: FakeTranscripts, starter: FakeStarter) -> None:
    msg = FakeMessage(_body())
    await rabbit.process_message(msg, transcripts, starter, Republisher())  # type: ignore[arg-type]
    assert msg.calls == [("ack", None)]
    assert starter.classify == ["x"]


async def test_process_broken_body_rejects_without_requeue(
    transcripts: FakeTranscripts, starter: FakeStarter
) -> None:
    unsafe_id = b'{"id": "../x", "text": "t", "received_at": "2026-01-01T00:00:00Z"}'
    for body in (b"{broken", b'{"id": "x"}', b"\xff\xfe", unsafe_id):
        msg = FakeMessage(body)
        await rabbit.process_message(msg, transcripts, starter, Republisher())  # type: ignore[arg-type]
        assert msg.calls == [("reject", False)]
    assert transcripts.items == {}


async def test_process_starter_error_republishes_with_counter(
    transcripts: FakeTranscripts, no_backoff: None
) -> None:
    msg = FakeMessage(_body(), headers={"x-retries": 2})
    republish = Republisher()
    await rabbit.process_message(msg, transcripts, BrokenStarter(), republish)  # type: ignore[arg-type]
    assert republish.sent == [(_body(), 3)]
    assert msg.calls == [("ack", None)]


async def test_process_first_failure_starts_counter(
    transcripts: FakeTranscripts, no_backoff: None
) -> None:
    msg = FakeMessage(_body())
    republish = Republisher()
    await rabbit.process_message(msg, transcripts, BrokenStarter(), republish)  # type: ignore[arg-type]
    assert republish.sent == [(_body(), 1)]


async def test_process_poison_goes_to_dlq(transcripts: FakeTranscripts, no_backoff: None) -> None:
    msg = FakeMessage(_body(), headers={"x-retries": rabbit.MAX_RETRIES})
    republish = Republisher()
    await rabbit.process_message(msg, transcripts, BrokenStarter(), republish)  # type: ignore[arg-type]
    assert republish.sent == []
    assert msg.calls == [("reject", False)]  # dead-letter -> <queue>.dlq


async def test_process_republish_failure_nacks_with_requeue(
    transcripts: FakeTranscripts, no_backoff: None
) -> None:
    msg = FakeMessage(_body())
    await rabbit.process_message(
        msg,  # type: ignore[arg-type]
        transcripts,
        BrokenStarter(),
        Republisher(broken=True),
    )
    assert msg.calls == [("nack", True)]
