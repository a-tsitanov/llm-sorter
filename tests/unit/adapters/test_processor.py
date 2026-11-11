import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llm_sorter.adapters.inbound.processor import TreeCache, handle_delivery
from llm_sorter.adapters.outbound.rabbit import reprocess_count
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.process import ProcessService
from llm_sorter.domain.changes import AddTopic
from llm_sorter.domain.messages import Transcript
from llm_sorter.domain.tree import Tree
from tests.fakes import FakeBus, FakeDecision, FakeGuard, RecordingMetrics
from tests.helpers import topic, tree3

NOW = datetime(2026, 10, 23, tzinfo=UTC)


class FakeMessage:
    def __init__(self, body: bytes, headers: dict[str, Any] | None = None, events=None) -> None:
        self.body = body
        self.headers = headers or {}
        self.events: list[str] = events if events is not None else []
        self.acked = self.rejected = self.nacked = 0
        self.requeue: bool | None = None

    async def ack(self) -> None:
        self.acked += 1
        self.events.append("ack")

    async def reject(self, requeue: bool = False) -> None:
        self.rejected += 1
        self.requeue = requeue

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None:
        self.nacked += 1
        self.requeue = requeue


class StaticCache:
    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    async def get(self, min_rev: int = 0) -> Tree:
        return self.tree


class Sleeps:
    def __init__(self, stop: asyncio.Event | None = None, stop_after: int | None = None) -> None:
        self.delays: list[float] = []
        self._stop, self._after = stop, stop_after

    async def __call__(self, stop: asyncio.Event, delay: float) -> None:
        self.delays.append(delay)
        if self._stop is not None and len(self.delays) == self._after:
            self._stop.set()


def body(id_: str = "m1", text: str = "текст") -> bytes:
    return Transcript(id=id_, text=text, received_at=NOW).model_dump_json().encode()


def svc(dec: FakeDecision, guard: FakeGuard | None = None) -> ProcessService:
    return ProcessService(dec, guard or FakeGuard([]), RecordingMetrics(), now=lambda: NOW)


@pytest.mark.parametrize("raw", [b"{not json", b'{"id": "../x", "text": "t"}', b'{"text": "t"}'])
async def test_invalid_body_rejected_to_dlq(raw: bytes) -> None:
    m, bus = FakeMessage(raw), FakeBus()
    await handle_delivery(m, svc(FakeDecision([])), StaticCache(tree3()), bus, asyncio.Event())
    assert (m.rejected, m.requeue, m.acked) == (1, False, 0)
    assert bus.events == []


async def test_publish_before_ack() -> None:
    events: list[str] = []
    m, bus = FakeMessage(body(), events=events), FakeBus(events)
    dec = FakeDecision(["finance", "cards"])
    await handle_delivery(
        m, svc(dec, FakeGuard([True])), StaticCache(tree3()), bus, asyncio.Event()
    )
    assert events == ["classified:finance.cards", "other:item", "history", "ack"]


async def test_multi_results_and_unmatched_messages_are_saved_in_history() -> None:
    bus = FakeBus()
    service = ProcessService(
        FakeDecision(["finance", "tech", "cards"]),
        FakeGuard([True, True]),
        RecordingMetrics(),
        multi_topic=True,
    )
    await handle_delivery(FakeMessage(body()), service, StaticCache(tree3()), bus, asyncio.Event())
    assert {c.topic.path for c in bus.classified_msgs} == {"finance.cards", "tech"}
    [snapshot] = bus.history_records
    assert {t.path for t in snapshot.topics} == {"finance.cards", "tech"}

    unmatched = FakeBus()
    await handle_delivery(
        FakeMessage(body("b")),
        svc(FakeDecision(["other"])),
        StaticCache(tree3()),
        unmatched,
        asyncio.Event(),
    )
    assert unmatched.history_records[0].topics == []


async def test_reprocess_header_reaches_service() -> None:
    m, bus = FakeMessage(body(), {"x-reprocess": 3}), FakeBus()
    await handle_delivery(
        m, svc(FakeDecision(["other"])), StaticCache(tree3()), bus, asyncio.Event()
    )
    assert bus.others[0].reprocess == 3 and bus.others[0].final is True


async def test_llm_down_waits_then_processes_once() -> None:
    m, bus = FakeMessage(body()), FakeBus()
    dec = FakeDecision([ConnectionError(), ConnectionError(), "finance", "cards"])
    sleeps = Sleeps()
    await handle_delivery(
        m, svc(dec, FakeGuard([True])), StaticCache(tree3()), bus, asyncio.Event(), wait=sleeps
    )
    assert sleeps.delays == [1.0, 2.0]
    assert (m.acked, m.rejected, m.nacked) == (1, 0, 0)
    assert [c.topic.path for c in bus.classified_msgs] == ["finance.cards"]
    assert not any(o.reason == "model_error" for o in bus.others)


async def test_backoff_is_capped() -> None:
    m, bus = FakeMessage(body()), FakeBus()
    dec = FakeDecision([ConnectionError()] * 12 + ["other"])
    sleeps = Sleeps()
    await handle_delivery(m, svc(dec), StaticCache(tree3()), bus, asyncio.Event(), wait=sleeps)
    assert max(sleeps.delays) == 300.0 and sleeps.delays[:3] == [1.0, 2.0, 4.0]


async def test_publish_failure_retried_without_duplicate_ack() -> None:
    class FlakyBus(FakeBus):
        fails = 1

        async def other(self, record):  # type: ignore[no-untyped-def]
            if self.fails:
                self.fails -= 1
                raise ConnectionError("rabbit down")
            await super().other(record)

    m, bus = FakeMessage(body()), FlakyBus()
    await handle_delivery(
        m,
        svc(FakeDecision(["other", "other"])),
        StaticCache(tree3()),
        bus,
        asyncio.Event(),
        wait=Sleeps(),
    )
    assert m.acked == 1 and len(bus.others) == 1


async def test_stop_during_outage_requeues() -> None:
    stop = asyncio.Event()
    m, bus = FakeMessage(body()), FakeBus()
    dec = FakeDecision([ConnectionError()] * 10)
    await handle_delivery(
        m, svc(dec), StaticCache(tree3()), bus, stop, wait=Sleeps(stop, stop_after=2)
    )
    assert (m.acked, m.nacked, m.requeue) == (0, 1, True)


@pytest.mark.parametrize(
    "headers,n",
    [
        (None, 0),
        ({}, 0),
        ({"x-reprocess": "abc"}, 0),
        ({"x-reprocess": -1}, 0),
        ({"x-reprocess": 2}, 2),
        ({"x-reprocess": "4"}, 4),
        ({"x-reprocess": 2.5}, 0),
    ],
)
def test_reprocess_count(headers: dict[str, Any] | None, n: int) -> None:
    assert reprocess_count(headers) == n


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteTopicStore]:
    s = await SqliteTopicStore.open(tmp_path / "s.db")
    yield s
    await s.close()


async def test_tree_cache_reloads_on_rev_change(store: SqliteTopicStore) -> None:
    clock = [0.0]
    cache = TreeCache(store, refresh_s=1.0, clock=lambda: clock[0])
    assert list((await cache.get()).topics) == ["root"]
    await store.apply([AddTopic(topic=topic("finance"))], "cli")
    clock[0] = 0.5
    assert "finance" not in (await cache.get()).topics  # ещё не время проверять rev
    clock[0] = 1.5
    assert "finance" in (await cache.get()).topics


async def test_stop_interrupts_long_backoff() -> None:
    import time

    stop = asyncio.Event()
    m, bus = FakeMessage(body()), FakeBus()
    dec = FakeDecision([ConnectionError()] * 100)
    loop = asyncio.get_running_loop()
    loop.call_later(0.2, stop.set)
    started = time.perf_counter()
    # настоящая пауза: 1 + 2 + 4 ... секунд, stop через 0.2 с должен её прервать
    await handle_delivery(m, svc(dec), StaticCache(tree3()), bus, stop)
    assert time.perf_counter() - started < 1.0
    assert (m.acked, m.nacked, m.requeue) == (0, 1, True)


async def test_tree_cache_reloads_for_newer_min_rev(store: SqliteTopicStore) -> None:
    cache = TreeCache(store, refresh_s=3600.0, clock=lambda: 0.0)
    assert (await cache.get()).rev == 0
    await store.apply([AddTopic(topic=topic("finance"))], "cli")
    assert (await cache.get()).rev == 0  # кеш свежий по времени
    assert "finance" in (await cache.get(min_rev=1)).topics  # перегон по новому дереву


async def test_reinjected_message_uses_header_rev() -> None:
    class Spy(StaticCache):
        asked: list[int] = []

        async def get(self, min_rev: int = 0) -> Tree:
            self.asked.append(min_rev)
            return self.tree

    m, bus, cache = FakeMessage(body(), {"x-tree-rev": 7}), FakeBus(), Spy(tree3())
    await handle_delivery(m, svc(FakeDecision(["other"])), cache, bus, asyncio.Event())
    assert cache.asked == [7]


@pytest.mark.parametrize(
    "headers,rev", [(None, 0), ({"x-tree-rev": "x"}, 0), ({"x-tree-rev": 4}, 4)]
)
def test_tree_rev_header(headers: dict[str, Any] | None, rev: int) -> None:
    from llm_sorter.adapters.outbound.rabbit import tree_rev_header

    assert tree_rev_header(headers) == rev
