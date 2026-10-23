from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.discover import Accepted, DiscoverResult
from llm_sorter.application.learner import OFFSET_KEY, Learner
from llm_sorter.domain.changes import AddTopic
from llm_sorter.domain.messages import Considered, OtherItem, Reason, Resolved
from llm_sorter.domain.tree import ROOT_ID, Settings, Tree
from tests.fakes import FakeBus, RecordingMetrics
from tests.helpers import topic

T0 = datetime(2026, 10, 23, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.t = T0 + timedelta(hours=1)

    def __call__(self) -> datetime:
        return self.t


class FakeDiscover:
    """Скрипт результатов; по умолчанию - всё в considered."""

    def __init__(self, results: list[DiscoverResult] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[str, list[str]]] = []

    async def propose(self, tree: Tree, node_id: str, texts: list[str]) -> DiscoverResult:
        self.calls.append((node_id, texts))
        if self.results:
            return self.results.pop(0)
        return DiscoverResult([], list(range(len(texts))), 0)


def item(
    id_: str, node: str = ROOT_ID, at: datetime = T0, reason: Reason = "no_match", **kw: object
) -> OtherItem:
    return OtherItem(
        id=id_,
        text=f"текст {id_}",
        metadata={},
        received_at=T0,
        node=node,
        trail=[],
        reason=reason,
        tree_rev=0,
        at=at,
        **kw,  # type: ignore[arg-type]
    )


def accepted(id_: str, parent: str, members: list[int]) -> Accepted:
    t = topic(id_, parent, guard="Разговор о чём-то важном?", origin="discovered")
    return Accepted(t, members)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteTopicStore]:
    s = await SqliteTopicStore.open(tmp_path / "s.db")
    await s.save_settings(
        Settings(discover=Settings().discover.model_copy(update={"buffer_threshold": 10}))
    )
    yield s
    await s.close()


def learner(
    store: SqliteTopicStore,
    discover: FakeDiscover,
    bus: FakeBus | None = None,
    metrics: RecordingMetrics | None = None,
    clock: Clock | None = None,
) -> Learner:
    return Learner(
        store, discover, bus or FakeBus(), metrics or RecordingMetrics(), now=clock or Clock()
    )


def fill(lr: Learner, n: int, node: str = ROOT_ID, start: int = 0, prefix: str = "m") -> None:
    for i in range(n):
        lr.feed(start + i, item(f"{prefix}{start + i}", node))


async def test_below_threshold_no_discover(store: SqliteTopicStore) -> None:
    d = FakeDiscover()
    lr = learner(store, d)
    fill(lr, 9)
    assert await lr.run_due() == [] and d.calls == []


async def test_threshold_triggers_discover_and_reinjects_members(store: SqliteTopicStore) -> None:
    d = FakeDiscover(
        [DiscoverResult([accepted("taxes", ROOT_ID, [0, 1, 2, 3, 4, 5])], [6, 7, 8, 9], 0)]
    )
    bus, m = FakeBus(), RecordingMetrics()
    lr = learner(store, d, bus, m)
    fill(lr, 10)
    assert await lr.run_due() == [ROOT_ID]
    tree = await store.load_tree()
    assert "taxes" in tree.topics and tree.get("taxes").origin == "discovered"
    assert [(t.id, r) for t, r in bus.reinjected] == [(f"m{i}", 1) for i in range(6)]
    assert set(bus.reinjected_revs) == {tree.rev}  # обработчик перечитает дерево до этой ревизии
    resolved = [r for r in bus.records if isinstance(r, Resolved)]
    considered = [r for r in bus.records if isinstance(r, Considered)]
    assert resolved[0].ids == [f"m{i}" for i in range(6)]
    assert considered[0].node == ROOT_ID and considered[0].ids == [f"m{i}" for i in range(6, 10)]
    assert [p.item.id for p in lr.buffer(ROOT_ID)] == [f"m{i}" for i in range(6, 10)]
    assert all(p.considered for p in lr.buffer(ROOT_ID))
    assert m.discover_runs == [(ROOT_ID, 1, 0)] and m.buffer_sizes[-1] == (ROOT_ID, 4)
    [entry] = [e for e in await store.changelog() if e.op == "add"]
    assert entry.actor == "learner"


async def test_considered_do_not_count_again(store: SqliteTopicStore) -> None:
    d = FakeDiscover()
    lr = learner(store, d)
    fill(lr, 10)
    assert await lr.run_due() == [ROOT_ID]
    assert await lr.run_due() == []  # те же 10 уже рассмотрены
    fill(lr, 9, start=10)
    assert await lr.run_due() == []
    fill(lr, 1, start=19)
    assert await lr.run_due() == [ROOT_ID]
    assert len(d.calls[-1][1]) == 20  # кластеризуется весь буфер, а не только новое


async def test_latest_copy_wins(store: SqliteTopicStore) -> None:
    lr = learner(store, FakeDiscover())
    lr.feed(0, item("a", ROOT_ID))
    lr.feed(1, item("a", "finance"))
    assert lr.buffer(ROOT_ID) == [] and [p.offset for p in lr.buffer("finance")] == [1]


async def test_final_items_ignored(store: SqliteTopicStore) -> None:
    lr = learner(store, FakeDiscover())
    lr.feed(0, item("a"))
    lr.feed(1, item("a", final=True))  # финальная копия убирает прежнюю и не буферизуется
    assert lr.buffer(ROOT_ID) == [] and lr.watermark() is None


async def test_resolved_does_not_drop_newer_copy(store: SqliteTopicStore) -> None:
    lr = learner(store, FakeDiscover())
    lr.feed(0, item("a", at=T0))
    lr.feed(
        1, item("a", "finance", at=T0 + timedelta(minutes=5))
    )  # перегнанная копия раньше Resolved
    lr.feed(2, Resolved(ids=["a"], at=T0 + timedelta(minutes=1)))
    assert [p.offset for p in lr.buffer("finance")] == [1]


async def test_watermark(store: SqliteTopicStore) -> None:
    lr = learner(store, FakeDiscover())
    assert lr.watermark() is None
    fill(lr, 3, start=5)
    lr.feed(9, Resolved(ids=["m5"], at=T0))
    assert lr.watermark() == 6
    await lr.save_offset(10)
    assert await store.get_state(OFFSET_KEY) == "6"
    lr.feed(10, Resolved(ids=["m6", "m7"], at=T0))
    await lr.save_offset(11)
    assert await store.get_state(OFFSET_KEY) == "11"  # буферы пусты - вся история разобрана


async def test_restart_replay_restores_buffers(store: SqliteTopicStore) -> None:
    records: list[OtherItem | Resolved | Considered] = [
        item("a"),
        item("b"),
        item("c"),
        item("d", "finance"),
        Resolved(ids=["a"], at=T0),
        Considered(node=ROOT_ID, ids=["b"], at=T0),
        item("e"),
        Resolved(ids=["d"], at=T0),
    ]

    def snapshot(lr: Learner) -> dict[str, list[tuple[str, bool]]]:
        return {n: [(p.item.id, p.considered) for p in lr.buffer(n)] for n in (ROOT_ID, "finance")}

    first = learner(store, FakeDiscover())
    for i, r in enumerate(records):
        first.feed(i, r)
    w = first.watermark()
    assert w is not None
    replay = learner(store, FakeDiscover())
    for i, r in enumerate(records):
        if i >= w:
            replay.feed(i, r)
    assert (
        snapshot(replay)
        == snapshot(first)
        == {
            ROOT_ID: [("b", True), ("c", False), ("e", False)],
            "finance": [],
        }
    )


async def test_leaf_split_reinjects_whole_leaf_buffer(store: SqliteTopicStore) -> None:
    await store.apply([AddTopic(topic=topic("cards"))], "cli")
    d = FakeDiscover(
        [
            DiscoverResult(
                [accepted("cashback", "cards", [0, 1, 2]), accepted("limits", "cards", [3, 4, 5])],
                [6, 7, 8, 9],
                0,
            )
        ]
    )
    bus = FakeBus()
    lr = learner(store, d, bus)
    for i in range(10):
        lr.feed(i, item(f"m{i}", "cards", reason="leaf"))
    assert await lr.run_due() == ["cards"]
    assert len(bus.reinjected) == 10 and lr.buffer("cards") == []
    assert not [r for r in bus.records if isinstance(r, Considered)]


async def test_blocked_node_reported(store: SqliteTopicStore) -> None:
    await store.save_settings(
        Settings(
            discover=Settings().discover.model_copy(
                update={"buffer_threshold": 1, "max_children": 1}
            )
        )
    )
    await store.apply([AddTopic(topic=topic("only"))], "cli")
    d, m = FakeDiscover(), RecordingMetrics()
    lr = learner(store, d, metrics=m)
    fill(lr, 3)
    assert await lr.run_due() == [] and d.calls == []
    assert m.discover_blocked_by_node == {ROOT_ID: True}


async def test_archived_node_skipped(store: SqliteTopicStore) -> None:
    d = FakeDiscover()
    lr = learner(store, d)
    fill(lr, 10, node="gone")
    assert await lr.run_due() == [] and d.calls == []


async def test_refresh_old_republishes(store: SqliteTopicStore) -> None:
    clock = Clock()
    bus = FakeBus()
    lr = learner(store, FakeDiscover(), bus, clock=clock)
    lr.feed(0, item("old", at=T0))
    lr.feed(1, item("fresh", at=T0 + timedelta(minutes=50)))
    lr.feed(2, Considered(node=ROOT_ID, ids=["old"], at=T0))
    assert await lr.refresh_old(timedelta(minutes=30)) == 1
    assert [r.kind for r in bus.records] == ["item", "considered"]
    copy = bus.records[0]
    assert isinstance(copy, OtherItem) and copy.id == "old" and copy.at == clock.t
    # копия читается из stream позже и заменяет старую, признак considered сохраняется
    lr.feed(3, copy)
    lr.feed(4, bus.records[1])
    assert [(p.item.id, p.offset, p.considered) for p in lr.buffer(ROOT_ID)] == [
        ("fresh", 1, False),
        ("old", 3, True),
    ]
    assert lr.watermark() == 1


async def test_replayed_offsets_are_ignored(store: SqliteTopicStore) -> None:
    """Переподключение robust-очереди отдаёт историю заново: состояние не должно меняться."""
    lr = learner(store, FakeDiscover())
    records: list[OtherItem | Resolved | Considered] = [
        item("a"),
        item("b"),
        Considered(node=ROOT_ID, ids=["a"], at=T0),
        Resolved(ids=["b"], at=T0),
    ]
    for i, r in enumerate(records):
        lr.feed(i, r)
    before = [(p.item.id, p.considered) for p in lr.buffer(ROOT_ID)]
    for i, r in enumerate(records[:2]):  # повтор с начала, оборван на середине
        lr.feed(i, r)
    assert [(p.item.id, p.considered) for p in lr.buffer(ROOT_ID)] == before == [("a", True)]


def test_marker_marks_caught_up(store: SqliteTopicStore) -> None:
    from llm_sorter.domain.messages import Marker

    lr = learner(store, FakeDiscover())
    assert not lr.has_seen("t1")
    lr.feed(0, Marker(token="t1", at=T0))
    assert lr.has_seen("t1") and lr.watermark() is None


async def test_failing_node_does_not_block_others(store: SqliteTopicStore) -> None:
    await store.apply(
        [AddTopic(topic=topic("finance")), AddTopic(topic=topic("cards", "finance"))], "cli"
    )

    class Flaky(FakeDiscover):
        async def propose(self, tree: Tree, node_id: str, texts: list[str]) -> DiscoverResult:
            if node_id == ROOT_ID:
                raise RuntimeError("context length exceeded")
            return await super().propose(tree, node_id, texts)

    m = RecordingMetrics()
    lr = learner(store, Flaky(), metrics=m)
    fill(lr, 10, node=ROOT_ID, prefix="r")
    fill(lr, 10, node="finance", start=10, prefix="f")
    assert await lr.run_due() == ["finance"]
    assert m.counts["model_error:discover"] == 1


async def test_refresh_schedule_survives_restart(store: SqliteTopicStore) -> None:
    clock, bus = Clock(), FakeBus()
    lr = learner(store, FakeDiscover(), bus, clock=clock)
    lr.feed(0, item("old", at=T0))
    every, age = timedelta(hours=24), timedelta(minutes=30)
    assert await lr.refresh_if_due(every, age) == 1  # ни разу не было - сразу
    assert await lr.refresh_if_due(every, age) is None
    restarted = learner(store, FakeDiscover(), bus, clock=clock)  # перезапуск процесса
    restarted.feed(0, item("old", at=T0))
    assert await restarted.refresh_if_due(every, age) is None
    clock.t += timedelta(hours=25)
    assert await restarted.refresh_if_due(every, age) == 1
