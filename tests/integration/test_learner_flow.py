import asyncio
from datetime import UTC, datetime
from pathlib import Path

import aio_pika
import pytest
from aio_pika.abc import AbstractRobustConnection

from llm_sorter.adapters.inbound.learner_runner import run_learner
from llm_sorter.adapters.outbound.rabbit import Names, declare_topology
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.discover import Accepted, DiscoverResult
from llm_sorter.application.learner import OFFSET_KEY
from llm_sorter.domain.messages import OtherItem, dump_record
from llm_sorter.domain.tree import ROOT_ID, Settings, Tree
from tests.fakes import RecordingMetrics
from tests.helpers import topic
from tests.integration.conftest import drain

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 23, tzinfo=UTC)


class AcceptAll:
    """Весь буфер узла - одна новая тема."""

    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, tree: Tree, node_id: str, texts: list[str]) -> DiscoverResult:
        self.calls += 1
        t = topic("taxes", node_id, guard="Разговор о налогах и вычетах?", origin="discovered")
        return DiscoverResult([Accepted(t, list(range(len(texts))))], [], 0)


async def _run_until(cond, store, names, rabbit_url, proposer, timeout: float = 15.0) -> None:  # type: ignore[no-untyped-def]
    stop = asyncio.Event()
    task = asyncio.create_task(
        run_learner(
            rabbit_url, names, store, proposer, RecordingMetrics(), 0.2, 3600.0, "30D", 10**9, stop
        )
    )
    try:
        deadline = asyncio.get_running_loop().time() + timeout
        while not await cond() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.2)
    finally:
        stop.set()
        await task


async def test_learner_discovers_reinjects_and_resumes(
    rabbit_url: str, names: Names, conn: AbstractRobustConnection, tmp_path: Path
) -> None:
    store = await SqliteTopicStore.open(tmp_path / "s.db")
    await store.save_settings(
        Settings(discover=Settings().discover.model_copy(update={"buffer_threshold": 10}))
    )
    ch = await conn.channel()
    await declare_topology(ch, names, "30D", 10**9)
    for i in range(12):
        item = OtherItem(
            id=f"m{i}",
            text=f"налоги {i}",
            metadata={},
            received_at=NOW,
            node=ROOT_ID,
            trail=[],
            reason="no_match",
            tree_rev=0,
            at=NOW,
        )
        await ch.default_exchange.publish(
            aio_pika.Message(dump_record(item)), routing_key=names.other
        )

    proposer = AcceptAll()

    async def has_topic() -> bool:
        return (
            "taxes" in (await store.load_tree()).topics
            and await store.get_state(OFFSET_KEY) == "13"
        )

    await _run_until(has_topic, store, names, rabbit_url, proposer)
    assert await has_topic(), await store.get_state(OFFSET_KEY)

    reinjected = await drain(ch, names.inbox, 12)
    assert sorted(m.headers["x-reprocess"] for m in reinjected) == [1] * 12

    # второй запуск продолжает с сохранённой позиции: повторного перегона нет
    async def never() -> bool:
        return False

    await _run_until(never, store, names, rabbit_url, proposer, timeout=1.5)
    assert proposer.calls == 1
    assert await drain(ch, names.inbox, 1, timeout=1.0) == []
    await store.close()
