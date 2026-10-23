import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aio_pika.abc import AbstractRobustConnection

from llm_sorter.adapters.inbound.processor import TreeCache, run_processor
from llm_sorter.adapters.outbound.rabbit import InboxPublisher, Names
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.process import ProcessService
from llm_sorter.domain.changes import AddTopic
from llm_sorter.domain.messages import Transcript, load_record
from tests.fakes import FakeGuard, RecordingMetrics
from tests.helpers import topic
from tests.integration.conftest import drain, read_stream

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 23, tzinfo=UTC)


class KeywordDecision:
    """Карта -> finance/cards, остальное -> other."""

    async def decide(self, text: str, candidates):  # type: ignore[no-untyped-def]
        ids = [c.id for c in candidates]
        for want in ("finance", "cards"):
            if want in ids and "карт" in text:
                return want
        return "other"


async def test_feed_process_classified_and_archive(
    rabbit_url: str, names: Names, conn: AbstractRobustConnection, tmp_path: Path
) -> None:
    store = await SqliteTopicStore.open(tmp_path / "s.db")
    await store.apply(
        [AddTopic(topic=topic("finance")), AddTopic(topic=topic("cards", "finance"))], "cli"
    )
    service = ProcessService(KeywordDecision(), FakeGuard([]), RecordingMetrics())
    stop = asyncio.Event()
    runner = asyncio.create_task(
        run_processor(rabbit_url, names, 4, service, TreeCache(store, 0.1), stop)
    )
    pub = InboxPublisher(rabbit_url, names)
    try:
        await asyncio.sleep(0.5)  # processor объявил топологию
        await pub.publish(Transcript(id="a", text="заблокировали карту", received_at=NOW))
        await pub.publish(Transcript(id="b", text="погода в Сочи", received_at=NOW))

        ch = await conn.channel()
        [msg] = await drain(ch, names.archive, 1)
        assert msg.routing_key == "finance.cards"
        assert json.loads(msg.body)["topic"]["path"] == "finance.cards"

        records = [load_record(m.body) for m in await read_stream(conn, names.other, 2)]
        assert sorted((r.id, r.node, r.reason) for r in records) == [  # type: ignore[union-attr]
            ("a", "cards", "leaf"),
            ("b", "root", "no_match"),
        ]
    finally:
        stop.set()
        await runner
        await pub.close()
        await store.close()


async def test_invalid_body_goes_to_dlq(
    rabbit_url: str, names: Names, conn: AbstractRobustConnection, tmp_path: Path
) -> None:
    import aio_pika

    store = await SqliteTopicStore.open(tmp_path / "s.db")
    service = ProcessService(KeywordDecision(), FakeGuard([]), RecordingMetrics())
    stop = asyncio.Event()
    runner = asyncio.create_task(
        run_processor(rabbit_url, names, 4, service, TreeCache(store, 0.1), stop)
    )
    try:
        await asyncio.sleep(0.5)
        ch = await conn.channel()
        await ch.default_exchange.publish(aio_pika.Message(b"{broken"), routing_key=names.inbox)
        [dead] = await drain(ch, names.dlq, 1)
        assert dead.body == b"{broken"
    finally:
        stop.set()
        await runner
        await store.close()
