"""Чтение истории stream до контрольной метки и повторная отправка исходных сообщений."""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

import aio_pika
from pydantic import ValidationError

from llm_sorter.adapters.outbound.rabbit import Names
from llm_sorter.domain.messages import ClassificationSnapshot

log = logging.getLogger(__name__)


def _update_latest(
    latest: dict[str, ClassificationSnapshot], record: ClassificationSnapshot
) -> None:
    previous = latest.get(record.transcript.id)
    if previous is None or (record.tree_rev, record.classified_at) >= (
        previous.tree_rev,
        previous.classified_at,
    ):
        latest[record.transcript.id] = record


def latest_snapshots(records: list[ClassificationSnapshot]) -> list[ClassificationSnapshot]:
    latest: dict[str, ClassificationSnapshot] = {}
    for record in records:
        _update_latest(latest, record)
    return sorted(latest.values(), key=lambda r: r.transcript.received_at, reverse=True)


async def read_history(
    url: str, names: Names, timeout_s: float = 120.0
) -> list[ClassificationSnapshot]:
    """Сканирует сохранённую историю до метки; читатели stream независимы."""
    conn = await aio_pika.connect_robust(url)
    try:
        ch = await conn.channel()
        await ch.set_qos(prefetch_count=500)
        queue = await ch.get_queue(names.history_name, ensure=True)
        token = uuid.uuid4().hex
        latest: dict[str, ClassificationSnapshot] = {}
        complete = asyncio.Event()

        async def on_message(msg: aio_pika.abc.AbstractIncomingMessage) -> None:
            try:
                data = json.loads(msg.body)
                if isinstance(data, dict) and "history_marker" in data:
                    if data["history_marker"] == token:
                        complete.set()
                else:
                    _update_latest(latest, ClassificationSnapshot.model_validate(data))
            except (ValueError, ValidationError) as exc:
                log.warning("Некорректная запись истории: %s", exc)
            await msg.ack()

        consumer = await queue.consume(on_message, arguments={"x-stream-offset": "first"})
        try:
            await ch.default_exchange.publish(
                aio_pika.Message(json.dumps({"history_marker": token}).encode()),
                routing_key=names.history_name,
            )
            await asyncio.wait_for(complete.wait(), timeout_s)
        finally:
            await queue.cancel(consumer)
        return sorted(latest.values(), key=lambda r: r.transcript.received_at, reverse=True)
    finally:
        await conn.close()


def filter_history(
    records: list[ClassificationSnapshot], topic: str | None, since: datetime | None
) -> list[ClassificationSnapshot]:
    return [
        r
        for r in records
        if (since is None or r.transcript.received_at >= since)
        and (topic is None or any(t.id == topic or t.path == topic for t in r.topics))
    ]


def utc_now() -> datetime:
    return datetime.now(UTC)
