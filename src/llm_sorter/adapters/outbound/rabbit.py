"""RabbitMQ: топология (inbox, exchange результатов, архив, stream other) и публикация."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection

from llm_sorter.config import AppConfig
from llm_sorter.domain.messages import (
    Classified,
    Considered,
    OtherItem,
    Resolved,
    Transcript,
    dump_record,
)

REPROCESS_HEADER = "x-reprocess"


@dataclass(frozen=True)
class Names:
    inbox: str
    exchange: str
    archive: str
    other: str

    @property
    def dlq(self) -> str:
        return f"{self.inbox}.dlq"

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "Names":
        return cls(
            inbox=cfg.inbox_queue,
            exchange=cfg.classified_exchange,
            archive=cfg.classified_archive,
            other=cfg.other_stream,
        )


def reprocess_count(headers: Mapping[str, Any] | None) -> int:
    """Номер перегона из заголовка; нет / мусор / отрицательное -> 0."""
    raw = (headers or {}).get(REPROCESS_HEADER)
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return max(raw, 0)
    if isinstance(raw, str | bytes) and raw.strip().isdigit():
        return int(raw)
    return 0


def _message(body: bytes, headers: dict[str, Any] | None = None) -> aio_pika.Message:
    return aio_pika.Message(
        body=body,
        headers=headers,
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )


async def declare_inbox(ch: AbstractChannel, n: Names) -> None:
    """Аргументы должны совпадать у всех объявлений: иначе PRECONDITION_FAILED."""
    await ch.declare_queue(n.dlq, durable=True)
    await ch.declare_queue(
        n.inbox,
        durable=True,
        arguments={"x-dead-letter-exchange": "", "x-dead-letter-routing-key": n.dlq},
    )


async def declare_topology(
    ch: AbstractChannel, n: Names, other_max_age: str, other_max_bytes: int
) -> AbstractExchange:
    await declare_inbox(ch, n)
    ex = await ch.declare_exchange(n.exchange, aio_pika.ExchangeType.TOPIC, durable=True)
    # архив: без подписчиков результаты не теряются
    archive = await ch.declare_queue(n.archive, durable=True)
    await archive.bind(ex, routing_key="#")
    await ch.declare_queue(
        n.other,
        durable=True,
        arguments={
            "x-queue-type": "stream",
            "x-max-age": other_max_age,
            "x-max-length-bytes": other_max_bytes,
        },
    )
    return ex


class RabbitBus:
    """Канал должен быть с publisher confirms (по умолчанию в aio-pika)."""

    def __init__(self, ch: AbstractChannel, exchange: AbstractExchange, n: Names) -> None:
        self._ch = ch
        self._ex = exchange
        self._n = n

    async def classified(self, msg: Classified) -> None:
        await self._ex.publish(_message(msg.model_dump_json().encode()), routing_key=msg.topic.path)

    async def other(self, record: OtherItem | Resolved | Considered) -> None:
        await self._ch.default_exchange.publish(
            _message(dump_record(record)), routing_key=self._n.other
        )

    async def reinject(self, t: Transcript, reprocess: int) -> None:
        await self._ch.default_exchange.publish(
            _message(t.model_dump_json().encode(), {REPROCESS_HEADER: reprocess}),
            routing_key=self._n.inbox,
        )


class InboxPublisher:
    """Публикация фидеров в inbox; соединение открывается лениво."""

    def __init__(self, url: str, n: Names) -> None:
        self._url = url
        self._n = n
        self._conn: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None

    async def _ensure_channel(self) -> AbstractChannel:
        if self._channel is None:
            if self._conn is None:
                self._conn = await aio_pika.connect_robust(self._url)
            channel = await self._conn.channel()
            await declare_inbox(channel, self._n)
            self._channel = channel  # только после успешного declare
        return self._channel

    async def publish(self, t: Transcript) -> None:
        channel = await self._ensure_channel()
        await channel.default_exchange.publish(
            _message(t.model_dump_json().encode()), routing_key=self._n.inbox
        )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
        self._conn = self._channel = None
