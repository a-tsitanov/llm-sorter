"""Интеграция с настоящим RabbitMQ (`make up`). Имена очередей уникальны на тест и удаляются."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import aio_pika
import pytest
from aio_pika.abc import AbstractChannel, AbstractRobustConnection

from llm_sorter.adapters.outbound.rabbit import Names
from llm_sorter.config import AppConfig


@pytest.fixture
def rabbit_url() -> str:
    return AppConfig().rabbit_url


@pytest.fixture
def names() -> Names:
    p = f"test-{uuid.uuid4().hex[:8]}"
    return Names(
        inbox=f"{p}.inbox", exchange=f"{p}.classified", archive=f"{p}.all", other=f"{p}.other"
    )


@pytest.fixture
async def conn(rabbit_url: str, names: Names) -> AsyncIterator[AbstractRobustConnection]:
    c = await aio_pika.connect_robust(rabbit_url)
    yield c
    ch = await c.channel()
    for q in (names.inbox, names.dlq, names.archive, names.other, names.history_name):
        await ch.queue_delete(q)
    await ch.exchange_delete(names.exchange)
    await c.close()


async def drain(ch: AbstractChannel, queue: str, expect: int, timeout: float = 10.0) -> list[Any]:
    """Сообщения обычной очереди (ack), ждёт пока придёт expect штук."""
    q = await ch.get_queue(queue)
    got: list[Any] = []
    deadline = asyncio.get_running_loop().time() + timeout
    while len(got) < expect and asyncio.get_running_loop().time() < deadline:
        msg = await q.get(fail=False)
        if msg is None:
            await asyncio.sleep(0.1)
            continue
        await msg.ack()
        got.append(msg)
    return got


async def read_stream(
    conn: AbstractRobustConnection, stream: str, expect: int, timeout: float = 10.0
) -> list[aio_pika.abc.AbstractIncomingMessage]:
    """Читает stream с начала: нужен prefetch и x-stream-offset."""
    ch = await conn.channel()
    await ch.set_qos(prefetch_count=100)
    q = await ch.get_queue(stream)
    got: list[aio_pika.abc.AbstractIncomingMessage] = []
    done = asyncio.Event()

    async def on(msg: aio_pika.abc.AbstractIncomingMessage) -> None:
        got.append(msg)
        await msg.ack()
        if len(got) >= expect:
            done.set()

    tag = await q.consume(on, arguments={"x-stream-offset": "first"})
    try:
        await asyncio.wait_for(done.wait(), timeout)
    finally:
        await q.cancel(tag)
        await ch.close()
    return got
