"""Обработчик inbox: N корутин, публикация результата до ack, ожидание при сбоях."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from pydantic import ValidationError

from llm_sorter.adapters.outbound.rabbit import (
    Names,
    RabbitBus,
    declare_topology,
    reprocess_count,
    tree_rev_header,
)
from llm_sorter.application.process import ProcessService
from llm_sorter.domain.messages import Transcript
from llm_sorter.domain.tree import Tree
from llm_sorter.ports.bus import Bus
from llm_sorter.ports.store import TopicStore

log = logging.getLogger(__name__)

BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 300.0

Wait = Callable[[asyncio.Event, float], Awaitable[None]]


async def wait_or_stop(stop: asyncio.Event, seconds: float) -> None:
    """Пауза, которую прерывает stop: остановка во время сбоя не ждёт backoff."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)


class Delivery(Protocol):
    """Подмножество AbstractIncomingMessage, которое нужно обработчику."""

    @property
    def body(self) -> bytes: ...

    @property
    def headers(self) -> Mapping[str, Any]: ...

    async def ack(self) -> None: ...

    async def reject(self, requeue: bool = False) -> None: ...

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None: ...


class TreeSource(Protocol):
    async def get(self, min_rev: int = 0) -> Tree: ...


class TreeCache:
    """Дерево в памяти; tree_rev сверяется не чаще refresh_s."""

    def __init__(
        self,
        store: TopicStore,
        refresh_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._refresh_s = refresh_s
        self._clock = clock
        self._tree: Tree | None = None
        self._checked = 0.0
        self._lock = asyncio.Lock()

    async def get(self, min_rev: int = 0) -> Tree:
        """min_rev: сообщение перегнано с этой ревизии - старее кеш использовать нельзя."""
        now = self._clock()
        fresh = self._tree is not None and now - self._checked < self._refresh_s
        if fresh and self._tree is not None and self._tree.rev >= min_rev:
            return self._tree
        async with self._lock:  # одна перезагрузка на все корутины
            if self._tree is None or await self._store.tree_rev() != self._tree.rev:
                self._tree = await self._store.load_tree()
            self._checked = now
        return self._tree


async def _process_and_publish(
    t: Transcript,
    reprocess: int,
    min_rev: int,
    service: ProcessService,
    cache: TreeSource,
    bus: Bus,
) -> None:
    out = await service.process(t, await cache.get(min_rev), reprocess)
    if out.classified is not None:
        await bus.classified(out.classified)
    for other in out.others:
        await bus.other(other)


async def handle_delivery(
    msg: Delivery,
    service: ProcessService,
    cache: TreeSource,
    bus: Bus,
    stop: asyncio.Event,
    wait: Wait = wait_or_stop,
) -> None:
    try:
        t = Transcript.model_validate_json(msg.body)
    except ValidationError:
        log.warning("invalid message body, dead-lettered")
        await msg.reject(requeue=False)
        return
    reprocess = reprocess_count(msg.headers)
    min_rev = tree_rev_header(msg.headers)
    delay = BACKOFF_START_S
    while True:
        try:
            await _process_and_publish(t, reprocess, min_rev, service, cache, bus)
        except asyncio.CancelledError:
            raise
        except Exception:
            # сбой LLM / Rabbit / SQLite: ждём, а не отправляем в other или DLQ
            log.exception("processing %s failed, retry in %.0fs", t.id, delay)
            await wait(stop, delay)
            delay = min(delay * 2, BACKOFF_MAX_S)
            if stop.is_set():
                await msg.nack(requeue=True)
                return
            continue
        await msg.ack()
        return


async def run_processor(
    url: str,
    n: Names,
    concurrency: int,
    service: ProcessService,
    cache: TreeSource,
    stop: asyncio.Event,
    other_max_age: str = "30D",
    other_max_bytes: int = 10 * 1024**3,
) -> None:
    """Читает inbox до stop; prefetch = concurrency ограничивает число сообщений в работе."""
    conn = await aio_pika.connect_robust(url)
    tasks: set[asyncio.Task[None]] = set()
    try:
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=concurrency)
        exchange = await declare_topology(channel, n, other_max_age, other_max_bytes)
        bus = RabbitBus(channel, exchange, n)
        queue = await channel.get_queue(n.inbox)

        async def on_message(message: AbstractIncomingMessage) -> None:
            task = asyncio.create_task(handle_delivery(message, service, cache, bus, stop))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        tag = await queue.consume(on_message)
        await stop.wait()
        await queue.cancel(tag)
        # обрабатываемые сообщения дорабатываются; ждущие при сбое вернутся в очередь
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks)
    finally:
        await conn.close()
