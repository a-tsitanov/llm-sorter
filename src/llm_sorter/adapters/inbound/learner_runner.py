"""Дообучение как процесс: читает stream other, раз в check_s запускает discover."""

import asyncio
import fcntl
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TextIO

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from llm_sorter.adapters.inbound.processor import Wait, wait_or_stop
from llm_sorter.adapters.outbound.rabbit import Names, RabbitBus, declare_topology
from llm_sorter.application.learner import OFFSET_KEY, Learner, Proposer
from llm_sorter.domain.messages import load_record
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.store import TopicStore

log = logging.getLogger(__name__)

STREAM_PREFETCH = 500  # stream без prefetch не отдаёт сообщения
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 300.0



class LearnerLock:
    """Единственный экземпляр дообучения на машине: flock рядом с файлом SQLite."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: TextIO | None = None

    def __enter__(self) -> "LearnerLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        f = self._path.open("w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            f.close()
            raise RuntimeError(f"another learner is running ({self._path})") from e
        self._file = f
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._file is not None:
            fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def stream_offset(headers: Mapping[str, Any] | None) -> int:
    raw = (headers or {}).get("x-stream-offset")
    if not isinstance(raw, int):
        raise ValueError(f"no x-stream-offset header: {raw!r}")
    return raw


@dataclass
class StreamPosition:
    next: int = 0  # offset следующей непрочитанной записи


class Maintainable(Protocol):
    async def run_due(self) -> list[str]: ...

    async def save_offset(self, next_offset: int) -> None: ...

    async def refresh_old(self, max_age: timedelta) -> int: ...


async def maintenance_loop(
    learner: Maintainable,
    pos: StreamPosition,
    check_s: float,
    refresh_s: float,
    max_age: timedelta,
    stop: asyncio.Event,
    wait: Wait = wait_or_stop,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Пороги -> discover -> позиция; переиздание старых раз в refresh_s.
    Сбой (LLM, Rabbit, SQLite) - backoff и повтор, а не падение процесса."""
    last_refresh = clock()
    while not stop.is_set():
        delay = BACKOFF_START_S
        while True:
            try:
                await learner.run_due()
                await learner.save_offset(pos.next)
                if clock() - last_refresh >= refresh_s:
                    await learner.refresh_old(max_age)
                    last_refresh = clock()
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("learner tick failed, retry in %.0fs", delay)
                await wait(stop, delay)
                delay = min(delay * 2, BACKOFF_MAX_S)
                if stop.is_set():
                    return
        await wait(stop, check_s)


async def run_learner(
    url: str,
    n: Names,
    store: TopicStore,
    discover: Proposer,
    metrics: Metrics,
    check_s: float,
    refresh_s: float,
    other_max_age: str,
    other_max_bytes: int,
    stop: asyncio.Event,
) -> None:
    conn = await aio_pika.connect_robust(url)
    try:
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=STREAM_PREFETCH)
        exchange = await declare_topology(channel, n, other_max_age, other_max_bytes)
        learner = Learner(store, discover, RabbitBus(channel, exchange, n), metrics)
        saved = await store.get_state(OFFSET_KEY)
        start: int | str = int(saved) if saved is not None else "first"
        pos = StreamPosition(next=start if isinstance(start, int) else 0)
        log.info("learner reads %s from offset %s", n.other, start)

        async def on_message(msg: AbstractIncomingMessage) -> None:
            offset = stream_offset(msg.headers)
            try:
                learner.feed(offset, load_record(msg.body))
            except ValueError as e:  # stream нельзя отправить в DLQ: пропускаем
                log.warning("skip unreadable stream record at %d: %s", offset, e)
            pos.next = offset + 1
            await msg.ack()

        queue = await channel.get_queue(n.other)
        tag = await queue.consume(on_message, arguments={"x-stream-offset": start})
        try:
            await maintenance_loop(
                learner, pos, check_s, refresh_s, max_age_timedelta(other_max_age) / 2, stop
            )
        finally:
            await queue.cancel(tag)
    finally:
        await conn.close()


_UNITS = {"Y": 365 * 86400, "M": 30 * 86400, "D": 86400, "h": 3600, "m": 60, "s": 1}


def max_age_timedelta(value: str) -> timedelta:
    """Формат x-max-age RabbitMQ: число и единица Y/M/D/h/m/s (30D, 12h)."""
    num, unit = value[:-1], value[-1:]
    if unit not in _UNITS or not num.isdigit():
        raise ValueError(f"bad x-max-age {value!r}")
    return timedelta(seconds=int(num) * _UNITS[unit])
