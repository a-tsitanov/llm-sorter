"""Дообучение как процесс: читает stream other, раз в check_s запускает discover."""

import asyncio
import fcntl
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TextIO

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from llm_sorter.adapters.inbound.processor import Wait, wait_or_stop
from llm_sorter.adapters.outbound.rabbit import Names, RabbitBus, declare_topology
from llm_sorter.application.learner import OFFSET_KEY, Learner, Proposer
from llm_sorter.domain.messages import Marker, load_record
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.store import TopicStore

log = logging.getLogger(__name__)

STREAM_PREFETCH = 500  # stream без prefetch не отдаёт сообщения
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 300.0
CATCH_UP_POLL_S = 0.2


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


def stream_gap(saved: int | None, first: int) -> int:
    """Сколько записей между сохранённой позицией и первой доставленной удалил ретеншн."""
    return max(first - saved, 0) if saved is not None else 0


@dataclass
class StreamPosition:
    next: int = 0  # offset следующей непрочитанной записи


class Maintainable(Protocol):
    async def run_due(self) -> list[str]: ...

    async def save_offset(self, next_offset: int) -> None: ...

    async def refresh_if_due(self, every: timedelta, max_age: timedelta) -> int | None: ...

    async def reconsider_all(self) -> int: ...


async def maintenance_loop(
    learner: Maintainable,
    pos: StreamPosition,
    check_s: float,
    refresh_s: float,
    max_age: timedelta,
    stop: asyncio.Event,
    wait: Wait = wait_or_stop,
    ready: Callable[[], bool] = lambda: True,
    reconsider: bool = False,
) -> None:
    """Пороги -> discover -> позиция; переиздание старых раз в refresh_s.
    Пока stream не дочитан до своей метки (ready), буферы неполны: discover ждёт.
    Сбой (LLM, Rabbit, SQLite) - backoff и повтор, а не падение процесса."""
    while not ready():
        if stop.is_set():
            return
        await wait(stop, CATCH_UP_POLL_S)
    if reconsider:
        count = await learner.reconsider_all()
        log.info("reconsider: %d waiting messages re-published as new", count)
    every = timedelta(seconds=refresh_s)
    delay = BACKOFF_START_S
    while not stop.is_set():
        # шаги независимы: падающий discover не останавливает позицию и переиздание
        failed = not await _step("run_due", learner.run_due)
        failed |= not await _step("save_offset", lambda: learner.save_offset(pos.next))
        failed |= not await _step("refresh", lambda: learner.refresh_if_due(every, max_age))
        if failed:
            await wait(stop, delay)
            delay = min(delay * 2, BACKOFF_MAX_S)
        else:
            delay = BACKOFF_START_S
            await wait(stop, check_s)


async def _step(name: str, fn: Callable[[], Awaitable[object]]) -> bool:
    try:
        await fn()
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("learner %s failed", name)
        return False
    return True


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
    reconsider: bool = False,
    max_batch: int = 2000,
) -> None:
    conn = await aio_pika.connect_robust(url)
    try:
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=STREAM_PREFETCH)
        exchange = await declare_topology(channel, n, other_max_age, other_max_bytes)
        learner = Learner(
            store, discover, RabbitBus(channel, exchange, n), metrics, max_batch=max_batch
        )
        saved = await store.get_state(OFFSET_KEY)
        start: int | str = int(saved) if saved is not None else "first"
        pos = StreamPosition(next=start if isinstance(start, int) else 0)
        log.info("learner reads %s from offset %s", n.other, start)

        first_seen = False

        async def on_message(msg: AbstractIncomingMessage) -> None:
            nonlocal first_seen
            offset = stream_offset(msg.headers)
            if not first_seen:
                first_seen = True
                lost = stream_gap(start if isinstance(start, int) else None, offset)
                if lost:
                    log.error(
                        "stream %s: %d records before offset %d were removed by retention "
                        "before the learner read them (OTHER_MAX_BYTES too small?)",
                        n.other,
                        lost,
                        offset,
                    )
                    metrics.other_lost(lost)
            try:
                learner.feed(offset, load_record(msg.body))
            except ValueError as e:  # stream нельзя отправить в DLQ: пропускаем
                log.warning("skip unreadable stream record at %d: %s", offset, e)
            pos.next = offset + 1
            await msg.ack()

        queue = await channel.get_queue(n.other)
        tag = await queue.consume(on_message, arguments={"x-stream-offset": start})
        token = uuid.uuid4().hex
        await RabbitBus(channel, exchange, n).other(Marker(token=token, at=datetime.now(UTC)))
        try:
            await maintenance_loop(
                learner,
                pos,
                check_s,
                refresh_s,
                max_age_timedelta(other_max_age) / 2,
                stop,
                ready=lambda: learner.has_seen(token),
                reconsider=reconsider,
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
