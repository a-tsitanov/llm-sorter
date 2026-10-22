"""RabbitMQ: публикация транскриптов и consumer, стартующий классификацию."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
    HeadersType,
)
from pydantic import ValidationError

from llm_sorter.domain.classification import Transcript
from llm_sorter.ports.io import WorkflowStarter
from llm_sorter.ports.storage import TranscriptStore

log = logging.getLogger(__name__)

PREFETCH = 10
NACK_BACKOFF_S = 1.0  # пауза перед повтором, чтобы не крутить горячий цикл
MAX_RETRIES = 5  # после стольких повторов сообщение уходит в <queue>.dlq
RETRY_HEADER = "x-retries"

# (body, номер повтора) -> публикация копии в ту же очередь
Republish = Callable[[bytes, int], Awaitable[None]]


def dlq_name(queue: str) -> str:
    return f"{queue}.dlq"


async def declare_queue(channel: AbstractChannel, queue: str) -> AbstractQueue:
    """Очередь с dead-letter в `<queue>.dlq`. Аргументы должны совпадать у всех объявлений:
    уже существующую очередь без них RabbitMQ не переобъявит (PRECONDITION_FAILED)."""
    await channel.declare_queue(dlq_name(queue), durable=True)
    return await channel.declare_queue(
        queue,
        durable=True,
        arguments={"x-dead-letter-exchange": "", "x-dead-letter-routing-key": dlq_name(queue)},
    )


def _message(body: bytes, headers: HeadersType | None = None) -> aio_pika.Message:
    return aio_pika.Message(
        body=body,
        headers=headers,
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )


class RabbitPublisher:
    """Публикует в durable-очередь через default exchange; соединение открывается лениво."""

    def __init__(self, url: str, queue: str) -> None:
        self._url = url
        self._queue = queue
        self._conn: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None

    async def _ensure_channel(self) -> AbstractChannel:
        if self._channel is None:
            if self._conn is None:
                self._conn = await aio_pika.connect_robust(self._url)
            channel = await self._conn.channel()
            await declare_queue(channel, self._queue)
            self._channel = channel  # только после успешного declare
        return self._channel

    async def publish(self, t: Transcript) -> None:
        channel = await self._ensure_channel()
        await channel.default_exchange.publish(
            _message(t.model_dump_json().encode()), routing_key=self._queue
        )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
        self._conn = self._channel = None


def parse_message(body: bytes) -> Transcript:
    """Битое тело -> ValidationError."""
    return Transcript.model_validate_json(body)


async def _apply(t: Transcript, transcripts: TranscriptStore, starter: WorkflowStarter) -> None:
    await transcripts.put(t)
    if not await starter.start_classify(t.id):
        log.info("classify %s already started", t.id)


async def handle_message(
    body: bytes, transcripts: TranscriptStore, starter: WorkflowStarter
) -> None:
    """Сохраняет транскрипт и стартует классификацию; повтор не ошибка."""
    await _apply(parse_message(body), transcripts, starter)


def _retries(message: AbstractIncomingMessage) -> int:
    value = message.headers.get(RETRY_HEADER)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


async def process_message(
    message: AbstractIncomingMessage,
    transcripts: TranscriptStore,
    starter: WorkflowStarter,
    republish: Republish,
) -> None:
    """Битое сообщение -> reject без requeue (в DLQ). Сбой побочных эффектов -> копия с
    увеличенным x-retries в конец очереди и ack; после MAX_RETRIES -> reject (в DLQ)."""
    try:
        t = parse_message(message.body)
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
        log.error("reject invalid message %s: %s", message.message_id, e)
        await message.reject(requeue=False)
        return
    try:
        await _apply(t, transcripts, starter)
    except Exception:
        retries = _retries(message)
        if retries >= MAX_RETRIES:
            log.exception("handle %s failed %d times, dead-letter", t.id, retries + 1)
            await message.reject(requeue=False)
            return
        log.exception("handle %s failed, retry %d", t.id, retries + 1)
        await asyncio.sleep(NACK_BACKOFF_S)
        try:
            await republish(message.body, retries + 1)
        except Exception:
            log.exception("republish %s failed, requeue", t.id)
            await message.nack(requeue=True)
            return
        await message.ack()
        return
    await message.ack()


async def run_consumer(
    url: str,
    queue: str,
    transcripts: TranscriptStore,
    starter: WorkflowStarter,
    stop: asyncio.Event | None = None,
) -> None:
    """Читает очередь до stop (или отмены задачи)."""
    stop = stop or asyncio.Event()
    conn = await aio_pika.connect_robust(url)
    try:
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=PREFETCH)
        q = await declare_queue(channel, queue)

        async def republish(body: bytes, retries: int) -> None:
            await channel.default_exchange.publish(
                _message(body, {RETRY_HEADER: retries}), routing_key=queue
            )

        async def on_message(message: AbstractIncomingMessage) -> None:
            await process_message(message, transcripts, starter, republish)

        tag = await q.consume(on_message)
        await stop.wait()
        await q.cancel(tag)
    finally:
        await conn.close()
