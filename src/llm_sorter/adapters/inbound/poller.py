import asyncio
import contextlib
import logging

from llm_sorter.ports.io import MessagePublisher, TranscriptSource

log = logging.getLogger(__name__)


async def run_poller(
    source: TranscriptSource,
    publisher: MessagePublisher,
    interval_s: float,
    stop: asyncio.Event,
) -> None:
    """Опрашивает источник раз в interval_s и публикует новые транскрипты до stop.
    ack — только после успешной публикации; при ошибке остаток батча ждёт следующего тика."""
    while not stop.is_set():
        try:
            for t in await source.fetch_new():
                await publisher.publish(t)
                await source.ack([t.id])
                log.info("published %s", t.id)
        except Exception:
            log.exception("poll iteration failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), interval_s)
