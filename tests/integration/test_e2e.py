"""Сквозной тест: каталог -> poller -> RabbitMQ -> consumer -> Temporal -> Postgres.
Требует `make up`; LM Studio не нужен (модели подменены), БД *_test готовит conftest."""

import asyncio
import contextlib
import uuid
from pathlib import Path

import aio_pika
import pytest
from temporalio.service import RPCError

from llm_sorter.adapters.inbound.poller import run_poller
from llm_sorter.adapters.inbound.rabbit import RabbitPublisher, dlq_name, run_consumer
from llm_sorter.adapters.inbound.stub_source import LocalDirSource
from llm_sorter.bootstrap import build_app
from llm_sorter.config import AppConfig
from llm_sorter.domain.classification import Classification, Status
from llm_sorter.orchestration.worker import run_worker
from tests.fakes import ScriptedDecision, ScriptedGuard

pytestmark = pytest.mark.integration


async def _delete_queue(url: str, name: str) -> None:
    conn = await aio_pika.connect_robust(url)
    try:
        channel = await conn.channel()
        await channel.queue_delete(name)
    finally:
        await conn.close()


async def test_cold_start_file_lands_in_root_other(tmp_path: Path, clean_db: str) -> None:
    run = uuid.uuid4().hex[:8]
    cfg = AppConfig(
        database_url=clean_db,  # холодный старт: правил нет, worker опубликует только root
        source_dir=tmp_path / "in",
        storage_root=tmp_path / "out",
        queue_name=f"e2e-{run}",
        task_queue=f"e2e-{run}",
        poll_interval_s=0.2,
        metrics_port=0,  # эфемерный порт, 9100 может быть занят dev-worker
        sweep_schedule_id=f"e2e-sweep-{run}",  # не трогать расписание dev-окружения
    )
    cfg.source_dir.mkdir()
    # workflow id = file_id, успешно завершённый id повторно не стартует: id уникален на прогон
    file_id = f"a-{run}"
    (cfg.source_dir / f"{file_id}.txt").write_text("разговор о погоде", encoding="utf-8")

    models = {"decision": ScriptedDecision(), "guard": ScriptedGuard()}
    got: Classification | None = None
    results: list[object] = []
    app = await build_app(cfg, models)
    try:
        publisher = RabbitPublisher(cfg.rabbit_url, cfg.queue_name)
        try:
            source = LocalDirSource(cfg.source_dir, cfg.source_dir / ".seen.json")
            stop = asyncio.Event()
            tasks = [
                asyncio.create_task(run_worker(cfg, models, stop)),
                asyncio.create_task(
                    run_consumer(cfg.rabbit_url, cfg.queue_name, app.transcripts, app.starter, stop)
                ),
                asyncio.create_task(run_poller(source, publisher, cfg.poll_interval_s, stop)),
            ]
            try:
                for _ in range(150):
                    if any(t.done() for t in tasks):
                        break  # упавшая задача: исключение проверяется ниже
                    got = await app.classifications.get(file_id)
                    if got is not None:
                        break
                    await asyncio.sleep(0.2)
            finally:
                stop.set()
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 30)
        finally:
            await publisher.close()
            await _delete_queue(cfg.rabbit_url, cfg.queue_name)
            await _delete_queue(cfg.rabbit_url, dlq_name(cfg.queue_name))
            if app.temporal is not None:
                with contextlib.suppress(RPCError):  # worker мог упасть до создания
                    await app.temporal.get_schedule_handle(cfg.sweep_schedule_id).delete()
    finally:
        await app.close()

    assert [r for r in results if isinstance(r, BaseException)] == []
    assert got is not None, "classification not written within 30s"
    assert got.status == Status.IN_OTHER
    assert got.path == "other"
    assert models["decision"].calls == 0  # у root только other: модель не вызывается
