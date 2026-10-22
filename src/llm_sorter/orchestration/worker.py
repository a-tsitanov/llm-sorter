"""Запуск Temporal worker со всеми workflow и activity."""

import asyncio
import logging
from datetime import timedelta
from typing import Any

from prometheus_client import start_http_server
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleIntervalSpec,
    ScheduleOverlapPolicy,
    SchedulePolicy,
    ScheduleSpec,
)
from temporalio.worker import Worker

from llm_sorter.bootstrap import build_app, connect_temporal
from llm_sorter.config import AppConfig
from llm_sorter.orchestration.activities import Activities
from llm_sorter.orchestration.workflows import ClassifyWorkflow, DiscoverWorkflow, SweepWorkflow

log = logging.getLogger(__name__)


async def ensure_sweep_schedule(client: Client, cfg: AppConfig) -> bool:
    """Создаёт расписание SweepWorkflow, если его нет. True — создано сейчас."""
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            SweepWorkflow.run,
            id=f"{cfg.sweep_schedule_id}-run",
            task_queue=cfg.task_queue,
        ),
        spec=ScheduleSpec(
            intervals=[ScheduleIntervalSpec(every=timedelta(seconds=cfg.sweep_interval_s))]
        ),
        policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
    )
    try:
        await client.create_schedule(cfg.sweep_schedule_id, schedule)
    except ScheduleAlreadyRunningError:
        return False
    log.info("sweep schedule %s created", cfg.sweep_schedule_id)
    return True


async def run_worker(
    cfg: AppConfig,
    overrides: dict[str, Any] | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    """Работает до stop (или отмены задачи). overrides — для тестов, см. build_app."""
    app = await build_app(cfg, overrides)
    http_server = None
    try:
        await app.rules_service.bootstrap_if_empty()
        try:
            version = (await app.rules.active()).version
            if version is not None:
                app.metrics.active_version(version)
        except LookupError:
            pass  # активных правил нет
        client = app.temporal or await connect_temporal(cfg)
        acts = Activities(
            classify=app.classify,
            actions=app.actions,
            discover=app.discover,
            starter=app.starter,
            metrics=app.metrics,
        )
        registry = getattr(app.metrics, "registry", None)
        if registry is not None:  # Noop-метрики из тестов порт не открывают
            http_server, _ = start_http_server(cfg.metrics_port, registry=registry)
        worker = Worker(
            client,
            task_queue=cfg.task_queue,
            workflows=[ClassifyWorkflow, DiscoverWorkflow, SweepWorkflow],
            activities=acts.all(),
            max_concurrent_activities=cfg.max_concurrent_activities,
        )
        await ensure_sweep_schedule(client, cfg)
        log.info("worker started on %s", cfg.task_queue)
        async with worker:
            await (stop or asyncio.Event()).wait()
    finally:
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        await app.close()
