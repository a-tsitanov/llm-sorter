"""Smoke: расписание sweep создаётся один раз. Требует Temporal из `make up`."""

import uuid
from datetime import timedelta

import pytest
from temporalio.client import ScheduleIntervalSpec, ScheduleOverlapPolicy

from llm_sorter.bootstrap import connect_temporal
from llm_sorter.config import AppConfig
from llm_sorter.orchestration.worker import ensure_sweep_schedule

pytestmark = pytest.mark.integration


async def test_ensure_sweep_schedule_idempotent() -> None:
    run = uuid.uuid4().hex[:8]
    cfg = AppConfig(
        sweep_schedule_id=f"it-sweep-{run}", task_queue=f"it-{run}", sweep_interval_s=90
    )
    client = await connect_temporal(cfg)
    handle = client.get_schedule_handle(cfg.sweep_schedule_id)
    try:
        assert await ensure_sweep_schedule(client, cfg) is True
        assert await ensure_sweep_schedule(client, cfg) is False
        desc = await handle.describe()
        assert desc.schedule.spec.intervals == [ScheduleIntervalSpec(every=timedelta(seconds=90))]
        assert desc.schedule.policy.overlap == ScheduleOverlapPolicy.SKIP
    finally:
        await handle.delete()
