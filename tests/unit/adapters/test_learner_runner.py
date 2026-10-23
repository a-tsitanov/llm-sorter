import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from llm_sorter.adapters.inbound.learner_runner import (
    LearnerLock,
    StreamPosition,
    maintenance_loop,
    stream_offset,
)


def test_second_lock_fails(tmp_path: Path) -> None:
    path = tmp_path / "s.db.learner.lock"
    with LearnerLock(path):
        with pytest.raises(RuntimeError, match="another learner"):
            with LearnerLock(path):
                pass
    with LearnerLock(path):  # после выхода блокировка свободна
        pass


@pytest.mark.parametrize(
    "headers,offset", [({"x-stream-offset": 7}, 7), ({"x-stream-offset": 0}, 0)]
)
def test_stream_offset_header(headers: dict[str, Any], offset: int) -> None:
    assert stream_offset(headers) == offset


def test_stream_offset_missing_raises() -> None:
    with pytest.raises(ValueError):
        stream_offset({})


class FakeLearner:
    def __init__(self, fails: int = 0) -> None:
        self.fails = fails
        self.runs = 0
        self.saved: list[int] = []
        self.refreshed: list[timedelta] = []

    async def run_due(self) -> list[str]:
        self.runs += 1
        if self.fails:
            self.fails -= 1
            raise ConnectionError("llm down")
        return []

    async def save_offset(self, next_offset: int) -> None:
        self.saved.append(next_offset)

    async def refresh_old(self, max_age: timedelta) -> int:
        self.refreshed.append(max_age)
        return 0


class Waits:
    """wait(stop, seconds): записывает задержки, после `stop_after` выставляет stop."""

    def __init__(self, stop: asyncio.Event, stop_after: int) -> None:
        self.delays: list[float] = []
        self._stop, self._after = stop, stop_after

    async def __call__(self, stop: asyncio.Event, seconds: float) -> None:
        self.delays.append(seconds)
        if len(self.delays) >= self._after:
            self._stop.set()


async def test_run_due_error_backs_off_then_saves_offset() -> None:
    stop = asyncio.Event()
    lr, pos = FakeLearner(fails=2), StreamPosition(next=42)
    waits = Waits(stop, stop_after=3)
    clock = iter([0.0, 0.0, 0.0, 0.0, 0.0])
    await maintenance_loop(
        lr, pos, 30.0, 3600.0, timedelta(days=15), stop, waits, lambda: next(clock)
    )  # type: ignore[arg-type]
    assert waits.delays == [1.0, 2.0, 30.0]
    assert lr.runs == 3 and lr.saved == [42]


async def test_refresh_runs_when_due() -> None:
    stop = asyncio.Event()
    lr, pos = FakeLearner(), StreamPosition(next=0)
    times = iter([0.0, 10.0, 100.0, 100.0, 100.0])
    await maintenance_loop(
        lr,
        pos,
        30.0,
        50.0,
        timedelta(days=15),
        stop,
        Waits(stop, stop_after=2),
        lambda: next(times),  # type: ignore[arg-type]
    )
    assert lr.refreshed == [timedelta(days=15)]


@pytest.mark.parametrize(
    "value,expected",
    [("30D", timedelta(days=30)), ("12h", timedelta(hours=12)), ("90s", timedelta(seconds=90))],
)
def test_max_age_timedelta(value: str, expected: timedelta) -> None:
    from llm_sorter.adapters.inbound.learner_runner import max_age_timedelta

    assert max_age_timedelta(value) == expected


@pytest.mark.parametrize("bad", ["30", "D", "3x", "-1D"])
def test_max_age_timedelta_rejects(bad: str) -> None:
    from llm_sorter.adapters.inbound.learner_runner import max_age_timedelta

    with pytest.raises(ValueError):
        max_age_timedelta(bad)
