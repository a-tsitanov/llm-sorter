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

    async def refresh_if_due(self, every: timedelta, max_age: timedelta) -> int | None:
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
    await maintenance_loop(lr, pos, 30.0, 3600.0, timedelta(days=15), stop, waits)  # type: ignore[arg-type]
    assert waits.delays == [1.0, 2.0, 30.0]
    assert lr.runs == 3 and lr.saved == [42, 42, 42]  # позиция сохраняется и при сбое discover


async def test_refresh_checked_every_tick() -> None:
    stop = asyncio.Event()
    lr, pos = FakeLearner(), StreamPosition(next=0)
    await maintenance_loop(lr, pos, 30.0, 50.0, timedelta(days=15), stop, Waits(stop, stop_after=2))  # type: ignore[arg-type]
    assert lr.refreshed == [timedelta(days=15)] * 2  # решает refresh_if_due по времени в SQLite


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


async def test_no_discover_until_caught_up() -> None:
    stop = asyncio.Event()
    lr, pos = FakeLearner(), StreamPosition(next=0)
    ready = iter([False, False, True])
    waits = Waits(stop, stop_after=3)
    await maintenance_loop(
        lr,
        pos,
        30.0,
        3600.0,
        timedelta(days=15),
        stop,
        waits,  # type: ignore[arg-type]
        ready=lambda: next(ready, True),
    )
    assert lr.runs == 1 and lr.saved == [0]  # только после догонялки
    assert waits.delays[:2] == [0.2, 0.2]


async def test_offset_saved_even_when_discover_keeps_failing() -> None:
    stop = asyncio.Event()
    lr, pos = FakeLearner(fails=99), StreamPosition(next=5)
    await maintenance_loop(
        lr,
        pos,
        30.0,
        0.0,
        timedelta(days=15),
        stop,
        Waits(stop, stop_after=2),  # type: ignore[arg-type]
    )
    assert lr.saved == [5, 5] and len(lr.refreshed) == 2


@pytest.mark.parametrize(
    "saved,first,lost", [(None, 0, 0), (None, 500, 0), (10, 10, 0), (10, 25, 15), (10, 3, 0)]
)
def test_stream_gap(saved: int | None, first: int, lost: int) -> None:
    """Ретеншн по размеру удалил записи до сохранённой позиции - их нужно заметить."""
    from llm_sorter.adapters.inbound.learner_runner import stream_gap

    assert stream_gap(saved, first) == lost


async def test_reconsider_runs_once_after_catch_up() -> None:
    class L(FakeLearner):
        reconsidered = 0

        async def reconsider_all(self) -> int:
            self.reconsidered += 1
            return 0

    stop = asyncio.Event()
    lr = L()
    await maintenance_loop(
        lr,
        StreamPosition(next=0),
        30.0,
        3600.0,
        timedelta(days=15),
        stop,  # type: ignore[arg-type]
        Waits(stop, stop_after=2),
        reconsider=True,
    )
    assert lr.reconsidered == 1 and lr.runs == 2
