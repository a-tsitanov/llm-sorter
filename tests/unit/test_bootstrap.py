from pathlib import Path

import pytest

from llm_sorter.adapters.outbound.noop_metrics import NoopMetrics
from llm_sorter.bootstrap import build_metrics, build_processor
from llm_sorter.config import AppConfig
from tests.fakes import FakeDecision, FakeGuard


def test_metrics_port_zero_disables() -> None:
    assert isinstance(build_metrics(0), NoopMetrics)


async def test_build_processor_uses_overrides(tmp_path: Path) -> None:
    cfg = AppConfig(_env_file=None, sqlite_path=tmp_path / "s.db")  # type: ignore[call-arg]
    dec = FakeDecision([])
    service, cache, store = await build_processor(
        cfg, metrics_port=0, overrides={"decision": dec, "guard": FakeGuard([])}
    )
    try:
        assert list((await cache.get()).topics) == ["root"]
        assert service._decision is dec
    finally:
        await store.close()


async def test_unknown_override_rejected(tmp_path: Path) -> None:
    cfg = AppConfig(_env_file=None, sqlite_path=tmp_path / "s.db")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unknown overrides"):
        await build_processor(cfg, metrics_port=0, overrides={"nope": 1})
