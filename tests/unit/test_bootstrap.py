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


async def test_build_learner_uses_overrides(tmp_path: Path) -> None:
    from llm_sorter.application.discover import DiscoverService
    from llm_sorter.bootstrap import build_learner
    from tests.fakes import FakeEmbedder, FakeText

    cfg = AppConfig(_env_file=None, sqlite_path=tmp_path / "s.db")  # type: ignore[call-arg]
    discover, metrics, store = await build_learner(
        cfg, metrics_port=0, overrides={"text": FakeText(), "embedder": FakeEmbedder()}
    )
    try:
        assert isinstance(discover, DiscoverService) and isinstance(metrics, NoopMetrics)
    finally:
        await store.close()


def test_guard_uses_own_endpoint_when_set() -> None:
    from llm_sorter.bootstrap import build_models

    cfg = AppConfig(  # type: ignore[call-arg]
        _env_file=None, llm_guard_url="http://guard:4014/v1", llm_guard_api_key="gk"
    )
    models = build_models(cfg)
    guard, decision = models["guard"]._client, models["decision"]._client
    assert str(guard.base_url) == "http://guard:4014/v1/" and guard.api_key == "gk"
    assert str(decision.base_url) == "http://localhost:1234/v1/"


def test_guard_shares_main_endpoint_by_default() -> None:
    from llm_sorter.bootstrap import build_models

    models = build_models(AppConfig(_env_file=None))  # type: ignore[call-arg]
    assert models["guard"]._client is models["decision"]._client
