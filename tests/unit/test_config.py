from pathlib import Path

import pytest

from llm_sorter.config import AppConfig

DEFAULTS = {
    "lmstudio_url": "http://localhost:1234/v1",
    "lmstudio_api_key": "lm-studio",
    "model_decide": "lfm2",
    "model_guard": "qwen3-4b",
    "model_text": "qwen3-4b",
    "model_embed": "bge-m3",
    "database_url": "postgresql+asyncpg://sorter:sorter@localhost:5432/sorter",
    "rabbit_url": "amqp://guest:guest@localhost:5672/",
    "queue_name": "transcripts",
    "temporal_address": "localhost:7233",
    "temporal_namespace": "default",
    "task_queue": "llm-sorter",
    "storage_root": Path("./data/out"),
    "source_dir": Path("./data/inbox"),
    "poll_interval_s": 10.0,
    "metrics_port": 9100,
    "sweep_interval_s": 600,
    "max_concurrent_activities": 8,
    "sweep_schedule_id": "llm-sorter-sweep",
}


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in AppConfig.model_fields:  # окружение разработчика (.env, export) не влияет
        monkeypatch.delenv(name.upper(), raising=False)
    cfg = AppConfig(_env_file=None)  # type: ignore[call-arg]
    assert set(DEFAULTS) == set(AppConfig.model_fields)
    assert cfg.model_dump() == DEFAULTS


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_DECIDE", "lfm2-1.2b")
    assert AppConfig(_env_file=None).model_decide == "lfm2-1.2b"  # type: ignore[call-arg]
