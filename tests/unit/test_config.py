from pathlib import Path

import pytest
from pydantic import SecretStr

from llm_sorter.config import AppConfig

DEFAULTS = {
    "llm_url": "http://localhost:1234/v1",
    "llm_api_key": SecretStr("lm-studio"),
    "llm_guard_url": "",
    "llm_guard_api_key": SecretStr(""),
    "llm_text_url": "",
    "llm_text_api_key": SecretStr(""),
    "model_decide": "lfm2",
    "decide_protocol": "chat",
    "model_guard": "qwen3-4b",
    "model_text": "qwen3-4b",
    "model_embed": "bge-m3",
    "guard_enabled": True,
    "multi_topic": True,
    "multi_max_choices": 3,
    "embed_max_chars": 1200,
    "embed_batch_size": 8,
    "sqlite_path": Path("./data/sorter.db"),
    "rabbit_url": "amqp://guest:guest@localhost:5672/",
    "inbox_queue": "llm-sorter.inbox",
    "classified_exchange": "llm-sorter.classified",
    "classified_archive": "llm-sorter.classified.all",
    "classified_history": "llm-sorter.classified.history",
    "history_max_age": "30D",
    "history_max_bytes": 10737418240,
    "other_stream": "llm-sorter.other",
    "other_max_age": "30D",
    "other_max_bytes": 10737418240,
    "process_concurrency": 16,
    "tree_refresh_s": 1.0,
    "learner_check_s": 30.0,
    "learner_max_batch": 2000,
    "other_refresh_s": 86400.0,
    "metrics_port": 9100,
    "learner_metrics_port": 9102,
    "source_dir": Path("./data/inbox"),
    "poll_interval_s": 10.0,
    "tg_api_id": 0,
    "tg_api_hash": SecretStr(""),
    "tg_session": SecretStr(""),
    "tg_folders": "",
    "tg_state_file": Path("./data/tg_state.json"),
    "tg_backfill": 20,
    "tg_min_len": 80,
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
