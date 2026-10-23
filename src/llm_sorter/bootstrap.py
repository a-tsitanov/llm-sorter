"""Единственное место, где адаптеры связываются с сервисами; тесты подменяют их через overrides."""

from typing import Any

from prometheus_client import start_http_server

from llm_sorter.adapters.inbound.processor import TreeCache
from llm_sorter.adapters.outbound.llm import (
    LLMDecision,
    LLMEmbedder,
    LLMGuard,
    LLMText,
    make_openai_client,
)
from llm_sorter.adapters.outbound.noop_metrics import NoopMetrics
from llm_sorter.adapters.outbound.prometheus_metrics import PrometheusMetrics
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.process import ProcessService
from llm_sorter.config import AppConfig
from llm_sorter.ports.metrics import Metrics

MODEL_KEYS = {"decision", "guard", "text", "embedder"}


def build_metrics(port: int) -> Metrics:
    """port 0 - без метрик (несколько обработчиков на одной машине не делят порт)."""
    if port == 0:
        return NoopMetrics()
    m = PrometheusMetrics()
    start_http_server(port, registry=m.registry)
    return m


def build_models(cfg: AppConfig, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    ov = dict(overrides or {})
    unknown = set(ov) - MODEL_KEYS
    if unknown:
        raise ValueError(f"unknown overrides: {sorted(unknown)}")
    if MODEL_KEYS <= set(ov):
        return ov
    client = make_openai_client(cfg.llm_url, cfg.llm_api_key.get_secret_value())
    models: dict[str, Any] = {
        "decision": LLMDecision(client, cfg.model_decide),
        "guard": LLMGuard(client, cfg.model_guard),
        "text": LLMText(client, cfg.model_text),
        "embedder": LLMEmbedder(client, cfg.model_embed),
    }
    return {**models, **ov}


async def build_processor(
    cfg: AppConfig, metrics_port: int, overrides: dict[str, Any] | None = None
) -> tuple[ProcessService, TreeCache, SqliteTopicStore]:
    models = build_models(cfg, overrides)
    store = await SqliteTopicStore.open(cfg.sqlite_path)
    service = ProcessService(models["decision"], models["guard"], build_metrics(metrics_port))
    return service, TreeCache(store, cfg.tree_refresh_s), store
