"""Единственное место, где адаптеры связываются с сервисами; тесты подменяют их через overrides."""

from typing import Any

from openai import AsyncOpenAI
from prometheus_client import start_http_server
from pydantic import SecretStr

from llm_sorter.adapters.inbound.processor import TreeCache
from llm_sorter.adapters.outbound.llm import (
    LLMDecision,
    LLMEmbedder,
    LLMGuard,
    LLMText,
    SystemOneDecision,
    make_openai_client,
)
from llm_sorter.adapters.outbound.noop_metrics import NoopMetrics
from llm_sorter.adapters.outbound.prometheus_metrics import PrometheusMetrics
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.discover import DiscoverService
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
    main_key = cfg.llm_api_key.get_secret_value()
    client = make_openai_client(cfg.llm_url, main_key)

    def own_client(url: str, key: SecretStr) -> AsyncOpenAI:
        if not url:
            return client
        return make_openai_client(url, key.get_secret_value() or main_key)

    decision_cls = SystemOneDecision if cfg.decide_protocol == "systemone" else LLMDecision
    models: dict[str, Any] = {
        "decision": decision_cls(client, cfg.model_decide),
        "guard": LLMGuard(own_client(cfg.llm_guard_url, cfg.llm_guard_api_key), cfg.model_guard),
        "text": LLMText(own_client(cfg.llm_text_url, cfg.llm_text_api_key), cfg.model_text),
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


async def build_learner(
    cfg: AppConfig, metrics_port: int, overrides: dict[str, Any] | None = None
) -> tuple[DiscoverService, Metrics, SqliteTopicStore]:
    models = build_models(cfg, overrides)
    metrics = build_metrics(metrics_port)
    store = await SqliteTopicStore.open(cfg.sqlite_path)
    return DiscoverService(models["embedder"], models["text"], metrics), metrics, store
