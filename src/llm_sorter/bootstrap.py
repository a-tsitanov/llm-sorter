from dataclasses import dataclass, fields
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from llm_sorter.adapters.outbound.fs_storage import LocalFileStorage
from llm_sorter.adapters.outbound.lmstudio import (
    LMStudioDecision,
    LMStudioEmbedder,
    LMStudioGuard,
    LMStudioText,
    make_openai_client,
)
from llm_sorter.adapters.outbound.postgres.classifications import PgClassifications
from llm_sorter.adapters.outbound.postgres.embeddings import PgEmbeddings
from llm_sorter.adapters.outbound.postgres.material import PgMaterial
from llm_sorter.adapters.outbound.postgres.rules import PgRules
from llm_sorter.adapters.outbound.postgres.tables import make_engine, make_sessionmaker
from llm_sorter.adapters.outbound.postgres.transcripts import PgTranscripts
from llm_sorter.adapters.outbound.prometheus_metrics import PrometheusMetrics
from llm_sorter.adapters.outbound.rocket_stub import LoggingNotifier
from llm_sorter.adapters.outbound.temporal_starter import TemporalStarter
from llm_sorter.application.actions import ActionService
from llm_sorter.application.classify import ClassifyService
from llm_sorter.application.discover import DiscoverService
from llm_sorter.application.rules import RulesService
from llm_sorter.config import AppConfig
from llm_sorter.ports.io import WorkflowStarter
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.models import DecisionModel, Embedder, GuardModel, TextLLM
from llm_sorter.ports.storage import (
    ClassificationRepository,
    EmbeddingCache,
    FileStorage,
    MaterialRepository,
    Notifier,
    RulesRepository,
    TranscriptStore,
)


@dataclass
class App:
    cfg: AppConfig
    engine: AsyncEngine
    sessions: "async_sessionmaker[AsyncSession]"
    rules: RulesRepository
    transcripts: TranscriptStore
    classifications: ClassificationRepository
    material: MaterialRepository
    embeddings: EmbeddingCache
    storage: FileStorage
    notifier: Notifier
    metrics: Metrics
    starter: WorkflowStarter
    decision: DecisionModel
    guard: GuardModel
    text_llm: TextLLM
    embedder: Embedder
    classify: ClassifyService
    actions: ActionService
    rules_service: RulesService
    discover: DiscoverService
    owns_engine: bool = True
    temporal: Client | None = None  # None, если starter подменён и клиент не передан

    async def close(self) -> None:
        if self.owns_engine:
            await self.engine.dispose()


async def connect_temporal(cfg: AppConfig) -> Client:
    return await Client.connect(
        cfg.temporal_address,
        namespace=cfg.temporal_namespace,
        data_converter=pydantic_data_converter,
    )


async def build_app(cfg: AppConfig, overrides: dict[str, Any] | None = None) -> App:
    """Собирает зависимости; ключи overrides совпадают с именами полей App."""
    ov = dict(overrides or {})
    unknown = (set(ov) - {f.name for f in fields(App)}) | ({"cfg"} & set(ov))
    if unknown:
        raise ValueError(f"unknown overrides: {sorted(unknown)}")

    owns_engine = "engine" not in ov
    engine = ov.pop("engine", None) or make_engine(cfg.database_url)
    sessions = ov.pop("sessions", None) or make_sessionmaker(engine)
    temporal = ov.pop("temporal", None)
    if temporal is None and "starter" not in ov:
        temporal = await connect_temporal(cfg)
    client = make_openai_client(cfg)
    parts: dict[str, Any] = {
        "rules": PgRules(sessions),
        "transcripts": PgTranscripts(sessions),
        "classifications": PgClassifications(sessions),
        "material": PgMaterial(sessions),
        "embeddings": PgEmbeddings(sessions),
        "storage": LocalFileStorage(cfg.storage_root),
        "notifier": LoggingNotifier(),
        "metrics": PrometheusMetrics(),
        "starter": TemporalStarter(temporal, cfg.task_queue) if temporal is not None else None,
        "decision": LMStudioDecision(client, cfg.model_decide),
        "guard": LMStudioGuard(client, cfg.model_guard),
        "text_llm": LMStudioText(client, cfg.model_text),
        "embedder": LMStudioEmbedder(client, cfg.model_embed),
    }
    parts.update({k: v for k, v in ov.items() if k in parts})
    p = parts
    classify = ov.get("classify") or ClassifyService(
        p["rules"],
        p["transcripts"],
        p["classifications"],
        p["material"],
        p["decision"],
        p["guard"],
        p["starter"],
        p["metrics"],
    )
    actions = ov.get("actions") or ActionService(
        p["transcripts"],
        p["storage"],
        p["notifier"],
        p["text_llm"],
        p["rules"],
        p["metrics"],
        "ru",  # запасной язык; рабочий берётся из settings.language активного дерева
    )
    rules_service = ov.get("rules_service") or RulesService(
        p["rules"], p["material"], p["starter"], p["metrics"]
    )
    discover = ov.get("discover") or DiscoverService(
        p["rules"],
        rules_service,
        p["material"],
        p["transcripts"],
        p["embeddings"],
        p["embedder"],
        p["text_llm"],
        p["metrics"],
    )
    return App(
        cfg=cfg,
        engine=engine,
        sessions=sessions,
        classify=classify,
        actions=actions,
        rules_service=rules_service,
        discover=discover,
        owns_engine=owns_engine,
        temporal=temporal,
        **p,
    )
