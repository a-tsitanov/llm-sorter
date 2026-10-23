"""Сквозной прогон: feed dir -> process -> learn -> перегон -> новая тема в архиве."""

import asyncio
import json
from pathlib import Path

import pytest
from aio_pika.abc import AbstractRobustConnection

from llm_sorter.adapters.inbound.learner_runner import run_learner
from llm_sorter.adapters.inbound.poller import run_poller
from llm_sorter.adapters.inbound.processor import TreeCache, run_processor
from llm_sorter.adapters.inbound.stub_source import LocalDirSource
from llm_sorter.adapters.outbound.rabbit import InboxPublisher, Names
from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.discover import Accepted, DiscoverResult
from llm_sorter.application.process import ProcessService
from llm_sorter.application.topics import TopicsService
from llm_sorter.domain.traversal import Candidate
from llm_sorter.domain.tree import ROOT_ID, Tree
from tests.fakes import RecordingMetrics
from tests.helpers import topic
from tests.integration.conftest import drain

pytestmark = pytest.mark.integration

SEED = """
settings: {discover: {buffer_threshold: 10}}
topics:
  - {id: finance, parent: root, title: Финансы, description: Деньги}
  - {id: cards, parent: finance, title: Карты, description: Карты, guard: "Разговор о картах?"}
  - {id: travel, parent: root, title: Поездки, description: Поездки}
  - {id: flights, parent: travel, title: Рейсы, description: Рейсы, guard: "Разговор о рейсах?"}
"""
KEYWORDS = {
    "finance": "карт",
    "cards": "карт",
    "travel": "рейс",
    "flights": "рейс",
    "taxes": "налог",
}
# оба процесса объявляют stream: аргументы должны совпадать
MAX_AGE, MAX_BYTES = "30D", 10**9


class KeywordDecision:
    async def decide(self, text: str, candidates: list[Candidate]) -> str:
        for c in candidates:
            if KEYWORDS.get(c.id, "\0") in text:
                return c.id
        return "other"


class PassGuard:
    async def check(self, text: str, question: str) -> bool:
        return True


class TaxesAtRoot:
    """Discover только на root: всё, что про налоги, - новая тема."""

    async def propose(self, tree: Tree, node_id: str, texts: list[str]) -> DiscoverResult:
        members = [i for i, t in enumerate(texts) if "налог" in t]
        if node_id != ROOT_ID or len(members) < 10:
            return DiscoverResult([], list(range(len(texts))), 0)
        t = topic("taxes", ROOT_ID, guard="Разговор о налогах?", origin="discovered")
        rest = [i for i in range(len(texts)) if i not in members]
        return DiscoverResult([Accepted(t, members)], rest, 0)


async def test_end_to_end(
    rabbit_url: str, names: Names, conn: AbstractRobustConnection, tmp_path: Path
) -> None:
    store = await SqliteTopicStore.open(tmp_path / "s.db")
    await TopicsService(store).import_yaml(SEED)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i in range(10):
        (inbox / f"card-{i}.txt").write_text(f"заблокировали карту {i}", encoding="utf-8")
        (inbox / f"fly-{i}.txt").write_text(f"задержали рейс {i}", encoding="utf-8")
        (inbox / f"tax-{i}.txt").write_text(f"налоговый вычет {i}", encoding="utf-8")

    stop = asyncio.Event()
    service = ProcessService(KeywordDecision(), PassGuard(), RecordingMetrics())
    publisher = InboxPublisher(rabbit_url, names)
    tasks = [
        asyncio.create_task(
            run_processor(
                rabbit_url, names, 8, service, TreeCache(store, 0.1), stop, MAX_AGE, MAX_BYTES
            )
        ),
        asyncio.create_task(
            run_learner(
                rabbit_url,
                names,
                store,
                TaxesAtRoot(),
                RecordingMetrics(),
                0.2,
                3600.0,
                MAX_AGE,
                MAX_BYTES,
                stop,
            )
        ),
    ]
    await asyncio.sleep(0.5)  # топология объявлена
    tasks.append(
        asyncio.create_task(
            run_poller(LocalDirSource(inbox, inbox / ".seen.json"), publisher, 0.2, stop)
        )
    )
    try:
        ch = await conn.channel()
        archived = await drain(ch, names.archive, 30, timeout=30.0)
        keys = sorted(m.routing_key for m in archived)
        assert keys == ["finance.cards"] * 10 + ["taxes"] * 10 + ["travel.flights"] * 10
        tax = [json.loads(m.body) for m in archived if m.routing_key == "taxes"]
        assert all(b["tree_rev"] >= 2 for b in tax)  # классифицированы по новому дереву
        log = await store.changelog()
        assert [(e.actor, e.op, e.topic_id) for e in log if e.actor == "learner"] == [
            ("learner", "add", "taxes")
        ]
    finally:
        stop.set()
        await asyncio.gather(*tasks)
        await publisher.close()
        await store.close()
