from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.application.topics import TopicsService
from llm_sorter.domain.validation import TreeValidationError

SEED = """
settings: {discover: {buffer_threshold: 7}}
topics:
  - {id: finance, parent: root, title: Финансы, description: Деньги}
  - {id: cards, parent: finance, title: Карты, description: Карты, guard: "Пост про карты банка?"}
"""


@pytest.fixture
async def svc(tmp_path: Path) -> AsyncIterator[TopicsService]:
    store = await SqliteTopicStore.open(tmp_path / "s.db")
    yield TopicsService(store)
    await store.close()


async def test_import_into_empty(svc: TopicsService) -> None:
    t = await svc.import_yaml(SEED)
    assert t.routing_key("cards") == "finance.cards"
    assert t.settings.discover.buffer_threshold == 7


async def test_import_into_non_empty_requires_replace(svc: TopicsService) -> None:
    await svc.import_yaml(SEED)
    with pytest.raises(ValueError, match="not empty"):
        await svc.import_yaml(SEED)
    t = await svc.import_yaml_replace(SEED.replace("Финансы", "Деньги"))
    assert t.get("finance").title == "Деньги"


async def test_export_roundtrip(svc: TopicsService) -> None:
    await svc.import_yaml(SEED)
    text = await svc.export_yaml()
    t = await svc.import_yaml_replace(text)
    assert t.routing_key("cards") == "finance.cards"
    assert t.get("cards").guard == "Пост про карты банка?"


async def test_show_prints_routing_keys(svc: TopicsService) -> None:
    await svc.import_yaml(SEED)
    lines = await svc.show()
    assert lines[0].startswith("/")
    assert any("cards" in ln and "finance.cards" in ln and ln.startswith("    ") for ln in lines)


async def test_cli_edits(svc: TopicsService) -> None:
    await svc.import_yaml(SEED)
    await svc.add("tech", "root", "Технологии", "Гаджеты")
    t = await svc.move("cards", "tech")
    assert t.routing_key("cards") == "tech.cards"
    t = await svc.edit("tech", title="IT")
    assert t.get("tech").title == "IT"
    t = await svc.archive("tech")
    assert "cards" not in t.topics
    with pytest.raises(TreeValidationError):
        await svc.add("Bad", "root", "x", "y")
