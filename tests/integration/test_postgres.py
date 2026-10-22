"""Интеграционные тесты Postgres-адаптеров. Требуют `make up`; БД *_test готовит conftest."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_sorter.adapters.outbound.postgres import (
    PgClassifications,
    PgEmbeddings,
    PgMaterial,
    PgRules,
    PgTranscripts,
    make_engine,
    make_sessionmaker,
)
from llm_sorter.domain.classification import (
    Classification,
    FinalClassificationExists,
    Status,
    Transcript,
)
from llm_sorter.domain.loader import load_tree
from llm_sorter.ports.storage import VersionConflict

pytestmark = pytest.mark.integration

TREE_YAML = """
node:
  id: root
  children:
    - id: finance
      title: Финансы
      description: Деньги
"""


@pytest.fixture
async def sessions(clean_db: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = make_engine(clean_db)
    yield make_sessionmaker(engine)
    await engine.dispose()


async def test_publish_and_active(sessions: async_sessionmaker[AsyncSession]) -> None:
    rules = PgRules(sessions)
    with pytest.raises(LookupError):
        await rules.active()
    tree = load_tree(TREE_YAML)
    v1 = await rules.publish(tree, None, "manual")
    assert v1 == 1
    active = await rules.active()
    assert active.version == 1
    assert active.root.uid == tree.root.uid
    assert [c.id for c in active.root.children] == ["finance"]
    v2 = await rules.publish(active, v1, "discover")
    assert v2 == 2
    assert (await rules.active()).version == 2
    old = await PgRules(sessions).get(1)  # свежий экземпляр без кэша
    assert old.version == 1
    with pytest.raises(LookupError):
        await rules.get(99)


async def test_publish_conflict(sessions: async_sessionmaker[AsyncSession]) -> None:
    rules = PgRules(sessions)
    tree = load_tree(TREE_YAML)
    v1 = await rules.publish(tree, None, "manual")
    await rules.publish(tree, v1, "manual")
    with pytest.raises(VersionConflict):
        await rules.publish(tree, v1, "manual")
    with pytest.raises(VersionConflict):  # активная есть, base=None
        await rules.publish(tree, None, "manual")


async def test_publish_concurrent_one_wins(sessions: async_sessionmaker[AsyncSession]) -> None:
    rules = PgRules(sessions)
    tree = load_tree(TREE_YAML)
    v1 = await rules.publish(tree, None, "manual")
    results = await asyncio.gather(
        *(rules.publish(tree, v1, "discover") for _ in range(4)), return_exceptions=True
    )
    assert sum(isinstance(r, int) for r in results) == 1
    assert sum(isinstance(r, VersionConflict) for r in results) == 3


async def test_publish_first_concurrent_one_wins(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    tree = load_tree(TREE_YAML)
    rules = PgRules(sessions)
    results = await asyncio.gather(
        *(rules.publish(tree, None, "manual") for _ in range(4)), return_exceptions=True
    )
    assert sum(isinstance(r, int) for r in results) == 1
    assert sum(isinstance(r, VersionConflict) for r in results) == 3
    assert (await rules.active()).version == 1


async def test_classification_concurrent_new_row_final_wins(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    repo = PgClassifications(sessions)
    for i in range(10):
        fid = f"race{i}"
        other = Classification(
            file_id=fid, tree_version=1, node_uid="n", path="a/other", status=Status.IN_OTHER
        )
        final = other.model_copy(update={"status": Status.CLASSIFIED, "path": "a/b"})
        calls = [repo.save(other), repo.save(final)]
        if i % 2:
            calls.reverse()
        results = await asyncio.gather(*calls, return_exceptions=True)
        for r in results:
            assert r is None or isinstance(r, FinalClassificationExists)
        assert await repo.get(fid) == final


async def test_classification_final_immutable(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    repo = PgClassifications(sessions)
    assert await repo.get("f1") is None
    other = Classification(
        file_id="f1", tree_version=1, node_uid="n1", path="a/other", status=Status.IN_OTHER
    )
    await repo.save(other)
    await repo.save(other.model_copy(update={"tree_version": 2}))  # in_other перезаписываем
    final = Classification(
        file_id="f1",
        tree_version=2,
        node_uid="n2",
        path="a/b",
        status=Status.CLASSIFIED,
        run_id="r1",
    )
    await repo.save(final)
    assert await repo.get("f1") == final
    with pytest.raises(FinalClassificationExists):
        await repo.save(other)
    assert await repo.get("f1") == final
    failed = final.model_copy(update={"file_id": "f2", "status": Status.GUARD_FAILED})
    await repo.save(failed)
    with pytest.raises(FinalClassificationExists):
        await repo.save(failed)


async def test_material_count_new_resets_after_mark(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    m = PgMaterial(sessions)
    await m.add("n", "a", "other")
    await m.add("n", "b", "leaf")
    await m.add("n", "a", "other")  # идемпотентно
    assert await m.count_new("n") == 2
    assert await m.other_size("n") == 1
    assert await m.due_nodes(2) == ["n"]
    assert await m.due_nodes(3) == []
    await m.mark_discovered("n")
    assert await m.count_new("n") == 0
    assert await m.due_nodes(1) == []
    await m.add("n", "c", "other")
    assert await m.count_new("n") == 1
    assert await m.due_nodes(1) == ["n"]
    assert await m.other_size("n") == 2
    assert [i.file_id for i in await m.list("n")] == ["a", "b", "c"]
    await m.remove("n", "a")
    assert [i.file_id for i in await m.list("n")] == ["b", "c"]


async def test_move_other(sessions: async_sessionmaker[AsyncSession]) -> None:
    m = PgMaterial(sessions)
    await m.add("src", "a", "other")
    await m.add("src", "b", "other")
    await m.add("src", "l", "leaf")
    await m.add("dst", "b", "leaf")  # уже есть в приёмнике: не затирается
    await m.move_other("src", "dst")
    assert [(i.file_id, i.kind) for i in await m.list("src")] == [("l", "leaf")]
    assert {(i.file_id, i.kind) for i in await m.list("dst")} == {("a", "other"), ("b", "leaf")}
    assert await m.other_size("dst") == 1


async def test_transcript_put_dedup(sessions: async_sessionmaker[AsyncSession]) -> None:
    store = PgTranscripts(sessions)
    t = Transcript(id="t1", text="привет", metadata={"k": [1, 2]}, received_at=datetime.now(UTC))
    assert await store.put(t) is True
    assert await store.put(t.model_copy(update={"text": "другое"})) is False
    got = await store.get("t1")
    assert got == t
    with pytest.raises(LookupError):
        await store.get("nope")


async def test_embeddings_roundtrip(sessions: async_sessionmaker[AsyncSession]) -> None:
    cache = PgEmbeddings(sessions)
    assert await cache.get_many([]) == {}
    v1 = [i / 1024 for i in range(1024)]
    v2 = [0.5] * 1024
    await cache.put_many({"a": v1, "b": v2})
    got = await cache.get_many(["a", "b", "missing"])
    assert set(got) == {"a", "b"}
    assert got["a"] == pytest.approx(v1, abs=1e-6)
    await cache.put_many({"a": v2})  # upsert
    assert (await cache.get_many(["a"]))["a"] == pytest.approx(v2, abs=1e-6)


async def test_material_mark_with_cutoff_keeps_later_files_new(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    from sqlalchemy import text

    m = PgMaterial(sessions)
    await m.add("n", "a", "other")
    async with sessions() as s:
        cutoff = await s.scalar(text("SELECT clock_timestamp()"))
    await m.add("n", "b", "other")
    await m.mark_discovered("n", at=cutoff)
    assert await m.count_new("n") == 1
    assert await m.due_nodes(1) == ["n"]
