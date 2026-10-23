import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from llm_sorter.adapters.outbound.sqlite_store import SqliteTopicStore
from llm_sorter.domain.changes import AddTopic, ArchiveTopic, EditTopic, MoveTopic
from llm_sorter.domain.tree import ROOT_ID, Settings
from llm_sorter.domain.validation import TreeValidationError
from tests.helpers import topic

Opener = Callable[[], Awaitable[SqliteTopicStore]]


@pytest.fixture
async def open_store(tmp_path: Path) -> AsyncIterator[Opener]:
    """Открывает коннекты к одному файлу и закрывает их все после теста."""
    opened: list[SqliteTopicStore] = []

    async def go() -> SqliteTopicStore:
        opened.append(await SqliteTopicStore.open(tmp_path / "s.db"))
        return opened[-1]

    yield go
    for s in opened:
        await s.close()


async def test_empty_db_has_root(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    t = await s.load_tree()
    assert list(t.topics) == [ROOT_ID] and t.rev == 0
    assert await s.tree_rev() == 0


async def test_wal_mode(open_store: Opener, tmp_path: Path) -> None:
    await open_store()
    with sqlite3.connect(tmp_path / "s.db") as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA user_version").fetchone()[0] == 1


async def test_apply_persists_and_logs(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    t = await s.apply([AddTopic(topic=topic("finance", examples=["a", "b"]))], "cli")
    assert t.rev == 1
    s2 = await open_store()  # второй коннект видит изменения
    t2 = await s2.load_tree()
    assert t2.routing_key("finance") == "finance" and t2.get("finance").examples == ["a", "b"]
    assert await s2.tree_rev() == 1
    [e] = await s2.changelog()
    assert (e.actor, e.op, e.topic_id) == ("cli", "add", "finance")


async def test_children_order_survives_reload(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    await s.apply([AddTopic(topic=topic("zeta")), AddTopic(topic=topic("alpha"))], "cli")
    t = await (await open_store()).load_tree()
    assert [c.id for c in t.children(ROOT_ID)] == ["zeta", "alpha"]


async def test_invalid_change_writes_nothing(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    with pytest.raises(TreeValidationError):
        await s.apply([AddTopic(topic=topic("ok")), AddTopic(topic=topic("Bad"))], "cli")
    assert await s.tree_rev() == 0
    assert await s.changelog() == []
    assert list((await s.load_tree()).topics) == [ROOT_ID]


async def test_move_edit_logged(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    await s.apply([AddTopic(topic=topic("a")), AddTopic(topic=topic("b"))], "cli")
    t = await s.apply(
        [MoveTopic(id="b", new_parent_id="a"), EditTopic(id="a", fields={"title": "A2"})],
        "learner",
    )
    assert t.routing_key("b") == "a.b" and t.get("a").title == "A2"
    log = await s.changelog()
    assert [(e.op, e.topic_id, e.actor) for e in log[:2]] == [
        ("edit", "a", "learner"),
        ("move", "b", "learner"),
    ]  # новые первыми
    assert log[1].details == {"from": "root", "to": "a"}
    reloaded = await (await open_store()).load_tree()
    assert reloaded.routing_key("b") == "a.b"


async def test_archive_hides_subtree_but_keeps_rows(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    await s.apply([AddTopic(topic=topic("a")), AddTopic(topic=topic("b", "a"))], "cli")
    t = await s.apply([ArchiveTopic(id="a")], "cli")
    assert list(t.topics) == [ROOT_ID]
    assert list((await s.load_tree()).topics) == [ROOT_ID]
    with sqlite3.connect(tmp_path / "s.db") as c:
        rows = c.execute("SELECT id FROM topics WHERE archived_at IS NOT NULL").fetchall()
    assert sorted(r[0] for r in rows) == ["a", "b"]


async def test_archived_id_can_be_reused(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    await s.apply([AddTopic(topic=topic("a"))], "cli")
    await s.apply([ArchiveTopic(id="a")], "cli")
    t = await s.apply([AddTopic(topic=topic("a", title="Снова"))], "cli")
    assert t.get("a").title == "Снова"


async def test_settings_roundtrip_bumps_rev(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    new = Settings(
        max_depth=4, discover=Settings().discover.model_copy(update={"buffer_threshold": 7})
    )
    t = await s.save_settings(new)
    assert t.rev == 1
    reloaded = await (await open_store()).load_tree()
    assert reloaded.settings == new
    assert (await s.changelog())[0].op == "settings"


async def test_state_roundtrip(open_store: Opener, tmp_path: Path) -> None:
    s = await open_store()
    assert await s.get_state("other_offset") is None
    await s.set_state("other_offset", "42")
    await s.set_state("other_offset", "43")
    assert await (await open_store()).get_state("other_offset") == "43"


async def test_load_tree_rev_never_newer_than_topics(open_store: Opener) -> None:
    """Запись, закоммиченная посреди load_tree, не даёт дерево с новым rev и старыми темами."""
    writer = await open_store()

    class Racy(SqliteTopicStore):
        raced = False

        async def tree_rev(self) -> int:
            # коммит другого процесса ровно в момент чтения ревизии
            if not self.raced:
                self.raced = True
                await writer.apply([AddTopic(topic=topic("late"))], "cli")
            return await super().tree_rev()

    reader = await open_store()
    racy = Racy(reader._db)
    t = await racy.load_tree()
    # либо старый rev (кеш перечитает позже), либо новый вместе с новой темой
    assert t.rev == 0 or "late" in t.topics
