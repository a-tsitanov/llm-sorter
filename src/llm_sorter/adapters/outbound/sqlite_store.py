"""TopicStore на SQLite (WAL): обработчики читают, дообучение и CLI пишут."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from llm_sorter.domain.changes import (
    AddTopic,
    ArchiveTopic,
    Change,
    EditTopic,
    MoveTopic,
    apply_changes,
)
from llm_sorter.domain.tree import ROOT_ID, Settings, Topic, Tree
from llm_sorter.ports.store import Actor, ChangelogEntry

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    seq INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    examples TEXT NOT NULL DEFAULT '[]',
    guard TEXT,
    origin TEXT NOT NULL DEFAULT 'manual',
    max_depth INTEGER,
    created_at TEXT NOT NULL,
    archived_at TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    op TEXT NOT NULL,
    topic_id TEXT,
    details TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS learner_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
_SETTINGS_KEY = "settings"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteTopicStore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    @classmethod
    async def open(cls, path: Path) -> "SqliteTopicStore":
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: транзакции явные (BEGIN IMMEDIATE), чтение без блокировок
        db = await aiosqlite.connect(path, isolation_level=None)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.executescript(_SCHEMA)
        await db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        await db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('tree_rev', 0)")
        await db.execute(
            "INSERT OR IGNORE INTO topics(id, parent_id, seq, created_at) VALUES (?, NULL, 0, ?)",
            (ROOT_ID, _now()),
        )
        return cls(db)

    async def close(self) -> None:
        await self._db.close()

    async def tree_rev(self) -> int:
        async with self._db.execute("SELECT value FROM meta WHERE key='tree_rev'") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def _settings(self) -> Settings:
        async with self._db.execute(
            "SELECT value FROM settings WHERE key=?", (_SETTINGS_KEY,)
        ) as cur:
            row = await cur.fetchone()
        return Settings.model_validate_json(row[0]) if row else Settings()

    async def load_tree(self) -> Tree:
        # ревизия раньше тем: если запись закоммитят между чтениями, дерево получит
        # старый rev с новыми темами и кеш обработчика перечитает его, а не наоборот
        rev = await self.tree_rev()
        async with self._db.execute(
            "SELECT id, parent_id, title, description, examples, guard, origin, max_depth "
            "FROM topics WHERE archived_at IS NULL ORDER BY seq"
        ) as cur:
            rows = await cur.fetchall()
        topics = {
            r[0]: Topic(
                id=r[0],
                parent_id=r[1],
                title=r[2],
                description=r[3],
                examples=json.loads(r[4]),
                guard=r[5],
                origin=r[6],
                max_depth=r[7],
            )
            for r in rows
        }
        return Tree(rev=rev, settings=await self._settings(), topics=topics)

    async def _log(self, actor: Actor, op: str, topic_id: str | None, details: Any) -> None:
        await self._db.execute(
            "INSERT INTO changelog(at, actor, op, topic_id, details) VALUES (?, ?, ?, ?, ?)",
            (_now(), actor, op, topic_id, json.dumps(details, ensure_ascii=False)),
        )

    async def _bump(self) -> None:
        await self._db.execute("UPDATE meta SET value = value + 1 WHERE key='tree_rev'")

    async def _write(self, old: Tree, ch: Change, actor: Actor) -> None:
        now = _now()
        if isinstance(ch, AddTopic):
            t = ch.topic
            # архивная строка с тем же id переиспользуется: id освобождается при архивации
            await self._db.execute(
                "INSERT INTO topics(id, parent_id, seq, title, description, examples, guard, "
                "origin, max_depth, created_at) "
                "VALUES (?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM topics), "
                "?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET parent_id=excluded.parent_id, seq=excluded.seq, "
                "title=excluded.title, description=excluded.description, "
                "examples=excluded.examples, guard=excluded.guard, origin=excluded.origin, "
                "max_depth=excluded.max_depth, created_at=excluded.created_at, archived_at=NULL",
                (
                    t.id,
                    t.parent_id,
                    t.title,
                    t.description,
                    json.dumps(t.examples, ensure_ascii=False),
                    t.guard,
                    t.origin,
                    t.max_depth,
                    now,
                ),
            )
            await self._log(actor, "add", t.id, {"parent": t.parent_id, "origin": t.origin})
        elif isinstance(ch, MoveTopic):
            await self._db.execute(
                "UPDATE topics SET parent_id=? WHERE id=?", (ch.new_parent_id, ch.id)
            )
            await self._log(
                actor, "move", ch.id, {"from": old.get(ch.id).parent_id, "to": ch.new_parent_id}
            )
        elif isinstance(ch, EditTopic):
            for field, value in ch.fields.items():
                stored = json.dumps(value, ensure_ascii=False) if field == "examples" else value
                # имя поля проверено apply_changes (EDITABLE_FIELDS)
                await self._db.execute(f"UPDATE topics SET {field}=? WHERE id=?", (stored, ch.id))
            await self._log(actor, "edit", ch.id, ch.fields)
        elif isinstance(ch, ArchiveTopic):
            gone = [t for t in old.topics if t not in apply_changes(old, [ch]).topics]
            await self._db.executemany(
                "UPDATE topics SET archived_at=? WHERE id=?", [(now, g) for g in gone]
            )
            await self._log(actor, "archive", ch.id, {"subtree": gone})

    async def apply(self, changes: Sequence[Change], actor: Actor) -> Tree:
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            # дерево читается под блокировкой записи: второй писатель ждёт и видит наши правки
            tree = await self.load_tree()
            new = apply_changes(tree, changes)  # TreeValidationError - до записи
            step = tree
            for ch in changes:
                await self._write(step, ch, actor)
                step = apply_changes(step, [ch])
            await self._bump()
            await self._db.execute("COMMIT")
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise
        return Tree(rev=new.rev, settings=new.settings, topics=new.topics)

    async def save_settings(self, s: Settings) -> Tree:
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            await self._db.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (_SETTINGS_KEY, s.model_dump_json()),
            )
            await self._log("cli", "settings", None, s.model_dump())
            await self._bump()
            await self._db.execute("COMMIT")
        except BaseException:
            await self._db.execute("ROLLBACK")
            raise
        return await self.load_tree()

    async def changelog(self, limit: int = 50) -> list[ChangelogEntry]:
        async with self._db.execute(
            "SELECT at, actor, op, topic_id, details FROM changelog ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            ChangelogEntry(
                at=datetime.fromisoformat(r[0]),
                actor=r[1],
                op=r[2],
                topic_id=r[3],
                details=json.loads(r[4]),
            )
            for r in rows
        ]

    async def get_state(self, key: str) -> str | None:
        async with self._db.execute("SELECT value FROM learner_state WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
        return str(row[0]) if row else None

    async def set_state(self, key: str, value: str) -> None:
        await self._db.execute(
            "INSERT INTO learner_state(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
