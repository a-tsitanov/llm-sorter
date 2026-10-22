import builtins
from datetime import datetime
from typing import Literal

from sqlalchemy import ColumnElement, delete, func, literal, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_sorter.adapters.outbound.postgres.tables import discover_state, material
from llm_sorter.ports.storage import MaterialItem

_NEVER = text("'-infinity'::timestamptz")


class PgMaterial:
    """Семантика как у FakeMaterial: count_new/due_nodes считают оба kind после
    последнего mark_discovered, other_size — только kind='other'."""

    def __init__(self, sessions: "async_sessionmaker[AsyncSession]") -> None:
        self._sessions = sessions

    async def add(self, node_uid: str, file_id: str, kind: Literal["other", "leaf"]) -> None:
        # повтор идемпотентен: added_at сохраняется, обновляется только kind (как в fake)
        stmt = insert(material).values(
            node_uid=node_uid, file_id=file_id, kind=kind, added_at=func.clock_timestamp()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["node_uid", "file_id"], set_={"kind": stmt.excluded.kind}
        )
        async with self._sessions.begin() as s:
            await s.execute(stmt)

    async def list(self, node_uid: str) -> builtins.list[MaterialItem]:
        async with self._sessions() as s:
            rows = await s.execute(
                select(material.c.file_id, material.c.kind)
                .where(material.c.node_uid == node_uid)
                .order_by(material.c.added_at, material.c.file_id)
            )
            return [MaterialItem(r.file_id, r.kind) for r in rows]

    async def remove(self, node_uid: str, file_id: str) -> None:
        async with self._sessions.begin() as s:
            await s.execute(
                delete(material).where(
                    material.c.node_uid == node_uid, material.c.file_id == file_id
                )
            )

    async def move_other(self, from_uid: str, to_uid: str) -> None:
        if from_uid == to_uid:
            return
        # одним оператором: DELETE ... RETURNING + INSERT, чтобы параллельный add не потерялся
        moved = (
            delete(material)
            .where(material.c.node_uid == from_uid, material.c.kind == "other")
            .returning(material.c.file_id)
            .cte("moved")
        )
        src = select(literal(to_uid), moved.c.file_id, literal("other"), func.clock_timestamp())
        ins = (
            insert(material)
            .from_select(["node_uid", "file_id", "kind", "added_at"], src)
            .on_conflict_do_nothing(index_elements=["node_uid", "file_id"])
        )
        async with self._sessions.begin() as s:
            await s.execute(ins)

    def _new_clause(self) -> ColumnElement[bool]:
        return material.c.added_at > func.coalesce(discover_state.c.last_discover_at, _NEVER)

    async def count_new(self, node_uid: str) -> int:
        stmt = (
            select(func.count())
            .select_from(
                material.outerjoin(discover_state, discover_state.c.node_uid == material.c.node_uid)
            )
            .where(material.c.node_uid == node_uid, self._new_clause())
        )
        async with self._sessions() as s:
            return int(await s.scalar(stmt) or 0)

    async def mark_discovered(self, node_uid: str, at: datetime | None = None) -> None:
        stmt = insert(discover_state).values(
            node_uid=node_uid, last_discover_at=at if at is not None else func.clock_timestamp()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["node_uid"], set_={"last_discover_at": stmt.excluded.last_discover_at}
        )
        async with self._sessions.begin() as s:
            await s.execute(stmt)

    async def due_nodes(self, threshold: int) -> builtins.list[str]:
        stmt = (
            select(material.c.node_uid)
            .select_from(
                material.outerjoin(discover_state, discover_state.c.node_uid == material.c.node_uid)
            )
            .where(self._new_clause())
            .group_by(material.c.node_uid)
            .having(func.count() >= threshold)
            .order_by(func.min(material.c.added_at), material.c.node_uid)
        )
        async with self._sessions() as s:
            return [r for r in (await s.scalars(stmt))]

    async def other_size(self, node_uid: str) -> int:
        stmt = (
            select(func.count())
            .select_from(material)
            .where(material.c.node_uid == node_uid, material.c.kind == "other")
        )
        async with self._sessions() as s:
            return int(await s.scalar(stmt) or 0)
