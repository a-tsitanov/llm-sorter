from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_sorter.adapters.outbound.postgres.tables import classifications
from llm_sorter.domain.classification import (
    Classification,
    FinalClassificationExists,
    Status,
)


class PgClassifications:
    def __init__(self, sessions: "async_sessionmaker[AsyncSession]") -> None:
        self._sessions = sessions

    async def get(self, file_id: str) -> Classification | None:
        async with self._sessions() as s:
            row = (
                await s.execute(select(classifications).where(classifications.c.file_id == file_id))
            ).one_or_none()
        if row is None:
            return None
        return Classification(
            file_id=row.file_id,
            tree_version=row.tree_version,
            node_uid=row.node_uid,
            path=row.path,
            status=Status(row.status),
            run_id=row.run_id,
        )

    async def save(self, c: Classification) -> None:
        async with self._sessions.begin() as s:
            status = await s.scalar(
                select(classifications.c.status)
                .where(classifications.c.file_id == c.file_id)
                .with_for_update()
            )
            if status is not None and Status(status) != Status.IN_OTHER:
                raise FinalClassificationExists(c.file_id)
            values = {
                "tree_version": c.tree_version,
                "node_uid": c.node_uid,
                "path": c.path,
                "status": c.status.value,
                "run_id": c.run_id,
                "updated_at": func.now(),
            }
            stmt = insert(classifications).values(file_id=c.file_id, **values)
            # на случай гонки первой вставки: финальную строку всё равно не затираем
            upsert = stmt.on_conflict_do_update(
                index_elements=["file_id"],
                set_=values,
                where=classifications.c.status == Status.IN_OTHER.value,
            ).returning(classifications.c.file_id)
            if await s.scalar(upsert) is None:
                raise FinalClassificationExists(c.file_id)
