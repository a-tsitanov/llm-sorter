from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_sorter.adapters.outbound.postgres.tables import embeddings


class PgEmbeddings:
    def __init__(self, sessions: "async_sessionmaker[AsyncSession]") -> None:
        self._sessions = sessions

    async def get_many(self, file_ids: list[str]) -> dict[str, list[float]]:
        if not file_ids:
            return {}
        async with self._sessions() as s:
            rows = await s.execute(
                select(embeddings.c.file_id, embeddings.c.vector).where(
                    embeddings.c.file_id.in_(file_ids)
                )
            )
            return {r.file_id: [float(x) for x in r.vector] for r in rows}

    async def put_many(self, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        stmt = insert(embeddings)
        stmt = stmt.on_conflict_do_update(
            index_elements=["file_id"], set_={"vector": stmt.excluded.vector}
        )
        async with self._sessions.begin() as s:
            await s.execute(stmt, [{"file_id": f, "vector": v} for f, v in vectors.items()])
