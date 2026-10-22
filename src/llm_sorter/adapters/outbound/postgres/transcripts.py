from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_sorter.adapters.outbound.postgres.tables import transcripts
from llm_sorter.domain.classification import Transcript


class PgTranscripts:
    def __init__(self, sessions: "async_sessionmaker[AsyncSession]") -> None:
        self._sessions = sessions

    async def put(self, t: Transcript) -> bool:
        """True, если транскрипт новый; повторный id не перезаписывается."""
        stmt = (
            insert(transcripts)
            .values(id=t.id, text=t.text, metadata=t.metadata, received_at=t.received_at)
            .on_conflict_do_nothing(index_elements=["id"])
            .returning(transcripts.c.id)
        )
        async with self._sessions.begin() as s:
            return await s.scalar(stmt) is not None

    async def get(self, file_id: str) -> Transcript:
        async with self._sessions() as s:
            row = (
                await s.execute(select(transcripts).where(transcripts.c.id == file_id))
            ).one_or_none()
        if row is None:
            raise LookupError(file_id)
        return Transcript(
            id=row.id, text=row.text, metadata=row.metadata, received_at=row.received_at
        )
