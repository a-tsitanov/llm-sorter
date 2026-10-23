from typing import Protocol

from llm_sorter.domain.classification import Transcript


class TranscriptSource(Protocol):
    """fetch_new отдаёт ещё не подтверждённые транскрипты (повторно, пока нет ack);
    ack(ids) вызывается после успешной публикации и исключает их из следующих выборок."""

    async def fetch_new(self) -> list[Transcript]: ...

    async def ack(self, ids: list[str]) -> None: ...


class MessagePublisher(Protocol):
    async def publish(self, t: Transcript) -> None: ...
