from typing import Protocol

from llm_sorter.domain.messages import Classified, Considered, OtherItem, Resolved, Transcript


class Bus(Protocol):
    """Выходы обработчика и дообучения; каждый вызов ждёт подтверждения брокера."""

    async def classified(self, msg: Classified) -> None:
        """В topic exchange, routing key = msg.topic.path."""
        ...

    async def other(self, record: OtherItem | Resolved | Considered) -> None:
        """В stream other."""
        ...

    async def reinject(self, t: Transcript, reprocess: int) -> None:
        """Обратно в inbox с заголовком x-reprocess."""
        ...
