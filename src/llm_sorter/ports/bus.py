from typing import Protocol

from llm_sorter.domain.messages import AnyRecord, Classified, Transcript


class Bus(Protocol):
    """Выходы обработчика и дообучения; каждый вызов ждёт подтверждения брокера."""

    async def classified(self, msg: Classified) -> None:
        """В topic exchange, routing key = msg.topic.path."""
        ...

    async def other(self, record: AnyRecord) -> None:
        """В stream other."""
        ...

    async def reinject(self, t: Transcript, reprocess: int, tree_rev: int = 0) -> None:
        """Обратно в inbox с заголовками x-reprocess и x-tree-rev."""
        ...
