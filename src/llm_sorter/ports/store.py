from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from llm_sorter.domain.changes import Change
from llm_sorter.domain.tree import Settings, Tree

Actor = Literal["learner", "cli"]


@dataclass(frozen=True)
class ChangelogEntry:
    at: datetime
    actor: str
    op: str
    topic_id: str | None
    details: dict[str, Any]


class TopicStore(Protocol):
    """Дерево тем, журнал изменений и служебное состояние дообучения."""

    async def tree_rev(self) -> int: ...

    async def load_tree(self) -> Tree: ...

    async def apply(self, changes: Sequence[Change], actor: Actor) -> Tree:
        """Все правки одной транзакцией: валидация, журнал, tree_rev + 1."""
        ...

    async def save_settings(self, s: Settings) -> Tree: ...

    async def changelog(self, limit: int = 50) -> list[ChangelogEntry]:
        """Новые записи первыми."""
        ...

    async def get_state(self, key: str) -> str | None: ...

    async def set_state(self, key: str, value: str) -> None: ...
