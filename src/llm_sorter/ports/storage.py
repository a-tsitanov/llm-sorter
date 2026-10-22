import builtins
from datetime import datetime
from typing import Literal, NamedTuple, Protocol

from llm_sorter.domain.classification import Classification, Transcript
from llm_sorter.domain.tree import RuleTree

PublishSource = Literal["manual", "discover", "rollback"]


class VersionConflict(Exception):
    """Активная версия правил не совпадает с base_version."""


class MaterialItem(NamedTuple):
    file_id: str
    kind: Literal["other", "leaf"]


class RulesRepository(Protocol):
    async def active(self) -> RuleTree: ...

    async def get(self, version: int) -> RuleTree: ...

    async def publish(
        self,
        tree: RuleTree,
        base_version: int | None,
        source: PublishSource,
    ) -> int: ...


class TranscriptStore(Protocol):
    async def put(self, t: Transcript) -> bool: ...

    async def get(self, file_id: str) -> Transcript: ...


class ClassificationRepository(Protocol):
    async def get(self, file_id: str) -> Classification | None: ...

    async def save(self, c: Classification) -> None: ...


class MaterialRepository(Protocol):
    """Материал узла для discover: файлы в `<node>/other` (kind=other) и в листе (kind=leaf).

    count_new и due_nodes считают оба вида, добавленные после последнего mark_discovered;
    other_size считает только kind=other.
    mark_discovered(at): материал, добавленный не позже `at` (по умолчанию — сейчас),
    считается увиденным; добавленный после `at` остаётся новым.
    """

    async def add(self, node_uid: str, file_id: str, kind: Literal["other", "leaf"]) -> None: ...

    async def list(self, node_uid: str) -> builtins.list[MaterialItem]: ...

    async def remove(self, node_uid: str, file_id: str) -> None: ...

    async def move_other(self, from_uid: str, to_uid: str) -> None: ...

    async def count_new(self, node_uid: str) -> int: ...

    async def mark_discovered(self, node_uid: str, at: datetime | None = None) -> None: ...

    async def due_nodes(self, threshold: int) -> builtins.list[str]: ...

    async def other_size(self, node_uid: str) -> int: ...


class EmbeddingCache(Protocol):
    async def get_many(self, file_ids: list[str]) -> dict[str, list[float]]: ...

    async def put_many(self, vectors: dict[str, list[float]]) -> None: ...


class FileStorage(Protocol):
    async def write(self, rel_path: str, content: str) -> str: ...


class Notifier(Protocol):
    async def send(
        self, channel: str, text: str, attachment_name: str, attachment: str
    ) -> None: ...
