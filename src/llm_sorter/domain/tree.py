from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_sorter.domain.actions import Action, SaveAction

OTHER_ID = "other"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DiscoverSettings(_Frozen):
    buffer_threshold: int = 50
    min_cluster_size: int = 10
    max_children: int = 12
    dedup_similarity: float = 0.85
    llm_dedup_floor: float = 0.70
    # до ~120k символов образцов (~40-50k токенов) на одно предложение темы:
    # модели guard/text нужен контекст >= 64k
    samples_per_cluster: int = 20
    sample_chars: int = 6000


class Settings(_Frozen):
    max_depth: int = 3
    language: str = "ru"
    discover: DiscoverSettings = Field(default_factory=DiscoverSettings)


class Defaults(_Frozen):
    guard_failed_action: Action = SaveAction(type="save", path="_guard_failed")


class Node(_Frozen):
    uid: str
    id: str
    title: str = ""
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    guard: str | None = None
    actions: list[Action] = Field(default_factory=list)
    origin: Literal["manual", "discovered"] = "manual"
    max_depth: int | None = None
    children: list["Node"] = Field(default_factory=list)


class ArchivedNode(_Frozen):
    uid: str
    path: str
    parent_uid: str | None


class RuleTree(_Frozen):
    version: int | None = None
    settings: Settings = Field(default_factory=Settings)
    defaults: Defaults = Field(default_factory=Defaults)
    root: Node
    archived: list[ArchivedNode] = Field(default_factory=list)

    def iter_nodes(self) -> Iterator[Node]:
        """Обход в глубину, root первым."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def _chain(self, uid: str) -> list[Node]:
        """Цепочка предков от root до узла включительно."""

        def walk(node: Node, acc: list[Node]) -> list[Node] | None:
            acc = [*acc, node]
            if node.uid == uid:
                return acc
            for child in node.children:
                found = walk(child, acc)
                if found is not None:
                    return found
            return None

        chain = walk(self.root, [])
        if chain is None:
            raise KeyError(uid)
        return chain

    def find(self, uid: str) -> Node:
        return self._chain(uid)[-1]

    def parent_of(self, uid: str) -> Node | None:
        chain = self._chain(uid)
        return chain[-2] if len(chain) > 1 else None

    def path_of(self, uid: str) -> str:
        return "/".join(n.id for n in self._chain(uid)[1:])

    def depth_of(self, uid: str) -> int:
        return len(self._chain(uid)) - 1

    def max_depth_for(self, uid: str) -> int:
        for node in reversed(self._chain(uid)):
            if node.max_depth is not None:
                return node.max_depth
        return self.settings.max_depth

    def has_other(self, uid: str) -> bool:
        return uid == self.root.uid or bool(self.find(uid).children)

    def other_path(self, uid: str) -> str:
        path = self.path_of(uid)
        return f"{path}/{OTHER_ID}" if path else OTHER_ID


Node.model_rebuild()
