"""Дерево тем: плоский словарь тем со ссылкой на родителя, корень - тема `root`."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OTHER_ID = "other"
ROOT_ID = "root"


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
    # после стольких перегонов сообщение, снова попавшее в other, остаётся там навсегда
    max_reprocess: int = 3
    discover: DiscoverSettings = Field(default_factory=DiscoverSettings)


class Topic(_Frozen):
    id: str
    parent_id: str | None
    title: str = ""
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    guard: str | None = None
    origin: Literal["manual", "discovered"] = "manual"
    max_depth: int | None = None


@dataclass(frozen=True)
class Tree:
    """Снимок дерева в памяти. Порядок `topics` (порядок вставки) - порядок детей."""

    rev: int
    settings: Settings
    topics: dict[str, Topic]
    _children: dict[str, list[str]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        children: dict[str, list[str]] = {}
        for t in self.topics.values():
            if t.parent_id is not None:
                children.setdefault(t.parent_id, []).append(t.id)
        object.__setattr__(self, "_children", children)

    @staticmethod
    def empty(settings: Settings | None = None) -> "Tree":
        return Tree(
            rev=0,
            settings=settings or Settings(),
            topics={ROOT_ID: Topic(id=ROOT_ID, parent_id=None)},
        )

    def get(self, id: str) -> Topic:
        return self.topics[id]

    def children(self, id: str) -> list[Topic]:
        return [self.topics[c] for c in self._children.get(id, [])]

    def parent(self, id: str) -> Topic | None:
        p = self.get(id).parent_id
        return None if p is None else self.topics[p]

    def _chain(self, id: str) -> list[Topic]:
        """Цепочка от root до темы включительно."""
        chain = [self.get(id)]
        while chain[-1].parent_id is not None:
            chain.append(self.topics[chain[-1].parent_id])
        return chain[::-1]

    def path(self, id: str) -> list[str]:
        return [t.id for t in self._chain(id)[1:]]

    def routing_key(self, id: str) -> str:
        return ".".join(self.path(id))

    def depth(self, id: str) -> int:
        return len(self._chain(id)) - 1

    def max_depth_for(self, id: str) -> int:
        for t in reversed(self._chain(id)):
            if t.max_depth is not None:
                return t.max_depth
        return self.settings.max_depth

    def has_other(self, id: str) -> bool:
        return id == ROOT_ID or bool(self._children.get(id))

    def is_leaf(self, id: str) -> bool:
        return id != ROOT_ID and not self._children.get(id)
