from typing import Literal

from pydantic import BaseModel, ConfigDict

from llm_sorter.domain.tree import OTHER_ID, DiscoverSettings, RuleTree
from llm_sorter.domain.validation import ID_RE


class TopicProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    description: str
    examples: list[str]
    guard: str


def slots_left(tree: RuleTree, uid: str) -> int:
    return tree.settings.discover.max_children - len(tree.find(uid).children)


def should_discover(tree: RuleTree, uid: str, new_count: int) -> bool:
    return (
        new_count >= tree.settings.discover.buffer_threshold
        and tree.depth_of(uid) < tree.max_depth_for(uid)
        and slots_left(tree, uid) > 0
    )


def discover_blocked(tree: RuleTree, uid: str, other_size: int) -> bool:
    """Узел упёрся в max_children, а буфер other не пуст: discover его уже не разберёт."""
    return slots_left(tree, uid) <= 0 and other_size > 0


def dedup_verdict(similarity: float, s: DiscoverSettings) -> Literal["dup", "ask", "new"]:
    if similarity >= s.dedup_similarity:
        return "dup"
    if similarity >= s.llm_dedup_floor:
        return "ask"
    return "new"


def proposal_is_valid(p: TopicProposal) -> bool:
    """Предложение пройдёт validate_tree: корректный id (не other), непустые title и description."""
    return (
        bool(ID_RE.match(p.id))
        and p.id != OTHER_ID
        and bool(p.title.strip())
        and bool(p.description.strip())
    )
