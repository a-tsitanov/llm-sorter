from typing import Literal

from pydantic import BaseModel, ConfigDict

from llm_sorter.domain.traversal import guard_question
from llm_sorter.domain.tree import DiscoverSettings, Topic, Tree
from llm_sorter.domain.validation import ID_RE, RESERVED_IDS, guard_is_meaningful


class TopicProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    description: str
    examples: list[str]
    guard: str


def slots_left(tree: Tree, node_id: str) -> int:
    return tree.settings.discover.max_children - len(tree.children(node_id))


def should_discover(tree: Tree, node_id: str, new_count: int) -> bool:
    """Узел (в том числе лист) набрал порог, есть запас глубины и свободные слоты."""
    return (
        new_count >= tree.settings.discover.buffer_threshold
        and tree.depth(node_id) < tree.max_depth_for(node_id)
        and slots_left(tree, node_id) > 0
    )


def dedup_verdict(similarity: float, s: DiscoverSettings) -> Literal["dup", "ask", "new"]:
    if similarity >= s.dedup_similarity:
        return "dup"
    if similarity >= s.llm_dedup_floor:
        return "ask"
    return "new"


def proposal_is_valid(p: TopicProposal) -> bool:
    """Предложение пройдёт validate_tree. guard не проверяется: его чинит fix_guard."""
    return (
        bool(ID_RE.match(p.id))
        and p.id not in RESERVED_IDS
        and bool(p.title.strip())
        and bool(p.description.strip())
    )


def fix_guard(p: TopicProposal) -> TopicProposal:
    """Модель пишет ответ («yes») вместо вопроса: подставляем стандартный вопрос из
    названия и описания, чтобы тема не терялась, а в дерево не попал пустой гейт."""
    if guard_is_meaningful(p.guard):
        return p
    question = guard_question(
        Topic(id=p.id, parent_id=None, title=p.title, description=p.description)
    )
    return p.model_copy(update={"guard": question})
