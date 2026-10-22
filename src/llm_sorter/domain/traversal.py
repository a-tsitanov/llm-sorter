from pydantic import BaseModel, ConfigDict, Field

from llm_sorter.domain.tree import OTHER_ID, Node, RuleTree


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    description: str
    examples: list[str] = Field(default_factory=list)


_OTHER = Candidate(id=OTHER_ID, title="Other", description="Ничего из перечисленного не подходит")


def candidates_for(tree: RuleTree, uid: str) -> list[Candidate]:
    """Живые дети по порядку + `other` последним; у листа кандидатов нет."""
    if not tree.has_other(uid):
        return []
    children = [
        Candidate(id=c.id, title=c.title, description=c.description, examples=list(c.examples))
        for c in tree.find(uid).children
    ]
    return [*children, _OTHER]


def resolve_choice(tree: RuleTree, uid: str, choice_id: str) -> str | None:
    """uid выбранного ребёнка; None для `other`; ValueError для неизвестного id."""
    if choice_id == OTHER_ID:
        return None
    for child in tree.find(uid).children:
        if child.id == choice_id:
            return child.uid
    raise ValueError(f"unknown choice {choice_id!r} at {tree.path_of(uid) or 'root'}")


def guard_question(node: Node) -> str:
    return node.guard or (
        f"Этот разговор действительно на тему «{node.title}»: {node.description}?"
    )
