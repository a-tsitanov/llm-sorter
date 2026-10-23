from pydantic import BaseModel, ConfigDict, Field

from llm_sorter.domain.tree import OTHER_ID, Topic, Tree


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    description: str
    examples: list[str] = Field(default_factory=list)


_OTHER = Candidate(id=OTHER_ID, title="Other", description="Ничего из перечисленного не подходит")


def candidates_for(tree: Tree, node_id: str) -> list[Candidate]:
    """Дети по порядку + `other` последним; у листа кандидатов нет."""
    if not tree.has_other(node_id):
        return []
    children = [
        Candidate(id=c.id, title=c.title, description=c.description, examples=list(c.examples))
        for c in tree.children(node_id)
    ]
    return [*children, _OTHER]


def resolve_choice(tree: Tree, node_id: str, choice_id: str) -> str | None:
    """id выбранного ребёнка; None для `other`; ValueError для неизвестного id."""
    if choice_id == OTHER_ID:
        return None
    if any(c.id == choice_id for c in tree.children(node_id)):
        return choice_id
    raise ValueError(f"unknown choice {choice_id!r} at {tree.routing_key(node_id) or 'root'}")


def guard_question(t: Topic) -> str:
    return t.guard or f"Этот разговор действительно на тему «{t.title}»: {t.description}?"
