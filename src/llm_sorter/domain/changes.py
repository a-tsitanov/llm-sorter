"""Инкрементальные правки дерева: добавить, перенести, изменить, архивировать тему."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_sorter.domain.tree import ROOT_ID, Topic, Tree
from llm_sorter.domain.validation import TreeValidationError, validate_tree

EDITABLE_FIELDS = {"title", "description", "examples", "guard", "max_depth"}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AddTopic(_Frozen):
    topic: Topic


class MoveTopic(_Frozen):
    id: str
    new_parent_id: str


class EditTopic(_Frozen):
    id: str
    fields: dict[str, Any]


class ArchiveTopic(_Frozen):
    """Убирает тему вместе с потомками."""

    id: str


Change = AddTopic | MoveTopic | EditTopic | ArchiveTopic


def _existing(topics: dict[str, Topic], id: str) -> Topic:
    if id not in topics:
        raise TreeValidationError([f"unknown topic {id!r}"])
    if id == ROOT_ID:
        raise TreeValidationError([f"{ROOT_ID!r} cannot be changed"])
    return topics[id]


def _subtree(topics: dict[str, Topic], id: str) -> set[str]:
    out = {id}
    grew = True
    while grew:
        new = {t.id for t in topics.values() if t.parent_id in out} - out
        out |= new
        grew = bool(new)
    return out


def apply_changes(tree: Tree, changes: Sequence[Change]) -> Tree:
    """Новое дерево (rev + 1) после всех правок; TreeValidationError - ничего не применено."""
    topics = dict(tree.topics)
    for ch in changes:
        if isinstance(ch, AddTopic):
            if ch.topic.id in topics:
                raise TreeValidationError([f"duplicate id {ch.topic.id!r}"])
            topics[ch.topic.id] = ch.topic
        elif isinstance(ch, MoveTopic):
            t = _existing(topics, ch.id)
            topics[ch.id] = t.model_copy(update={"parent_id": ch.new_parent_id})
        elif isinstance(ch, EditTopic):
            t = _existing(topics, ch.id)
            bad = set(ch.fields) - EDITABLE_FIELDS
            if bad:
                raise TreeValidationError([f"field(s) {sorted(bad)} cannot be edited"])
            # через валидацию модели, а не model_copy: типы полей проверяются
            topics[ch.id] = Topic(**{**t.model_dump(), **ch.fields})
        else:
            _existing(topics, ch.id)
            for gone in _subtree(topics, ch.id):
                del topics[gone]
    new = Tree(rev=tree.rev + 1, settings=tree.settings, topics=topics)
    validate_tree(new)
    return new
