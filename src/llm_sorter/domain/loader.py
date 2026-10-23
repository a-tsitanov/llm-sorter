"""YAML дерева тем: настройки и плоский список тем со ссылкой на родителя."""

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_sorter.domain.tree import ROOT_ID, Settings, Topic, Tree


class TreeParseError(ValueError):
    """YAML не читается или не соответствует схеме."""


class _TopicIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    parent: str
    title: str = ""
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    guard: str | None = None
    origin: str = "manual"
    max_depth: int | None = None


class _FileIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: Settings = Field(default_factory=Settings)
    topics: list[_TopicIn] = Field(default_factory=list)


def load_yaml(text: str) -> tuple[Settings, list[Topic]]:
    try:
        raw = yaml.safe_load(text) or {}
        data = _FileIn.model_validate(raw)
        topics = [
            Topic(**t.model_dump(exclude={"parent"}), parent_id=t.parent) for t in data.topics
        ]
    except (yaml.YAMLError, ValidationError) as e:
        raise TreeParseError(str(e)) from e
    return data.settings, topics


def _dump_topic(t: Topic) -> dict[str, Any]:
    out: dict[str, Any] = {"id": t.id, "parent": t.parent_id, "title": t.title}
    out["description"] = t.description
    if t.examples:
        out["examples"] = list(t.examples)
    if t.guard is not None:
        out["guard"] = t.guard
    if t.origin != "manual":
        out["origin"] = t.origin
    if t.max_depth is not None:
        out["max_depth"] = t.max_depth
    return out


def dump_yaml(tree: Tree) -> str:
    """Темы в порядке обхода в глубину: родитель раньше детей; root не пишется."""
    ordered: list[Topic] = []

    def walk(id: str) -> None:
        for c in tree.children(id):
            ordered.append(c)
            walk(c.id)

    walk(ROOT_ID)
    doc = {
        "settings": tree.settings.model_dump(),
        "topics": [_dump_topic(t) for t in ordered],
    }
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
