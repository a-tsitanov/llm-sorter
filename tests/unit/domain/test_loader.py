import pytest

from llm_sorter.domain.loader import TreeParseError, dump_yaml, load_yaml
from llm_sorter.domain.tree import Tree
from tests.helpers import make_tree, topic

YAML = """
settings:
  max_depth: 4
  discover: {buffer_threshold: 7}
topics:
  - {id: finance, parent: root, title: Финансы, description: Деньги}
  - id: cards
    parent: finance
    title: Карты
    description: Банковские карты
    examples: [Кэшбэк по карте]
    guard: Пост про банковские карты?
"""


def test_load_yaml() -> None:
    settings, topics = load_yaml(YAML)
    assert settings.max_depth == 4 and settings.discover.buffer_threshold == 7
    assert [(t.id, t.parent_id) for t in topics] == [("finance", "root"), ("cards", "finance")]
    assert topics[1].examples == ["Кэшбэк по карте"]
    assert topics[1].guard == "Пост про банковские карты?"


def test_settings_optional() -> None:
    settings, topics = load_yaml("topics: [{id: a, parent: root, title: A, description: d}]")
    assert settings.max_depth == 3 and len(topics) == 1


def test_dump_roundtrip() -> None:
    t = make_tree(topic("finance"), topic("cards", "finance", examples=["x"]), topic("tech"))
    settings, topics = load_yaml(dump_yaml(t))
    again = Tree(
        rev=0, settings=settings, topics={"root": t.get("root"), **{x.id: x for x in topics}}
    )
    assert again.topics == t.topics and again.settings == t.settings


def test_dump_parents_before_children() -> None:
    t = make_tree(topic("finance"), topic("cards", "finance"))
    assert dump_yaml(t).index("id: finance") < dump_yaml(t).index("id: cards")
    assert "id: root" not in dump_yaml(t)


@pytest.mark.parametrize(
    "bad",
    [
        "{not yaml",
        "topics: 5",
        "topics: [{id: a}]",  # нет parent
        "topics: [{id: a, parent: root, colour: red}]",  # лишнее поле
        "settings: {max_depth: many}",
    ],
)
def test_parse_errors(bad: str) -> None:
    with pytest.raises(TreeParseError):
        load_yaml(bad)
