import pytest

from llm_sorter.domain.tree import ROOT_ID, Tree
from tests.helpers import make_tree, topic


def test_navigation(tree: Tree) -> None:
    assert [t.id for t in tree.children(ROOT_ID)] == ["finance", "tech"]
    assert tree.parent("cards").id == "finance"
    assert tree.parent(ROOT_ID) is None
    assert tree.path("cards") == ["finance", "cards"]
    assert tree.path(ROOT_ID) == []
    assert tree.depth("cards") == 2 and tree.depth(ROOT_ID) == 0


def test_routing_key_and_leaf(tree: Tree) -> None:
    assert tree.routing_key("cards") == "finance.cards"
    assert tree.routing_key(ROOT_ID) == ""
    assert tree.is_leaf("cards") and tree.is_leaf("tech")
    assert not tree.is_leaf("finance") and not tree.is_leaf(ROOT_ID)
    assert tree.has_other(ROOT_ID) and tree.has_other("finance")
    assert not tree.has_other("cards")


def test_empty_tree_is_root_only() -> None:
    t = Tree.empty()
    assert list(t.topics) == [ROOT_ID] and t.rev == 0
    assert t.has_other(ROOT_ID) and not t.is_leaf(ROOT_ID)


def test_get_missing_raises(tree: Tree) -> None:
    with pytest.raises(KeyError):
        tree.get("nope")


def test_max_depth_for_nearest_ancestor() -> None:
    t = make_tree(topic("a", max_depth=5), topic("b", "a"), topic("c"), max_depth=2)
    assert t.max_depth_for("b") == 5
    assert t.max_depth_for("c") == 2
    assert t.max_depth_for(ROOT_ID) == 2
