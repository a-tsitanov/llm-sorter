import pytest

from llm_sorter.domain.traversal import candidates_for, guard_question, resolve_choice
from llm_sorter.domain.tree import ROOT_ID, Tree
from tests.helpers import topic


def test_candidates_include_other_last(tree: Tree) -> None:
    assert [c.id for c in candidates_for(tree, ROOT_ID)] == ["finance", "tech", "other"]
    assert [c.id for c in candidates_for(tree, "finance")] == ["cards", "other"]


def test_empty_root_has_only_other() -> None:
    assert [c.id for c in candidates_for(Tree.empty(), ROOT_ID)] == ["other"]


def test_leaf_has_no_candidates(tree: Tree) -> None:
    assert candidates_for(tree, "cards") == []


def test_resolve_choice(tree: Tree) -> None:
    assert resolve_choice(tree, ROOT_ID, "finance") == "finance"
    assert resolve_choice(tree, ROOT_ID, "other") is None
    with pytest.raises(ValueError, match="unknown choice"):
        resolve_choice(tree, ROOT_ID, "cards")  # не ребёнок root


def test_guard_question_custom_and_default() -> None:
    assert guard_question(topic("a", guard="Это про A?")) == "Это про A?"
    q = guard_question(topic("a", title="Банки", description="Банковские услуги"))
    assert "Банки" in q and "Банковские услуги" in q
