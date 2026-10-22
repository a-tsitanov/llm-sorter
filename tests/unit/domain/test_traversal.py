import pytest

from llm_sorter.domain.traversal import candidates_for, guard_question, resolve_choice
from llm_sorter.domain.tree import Node
from tests.helpers import uid


def test_candidates_include_other_last(tree):
    ids = [c.id for c in candidates_for(tree, uid(tree, "finance"))]
    assert ids == ["crypto", "banks", "other"]


def test_root_has_other(tree):
    assert [c.id for c in candidates_for(tree, tree.root.uid)] == ["finance", "other"]


def test_leaf_has_no_candidates(tree):
    assert candidates_for(tree, uid(tree, "crypto")) == []


def test_resolve_choice(tree):
    f = uid(tree, "finance")
    assert resolve_choice(tree, f, "crypto") == uid(tree, "crypto")
    assert resolve_choice(tree, f, "other") is None
    with pytest.raises(ValueError):
        resolve_choice(tree, f, "nft")


def test_guard_question_custom_and_default():
    assert guard_question(Node(uid="u", id="a", guard="Q?")) == "Q?"
    n = Node(uid="u", id="a", title="Крипта", description="Монеты")
    assert guard_question(n) == "Этот разговор действительно на тему «Крипта»: Монеты?"
