import pytest

from llm_sorter.domain.changes import AddTopic, ArchiveTopic, EditTopic, MoveTopic, apply_changes
from llm_sorter.domain.tree import Topic, Tree
from llm_sorter.domain.validation import TreeValidationError
from tests.helpers import topic


def test_apply_changes_bumps_rev_and_keeps_order(tree: Tree) -> None:
    t = apply_changes(
        tree,
        [AddTopic(topic=Topic(id="loans", parent_id="finance", title="Кредиты", description="d"))],
    )
    assert t.rev == tree.rev + 1
    assert [c.id for c in t.children("finance")] == ["cards", "loans"]
    assert "loans" not in tree.topics  # исходное дерево не меняется


def test_duplicate_id_anywhere_rejected(tree: Tree) -> None:
    with pytest.raises(TreeValidationError, match="duplicate id"):
        apply_changes(tree, [AddTopic(topic=topic("cards", "tech"))])


def test_move_into_own_subtree_rejected(tree: Tree) -> None:
    with pytest.raises(TreeValidationError, match="cycle"):
        apply_changes(tree, [MoveTopic(id="finance", new_parent_id="cards")])


def test_move_changes_routing_key(tree: Tree) -> None:
    t = apply_changes(tree, [MoveTopic(id="cards", new_parent_id="tech")])
    assert t.routing_key("cards") == "tech.cards"
    assert t.is_leaf("finance")


def test_unknown_parent_rejected(tree: Tree) -> None:
    with pytest.raises(TreeValidationError, match="unknown parent"):
        apply_changes(tree, [AddTopic(topic=topic("x", "nope"))])


def test_archive_removes_subtree(tree: Tree) -> None:
    t = apply_changes(tree, [ArchiveTopic(id="finance")])
    assert "finance" not in t.topics and "cards" not in t.topics
    assert "tech" in t.topics


def test_root_cannot_be_archived_or_moved(tree: Tree) -> None:
    with pytest.raises(TreeValidationError):
        apply_changes(tree, [ArchiveTopic(id="root")])
    with pytest.raises(TreeValidationError):
        apply_changes(tree, [MoveTopic(id="root", new_parent_id="tech")])


def test_edit_updates_fields(tree: Tree) -> None:
    t = apply_changes(tree, [EditTopic(id="tech", fields={"title": "Технологии"})])
    assert t.get("tech").title == "Технологии"


def test_edit_rejects_unknown_field(tree: Tree) -> None:
    with pytest.raises(TreeValidationError, match="field"):
        apply_changes(tree, [EditTopic(id="tech", fields={"parent_id": "finance"})])


def test_unknown_topic_rejected(tree: Tree) -> None:
    with pytest.raises(TreeValidationError, match="unknown topic"):
        apply_changes(tree, [ArchiveTopic(id="nope")])
