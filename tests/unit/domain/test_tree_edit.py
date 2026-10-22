from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.tree_edit import (
    add_children,
    gained_children,
    live_ancestor_with_other,
    merge_archived,
)
from tests.helpers import remove_node, uid


def _nft() -> TopicProposal:
    return TopicProposal(id="nft", title="NFT", description="d", examples=[], guard="g")


def test_add_children_marks_discovered(tree):
    t2 = add_children(tree, uid(tree, "crypto"), [_nft()])
    nft = next(n for n in t2.iter_nodes() if n.id == "nft")
    assert nft.origin == "discovered" and t2.path_of(nft.uid) == "finance/crypto/nft"
    assert nft.actions == [] and t2.version is None
    assert gained_children(tree, t2) == [uid(tree, "crypto")]
    assert not any(n.id == "nft" for n in tree.iter_nodes())  # исходное не изменено


def test_gained_children_none_when_equal(tree):
    assert gained_children(tree, tree) == []


def test_merge_archived_and_live_ancestor(tree):
    new = remove_node(tree, "crypto")
    merged = merge_archived(tree, new)
    assert [a.path for a in merged.archived] == ["finance/crypto"]
    assert merged.archived[0].parent_uid == uid(tree, "finance")
    assert live_ancestor_with_other(tree, merged, uid(tree, "crypto")) == uid(tree, "finance")


def test_merge_archived_is_idempotent(tree):
    merged = merge_archived(tree, remove_node(tree, "crypto"))
    assert len(merge_archived(tree, merged).archived) == 1


def test_live_ancestor_skips_ancestor_without_other(tree):
    # finance потерял всех детей -> other у него нет, берём root
    new = remove_node(remove_node(tree, "crypto"), "banks")
    assert live_ancestor_with_other(tree, new, uid(tree, "crypto")) == tree.root.uid
