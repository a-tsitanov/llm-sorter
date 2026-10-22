import pytest

from llm_sorter.domain.classification import Classification, Status
from llm_sorter.domain.discover import dedup_verdict, should_discover, slots_left
from llm_sorter.domain.tree import DiscoverSettings
from tests.helpers import uid, with_settings


def test_should_discover_threshold_depth_slots(tree):
    f = uid(tree, "finance")
    assert should_discover(tree, f, 50) and not should_discover(tree, f, 49)
    deep = with_settings(tree, max_depth=1)
    assert not should_discover(deep, f, 100)
    full = with_settings(tree, max_children=2)
    assert slots_left(full, f) == 0
    assert not should_discover(full, f, 100)


@pytest.mark.parametrize("sim,verdict", [(0.9, "dup"), (0.85, "dup"), (0.75, "ask"), (0.5, "new")])
def test_dedup_verdict(sim, verdict):
    assert dedup_verdict(sim, DiscoverSettings()) == verdict


@pytest.mark.parametrize(
    "status,final",
    [(Status.CLASSIFIED, True), (Status.GUARD_FAILED, True), (Status.IN_OTHER, False)],
)
def test_classification_final(status, final):
    c = Classification(file_id="f", tree_version=1, node_uid="u", path="p", status=status)
    assert c.final is final
