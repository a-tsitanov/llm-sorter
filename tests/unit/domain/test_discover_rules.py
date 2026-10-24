import pytest

from llm_sorter.domain.discover import (
    TopicProposal,
    dedup_verdict,
    proposal_is_valid,
    should_discover,
    slots_left,
)
from llm_sorter.domain.tree import ROOT_ID, DiscoverSettings
from tests.helpers import tree3


def test_should_discover_threshold_depth_slots() -> None:
    t = tree3()
    assert should_discover(t, "finance", 50) and not should_discover(t, "finance", 49)
    assert not should_discover(tree3(max_depth=1), "finance", 100)
    full = tree3(max_children=1)
    assert slots_left(full, "finance") == 0
    assert not should_discover(full, "finance", 100)


def test_leaf_can_discover_below_max_depth() -> None:
    t = tree3()
    assert should_discover(t, "cards", 50)  # глубина 2 < 3
    assert not should_discover(tree3(max_depth=2), "cards", 50)
    assert should_discover(t, ROOT_ID, 50)


@pytest.mark.parametrize("sim,verdict", [(0.9, "dup"), (0.85, "dup"), (0.75, "ask"), (0.5, "new")])
def test_dedup_verdict(sim: float, verdict: str) -> None:
    assert dedup_verdict(sim, DiscoverSettings()) == verdict


def _p(**kw: str) -> TopicProposal:
    base = {
        "id": "cards",
        "title": "Карты",
        "description": "Банковские карты",
        "guard": "Пост про банковские карты?",
    }
    return TopicProposal(examples=[], **{**base, **kw})


@pytest.mark.parametrize(
    "kw,ok",
    [
        ({}, True),
        ({"id": "Bad-Id"}, False),
        ({"id": "other"}, False),
        ({"id": "root"}, False),
        ({"title": " "}, False),
        ({"description": ""}, False),
        ({"guard": "yes"}, True),  # guard чинит discover, тема из-за него не теряется
    ],
)
def test_proposal_is_valid(kw: dict[str, str], ok: bool) -> None:
    assert proposal_is_valid(_p(**kw)) is ok
