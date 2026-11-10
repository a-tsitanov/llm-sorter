from datetime import UTC, datetime

import pytest

from llm_sorter.application.process import ProcessService
from llm_sorter.domain.messages import Transcript
from llm_sorter.domain.tree import Tree
from llm_sorter.ports.models import InvalidModelOutput
from tests.fakes import FakeDecision, FakeGuard, RecordingMetrics
from tests.helpers import make_tree, topic, tree3

NOW = datetime(2026, 10, 23, tzinfo=UTC)


def t(id_: str = "m1", text: str = "текст про карту") -> Transcript:
    return Transcript(id=id_, text=text, metadata={"src": "x"}, received_at=NOW)


def svc(
    dec: FakeDecision | None = None,
    guard: FakeGuard | None = None,
    metrics: RecordingMetrics | None = None,
) -> ProcessService:
    return ProcessService(
        dec or FakeDecision([]),
        guard or FakeGuard([]),
        metrics or RecordingMetrics(),
        now=lambda: NOW,
    )


async def test_empty_tree_goes_to_root_other_without_llm() -> None:
    dec = FakeDecision([])
    out = await svc(dec).process(t(), Tree.empty())
    assert out.classified is None
    assert [(o.node, o.reason, o.trail) for o in out.others] == [("root", "no_match", [])]
    assert dec.calls == []


async def test_empty_text_goes_to_other_without_llm() -> None:
    dec = FakeDecision([])
    out = await svc(dec).process(t(text="   "), tree3())
    assert [(o.node, o.reason) for o in out.others] == [("root", "no_match")]
    assert dec.calls == []


async def test_descends_to_leaf_and_passes_gate() -> None:
    tree = tree3()
    guard = FakeGuard([True])
    out = await svc(FakeDecision(["finance", "cards"]), guard).process(t(), tree)
    c = out.classified
    assert c is not None
    assert (c.topic.id, c.topic.path, c.topic.title) == ("cards", "finance.cards", "Cards")
    assert c.trail == ["finance", "cards"] and c.tree_rev == tree.rev and c.classified_at == NOW
    assert (c.id, c.text, c.metadata) == ("m1", "текст про карту", {"src": "x"})
    assert guard.calls == [("текст про карту", "Пост про банковские карты?")]
    # глубина 2 < max_depth 3: копия - материал для прорастания листа
    assert [(o.node, o.reason, o.final) for o in out.others] == [("cards", "leaf", False)]


async def test_guard_can_be_disabled_without_claiming_it_passed() -> None:
    guard = FakeGuard([])
    metrics = RecordingMetrics()
    service = ProcessService(FakeDecision(["tech"]), guard, metrics, guard_enabled=False)
    out = await service.process(t(), tree3())
    assert out.classified is not None and out.classified.gate == "skipped"
    assert guard.calls == []


async def test_no_leaf_copy_at_max_depth() -> None:
    out = await svc(FakeDecision(["finance", "cards"]), FakeGuard([True])).process(
        t(), tree3(max_depth=2)
    )
    assert out.classified is not None and out.others == []


async def test_decide_other_stops_at_node() -> None:
    out = await svc(FakeDecision(["finance", "other"])).process(t(), tree3())
    assert out.classified is None
    assert [(o.node, o.reason, o.trail) for o in out.others] == [
        ("finance", "no_match", ["finance"])
    ]


async def test_gate_failure_goes_to_parent_other() -> None:
    out = await svc(FakeDecision(["finance", "cards"]), FakeGuard([False])).process(t(), tree3())
    assert out.classified is None
    assert [(o.node, o.reason, o.trail) for o in out.others] == [
        ("finance", "gate_failed", ["finance", "cards"])
    ]


async def test_invalid_choice_three_times_falls_back() -> None:
    m = RecordingMetrics()
    dec = FakeDecision(["bogus", "bogus", "bogus"])
    out = await svc(dec, metrics=m).process(t(), tree3())
    assert [(o.node, o.reason) for o in out.others] == [("root", "model_error")]
    assert len(dec.calls) == 3 and m.counts["model_error:decide"] == 1


async def test_invalid_output_then_valid_recovers() -> None:
    dec = FakeDecision([InvalidModelOutput("x"), "tech"])
    out = await svc(dec, FakeGuard([True])).process(t(), tree3())
    assert out.classified is not None and out.classified.topic.path == "tech"


async def test_guard_invalid_output_goes_to_parent_as_model_error() -> None:
    bad = InvalidModelOutput("x")
    out = await svc(FakeDecision(["tech"]), FakeGuard([bad, bad, bad])).process(t(), tree3())
    assert [(o.node, o.reason) for o in out.others] == [("root", "model_error")]


@pytest.mark.parametrize("reprocess,final", [(0, False), (2, False), (3, True), (5, True)])
async def test_final_after_max_reprocess(reprocess: int, final: bool) -> None:
    out = await svc(FakeDecision(["other"])).process(t(), tree3(), reprocess=reprocess)
    assert out.others[0].final is final and out.others[0].reprocess == reprocess


async def test_leaf_copy_never_final() -> None:
    out = await svc(FakeDecision(["finance", "cards"]), FakeGuard([True])).process(
        t(), tree3(), reprocess=9
    )
    assert out.others[0].reason == "leaf" and out.others[0].final is False


async def test_connection_error_propagates() -> None:
    with pytest.raises(ConnectionError):
        await svc(FakeDecision([ConnectionError()])).process(t(), tree3())


async def test_metrics_outcomes() -> None:
    m = RecordingMetrics()
    s = svc(FakeDecision(["tech", "other"]), FakeGuard([True]), m)
    await s.process(t("a"), tree3())
    await s.process(t("b"), tree3())
    await s.process(t("c"), tree3(), reprocess=3)
    assert m.processed_by == {"classified": 1, "other": 1, "final_other": 1}
    assert m.counts["guard:tech:True"] == 1


async def test_deep_tree_records_trail() -> None:
    tree = make_tree(topic("a"), topic("b", "a"), topic("c", "b"), max_depth=3)
    out = await svc(FakeDecision(["a", "b", "c"]), FakeGuard([True])).process(t(), tree)
    assert out.classified is not None and out.classified.trail == ["a", "b", "c"]
    assert out.others == []
