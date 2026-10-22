from datetime import UTC, datetime

import pytest

from llm_sorter.application.classify import ClassifyService, StepResult
from llm_sorter.domain.classification import FinalClassificationExists, Status, Transcript
from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.loader import load_tree
from llm_sorter.domain.tree_edit import add_children
from llm_sorter.ports.models import InvalidModelOutput
from llm_sorter.ports.storage import MaterialItem
from tests.fakes import (
    FakeClassifications,
    FakeMaterial,
    FakeRules,
    FakeStarter,
    FakeTranscripts,
    RecordingMetrics,
    ScriptedDecision,
    ScriptedGuard,
)
from tests.helpers import uid

TREE_YAML = """
settings:
  discover:
    buffer_threshold: 2
node:
  id: root
  children:
    - id: finance
      title: Финансы
      description: Деньги
      children:
        - id: crypto
          title: Крипта
          description: Криптовалюты
          children:
            - id: trading
              title: Трейдинг
              description: Торговля
        - id: banks
          title: Банки
          description: Банки
"""

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class Env:
    def __init__(self, answers: dict | None = None) -> None:
        self.tree = load_tree(TREE_YAML)
        self.rules = FakeRules(self.tree)
        self.transcripts = FakeTranscripts()
        self.classifications = FakeClassifications()
        self.material = FakeMaterial()
        self.decision = ScriptedDecision(answers)
        self.guard = ScriptedGuard()
        self.starter = FakeStarter()
        self.metrics = RecordingMetrics()
        self.svc = ClassifyService(
            self.rules,
            self.transcripts,
            self.classifications,
            self.material,
            self.decision,
            self.guard,
            self.starter,
            self.metrics,
        )
        self.root = self.rules.versions[1].root.uid

    def uid(self, id_: str) -> str:
        return uid(self.rules.versions[1], id_)


async def make(answers: dict | None = None) -> Env:
    env = Env(answers)
    await env.transcripts.put(Transcript(id="f1", text="про биткоин", received_at=NOW))
    return env


async def test_step_descends_then_leaf() -> None:
    e = await make({("finance", "other"): "finance"})
    assert await e.svc.step(1, e.root, "f1") == StepResult(
        kind="descend", next_uid=e.uid("finance")
    )
    assert (await e.svc.step(1, e.uid("trading"), "f1")).kind == "leaf"
    assert e.metrics.counts["model_call"] == 1


async def test_step_other() -> None:
    e = await make({("finance", "other"): "other"})
    assert await e.svc.step(1, e.root, "f1") == StepResult(kind="other", next_uid=None)


async def test_unknown_choice_raises() -> None:
    e = await make({("finance", "other"): "nft"})
    with pytest.raises(InvalidModelOutput):
        await e.svc.step(1, e.root, "f1")


async def test_empty_text_goes_other_without_model() -> None:
    e = await make()
    await e.transcripts.put(Transcript(id="e", text="  \n", received_at=NOW))
    assert (await e.svc.step(1, e.root, "e")).kind == "other"
    assert e.decision.calls == 0


async def test_guard_check_records_metric() -> None:
    e = await make()
    assert await e.svc.guard_check(1, e.uid("finance"), "f1") is True
    assert e.metrics.counts["guard_result"] == 1
    assert e.metrics.counts["model_call"] == 1


async def test_active_version_and_root() -> None:
    e = await make()
    v = await e.svc.active_version()
    assert v == 1
    assert await e.svc.root_uid(v) == e.root


async def test_finalize_other_adds_material() -> None:
    e = await make()
    r = await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER)
    assert r.path == "finance/other"
    assert (await e.classifications.get("f1")).path == "finance/other"
    assert await e.material.list(e.uid("finance")) == [MaterialItem(file_id="f1", kind="other")]
    assert e.metrics.buffer_sizes == [("finance/other", 1)]


async def test_finalize_leaf_below_max_depth_is_material() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("crypto"), Status.CLASSIFIED)
    assert MaterialItem(file_id="f1", kind="leaf") in await e.material.list(e.uid("crypto"))


async def test_finalize_at_max_depth_no_material() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("trading"), Status.CLASSIFIED)
    assert await e.material.list(e.uid("trading")) == []


async def test_guard_failed_path() -> None:
    e = await make()
    r = await e.svc.finalize("f1", 1, e.uid("banks"), Status.GUARD_FAILED)
    assert r.path == "_guard_failed/finance/banks"
    assert await e.material.list(e.uid("banks")) == []


async def test_final_is_immutable() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("trading"), Status.CLASSIFIED)
    r = await e.svc.finalize("f1", 1, e.uid("banks"), Status.CLASSIFIED)
    assert r.skipped
    assert (await e.classifications.get("f1")).path == "finance/crypto/trading"
    assert e.metrics.counts["topic_move_attempt"] == 1
    assert e.metrics.counts["file_classified"] == 1
    assert await e.material.list(e.uid("banks")) == []


async def test_reprocess_moves_out_of_other() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER)
    await e.svc.finalize(
        "f1", 1, e.uid("banks"), Status.CLASSIFIED, from_other_uid=e.uid("finance")
    )
    assert await e.material.list(e.uid("finance")) == []
    assert e.metrics.counts["reprocess_moved"] == 1
    assert e.metrics.buffer_sizes == [("finance/other", 1), ("finance/other", 0)]


async def test_reprocess_staying_in_same_other_does_not_duplicate() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER)
    await e.svc.finalize(
        "f1", 1, e.uid("finance"), Status.IN_OTHER, from_other_uid=e.uid("finance")
    )
    assert await e.material.list(e.uid("finance")) == [MaterialItem("f1", "other")]


async def test_threshold_triggers_discover() -> None:
    e = await make()
    for f in ("a", "b"):
        await e.svc.finalize(f, 1, e.uid("finance"), Status.IN_OTHER)
    assert e.starter.discover == [e.uid("finance")]


async def test_below_threshold_no_discover() -> None:
    e = await make()
    await e.svc.finalize("a", 1, e.uid("finance"), Status.IN_OTHER)
    assert e.starter.discover == []


async def test_reprocess_lands_in_deeper_other() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER)
    await e.svc.finalize("f1", 1, e.uid("crypto"), Status.IN_OTHER, from_other_uid=e.uid("finance"))
    assert await e.material.list(e.uid("finance")) == []
    assert await e.material.list(e.uid("crypto")) == [MaterialItem("f1", "other")]
    assert e.metrics.counts["reprocess_moved"] == 1


async def test_reprocess_same_other_no_moved_metric() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER)
    await e.svc.finalize(
        "f1", 1, e.uid("finance"), Status.IN_OTHER, from_other_uid=e.uid("finance")
    )
    assert e.metrics.counts["reprocess_moved"] == 0


async def test_guard_failed_never_starts_discover() -> None:
    e = await make()
    for f in ("a", "b", "c"):
        await e.svc.finalize(f, 1, e.uid("finance"), Status.GUARD_FAILED)
    assert e.starter.discover == []


class _FlakyMaterial(FakeMaterial):
    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    async def add(self, node_uid, file_id, kind) -> None:  # type: ignore[no-untyped-def]
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("boom")
        await super().add(node_uid, file_id, kind)


async def test_finalize_retry_with_same_run_id_reapplies_effects() -> None:
    e = await make()
    e.material = _FlakyMaterial()
    e.svc._material = e.material
    with pytest.raises(RuntimeError):
        await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER, run_id="r1")
    r = await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER, run_id="r1")
    assert not r.skipped
    assert await e.material.list(e.uid("finance")) == [MaterialItem("f1", "other")]
    assert e.metrics.counts["topic_move_attempt"] == 0


async def test_finalize_different_run_id_is_skipped() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER, run_id="r1")
    await e.svc.finalize("f1", 1, e.uid("trading"), Status.CLASSIFIED, run_id="r1")
    r = await e.svc.finalize("f1", 1, e.uid("trading"), Status.CLASSIFIED, run_id="r2")
    assert r.skipped
    assert e.metrics.counts["topic_move_attempt"] == 1


async def test_invalid_choice_retried_three_times_then_raises() -> None:
    e = await make({("finance", "other"): "nft"})
    with pytest.raises(InvalidModelOutput):
        await e.svc.step(1, e.root, "f1")
    assert e.decision.calls == 3
    assert e.metrics.counts["model_call"] == 3


async def test_invalid_choice_then_valid_succeeds() -> None:
    e = await make({("finance", "other"): ["nft", "finance"]})
    assert (await e.svc.step(1, e.root, "f1")).kind == "descend"
    assert e.decision.calls == 2


class _DownGuard(ScriptedGuard):
    async def check(self, text: str, question: str) -> bool:
        self.calls += 1
        raise ConnectionError("down")


async def test_transport_error_is_not_retried_in_service() -> None:
    # недоступность модели отдаётся Temporal сразу, без внутреннего цикла
    e = await make()
    e.guard = _DownGuard()
    e.svc._guard = e.guard
    with pytest.raises(ConnectionError):
        await e.svc.guard_check(1, e.uid("finance"), "f1")
    assert e.guard.calls == 1


class _SpyClassifications(FakeClassifications):
    def __init__(self) -> None:
        super().__init__()
        self.final_rejections = 0

    async def save(self, c) -> None:  # type: ignore[no-untyped-def]
        try:
            await super().save(c)
        except FinalClassificationExists:
            self.final_rejections += 1
            raise


async def test_final_status_retry_with_same_run_id_reapplies_effects() -> None:
    e = await make()
    e.material = _FlakyMaterial()
    e.svc._material = e.material
    e.classifications = _SpyClassifications()
    e.svc._classifications = e.classifications
    with pytest.raises(RuntimeError):
        await e.svc.finalize("f1", 1, e.uid("crypto"), Status.CLASSIFIED, run_id="r1")
    r = await e.svc.finalize("f1", 1, e.uid("crypto"), Status.CLASSIFIED, run_id="r1")
    assert e.classifications.final_rejections == 1  # ветка повтора реально пройдена
    assert not r.skipped
    assert await e.material.list(e.uid("crypto")) == [MaterialItem("f1", "leaf")]
    assert e.metrics.counts["topic_move_attempt"] == 0


async def test_max_depth_uses_largest_override() -> None:
    e = await make()
    assert await e.svc.max_depth(1) == 3
    tree = e.rules.versions[1]
    deep = tree.model_copy(update={"root": tree.root.model_copy(update={"max_depth": 5})})
    e.rules.versions[2] = deep.model_copy(update={"version": 2})
    assert await e.svc.max_depth(2) == 5


async def test_same_run_id_with_different_decision_replays_stored() -> None:
    e = await make()
    first = await e.svc.finalize("f1", 1, e.uid("trading"), Status.CLASSIFIED, run_id="r1")
    assert (first.node_uid, first.status, first.version) == (
        e.uid("trading"),
        Status.CLASSIFIED,
        1,
    )
    r = await e.svc.finalize("f1", 1, e.uid("banks"), Status.GUARD_FAILED, run_id="r1")
    assert not r.skipped
    assert (r.node_uid, r.status, r.path, r.version) == (
        e.uid("trading"),
        Status.CLASSIFIED,
        "finance/crypto/trading",
        1,
    )
    assert e.metrics.counts["topic_move_attempt"] == 0
    assert (await e.classifications.get("f1")).node_uid == e.uid("trading")
    assert await e.material.list(e.uid("banks")) == []


async def test_skipped_reports_stored_state() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("trading"), Status.CLASSIFIED, run_id="r1")
    r = await e.svc.finalize("f1", 1, e.uid("banks"), Status.GUARD_FAILED, run_id="r2")
    assert r.skipped
    assert (r.node_uid, r.status, r.path) == (
        e.uid("trading"),
        Status.CLASSIFIED,
        "finance/crypto/trading",
    )


async def test_only_other_candidate_skips_model() -> None:
    # холодный старт: у root нет детей, выбирать не из чего
    e = await make()
    e.rules = FakeRules(load_tree("node: {id: root}"))
    svc = ClassifyService(
        e.rules,
        e.transcripts,
        e.classifications,
        e.material,
        e.decision,
        e.guard,
        e.starter,
        e.metrics,
    )
    root = e.rules.versions[1].root.uid
    assert await svc.step(1, root, "f1") == StepResult(kind="other", next_uid=None)
    assert e.decision.calls == 0
    assert e.metrics.counts["model_call"] == 0


async def _publish_with_new_child(e: Env, parent_id: str) -> int:
    """Публикация в обход RulesService: имитирует сбой до старта reprocess."""
    active = await e.rules.active()
    tree = add_children(active, e.uid(parent_id), [_proposal("loans")])
    return await e.rules.publish(tree, active.version, "manual")


def _proposal(id_: str) -> TopicProposal:
    return TopicProposal(id=id_, title=id_, description=id_, examples=[], guard="q?")


async def test_reconcile_starts_reprocess_stranded_by_partial_publish() -> None:
    e = await make()
    await e.svc.finalize("f1", 1, e.uid("finance"), Status.IN_OTHER)
    v2 = await _publish_with_new_child(e, "finance")
    assert await e.svc.reconcile_reprocess() == 1
    assert e.starter.reprocess == [("f1", e.uid("finance"), v2)]


async def test_reconcile_skips_items_already_at_current_children() -> None:
    e = await make()
    v2 = await _publish_with_new_child(e, "finance")
    await e.svc.finalize("f1", v2, e.uid("finance"), Status.IN_OTHER)
    assert await e.svc.reconcile_reprocess() == 0
    assert e.starter.reprocess == []


async def test_overlapping_reprocess_after_final_is_silent_skip() -> None:
    e = await make()
    fin = e.uid("finance")
    await e.svc.finalize("f1", 1, fin, Status.IN_OTHER)
    await e.svc.finalize(
        "f1", 1, e.uid("trading"), Status.CLASSIFIED, from_other_uid=fin, run_id="a"
    )
    r = await e.svc.finalize(
        "f1", 1, e.uid("banks"), Status.CLASSIFIED, from_other_uid=fin, run_id="b"
    )
    assert r.skipped and r.path == "finance/crypto/trading"
    assert e.metrics.counts["topic_move_attempt"] == 0
    assert await e.material.list(e.uid("banks")) == []


async def test_overlapping_reprocess_after_move_to_other_other_is_skipped() -> None:
    e = await make()
    fin, crypto = e.uid("finance"), e.uid("crypto")
    await e.svc.finalize("f1", 1, fin, Status.IN_OTHER)
    await e.svc.finalize("f1", 1, crypto, Status.IN_OTHER, from_other_uid=fin, run_id="a")
    r = await e.svc.finalize(
        "f1", 1, e.uid("banks"), Status.CLASSIFIED, from_other_uid=fin, run_id="b"
    )
    assert r.skipped and r.path == "finance/crypto/other"
    assert e.metrics.counts["topic_move_attempt"] == 0
    assert await e.material.list(crypto) == [MaterialItem("f1", "other")]
    assert await e.material.list(e.uid("banks")) == []
    assert (await e.classifications.get("f1")).node_uid == crypto


async def test_still_in_other() -> None:
    e = await make()
    fin = e.uid("finance")
    assert not await e.svc.still_in_other("f1", fin)  # классификации нет
    await e.svc.finalize("f1", 1, fin, Status.IN_OTHER)
    assert await e.svc.still_in_other("f1", fin)
    await e.svc.finalize("f1", 1, e.uid("crypto"), Status.IN_OTHER, from_other_uid=fin)
    assert not await e.svc.still_in_other("f1", fin)
