from datetime import UTC, datetime

import pytest

from llm_sorter.application.actions import ActionContext, ActionService
from llm_sorter.domain.actions import AnnotateAction, RocketAction, SaveAction, TranslateAction
from llm_sorter.domain.classification import Transcript
from llm_sorter.domain.loader import load_tree
from tests.fakes import (
    FakeNotifier,
    FakeRules,
    FakeStorage,
    FakeText,
    FakeTranscripts,
    NotifierCall,
    RecordingMetrics,
)
from tests.helpers import uid

TREE_YAML = """
node:
  id: root
  children:
    - id: finance
      children:
        - id: crypto
          children:
            - id: trading
              actions:
                - {type: save}
"""


@pytest.fixture
def storage():
    return FakeStorage()


@pytest.fixture
def notifier():
    return FakeNotifier()


@pytest.fixture
def metrics():
    return RecordingMetrics()


@pytest.fixture
def v():
    return 1


@pytest.fixture
async def svc(storage, notifier, metrics):
    transcripts = FakeTranscripts()
    await transcripts.put(
        Transcript(id="f1", text="hello world", received_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    rules = FakeRules(load_tree(TREE_YAML))
    return ActionService(transcripts, storage, notifier, FakeText(), rules, metrics, "ru")


async def test_chain_translate_annotate_save(svc, storage):
    ctx = await svc.load_context("f1")
    for a in [
        TranslateAction(type="translate", to="en"),
        AnnotateAction(type="annotate"),
        SaveAction(type="save"),
    ]:
        ctx = await svc.run(ctx, a, "finance/crypto/trading")
    assert set(storage.files) == {
        "finance/crypto/trading/f1.txt",
        "finance/crypto/trading/f1.summary.txt",
        "finance/crypto/trading/f1.en.txt",
    }
    assert storage.files["finance/crypto/trading/f1.en.txt"] == "[en] hello world"


async def test_save_explicit_path(svc, storage):
    ctx = await svc.load_context("f1")
    await svc.run(ctx, SaveAction(type="save", path="out"), "finance/crypto/trading")
    assert set(storage.files) == {"out/f1.txt"}


async def test_rocket_gets_summary(svc, notifier):
    ctx = ActionContext(file_id="f1", text="t", summary="s")
    await svc.run(ctx, RocketAction(type="rocket", channel="#c"), "x")
    assert notifier.calls == [NotifierCall("#c", "s", "f1.txt", "t")]


async def test_rocket_without_summary_truncates(svc, notifier):
    ctx = ActionContext(file_id="f1", text="a" * 600)
    await svc.run(ctx, RocketAction(type="rocket", channel="#c"), "x")
    assert notifier.calls == [NotifierCall("#c", "a" * 500, "f1.txt", "a" * 600)]


async def test_node_plan_regular(svc, v):
    tree = await svc._rules.get(v)
    actions, path = await svc.node_plan(v, uid(tree, "trading"), guard_failed=False)
    assert actions == [SaveAction(type="save")]
    assert path == "finance/crypto/trading"


async def test_guard_failed_plan(svc, v):
    tree = await svc._rules.get(v)
    actions, path = await svc.node_plan(v, uid(tree, "trading"), guard_failed=True)
    assert actions == [SaveAction(type="save", path="_guard_failed/finance/crypto/trading")]
    assert path == "finance/crypto/trading"


async def test_metrics_and_error(svc, metrics):
    ctx = await svc.load_context("f1")
    await svc.run(ctx, AnnotateAction(type="annotate"), "x")
    assert metrics.counts["action_done"] == 1

    async def boom(*a, **k):
        raise RuntimeError("x")

    svc._storage.write = boom
    with pytest.raises(RuntimeError):
        await svc.run(ctx, SaveAction(type="save"), "x")
    assert metrics.counts["action_done"] == 2
