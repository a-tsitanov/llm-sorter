import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from llm_sorter.application.actions import ActionService
from llm_sorter.application.classify import ClassifyService
from llm_sorter.application.discover import DiscoverService
from llm_sorter.application.rules import RulesService
from llm_sorter.domain.classification import Status, Transcript
from llm_sorter.domain.loader import load_tree
from llm_sorter.orchestration.activities import Activities
from llm_sorter.orchestration.workflows import ClassifyInput, ClassifyOutcome, ClassifyWorkflow
from llm_sorter.ports.models import InvalidModelOutput
from tests.fakes import (
    FakeClassifications,
    FakeEmbedder,
    FakeEmbeddings,
    FakeMaterial,
    FakeNotifier,
    FakeRules,
    FakeStarter,
    FakeStorage,
    FakeText,
    FakeTranscripts,
    RecordingMetrics,
    ScriptedDecision,
    ScriptedGuard,
)
from tests.helpers import remove_node, uid, with_settings

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
                - {type: translate, to: en}
                - {type: save}
        - id: banks
"""

ANSWERS = {
    "finance,other": "finance",
    "crypto,banks,other": "crypto",
    "trading,other": "trading",
}


class FlakyStorage(FakeStorage):
    def __init__(self) -> None:
        super().__init__()
        self.broken = False  # ValueError: не ретраится, workflow падает
        self.failures = 0  # столько первых записей падают с временной ошибкой
        self.attempts = 0

    async def write(self, rel_path: str, content: str) -> str:
        self.attempts += 1
        if self.broken:
            raise ValueError("bad path")
        if self.attempts <= self.failures:
            raise RuntimeError("disk busy")
        return await super().write(rel_path, content)


class BrokenGuard(ScriptedGuard):
    """Модель всегда отвечает мусором."""

    async def check(self, text: str, question: str) -> bool:
        self.calls += 1
        raise InvalidModelOutput("garbage")


class FlakyGuard(ScriptedGuard):
    """Первые `failures` вызовов — обрыв связи с LM Studio."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def check(self, text: str, question: str) -> bool:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("lm studio down")
        return True


@dataclass
class Deps:
    decision: ScriptedDecision = field(default_factory=lambda: ScriptedDecision(dict(ANSWERS)))
    guard: ScriptedGuard = field(default_factory=ScriptedGuard)
    rules: FakeRules = field(default_factory=lambda: FakeRules(load_tree(TREE_YAML)))
    transcripts: FakeTranscripts = field(default_factory=FakeTranscripts)
    classifications: FakeClassifications = field(default_factory=FakeClassifications)
    material: FakeMaterial = field(default_factory=FakeMaterial)
    storage: FlakyStorage = field(default_factory=FlakyStorage)
    metrics: RecordingMetrics = field(default_factory=RecordingMetrics)

    def activities(self) -> Activities:
        classify = ClassifyService(
            self.rules,
            self.transcripts,
            self.classifications,
            self.material,
            self.decision,
            self.guard,
            FakeStarter(),
            self.metrics,
        )
        actions = ActionService(
            self.transcripts,
            self.storage,
            FakeNotifier(),
            FakeText(),
            self.rules,
            self.metrics,
            "ru",
        )
        starter = FakeStarter()
        rules_service = RulesService(self.rules, self.material, starter, self.metrics)
        discover = DiscoverService(
            self.rules,
            rules_service,
            self.material,
            self.transcripts,
            FakeEmbeddings(),
            FakeEmbedder(),
            FakeText(),
            self.metrics,
        )
        return Activities(
            classify=classify,
            actions=actions,
            discover=discover,
            starter=starter,
            metrics=self.metrics,
        )


@pytest.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as e:
        yield e


@pytest.fixture
async def deps() -> Deps:
    d = Deps()
    await d.transcripts.put(
        Transcript(id="f1", text="про биткоин", received_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    return d


async def run_classify(
    env: WorkflowEnvironment,
    deps: Deps,
    file_id: str,
    inp: ClassifyInput | None = None,
    workflow_id: str | None = None,
) -> ClassifyOutcome:
    acts = deps.activities()
    queue = f"q-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=queue,
        workflows=[ClassifyWorkflow],
        activities=acts.all(),
    ):
        return await env.client.execute_workflow(
            ClassifyWorkflow.run,
            inp or ClassifyInput(file_id=file_id),
            id=workflow_id or file_id,
            task_queue=queue,
        )


async def test_happy_path_runs_actions(env, deps):
    out = await run_classify(env, deps, "f1")
    assert out.status == Status.CLASSIFIED and out.path == "finance/crypto/trading"
    assert not out.skipped
    assert "finance/crypto/trading/f1.txt" in deps.storage.files
    assert deps.storage.files["finance/crypto/trading/f1.en.txt"] == "[en] про биткоин"


async def test_guard_fail_default_action(env, deps):
    deps.guard.default = False
    out = await run_classify(env, deps, "f1")
    assert out.status == Status.GUARD_FAILED
    assert "_guard_failed/finance/crypto/trading/f1.txt" in deps.storage.files


async def test_invalid_decide_falls_to_other(env, deps):
    deps.decision = ScriptedDecision(lambda text, candidates: "nft")
    out = await run_classify(env, deps, "f1")
    assert out.status == Status.IN_OTHER and out.path == "other"
    assert deps.metrics.counts["model_error:decide"] == 1
    assert deps.decision.calls == 3  # 3 попытки на невалидный ответ
    assert deps.storage.files == {}


async def test_pinned_version(env, deps):
    v1 = (await deps.rules.active()).version
    tree_without_crypto = remove_node(await deps.rules.active(), "crypto")
    await deps.rules.publish(tree_without_crypto, base_version=v1, source="manual")
    out = await run_classify(env, deps, "f1", ClassifyInput(file_id="f1", version=v1))
    assert out.version == v1 and out.path == "finance/crypto/trading"


async def test_second_run_skips_actions(env, deps):
    await run_classify(env, deps, "f1")
    deps.storage.files.clear()
    out = await run_classify(env, deps, "f1", ClassifyInput(file_id="f1"), workflow_id="f1-again")
    assert out.skipped and deps.storage.files == {}


async def test_rerun_after_failed_actions_replays_stored_decision(env, deps):
    deps.storage.broken = True
    with pytest.raises(WorkflowFailureError):
        await run_classify(env, deps, "f1")
    assert deps.storage.attempts == 1  # ValueError не ретраится
    assert deps.classifications.items["f1"].path == "finance/crypto/trading"

    # при повторе модель «передумала», но решение уже зафиксировано
    deps.storage.broken = False
    deps.decision = ScriptedDecision({**ANSWERS, "crypto,banks,other": "banks"})
    deps.guard.default = False
    out = await run_classify(env, deps, "f1")
    assert not out.skipped
    assert out.status == Status.CLASSIFIED and out.path == "finance/crypto/trading"
    assert set(deps.storage.files) == {
        "finance/crypto/trading/f1.txt",
        "finance/crypto/trading/f1.en.txt",
    }
    assert deps.metrics.counts["topic_move_attempt"] == 0


async def test_guard_error_counts_as_failed(env, deps):
    deps.guard = BrokenGuard()
    out = await run_classify(env, deps, "f1")
    assert out.status == Status.GUARD_FAILED
    assert deps.guard.calls == 3
    assert "_guard_failed/finance/crypto/trading/f1.txt" in deps.storage.files
    assert deps.metrics.counts["model_error:guard"] == 1


async def test_traversal_overflow_fails_workflow(env, deps):
    # дерево глубже max_depth: шагов не хватает — ошибка, а не in_other на листе
    deps.rules = FakeRules(with_settings(load_tree(TREE_YAML), max_depth=1))
    with pytest.raises(WorkflowFailureError):
        await run_classify(env, deps, "f1")
    assert deps.classifications.items == {}


async def test_guard_outage_waits_instead_of_failing(env, deps):
    # недоступность модели — не ответ: activity ретраится, файл не уходит в guard_failed
    deps.guard = FlakyGuard(failures=4)
    out = await run_classify(env, deps, "f1")
    assert out.status == Status.CLASSIFIED
    assert deps.guard.calls == 5
    assert deps.metrics.counts["model_error:guard"] == 0


async def test_transient_action_error_is_retried_until_success(env, deps):
    deps.storage.failures = 6  # больше прежнего лимита в 5 попыток
    out = await run_classify(env, deps, "f1")
    assert out.status == Status.CLASSIFIED
    assert deps.storage.attempts == 6 + 2
    assert "finance/crypto/trading/f1.txt" in deps.storage.files


async def test_decide_outage_waits_instead_of_other(env, deps):
    answers = ScriptedDecision(dict(ANSWERS))

    class FlakyDecision:
        calls = 0

        async def decide(self, text, candidates):
            FlakyDecision.calls += 1
            if FlakyDecision.calls <= 4:
                raise ConnectionError("lm studio down")
            return await answers.decide(text, candidates)

    deps.decision = FlakyDecision()  # type: ignore[assignment]
    out = await run_classify(env, deps, "f1")
    assert out.status == Status.CLASSIFIED and out.path == "finance/crypto/trading"
    assert deps.metrics.counts["model_error:decide"] == 0


async def test_reprocess_of_already_moved_file_skips_without_llm(env, deps):
    await run_classify(env, deps, "f1")  # classified: finance/crypto/trading
    calls = deps.decision.calls
    tree = await deps.rules.active()
    fin = uid(tree, "finance")
    inp = ClassifyInput(file_id="f1", start_node_uid=fin, version=tree.version, from_other_uid=fin)
    out = await run_classify(env, deps, "f1", inp, workflow_id="reprocess:f1:1")
    assert out.skipped and out.path == "finance/crypto/trading"
    assert deps.decision.calls == calls and deps.guard.calls == 1
    assert deps.metrics.counts["topic_move_attempt"] == 0
