import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from llm_sorter.adapters.outbound.temporal_starter import DISCOVER_WORKFLOW
from llm_sorter.application.actions import ActionService
from llm_sorter.application.classify import ClassifyService
from llm_sorter.application.discover import DiscoverResult, DiscoverService
from llm_sorter.application.rules import RulesService
from llm_sorter.domain.classification import Classification, Status, Transcript
from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.loader import load_tree
from llm_sorter.domain.tree_edit import add_children
from llm_sorter.orchestration.activities import Activities
from llm_sorter.orchestration.workflows import DiscoverWorkflow, SweepWorkflow
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
from tests.helpers import uid, with_settings

TREE_YAML = """
node:
  id: root
  children:
    - id: finance
      title: Финансы
      description: Деньги
      children:
        - id: banks
          title: Банки
          description: Банковские услуги
"""


class Env:
    def __init__(self) -> None:
        self.rules = FakeRules(
            with_settings(load_tree(TREE_YAML), buffer_threshold=3, min_cluster_size=3)
        )
        self.material = FakeMaterial()
        self.transcripts = FakeTranscripts()
        self.starter = FakeStarter()
        self.metrics = RecordingMetrics()
        self.text = FakeText()
        self.embedder = FakeEmbedder()
        self.classifications = FakeClassifications()

    def activities(self) -> Activities:
        classify = ClassifyService(
            self.rules,
            self.transcripts,
            self.classifications,
            self.material,
            ScriptedDecision(),
            ScriptedGuard(),
            self.starter,
            self.metrics,
        )
        actions = ActionService(
            self.transcripts,
            FakeStorage(),
            FakeNotifier(),
            self.text,
            self.rules,
            self.metrics,
            "ru",
        )
        rules_service = RulesService(self.rules, self.material, self.starter, self.metrics)
        discover = DiscoverService(
            self.rules,
            rules_service,
            self.material,
            self.transcripts,
            FakeEmbeddings(),
            self.embedder,
            self.text,
            self.metrics,
        )
        return Activities(
            classify=classify,
            actions=actions,
            discover=discover,
            starter=self.starter,
            metrics=self.metrics,
        )

    async def seed(self, node_uid: str, prefix: str, n: int = 3) -> None:
        for i in range(n):
            fid = f"{prefix}{i}"
            text = f"{prefix} {i}"
            self.embedder.mapping[text] = [1.0, 0.001 * i, 0, 0, 0, 0, 0, 0]
            await self.transcripts.put(
                Transcript(id=fid, text=text, received_at=datetime(2026, 1, 1, tzinfo=UTC))
            )
            await self.material.add(node_uid, fid, "other")


@pytest.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as e:
        yield e


def worker(env: WorkflowEnvironment, deps: Env, queue: str) -> Worker:
    return Worker(
        env.client,
        task_queue=queue,
        workflows=[DiscoverWorkflow, SweepWorkflow],
        activities=deps.activities().all(),
    )


async def test_sweep_starts_discover_for_due_nodes(env):
    deps = Env()
    tree = await deps.rules.active()
    u1, u2 = tree.root.uid, uid(tree, "finance")
    await deps.seed(u1, "a")
    await deps.seed(u2, "b")
    await deps.seed(uid(tree, "banks"), "c", n=2)  # ниже порога
    queue = f"q-{uuid.uuid4()}"
    async with worker(env, deps, queue):
        started = await env.client.execute_workflow(
            SweepWorkflow.run, id=f"sweep-{queue}", task_queue=queue
        )
    assert started == 2
    assert deps.starter.discover == [u1, u2]


async def test_sweep_counts_only_started(env):
    deps = Env()
    tree = await deps.rules.active()
    await deps.seed(tree.root.uid, "a")
    deps.starter.discover.append(tree.root.uid)  # уже бежит
    queue = f"q-{uuid.uuid4()}"
    async with worker(env, deps, queue):
        started = await env.client.execute_workflow(
            SweepWorkflow.run, id=f"sweep-{queue}", task_queue=queue
        )
    assert started == 0


async def test_discover_workflow_by_name(env):
    deps = Env()
    tree = await deps.rules.active()
    await deps.seed(tree.root.uid, "a")
    deps.text.next_proposal = TopicProposal(
        id="taxes", title="Налоги", description="Налоговые вопросы", examples=[], guard="q?"
    )
    queue = f"q-{uuid.uuid4()}"
    async with worker(env, deps, queue):
        res = await env.client.execute_workflow(
            DISCOVER_WORKFLOW,
            tree.root.uid,
            id=f"discover:{tree.root.uid}",
            task_queue=queue,
            result_type=DiscoverResult,
        )
    assert res.created == ["taxes"] and res.version == 2
    assert len(deps.starter.reprocess) == 3


async def test_sweep_reconciles_stranded_reprocess(env):
    deps = Env()
    tree = await deps.rules.active()
    fin = uid(tree, "finance")
    await deps.material.add(fin, "f1", "other")
    await deps.classifications.save(
        Classification(
            file_id="f1", tree_version=1, node_uid=fin, path="finance/other", status=Status.IN_OTHER
        )
    )
    # публикация прошла, а старт reprocess — нет
    loans = TopicProposal(id="loans", title="Кредиты", description="Займы", examples=[], guard="q?")
    v2 = await deps.rules.publish(add_children(tree, fin, [loans]), 1, "manual")
    queue = f"q-{uuid.uuid4()}"
    async with worker(env, deps, queue):
        await env.client.execute_workflow(SweepWorkflow.run, id=f"sweep-{queue}", task_queue=queue)
    assert deps.starter.reprocess == [("f1", fin, v2)]
