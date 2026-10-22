import uuid
from collections.abc import AsyncIterator

import pytest
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from llm_sorter.adapters.outbound.temporal_starter import TemporalStarter
from llm_sorter.orchestration.workflows import ClassifyInput


@pytest.fixture(scope="module")
async def env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as e:
        yield e


@pytest.fixture
def starter(env: WorkflowEnvironment) -> TemporalStarter:
    # воркера нет: workflow остаются запущенными, этого достаточно для проверки id
    return TemporalStarter(env.client, f"q-{uuid.uuid4()}")


async def test_classify_started_once(env: WorkflowEnvironment, starter: TemporalStarter) -> None:
    fid = f"f-{uuid.uuid4()}"
    assert await starter.start_classify(fid) is True
    assert await starter.start_classify(fid) is False
    desc = await env.client.get_workflow_handle(fid).describe()
    assert desc.workflow_type == "ClassifyWorkflow"


async def test_classify_restarts_after_failed_run(
    env: WorkflowEnvironment, starter: TemporalStarter
) -> None:
    fid = f"f-{uuid.uuid4()}"
    assert await starter.start_classify(fid)
    await env.client.get_workflow_handle(fid).terminate("boom")
    assert await starter.start_classify(fid) is True


async def test_reprocess_id_and_input(env: WorkflowEnvironment, starter: TemporalStarter) -> None:
    fid = f"f-{uuid.uuid4()}"
    assert await starter.start_reprocess(fid, "n1", 7) is True
    assert await starter.start_reprocess(fid, "n1", 7) is False
    handle = env.client.get_workflow_handle(f"reprocess:{fid}:7")
    events = [e async for e in handle.fetch_history_events()]
    payload = events[0].workflow_execution_started_event_attributes.input.payloads[0]
    [inp] = await env.client.data_converter.decode([payload], [ClassifyInput])
    assert inp == ClassifyInput(file_id=fid, start_node_uid="n1", version=7, from_other_uid="n1")


async def test_discover_by_type_name(env: WorkflowEnvironment, starter: TemporalStarter) -> None:
    uid = f"u-{uuid.uuid4()}"
    assert await starter.start_discover(uid) is True
    assert await starter.start_discover(uid) is False
    desc = await env.client.get_workflow_handle(f"discover:{uid}").describe()
    assert desc.workflow_type == "DiscoverWorkflow"
