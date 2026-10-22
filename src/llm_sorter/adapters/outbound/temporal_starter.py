from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from llm_sorter.orchestration.workflows import ClassifyInput, ClassifyWorkflow

DISCOVER_WORKFLOW = "DiscoverWorkflow"  # по имени: определяется отдельно


class TemporalStarter:
    """WorkflowStarter поверх Temporal. False — workflow с этим id уже запущен
    (или завершён успешно, если политика повтора id это запрещает)."""

    def __init__(self, client: Client, task_queue: str) -> None:
        self._client = client
        self._queue = task_queue

    async def start_classify(self, file_id: str) -> bool:
        # повтор после неуспешного запуска разрешён: finalize воспроизводит
        # сохранённое решение по run_id (= workflow_id)
        return await self._start_classify(ClassifyInput(file_id=file_id), file_id)

    async def start_reprocess(self, file_id: str, node_uid: str, version: int) -> bool:
        inp = ClassifyInput(
            file_id=file_id, start_node_uid=node_uid, version=version, from_other_uid=node_uid
        )
        return await self._start_classify(inp, f"reprocess:{file_id}:{version}")

    async def start_discover(self, node_uid: str) -> bool:
        # discover узла запускается многократно: дубль отклоняется только пока бежит
        try:
            await self._client.start_workflow(
                DISCOVER_WORKFLOW,
                node_uid,
                id=f"discover:{node_uid}",
                task_queue=self._queue,
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            )
        except WorkflowAlreadyStartedError:
            return False
        return True

    async def _start_classify(self, inp: ClassifyInput, workflow_id: str) -> bool:
        try:
            await self._client.start_workflow(
                ClassifyWorkflow.run,
                inp,
                id=workflow_id,
                task_queue=self._queue,
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            return False
        return True
