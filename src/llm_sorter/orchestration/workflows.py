"""Temporal workflow: классификация файла, discover узла, периодический sweep."""

from datetime import timedelta

from pydantic import BaseModel, ConfigDict
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from llm_sorter.application.classify import FinalizeResult
    from llm_sorter.application.discover import DiscoverResult
    from llm_sorter.domain.classification import Status
    from llm_sorter.orchestration.activities import Activities

MODEL_TIMEOUT = timedelta(seconds=120)
# невалидный ответ ретраится внутри activity (MODEL_ATTEMPTS) и дальше не повторяется;
# недоступность LM Studio/Postgres — бесконечные ретраи с backoff: файл ждёт
MODEL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=0,
    non_retryable_error_types=["InvalidModelOutput"],
)
ACTION_RETRY = RetryPolicy(
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=0,
    non_retryable_error_types=["ValueError"],
)
SHORT_TIMEOUT = timedelta(seconds=30)
DISCOVER_TIMEOUT = timedelta(minutes=30)
DISCOVER_RETRY = RetryPolicy(maximum_attempts=2)
DISCOVER_HEARTBEAT = timedelta(minutes=5)  # один вызов LLM до 120 с плюс ретраи


def _invalid_model_output(e: ActivityError) -> bool:
    cause = e.cause
    return isinstance(cause, ApplicationError) and cause.type == "InvalidModelOutput"


class ClassifyInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: str
    start_node_uid: str | None = None
    version: int | None = None
    from_other_uid: str | None = None


class ClassifyOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Status
    path: str
    version: int
    skipped: bool


@workflow.defn
class ClassifyWorkflow:
    @workflow.run
    async def run(self, inp: ClassifyInput) -> ClassifyOutcome:
        version = inp.version
        if version is None:
            version = await workflow.execute_activity_method(
                Activities.active_version, start_to_close_timeout=SHORT_TIMEOUT
            )
        node = inp.start_node_uid
        if node is None:
            node = await workflow.execute_activity_method(
                Activities.root_uid, version, start_to_close_timeout=SHORT_TIMEOUT
            )
        if inp.from_other_uid is not None and not await workflow.execute_activity_method(
            Activities.still_in_other,
            args=[inp.file_id, inp.from_other_uid],
            start_to_close_timeout=SHORT_TIMEOUT,
        ):
            # файл уже увёл другой reprocess (или это повтор этого запуска): без вызовов LLM,
            # finalize вернёт сохранённое состояние
            result = await self._finalize_raw(inp, version, node, Status.IN_OTHER)
            return await self._run_actions(inp, result)
        max_depth = await workflow.execute_activity_method(
            Activities.max_depth, version, start_to_close_timeout=SHORT_TIMEOUT
        )

        # защита от зацикливания: шагов вниз не больше, чем глубина дерева
        for _ in range(max_depth + 2):
            try:
                step = await workflow.execute_activity_method(
                    Activities.step,
                    args=[version, node, inp.file_id],
                    start_to_close_timeout=MODEL_TIMEOUT,
                    retry_policy=MODEL_RETRY,
                )
            except ActivityError as e:
                if not _invalid_model_output(e):
                    raise
                await self._model_error("decide")
                return await self._finalize(inp, version, node, Status.IN_OTHER)
            if step.kind == "other":
                return await self._finalize(inp, version, node, Status.IN_OTHER)
            if step.kind == "leaf":
                break
            if step.next_uid is None:
                raise ApplicationError("descend without next_uid", non_retryable=True)
            node = step.next_uid
        else:
            raise ApplicationError(
                f"tree traversal exceeded {max_depth + 2} steps", non_retryable=True
            )

        try:
            passed = await workflow.execute_activity_method(
                Activities.guard_check,
                args=[version, node, inp.file_id],
                start_to_close_timeout=MODEL_TIMEOUT,
                retry_policy=MODEL_RETRY,
            )
        except ActivityError as e:
            if not _invalid_model_output(e):
                raise
            await self._model_error("guard")
            passed = False
        status = Status.CLASSIFIED if passed else Status.GUARD_FAILED
        result = await self._finalize_raw(inp, version, node, status)
        return await self._run_actions(inp, result)

    async def _run_actions(self, inp: ClassifyInput, result: FinalizeResult) -> ClassifyOutcome:
        outcome = self._outcome(result)
        if result.skipped or result.status == Status.IN_OTHER:
            return outcome

        # план строится по зафиксированному решению: при повторе запуска оно
        # может отличаться от только что вычисленного
        plan = await workflow.execute_activity_method(
            Activities.node_plan,
            args=[result.version, result.node_uid, result.status == Status.GUARD_FAILED],
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        ctx = await workflow.execute_activity_method(
            Activities.load_context, inp.file_id, start_to_close_timeout=SHORT_TIMEOUT
        )
        for action in plan.actions:
            ctx = await workflow.execute_activity_method(
                Activities.run_action,
                args=[ctx, action, plan.path],
                start_to_close_timeout=MODEL_TIMEOUT,
                retry_policy=ACTION_RETRY,
            )
        return outcome

    async def _finalize(
        self, inp: ClassifyInput, version: int, node: str, status: Status
    ) -> ClassifyOutcome:
        return await self._run_actions(inp, await self._finalize_raw(inp, version, node, status))

    @staticmethod
    def _outcome(result: FinalizeResult) -> ClassifyOutcome:
        return ClassifyOutcome(
            status=result.status, path=result.path, version=result.version, skipped=result.skipped
        )

    async def _finalize_raw(
        self, inp: ClassifyInput, version: int, node: str, status: Status
    ) -> FinalizeResult:
        # run_id = workflow_id: повтор того же запуска после частичного сбоя не считается skipped
        return await workflow.execute_activity_method(
            Activities.finalize,
            args=[
                inp.file_id,
                version,
                node,
                status,
                inp.from_other_uid,
                workflow.info().workflow_id,
            ],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

    async def _model_error(self, role: str) -> None:
        await workflow.execute_activity_method(
            Activities.record_model_error, role, start_to_close_timeout=SHORT_TIMEOUT
        )


@workflow.defn(name="DiscoverWorkflow")  # стартует по имени из TemporalStarter
class DiscoverWorkflow:
    @workflow.run
    async def run(self, node_uid: str) -> DiscoverResult:
        return await workflow.execute_activity_method(
            Activities.discover_run,
            node_uid,
            start_to_close_timeout=DISCOVER_TIMEOUT,
            heartbeat_timeout=DISCOVER_HEARTBEAT,
            retry_policy=DISCOVER_RETRY,
        )


@workflow.defn
class SweepWorkflow:
    """Запускает discover для узлов, набравших порог нового материала, и reconcile reprocess."""

    @workflow.run
    async def run(self) -> int:
        nodes = await workflow.execute_activity_method(
            Activities.due_nodes, start_to_close_timeout=SHORT_TIMEOUT
        )
        started = 0
        for uid in nodes:
            if await workflow.execute_activity_method(
                Activities.start_discover, uid, start_to_close_timeout=SHORT_TIMEOUT
            ):
                started += 1
        # догоняем reprocess, не стартовавший после частично прошедшей публикации
        await workflow.execute_activity_method(
            Activities.reconcile_reprocess, start_to_close_timeout=DISCOVER_TIMEOUT
        )
        return started
