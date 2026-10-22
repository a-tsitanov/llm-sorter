"""Тонкие activity-прокси к прикладным сервисам."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict
from temporalio import activity

from llm_sorter.application.actions import ActionContext, ActionService
from llm_sorter.application.classify import ClassifyService, FinalizeResult, StepResult
from llm_sorter.application.discover import DiscoverResult, DiscoverService
from llm_sorter.domain.actions import Action
from llm_sorter.domain.classification import Status
from llm_sorter.ports.io import WorkflowStarter
from llm_sorter.ports.metrics import Metrics


class NodePlan(BaseModel):
    """Обёртка над (actions, path) для сериализации через pydantic-конвертер."""

    model_config = ConfigDict(frozen=True)

    actions: list[Action]
    path: str


class Activities:
    def __init__(
        self,
        *,
        classify: ClassifyService,
        actions: ActionService,
        discover: DiscoverService,
        starter: WorkflowStarter,
        metrics: Metrics,
    ) -> None:
        self._classify = classify
        self._actions = actions
        self._discover = discover
        self._starter = starter
        self._metrics = metrics

    def all(self) -> list[Callable[..., Any]]:
        """Список activity для регистрации в Worker."""
        return [
            self.active_version,
            self.root_uid,
            self.max_depth,
            self.step,
            self.guard_check,
            self.finalize,
            self.load_context,
            self.node_plan,
            self.run_action,
            self.record_model_error,
            self.discover_run,
            self.due_nodes,
            self.start_discover,
            self.still_in_other,
            self.reconcile_reprocess,
        ]

    @activity.defn
    async def active_version(self) -> int:
        return await self._classify.active_version()

    @activity.defn
    async def root_uid(self, version: int) -> str:
        return await self._classify.root_uid(version)

    @activity.defn
    async def max_depth(self, version: int) -> int:
        return await self._classify.max_depth(version)

    @activity.defn
    async def step(self, version: int, node_uid: str, file_id: str) -> StepResult:
        return await self._classify.step(version, node_uid, file_id)

    @activity.defn
    async def guard_check(self, version: int, node_uid: str, file_id: str) -> bool:
        return await self._classify.guard_check(version, node_uid, file_id)

    @activity.defn
    async def finalize(
        self,
        file_id: str,
        version: int,
        node_uid: str,
        status: Status,
        from_other_uid: str | None,
        run_id: str | None,
    ) -> FinalizeResult:
        return await self._classify.finalize(
            file_id, version, node_uid, status, from_other_uid=from_other_uid, run_id=run_id
        )

    @activity.defn
    async def load_context(self, file_id: str) -> ActionContext:
        return await self._actions.load_context(file_id)

    @activity.defn
    async def node_plan(self, version: int, node_uid: str, guard_failed: bool) -> NodePlan:
        actions, path = await self._actions.node_plan(version, node_uid, guard_failed)
        return NodePlan(actions=actions, path=path)

    @activity.defn
    async def run_action(self, ctx: ActionContext, action: Action, node_path: str) -> ActionContext:
        return await self._actions.run(ctx, action, node_path)

    @activity.defn
    async def record_model_error(self, role: str) -> None:
        self._metrics.model_error(role)

    @activity.defn
    async def discover_run(self, node_uid: str) -> DiscoverResult:
        return await self._discover.run(node_uid, progress=activity.heartbeat)

    @activity.defn
    async def due_nodes(self) -> list[str]:
        return await self._discover.due_nodes()

    @activity.defn
    async def start_discover(self, node_uid: str) -> bool:
        return await self._starter.start_discover(node_uid)

    @activity.defn
    async def still_in_other(self, file_id: str, from_other_uid: str) -> bool:
        return await self._classify.still_in_other(file_id, from_other_uid)

    @activity.defn
    async def reconcile_reprocess(self) -> int:
        return await self._classify.reconcile_reprocess()
