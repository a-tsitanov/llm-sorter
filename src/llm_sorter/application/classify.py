import time
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from llm_sorter.domain.classification import (
    Classification,
    FinalClassificationExists,
    Status,
)
from llm_sorter.domain.discover import should_discover
from llm_sorter.domain.traversal import candidates_for, guard_question, resolve_choice
from llm_sorter.domain.tree import OTHER_ID, RuleTree
from llm_sorter.ports.io import WorkflowStarter
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.models import DecisionModel, GuardModel, InvalidModelOutput
from llm_sorter.ports.storage import (
    ClassificationRepository,
    MaterialRepository,
    RulesRepository,
    TranscriptStore,
)

T = TypeVar("T")

# попыток на невалидный ответ модели; сбои связи ретраит Temporal без ограничения
MODEL_ATTEMPTS = 3


class StepResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["descend", "other", "leaf"]
    next_uid: str | None


class FinalizeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    skipped: bool
    path: str
    node_uid: str  # при повторе/skipped — сохранённые значения, а не переданные
    status: Status
    version: int


class ClassifyService:
    def __init__(
        self,
        rules: RulesRepository,
        transcripts: TranscriptStore,
        classifications: ClassificationRepository,
        material: MaterialRepository,
        decision: DecisionModel,
        guard: GuardModel,
        starter: WorkflowStarter,
        metrics: Metrics,
    ) -> None:
        self._rules = rules
        self._transcripts = transcripts
        self._classifications = classifications
        self._material = material
        self._decision = decision
        self._guard = guard
        self._starter = starter
        self._metrics = metrics
        self._trees: dict[int, RuleTree] = {}

    async def _tree(self, version: int) -> RuleTree:
        # версии неизменяемы, поэтому кэш по номеру безопасен
        if version not in self._trees:
            self._trees[version] = await self._rules.get(version)
        return self._trees[version]

    async def active_version(self) -> int:
        tree = await self._rules.active()
        if tree.version is None:
            raise LookupError("active tree has no version")
        self._trees[tree.version] = tree
        return tree.version

    async def root_uid(self, version: int) -> str:
        return (await self._tree(version)).root.uid

    async def max_depth(self, version: int) -> int:
        """Наибольшая допустимая глубина в дереве (с учётом переопределений в узлах)."""
        tree = await self._tree(version)
        overrides = [n.max_depth for n in tree.iter_nodes() if n.max_depth is not None]
        return max([tree.settings.max_depth, *overrides])

    async def step(self, version: int, node_uid: str, file_id: str) -> StepResult:
        tree = await self._tree(version)
        candidates = candidates_for(tree, node_uid)
        if not candidates:
            return StepResult(kind="leaf", next_uid=None)
        if [c.id for c in candidates] == [OTHER_ID]:
            return StepResult(kind="other", next_uid=None)  # выбирать не из чего
        text = (await self._transcripts.get(file_id)).text
        if not text.strip():
            return StepResult(kind="other", next_uid=None)

        async def decide() -> str:
            choice = await self._decision.decide(text, candidates)
            if choice not in {c.id for c in candidates}:
                raise InvalidModelOutput(f"unknown choice {choice!r}")
            return choice

        choice = await self._call_model("decide", decide)
        next_uid = resolve_choice(tree, node_uid, choice)
        if next_uid is None:
            return StepResult(kind="other", next_uid=None)
        return StepResult(kind="descend", next_uid=next_uid)

    async def guard_check(self, version: int, node_uid: str, file_id: str) -> bool:
        tree = await self._tree(version)
        question = guard_question(tree.find(node_uid))
        text = (await self._transcripts.get(file_id)).text
        passed = await self._call_model("guard", lambda: self._guard.check(text, question))
        self._metrics.guard_result(tree.path_of(node_uid), passed)
        return passed

    async def _call_model(self, role: str, call: Callable[[], Awaitable[T]]) -> T:
        """До MODEL_ATTEMPTS вызовов на InvalidModelOutput; прочие ошибки — сразу наверх."""
        for attempt in range(MODEL_ATTEMPTS):
            started = time.perf_counter()
            try:
                result = await call()
            except InvalidModelOutput:
                self._metrics.model_call(role, time.perf_counter() - started, False)
                if attempt == MODEL_ATTEMPTS - 1:
                    raise
                continue
            except Exception:
                self._metrics.model_call(role, time.perf_counter() - started, False)
                raise
            self._metrics.model_call(role, time.perf_counter() - started, True)
            return result
        raise AssertionError("unreachable")

    async def finalize(
        self,
        file_id: str,
        version: int,
        node_uid: str,
        status: Status,
        from_other_uid: str | None = None,
        run_id: str | None = None,
    ) -> FinalizeResult:
        tree = await self._tree(version)
        path = self._path_for(tree, node_uid, status)
        if from_other_uid is not None:
            overtaken = await self._overtaken(file_id, from_other_uid, run_id)
            if overtaken is not None:
                return overtaken
        try:
            await self._classifications.save(
                Classification(
                    file_id=file_id,
                    tree_version=version,
                    node_uid=node_uid,
                    path=path,
                    status=status,
                    run_id=run_id,
                )
            )
        except FinalClassificationExists:
            existing = await self._classifications.get(file_id)
            if existing is None:
                raise
            stored = FinalizeResult(
                skipped=True,
                path=existing.path,
                node_uid=existing.node_uid,
                status=existing.status,
                version=existing.tree_version,
            )
            if run_id is None or existing.run_id != run_id:
                self._metrics.topic_move_attempt()
                return stored
            # повтор того же запуска: решение уже зафиксировано, побочные эффекты
            # применяются заново (идемпотентно) для сохранённых узла и статуса
            version, node_uid, status, path = (
                existing.tree_version,
                existing.node_uid,
                existing.status,
                existing.path,
            )
            tree = await self._tree(version)

        if from_other_uid is not None and (status != Status.IN_OTHER or node_uid != from_other_uid):
            await self._material.remove(from_other_uid, file_id)
            self._metrics.reprocess_moved(tree.other_path(from_other_uid), path)
        if status == Status.IN_OTHER:
            if node_uid != from_other_uid:  # иначе материал уже лежит в этом other
                await self._material.add(node_uid, file_id, "other")
        elif status == Status.CLASSIFIED and tree.depth_of(node_uid) < tree.max_depth_for(node_uid):
            await self._material.add(node_uid, file_id, "leaf")

        for uid in dict.fromkeys(  # размер буфера other изменился у затронутых узлов
            u for u in (from_other_uid, node_uid if status == Status.IN_OTHER else None) if u
        ):
            self._metrics.buffer_size(tree.other_path(uid), await self._material.other_size(uid))

        if status != Status.GUARD_FAILED:  # guard_failed не добавляет материал
            await self._maybe_discover(node_uid)
        self._metrics.file_classified(path, status, version)
        return FinalizeResult(
            skipped=False, path=path, node_uid=node_uid, status=status, version=version
        )

    async def _in_other_buffer(self, node_uid: str, file_id: str) -> bool:
        return any(
            item.file_id == file_id and item.kind == "other"
            for item in await self._material.list(node_uid)
        )

    async def still_in_other(self, file_id: str, from_other_uid: str) -> bool:
        """Файл всё ещё ждёт переобработки в `other` узла (его не увёл другой reprocess)."""
        existing = await self._classifications.get(file_id)
        if existing is None or existing.final:
            return False
        return existing.node_uid == from_other_uid or await self._in_other_buffer(
            from_other_uid, file_id
        )

    async def _overtaken(
        self, file_id: str, from_other_uid: str, run_id: str | None
    ) -> FinalizeResult | None:
        """Reprocess опоздал: файл уже финален или перенесён другим reprocess.
        Это не попытка переноса темы — skipped без метрики и без изменений материала."""
        existing = await self._classifications.get(file_id)
        if existing is None or (run_id is not None and existing.run_id == run_id):
            return None  # повтор того же запуска идёт обычным путём
        if not existing.final and await self.still_in_other(file_id, from_other_uid):
            return None
        return FinalizeResult(
            skipped=True,
            path=existing.path,
            node_uid=existing.node_uid,
            status=existing.status,
            version=existing.tree_version,
        )

    async def reconcile_reprocess(self) -> int:
        """Догоняет reprocess, не стартовавший после публикации (сбой между publish и стартом):
        файл в `other` узла X, классифицированный по версии, где у X было меньше живых детей.
        Id `reprocess:{file}:{version}` делает повторный старт идемпотентным."""
        active = await self._rules.active()
        if active.version is None:
            raise LookupError("active tree has no version")
        self._trees[active.version] = active
        started = 0
        for node in active.iter_nodes():
            if not node.children or await self._material.other_size(node.uid) == 0:
                continue
            live = {c.uid for c in node.children}
            for item in await self._material.list(node.uid):
                if item.kind != "other":
                    continue
                c = await self._classifications.get(item.file_id)
                if c is None or c.final:
                    continue
                try:
                    seen = {
                        ch.uid for ch in (await self._tree(c.tree_version)).find(node.uid).children
                    }
                except KeyError:  # узла не было в той версии
                    seen = set()
                if live - seen and await self._starter.start_reprocess(
                    item.file_id, node.uid, active.version
                ):
                    started += 1
        return started

    @staticmethod
    def _path_for(tree: RuleTree, node_uid: str, status: Status) -> str:
        node_path = tree.path_of(node_uid)
        if status == Status.IN_OTHER:
            return tree.other_path(node_uid)
        if status == Status.GUARD_FAILED:
            return "_guard_failed/" + node_path
        return node_path

    async def _maybe_discover(self, node_uid: str) -> None:
        # узел мог измениться после закрепления версии, поэтому смотрим активное дерево
        active = await self._rules.active()
        try:
            active.find(node_uid)
        except KeyError:
            return
        if should_discover(active, node_uid, await self._material.count_new(node_uid)):
            await self._starter.start_discover(node_uid)
