"""Обработка сообщения: спуск по дереву до листа, гейт листа, исход classified/other."""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from llm_sorter.domain.messages import Classified, OtherItem, Reason, TopicRef, Transcript
from llm_sorter.domain.traversal import candidates_for, guard_question, resolve_choice
from llm_sorter.domain.tree import OTHER_ID, ROOT_ID, Tree
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.models import DecisionModel, GuardModel, InvalidModelOutput

T = TypeVar("T")

# попыток на невалидный ответ модели; сбои связи ждёт обработчик (без ограничения)
MODEL_ATTEMPTS = 3


@dataclass(frozen=True)
class Outcome:
    classified: Classified | None
    # 0..2: остановка/отказ гейта, плюс копия материала листа (reason=leaf)
    others: list[OtherItem]


class ProcessService:
    def __init__(
        self,
        decision: DecisionModel,
        guard: GuardModel,
        metrics: Metrics,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        guard_enabled: bool = True,
    ) -> None:
        self._decision = decision
        self._guard = guard
        self._metrics = metrics
        self._now = now
        self._guard_enabled = guard_enabled

    async def _call_model(self, role: str, call: Callable[[], Awaitable[T]]) -> T:
        """До MODEL_ATTEMPTS вызовов на InvalidModelOutput; прочие ошибки - сразу наверх."""
        for attempt in range(MODEL_ATTEMPTS):
            started = time.perf_counter()
            try:
                result = await call()
            except InvalidModelOutput:
                self._metrics.model_call(role, time.perf_counter() - started, False)
                if attempt == MODEL_ATTEMPTS - 1:
                    self._metrics.model_error(role)
                    raise
                continue
            except Exception:
                self._metrics.model_call(role, time.perf_counter() - started, False)
                raise
            self._metrics.model_call(role, time.perf_counter() - started, True)
            return result
        raise AssertionError("unreachable")

    async def _choose(self, text: str, tree: Tree, node: str) -> str | None:
        """id выбранного ребёнка; None - other (в том числе при невалидном ответе)."""
        candidates = candidates_for(tree, node)
        ids = {c.id for c in candidates}

        async def decide() -> str:
            choice = await self._decision.decide(text, candidates)
            if choice not in ids:
                raise InvalidModelOutput(f"unknown choice {choice!r}")
            return choice

        return resolve_choice(tree, node, await self._call_model("decide", decide))

    def _other(
        self, t: Transcript, tree: Tree, node: str, trail: list[str], reason: Reason, reprocess: int
    ) -> OtherItem:
        final = reason != "leaf" and reprocess >= tree.settings.max_reprocess
        return OtherItem(
            id=t.id,
            text=t.text,
            metadata=t.metadata,
            received_at=t.received_at,
            node=node,
            trail=trail,
            reason=reason,
            tree_rev=tree.rev,
            reprocess=reprocess,
            final=final,
            at=self._now(),
        )

    def _done(self, out: Outcome) -> Outcome:
        if out.classified is not None:
            self._metrics.processed("classified")
        elif any(o.final for o in out.others):
            self._metrics.processed("final_other")
        else:
            self._metrics.processed("other")
        return out

    async def process(self, t: Transcript, tree: Tree, reprocess: int = 0) -> Outcome:
        node = ROOT_ID
        trail: list[str] = []
        while not tree.is_leaf(node):
            only_other = [c.id for c in candidates_for(tree, node)] == [OTHER_ID]
            if only_other or not t.text.strip():
                # выбирать не из чего / нечего классифицировать: LLM не нужна
                other = self._other(t, tree, node, trail, "no_match", reprocess)
                return self._done(Outcome(None, [other]))
            try:
                choice = await self._choose(t.text, tree, node)
            except InvalidModelOutput:
                other = self._other(t, tree, node, trail, "model_error", reprocess)
                return self._done(Outcome(None, [other]))
            if choice is None:
                other = self._other(t, tree, node, trail, "no_match", reprocess)
                return self._done(Outcome(None, [other]))
            node = choice
            trail = [*trail, node]

        leaf = tree.get(node)
        parent = tree.parent(node)
        assert parent is not None  # у листа всегда есть родитель (root - не лист)
        path = tree.routing_key(node)
        question = guard_question(leaf)
        passed = True
        if self._guard_enabled:
            try:
                passed = await self._call_model(
                    "guard", lambda: self._guard.check(t.text, question)
                )
            except InvalidModelOutput:
                other = self._other(t, tree, parent.id, trail, "model_error", reprocess)
                return self._done(Outcome(None, [other]))
            self._metrics.guard_result(path, passed)
        if not passed:
            # ошиблась ветка: материал для соседней темы у родителя
            other = self._other(t, tree, parent.id, trail, "gate_failed", reprocess)
            return self._done(Outcome(None, [other]))

        classified = Classified(
            id=t.id,
            text=t.text,
            metadata=t.metadata,
            topic=TopicRef(id=leaf.id, path=path, title=leaf.title),
            gate="passed" if self._guard_enabled else "skipped",
            trail=trail,
            tree_rev=tree.rev,
            classified_at=self._now(),
        )
        others = []
        if tree.depth(node) < tree.max_depth_for(node):
            others.append(self._other(t, tree, node, trail, "leaf", reprocess))
        return self._done(Outcome(classified, others))
