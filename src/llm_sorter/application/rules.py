from dataclasses import dataclass

from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.loader import dump_tree, load_tree
from llm_sorter.domain.tree import RuleTree
from llm_sorter.domain.tree_edit import (
    add_children,
    gained_children,
    live_ancestor_with_other,
    merge_archived,
)
from llm_sorter.domain.validation import TreeValidationError, validate_tree
from llm_sorter.ports.io import WorkflowStarter
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.storage import (
    MaterialRepository,
    PublishSource,
    RulesRepository,
    VersionConflict,
)

_DISCOVER_ATTEMPTS = 3


@dataclass(frozen=True)
class PublishResult:
    version: int
    reprocess_nodes: list[str]
    reprocess_started: int


class RulesService:
    def __init__(
        self,
        rules: RulesRepository,
        material: MaterialRepository,
        starter: WorkflowStarter,
        metrics: Metrics,
    ) -> None:
        self._rules = rules
        self._material = material
        self._starter = starter
        self._metrics = metrics

    async def _active_or_none(self) -> RuleTree | None:
        try:
            return await self._rules.active()
        except LookupError:
            return None

    async def publish_tree(
        self, new: RuleTree, base_version: int | None, source: PublishSource
    ) -> PublishResult:
        validate_tree(new)
        old = await self._active_or_none()
        merged = merge_archived(old, new) if old is not None else new
        version = await self._rules.publish(merged, base_version, source)
        reprocess_nodes: list[str] = []
        started = 0
        if old is not None:
            live = {n.uid for n in merged.iter_nodes()}
            for node in old.iter_nodes():
                # архивирован или лишился всех детей (у листа нет other): буфер наверх
                archived = node.uid not in live
                if archived or (old.has_other(node.uid) and not merged.has_other(node.uid)):
                    await self._material.move_other(
                        node.uid, live_ancestor_with_other(old, merged, node.uid)
                    )
            for uid in gained_children(old, merged):
                if await self._material.other_size(uid) > 0:
                    reprocess_nodes.append(uid)
            for uid in reprocess_nodes:
                for item in await self._material.list(uid):
                    if item.kind == "other" and await self._starter.start_reprocess(
                        item.file_id, uid, version
                    ):
                        started += 1
        self._metrics.rules_published(source)
        self._metrics.active_version(version)
        return PublishResult(version, reprocess_nodes, started)

    async def publish_yaml(self, text: str, base_version: int | None) -> PublishResult:
        tree = load_tree(text)
        old = await self._active_or_none()
        if old is not None and tree.root.uid != old.root.uid:
            raise TreeValidationError(
                [f"root uid mismatch ({tree.root.uid} != {old.root.uid}) — run `rules pull` first"]
            )
        return await self.publish_tree(tree, base_version, "manual")

    async def add_discovered(
        self, parent_uid: str, proposals: list[TopicProposal]
    ) -> PublishResult:
        for attempt in range(_DISCOVER_ATTEMPTS):
            active = await self._rules.active()
            tree = add_children(active, parent_uid, proposals)
            try:
                return await self.publish_tree(tree, active.version, "discover")
            except VersionConflict:
                if attempt == _DISCOVER_ATTEMPTS - 1:
                    raise
        raise AssertionError("unreachable")

    async def pull(self) -> str:
        return dump_tree(await self._rules.active())

    async def rollback(self, version: int) -> PublishResult:
        target = (await self._rules.get(version)).model_copy(update={"version": None})
        active = await self._rules.active()
        return await self.publish_tree(target, active.version, "rollback")

    async def bootstrap_if_empty(self) -> None:
        if await self._active_or_none() is not None:
            return
        await self.publish_yaml("node: {id: root}", None)
