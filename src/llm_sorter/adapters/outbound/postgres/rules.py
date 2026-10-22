from typing import Literal

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_sorter.adapters.outbound.postgres.tables import active_rules, rule_versions
from llm_sorter.domain.loader import dump_tree, load_tree
from llm_sorter.domain.tree import RuleTree
from llm_sorter.ports.storage import VersionConflict

# ключ advisory-блокировки, сериализующей все publish (в т.ч. первый, когда active_rules пуста)
_PUBLISH_LOCK = 7_411_001


class PgRules:
    def __init__(self, sessions: "async_sessionmaker[AsyncSession]") -> None:
        self._sessions = sessions
        self._cache: dict[int, RuleTree] = {}  # версии неизменяемы

    async def active(self) -> RuleTree:
        async with self._sessions() as s:
            version = await s.scalar(select(active_rules.c.version).where(active_rules.c.id == 1))
        if version is None:
            raise LookupError("no active rules")
        return await self.get(version)

    async def get(self, version: int) -> RuleTree:
        cached = self._cache.get(version)
        if cached is not None:
            return cached
        async with self._sessions() as s:
            text = await s.scalar(
                select(rule_versions.c.yaml).where(rule_versions.c.version == version)
            )
        if text is None:
            raise LookupError(f"unknown version {version}")
        tree = load_tree(text).model_copy(update={"version": version})
        self._cache[version] = tree
        return tree

    async def publish(
        self,
        tree: RuleTree,
        base_version: int | None,
        source: Literal["manual", "discover", "rollback"],
    ) -> int:
        async with self._sessions.begin() as s:
            # FOR UPDATE при пустой active_rules ничего не блокирует, поэтому
            # сериализуем публикации advisory-локом на время транзакции
            await s.execute(select(func.pg_advisory_xact_lock(_PUBLISH_LOCK)))
            current = await s.scalar(
                select(active_rules.c.version).where(active_rules.c.id == 1).with_for_update()
            )
            if current != base_version:
                raise VersionConflict(f"base={base_version}, active={current}")
            text = dump_tree(tree.model_copy(update={"version": None}))
            new = await s.scalar(
                insert(rule_versions)
                .values(yaml=text, source=source, base_version=base_version)
                .returning(rule_versions.c.version)
            )
            assert new is not None
            if current is None:
                await s.execute(insert(active_rules).values(id=1, version=new))
            else:
                await s.execute(
                    update(active_rules).where(active_rules.c.id == 1).values(version=new)
                )
        return int(new)
