"""Ручная работа с деревом тем: просмотр, правки, импорт/экспорт YAML."""

from typing import Any

from llm_sorter.domain.changes import AddTopic, ArchiveTopic, Change, EditTopic, MoveTopic
from llm_sorter.domain.loader import dump_yaml, load_yaml
from llm_sorter.domain.tree import ROOT_ID, Topic, Tree
from llm_sorter.ports.store import ChangelogEntry, TopicStore


class TopicsService:
    def __init__(self, store: TopicStore) -> None:
        self._store = store

    async def show(self) -> list[str]:
        """Строки дерева: отступ по глубине, id, происхождение, routing key."""
        tree = await self._store.load_tree()
        lines = [f"/  rev {tree.rev}"]

        def walk(id: str, depth: int) -> None:
            for c in tree.children(id):
                lines.append(f"{'  ' * depth}{c.id}  [{c.origin}]  {tree.routing_key(c.id)}")
                walk(c.id, depth + 1)

        walk(ROOT_ID, 1)
        return lines

    async def _apply(self, changes: list[Change]) -> Tree:
        return await self._store.apply(changes, "cli")

    async def add(
        self, id: str, parent: str, title: str, description: str, guard: str | None = None
    ) -> Tree:
        t = Topic(id=id, parent_id=parent, title=title, description=description, guard=guard)
        return await self._apply([AddTopic(topic=t)])

    async def move(self, id: str, parent: str) -> Tree:
        return await self._apply([MoveTopic(id=id, new_parent_id=parent)])

    async def edit(self, id: str, **fields: Any) -> Tree:
        return await self._apply([EditTopic(id=id, fields=fields)])

    async def archive(self, id: str) -> Tree:
        return await self._apply([ArchiveTopic(id=id)])

    async def _import(self, text: str, replace: bool) -> Tree:
        settings, topics = load_yaml(text)
        tree = await self._store.load_tree()
        existing = tree.children(ROOT_ID)
        if existing and not replace:
            raise ValueError("tree not empty; use --replace")
        # настройки раньше тем: от max_depth зависит валидация
        if settings != tree.settings:
            await self._store.save_settings(settings)
        archive: list[Change] = [ArchiveTopic(id=c.id) for c in existing]
        return await self._apply([*archive, *(AddTopic(topic=t) for t in topics)])

    async def import_yaml(self, text: str) -> Tree:
        return await self._import(text, replace=False)

    async def import_yaml_replace(self, text: str) -> Tree:
        return await self._import(text, replace=True)

    async def changelog(self, limit: int = 50) -> list[ChangelogEntry]:
        return await self._store.changelog(limit)

    async def export_yaml(self) -> str:
        return dump_yaml(await self._store.load_tree())
