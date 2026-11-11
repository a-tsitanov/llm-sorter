"""Дообучение: буферы по узлам из stream `other`, discover по порогу, перегон сообщений.

Состояние сообщений живёт только в stream: Resolved/Considered - служебные записи,
которые дообучение само туда пишет. После перезапуска буферы восстанавливаются
чтением stream с сохранённой позиции (watermark)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from llm_sorter.application.discover import DiscoverResult
from llm_sorter.domain.changes import AddTopic
from llm_sorter.domain.discover import should_discover, slots_left
from llm_sorter.domain.messages import AnyRecord, Considered, Marker, OtherItem, Resolved
from llm_sorter.domain.tree import Tree
from llm_sorter.domain.validation import TreeValidationError
from llm_sorter.ports.bus import Bus
from llm_sorter.ports.metrics import Metrics
from llm_sorter.ports.store import TopicStore

log = logging.getLogger(__name__)

OFFSET_KEY = "other_offset"
REFRESH_KEY = "last_refresh_at"


class Proposer(Protocol):
    async def propose(self, tree: Tree, node_id: str, texts: list[str]) -> DiscoverResult: ...


@dataclass
class Pending:
    offset: int
    item: OtherItem
    considered: bool = False


class Learner:
    def __init__(
        self,
        store: TopicStore,
        discover: Proposer,
        bus: Bus,
        metrics: Metrics,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_batch: int = 2000,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self._store = store
        self._discover = discover
        self._bus = bus
        self._metrics = metrics
        self._now = now
        self._max_batch = max_batch
        self._buffers: dict[str, dict[str, Pending]] = {}  # узел -> id -> копия
        self._last_offset = -1  # robust-переподключение отдаёт stream заново - не применять
        self._markers: set[str] = set()

    # --- состояние из stream ---

    def _drop(self, id: str, node: str) -> Pending | None:
        buf = self._buffers.get(node)
        if buf is None or id not in buf:
            return None
        p = buf.pop(id)
        if not self._buffers[node]:
            del self._buffers[node]
        return p

    def _resolve(self, ids: list[str], at: datetime, node: str | None = None) -> None:
        # копия новее записи - уже следующий круг этого сообщения, её не трогаем
        for key in [node] if node is not None else list(self._buffers):
            buf = self._buffers.get(key, {})
            for id in ids:
                if id in buf and buf[id].item.at <= at:
                    self._drop(id, key)

    def _consider(self, node: str, ids: list[str], at: datetime) -> None:
        buf = self._buffers.get(node, {})
        for id in ids:
            p = buf.get(id)
            if p is not None and p.item.at <= at:
                p.considered = True

    def has_seen(self, token: str) -> bool:
        return token in self._markers

    def feed(self, offset: int, record: AnyRecord) -> None:
        if offset <= self._last_offset:
            return
        self._last_offset = offset
        if isinstance(record, Marker):
            self._markers.add(record.token)
        elif isinstance(record, Resolved):
            self._resolve(record.ids, record.at, record.node)
        elif isinstance(record, Considered):
            self._consider(record.node, record.ids, record.at)
        else:
            # Старые записи: последняя копия ID глобальна. Новые: отдельно по веткам.
            for node in [record.node] if record.branch_scoped else list(self._buffers):
                self._drop(record.id, node)
            if record.final:
                return
            self._buffers.setdefault(record.node, {})[record.id] = Pending(offset, record)

    def buffer(self, node: str) -> list[Pending]:
        return sorted(self._buffers.get(node, {}).values(), key=lambda p: p.offset)

    def watermark(self) -> int | None:
        offsets = [p.offset for buf in self._buffers.values() for p in buf.values()]
        return min(offsets) if offsets else None

    async def save_offset(self, next_offset: int) -> None:
        """Позиция для перезапуска: самая старая ждущая копия, иначе всё прочитанное."""
        w = self.watermark()
        await self._store.set_state(OFFSET_KEY, str(next_offset if w is None else w))

    # --- discover ---

    async def run_due(self) -> list[str]:
        tree = await self._store.load_tree()
        self._metrics.tree_rev(tree.rev)
        done: list[str] = []
        for node in list(self._buffers):
            if node not in tree.topics:
                continue  # узел архивирован; материал ждёт ручной правки дерева
            buf = self.buffer(node)
            if slots_left(tree, node) <= 0:
                self._metrics.discover_blocked(node, True)
                log.warning(
                    "discover blocked at %s: max_children reached, %d waiting", node, len(buf)
                )
                continue
            self._metrics.discover_blocked(node, False)
            new_count = sum(not p.considered for p in buf)
            if not should_discover(tree, node, new_count):
                continue
            # Сначала новые сообщения, затем ранее рассмотренные для контекста.
            # Порог может быть больше лимита: иначе discover никогда не наберёт материал.
            limit = max(
                self._max_batch,
                tree.settings.discover.buffer_threshold,
                tree.settings.discover.min_cluster_size,
            )
            batch = ([p for p in buf if not p.considered] + [p for p in buf if p.considered])[
                :limit
            ]
            try:
                tree = await self._run_node(tree, node, batch, buf)
            except Exception:
                # узел, на котором discover падает (контекст модели, сбой), не держит остальные
                log.exception("discover at %s failed", node)
                self._metrics.model_error("discover")
                continue
            done.append(node)
        return done

    async def _run_node(
        self, tree: Tree, node: str, buf: list[Pending], full_buffer: list[Pending]
    ) -> Tree:
        started = self._now()
        was_leaf = tree.is_leaf(node)
        res = await self._discover.propose(tree, node, [p.item.text for p in buf])
        moved: list[Pending] = []
        if res.accepted:
            try:
                tree = await self._store.apply(
                    [AddTopic(topic=a.topic) for a in res.accepted], "learner"
                )
            except TreeValidationError as e:
                # дерево поменяли параллельно (CLI): прогон считается неудачным
                log.warning("discover at %s not applied: %s", node, e)
                res = DiscoverResult([], list(range(len(buf))), res.rejected)
            else:
                # проросший лист: всё его содержимое спускается в новые подтемы
                if was_leaf:
                    moved = full_buffer
                else:
                    members = sorted({i for a in res.accepted for i in a.members})
                    moved = [buf[i] for i in members]
        for p in moved:
            await self._bus.reinject(p.item.transcript(), p.item.reprocess + 1, tree.rev)
        if moved:
            ids = [p.item.id for p in moved]
            await self._bus.other(Resolved(ids=ids, at=started, node=node))
            self._resolve(ids, started, node)
        moved_ids = {p.item.id for p in moved}
        rest = [buf[i].item.id for i in res.considered if buf[i].item.id not in moved_ids]
        if rest:
            await self._bus.other(Considered(node=node, ids=rest, at=started))
            self._consider(node, rest, started)
        self._metrics.discover_run(node, len(res.accepted), res.rejected)
        self._metrics.buffer_size(node, len(self.buffer(node)))
        return tree

    async def refresh_old(self, max_age: timedelta) -> int:
        """Переиздаёт в конец stream копии старше max_age, чтобы их не удалил ретеншн.
        Копия получает at=now (иначе переиздавалась бы каждый цикл); признак considered
        переносится следующей за ней записью Considered с at не раньше копии."""
        now = self._now()
        cutoff = now - max_age
        count = 0
        for node in list(self._buffers):
            old = [p for p in self.buffer(node) if p.item.at < cutoff]
            for p in old:
                await self._bus.other(p.item.model_copy(update={"at": now}))
            considered = [p.item.id for p in old if p.considered]
            if considered:
                await self._bus.other(Considered(node=node, ids=considered, at=now))
            count += len(old)
        return count

    async def reconsider_all(self) -> int:
        """Переиздаёт все ждущие копии с at=now и без Considered: пометки «уже рассмотрено»
        снимаются (например, после discover, упавшего из-за неверной конфигурации моделей)."""
        now = self._now()
        count = 0
        for node in list(self._buffers):
            for p in self.buffer(node):
                await self._bus.other(p.item.model_copy(update={"at": now}))
                count += 1
        return count

    async def refresh_if_due(self, every: timedelta, max_age: timedelta) -> int | None:
        """refresh_old не чаще every; время последнего хранится в SQLite и переживает
        перезапуски (иначе при деплоях чаще раза в сутки переиздание не случалось бы)."""
        last = await self._store.get_state(REFRESH_KEY)
        now = self._now()
        if last is not None and now - datetime.fromisoformat(last) < every:
            return None
        count = await self.refresh_old(max_age)
        await self._store.set_state(REFRESH_KEY, now.isoformat())
        return count
