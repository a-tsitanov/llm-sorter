"""Обвязка не должна быть узким местом: 1000 сообщений на фейковых моделях < 5 с."""

import asyncio
import time
from datetime import UTC, datetime

from llm_sorter.adapters.inbound.processor import handle_delivery
from llm_sorter.application.process import ProcessService
from llm_sorter.domain.messages import Transcript
from llm_sorter.domain.traversal import Candidate
from llm_sorter.domain.tree import Tree
from tests.fakes import FakeBus, RecordingMetrics
from tests.helpers import make_tree, topic

NOW = datetime(2026, 10, 23, tzinfo=UTC)
N = 1000
CONCURRENCY = 16


class FirstChild:
    async def decide(self, text: str, candidates: list[Candidate]) -> str:
        await asyncio.sleep(0)  # как сетевой вызов: отдаёт управление
        return candidates[0].id


class Pass:
    async def check(self, text: str, question: str) -> bool:
        await asyncio.sleep(0)
        return True


class Msg:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.headers: dict[str, object] = {}
        self.acked = False

    async def ack(self) -> None:
        self.acked = True

    async def reject(self, requeue: bool = False) -> None: ...

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None: ...


class Static:
    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    async def get(self, min_rev: int = 0) -> Tree:
        return self.tree


async def test_thousand_messages_under_five_seconds() -> None:
    tree = make_tree(topic("a"), topic("b", "a"), topic("c", "b"), topic("x"), topic("y", "x"))
    service = ProcessService(FirstChild(), Pass(), RecordingMetrics())
    bus, cache, stop = FakeBus(), Static(tree), asyncio.Event()
    msgs = [
        Msg(Transcript(id=f"m{i}", text="текст", received_at=NOW).model_dump_json().encode())
        for i in range(N)
    ]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(m: Msg) -> None:
        async with sem:
            await handle_delivery(m, service, cache, bus, stop)

    started = time.perf_counter()
    await asyncio.gather(*(one(m) for m in msgs))
    elapsed = time.perf_counter() - started
    assert all(m.acked for m in msgs) and len(bus.classified_msgs) == N
    assert elapsed < 5.0, f"{N} messages took {elapsed:.2f}s"
