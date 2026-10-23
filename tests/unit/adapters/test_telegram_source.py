import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from telethon import utils
from telethon.tl.types import DialogFilter, InputPeerChannel, TextWithEntities

from llm_sorter.adapters.inbound.telegram_source import (
    TelegramSource,
    TelethonGateway,
    TgChat,
    TgMessage,
)

LONG = "x" * 100
DATE = datetime(2026, 10, 1, tzinfo=UTC)


class FakeGateway:
    def __init__(self, history: dict[int, list[tuple[int, str]]]) -> None:
        self.history = history
        self.calls: list[tuple[int, int | None, int]] = []

    async def folder_chats(self, folders: Sequence[str]) -> list[TgChat]:
        return [TgChat(cid, f"chat{cid}", f"u{cid}", "News", None) for cid in self.history]

    async def messages(self, chat: TgChat, min_id: int | None, limit: int) -> list[TgMessage]:
        self.calls.append((chat.id, min_id, limit))
        msgs = [TgMessage(i, t, DATE) for i, t in self.history[chat.id]]
        if min_id is None:
            return msgs[-limit:]
        return [m for m in msgs if m.id > min_id][:limit]


async def test_backfill_then_cursor_after_ack(tmp_path: Path) -> None:
    gw = FakeGateway({7: [(i, LONG) for i in range(1, 31)]})
    src = TelegramSource(gw, ["News"], tmp_path / "state.json", backfill=5)
    got = await src.fetch_new()
    assert [t.id for t in got] == [f"tg-7-{i}" for i in range(26, 31)]
    assert got[0].metadata["url"] == "https://t.me/u7/26"
    assert got[0].metadata["folder"] == "News"
    assert [t.id for t in await src.fetch_new()] == [t.id for t in got]  # без ack - снова

    await src.ack([t.id for t in got])
    assert json.loads((tmp_path / "state.json").read_text()) == {"7": 30}
    assert await src.fetch_new() == []
    assert gw.calls[-1] == (7, 30, 100)

    gw.history[7].append((31, LONG))
    # новый экземпляр читает курсор с диска
    src2 = TelegramSource(gw, ["News"], tmp_path / "state.json")
    assert [t.id for t in await src2.fetch_new()] == ["tg-7-31"]


async def test_short_messages_move_cursor_only_before_first_post(tmp_path: Path) -> None:
    gw = FakeGateway({7: [(1, "фото"), (2, ""), (3, LONG), (4, "коротко"), (5, LONG)]})
    state = tmp_path / "state.json"
    src = TelegramSource(gw, ["News"], state, backfill=10)
    assert [t.id for t in await src.fetch_new()] == ["tg-7-3", "tg-7-5"]
    # 1, 2 пропущены сразу; 4 стоит после неподтверждённого 3 - курсор не трогаем
    assert json.loads(state.read_text()) == {"7": 2}

    await src.ack(["tg-7-3"])
    assert [t.id for t in await src.fetch_new()] == ["tg-7-5"]
    assert json.loads(state.read_text()) == {"7": 4}


async def test_only_short_messages_do_not_block(tmp_path: Path) -> None:
    gw = FakeGateway({7: [(i, "ok") for i in range(1, 4)]})
    src = TelegramSource(gw, ["News"], tmp_path / "state.json")
    assert await src.fetch_new() == []
    assert gw.calls[-1] == (7, None, 20)
    await src.fetch_new()
    assert gw.calls[-1] == (7, 3, 100)


async def test_ack_keeps_max_and_ignores_foreign_ids(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    src = TelegramSource(FakeGateway({}), [], state)
    await src.ack(["tg-7-10", "tg-7-4", "tg-8-1", "a.txt", "tg-x-1"])
    assert json.loads(state.read_text()) == {"7": 10, "8": 1}


def _peer(cid: int) -> InputPeerChannel:
    return InputPeerChannel(channel_id=cid, access_hash=cid * 10)


def _folder(title: str, include: list[Any], **kw: Any) -> DialogFilter:
    return DialogFilter(
        id=len(title),
        title=TextWithEntities(text=title, entities=[]),
        pinned_peers=[],
        include_peers=include,
        exclude_peers=kw.pop("exclude", []),
        **kw,
    )


class FakeClient:
    def __init__(self, filters: list[Any], dialogs: list[Any]) -> None:
        self.filters = filters
        self.dialogs = dialogs
        self.entity_calls = 0

    async def __call__(self, request: Any) -> Any:
        return SimpleNamespace(filters=self.filters)

    async def get_entity(self, peer: Any) -> Any:
        self.entity_calls += 1
        return SimpleNamespace(title=f"chan{peer.channel_id}", username=None)

    async def iter_dialogs(self) -> AsyncIterator[Any]:
        for d in self.dialogs:
            yield d


def _dialog(cid: int, *, archived: bool = False) -> Any:
    peer = _peer(cid)
    return SimpleNamespace(
        id=utils.get_peer_id(peer),
        input_entity=peer,
        entity=SimpleNamespace(title=f"dlg{cid}", username=f"d{cid}"),
        is_channel=True,
        is_group=False,
        archived=archived,
    )


async def test_gateway_resolves_folders() -> None:
    client = FakeClient(
        [
            _folder("News", [_peer(1), _peer(2)], exclude=[_peer(2)]),
            _folder("Каналы", [_peer(1)], broadcasts=True, exclude_archived=True),
            _folder("Other", [_peer(9)]),
        ],
        [_dialog(3), _dialog(4, archived=True)],
    )
    gw = TelethonGateway(client)
    chats = await gw.folder_chats(["News", "Каналы"])
    # 2 исключён, 1 уже взят из News, 4 в архиве
    assert [(c.id, c.title, c.folder) for c in chats] == [
        (1, "chan1", "News"),
        (3, "dlg3", "Каналы"),
    ]
    assert await gw.folder_chats(["News", "Каналы"]) == chats  # кеш
    assert client.entity_calls == 1
    assert await gw.folder_titles() == ["News", "Каналы", "Other"]


async def test_gateway_unknown_folder() -> None:
    gw = TelethonGateway(FakeClient([_folder("News", [])], []))
    with pytest.raises(ValueError, match=r"unknown telegram folders \['Nope'\]"):
        await gw.folder_chats(["Nope"])
