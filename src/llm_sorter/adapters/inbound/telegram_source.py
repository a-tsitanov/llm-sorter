"""Источник транскриптов из Telegram: посты чатов из выбранных папок (dialog filters)."""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from llm_sorter.domain.classification import Transcript

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TgChat:
    id: int  # id без маркера типа (как entity.id)
    title: str
    username: str | None
    folder: str
    peer: Any  # InputPeer для запросов истории


@dataclass(frozen=True)
class TgMessage:
    id: int
    text: str
    date: datetime


class TelegramGateway(Protocol):
    async def folder_chats(self, folders: Sequence[str]) -> list[TgChat]: ...

    async def messages(self, chat: TgChat, min_id: int | None, limit: int) -> list[TgMessage]:
        """По возрастанию id. min_id=None - последние limit сообщений, иначе limit после min_id."""
        ...


def transcript_id(chat_id: int, message_id: int) -> str:
    return f"tg-{chat_id}-{message_id}"


def _parse_id(tid: str) -> tuple[int, int] | None:
    prefix, _, rest = tid.partition("-")
    chat, _, msg = rest.partition("-")
    if prefix != "tg" or not chat.isdigit() or not msg.isdigit():
        return None
    return int(chat), int(msg)


class TelegramSource:
    """Курсор - последний подтверждённый id сообщения по каждому чату (JSON в state_file).
    Первый опрос чата без курсора берёт последние backfill сообщений.
    Сообщения короче min_len (подписи к медиа, сервисные) не отдаются; курсор
    сдвигается через них сразу, только пока перед ними нет неподтверждённых постов."""

    def __init__(
        self,
        gateway: TelegramGateway,
        folders: Sequence[str],
        state_file: Path,
        *,
        backfill: int = 20,
        batch: int = 100,
        min_len: int = 80,
    ) -> None:
        self._gw = gateway
        self._folders = list(folders)
        self._state_file = state_file
        self._backfill = backfill
        self._batch = batch
        self._min_len = min_len

    def _load(self) -> dict[str, int]:
        if not self._state_file.exists():
            return {}
        return {k: int(v) for k, v in json.loads(self._state_file.read_text("utf-8")).items()}

    def _save(self, state: dict[str, int]) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_file.with_name(self._state_file.name + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        tmp.replace(self._state_file)

    async def fetch_new(self) -> list[Transcript]:
        state = self._load()
        now = datetime.now(UTC)
        out: list[Transcript] = []
        moved = False
        for chat in await self._gw.folder_chats(self._folders):
            key = str(chat.id)
            last = state.get(key)
            limit = self._backfill if last is None else self._batch
            emitted = False
            for m in await self._gw.messages(chat, last, limit):
                text = m.text.strip()
                if len(text) < self._min_len:
                    if not emitted:
                        state[key] = m.id
                        moved = True
                    continue
                emitted = True
                meta: dict[str, Any] = {
                    "source": "telegram",
                    "folder": chat.folder,
                    "chat_id": chat.id,
                    "chat_title": chat.title,
                    "chat_username": chat.username,
                    "message_id": m.id,
                    "date": m.date.isoformat(),
                }
                if chat.username:
                    meta["url"] = f"https://t.me/{chat.username}/{m.id}"
                out.append(
                    Transcript(
                        id=transcript_id(chat.id, m.id), text=text, metadata=meta, received_at=now
                    )
                )
        if moved:
            self._save(state)
        return out

    async def ack(self, ids: list[str]) -> None:
        state = self._load()
        changed = False
        for tid in ids:
            parsed = _parse_id(tid)
            if parsed is None:
                log.warning("ack of foreign id %s ignored", tid)
                continue
            chat_id, msg_id = parsed
            if msg_id > state.get(str(chat_id), 0):
                state[str(chat_id)] = msg_id
                changed = True
        if changed:
            self._save(state)


def _title(f: Any) -> str:
    # с layer 185 title - TextWithEntities, раньше str
    return str(getattr(f.title, "text", f.title))


class TelethonGateway:
    """Список чатов папок кешируется на refresh_s: папки меняются редко, а
    GetDialogFilters + iter_dialogs на каждом тике упираются во flood wait."""

    def __init__(self, client: Any, *, refresh_s: float = 600.0) -> None:
        self._client = client
        self._refresh_s = refresh_s
        self._cache: tuple[float, tuple[str, ...], list[TgChat]] | None = None
        self._entities: dict[int, Any] = {}

    @classmethod
    async def connect(cls, api_id: int, api_hash: str, session: str) -> "TelethonGateway":
        from telethon import TelegramClient
        from telethon.sessions import StringSession

        client = TelegramClient(StringSession(session), api_id, api_hash)
        await client.connect()
        if not await client.is_user_authorized():
            await client.disconnect()
            raise RuntimeError("telegram session is not authorized: regenerate TG_SESSION")
        return cls(client)

    async def close(self) -> None:
        await self._client.disconnect()

    async def _filters(self) -> list[Any]:
        from telethon.tl.functions.messages import GetDialogFiltersRequest
        from telethon.tl.types import DialogFilter, DialogFilterChatlist

        res = await self._client(GetDialogFiltersRequest())
        filters = getattr(
            res, "filters", res
        )  # messages.DialogFilters в новых layer, список в старых
        return [f for f in filters if isinstance(f, DialogFilter | DialogFilterChatlist)]

    async def folder_titles(self) -> list[str]:
        return [_title(f) for f in await self._filters()]

    async def _entity(self, pid: int, peer: Any) -> Any:
        if pid not in self._entities:
            self._entities[pid] = await self._client.get_entity(peer)
        return self._entities[pid]

    async def folder_chats(self, folders: Sequence[str]) -> list[TgChat]:
        key = tuple(folders)
        if (
            self._cache
            and self._cache[1] == key
            and time.monotonic() - self._cache[0] < self._refresh_s
        ):
            return self._cache[2]
        chats = await self._resolve(folders)
        self._cache = (time.monotonic(), key, chats)
        return chats

    async def _resolve(self, folders: Sequence[str]) -> list[TgChat]:
        from telethon import utils
        from telethon.tl.types import DialogFilter

        by_title = {_title(f): f for f in await self._filters()}
        missing = [n for n in folders if n not in by_title]
        if missing:
            raise ValueError(f"unknown telegram folders {missing}; available: {sorted(by_title)}")

        dialogs: list[Any] | None = None
        chats: list[TgChat] = []
        seen: set[int] = set()
        for name in folders:
            f = by_title[name]
            excluded = {utils.get_peer_id(p) for p in getattr(f, "exclude_peers", None) or []}
            # (marked peer id, input peer, entity или None - тогда get_entity)
            members: list[tuple[int, Any, Any]] = [
                (utils.get_peer_id(p), p, None) for p in [*f.pinned_peers, *f.include_peers]
            ]
            if isinstance(f, DialogFilter) and (f.broadcasts or f.groups):
                if dialogs is None:
                    dialogs = [d async for d in self._client.iter_dialogs()]
                for d in dialogs:
                    if f.exclude_archived and d.archived:
                        continue
                    if (f.broadcasts and d.is_channel and not d.is_group) or (
                        f.groups and d.is_group
                    ):
                        members.append((d.id, d.input_entity, d.entity))
            if isinstance(f, DialogFilter) and (f.contacts or f.non_contacts or f.bots):
                log.warning("folder %r: contacts/non_contacts/bots flags are ignored", name)

            for pid, peer, entity in members:
                if pid in excluded or pid in seen:
                    continue
                seen.add(pid)
                if entity is None:
                    entity = await self._entity(pid, peer)
                chats.append(
                    TgChat(
                        id=utils.resolve_id(pid)[0],
                        title=getattr(entity, "title", None) or utils.get_display_name(entity),
                        username=getattr(entity, "username", None),
                        folder=name,
                        peer=peer,
                    )
                )
        return chats

    async def messages(self, chat: TgChat, min_id: int | None, limit: int) -> list[TgMessage]:
        if min_id is None:
            raw = [m async for m in self._client.iter_messages(chat.peer, limit=limit)]
            raw.reverse()
        else:
            raw = [
                m
                async for m in self._client.iter_messages(
                    chat.peer, min_id=min_id, limit=limit, reverse=True
                )
            ]
        # MessageService не имеет текста - станет коротким и будет пропущен
        return [TgMessage(m.id, getattr(m, "message", None) or "", m.date) for m in raw]
