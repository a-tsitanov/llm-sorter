"""Заглушка источника транскриптов: локальный каталог с *.txt и *.json."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from llm_sorter.domain.classification import Transcript

log = logging.getLogger(__name__)


class LocalDirSource:
    """*.txt: id = имя без расширения; *.json: {"id","text","metadata"}.
    Подтверждённые через ack id хранятся JSON-списком в seen_file."""

    def __init__(self, dir: Path, seen_file: Path) -> None:
        self._dir = dir
        self._seen_file = seen_file

    def _load_seen(self) -> set[str]:
        if not self._seen_file.exists():
            return set()
        return set(json.loads(self._seen_file.read_text(encoding="utf-8")))

    def _read(self, path: Path, now: datetime) -> Transcript | None:
        try:
            if path.suffix == ".txt":
                return Transcript(
                    id=path.stem, text=path.read_text(encoding="utf-8"), received_at=now
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            return Transcript(
                id=data["id"],
                text=data["text"],
                metadata=data.get("metadata") or {},
                received_at=now,
            )
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as e:
            log.warning("skip unreadable transcript %s: %s", path, e)
            return None

    async def fetch_new(self) -> list[Transcript]:
        seen = self._load_seen()
        now = datetime.now(UTC)
        seen_path = self._seen_file.resolve()
        files = sorted(
            p
            for p in self._dir.iterdir()
            if p.is_file() and p.suffix in (".txt", ".json") and p.resolve() != seen_path
        )
        new: list[Transcript] = []
        for path in files:
            t = self._read(path, now)
            if t is not None and t.id not in seen:
                seen.add(t.id)  # защита от дублей id внутри одной выборки
                new.append(t)
        return new

    async def ack(self, ids: list[str]) -> None:
        if not ids:
            return
        seen = self._load_seen() | set(ids)
        self._seen_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._seen_file.with_name(self._seen_file.name + ".tmp")
        tmp.write_text(json.dumps(sorted(seen), ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._seen_file)
