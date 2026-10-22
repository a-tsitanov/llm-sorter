import asyncio
from pathlib import Path


class LocalFileStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def write(self, rel_path: str, content: str) -> str:
        if Path(rel_path).is_absolute():
            raise ValueError(f"absolute path not allowed: {rel_path}")
        target = (self._root / rel_path).resolve()
        if not target.is_relative_to(self._root):
            raise ValueError(f"path escapes storage root: {rel_path}")
        await asyncio.to_thread(self._write, target, content)
        return str(target.relative_to(self._root))

    @staticmethod
    def _write(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
