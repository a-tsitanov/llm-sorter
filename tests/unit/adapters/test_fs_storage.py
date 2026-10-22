import pytest

from llm_sorter.adapters.outbound.fs_storage import LocalFileStorage


async def test_fs_rejects_traversal(tmp_path):
    s = LocalFileStorage(tmp_path)
    with pytest.raises(ValueError):
        await s.write("../evil.txt", "x")
    with pytest.raises(ValueError):
        await s.write("/etc/evil.txt", "x")
    assert not (tmp_path.parent / "evil.txt").exists()


async def test_fs_writes_nested(tmp_path):
    await LocalFileStorage(tmp_path).write("a/b/c.txt", "привет")
    assert (tmp_path / "a/b/c.txt").read_text("utf-8") == "привет"
