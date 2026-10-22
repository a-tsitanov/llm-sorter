import json
from pathlib import Path

from llm_sorter.adapters.inbound.stub_source import LocalDirSource


async def test_stub_source_dedups(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "in/a.txt").write_text("hi")
    src = LocalDirSource(tmp_path / "in", tmp_path / "seen.json")
    assert [t.id for t in await src.fetch_new()] == ["a"]
    assert [t.id for t in await src.fetch_new()] == ["a"]  # без ack файл отдаётся снова
    await src.ack(["a"])
    assert await src.fetch_new() == []


async def test_stub_source_reads_json_and_persists_seen(tmp_path: Path) -> None:
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "b.txt").write_text("текст b", encoding="utf-8")
    (inbox / "c.json").write_text(
        json.dumps({"id": "c-1", "text": "текст c", "metadata": {"src": "x"}}), encoding="utf-8"
    )
    seen = inbox / ".seen.json"  # seen-файл внутри каталога не считается транскриптом
    got = await LocalDirSource(inbox, seen).fetch_new()
    assert {(t.id, t.text) for t in got} == {("b", "текст b"), ("c-1", "текст c")}
    assert next(t for t in got if t.id == "c-1").metadata == {"src": "x"}
    assert not seen.exists()  # fetch_new ничего не помечает
    await LocalDirSource(inbox, seen).ack([t.id for t in got])
    assert sorted(json.loads(seen.read_text())) == ["b", "c-1"]

    # новый экземпляр читает seen с диска
    (inbox / "d.txt").write_text("d")
    assert [t.id for t in await LocalDirSource(inbox, seen).fetch_new()] == ["d"]


async def test_stub_source_skips_broken_json(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "ok.txt").write_text("ok")
    src = LocalDirSource(tmp_path, tmp_path / "seen.json")
    assert [t.id for t in await src.fetch_new()] == ["ok"]
