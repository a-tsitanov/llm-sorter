from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_sorter.adapters.inbound.cli import app

SEED = """
topics:
  - {id: finance, parent: root, title: Финансы, description: Деньги}
  - {id: cards, parent: finance, title: Карты, description: Карты}
"""

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "s.db"))
    seed = tmp_path / "seed.yml"
    seed.write_text(SEED, encoding="utf-8")
    return seed


def test_help_lists_commands() -> None:
    out = runner.invoke(app, ["--help"]).output
    assert "topics" in out and "tg-folders" in out


def test_import_show_log(env: Path) -> None:
    assert runner.invoke(app, ["topics", "import", str(env)]).exit_code == 0
    shown = runner.invoke(app, ["topics", "show"])
    assert shown.exit_code == 0 and "finance.cards" in shown.output
    log = runner.invoke(app, ["topics", "log"]).output
    assert "cli" in log and "add" in log and "cards" in log


def test_move_and_export(env: Path, tmp_path: Path) -> None:
    runner.invoke(app, ["topics", "import", str(env)])
    assert (
        runner.invoke(
            app, ["topics", "add", "tech", "--title", "IT", "--description", "d"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["topics", "move", "cards", "tech"]).exit_code == 0
    out = tmp_path / "out.yml"
    assert runner.invoke(app, ["topics", "export", "-o", str(out)]).exit_code == 0
    assert "parent: tech" in out.read_text(encoding="utf-8")


def test_validation_error_exit_code(env: Path) -> None:
    r = runner.invoke(app, ["topics", "add", "Bad-Id", "--title", "x", "--description", "y"])
    assert r.exit_code == 1 and "invalid id" in r.output


def test_import_twice_needs_replace(env: Path) -> None:
    runner.invoke(app, ["topics", "import", str(env)])
    assert runner.invoke(app, ["topics", "import", str(env)]).exit_code == 1
    assert runner.invoke(app, ["topics", "import", str(env), "--replace"]).exit_code == 0
