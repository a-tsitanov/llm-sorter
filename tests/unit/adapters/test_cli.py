from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from llm_sorter.adapters.inbound import cli
from llm_sorter.adapters.outbound.null_starter import NullStarter
from llm_sorter.application.rules import RulesService
from llm_sorter.domain.loader import dump_tree, load_tree
from tests.fakes import FakeMaterial, FakeRules, FakeStarter, RecordingMetrics

TREE_YAML = """
node:
  id: root
  children:
    - id: finance
      title: Финансы
      description: Деньги
"""

runner = CliRunner()


@pytest.fixture
def rules():
    return FakeRules(load_tree(TREE_YAML))


@pytest.fixture
def patched(monkeypatch, rules):
    def install(r):
        svc = RulesService(r, FakeMaterial(), FakeStarter(), RecordingMetrics())

        async def fake_build(cfg, overrides=None):
            built.append(overrides or {})

            async def close():
                pass

            return SimpleNamespace(rules_service=svc, close=close)

        monkeypatch.setattr(cli, "build_app", fake_build)

    built: list[dict] = []
    install(rules)
    install.built = built
    return install


def test_help_lists_rules_group():
    res = runner.invoke(cli.app, ["--help"])
    assert res.exit_code == 0
    assert "rules" in res.output


def _pulled(rules) -> str:
    return dump_tree(rules.versions[1])


def test_publish_uses_version_from_file(patched, rules, tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(_pulled(rules))
    res = runner.invoke(cli.app, ["rules", "publish", str(f)])
    assert res.exit_code == 0, res.output
    assert "published version 2" in res.output


def test_publish_base_option_overrides(patched, rules, tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(_pulled(rules))
    res = runner.invoke(cli.app, ["rules", "publish", str(f), "--base", "7"])
    assert res.exit_code == 1
    assert "конфликт версий" in res.output


def test_publish_without_version_asks_for_pull(patched, tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(TREE_YAML)
    res = runner.invoke(cli.app, ["rules", "publish", str(f)])
    assert res.exit_code == 1
    assert "rules pull" in res.output


def test_publish_without_version_ok_when_nothing_active(patched, tmp_path):
    patched(FakeRules())
    f = tmp_path / "t.yaml"
    f.write_text(TREE_YAML)
    res = runner.invoke(cli.app, ["rules", "publish", str(f)])
    assert res.exit_code == 0, res.output


def test_publish_validation_error(patched, rules, tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("version: 1\nnode: {id: root, children: [{id: other, title: x, description: y}]}")
    res = runner.invoke(cli.app, ["rules", "publish", str(f)])
    assert res.exit_code == 1
    assert "Traceback" not in res.output


def test_publish_bad_yaml_and_bad_version(patched, tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("a: [unclosed")
    assert runner.invoke(cli.app, ["rules", "publish", str(f)]).exit_code == 1
    f.write_text("version: abc\nnode: {id: root}")
    assert runner.invoke(cli.app, ["rules", "publish", str(f)]).exit_code == 1


def test_pull_writes_file(patched, tmp_path):
    out = tmp_path / "o.yaml"
    res = runner.invoke(cli.app, ["rules", "pull", "-o", str(out)])
    assert res.exit_code == 0
    assert load_tree(out.read_text()).version == 1


def test_rollback_ok_and_errors(patched):
    assert runner.invoke(cli.app, ["rules", "rollback", "1"]).exit_code == 0
    res = runner.invoke(cli.app, ["rules", "rollback", "99"])
    assert res.exit_code == 1
    assert "Traceback" not in res.output


def test_rollback_conflict(patched, rules, monkeypatch):
    rules.conflict_once = True
    res = runner.invoke(cli.app, ["rules", "rollback", "1"])
    assert res.exit_code == 1
    assert "конфликт" in res.output


def test_show_prints_paths(patched):
    res = runner.invoke(cli.app, ["rules", "show"])
    assert res.exit_code == 0
    assert "finance" in res.output
    assert "[manual]" in res.output


def test_read_only_commands_do_not_need_temporal(patched, tmp_path):
    assert runner.invoke(cli.app, ["rules", "show"]).exit_code == 0
    assert runner.invoke(cli.app, ["rules", "pull"]).exit_code == 0
    assert all(isinstance(o.get("starter"), NullStarter) for o in patched.built)
    assert len(patched.built) == 2
    assert runner.invoke(cli.app, ["rules", "rollback", "1"]).exit_code in (0, 1)
    assert "starter" not in patched.built[-1]  # пишущие команды запускают workflow
