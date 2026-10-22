import pytest

from llm_sorter.domain.actions import RocketAction, SaveAction
from llm_sorter.domain.loader import TreeParseError, dump_tree, load_tree

YAML = """
node:
  id: root
  children:
    - id: finance
      title: Финансы
      description: Деньги
      children:
        - id: crypto
          title: Крипта
          description: Крипта
          actions: [{type: save}, {type: rocket, channel: "#c"}]
"""


def _by_id(tree, node_id):
    return next(n for n in tree.iter_nodes() if n.id == node_id)


def test_load_assigns_uids_and_paths():
    t = load_tree(YAML)
    crypto = _by_id(t, "crypto")
    assert len(crypto.uid) == 12
    assert t.path_of(crypto.uid) == "finance/crypto"
    assert t.depth_of(crypto.uid) == 2
    assert t.settings.max_depth == 3


def test_other_only_where_children():
    t = load_tree(YAML)
    fin = _by_id(t, "finance")
    crypto = _by_id(t, "crypto")
    assert t.has_other(t.root.uid) and t.has_other(fin.uid)
    assert not t.has_other(crypto.uid)
    assert t.other_path(fin.uid) == "finance/other"
    assert t.other_path(t.root.uid) == "other"


def test_roundtrip_keeps_uids():
    t = load_tree(YAML)
    t2 = load_tree(dump_tree(t))
    assert [n.uid for n in t.iter_nodes()] == [n.uid for n in t2.iter_nodes()]
    assert t2 == t


def test_roundtrip_keeps_archived_and_settings():
    text = (
        "version: 4\n"
        "settings: {max_depth: 2, discover: {max_children: 5}}\n"
        "defaults: {guard_failed_action: {type: save, path: bad}}\n"
        "node: {id: root, max_depth: 1}\n"
        "archived: [{uid: abc, path: old/x, parent_uid: null}]\n"
    )
    t = load_tree(text)
    assert t.version == 4 and t.settings.discover.max_children == 5
    assert t.defaults.guard_failed_action == SaveAction(type="save", path="bad")
    assert load_tree(dump_tree(t)) == t


def test_dump_omits_defaults_but_keeps_ids():
    out = dump_tree(load_tree(YAML))
    assert "uid:" in out and "id: root" in out
    assert "examples" not in out and "origin" not in out and "buffer_threshold" not in out
    assert "Финансы" in out


def test_actions_parsed():
    crypto = _by_id(load_tree(YAML), "crypto")
    assert isinstance(crypto.actions[1], RocketAction) and crypto.actions[1].channel == "#c"


def test_minimal_root_only():
    t = load_tree("node: {id: root}")
    assert t.root.children == [] and t.has_other(t.root.uid)


def test_root_without_id():
    assert load_tree("node: {}").root.id == "root"


def test_bad_yaml_raises():
    with pytest.raises(TreeParseError):
        load_tree("node: [")


def test_invalid_structure_raises():
    with pytest.raises(TreeParseError):
        load_tree("node: {id: root, actions: [{type: nope}]}")
    with pytest.raises(TreeParseError):
        load_tree("- just a list")
    with pytest.raises(TreeParseError):
        load_tree("settings: {}")


def test_unknown_key_on_node_raises():
    with pytest.raises(TreeParseError):
        load_tree("node:\n  id: root\n  children:\n    - id: a\n      descripton: опечатка\n")
