import pytest

from llm_sorter.domain.loader import load_tree
from llm_sorter.domain.validation import TreeValidationError, validate_tree

VALID = """
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


def _child(id_: str, uid: str | None = None, title: str = "T", desc: str = "D") -> str:
    uid_part = f", uid: {uid}" if uid else ""
    return f"{{id: {id_}, title: '{title}', description: '{desc}'{uid_part}}}"


def _errors(text: str) -> list[str]:
    with pytest.raises(TreeValidationError) as exc:
        validate_tree(load_tree(text))
    return exc.value.errors


def test_valid_tree_passes():
    validate_tree(load_tree(VALID))


def test_id_pattern():
    errs = _errors(f"node: {{id: root, children: [{_child('Bad-Id')}]}}")
    assert len(errs) == 1 and "id" in errs[0]


def test_other_reserved():
    errs = _errors(f"node: {{id: root, children: [{_child('other')}]}}")
    assert any("other" in e for e in errs)


def test_duplicate_sibling_id():
    errs = _errors(f"node: {{id: root, children: [{_child('a')}, {_child('a')}]}}")
    assert any("duplicate id" in e for e in errs)


def test_same_id_in_different_branches_ok():
    nested = f"{{id: p, title: T, description: D, children: [{_child('a')}]}}"
    validate_tree(load_tree(f"node: {{id: root, children: [{nested}, {_child('a')}]}}"))


def test_duplicate_uid():
    errs = _errors(f"node: {{id: root, children: [{_child('a', 'u1')}, {_child('b', 'u1')}]}}")
    assert any("duplicate uid" in e for e in errs)


def test_duplicate_uid_with_archived():
    text = (
        f"node: {{id: root, children: [{_child('a', 'u1')}]}}\n"
        "archived: [{uid: u1, path: x, parent_uid: null}]\n"
    )
    assert any("duplicate uid" in e for e in _errors(text))


def test_depth_exceeded():
    inner = _child("b")
    outer = f"{{id: a, title: T, description: D, children: [{inner}]}}"
    errs = _errors(f"settings: {{max_depth: 1}}\nnode: {{id: root, children: [{outer}]}}")
    assert any("depth" in e for e in errs)


def test_node_max_depth_override():
    inner = _child("b")
    outer = f"{{id: a, title: T, description: D, max_depth: 2, children: [{inner}]}}"
    validate_tree(load_tree(f"settings: {{max_depth: 1}}\nnode: {{id: root, children: [{outer}]}}"))


def test_empty_title():
    errs = _errors(f"node: {{id: root, children: [{_child('a', title='')}]}}")
    assert any("title" in e for e in errs)


def test_empty_description():
    errs = _errors(f"node: {{id: root, children: [{_child('a', desc='')}]}}")
    assert any("description" in e for e in errs)


def test_root_needs_no_title():
    validate_tree(load_tree("node: {id: root}"))


def test_collects_all_errors():
    errs = _errors(f"node: {{id: root, children: [{_child('other')}, {_child('a', title='')}]}}")
    assert len(errs) == 2
