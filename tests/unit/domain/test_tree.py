import pytest
from pydantic import ValidationError

from llm_sorter.domain.loader import load_tree

YAML = """
settings: {max_depth: 4}
node:
  id: root
  children:
    - id: a
      max_depth: 2
      children:
        - id: b
          children:
            - id: c
    - id: d
"""


def uid(t, node_id):
    return next(n.uid for n in t.iter_nodes() if n.id == node_id)


def test_navigation():
    t = load_tree(YAML)
    assert [n.id for n in t.iter_nodes()] == ["root", "a", "b", "c", "d"]
    assert t.find(uid(t, "b")).id == "b"
    assert t.parent_of(uid(t, "b")).id == "a"
    assert t.parent_of(t.root.uid) is None
    assert t.path_of(t.root.uid) == "" and t.path_of(uid(t, "c")) == "a/b/c"
    assert t.depth_of(t.root.uid) == 0 and t.depth_of(uid(t, "c")) == 3


def test_find_missing_raises():
    with pytest.raises(KeyError):
        load_tree("node: {id: root}").find("nope")


def test_max_depth_for_nearest_ancestor():
    t = load_tree(YAML)
    assert t.max_depth_for(t.root.uid) == 4
    assert t.max_depth_for(uid(t, "a")) == 2
    assert t.max_depth_for(uid(t, "c")) == 2
    assert t.max_depth_for(uid(t, "d")) == 4


def test_frozen():
    t = load_tree(YAML)
    with pytest.raises(ValidationError):
        t.root.id = "x"  # type: ignore[misc]
