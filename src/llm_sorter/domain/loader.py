import uuid
from typing import Any

import yaml
from pydantic import ValidationError

from llm_sorter.domain.tree import Node, RuleTree


class TreeParseError(ValueError):
    """Некорректный YAML дерева правил."""


def _new_uid() -> str:
    return uuid.uuid4().hex[:12]


def _fill_node(raw: Any, *, is_root: bool) -> Any:
    if not isinstance(raw, dict):
        return raw
    node = dict(raw)
    node.setdefault("uid", _new_uid())
    if is_root:
        node.setdefault("id", "root")
    children = node.get("children")
    if isinstance(children, list):
        node["children"] = [_fill_node(c, is_root=False) for c in children]
    return node


def load_tree(text: str) -> RuleTree:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise TreeParseError(f"невалидный YAML: {e}") from e
    if not isinstance(data, dict) or "node" not in data:
        raise TreeParseError("ожидается mapping с ключом 'node'")
    payload = {k: v for k, v in data.items() if k != "node"}
    payload["root"] = _fill_node(data["node"] or {}, is_root=True)
    try:
        return RuleTree.model_validate(payload)
    except ValidationError as e:
        raise TreeParseError(str(e)) from e


def _dump_node(node: Node) -> dict[str, Any]:
    out: dict[str, Any] = {"uid": node.uid, "id": node.id}
    out.update(node.model_dump(exclude_defaults=True, exclude={"uid", "id", "children"}))
    if node.children:
        out["children"] = [_dump_node(c) for c in node.children]
    return out


def dump_tree(tree: RuleTree) -> str:
    data: dict[str, Any] = {}
    if tree.version is not None:
        data["version"] = tree.version
    rest = tree.model_dump(exclude_defaults=True, exclude={"version", "root", "archived"})
    data.update(rest)
    data["node"] = _dump_node(tree.root)
    if tree.archived:
        data["archived"] = [a.model_dump() for a in tree.archived]
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
