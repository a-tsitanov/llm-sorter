import re

from llm_sorter.domain.tree import OTHER_ID, Node, RuleTree

ID_RE = re.compile(r"^[a-z0-9_]+$")


class TreeValidationError(ValueError):
    """Дерево правил нарушает инварианты; все найденные ошибки в `errors`."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def validate_tree(tree: RuleTree) -> None:
    errors: list[str] = []
    seen_uids: set[str] = set()

    def check_uid(uid: str) -> None:
        if uid in seen_uids:
            errors.append(f"duplicate uid: {uid}")
        seen_uids.add(uid)

    def walk(node: Node, ancestors: list[Node]) -> None:
        chain = [*ancestors, node]
        depth = len(chain) - 1
        check_uid(node.uid)
        if ancestors:
            if not ID_RE.match(node.id):
                errors.append(f"invalid id {node.id!r}: ожидается ^[a-z0-9_]+$")
            if node.id == OTHER_ID:
                errors.append(f"id {OTHER_ID!r} зарезервирован (uid {node.uid})")
            if not node.title.strip():
                errors.append(f"empty title at {node.id!r}")
            if not node.description.strip():
                errors.append(f"empty description at {node.id!r}")
        limit = next(
            (n.max_depth for n in reversed(chain) if n.max_depth is not None),
            tree.settings.max_depth,
        )
        if depth > limit:
            errors.append(f"depth {depth} exceeds max_depth {limit} at {node.id!r}")
        sibling_ids: set[str] = set()
        for child in node.children:
            if child.id in sibling_ids:
                errors.append(f"duplicate id {child.id!r} under {node.id!r}")
            sibling_ids.add(child.id)
            walk(child, chain)

    walk(tree.root, [])
    for archived in tree.archived:
        check_uid(archived.uid)
    if errors:
        raise TreeValidationError(errors)
