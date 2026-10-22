import uuid

from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.tree import ArchivedNode, Node, RuleTree


def _new_uid() -> str:
    return uuid.uuid4().hex[:12]


def _with_children(node: Node, parent_uid: str, extra: list[Node]) -> Node:
    if node.uid == parent_uid:
        return node.model_copy(update={"children": [*node.children, *extra]})
    return node.model_copy(
        update={"children": [_with_children(c, parent_uid, extra) for c in node.children]}
    )


def add_children(tree: RuleTree, parent_uid: str, proposals: list[TopicProposal]) -> RuleTree:
    tree.find(parent_uid)  # KeyError, если родителя нет
    new_nodes = [
        Node(
            uid=_new_uid(),
            id=p.id,
            title=p.title,
            description=p.description,
            examples=list(p.examples),
            guard=p.guard,
            origin="discovered",
        )
        for p in proposals
    ]
    return tree.model_copy(
        update={"root": _with_children(tree.root, parent_uid, new_nodes), "version": None}
    )


def merge_archived(old: RuleTree, new: RuleTree) -> RuleTree:
    """Узлы, исчезнувшие из new, переносятся в archived (путь берётся из old)."""
    new_uids = {n.uid for n in new.iter_nodes()}
    known = {a.uid for a in new.archived}
    added: list[ArchivedNode] = []
    for node in old.iter_nodes():
        if node.uid in new_uids or node.uid in known:
            continue
        parent = old.parent_of(node.uid)
        added.append(
            ArchivedNode(
                uid=node.uid,
                path=old.path_of(node.uid),
                parent_uid=parent.uid if parent else None,
            )
        )
    return new.model_copy(update={"archived": [*new.archived, *added]})


def gained_children(old: RuleTree, new: RuleTree) -> list[str]:
    old_children = {n.uid: {c.uid for c in n.children} for n in old.iter_nodes()}
    result: list[str] = []
    for node in new.iter_nodes():
        if node.uid not in old_children:
            continue
        if {c.uid for c in node.children} - old_children[node.uid]:
            result.append(node.uid)
    return result


def live_ancestor_with_other(old: RuleTree, new: RuleTree, uid: str) -> str:
    """Ближайший предок uid (по old), живой в new и имеющий `other`; иначе root."""
    new_uids = {n.uid for n in new.iter_nodes()}
    ancestors: list[str] = []
    cur = old.parent_of(uid)
    while cur is not None:
        ancestors.append(cur.uid)
        cur = old.parent_of(cur.uid)
    for a in ancestors:
        if a in new_uids and new.has_other(a):
            return a
    return new.root.uid
