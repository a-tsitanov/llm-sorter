from llm_sorter.domain.tree import Node, RuleTree


def uid(tree: RuleTree, id_: str) -> str:
    return next(n.uid for n in tree.iter_nodes() if n.id == id_)


def _without(node: Node, id_: str) -> Node:
    return node.model_copy(
        update={"children": [_without(c, id_) for c in node.children if c.id != id_]}
    )


def remove_node(tree: RuleTree, id_: str) -> RuleTree:
    return tree.model_copy(update={"root": _without(tree.root, id_)})


def with_settings(tree: RuleTree, **kw: int) -> RuleTree:
    """kw: max_depth -> Settings, остальное -> DiscoverSettings."""
    top = {k: v for k, v in kw.items() if k == "max_depth"}
    disc = {k: v for k, v in kw.items() if k != "max_depth"}
    s = tree.settings
    if disc:
        s = s.model_copy(update={"discover": s.discover.model_copy(update=disc)})
    s = s.model_copy(update=top)
    return tree.model_copy(update={"settings": s})
