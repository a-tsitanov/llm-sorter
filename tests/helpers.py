from typing import Any

from llm_sorter.domain.tree import ROOT_ID, DiscoverSettings, Settings, Topic, Tree


def topic(id_: str, parent: str = ROOT_ID, **kw: Any) -> Topic:
    kw.setdefault("title", id_.capitalize())
    kw.setdefault("description", f"Про {id_}")
    return Topic(id=id_, parent_id=parent, **kw)


def make_tree(*topics: Topic, rev: int = 0, **settings: Any) -> Tree:
    """settings: max_depth/language/max_reprocess -> Settings, остальное -> DiscoverSettings."""
    top = {k: v for k, v in settings.items() if k in Settings.model_fields and k != "discover"}
    disc = {k: v for k, v in settings.items() if k not in top}
    s = Settings(**top, discover=DiscoverSettings(**disc))
    root = Topic(id=ROOT_ID, parent_id=None)
    return Tree(rev=rev, settings=s, topics={ROOT_ID: root, **{t.id: t for t in topics}})


def tree3(**settings: Any) -> Tree:
    """root -> finance -> cards; root -> tech."""
    return make_tree(
        topic("finance"),
        topic("cards", "finance", guard="Пост про банковские карты?"),
        topic("tech"),
        **settings,
    )
