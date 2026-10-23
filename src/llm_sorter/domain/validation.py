import re

from llm_sorter.domain.tree import OTHER_ID, ROOT_ID, Tree

ID_RE = re.compile(r"^[a-z0-9_]+$")
RESERVED_IDS = {OTHER_ID, ROOT_ID}
# ответ вместо вопроса - так discover заполнял guard в тесте 2026-10-22
_ANSWER_WORDS = {"yes", "no", "да", "нет"}
_MIN_GUARD_LEN = 15


class TreeValidationError(ValueError):
    """Дерево нарушает инварианты; все найденные ошибки в `errors`."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def guard_is_meaningful(q: str | None) -> bool:
    """Гейт - вопрос к модели: не короче 15 символов, с `?`, не начинается с да/нет."""
    if q is None:
        return False
    q = q.strip()
    if len(q) < _MIN_GUARD_LEN or "?" not in q:
        return False
    first = re.split(r"\W+", q.lower(), maxsplit=1)[0]
    return first not in _ANSWER_WORDS


def validate_tree(tree: Tree) -> None:
    errors: list[str] = []
    root = tree.topics.get(ROOT_ID)
    if root is None or root.parent_id is not None:
        errors.append(f"id {ROOT_ID!r} reserved for the root (no parent)")
    for key, t in tree.topics.items():
        if key != t.id:
            errors.append(f"key {key!r} != id {t.id!r}")
        if t.id == ROOT_ID:
            continue
        if t.id in RESERVED_IDS:
            errors.append(f"id {t.id!r} is reserved")
            continue
        if not ID_RE.match(t.id):
            errors.append(f"invalid id {t.id!r}: ожидается ^[a-z0-9_]+$")
        if not t.title.strip():
            errors.append(f"empty title at {t.id!r}")
        if not t.description.strip():
            errors.append(f"empty description at {t.id!r}")
        if t.parent_id is None or t.parent_id not in tree.topics:
            errors.append(f"unknown parent {t.parent_id!r} at {t.id!r}")
    if errors:
        raise TreeValidationError(errors)
    for t in tree.topics.values():
        seen = {t.id}
        cur = t
        while cur.parent_id is not None:
            if cur.parent_id in seen:
                raise TreeValidationError([f"cycle through {t.id!r}"])
            seen.add(cur.parent_id)
            cur = tree.topics[cur.parent_id]
        depth = len(seen) - 1
        if depth > (limit := tree.max_depth_for(t.id)):
            errors.append(f"depth {depth} exceeds max_depth {limit} at {t.id!r}")
    if errors:
        raise TreeValidationError(errors)
