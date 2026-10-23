import pytest

from llm_sorter.domain.tree import Tree
from llm_sorter.domain.validation import TreeValidationError, guard_is_meaningful, validate_tree
from tests.helpers import make_tree, topic


def test_valid_tree_passes(tree: Tree) -> None:
    validate_tree(tree)


@pytest.mark.parametrize("bad", ["Finance", "a-b", "", "ф"])
def test_id_pattern(bad: str) -> None:
    with pytest.raises(TreeValidationError, match="invalid id"):
        validate_tree(make_tree(topic(bad)))


@pytest.mark.parametrize("reserved", ["other", "root"])
def test_reserved_ids(reserved: str) -> None:
    # topic("root") подменяет корень: у него появляется родитель
    with pytest.raises(TreeValidationError, match="reserved"):
        validate_tree(make_tree(topic(reserved)))


def test_depth_exceeded() -> None:
    t = make_tree(topic("a"), topic("b", "a"), max_depth=1)
    with pytest.raises(TreeValidationError, match="exceeds max_depth"):
        validate_tree(t)


def test_node_max_depth_override() -> None:
    validate_tree(make_tree(topic("a", max_depth=2), topic("b", "a"), max_depth=1))


@pytest.mark.parametrize("field", ["title", "description"])
def test_empty_text_fields(field: str) -> None:
    with pytest.raises(TreeValidationError, match=f"empty {field}"):
        validate_tree(make_tree(topic("a", **{field: " "})))


def test_collects_all_errors() -> None:
    with pytest.raises(TreeValidationError) as e:
        validate_tree(make_tree(topic("A"), topic("b", title="")))
    assert len(e.value.errors) == 2


@pytest.mark.parametrize(
    "q,ok",
    [
        (None, False),
        ("", False),
        ("yes", False),
        ("Да", False),
        ("Да, если в разговоре упоминаются технологии, тренды, обновления", False),
        ("Пост о банковских картах или их тарифах?", True),
        ("Is this about mobile phones?", True),
        ("Нет?", False),
    ],
)
def test_guard_is_meaningful(q: str | None, ok: bool) -> None:
    assert guard_is_meaningful(q) is ok
