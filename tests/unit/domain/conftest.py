import pytest

from llm_sorter.domain.tree import Tree
from tests.helpers import tree3


@pytest.fixture
def tree() -> Tree:
    return tree3()
