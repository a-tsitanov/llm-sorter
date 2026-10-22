import pytest

from llm_sorter.domain.loader import load_tree
from llm_sorter.domain.tree import RuleTree

TREE_YAML = """
node:
  id: root
  children:
    - id: finance
      title: Финансы
      description: Деньги
      children:
        - id: crypto
          title: Крипта
          description: Криптовалюты
        - id: banks
          title: Банки
          description: Банковские услуги
"""


@pytest.fixture
def tree() -> RuleTree:
    return load_tree(TREE_YAML)
