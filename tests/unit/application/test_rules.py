import pytest

from llm_sorter.application.rules import RulesService
from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.loader import dump_tree, load_tree
from llm_sorter.domain.validation import TreeValidationError
from llm_sorter.ports.storage import MaterialItem, VersionConflict
from tests.fakes import FakeMaterial, FakeRules, FakeStarter, RecordingMetrics
from tests.helpers import remove_node, uid

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
          children:
            - id: trading
              title: Трейдинг
              description: Торговля
        - id: banks
          title: Банки
          description: Банковские услуги
"""


def proposal(id_: str) -> TopicProposal:
    return TopicProposal(id=id_, title=id_, description=f"about {id_}", examples=[], guard="q?")


@pytest.fixture
def tree():
    return load_tree(TREE_YAML)


@pytest.fixture
def rules(tree):
    return FakeRules(tree)


@pytest.fixture
def material():
    return FakeMaterial()


@pytest.fixture
def starter():
    return FakeStarter()


@pytest.fixture
def metrics():
    return RecordingMetrics()


@pytest.fixture
def svc(rules, material, starter, metrics):
    return RulesService(rules, material, starter, metrics)


async def test_publish_invalid_rejected(svc, rules):
    with pytest.raises(TreeValidationError):
        await svc.publish_yaml(
            "node: {id: root, children: [{id: other, title: x, description: y}]}",
            base_version=1,
        )
    assert (await rules.active()).version == 1


async def test_manual_add_child_reprocesses_other_buffer(svc, rules, material, starter, metrics):
    active = await rules.active()
    root_uid = active.root.uid
    await material.add(root_uid, "f1", "other")
    await material.add(root_uid, "f2", "leaf")
    text = dump_tree(active).replace(
        "  children:\n  - uid",
        "  children:\n  - id: news\n    title: Новости\n    description: Новости дня\n  - uid",
        1,
    )
    res = await svc.publish_yaml(text, base_version=1)
    assert res.version == 2
    assert res.reprocess_nodes == [root_uid]
    assert res.reprocess_started == 1
    assert starter.reprocess == [("f1", root_uid, 2)]
    assert metrics.counts["rules_published"] == 1
    assert metrics.counts["active_version"] == 1


async def test_delete_moves_other_buffer_up(svc, rules, material):
    active = await rules.active()
    await material.add(uid(active, "crypto"), "f2", "other")
    res = await svc.publish_tree(remove_node(active, "crypto"), base_version=1, source="manual")
    assert MaterialItem(file_id="f2", kind="other") in await material.list(uid(active, "finance"))
    assert await material.list(uid(active, "crypto")) == []
    assert (await rules.active()).archived[0].path == "finance/crypto"
    assert res.reprocess_nodes == []


async def test_delete_last_child_moves_other_to_root(svc, rules, material):
    active = await rules.active()
    await material.add(uid(active, "trading"), "f3", "other")
    # trading удалён, crypto осталась листом без other -> ближайший предок с other: finance
    await svc.publish_tree(remove_node(active, "trading"), base_version=1, source="manual")
    assert MaterialItem(file_id="f3", kind="other") in await material.list(uid(active, "finance"))


async def test_add_discovered_retries_on_conflict(svc, rules):
    rules.conflict_once = True
    root_uid = (await rules.active()).root.uid
    await svc.add_discovered(root_uid, [proposal("taxes")])
    assert any(n.id == "taxes" for n in (await rules.active()).iter_nodes())
    assert rules.sources[2] == "discover"


async def test_add_discovered_gives_up_after_three_conflicts(svc, rules):
    root_uid = (await rules.active()).root.uid
    orig = rules.publish

    async def always_conflict(*a, **k):
        raise VersionConflict("x")

    rules.publish = always_conflict
    with pytest.raises(VersionConflict):
        await svc.add_discovered(root_uid, [proposal("taxes")])
    rules.publish = orig


async def test_rollback_archives_discovered(svc, rules):
    v1 = (await rules.active()).version
    root_uid = (await rules.active()).root.uid
    await svc.add_discovered(root_uid, [proposal("taxes")])
    await svc.rollback(v1)
    active = await rules.active()
    assert [a.path for a in active.archived] == ["taxes"]
    assert rules.sources[active.version] == "rollback"


async def test_pull_roundtrips(svc, rules):
    assert load_tree(await svc.pull()).version == 1


async def test_bootstrap_if_empty(material, starter, metrics):
    empty = FakeRules()
    svc = RulesService(empty, material, starter, metrics)
    await svc.bootstrap_if_empty()
    assert (await empty.active()).root.id == "root"
    assert empty.active_version == 1
    await svc.bootstrap_if_empty()
    assert empty.active_version == 1


async def test_manual_yaml_with_foreign_root_uid_rejected(svc, rules):
    # YAML без uid (или из другого окружения) получит новый root uid: весь материал осиротеет
    with pytest.raises(TreeValidationError, match="rules pull"):
        await svc.publish_yaml(TREE_YAML, base_version=1)
    assert (await rules.active()).version == 1


async def test_removing_all_children_moves_other_buffer_up(svc, rules, material):
    active = await rules.active()
    crypto, finance = uid(active, "crypto"), uid(active, "finance")
    await material.add(crypto, "f4", "other")
    await material.add(crypto, "f5", "leaf")
    # crypto жива, но потеряла единственного ребёнка -> у неё больше нет other
    await svc.publish_tree(remove_node(active, "trading"), base_version=1, source="manual")
    assert MaterialItem("f4", "other") in await material.list(finance)
    assert await material.list(crypto) == [MaterialItem("f5", "leaf")]
