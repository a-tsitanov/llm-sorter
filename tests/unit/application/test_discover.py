from datetime import UTC, datetime

import pytest

from llm_sorter.application.discover import DiscoverResult, DiscoverService
from llm_sorter.application.rules import RulesService
from llm_sorter.domain.classification import Classification, Status, Transcript
from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.loader import load_tree
from llm_sorter.domain.tree import RuleTree
from tests.fakes import (
    FakeClassifications,
    FakeEmbedder,
    FakeEmbeddings,
    FakeMaterial,
    FakeRules,
    FakeStarter,
    FakeText,
    FakeTranscripts,
    RecordingMetrics,
)
from tests.helpers import remove_node, uid, with_settings

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

DIM = 8


def unit(axis: int, eps: float = 0.0) -> list[float]:
    v = [0.0] * DIM
    v[axis] = 1.0
    v[(axis + 1) % DIM] = eps
    return v


# эмбеддинги тем: ортогональны, чтобы дедуп был детерминирован
TOPIC_VECTORS = {
    "Финансы. Деньги": unit(0),
    "Крипта. Криптовалюты": unit(1),
    "Банки. Банковские услуги": unit(2),
    "Налоги. Налоговые вопросы": unit(7),
}


def proposal(
    id_: str, title: str = "Налоги", description: str = "Налоговые вопросы"
) -> TopicProposal:
    return TopicProposal(id=id_, title=title, description=description, examples=[], guard="q?")


@pytest.fixture
def tree() -> RuleTree:
    return with_settings(load_tree(TREE_YAML), buffer_threshold=10, min_cluster_size=10)


@pytest.fixture
def rules(tree):
    return FakeRules(tree)


@pytest.fixture
def material():
    return FakeMaterial()


@pytest.fixture
def transcripts():
    return FakeTranscripts()


@pytest.fixture
def classifications():
    return FakeClassifications()


@pytest.fixture
def cache():
    return FakeEmbeddings()


@pytest.fixture
def embedder():
    return FakeEmbedder(dict(TOPIC_VECTORS), dim=DIM)


@pytest.fixture
def starter():
    return FakeStarter()


@pytest.fixture
def metrics():
    return RecordingMetrics()


@pytest.fixture
def text_llm():
    t = FakeText()
    t.next_proposal = proposal("taxes")
    return t


@pytest.fixture
def svc(rules, material, transcripts, cache, embedder, text_llm, starter, metrics):
    rules_service = RulesService(rules, material, starter, metrics)
    return DiscoverService(
        rules, rules_service, material, transcripts, cache, embedder, text_llm, metrics
    )


async def seed(
    material: FakeMaterial,
    transcripts: FakeTranscripts,
    embedder: FakeEmbedder,
    node_uid: str,
    prefix: str,
    axis: int,
    kind: str = "other",
    n: int = 12,
) -> list[str]:
    """n файлов с почти одинаковыми эмбеддингами вдоль оси axis."""
    ids = []
    for i in range(n):
        fid = f"{prefix}{i}"
        text = f"{prefix} текст {i}"
        embedder.mapping[text] = unit(axis, eps=0.001 * i)
        await transcripts.put(
            Transcript(id=fid, text=text, received_at=datetime(2026, 1, 1, tzinfo=UTC))
        )
        await material.add(node_uid, fid, kind)  # type: ignore[arg-type]
        ids.append(fid)
    return ids


async def test_discover_creates_topic_and_reprocesses(
    svc, rules, tree, material, transcripts, embedder, starter, metrics
):
    root_uid = tree.root.uid
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)
    res = await svc.run(root_uid)
    assert res.created == ["taxes"]
    assert res.version == rules.active_version == 2
    assert rules.sources[2] == "discover"
    assert len(starter.reprocess) == 12
    assert {node for _, node, _ in starter.reprocess} == {root_uid}
    assert metrics.counts["discover_run"] == 1
    assert metrics.buffer_sizes == [("other", 12)]
    new = (await rules.active()).root.children[-1]
    assert (new.id, new.origin) == ("taxes", "discovered")


async def test_discover_leaf_material_not_reprocessed(
    svc, rules, tree, material, transcripts, embedder, starter, classifications, text_llm
):
    crypto = uid(tree, "crypto")
    leaf_files = await seed(material, transcripts, embedder, crypto, "defi", axis=4, kind="leaf")
    for f in leaf_files:
        await classifications.save(
            Classification(
                file_id=f,
                tree_version=1,
                node_uid=crypto,
                path="finance/crypto",
                status=Status.CLASSIFIED,
            )
        )
    text_llm.next_proposal = proposal(
        "defi", title="DeFi", description="Децентрализованные финансы"
    )
    res = await svc.run(crypto)
    assert res.created == ["defi"] and starter.reprocess == []
    assert [c.id for c in (await rules.active()).find(crypto).children] == ["defi"]
    assert [(await classifications.get(f)).path for f in leaf_files] == ["finance/crypto"] * 12


async def test_dedup_against_sibling(svc, rules, tree, material, transcripts, embedder, text_llm):
    finance = uid(tree, "finance")
    await seed(material, transcripts, embedder, finance, "bank", axis=3)
    text_llm.next_proposal = proposal("banks_2", title="Банки", description="Банковские услуги")
    res = await svc.run(finance)
    assert res.created == [] and res.rejected == 1
    assert res.version is None and rules.active_version == 1


async def test_ask_zone_uses_llm(svc, tree, material, transcripts, embedder, text_llm):
    finance = uid(tree, "finance")
    await seed(material, transcripts, embedder, finance, "bank", axis=3)
    # cos(unit(2), [0.6, 0, 0.8, ...]) = 0.8: между llm_dedup_floor и dedup_similarity
    embedder.mapping["Вклады. Депозиты"] = [0.6, 0, 0.8, 0, 0, 0, 0, 0]
    text_llm.next_proposal = proposal("deposits", title="Вклады", description="Депозиты")
    text_llm.same = True
    res = await svc.run(finance)
    assert res.created == [] and res.rejected == 1
    assert text_llm.calls["same_topic"] == 1


async def test_ask_zone_new_when_llm_says_different(
    svc, tree, material, transcripts, embedder, text_llm
):
    finance = uid(tree, "finance")
    await seed(material, transcripts, embedder, finance, "bank", axis=3)
    embedder.mapping["Вклады. Депозиты"] = [0.6, 0, 0.8, 0, 0, 0, 0, 0]
    text_llm.next_proposal = proposal("deposits", title="Вклады", description="Депозиты")
    res = await svc.run(finance)
    assert res.created == ["deposits"] and res.rejected == 0


async def test_id_conflict_gets_suffix(svc, rules, tree, material, transcripts, embedder, text_llm):
    finance = uid(tree, "finance")
    await seed(material, transcripts, embedder, finance, "bank", axis=3)
    text_llm.next_proposal = proposal("banks")  # id занят, но тема другая
    res = await svc.run(finance)
    assert res.created == ["banks_2"]
    assert [c.id for c in (await rules.active()).find(finance).children] == [
        "crypto",
        "banks",
        "banks_2",
    ]


class SequenceText(FakeText):
    def __init__(self, proposals: list[TopicProposal]) -> None:
        super().__init__()
        self._queue = list(proposals)

    async def propose_topic(self, samples, parent_path, siblings, language):  # type: ignore[no-untyped-def]
        self.calls["propose_topic"] += 1
        self.samples = samples
        self.siblings = siblings
        return self._queue.pop(0)


async def test_two_clusters_conflict_with_accepted(
    rules, tree, material, transcripts, cache, embedder, starter, metrics
):
    root_uid = tree.root.uid
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)
    await seed(material, transcripts, embedder, root_uid, "law", axis=6, n=11)
    embedder.mapping["Право. Юридические вопросы"] = unit(3)
    text = SequenceText(
        [proposal("taxes"), proposal("taxes", title="Право", description="Юридические вопросы")]
    )
    svc = DiscoverService(
        rules,
        RulesService(rules, material, starter, metrics),
        material,
        transcripts,
        cache,
        embedder,
        text,
        metrics,
    )
    res = await svc.run(root_uid)
    assert res.created == ["taxes", "taxes_2"]
    assert len(starter.reprocess) == 23
    assert len(text.samples) == 5  # samples_per_cluster
    assert [c.id for c in text.siblings] == ["finance"]  # без other


async def test_slots_limit_clusters(
    rules, tree, material, transcripts, cache, embedder, starter, metrics
):
    rules.versions[1] = with_settings(tree, max_children=2).model_copy(update={"version": 1})
    root_uid = tree.root.uid
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)
    await seed(material, transcripts, embedder, root_uid, "law", axis=6, n=11)
    text = SequenceText([proposal("taxes"), proposal("law")])
    svc = DiscoverService(
        rules,
        RulesService(rules, material, starter, metrics),
        material,
        transcripts,
        cache,
        embedder,
        text,
        metrics,
    )
    res = await svc.run(root_uid)
    assert res.created == ["taxes"] and text.calls["propose_topic"] == 1


async def test_samples_truncated(
    rules, tree, material, transcripts, cache, embedder, starter, metrics
):
    rules.versions[1] = with_settings(tree, sample_chars=4).model_copy(update={"version": 1})
    await seed(material, transcripts, embedder, tree.root.uid, "tax", axis=5)
    text = SequenceText([proposal("taxes")])
    svc = DiscoverService(
        rules,
        RulesService(rules, material, starter, metrics),
        material,
        transcripts,
        cache,
        embedder,
        text,
        metrics,
    )
    await svc.run(tree.root.uid)
    assert all(len(s) <= 4 for s in text.samples)


async def test_respects_max_children(svc, rules, tree, material, transcripts, embedder, text_llm):
    rules.versions[1] = with_settings(tree, max_children=2).model_copy(update={"version": 1})
    finance = uid(tree, "finance")
    await seed(material, transcripts, embedder, finance, "bank", axis=3)
    assert (await svc.run(finance)).version is None
    assert text_llm.calls["propose_topic"] == 0
    assert await material.count_new(finance) == 12  # не дошли до discover — счётчик цел


async def test_below_threshold_noop(svc, tree, material, transcripts, embedder, text_llm):
    await seed(material, transcripts, embedder, tree.root.uid, "tax", axis=5, n=9)
    assert await svc.run(tree.root.uid) == DiscoverResult(None, [], 0)
    assert text_llm.calls["propose_topic"] == 0


async def test_archived_node_noop(svc, rules, tree, material, transcripts, embedder, text_llm):
    banks = uid(tree, "banks")
    await seed(material, transcripts, embedder, banks, "bank", axis=3, kind="leaf")
    await rules.publish(remove_node(tree, "banks"), base_version=1, source="manual")
    assert await svc.run(banks) == DiscoverResult(None, [], 0)
    assert text_llm.calls["propose_topic"] == 0


async def test_counter_resets(svc, tree, material, transcripts, embedder):
    root_uid = tree.root.uid
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)
    await svc.run(root_uid)
    assert await material.count_new(root_uid) == 0


async def test_counter_resets_without_accepted(
    svc, tree, material, transcripts, embedder, text_llm
):
    finance = uid(tree, "finance")
    await seed(material, transcripts, embedder, finance, "bank", axis=3)
    text_llm.next_proposal = proposal("banks_2", title="Банки", description="Банковские услуги")
    await svc.run(finance)
    assert await material.count_new(finance) == 0


async def test_embedding_cache_used(svc, tree, material, transcripts, embedder, cache):
    root_uid = tree.root.uid
    ids = await seed(material, transcripts, embedder, root_uid, "tax", axis=5, n=20)
    cache.vectors[ids[0]] = unit(5)
    await svc.run(root_uid)
    assert set(cache.vectors) == set(ids)
    # 19 недостающих → 2 батча по 16; плюс эмбеддинги тем
    assert embedder.calls == 2 + 2


async def test_due_nodes_filters_ineligible(svc, rules, tree, material, transcripts, embedder):
    root_uid = tree.root.uid
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)
    banks = uid(tree, "banks")
    await seed(material, transcripts, embedder, banks, "bank", axis=3, kind="leaf")
    await rules.publish(remove_node(tree, "banks"), base_version=1, source="manual")
    assert await svc.due_nodes() == [root_uid]


async def test_empty_transcripts_ignored(svc, tree, material, transcripts, embedder, text_llm):
    root_uid = tree.root.uid
    for i in range(12):
        await transcripts.put(
            Transcript(id=f"e{i}", text=" \n\t ", received_at=datetime(2026, 1, 1, tzinfo=UTC))
        )
        await material.add(root_uid, f"e{i}", "other")
    res = await svc.run(root_uid)
    assert res == DiscoverResult(None, [], 0)
    assert embedder.calls == 0 and text_llm.calls["propose_topic"] == 0
    assert await material.count_new(root_uid) == 0


async def test_invalid_proposal_skipped_others_created(
    rules, tree, material, transcripts, cache, embedder, starter, metrics
):
    root_uid = tree.root.uid
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)
    await seed(material, transcripts, embedder, root_uid, "law", axis=6, n=11)
    embedder.mapping["Право. Юридические вопросы"] = unit(3)
    text = SequenceText(
        [
            proposal("taxes", title="  "),
            proposal("law", title="Право", description="Юридические вопросы"),
        ]
    )
    svc = DiscoverService(
        rules,
        RulesService(rules, material, starter, metrics),
        material,
        transcripts,
        cache,
        embedder,
        text,
        metrics,
    )
    res = await svc.run(root_uid)
    assert res.created == ["law"]
    assert metrics.counts["model_error:discover"] == 1
    assert await material.count_new(root_uid) == 0


@pytest.mark.parametrize(
    "bad",
    [
        proposal("taxes", description=""),
        proposal("Налоги"),
        proposal("other"),
    ],
)
async def test_invalid_proposal_variants(
    svc, tree, material, transcripts, embedder, text_llm, metrics, bad
):
    await seed(material, transcripts, embedder, tree.root.uid, "tax", axis=5)
    text_llm.next_proposal = bad
    res = await svc.run(tree.root.uid)
    assert res.created == [] and metrics.counts["model_error:discover"] == 1


class FailingRulesService:
    async def add_discovered(self, parent_uid, proposals):  # type: ignore[no-untyped-def]
        from llm_sorter.domain.validation import TreeValidationError

        raise TreeValidationError(["boom"])


async def test_tree_validation_error_marks_and_returns_empty(
    rules, tree, material, transcripts, cache, embedder, text_llm, metrics
):
    svc = DiscoverService(
        rules,
        FailingRulesService(),
        material,
        transcripts,
        cache,
        embedder,
        text_llm,
        metrics,  # type: ignore[arg-type]
    )
    await seed(material, transcripts, embedder, tree.root.uid, "tax", axis=5)
    res = await svc.run(tree.root.uid)
    assert res == DiscoverResult(None, [], 0)
    assert metrics.counts["model_error:discover"] == 1
    assert await material.count_new(tree.root.uid) == 0


class AddingText(FakeText):
    """Во время propose_topic в узел приходит новый файл."""

    def __init__(self, material: FakeMaterial, node_uid: str) -> None:
        super().__init__()
        self.material = material
        self.node_uid = node_uid
        self.next_proposal = proposal("taxes")

    async def propose_topic(self, samples, parent_path, siblings, language):  # type: ignore[no-untyped-def]
        await self.material.add(self.node_uid, "late", "other")
        return await super().propose_topic(samples, parent_path, siblings, language)


async def test_material_added_during_run_stays_new(
    rules, tree, material, transcripts, cache, embedder, starter, metrics
):
    root_uid = tree.root.uid
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)
    svc = DiscoverService(
        rules,
        RulesService(rules, material, starter, metrics),
        material,
        transcripts,
        cache,
        embedder,
        AddingText(material, root_uid),
        metrics,
    )
    await svc.run(root_uid)
    assert await material.count_new(root_uid) == 1


async def test_progress_called(svc, tree, material, transcripts, embedder):
    await seed(material, transcripts, embedder, tree.root.uid, "tax", axis=5, n=20)
    ticks: list[int] = []
    await svc.run(tree.root.uid, progress=lambda: ticks.append(1))
    # 2 батча файлов + 1 кластер + эмбеддинг соседей + эмбеддинг предложения
    assert len(ticks) == 5


async def test_due_node_at_max_children_reports_blocked(
    svc, rules, tree, material, transcripts, embedder, metrics
):
    await rules.publish(with_settings(tree, max_children=2), base_version=1, source="manual")
    root_uid, fin = tree.root.uid, uid(tree, "finance")
    await seed(material, transcripts, embedder, fin, "fin", axis=4)  # у finance 2 из 2 детей
    await seed(material, transcripts, embedder, root_uid, "tax", axis=5)  # у root 1 из 2
    assert await svc.due_nodes() == [root_uid]
    assert metrics.discover_blocked_by_path == {"finance": True, "root": False}
