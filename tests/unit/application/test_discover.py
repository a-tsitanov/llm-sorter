import pytest

from llm_sorter.application.discover import DiscoverService
from llm_sorter.domain.discover import TopicProposal
from llm_sorter.domain.tree import ROOT_ID, Tree
from llm_sorter.ports.models import InvalidModelOutput
from tests.fakes import FakeEmbedder, FakeText, RecordingMetrics
from tests.helpers import make_tree, topic

DIM = 8


def unit(axis: int, eps: float = 0.0) -> list[float]:
    v = [0.0] * DIM
    v[axis] = 1.0
    v[(axis + 1) % DIM] = eps
    return v


def near(axis: int, other: int, w: float) -> list[float]:
    """Вектор между осями: косинус к unit(axis) = 1/sqrt(1+w^2)."""
    v = [0.0] * DIM
    v[axis], v[other] = 1.0, w
    return v


def tree(**settings: int) -> Tree:
    """root -> finance -> cards; root -> tech."""
    settings.setdefault("min_cluster_size", 10)
    return make_tree(
        topic("finance", title="Финансы", description="Деньги"),
        topic("cards", "finance", title="Карты", description="Банковские карты"),
        topic("tech", title="Техника", description="Гаджеты"),
        **settings,
    )


TOPIC_VECTORS = {
    "Финансы. Деньги": unit(0),
    "Карты. Банковские карты": unit(1),
    "Техника. Гаджеты": unit(2),
    "Налоги. Налоговые вопросы": unit(7),
    "Кэшбэк. Кэшбэк по картам": unit(6),
    "Лимиты. Лимиты по картам": unit(4),
}


def proposal(
    id_: str = "taxes",
    title: str = "Налоги",
    description: str = "Налоговые вопросы",
    guard: str = "Разговор о налогах или налоговых вычетах?",
) -> TopicProposal:
    return TopicProposal(id=id_, title=title, description=description, examples=[], guard=guard)


def cluster(embedder: FakeEmbedder, prefix: str, axis: int, n: int = 12) -> list[str]:
    texts = []
    for i in range(n):
        text = f"{prefix} текст {i}"
        embedder.mapping[text] = unit(axis, eps=0.001 * i)
        texts.append(text)
    return texts


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dict(TOPIC_VECTORS), dim=DIM)


@pytest.fixture
def metrics() -> RecordingMetrics:
    return RecordingMetrics()


def svc(embedder: FakeEmbedder, text: FakeText, metrics: RecordingMetrics) -> DiscoverService:
    return DiscoverService(embedder, text, metrics)


async def test_creates_topic_from_cluster(
    embedder: FakeEmbedder, metrics: RecordingMetrics
) -> None:
    text = FakeText([proposal()])
    texts = cluster(embedder, "tax", 5)
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, texts)
    [a] = res.accepted
    assert (a.topic.id, a.topic.parent_id, a.topic.origin) == ("taxes", ROOT_ID, "discovered")
    assert a.topic.guard == "Разговор о налогах или налоговых вычетах?"
    assert sorted(a.members) == list(range(12))
    assert res.considered == [] and res.rejected == 0
    assert text.levels == [(1, 3)]  # дети root - уровень 1 при max_depth 3
    assert text.siblings == [["finance", "tech"]]
    assert 10 <= len(text.samples[0]) <= 12  # весь кластер: он меньше samples_per_cluster


async def test_noise_is_considered_without_llm(
    embedder: FakeEmbedder, metrics: RecordingMetrics
) -> None:
    text = FakeText([proposal()])
    texts = [f"шум {i}" for i in range(12)]
    for i, t in enumerate(texts):  # разбросаны по осям: ни кластера, ни одной темы
        embedder.mapping[t] = near(i % DIM, (i + 3) % DIM, -1.0 if i % 2 else 0.5)
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, texts)
    assert res.accepted == [] and sorted(res.considered) == list(range(12))
    assert text.calls["propose_topic"] == 0


async def test_id_collision_in_other_branch_gets_suffix(
    embedder: FakeEmbedder, metrics: RecordingMetrics
) -> None:
    text = FakeText([proposal(id_="cards")])  # cards уже есть под finance
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, cluster(embedder, "x", 5))
    assert [a.topic.id for a in res.accepted] == ["cards_2"]


@pytest.mark.parametrize("guard", ["yes", "Да, если про налоги", ""])
async def test_meaningless_guard_replaced_with_default(
    embedder: FakeEmbedder, metrics: RecordingMetrics, guard: str
) -> None:
    """Модель пишет ответ вместо вопроса: тема принимается, guard - стандартный вопрос."""
    text = FakeText([proposal(guard=guard)])
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, cluster(embedder, "tax", 5))
    [a] = res.accepted
    assert a.topic.guard == "Этот разговор действительно на тему «Налоги»: Налоговые вопросы?"


async def test_invalid_output_skipped(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([InvalidModelOutput("bad json")])
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, cluster(embedder, "t", 5))
    assert res.accepted == [] and len(res.considered) == 12
    assert metrics.counts["model_error:discover"] == 1


async def test_dup_of_sibling_rejected(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([proposal(id_="gadgets", title="Техника", description="Гаджеты")])
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, cluster(embedder, "g", 5))
    assert res.accepted == [] and res.rejected == 1 and len(res.considered) == 12


@pytest.mark.parametrize("same,accepted", [(True, 0), (False, 1)])
async def test_ask_zone_uses_llm(
    embedder: FakeEmbedder, metrics: RecordingMetrics, same: bool, accepted: int
) -> None:
    embedder.mapping["Гаджеты2. Около техники"] = near(2, 3, 0.8)  # косинус ~0.78: зона ask
    text = FakeText([proposal(id_="gadgets", title="Гаджеты2", description="Около техники")], same)
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, cluster(embedder, "g", 5))
    assert len(res.accepted) == accepted and text.calls["same_topic"] == 1


async def test_dedup_between_accepted(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([proposal(), proposal(id_="taxes_again")])
    texts = cluster(embedder, "a", 5) + cluster(embedder, "b", 3)
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, texts)
    assert [a.topic.id for a in res.accepted] == ["taxes"] and res.rejected == 1
    assert sorted(res.considered) == list(range(12, 24))


async def test_two_clusters_two_topics(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText(
        [proposal(), proposal(id_="cashback", title="Кэшбэк", description="Кэшбэк по картам")]
    )
    texts = cluster(embedder, "a", 5) + cluster(embedder, "b", 3)
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, texts)
    assert {a.topic.id: sorted(a.members) for a in res.accepted} == {
        "taxes": list(range(12)),
        "cashback": list(range(12, 24)),
    }


async def test_leaf_needs_two_subtopics(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([proposal(id_="cashback", title="Кэшбэк", description="Кэшбэк по картам")])
    res = await svc(embedder, text, metrics).propose(tree(), "cards", cluster(embedder, "a", 5))
    assert res.accepted == [] and sorted(res.considered) == list(range(12))


async def test_leaf_splits_into_two(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText(
        [
            proposal(id_="cashback", title="Кэшбэк", description="Кэшбэк по картам"),
            proposal(id_="limits", title="Лимиты", description="Лимиты по картам"),
        ]
    )
    texts = cluster(embedder, "a", 5) + cluster(embedder, "b", 3)
    res = await svc(embedder, text, metrics).propose(tree(), "cards", texts)
    assert [(a.topic.id, a.topic.parent_id) for a in res.accepted] == [
        ("cashback", "cards"),
        ("limits", "cards"),
    ]
    assert text.levels[0] == (3, 3) and text.siblings[0] == []


async def test_leaf_subtopic_same_as_leaf_rejected(
    embedder: FakeEmbedder, metrics: RecordingMetrics
) -> None:
    same_as_leaf = proposal(id_="cards_more", title="Карты", description="Банковские карты")
    text = FakeText(
        [same_as_leaf, proposal(id_="limits", title="Лимиты", description="Лимиты по картам")]
    )
    texts = cluster(embedder, "a", 5) + cluster(embedder, "b", 3)
    res = await svc(embedder, text, metrics).propose(tree(), "cards", texts)
    assert res.accepted == [] and res.rejected == 1  # одна подтема - лист не делится


async def test_respects_slots_left(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([proposal()])
    res = await svc(embedder, text, metrics).propose(
        tree(max_children=2), ROOT_ID, cluster(embedder, "a", 5)
    )
    assert res.accepted == [] and len(res.considered) == 12
    assert text.calls["propose_topic"] == 0


async def test_slots_limit_clusters(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText(
        [proposal(), proposal(id_="cashback", title="Кэшбэк", description="Кэшбэк по картам")]
    )
    texts = cluster(embedder, "a", 5) + cluster(embedder, "b", 3)
    res = await svc(embedder, text, metrics).propose(tree(max_children=3), ROOT_ID, texts)
    assert len(res.accepted) == 1 and text.calls["propose_topic"] == 1
    assert len(res.considered) == 12


async def test_no_depth_left(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([proposal()])
    res = await svc(embedder, text, metrics).propose(
        tree(max_depth=2), "cards", cluster(embedder, "a", 5)
    )
    assert res.accepted == [] and text.calls["propose_topic"] == 0


async def test_samples_truncated(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([proposal()])
    await svc(embedder, text, metrics).propose(
        tree(sample_chars=4), ROOT_ID, cluster(embedder, "длинный", 5)
    )
    assert all(len(s) <= 4 for s in text.samples[0])


async def test_empty_texts_ignored(embedder: FakeEmbedder, metrics: RecordingMetrics) -> None:
    text = FakeText([proposal()])
    texts = cluster(embedder, "a", 5) + ["", "   "]
    res = await svc(embedder, text, metrics).propose(tree(), ROOT_ID, texts)
    [a] = res.accepted
    assert sorted(a.members) == list(range(12)) and sorted(res.considered) == [12, 13]


async def test_embeddings_cached_between_runs(
    embedder: FakeEmbedder, metrics: RecordingMetrics
) -> None:
    class Counting(FakeEmbedder):
        texts: list[str] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.texts.extend(texts)
            return await super().embed(texts)

    e = Counting(dict(embedder.mapping), dim=DIM)
    texts = cluster(e, "tax", 5)
    s = svc(e, FakeText([InvalidModelOutput("x")]), metrics)
    await s.propose(tree(), ROOT_ID, texts)
    before = len(e.texts)
    await s.propose(tree(), ROOT_ID, texts)  # тот же буфер на следующем discover
    assert len(e.texts) == before and before >= 12


async def test_embed_more_texts_than_cache(
    monkeypatch: pytest.MonkeyPatch, embedder: FakeEmbedder, metrics: RecordingMetrics
) -> None:
    import llm_sorter.application.discover as d

    monkeypatch.setattr(d, "EMBED_CACHE_SIZE", 3)
    s = svc(embedder, FakeText([]), metrics)
    assert len(await s._embed([f"t{i}" for i in range(10)])) == 10


async def test_accepted_topics_shown_to_model_as_siblings(
    embedder: FakeEmbedder, metrics: RecordingMetrics
) -> None:
    text = FakeText(
        [proposal(), proposal(id_="cashback", title="Кэшбэк", description="Кэшбэк по картам")]
    )
    texts = cluster(embedder, "a", 5) + cluster(embedder, "b", 3)
    await svc(embedder, text, metrics).propose(tree(), ROOT_ID, texts)
    assert text.siblings == [["finance", "tech"], ["finance", "tech", "taxes"]]
