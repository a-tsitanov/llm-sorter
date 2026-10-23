import numpy as np
import pytest

from llm_sorter.application.clustering import (
    cluster,
    cosine,
    diverse_samples,
    nearest_to_centroid,
)


def test_cluster_two_groups():
    rng = np.random.default_rng(0)
    a = rng.normal([1, 0, 0], 0.01, (12, 3))
    b = rng.normal([0, 1, 0], 0.01, (12, 3))
    groups = cluster(np.vstack([a, b]), min_cluster_size=10)
    assert sorted(len(g) for g in groups) == [12, 12]


def test_cluster_single_group_is_kept():
    rng = np.random.default_rng(1)
    groups = cluster(rng.normal([1, 0, 0, 0], 0.001, (12, 4)), min_cluster_size=10)
    assert len(groups) == 1 and len(groups[0]) >= 10


def test_cluster_sorted_by_size():
    rng = np.random.default_rng(2)
    a = rng.normal([1, 0, 0], 0.01, (11, 3))
    b = rng.normal([0, 1, 0], 0.01, (15, 3))
    groups = cluster(np.vstack([a, b]), min_cluster_size=10)
    assert [len(g) for g in groups] == [15, 11]
    assert set(groups[0]) == set(range(11, 26))


def test_cluster_too_small():
    assert cluster(np.eye(3), min_cluster_size=10) == []


def test_nearest_to_centroid():
    vectors = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [1.0, -0.1]])
    assert nearest_to_centroid(vectors, [0, 1, 3], k=1) == [0]
    assert set(nearest_to_centroid(vectors, [0, 1, 2, 3], k=4)) == {0, 1, 2, 3}
    assert nearest_to_centroid(vectors, [0, 1, 2, 3], k=4)[-1] == 2


def test_cosine():
    assert cosine([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def _blobs(rng: np.random.Generator, k: int, n: int, dim: int, noise: float) -> np.ndarray:
    # как у эмбеддингов транскриптов: общий «стиль разговора» + слабое смещение темы
    common = rng.normal(0, 1, dim)
    centers = [common + rng.normal(0, 0.9, dim) for _ in range(k)]
    return np.vstack([c + rng.normal(0, noise, (n, dim)) for c in centers])


def test_cluster_high_dim_weak_topics_are_separated():
    # 1024-d, межтемная близость ~ внутритемной: сырой HDBSCAN сливает всё в один кластер
    rng = np.random.default_rng(3)
    vectors = _blobs(rng, k=4, n=12, dim=1024, noise=1.2)
    groups = cluster(vectors, min_cluster_size=10)
    assert len(groups) == 4
    for g in groups:
        assert len({i // 12 for i in g}) == 1  # каждая группа из одной темы


def test_cluster_unstructured_material_is_not_one_topic():
    # случайные векторы без общей темы не должны становиться одной темой
    rng = np.random.default_rng(4)
    assert cluster(rng.normal(0, 1, (40, 1024)), min_cluster_size=10) == []


def test_cluster_is_deterministic():
    rng = np.random.default_rng(5)
    vectors = _blobs(rng, k=3, n=12, dim=1024, noise=1.2)
    assert cluster(vectors, 10) == cluster(vectors.copy(), 10)


def test_cluster_near_duplicates_stay_one_group():
    # 20 почти одинаковых векторов: PCA не должен раздуть шум до двух тем
    base = np.zeros(1024)
    base[5] = 1.0
    vectors = np.array([base + 0.001 * i * np.eye(1024)[i % 7] for i in range(20)])
    assert [len(g) for g in cluster(vectors, min_cluster_size=10)] == [20]


def test_diverse_samples_covers_subgroups():
    # кластер из двух подгрупп: центральные образцы из большей, но меньшая тоже представлена
    rng = np.random.default_rng(6)
    big = rng.normal([1, 0.2, 0], 0.01, (15, 3))
    small = rng.normal([1, -0.2, 0], 0.01, (5, 3))
    vectors = np.vstack([big, small])
    picked = diverse_samples(vectors, list(range(20)), k=4)
    assert len(picked) == len(set(picked)) == 4
    assert picked[0] < 15  # первый - центральный
    assert any(i >= 15 for i in picked)


def test_diverse_samples_small_cluster_returns_all():
    vectors = np.eye(3)
    assert sorted(diverse_samples(vectors, [0, 2], k=5)) == [0, 2]


def test_diverse_samples_deterministic():
    rng = np.random.default_rng(7)
    vectors = rng.normal(0, 1, (30, 16))
    idxs = list(range(30))
    assert diverse_samples(vectors, idxs, k=8) == diverse_samples(vectors.copy(), idxs, k=8)
