"""Кластеризация эмбеддингов материала узла: PCA + HDBSCAN над L2-нормализованными векторами."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

# Подобрано на datasets/synthetic-v1 (bge-m3): на сырых 1024-d векторах темы
# разговоров почти не отделяются (косинус внутри темы ~0.62, между ~0.51),
# и HDBSCAN видит один кластер или шум.
MAX_DIM = 10  # PCA: не больше MAX_DIM и не больше n/10 компонент
MIN_DIM = 3
MIN_SAMPLES = 2  # по умолчанию = min_cluster_size: слишком строго для тем по 10-15 файлов
# HDBSCAN не нашёл групп: весь материал считается одной темой только при высокой
# связности (средний косинус к центроиду). Одна тема ~0.78-0.82, две слитые ~0.74-0.78,
# поэтому порог строгий: при сомнении файлы остаются в other.
SINGLE_TOPIC_COHESION = 0.80
# почти одинаковый материал (автоответчик, дубли): PCA раздул бы шум между копиями
NEAR_DUPLICATE_COHESION = 0.95


def _normalize(vectors: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: NDArray[np.float64] = x / norms
    return out


def _reduce(x: NDArray[np.float64]) -> NDArray[np.float64]:
    dim = max(MIN_DIM, min(MAX_DIM, len(x) // 10))
    if x.shape[1] <= dim or len(x) <= dim:
        return x
    z = PCA(n_components=dim, svd_solver="full", random_state=0).fit_transform(x)
    return _normalize(z)


def _cohesion(x: NDArray[np.float64]) -> float:
    centroid = x.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return float((x @ centroid).mean()) / norm if norm else 0.0


def cluster(vectors: NDArray[np.floating], min_cluster_size: int) -> list[list[int]]:
    """Группы индексов по убыванию размера; шум (-1) отбрасывается.

    allow_single_cluster не используется: на неоднородном материале он сливает
    всё в одну тему. Случай «весь материал узла — одна тема» проверяется отдельно.
    """
    if len(vectors) < min_cluster_size:
        return []
    x = _normalize(vectors)
    if _cohesion(x) >= NEAR_DUPLICATE_COHESION:
        return [list(range(len(x)))]
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        copy=True,
    ).fit_predict(_reduce(x))
    groups: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:
            groups[int(label)].append(i)
    if not groups and _cohesion(x) >= SINGLE_TOPIC_COHESION:
        return [list(range(len(x)))]
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def nearest_to_centroid(vectors: NDArray[np.floating], idxs: list[int], k: int) -> list[int]:
    """k индексов из idxs, ближайших (по косинусу) к центроиду группы."""
    x = _normalize(vectors[idxs])
    centroid = x.mean(axis=0)
    order = np.argsort(-(x @ centroid), kind="stable")
    return [idxs[int(i)] for i in order[:k]]


def cosine(
    a: Sequence[float] | NDArray[np.floating], b: Sequence[float] | NDArray[np.floating]
) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb) / denom if denom else 0.0
