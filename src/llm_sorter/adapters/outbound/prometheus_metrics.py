from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from llm_sorter.ports.metrics import Outcome


class PrometheusMetrics:
    """Реализация порта Metrics; registry по умолчанию свой, не глобальный."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        p = "llm_sorter_"
        self._processed = Counter(
            p + "processed", "Обработанные сообщения", ["outcome"], registry=r
        )
        self._calls = Counter(p + "model_calls", "Вызовы моделей", ["role", "ok"], registry=r)
        self._latency = Histogram(
            p + "model_latency_seconds", "Латентность модели", ["role"], registry=r
        )
        self._errors = Counter(p + "model_errors", "Ошибки модели", ["role"], registry=r)
        self._guard = Counter(
            p + "guard_results", "Результаты гейта", ["path", "passed"], registry=r
        )
        self._buffer = Gauge(p + "other_buffer_size", "Размер буфера узла", ["node"], registry=r)
        self._disc_runs = Counter(p + "discover_runs", "Прогоны discover", ["node"], registry=r)
        self._disc_created = Counter(
            p + "discover_topics_created", "Созданные темы", ["node"], registry=r
        )
        self._disc_rejected = Counter(
            p + "discover_topics_rejected", "Отклонённые темы", ["node"], registry=r
        )
        self._blocked = Gauge(
            p + "discover_blocked",
            "Узел в max_children, а буфер не пуст (1/0)",
            ["node"],
            registry=r,
        )
        self._rev = Gauge(p + "tree_rev", "Ревизия дерева тем", registry=r)

    def processed(self, outcome: Outcome) -> None:
        self._processed.labels(outcome).inc()

    def model_call(self, role: str, latency_s: float, ok: bool) -> None:
        self._calls.labels(role, str(ok).lower()).inc()
        self._latency.labels(role).observe(latency_s)

    def model_error(self, role: str) -> None:
        self._errors.labels(role).inc()

    def guard_result(self, path: str, passed: bool) -> None:
        self._guard.labels(path, str(passed).lower()).inc()

    def buffer_size(self, node: str, size: int) -> None:
        self._buffer.labels(node).set(size)

    def discover_run(self, node: str, created: int, rejected: int) -> None:
        self._disc_runs.labels(node).inc()
        self._disc_created.labels(node).inc(created)
        self._disc_rejected.labels(node).inc(rejected)

    def discover_blocked(self, node: str, blocked: bool) -> None:
        self._blocked.labels(node).set(1 if blocked else 0)

    def tree_rev(self, rev: int) -> None:
        self._rev.set(rev)
