from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from llm_sorter.domain.classification import Status


class PrometheusMetrics:
    """Реализация порта Metrics; registry по умолчанию свой, не глобальный."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        p = "llm_sorter_"
        self._files = Counter(
            p + "files_classified",
            "Классифицированные файлы",
            ["path", "status", "version"],
            registry=r,
        )
        self._calls = Counter(p + "model_calls", "Вызовы моделей", ["role", "ok"], registry=r)
        self._latency = Histogram(
            p + "model_latency_seconds", "Латентность модели", ["role"], registry=r
        )
        self._errors = Counter(p + "model_errors", "Ошибки модели", ["role"], registry=r)
        self._guard = Counter(
            p + "guard_results", "Результаты guard", ["path", "passed"], registry=r
        )
        self._actions = Counter(p + "actions", "Выполненные actions", ["type", "ok"], registry=r)
        self._action_latency = Histogram(
            p + "action_latency_seconds", "Латентность action", ["type"], registry=r
        )
        self._buffer = Gauge(p + "other_buffer_size", "Размер буфера other", ["path"], registry=r)
        self._disc_runs = Counter(p + "discover_runs", "Прогоны discover", ["path"], registry=r)
        self._disc_created = Counter(
            p + "discover_topics_created", "Созданные темы", ["path"], registry=r
        )
        self._disc_rejected = Counter(
            p + "discover_topics_rejected", "Отклонённые темы", ["path"], registry=r
        )
        self._moved = Counter(
            p + "reprocess_moved",
            "Перенесено при переобработке",
            ["from_path", "to_path"],
            registry=r,
        )
        self._blocked = Gauge(
            p + "discover_blocked",
            "Узел в max_children, а буфер other не пуст (1/0)",
            ["path"],
            registry=r,
        )
        self._move_attempts = Counter(
            p + "topic_move_attempts", "Попытки изменить финальную классификацию", registry=r
        )
        self._version = Gauge(p + "active_rules_version", "Активная версия правил", registry=r)
        self._published = Counter(
            p + "rules_published", "Публикации правил", ["source"], registry=r
        )

    def file_classified(self, path: str, status: Status, version: int) -> None:
        self._files.labels(path, status.value, str(version)).inc()

    def model_call(self, role: str, latency_s: float, ok: bool) -> None:
        self._calls.labels(role, str(ok).lower()).inc()
        self._latency.labels(role).observe(latency_s)

    def model_error(self, role: str) -> None:
        self._errors.labels(role).inc()

    def guard_result(self, path: str, passed: bool) -> None:
        self._guard.labels(path, str(passed).lower()).inc()

    def action_done(self, action_type: str, ok: bool, latency_s: float) -> None:
        self._actions.labels(action_type, str(ok).lower()).inc()
        self._action_latency.labels(action_type).observe(latency_s)

    def buffer_size(self, path: str, size: int) -> None:
        self._buffer.labels(path).set(size)

    def discover_run(self, path: str, created: int, rejected: int) -> None:
        self._disc_runs.labels(path).inc()
        self._disc_created.labels(path).inc(created)
        self._disc_rejected.labels(path).inc(rejected)

    def reprocess_moved(self, from_path: str, to_path: str) -> None:
        self._moved.labels(from_path, to_path).inc()

    def discover_blocked(self, path: str, blocked: bool) -> None:
        self._blocked.labels(path).set(1 if blocked else 0)

    def topic_move_attempt(self) -> None:
        self._move_attempts.inc()

    def active_version(self, v: int) -> None:
        self._version.set(v)

    def rules_published(self, source: str) -> None:
        self._published.labels(source).inc()
