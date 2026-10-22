from prometheus_client import CollectorRegistry

from llm_sorter.adapters.outbound.prometheus_metrics import PrometheusMetrics
from llm_sorter.domain.classification import Status


def test_counters_and_labels() -> None:
    reg = CollectorRegistry()
    m = PrometheusMetrics(reg)
    m.topic_move_attempt()
    m.topic_move_attempt()
    m.file_classified("finance", Status.CLASSIFIED, 3)
    assert reg.get_sample_value("llm_sorter_topic_move_attempts_total") == 2
    labels = {"path": "finance", "status": "classified", "version": "3"}
    assert reg.get_sample_value("llm_sorter_files_classified_total", labels) == 1


def test_all_methods_emit() -> None:
    reg = CollectorRegistry()
    m = PrometheusMetrics(reg)
    m.model_call("decide", 0.5, True)
    m.model_error("decide")
    m.guard_result("a", False)
    m.action_done("save", True, 0.1)
    m.buffer_size("a/other", 7)
    m.discover_run("a", 2, 1)
    m.reprocess_moved("a/other", "a/b")
    m.active_version(4)
    m.rules_published("cli")
    m.discover_blocked("a", True)
    m.discover_blocked("b", True)
    m.discover_blocked("b", False)
    g = reg.get_sample_value
    assert g("llm_sorter_model_calls_total", {"role": "decide", "ok": "true"}) == 1
    assert g("llm_sorter_model_latency_seconds_count", {"role": "decide"}) == 1
    assert g("llm_sorter_model_errors_total", {"role": "decide"}) == 1
    assert g("llm_sorter_guard_results_total", {"path": "a", "passed": "false"}) == 1
    assert g("llm_sorter_actions_total", {"type": "save", "ok": "true"}) == 1
    assert g("llm_sorter_action_latency_seconds_count", {"type": "save"}) == 1
    assert g("llm_sorter_other_buffer_size", {"path": "a/other"}) == 7
    assert g("llm_sorter_discover_topics_created_total", {"path": "a"}) == 2
    assert g("llm_sorter_discover_topics_rejected_total", {"path": "a"}) == 1
    assert g("llm_sorter_reprocess_moved_total", {"from_path": "a/other", "to_path": "a/b"}) == 1
    assert g("llm_sorter_active_rules_version") == 4
    assert g("llm_sorter_rules_published_total", {"source": "cli"}) == 1
    assert g("llm_sorter_discover_blocked", {"path": "a"}) == 1
    assert g("llm_sorter_discover_blocked", {"path": "b"}) == 0
