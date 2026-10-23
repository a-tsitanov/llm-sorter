from prometheus_client import CollectorRegistry

from llm_sorter.adapters.outbound.prometheus_metrics import PrometheusMetrics


def test_all_methods_emit() -> None:
    reg = CollectorRegistry()
    m = PrometheusMetrics(reg)
    m.processed("classified")
    m.processed("other")
    m.processed("other")
    m.model_call("decide", 0.5, True)
    m.model_error("decide")
    m.guard_result("finance.cards", False)
    m.buffer_size("root", 7)
    m.discover_run("root", 2, 1)
    m.discover_blocked("a", True)
    m.discover_blocked("b", True)
    m.discover_blocked("b", False)
    m.tree_rev(4)
    g = reg.get_sample_value
    assert g("llm_sorter_processed_total", {"outcome": "classified"}) == 1
    assert g("llm_sorter_processed_total", {"outcome": "other"}) == 2
    assert g("llm_sorter_model_calls_total", {"role": "decide", "ok": "true"}) == 1
    assert g("llm_sorter_model_latency_seconds_count", {"role": "decide"}) == 1
    assert g("llm_sorter_model_errors_total", {"role": "decide"}) == 1
    assert g("llm_sorter_guard_results_total", {"path": "finance.cards", "passed": "false"}) == 1
    assert g("llm_sorter_other_buffer_size", {"node": "root"}) == 7
    assert g("llm_sorter_discover_runs_total", {"node": "root"}) == 1
    assert g("llm_sorter_discover_topics_created_total", {"node": "root"}) == 2
    assert g("llm_sorter_discover_topics_rejected_total", {"node": "root"}) == 1
    assert g("llm_sorter_discover_blocked", {"node": "a"}) == 1
    assert g("llm_sorter_discover_blocked", {"node": "b"}) == 0
    assert g("llm_sorter_tree_rev") == 4
