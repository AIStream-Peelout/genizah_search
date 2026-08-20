"""Tests for the eval runner's metrics records and the model-comparison aggregation."""

from typing import Any, Dict

from scripts.run_agentic_rag_eval import build_result_record, summarize_metrics
from scripts.run_synthesis_model_comparison import aggregate_run, slug


def response_with_metrics(**overrides: Any) -> Dict[str, Any]:
    """Build a RAG response carrying a metrics block like the backend emits.

    :param overrides: Response keys to override.
    :returns: Response dictionary.
    :rtype: Dict[str, Any]
    """
    response = {
        "answer": "x" * 500,
        "success": True,
        "error_type": None,
        "verified_claims": [{}, {}, {}],
        "flagged_claims": [{}],
        "metrics": {
            "total_seconds": 100.0,
            "synthesis_model": "qwen/qwen3.6-27b",
            "stage_timings": {"synthesize_answer": 40.0, "verify_claims": 50.0},
            "stage_calls": {"synthesize_answer": 1, "verify_claims": 2, "repair_answer": 1},
            "verification_cycles": 2,
            "repair_attempts": 1,
            "llm_calls": [{}, {}, {}],
            "llm_by_stage": {
                "synthesize_answer": {"calls": 1, "seconds": 40.0, "prompt_tokens": 6000, "completion_tokens": 800},
                "verify_claims": {"calls": 2, "seconds": 45.0, "prompt_tokens": 9000, "completion_tokens": 700},
            },
        },
    }
    response.update(overrides)
    return response


def test_summarize_metrics_extracts_efficiency_fields() -> None:
    """Latency, cycles, and synthesis throughput are lifted into a flat record."""
    metrics = summarize_metrics(response_with_metrics(), elapsed_seconds=101.5, synthesis_model="qwen/qwen3.6-27b")

    assert metrics["elapsed_seconds"] == 101.5
    assert metrics["pipeline_seconds"] == 100.0
    assert metrics["synthesis_model_used"] == "qwen/qwen3.6-27b"
    assert metrics["verification_cycles"] == 2
    assert metrics["repair_attempts"] == 1
    assert metrics["synthesis_seconds"] == 40.0
    assert metrics["synthesis_completion_tokens"] == 800
    assert metrics["synthesis_tokens_per_second"] == 20.0
    assert metrics["verification_seconds"] == 45.0
    assert metrics["answer_chars"] == 500
    assert metrics["verified_claims"] == 3
    assert metrics["flagged_claims"] == 1


def test_summarize_metrics_tolerates_missing_metrics_block() -> None:
    """Responses from an older backend (no metrics) still produce a record."""
    metrics = summarize_metrics({"answer": "a"}, elapsed_seconds=None, synthesis_model=None)

    assert metrics["synthesis_tokens_per_second"] is None
    assert metrics["verification_cycles"] is None
    assert metrics["answer_chars"] == 1


def test_result_record_carries_metrics_and_judge_error() -> None:
    """The JSONL row keeps latency/metrics and a judge failure beside the response."""
    record = build_result_record(
        {"dataset_id": "d"}, {"id": "c", "question": "q"}, response_with_metrics(),
        {"overall_pass": True}, None, elapsed_seconds=12.0,
        synthesis_model="qwen/qwen3.6-27b", judge_error="ValueError: bad json",
    )

    assert record["metrics"]["elapsed_seconds"] == 12.0
    assert record["synthesis_model"] == "qwen/qwen3.6-27b"
    assert record["judge_error"] == "ValueError: bad json"
    assert record["judge"] is None
    assert record["error"] is None


def test_aggregate_run_computes_rates_and_means() -> None:
    """Pass rates, judge means, and efficiency means aggregate per run."""
    records = [
        {
            "deterministic": {"overall_pass": True},
            "judge": {"scores": {"a": 4, "b": 2}, "score_mean": 3.0, "computed_overall_pass": True, "critical_failures": []},
            "metrics": {"elapsed_seconds": 100.0, "synthesis_seconds": 40.0, "synthesis_tokens_per_second": 20.0,
                        "verification_cycles": 1, "repair_attempts": 0, "flagged_claims": 0, "success": True},
        },
        {
            "deterministic": {"overall_pass": False},
            "judge": {"scores": {"a": 2, "b": 2}, "score_mean": 2.0, "computed_overall_pass": False,
                      "critical_failures": ["fabricated_quote"]},
            "metrics": {"elapsed_seconds": 300.0, "synthesis_seconds": 80.0, "synthesis_tokens_per_second": 10.0,
                        "verification_cycles": 3, "repair_attempts": 2, "flagged_claims": 2, "success": True},
        },
        {"error": {"type": "ReadTimeout", "message": "timeout"}},
    ]

    aggregate = aggregate_run(records)

    assert aggregate["cases"] == 3
    assert aggregate["errors"] == 1
    assert aggregate["deterministic_pass_rate"] == 0.5
    assert aggregate["judge_pass_rate"] == 0.5
    assert aggregate["judge_score_mean"] == 2.5
    assert aggregate["judge_dimension_means"] == {"a": 3.0, "b": 2.0}
    assert aggregate["critical_failures_total"] == 1
    assert aggregate["elapsed_seconds_mean"] == 200.0
    assert aggregate["elapsed_seconds_max"] == 300.0
    assert aggregate["synthesis_tokens_per_second_mean"] == 15.0
    assert aggregate["verification_cycles_mean"] == 2.0
    assert aggregate["repair_attempts_total"] == 2
    assert aggregate["first_pass_verification_rate"] == 0.5


def test_slug_is_filesystem_safe() -> None:
    """Model ids with slashes and dots become safe file name parts."""
    assert slug("qwen/qwen3.6-35b-a3b") == "qwen_qwen3_6_35b_a3b"


def test_bounded_graph_result_keeps_counts_and_truncates_lists() -> None:
    """A prolific scholar's neighborhood must not overflow the judge context."""
    from scripts.run_agentic_rag_eval import bounded_evidence, bounded_graph_result

    works = [{"title": f"Work {i}", "year": 1960 + i, "referenced_fragment_count": i,
              "referenced_fragment_samples": [{"shelfmark": f"T-S {i}.{j}"} for j in range(50)]}
             for i in range(106)]
    result = {
        "scholar": {"name": "Shelomo Dov Goitein"},
        "works": works,
        "studied_fragment_count": 7,
        "studied_fragment_samples": [{"shelfmark": f"T-S 8J{i}"} for i in range(30)],
        "relationships": [{"type": "COLLABORATED_WITH", "name": f"Person {i}"} for i in range(40)],
    }

    compact = bounded_graph_result(result)

    assert compact["work_count"] == 106
    assert len(compact["works_sample"]) == 15
    assert compact["works_sample"][0] == {"title": "Work 0", "year": 1960, "referenced_fragment_count": 0}
    assert compact["studied_fragment_samples"] == [f"T-S 8J{i}" for i in range(10)]
    assert compact["relationship_count"] == 40 and len(compact["relationships_sample"]) == 10
    import json
    assert len(json.dumps(compact)) < 6000
    assert bounded_evidence({"graph_results": [result]})["graph"][0]["work_count"] == 106


def test_busy_models_from_entries_flags_generating_and_queued() -> None:
    """Only non-idle or queued models count as busy."""
    from scripts.run_agentic_rag_eval import busy_models_from_entries

    entries = [
        {"identifier": "qwen/qwen3-4b-2507", "status": "idle", "queued": 0},
        {"identifier": "qwen3-vl-8b-heb-v18b-step700", "status": "generating", "queued": 0},
        {"identifier": "qwen/qwen3.6-35b-a3b", "status": "idle", "queued": 2},
    ]

    busy = busy_models_from_entries(entries)

    assert busy == ["qwen3-vl-8b-heb-v18b-step700 (generating, queued 0)", "qwen/qwen3.6-35b-a3b (idle, queued 2)"]
    assert busy_models_from_entries([{"identifier": "x", "status": "idle", "queued": 0}]) == []


def test_wait_for_lm_studio_idle_requires_a_quiet_window_and_times_out() -> None:
    """The gate returns only after LM Studio stays idle for the quiet window."""
    from scripts.run_agentic_rag_eval import wait_for_lm_studio_idle

    clock = {"t": 0.0}
    states = iter([["vl (generating)"], ["vl (generating)"], [], [], [], []])

    def check():
        return next(states)

    def sleep(seconds):
        clock["t"] += seconds

    waited = wait_for_lm_studio_idle(quiet_seconds=30, max_wait_seconds=1000, poll_seconds=15,
                                     check=check, log=lambda _m: None, sleep=sleep, clock=lambda: clock["t"])
    # busy at t=0,15; idle from t=30; quiet window satisfied at t=60
    assert waited == 60.0

    import pytest as _pytest
    clock["t"] = 0.0
    with _pytest.raises(TimeoutError):
        wait_for_lm_studio_idle(quiet_seconds=30, max_wait_seconds=100, poll_seconds=50,
                                check=lambda: ["vl (generating)"], log=lambda _m: None,
                                sleep=sleep, clock=lambda: clock["t"])
