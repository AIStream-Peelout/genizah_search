"""Tests for the eval runner's metrics records and the model-comparison aggregation."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest

from scripts.run_agentic_rag_eval import build_result_record, summarize_metrics
from scripts.run_synthesis_model_comparison import aggregate_run, ensure_model_loaded, slug, write_summary


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
        {"dataset_id": "d"},
        {"id": "c", "question": "q", "conversation_history": [{"role": "user", "content": "earlier"}]},
        response_with_metrics(),
        {"overall_pass": True}, None, elapsed_seconds=12.0,
        synthesis_model="qwen/qwen3.6-27b", judge_error="ValueError: bad json",
    )

    assert record["metrics"]["elapsed_seconds"] == 12.0
    assert record["synthesis_model"] == "qwen/qwen3.6-27b"
    assert record["judge_error"] == "ValueError: bad json"
    assert record["judge"] is None
    assert record["error"] is None
    assert record["conversation_history"] == [{"role": "user", "content": "earlier"}]


def test_aggregate_run_computes_rates_and_means() -> None:
    """Pass rates, judge means, and efficiency means aggregate per run."""
    judge_schema = {
        "judge_version": 1,
        "score_scale": {"min": 0, "max": 4},
        "score_dimensions": ["a", "b"],
    }
    records = [
        {
            "deterministic": {"overall_pass": True},
            "judge": {**judge_schema, "scores": {"a": 4, "b": 2}, "score_mean": 3.0,
                      "computed_overall_pass": True, "critical_failures": []},
            "metrics": {"elapsed_seconds": 100.0, "synthesis_seconds": 40.0, "synthesis_tokens_per_second": 20.0,
                        "verification_cycles": 1, "repair_attempts": 0, "flagged_claims": 0, "success": True},
        },
        {
            "deterministic": {"overall_pass": False},
            "judge": {**judge_schema, "scores": {"a": 2, "b": 2}, "score_mean": 2.0,
                      "computed_overall_pass": False,
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
    assert aggregate["judge_score_normalized_mean"] == 0.625
    assert aggregate["judge_dimension_normalized_means"] == {"a": 0.75, "b": 0.5}
    assert aggregate["judge_scores_comparable"]
    assert aggregate["critical_failures_total"] == 1
    assert aggregate["elapsed_seconds_mean"] == 200.0
    assert aggregate["elapsed_seconds_max"] == 300.0
    assert aggregate["synthesis_tokens_per_second_mean"] == 15.0
    assert aggregate["verification_cycles_mean"] == 2.0
    assert aggregate["repair_attempts_total"] == 2
    assert aggregate["first_pass_verification_rate"] == 0.5


def test_aggregate_run_omits_legacy_judge_scores_without_schema() -> None:
    """Unknown historical scales must not be averaged beside current scores."""
    aggregate = aggregate_run([{
        "deterministic": {"overall_pass": True},
        "judge": {
            "scores": {"a": 4}, "score_mean": 4.0,
            "computed_overall_pass": True, "critical_failures": [],
        },
        "metrics": {"success": True},
    }])

    assert not aggregate["judge_scores_comparable"]
    assert aggregate["judge_score_mean"] is None
    assert aggregate["judge_score_normalized_mean"] is None


def test_write_summary_excludes_failed_partial_run(tmp_path: Path) -> None:
    """A nonzero runner exit is reported but excluded from comparisons."""
    complete_path = tmp_path / "complete.jsonl"
    partial_path = tmp_path / "partial.jsonl"
    row = {"case_id": "case", "deterministic": {"overall_pass": True}, "metrics": {"success": True}}
    complete_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    partial_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    runs = [
        {"dataset_id": "d", "dataset": None, "model": "complete", "path": str(complete_path),
         "exit_code": 0, "seconds": 1.0},
        {"dataset_id": "d", "dataset": None, "model": "partial", "path": str(partial_path),
         "exit_code": 1, "seconds": 1.0},
    ]
    meta = {"finished_at": "now", "models": ["complete", "partial"], "judge_model": None}

    write_summary(tmp_path, runs, meta)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["eligible_for_comparison"]
    assert summary["runs"][1]["aggregate"] is None
    assert summary["runs"][1]["partial_aggregate"]["cases"] == 1
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Excluded incomplete runs" in markdown
    assert "partial` exited with code 1" in markdown


def test_write_summary_separates_incompatible_judge_schemas(tmp_path: Path) -> None:
    """Different scale/dimension sets render in separate comparison sections."""
    runs = []
    for model, score_max, dimensions in [("legacy", 4, ["a"]), ("current", 10, ["a", "b"])]:
        path = tmp_path / f"{model}.jsonl"
        scores = {dimension: score_max for dimension in dimensions}
        row = {
            "case_id": "case", "deterministic": {"overall_pass": True},
            "judge": {
                "judge_version": 1 if score_max == 4 else 2,
                "score_scale": {"min": 0, "max": score_max},
                "score_dimensions": dimensions,
                "scores": scores, "score_mean": float(score_max),
                "computed_overall_pass": True, "critical_failures": [],
            },
            "metrics": {"success": True},
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        runs.append({
            "dataset_id": "d", "dataset": None, "model": model, "path": str(path),
            "exit_code": 0, "seconds": 1.0,
        })

    write_summary(
        tmp_path,
        runs,
        {"finished_at": "now", "models": ["legacy", "current"], "judge_model": "judge"},
    )

    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Judge v1 · 0.00–4.00 · 1 dimensions" in markdown
    assert "Judge v2 · 0.00–10.00 · 2 dimensions" in markdown
    assert "comparable only within the same subsection" in markdown


@pytest.mark.asyncio
async def test_failed_rejudge_clears_stale_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rejudge cannot leave an earlier judge result looking current."""
    from scripts.rejudge_eval_results import rejudge_file

    result_path = tmp_path / "results.jsonl"
    row = {
        "dataset_id": "d", "case_id": "c", "question": "q",
        "response": {"answer": "A sufficiently long answer for deterministic evaluation."},
        "deterministic": {"overall_pass": True},
        "judge": {"score_mean": 4.0}, "judge_error": None, "error": None,
    }
    result_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.rejudge_eval_results.judge_case",
        AsyncMock(side_effect=ValueError("bad judge output")),
    )
    args = argparse.Namespace(
        timeout=1.0, all=True, judge_base_url="http://judge", judge_model="judge",
        judge_provider="lmstudio",
    )

    counts = await rejudge_file(
        result_path,
        {"d": {"dataset_id": "d", "cases": [{"id": "c", "question": "q"}]}},
        args,
        "instructions",
    )

    updated = json.loads(result_path.read_text(encoding="utf-8"))
    assert counts == {"judged": 1, "failed": 1}
    assert updated["judge"] is None
    assert "bad judge output" in updated["judge_error"]


def test_slug_is_filesystem_safe() -> None:
    """Model ids with slashes and dots become safe file name parts."""
    assert slug("qwen/qwen3.6-35b-a3b") == "qwen_qwen3_6_35b_a3b"


def test_ensure_model_loaded_refuses_to_reload_resident_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared resident model must never be unloaded to change its context."""
    monkeypatch.setattr(
        "scripts.run_synthesis_model_comparison.loaded_models",
        lambda _url: {"model": {"id": "model", "state": "loaded", "loaded_context_length": 4096}},
    )
    run = Mock()
    monkeypatch.setattr("scripts.run_synthesis_model_comparison.subprocess.run", run)

    with pytest.raises(RuntimeError, match="Refusing to unload"):
        ensure_model_loaded("model", "http://127.0.0.1:1234", 32768, 7200, Mock())
    run.assert_not_called()


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
