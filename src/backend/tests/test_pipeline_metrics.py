"""Tests for per-request pipeline metrics (stage timings, cycles, LLM usage)."""

from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from src.backend import lms_agentic_search as agent_module
from src.backend.lms_agentic_search import AgenticRAGService, record_llm_call


@pytest.mark.asyncio
async def test_timed_node_accumulates_time_calls_and_stage_attribution() -> None:
    """The wrapper counts calls, sums time, and attributes LLM calls to the stage."""
    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        record_llm_call("m", {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}, 0.5)
        return state

    timed = AgenticRAGService._timed_node("verify_claims", node)
    state: Dict[str, Any] = {}
    sink: Dict[str, Any] = {"llm_calls": []}
    token = agent_module._REQUEST_METRICS.set(sink)
    try:
        await timed(state)
        await timed(state)
    finally:
        agent_module._REQUEST_METRICS.reset(token)

    assert state["stage_calls"] == {"verify_claims": 2}
    assert state["stage_timings"]["verify_claims"] >= 0.0
    assert [call["stage"] for call in sink["llm_calls"]] == ["verify_claims", "verify_claims"]
    assert sink["llm_calls"][0]["completion_tokens"] == 5
    assert agent_module._CURRENT_STAGE.get() == ""


def test_record_llm_call_is_a_no_op_outside_a_request() -> None:
    """No sink means nothing is recorded (unit tests, warm-up calls)."""
    record_llm_call("m", {"usage": {"prompt_tokens": 1}}, 0.1)  # must not raise


def test_build_metrics_reports_cycles_and_per_stage_usage() -> None:
    """Verification cycles and repair attempts come from stage call counts."""
    service = AgenticRAGService()
    state = {
        "synthesis_model_override": "qwen/qwen3.6-27b",
        "stage_timings": {"synthesize_answer": 12.5, "verify_claims": 20.0, "repair_answer": 3.0},
        "stage_calls": {"synthesize_answer": 1, "verify_claims": 3, "repair_answer": 2},
        "retry_count": 2,
    }
    request_metrics = {"llm_calls": [
        {"stage": "synthesize_answer", "model": "qwen/qwen3.6-27b", "prompt_tokens": 4000,
         "completion_tokens": 600, "seconds": 12.0, "with_tools": False},
        {"stage": "verify_claims", "model": "v", "prompt_tokens": 5000, "completion_tokens": 300,
         "seconds": 9.0, "with_tools": False},
        {"stage": "verify_claims", "model": "v", "prompt_tokens": 5000, "completion_tokens": 250,
         "seconds": 8.0, "with_tools": False},
    ]}

    metrics = service._build_metrics(state, started_at=agent_module.time.monotonic() - 40.0,
                                     request_metrics=request_metrics)

    assert metrics["synthesis_model"] == "qwen/qwen3.6-27b"
    assert metrics["verification_cycles"] == 3
    assert metrics["repair_attempts"] == 2
    assert metrics["retry_count"] == 2
    assert metrics["total_seconds"] >= 40.0
    assert metrics["llm_by_stage"]["synthesize_answer"] == {
        "calls": 1, "seconds": 12.0, "prompt_tokens": 4000, "completion_tokens": 600,
        "reasoning_tokens": 0,
    }
    assert metrics["llm_by_stage"]["verify_claims"]["calls"] == 2
    assert metrics["llm_by_stage"]["verify_claims"]["completion_tokens"] == 550
    assert len(metrics["llm_calls"]) == 3


@pytest.mark.asyncio
async def test_chat_response_carries_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """The metrics sink is active during the graph run and reported on the response."""
    service = AgenticRAGService()

    async def fake_invoke(initial_state: Dict[str, Any]) -> Dict[str, Any]:
        record_llm_call("qwen/qwen3.6-35b-a3b", {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}, 1.0)
        return {**initial_state, "final_answer": "answer", "verified_claims": [],
                "verification_summary": {}, "stage_calls": {"verify_claims": 1},
                "stage_timings": {"verify_claims": 1.0}}

    monkeypatch.setattr(service.graph, "ainvoke", fake_invoke)

    response = await service.chat("Question", conversation_history=None)

    assert response.metrics["verification_cycles"] == 1
    assert response.metrics["repair_attempts"] == 0
    assert response.metrics["llm_calls"][0]["completion_tokens"] == 3
    assert response.metrics["synthesis_model"] == service.synthesis_model
    assert agent_module._REQUEST_METRICS.get() is None
