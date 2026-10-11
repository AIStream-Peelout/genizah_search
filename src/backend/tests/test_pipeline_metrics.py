"""Tests for per-request pipeline metrics (stage timings, cycles, LLM usage)."""

import json
from typing import Any, Dict, List, Optional
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


@pytest.mark.asyncio
async def test_case23_record_llm_call_tracks_finish_reason_per_call_and_stage() -> None:
    """Case 23 (C6, finish_reason part): each call records its finish_reason
    and llm_by_stage counts them per stage; calls without one are not counted."""
    async def node(state: Dict[str, Any]) -> Dict[str, Any]:
        record_llm_call("v", {"choices": [{"finish_reason": "stop"}], "usage": {}}, 0.1)
        record_llm_call("v", {"choices": [{"finish_reason": "length"}], "usage": {}}, 0.1)
        record_llm_call("v", {"choices": [{"finish_reason": "length"}], "usage": {}}, 0.1)
        record_llm_call("v", {"usage": {"prompt_tokens": 1}}, 0.1)
        return state

    service = AgenticRAGService()
    state: Dict[str, Any] = {}
    sink: Dict[str, Any] = {"llm_calls": []}
    token = agent_module._REQUEST_METRICS.set(sink)
    try:
        await AgenticRAGService._timed_node("verify_claims", node)(state)
    finally:
        agent_module._REQUEST_METRICS.reset(token)

    assert [call["finish_reason"] for call in sink["llm_calls"]] == ["stop", "length", "length", None]
    metrics = service._build_metrics(state, started_at=agent_module.time.monotonic(),
                                     request_metrics=sink)
    assert metrics["llm_by_stage"]["verify_claims"]["calls"] == 4
    assert metrics["llm_by_stage"]["verify_claims"]["finish_reasons"] == {"stop": 1, "length": 2}


@pytest.mark.asyncio
async def test_case30_timed_node_names_the_active_stage_until_it_returns() -> None:
    """Case 30 (C5.2 prerequisite): while a node runs, the request sink names
    it as active_stage; a normal return clears it, an exception leaves the
    failing node's name for the stream backstop."""
    seen: List[Optional[str]] = []

    async def succeeds(state: Dict[str, Any]) -> Dict[str, Any]:
        """Record the active stage seen from inside the node.

        :param state: Pipeline state.
        :returns: The same state.
        :rtype: Dict[str, Any]
        """
        seen.append(agent_module._REQUEST_METRICS.get()["active_stage"])
        return state

    async def fails(state: Dict[str, Any]) -> Dict[str, Any]:
        """Raise like a buggy node.

        :param state: Pipeline state.
        :returns: Never returns.
        :rtype: Dict[str, Any]
        :raises RuntimeError: Always.
        """
        raise RuntimeError("node bug")

    sink: Dict[str, Any] = {"llm_calls": []}
    token = agent_module._REQUEST_METRICS.set(sink)
    try:
        await AgenticRAGService._timed_node("verify_claims", succeeds)({})
        assert seen == ["verify_claims"]
        assert sink["active_stage"] is None
        with pytest.raises(RuntimeError, match="node bug"):
            await AgenticRAGService._timed_node("repair_answer", fails)({})
        assert sink["active_stage"] == "repair_answer"
    finally:
        agent_module._REQUEST_METRICS.reset(token)
    # Outside a request there is no sink; the wrapper still runs the node.
    state: Dict[str, Any] = {}
    assert await AgenticRAGService._timed_node("verify_claims", AsyncMock(return_value=state))(state) is state
    assert state["stage_calls"] == {"verify_claims": 1}


def test_case23_build_metrics_reports_quote_check_noop_repair_and_verification_unavailable() -> None:
    """Case 23 (C6): _build_metrics flattens quote_check into mode, counts and
    outcome (never the verified or unverified lists), reports noop_repair and
    verification_unavailable, and keeps each call's finish_reason."""
    service = AgenticRAGService()
    state = {
        "stage_calls": {"check_quotes": 1, "verify_claims": 1, "repair_answer": 1},
        "quote_check": {
            "mode": "delete",
            "verified": [{
                "type": "quote", "text": "documents of the India Book", "supported": True,
                "source_number": 1, "reasoning": "Quoted wording found in SOURCE 1",
                "start": 10, "end": 39,
            }],
            "unverified_spans": [(50, 82, "there was no court in Fustat", "quote")],
            "counts": {
                "quotations": 3, "verified": 1, "exempt_title": 1, "exempt_question": 0,
                "terms_skipped": 2, "unverified": 1, "shelf_marks_unverified": 1,
            },
            "sentences_deleted": 2,
            "smoothing": "rejected:new_number",
            "smoothing_paragraphs": {0: "rejected:new_number"},
            "answer_too_short": False,
        },
        "repair_noop": True,
        "verification_unavailable": True,
    }
    request_metrics = {"llm_calls": [
        {"stage": "check_quotes", "model": "v", "prompt_tokens": 228, "completion_tokens": 167,
         "reasoning_tokens": None, "finish_reason": "stop", "seconds": 2.5, "with_tools": False},
        {"stage": "verify_claims", "model": "v", "prompt_tokens": 9000, "completion_tokens": 8192,
         "reasoning_tokens": None, "finish_reason": "length", "seconds": 90.0, "with_tools": False},
    ]}

    metrics = service._build_metrics(state, agent_module.time.monotonic(), request_metrics)

    assert metrics["quote_check"] == {
        "mode": "delete", "quotations": 3, "verified": 1, "exempt_title": 1,
        "exempt_question": 0, "terms_skipped": 2, "unverified": 1, "shelf_marks_unverified": 1,
        "sentences_deleted": 2, "smoothing": "rejected:new_number", "answer_too_short": False,
    }
    assert metrics["noop_repair"] is True
    assert metrics["verification_unavailable"] is True
    assert [call["finish_reason"] for call in metrics["llm_calls"]] == ["stop", "length"]
    assert metrics["llm_by_stage"]["check_quotes"]["finish_reasons"] == {"stop": 1}
    assert metrics["llm_by_stage"]["verify_claims"]["finish_reasons"] == {"length": 1}
    # The metrics travel in the SSE final event, so they must serialise.
    json.dumps(metrics)


@pytest.mark.parametrize("quote_check, mode", [
    (None, None),
    ({}, None),
    ({"mode": "repair"}, "repair"),
], ids=["key_missing", "check_never_ran", "repair_mode"])
def test_case23_build_metrics_quote_check_defaults(
    quote_check: Optional[Dict[str, Any]],
    mode: Optional[str],
) -> None:
    """Case 23 (C6): with no quote check, an empty one, or the repair-mode
    pass-through, every quote_check metric is still reported with a neutral
    default, and the two booleans default to False."""
    service = AgenticRAGService()
    state: Dict[str, Any] = {} if quote_check is None else {"quote_check": quote_check}

    metrics = service._build_metrics(state, agent_module.time.monotonic(), None)

    assert metrics["quote_check"] == {
        "mode": mode, "quotations": 0, "verified": 0, "exempt_title": 0, "exempt_question": 0,
        "terms_skipped": 0, "unverified": 0, "shelf_marks_unverified": 0,
        "sentences_deleted": 0, "smoothing": "skipped", "answer_too_short": False,
    }
    assert metrics["noop_repair"] is False
    assert metrics["verification_unavailable"] is False


def test_llm_finish_reason_tolerates_malformed_responses() -> None:
    """Case 23 (C6): a response without choices reports no finish reason."""
    assert agent_module.llm_finish_reason({"choices": [{"finish_reason": "tool_calls"}]}) == "tool_calls"
    assert agent_module.llm_finish_reason({"choices": []}) is None
    assert agent_module.llm_finish_reason({"choices": [{"message": {}}]}) is None
    assert agent_module.llm_finish_reason("not a dict") is None


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
        "reasoning_tokens": 0, "finish_reasons": {},
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


@pytest.mark.parametrize("state, expected", [
    ({}, {"keeps_verified": 0, "keeps_rejected": 0, "forced_removals": 0, "deterministic": False}),
    (
        {
            "repair_counts": {"keeps_verified": 1, "keeps_rejected": 2, "forced_removals": 3},
            "repair_deterministic": True,
        },
        {"keeps_verified": 1, "keeps_rejected": 2, "forced_removals": 3, "deterministic": True},
    ),
], ids=["no_repair", "after_repairs"])
def test_build_metrics_reports_repair_keep_protocol_totals(
    state: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    """KEEP protocol: metrics["repair"] reports the request totals of verified
    and rejected keeps and forced removals, and whether the last repair was
    deterministic; every key defaults to zero/False."""
    service = AgenticRAGService()

    metrics = service._build_metrics(state, agent_module.time.monotonic(), None)

    assert metrics["repair"] == expected
    json.dumps(metrics)
