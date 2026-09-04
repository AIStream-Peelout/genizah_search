"""Regression tests for multi-turn (follow-up) handling in the agentic RAG pipeline.

Production delivers conversation history as pydantic ``ChatMessage`` objects;
several consumers only handled plain dicts and silently saw an empty
conversation. These tests exercise every history shape, the follow-up
resolver, the router's use of the resolved query, and the eval runner's
multi-turn support — all with the LLM mocked.
"""

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts.run_agentic_rag_eval import (
    evaluate_deterministically,
    load_json,
    run_chat_case,
)
from src.backend.lms_agentic_search import (
    AgenticRAGService,
    ConversationTurn,
    ModelUnavailableError,
    QueryPlan,
    SearchAction,
    accept_resolved_query,
    answer_prose,
    conversation_grounding_text,
    normalize_conversation_history,
    parse_resolution_reply,
    reference_cues,
    render_conversation,
    search_query,
)
from src.backend.ollama_rag_service import ChatMessage


TURN1_ANSWER = (
    "Scholarly analysis of Yom Kippur piyyut fragments centers on a single manuscript. "
    "Meir Wallenstein examines a unique Kol-Nidré piyyut preserved in the Gaster "
    "Collection, citing the shelf mark Rylands Genizah Fragment 1. Wallenstein notes "
    "that the poem is \"framed in the liturgical composition Kol-Nidrê\" (Wallenstein, "
    "p. 489). The work comprises eighteen verses, each containing three rhyming lines "
    "(Wallenstein, p. 489)."
    "\n\n---\n**Related catalog entries:**\n\n"
    "- **[Manchester: Rylands Genizah fragment 1](doc:abc123)**\n  Kol Nidre poem"
    "\n\n---\n**Works cited:**\n\n"
    "- Meir Wallenstein, [A Unique Kol-nidré Piyyut](https://doi.org/10.7227/bjrl.41.2.11)"
    "\n\n---\n**Manuscripts these works are based on:**\n\n"
    "- Jewish Marriage in Palestine — [T-S NS J48](doc:d1), [T-S 8.268](doc:d2)"
)
FOLLOWUP = "Can you give some samples of what the verses actually state?"
RESOLVED = (
    "What do the verses of the Kol-Nidré piyyut in Rylands Genizah Fragment 1 actually state?"
)


def chat_message_history() -> List[ChatMessage]:
    """Build the history exactly as the ``/chat`` endpoint delivers it.

    :returns: Pydantic ``ChatMessage`` turns.
    :rtype: List[ChatMessage]
    """
    return [
        ChatMessage(role="user", content="Yom Kippur Piyyut Fragments"),
        ChatMessage(role="assistant", content=TURN1_ANSWER),
    ]


def base_state(**overrides: Any) -> Dict[str, Any]:
    """Build a minimal RAG state for node-level tests.

    :param overrides: State keys to set or replace.
    :returns: State dictionary.
    :rtype: Dict[str, Any]
    """
    state: Dict[str, Any] = {
        "user_query": FOLLOWUP,
        "conversation_history": chat_message_history(),
        "resolved_query": None,
        "is_followup": False,
        "resolved_entities": [],
        "processing_steps": [],
    }
    state.update(overrides)
    return state


def resolver_reply(followup: str, question: str) -> str:
    """Format a resolver-model reply.

    :param followup: ``yes`` or ``no``.
    :param question: Standalone question line.
    :returns: Two-line reply text.
    :rtype: str
    """
    return f"FOLLOWUP: {followup}\nQUESTION: {question}"


def tool_call_response(arguments: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Build a chat-completion response carrying a ``create_search_plan`` call.

    :param arguments: Tool-call arguments.
    :param content: Optional assistant prose alongside the call.
    :returns: Response payload.
    :rtype: Dict[str, Any]
    """
    return {
        "choices": [{
            "message": {
                "content": content,
                "tool_calls": [{
                    "function": {
                        "name": "create_search_plan",
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
        }],
    }


# ---------------------------------------------------------------------------
# History normalization
# ---------------------------------------------------------------------------

def test_normalize_history_accepts_every_production_and_test_shape() -> None:
    """Pydantic messages, role dicts, legacy turn dicts, and turn objects all normalize."""
    legacy_turn = ConversationTurn(
        timestamp="2024-01-01T00:00:00",
        user_query="Tell me about T-S 8J22.22",
        answer="T-S 8J22.22 is a business letter.",
    )
    history = [
        ChatMessage(role="user", content="first question"),
        {"role": "assistant", "content": "first answer"},
        {"user_query": "second question", "answer": "second answer"},
        legacy_turn,
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "   "},
    ]

    turns = normalize_conversation_history(history)

    assert turns == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "Tell me about T-S 8J22.22"},
        {"role": "assistant", "content": "T-S 8J22.22 is a business letter."},
    ]
    assert normalize_conversation_history(None) == []
    assert normalize_conversation_history(turns) == turns


def test_answer_prose_strips_appendices_links_citations_and_flags() -> None:
    """Prior answers reach prompts as prose only."""
    prose = answer_prose(TURN1_ANSWER + " ⟦flag:1⟧flagged⟦/flag⟧")

    assert "Works cited" not in prose
    assert "Related catalog entries" not in prose
    assert "doc:" not in prose
    assert "(Wallenstein, p. 489)" not in prose
    assert "⟦" not in prose
    assert "Rylands Genizah Fragment 1" in prose
    assert "eighteen verses" in prose


def test_render_conversation_bounds_turns_and_uses_prose() -> None:
    """Only the most recent turns are rendered, assistant turns as prose."""
    turns = normalize_conversation_history(chat_message_history())

    rendered = render_conversation(turns, max_turns=1, user_chars=50, assistant_chars=80)

    assert rendered.startswith("Assistant: Scholarly analysis")
    assert "User:" not in rendered
    assert len(rendered) <= len("Assistant: ") + 80


def test_grounding_text_includes_appendix_shelf_marks_from_pydantic_history() -> None:
    """Shelf marks a previous answer printed (even in appendices) count as cited."""
    state = base_state(resolved_query=RESOLVED)

    text = conversation_grounding_text(state)

    assert "Rylands Genizah Fragment 1" in text
    assert "T-S NS J48" in text
    assert FOLLOWUP in text and RESOLVED in text


# ---------------------------------------------------------------------------
# Follow-up detection and rewrite guards
# ---------------------------------------------------------------------------

def test_reference_cues_flag_definite_phrases_pronouns_and_openers() -> None:
    """Cues fire for references to earlier turns, not for collection words."""
    turns = normalize_conversation_history(chat_message_history())

    assert reference_cues(FOLLOWUP, turns) == ["the verses"]
    assert "it" in reference_cues("When was it copied?", turns)
    assert "that" in reference_cues("Who wrote that?", turns)
    assert "and" in reference_cues("and what about the Gaster Collection?", turns)
    assert reference_cues("Tell me about ketubbot in the Genizah", turns) == []
    assert reference_cues("What are the Tisha B'Av kinnot fragments?", turns) == []
    assert reference_cues(FOLLOWUP, []) == []


def test_parse_resolution_reply_handles_labels_case_and_missing_lines() -> None:
    """The two-line format parses leniently and fails closed."""
    assert parse_resolution_reply("FOLLOWUP: yes\nQUESTION: What is X?", "orig") == (True, "What is X?")
    assert parse_resolution_reply("followup: No\nquestion: orig", "orig") == (False, "orig")
    assert parse_resolution_reply("Follow-up: true\n", "orig") == (True, "orig")
    assert parse_resolution_reply("garbage", "orig") == (False, "orig")
    assert parse_resolution_reply("", "orig") == (False, "orig")


@pytest.mark.parametrize(
    ("rewrite", "reason_fragment"),
    [
        (FOLLOWUP, "unchanged"),
        ("What specific verses does the Kol-Nidré piyyut contain? " * 6, "too long"),
        ('What do the verses "framed in the liturgical composition" state?', "quoted"),
        ("What do the verses of T-S 13J13.30 state?", "shelf mark"),
        (
            "Scholarly analysis of ketubbot focusing on their content, historical context, "
            "comparative legal significance and emphasis",
            "introduces terms",
        ),
        ("Where is the Kol-Nidré piyyut preserved?", "drops the user's terms"),
    ],
)
def test_accept_resolved_query_rejects_bad_rewrites(rewrite: str, reason_fragment: str) -> None:
    """Each guard was hit by a real production-model rewrite; all fall back safely."""
    grounding = conversation_grounding_text(base_state())

    accepted, reason = accept_resolved_query(FOLLOWUP, rewrite, grounding)

    assert accepted is None
    assert reason_fragment in reason


def test_accept_resolved_query_keeps_grounded_rewrite_and_strips_citations() -> None:
    """A grounded rewrite is accepted; a copied page citation is stripped, not fatal."""
    grounding = conversation_grounding_text(base_state())

    accepted, reason = accept_resolved_query(
        FOLLOWUP,
        RESOLVED.rstrip("?") + " (Wallenstein, p. 489)?",
        grounding,
    )

    assert reason == "accepted"
    assert accepted == RESOLVED
    assert "p. 489" not in accepted


# ---------------------------------------------------------------------------
# Resolver node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolver_skips_llm_without_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first message is never rewritten and costs no model call."""
    service = AgenticRAGService()
    llm = AsyncMock()
    monkeypatch.setattr(service, "_call_llm", llm)
    state = base_state(conversation_history=None)

    result = await service._resolve_query_node(state)

    llm.assert_not_awaited()
    assert result["resolved_query"] == FOLLOWUP
    assert result["is_followup"] is False
    assert result["conversation_history"] == []


@pytest.mark.asyncio
async def test_resolver_rewrites_followup_from_pydantic_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production history shape yields a contextualized standalone question."""
    service = AgenticRAGService()
    llm = AsyncMock(return_value=resolver_reply("yes", RESOLVED))
    monkeypatch.setattr(service, "_call_llm", llm)
    state = base_state()

    result = await service._resolve_query_node(state)

    assert result["resolved_query"] == RESOLVED
    assert result["is_followup"] is True
    assert result["conversation_history"] == normalize_conversation_history(chat_message_history())
    assert search_query(result) == RESOLVED
    prompt = llm.call_args.kwargs["messages"][1]["content"]
    assert "Rylands Genizah Fragment 1" in prompt
    assert "Works cited" not in prompt
    assert "'the verses'" in prompt  # deterministic cue passed as a hint
    assert llm.call_args.kwargs["model"] == service.router_model
    assert any("Interpreted follow-up as" in step for step in result["processing_steps"])


@pytest.mark.asyncio
async def test_resolver_leaves_topic_change_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-contained message is searched as written even with history present."""
    service = AgenticRAGService()
    message = "Tell me about ketubbot in the Cairo Genizah"
    llm = AsyncMock(return_value=resolver_reply("no", message))
    monkeypatch.setattr(service, "_call_llm", llm)

    result = await service._resolve_query_node(base_state(user_query=message))

    assert result["resolved_query"] == message
    assert result["is_followup"] is False
    assert llm.await_count == 1


@pytest.mark.asyncio
async def test_resolver_runs_dedicated_rewrite_when_first_pass_repeats_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'FOLLOWUP: yes' with the message repeated triggers one rewrite-only pass."""
    service = AgenticRAGService()
    llm = AsyncMock(side_effect=[
        resolver_reply("yes", FOLLOWUP),
        f"Standalone question: {RESOLVED}",
    ])
    monkeypatch.setattr(service, "_call_llm", llm)

    result = await service._resolve_query_node(base_state())

    assert llm.await_count == 2
    assert result["resolved_query"] == RESOLVED
    assert result["is_followup"] is True
    second_system_prompt = llm.call_args_list[1].kwargs["messages"][0]["content"]
    assert "rewrite the latest user message" in second_system_prompt


@pytest.mark.asyncio
async def test_resolver_falls_back_to_literal_message_on_bad_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hallucinated rewrite is rejected by the guards; the user's words are searched."""
    service = AgenticRAGService()
    polluted = (
        "Scholarly analysis of ketubbot focusing on content, structure, historical context, "
        "comparative legal significance and liturgical emphasis"
    )
    llm = AsyncMock(side_effect=[resolver_reply("yes", polluted), polluted])
    monkeypatch.setattr(service, "_call_llm", llm)

    result = await service._resolve_query_node(base_state())

    assert result["resolved_query"] == FOLLOWUP
    assert result["is_followup"] is True
    assert any("kept as written" in step for step in result["processing_steps"])


@pytest.mark.asyncio
async def test_resolver_degrades_gracefully_but_propagates_capacity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolver failure never blocks the turn; model unavailability surfaces as such."""
    service = AgenticRAGService()
    monkeypatch.setattr(service, "_call_llm", AsyncMock(side_effect=RuntimeError("boom")))

    result = await service._resolve_query_node(base_state())

    assert result["resolved_query"] == FOLLOWUP
    assert result["is_followup"] is False

    monkeypatch.setattr(
        service, "_call_llm", AsyncMock(side_effect=ModelUnavailableError("busy"))
    )
    with pytest.raises(ModelUnavailableError):
        await service._resolve_query_node(base_state())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_plans_on_resolved_query_and_keeps_grounded_shelf_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The router sees the standalone question; a shelf mark cited in a prior answer survives."""
    service = AgenticRAGService()
    router = AsyncMock(return_value=tool_call_response({
        "actions": [
            {"search_type": "primary_shelfmark", "query": "Rylands Genizah Fragment 1"},
            {"search_type": "bibliography_hybrid", "query": "Kol Nidre piyyut Wallenstein",
             "keyword_weight": 70, "semantic_weight": 30},
        ],
        "needs_primary_secondary_linking": True,
        "is_followup": True,
        "reasoning": "Follow-up about a known manuscript",
    }))
    monkeypatch.setattr(service, "_call_llm_with_tools", router)
    monkeypatch.setattr(service, "_add_hebrew_search_variant", AsyncMock(side_effect=lambda plan, state: plan))
    state = base_state(resolved_query=RESOLVED, is_followup=True)

    result = await service._route_query_node(state)

    router_messages = router.call_args.kwargs["messages"]
    assert router_messages[1]["content"].startswith(RESOLVED)
    assert FOLLOWUP in router_messages[1]["content"]
    assert "Rylands Genizah Fragment 1" in router_messages[0]["content"]  # history block
    assert "Works cited" not in router_messages[0]["content"]
    assert [a.search_type for a in result["query_plan"].actions] == [
        "primary_shelfmark", "bibliography_hybrid",
    ]
    assert router.await_count == 1


@pytest.mark.asyncio
async def test_router_adds_bibliography_search_to_shelfmark_only_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manuscript-only plan still retrieves the scholarship on that manuscript."""
    service = AgenticRAGService()
    monkeypatch.setattr(service, "_call_llm_with_tools", AsyncMock(return_value=tool_call_response({
        "actions": [{"search_type": "primary_shelfmark", "query": "Rylands Genizah Fragment 1"}],
        "needs_primary_secondary_linking": True,
        "reasoning": "Shelf mark lookup",
    })))
    monkeypatch.setattr(service, "_add_hebrew_search_variant", AsyncMock(side_effect=lambda plan, state: plan))

    result = await service._route_query_node(base_state(resolved_query=RESOLVED, is_followup=True))

    actions = result["query_plan"].actions
    assert actions[0].search_type == "primary_shelfmark"
    assert actions[1] == SearchAction(
        search_type="bibliography_hybrid", query=RESOLVED,
        keyword_weight=70, semantic_weight=30, num_results=5,
    )


@pytest.mark.asyncio
async def test_router_fallback_uses_resolved_query_after_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tool call twice → bibliography fallback on the standalone question, flagged as follow-up."""
    service = AgenticRAGService()
    router = AsyncMock(return_value={"choices": [{"message": {"content": "", "tool_calls": []}}]})
    monkeypatch.setattr(service, "_call_llm_with_tools", router)
    monkeypatch.setattr(service, "_add_hebrew_search_variant", AsyncMock(side_effect=lambda plan, state: plan))

    result = await service._route_query_node(base_state(resolved_query=RESOLVED, is_followup=True))

    assert router.await_count == 2
    retry_prompt = router.call_args_list[1].kwargs["messages"][1]["content"]
    assert "calling the create_search_plan tool" in retry_prompt
    plan = result["query_plan"]
    assert plan.actions == [SearchAction(
        search_type="bibliography_hybrid", query=RESOLVED,
        keyword_weight=70, semantic_weight=30, num_results=5,
    )]
    assert plan.is_followup is True
    assert "Fallback" in plan.reasoning


@pytest.mark.asyncio
async def test_router_salvages_plan_from_prose_when_tool_call_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan the server returned as text (unparsed tool call) is still used."""
    service = AgenticRAGService()
    plan_json = json.dumps({
        "actions": [{"search_type": "bibliography_hybrid", "query": "Kol Nidre piyyut",
                     "keyword_weight": 70, "semantic_weight": 30}],
        "needs_primary_secondary_linking": True,
        "reasoning": "prose plan",
    })
    router = AsyncMock(return_value={"choices": [{"message": {
        "content": f"Here is the plan:\n{plan_json}", "tool_calls": None,
    }}]})
    monkeypatch.setattr(service, "_call_llm_with_tools", router)
    monkeypatch.setattr(service, "_add_hebrew_search_variant", AsyncMock(side_effect=lambda plan, state: plan))

    result = await service._route_query_node(base_state(resolved_query=RESOLVED, is_followup=True))

    assert router.await_count == 1
    assert result["query_plan"].actions[0].query == "Kol Nidre piyyut"
    assert result["query_plan"].reasoning == "prose plan"


def test_grounding_filter_keeps_shelf_mark_cited_in_pydantic_history() -> None:
    """Regression: ChatMessage history used to look empty, so cited marks were 'invented'."""
    plan = QueryPlan(
        actions=[
            SearchAction(search_type="primary_shelfmark", query="Rylands Genizah Fragment 1"),
            SearchAction(search_type="primary_shelfmark", query="T-S 13J13.30"),
        ],
        needs_primary_secondary_linking=True,
        reasoning="test",
    )
    state = base_state()

    filtered = AgenticRAGService._drop_ungrounded_shelfmark_actions(plan, state)

    assert [a.query for a in filtered.actions] == ["Rylands Genizah Fragment 1"]
    assert any("T-S 13J13.30" in step for step in state["processing_steps"])


# ---------------------------------------------------------------------------
# Synthesis and finalize
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesis_prompt_carries_prior_turns_and_resolved_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with ChatMessage history the synthesis prompt rendered zero prior turns."""
    service = AgenticRAGService()
    llm = AsyncMock(return_value="Answer")
    monkeypatch.setattr(service, "_call_llm", llm)
    state = base_state(
        resolved_query=RESOLVED,
        is_followup=True,
        bibliography_results=[],
        graph_results=[{
            "scholar": {"name": "Meir Wallenstein", "data_sources": ["biblio"]},
            "works": [], "studied_fragment_count": 0, "studied_fragment_samples": [],
            "relationships": [],
        }],
        synthesis_model_override=None,
        excluded_claims=[],
        retry_count=0,
        error=None,
        error_type=None,
    )

    await service._synthesize_answer_node(state)

    prompt = llm.call_args.kwargs["messages"][0]["content"]
    assert "EARLIER IN THIS CONVERSATION" in prompt
    assert "User: Yom Kippur Piyyut Fragments" in prompt
    assert "Assistant: Scholarly analysis" in prompt
    assert "Works cited" not in prompt
    assert f"USER QUERY:\n{FOLLOWUP}\n(In the context of the conversation, this asks: {RESOLVED})" in prompt


@pytest.mark.asyncio
async def test_finalize_lists_catalog_entries_when_no_scholarship_found() -> None:
    """A shelf-mark hit is shown even when no scholarship on it was retrieved."""
    service = AgenticRAGService()
    state = base_state(
        error_type="NO_RELEVANT_SOURCES",
        draft_answer="I wasn't able to find relevant information about this topic.",
        primary_source_results=[{
            "doc_id": "abc123", "shelf_mark": "Rylands Genizah Fragment 1",
            "title": "Kol Nidre poem", "description": "Piyyut Bible: Texts",
        }],
        shelf_mark_lookup={},
    )

    result = await service._finalize_response_node(state)

    assert result["final_answer"].startswith("I wasn't able to find")
    assert "**Related catalog entries:**" in result["final_answer"]
    assert "[Rylands Genizah Fragment 1](doc:abc123)" in result["final_answer"]


@pytest.mark.asyncio
async def test_chat_returns_resolved_query_and_normalizes_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response exposes the interpretation and the graph receives normalized turns."""
    service = AgenticRAGService()
    captured: Dict[str, Any] = {}

    async def fake_invoke(initial_state: Dict[str, Any]) -> Dict[str, Any]:
        captured["state"] = initial_state
        return {
            **initial_state,
            "resolved_query": RESOLVED,
            "is_followup": True,
            "final_answer": "answer",
            "verified_claims": [],
            "verification_summary": {},
        }

    monkeypatch.setattr(service.graph, "ainvoke", fake_invoke)

    response = await service.chat(FOLLOWUP, conversation_history=chat_message_history())

    assert response.resolved_query == RESOLVED
    assert response.is_followup is True
    assert captured["state"]["conversation_history"] == normalize_conversation_history(
        chat_message_history()
    )


def test_resolved_query_is_reported_only_when_it_differs_from_the_message() -> None:
    """The UI shows the resolved query as 'Understanding:'; echoing the message is noise."""
    assert AgenticRAGService._reported_resolved_query(
        {"resolved_query": RESOLVED}, FOLLOWUP
    ) == RESOLVED
    assert AgenticRAGService._reported_resolved_query(
        {"resolved_query": FOLLOWUP}, FOLLOWUP
    ) is None
    assert AgenticRAGService._reported_resolved_query({}, FOLLOWUP) is None


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eval_runner_sends_case_conversation_history() -> None:
    """Multi-turn cases post their fixed history; single-turn cases post none."""
    bodies: List[Dict[str, Any]] = []

    async def handle_request(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True})

    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
        await run_chat_case(client, "http://backend.test", {"question": "follow-up", "conversation_history": history}, None)
        await run_chat_case(client, "http://backend.test", {"question": "single"}, None)

    assert bodies[0]["conversation_history"] == history
    assert bodies[1]["conversation_history"] is None


def test_eval_resolution_checks_contextualization_and_topic_changes() -> None:
    """`resolution` expectations pass or fail on the response's resolved_query."""
    followup_case = {
        "question": FOLLOWUP,
        "routing": {}, "retrieval": {},
        "resolution": {"must_contain_any": ["kol", "nidr"]},
    }
    topic_change_case = {
        "question": "Tell me about ketubbot",
        "routing": {}, "retrieval": {},
        "resolution": {"must_not_be_rewritten": True},
    }

    prose = {"answer": "A grounded answer with real prose content for the check to accept."}
    good = evaluate_deterministically(followup_case, {**prose, "resolved_query": RESOLVED})
    bad = evaluate_deterministically(followup_case, {**prose, "resolved_query": FOLLOWUP})
    unchanged = evaluate_deterministically(topic_change_case, {**prose, "resolved_query": "Tell me about ketubbot"})
    polluted = evaluate_deterministically(
        topic_change_case, {**prose, "resolved_query": "Tell me about ketubbot studied by Goitein"}
    )
    single_turn = evaluate_deterministically({"question": "x", "routing": {}, "retrieval": {}}, prose)

    assert good["resolution_pass"] and good["overall_pass"]
    assert not bad["resolution_pass"] and not bad["overall_pass"]
    assert unchanged["resolution_pass"]
    assert not polluted["resolution_pass"]
    assert single_turn["resolution_pass"] and not single_turn["resolution"]["applicable"]


def test_multiturn_dataset_is_well_formed_and_separate() -> None:
    """Every multi-turn case has history, a resolution check, and the shared rubric surfaces."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    dataset = load_json(root / "evals" / "agentic_rag_multiturn_v1.json")
    single_turn = load_json(root / "evals" / "agentic_rag_v1.json")

    assert dataset["dataset_id"] != single_turn["dataset_id"]
    assert all("conversation_history" not in case for case in single_turn["cases"])
    ids = [case["id"] for case in dataset["cases"]]
    assert len(ids) == len(set(ids)) >= 4
    for case in dataset["cases"]:
        assert case["conversation_history"], case["id"]
        assert all(turn["role"] in {"user", "assistant"} and turn["content"].strip()
                   for turn in case["conversation_history"])
        assert case["conversation_history"][-1]["role"] == "assistant"
        assert case["resolution"], case["id"]
        assert case["routing"]["must_include_any"]
        assert case["retrieval"]["must_find_any"]
        assert case["answer_rubric"]["must_include"]
        assert case["answer_rubric"]["limitation_behavior"].strip()
    assert any(case["resolution"].get("must_not_be_rewritten") for case in dataset["cases"])
