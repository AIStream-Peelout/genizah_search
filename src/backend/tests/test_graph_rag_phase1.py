"""Focused Phase 1 regression tests for graph-first scholar retrieval."""

import json
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.backend import lms_agentic_search as agent_module
from src.backend.lms_agentic_search import (
    AgenticRAGService,
    ModelUnavailableError,
    QueryPlan,
    SearchAction,
    annotate_answer_with_flags,
    bound_direct_quote,
    build_bibliography_source_context,
    build_verification_sources,
    extract_quoteable_main_text,
    find_quote_source,
    locate_claim_sentence,
    remove_sentences_containing,
    should_retry_verification,
    strip_flag_markers,
)
from src.backend.neo4j_service import Neo4jService
from src.backend.search_bibliography import (
    BibliographyHybridSearchRequest,
    BibliographySearchResponse,
    BibliographySearchResult,
    ElasticsearchBibliographyService,
    embedding_client,
)


@pytest.mark.asyncio
async def test_resolve_scholar_name_from_natural_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefer the complete Estara Arrant Scholar node over a surname-only collision."""
    service = Neo4jService("bolt://unused", "neo4j", "password")
    monkeypatch.setattr(
        service,
        "run_query",
        AsyncMock(return_value=[
            {"name": "Estara J Arrant", "data_sources": ["biblio"]},
            {"name": "Baker Arrant", "data_sources": ["biblio"]},
        ]),
    )

    candidates = await service.resolve_scholars_in_text(
        "Can you summarize Estara J Arrant's work?",
        limit=3,
    )

    assert candidates[0]["name"] == "Estara J Arrant"
    assert candidates[0]["match_score"] > candidates[1]["match_score"]


@pytest.mark.asyncio
async def test_scholar_rag_evidence_is_bounded_and_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return separate profile, work, studied-fragment, and relationship evidence."""
    service = Neo4jService("bolt://unused", "neo4j", "password")
    monkeypatch.setattr(
        service,
        "run_query",
        AsyncMock(side_effect=[
            [{"name": "Estara J Arrant", "data_sources": ["biblio"], "source_books": ["ej_arrant"]}],
            [{
                "article_id": "work-1",
                "title": "Torah Codices",
                "referenced_fragment_count": 160,
                "referenced_fragment_samples": [None, {"shelfmark": "T_S_A25_193"}],
            }],
            [{"studied_fragment_count": 22, "studied_fragment_samples": [{"shelfmark": "T_S_1"}]}],
            [{"relationship": "AFFILIATED_WITH", "labels": ["Institution"], "name": "Cambridge"}],
        ]),
    )

    evidence = await service.get_scholar_rag_evidence("Estara J Arrant")

    assert evidence is not None
    assert evidence["scholar"]["name"] == "Estara J Arrant"
    assert evidence["works"][0]["referenced_fragment_count"] == 160
    assert evidence["works"][0]["referenced_fragment_samples"] == [{"shelfmark": "T_S_A25_193"}]
    assert evidence["studied_fragment_count"] == 22


@pytest.mark.asyncio
async def test_hybrid_keyword_search_uses_mapped_metadata_fields() -> None:
    """Use mapped full text and author/title metadata while preserving lexical rank."""
    service = ElasticsearchBibliographyService.__new__(ElasticsearchBibliographyService)
    service.index_name = "bibliography-test"
    service.es = MagicMock()
    service.es.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "one", "_score": 20.0, "_source": {"doc_id": "one", "author": "Estara Arrant"}},
                {"_id": "two", "_score": 5.0, "_source": {"doc_id": "two", "author": "Other Scholar"}},
            ]
        }
    }

    response = await service.search_hybrid(BibliographyHybridSearchRequest(
        query="Estara Arrant",
        semanticWeight=0,
        keywordWeight=100,
        num_results=2,
    ))

    query = service.es.search.call_args.kwargs["query"]
    query_text = str(query)
    assert "full_text_content^2.5" in query_text
    assert "author^6.0" in query_text
    assert "title^4.0" in query_text
    assert response.results[0].doc_id == "one"
    assert response.results[0].retrieval_details["keyword_rank"] == 1


@pytest.mark.asyncio
async def test_author_search_removes_middle_initial_for_exact_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map the graph name ``Estara J Arrant`` to indexed author ``Estara Arrant``."""
    service = ElasticsearchBibliographyService.__new__(ElasticsearchBibliographyService)
    service.index_name = "bibliography-test"
    service.es = MagicMock()
    service.es.search.return_value = {
        "hits": {
            "total": {"value": 139, "relation": "eq"},
            "hits": [{
                "_id": "ej_arrant_p073",
                "_score": 3.0,
                "_source": {
                    "doc_id": "ej_arrant_p073",
                    "author": "Estara Arrant",
                    "authors": ["Estara Arrant"],
                    "title": "A Codicological and Linguistic Typology of Common Torah Codices",
                    "full_text_content": "Relevant source text.",
                    "extracted_page_number": 73,
                },
            }],
        }
    }
    monkeypatch.setattr(
        embedding_client,
        "get_embedding",
        AsyncMock(side_effect=RuntimeError("embedding unavailable")),
    )

    response = await service.search_by_author(
        "Estara J Arrant",
        "Summarize Estara J Arrant's work",
        num_results=5,
    )

    query = service.es.search.call_args.kwargs["query"]
    assert "Estara Arrant" in str(query)
    assert response.total == 139
    assert response.results[0].author == "Estara Arrant"
    assert response.results[0].retrieval_details["mode"] == "author_constrained_lexical_fallback"


@pytest.mark.asyncio
async def test_router_uses_planner_graph_action_for_named_scholar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honor a planner-selected graph action for a specific human scholar."""
    service = AgenticRAGService()
    router_call = AsyncMock(return_value={
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "create_search_plan",
                        "arguments": (
                            '{"actions": [{"search_type": "graph_scholar", '
                            '"query": "Estara J Arrant", "num_results": 1}], '
                            '"needs_primary_secondary_linking": true, '
                            '"is_followup": true, '
                            '"reasoning": "The query names a specific scholar"}'
                        ),
                    },
                }],
            },
        }],
    })
    monkeypatch.setattr(service, "_call_llm_with_tools", router_call)
    state = {
        "user_query": "Summarize Estara J Arrant's work",
        "conversation_history": [{"role": "user", "content": "Focus on her methodology."}],
        "resolved_entities": [],
        "processing_steps": [],
    }

    result = await service._route_query_node(state)

    assert result["query_plan"].actions[0].search_type == "graph_scholar"
    assert result["query_plan"].actions[0].query == "Estara J Arrant"
    assert result["query_plan"].is_followup is True
    assert result["resolved_entities"] == []
    assert router_call.call_args.kwargs["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_router_normalizes_direct_graph_scholar_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap a direct planner action in a complete ``QueryPlan``."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service,
        "_call_llm_with_tools",
        AsyncMock(return_value={
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "graph_scholar",
                            "arguments": '{"query": "S.D. Goitein"}',
                        },
                    }],
                },
            }],
        }),
    )
    state = {
        "user_query": "What did S.D. Goitein write about?",
        "conversation_history": [],
        "resolved_entities": [],
        "processing_steps": [],
    }

    result = await service._route_query_node(state)

    plan = result["query_plan"]
    assert isinstance(plan, QueryPlan)
    assert plan.actions == [SearchAction(search_type="graph_scholar", query="S.D. Goitein")]
    assert "Normalized the planner's direct graph_scholar action" in plan.reasoning


@pytest.mark.asyncio
async def test_goitein_alias_queries_only_the_canonical_graph_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand the common Goitein abbreviation without changing graph data."""
    service = Neo4jService("bolt://unused", "neo4j", "password")
    query = AsyncMock(return_value=[{
        "name": "Shelomo Dov Goitein",
        "data_sources": ["biblio"],
    }])
    monkeypatch.setattr(service, "run_query", query)

    candidates = await service.find_scholars("S.D. Goitein")

    assert candidates[0]["name"] == "Shelomo Dov Goitein"
    assert query.await_args.args[1] == {"tokens": ["shelomo", "dov", "goitein"]}


@pytest.mark.asyncio
async def test_router_keeps_genizah_topic_query_out_of_neo4j(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honor hybrid retrieval for ketubbot instead of pre-resolving a false scholar."""
    service = AgenticRAGService()
    resolver = AsyncMock(return_value=[{
        "name": "Cairo Genizah",
        "data_sources": ["biblio"],
        "match_score": 1.0,
    }])
    monkeypatch.setattr(agent_module.neo4j_service, "resolve_scholars_in_text", resolver)
    router_call = AsyncMock(return_value={
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "create_search_plan",
                        "arguments": (
                            '{"actions": [{"search_type": "bibliography_hybrid", '
                            '"query": "ketubbot Cairo Genizah", "semantic_weight": 30, '
                            '"keyword_weight": 70, "num_results": 8}], '
                            '"needs_primary_secondary_linking": true, '
                            '"reasoning": "Ketubbot are a topic, not a scholar"}'
                        ),
                    },
                }],
            },
        }],
    })
    monkeypatch.setattr(service, "_call_llm_with_tools", router_call)
    state = {
        "user_query": "Tell me about ketubbot in the Cairo Genizah",
        "conversation_history": [],
        "resolved_entities": [],
        "processing_steps": [],
    }

    result = await service._route_query_node(state)

    assert result["query_plan"].actions[0].search_type == "bibliography_hybrid"
    assert all(
        action.search_type != "graph_scholar"
        for action in result["query_plan"].actions
    )
    resolver.assert_not_awaited()
    router_prompt = router_call.call_args.kwargs["messages"][0]["content"]
    assert '"Cairo Genizah" is the collection/context, not an' in router_prompt


@pytest.mark.asyncio
async def test_router_falls_back_when_tool_arguments_are_not_a_query_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not crash when a tool model emits only ``query`` plan arguments."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service,
        "_call_llm_with_tools",
        AsyncMock(return_value={
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "create_search_plan",
                            "arguments": '{"query": "Estara Arrant"}',
                        },
                    }],
                },
            }],
        }),
    )
    state = {
        "user_query": "Estara Arrant",
        "conversation_history": [],
        "resolved_entities": [],
        "processing_steps": [],
    }

    result = await service._route_query_node(state)

    assert result["query_plan"].actions[0].search_type == "bibliography_hybrid"
    assert "Fallback" in result["query_plan"].reasoning


@pytest.mark.asyncio
async def test_graph_action_drives_author_constrained_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute Neo4j first and retain graph evidence alongside author text."""
    service = AgenticRAGService()
    evidence = {
        "scholar": {"name": "Estara J Arrant", "data_sources": ["biblio"]},
        "works": [{"article_id": "work-1", "title": "Torah Codices"}],
        "studied_fragment_count": 22,
        "studied_fragment_samples": [],
        "relationships": [],
    }
    monkeypatch.setattr(
        agent_module.neo4j_service,
        "find_scholars",
        AsyncMock(return_value=[{"name": "Estara J Arrant", "match_score": 1.0}]),
    )
    monkeypatch.setattr(
        agent_module.neo4j_service,
        "get_scholar_rag_evidence",
        AsyncMock(return_value=evidence),
    )
    author_response = BibliographySearchResponse(
        results=[BibliographySearchResult(
            doc_id="ej_arrant_p073",
            similarity_score=0.8,
            author="Estara Arrant",
            authors=["Estara Arrant"],
            title="Torah Codices",
            full_text="Relevant source text.",
            extracted_page_number=73,
        )],
        count=1,
        processing_time_ms=1.0,
        total=1,
    )
    monkeypatch.setattr(
        agent_module.bibliography_search_service,
        "search_by_author",
        AsyncMock(return_value=author_response),
    )
    state = {
        "user_query": "Summarize Estara J Arrant's work",
        "query_plan": QueryPlan(
            actions=[SearchAction(search_type="graph_scholar", query="Estara J Arrant")],
            needs_primary_secondary_linking=False,
            reasoning="Named scholar",
        ),
        "bibliography_results": [],
        "primary_source_results": [],
        "graph_results": [],
        "resolved_entities": [],
        "shelf_marks_in_bibliography": set(),
        "shelf_marks_from_search": set(),
        "shelf_mark_lookup": {},
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._execute_searches_node(state)

    assert result["graph_results"][0]["scholar"]["name"] == "Estara J Arrant"
    assert result["bibliography_results"][0]["author"] == "Estara Arrant"
    assert result["error_type"] is None


@pytest.mark.asyncio
async def test_synthesis_receives_graph_evidence_with_provenance_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass graph-only evidence to synthesis without presenting it as prose scholarship."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value="Graph-grounded answer")
    monkeypatch.setattr(service, "_call_llm", llm_call)
    state = {
        "user_query": "What works are associated with Estara J Arrant?",
        "bibliography_results": [],
        "graph_results": [{
            "scholar": {"name": "Estara J Arrant", "data_sources": ["biblio"]},
            "works": [{
                "article_id": "work-1",
                "title": "Torah Codices",
                "year": "2021",
                "referenced_fragment_count": 160,
                "referenced_fragment_samples": [],
            }],
            "studied_fragment_count": 22,
            "studied_fragment_samples": [],
            "relationships": [],
        }],
        "synthesis_model_override": None,
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._synthesize_answer_node(state)

    prompt = llm_call.call_args.kwargs["messages"][0]["content"]
    assert "NEO4J GRAPH EVIDENCE" in prompt
    assert "Torah Codices" in prompt
    assert "structured catalog and relationship metadata" in prompt
    assert result["draft_answer"] == "Graph-grounded answer"


def test_synthesis_and_verification_share_identical_bibliography_evidence() -> None:
    """Keep descriptions visible to both stages instead of truncating verifier input."""
    bibliography = [{
        "authors": ["Estara Arrant"],
        "title": "Torah Codices",
        "extracted_page_number": 73,
        "full_text": "A" * 1400,
        "description": "Distinct attribution evidence near the end of the summary.",
        "shelf_marks_mentioned": ["T-S 8.133"],
    }]

    synthesis_source = build_bibliography_source_context(bibliography)[0]
    verification_source = build_verification_sources(bibliography, [])[0]

    assert synthesis_source["prompt_text"] == verification_source["prompt_text"]
    assert "Distinct attribution evidence" in verification_source["prompt_text"]


def test_only_main_text_is_quoteable_bibliography_evidence() -> None:
    """Keep generated summaries visible for orientation but ineligible as quotes."""
    generated = "This generated summary claims a remarkable seventh-century origin."
    original = "The manuscript preserves a distinctive piyyut for the Day of Atonement."
    bibliography = [{
        "authors": ["Meir Wallenstein"],
        "title": "A Unique Kol-Nidre Piyyut",
        "extracted_page_number": 489,
        "full_text": f"Metadata and generated prose. Main text: {original}",
        "description": generated,
        "shelf_marks_mentioned": [],
    }]

    source = build_bibliography_source_context(bibliography)[0]

    assert extract_quoteable_main_text(bibliography[0]["full_text"]) == original
    assert source["quoteable_text"] == original
    assert "orientation only; never quote" in source["prompt_text"]
    assert find_quote_source(generated, [source]) is None
    assert find_quote_source(original, [source]) == 1


@pytest.mark.asyncio
async def test_verifier_checks_quotes_deterministically_and_claim_propositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept an exact quote and soft-flag an unsupported attributed proposition."""
    service = AgenticRAGService()
    # The verifier's own quote hard gate is the QUOTE_FAILURE_MODE=repair path.
    service.quote_failure_mode = "repair"
    monkeypatch.setattr(
        service,
        "_chat_completion",
        AsyncMock(return_value=((
            '{"verified_claims": [{"type": "attribution", '
            '"text": "Arrant argues that this codex was copied in 1100", '
            '"verdict": "unsupported", "source_number": null, '
            '"reasoning": "SOURCE 1 names Arrant but gives no such date"}], '
            '"summary": "Unsupported attribution"}'
        ), "stop")),
    )
    state = {
        "draft_answer": (
            'Arrant writes that "the leaves preserve several scribal corrections." '
            "She argues that this codex was copied in 1100."
        ),
        "bibliography_results": [{
            "authors": ["Estara Arrant"],
            "title": "Torah Codices",
            "extracted_page_number": 73,
            "full_text": "The leaves preserve several scribal corrections.",
            "description": "",
            "shelf_marks_mentioned": [],
        }],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    # An unsupported-but-not-contradicted claim no longer triggers repair; it
    # is kept in the answer and flagged for the user instead.
    assert result["verification_summary"] == {
        "SUPPORTED": 1,
        "NOT_SUPPORTED": 1,
        "CONTRADICTED_OR_FABRICATED": 0,
    }
    assert result["error_type"] is None
    assert result["excluded_claims"] == []
    assert result["soft_flagged_claims"][0]["type"] == "attribution"
    assert "no such date" in result["soft_flagged_claims"][0]["reason"]


@pytest.mark.asyncio
async def test_hallucinated_citation_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'supported' verdict citing a source that never mentions the claim's
    subject is downgraded to a user-visible flag, not published as verified."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service,
        "_chat_completion",
        AsyncMock(return_value=((
            '{"verified_claims": ['
            '{"type": "factual_claim", '
            '"text": "Maimonides settled in Fustat around 1168", '
            '"verdict": "supported", "source_number": 1, '
            '"reasoning": "Supported by SOURCE 1"},'
            '{"type": "factual_claim", '
            '"text": "The piyyut manuscript shows two distinct pens", '
            '"verdict": "supported", "source_number": 1, '
            '"reasoning": "Supported by SOURCE 1"}'
            '], "summary": "two claims"}'
        ), "stop")),
    )
    state = {
        "draft_answer": (
            "Maimonides settled in Fustat around 1168. "
            "The piyyut manuscript shows two distinct pens."
        ),
        "bibliography_results": [{
            "authors": ["Meir Wallenstein"],
            "title": "A Unique Kol-nidre Piyyut",
            "extracted_page_number": 489,
            "full_text": "Main text: The piyyut manuscript betrays two distinct pens.",
            "description": "",
            "shelf_marks_mentioned": [],
        }],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    flagged_texts = [f["text"] for f in result["soft_flagged_claims"]]
    # The Maimonides claim shares no distinctive term with the cited source.
    assert any("Maimonides" in text for text in flagged_texts)
    # The genuinely supported claim (terms overlap: piyyut, pens…) stays green.
    assert all("distinct pens" not in text for text in flagged_texts)
    statuses = {c.claim: c.verification_status for c in result["verified_claims"]}
    assert any(
        status == "SUPPORTED" for claim, status in statuses.items() if "pens" in claim
    )


@pytest.mark.asyncio
async def test_accurate_limitation_statements_verify_as_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'The sources do not mention X' is supported when X is truly absent —
    and contradicted (hard) when a source plainly contains X."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service,
        "_chat_completion",
        AsyncMock(return_value=((
            '{"verified_claims": ['
            '{"type": "evidence_limitation", '
            '"text": "The sources do not identify the scribe of the manuscript", '
            '"verdict": "supported", "source_number": null, '
            '"reasoning": "No source names a scribe"},'
            '{"type": "evidence_limitation", '
            '"text": "The sources do not suggest a date for the manuscript", '
            '"verdict": "contradicted", "source_number": 1, '
            '"reasoning": "SOURCE 1 suggests the end of the eleventh century"}'
            '], "summary": "One accurate limitation, one false one"}'
        ), "stop")),
    )
    state = {
        "draft_answer": (
            "The sources do not identify the scribe of the manuscript. "
            "The sources do not suggest a date for the manuscript."
        ),
        "bibliography_results": [{
            "authors": ["Meir Wallenstein"],
            "title": "A Unique Kol-nidre Piyyut",
            "extracted_page_number": 489,
            "full_text": "Main text: Perhaps the end of the eleventh century.",
            "description": "",
            "shelf_marks_mentioned": [],
        }],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    # The accurate absence note is SUPPORTED — never a red flag.
    assert result["soft_flagged_claims"] == []
    statuses = {c.claim: c.verification_status for c in result["verified_claims"]}
    assert any(
        status == "SUPPORTED" for claim, status in statuses.items() if "scribe" in claim
    )
    # The false absence note is a hard failure and goes to repair.
    assert result["error_type"] == "FABRICATED_CLAIMS"
    assert "date" in result["excluded_claims"][0]["text"]
    # Schema advertises the new claim type to the structured-output grammar.
    import json as _json
    assert "evidence_limitation" in _json.dumps(
        AgenticRAGService._VERIFIER_RESPONSE_FORMAT
    )


@pytest.mark.asyncio
async def test_contradicted_claim_triggers_targeted_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim the sources contradict is a hard failure that forces repair."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service,
        "_chat_completion",
        AsyncMock(return_value=((
            '{"verified_claims": [{"type": "factual_claim", '
            '"text": "The piyyut was composed for Passover", '
            '"verdict": "contradicted", "source_number": null, '
            '"reasoning": "SOURCE 1 identifies the piyyut as a Kol Nidre composition"}], '
            '"summary": "One contradicted claim"}'
        ), "stop")),
    )
    state = {
        "draft_answer": "The piyyut was composed for Passover.",
        "bibliography_results": [{
            "authors": ["Meir Wallenstein"],
            "title": "A Unique Kol-Nidre Piyyut",
            "extracted_page_number": 489,
            "full_text": "Main text: This Kol Nidre piyyut is unique.",
            "description": "",
            "shelf_marks_mentioned": [],
        }],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    assert result["error_type"] == "FABRICATED_CLAIMS"
    assert result["excluded_claims"][0]["type"] == "factual_claim"
    assert result["verification_feedback_history"][0]["rejected_claims"] == result["excluded_claims"]
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_prior_rejections_are_advisory_not_a_blacklist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repaired claim must be judged on current source evidence, not history."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value=('{"verified_claims": [], "summary": "ok"}', "stop"))
    monkeypatch.setattr(service, "_chat_completion", llm_call)
    state = {
        "draft_answer": "Friedman corrected the earlier claim.",
        "bibliography_results": [],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 1,
        "processing_steps": [],
        "error": None,
        "error_type": None,
        "verification_feedback_history": [{
            "attempt": 1,
            "summary": "one rejection",
            "rejected_claims": [{
                "type": "quote",
                "text": "The basic similarity misled students",
                "reason": "misquote",
            }],
            "supported_claims": [],
        }],
    }

    await service._verify_claims_node(state)

    prompt = llm_call.call_args.kwargs["messages"][0]["content"]
    # Prior rejections are context that source evidence always outranks…
    assert "NEVER overrides current source evidence" in prompt
    assert "The basic similarity misled students" in prompt
    # …and the old permanent-blacklist instruction must not return.
    assert "mark the complete proposition NOT_SUPPORTED" not in prompt


@pytest.mark.asyncio
async def test_verifier_invalid_json_retries_then_flags_visibly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry invalid structured output once, then flag rather than destroy."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value=("not json", "stop"))
    monkeypatch.setattr(service, "_chat_completion", llm_call)
    state = {
        "draft_answer": "Arrant established an unsupported conclusion.",
        "bibliography_results": [],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    assert llm_call.call_count == 2
    assert result["error_type"] is None
    assert result["verification_summary"]["parse_error"] == 1
    assert result["soft_flagged_claims"][0]["type"] == "verification_error"


_OUTAGE_DRAFT = (
    'Arrant writes that "the leaves preserve several scribal corrections." '
    "She argues that this codex was copied in 1100."
)


def _outage_bibliography() -> List[Dict[str, Any]]:
    """Build one bibliography result whose main text holds the draft's quote.

    :returns: Bibliography results for the verifier-outage tests.
    :rtype: List[Dict[str, Any]]
    """
    return [{
        "authors": ["Estara Arrant"],
        "title": "Torah Codices",
        "extracted_page_number": 73,
        "full_text": "The leaves preserve several scribal corrections.",
        "description": "",
        "shelf_marks_mentioned": [],
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outage",
    [ModelUnavailableError("verifier unloaded"), httpx.ConnectError("connection refused")],
    ids=["model_unavailable", "connect_error"],
)
async def test_case21_verifier_outage_keeps_draft_and_flags_it(
    monkeypatch: pytest.MonkeyPatch,
    outage: Exception,
) -> None:
    """Case 21 (C5.1): a verifier outage keeps the draft, sources and the
    deterministic quote result under a visible verification_error flag."""
    service = AgenticRAGService()
    verifier = AsyncMock(side_effect=outage)
    monkeypatch.setattr(service, "_call_verifier", verifier)
    bibliography = _outage_bibliography()
    state = {
        "draft_answer": _OUTAGE_DRAFT,
        "bibliography_results": bibliography,
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    verifier.assert_awaited_once()
    assert result["draft_answer"] == _OUTAGE_DRAFT
    assert result["bibliography_results"] == bibliography
    assert result["verification_unavailable"] is True
    assert result["error_type"] is None
    assert result["excluded_claims"] == []
    assert result["soft_flagged_claims"] == [{
        "type": "verification_error",
        "text": "Automatic claim verification did not complete for this answer",
        "reason": "The verification model was unavailable; this answer could not be double-checked.",
    }]
    # The deterministic quote check still ran and verified the exact quote.
    assert result["verification_summary"]["SUPPORTED"] == 1


@pytest.mark.asyncio
async def test_case21_verifier_bug_still_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case 21 (C5.1): only model outages are absorbed; other errors still raise."""
    service = AgenticRAGService()
    monkeypatch.setattr(service, "_call_verifier", AsyncMock(side_effect=RuntimeError("bug")))
    state = {
        "draft_answer": _OUTAGE_DRAFT,
        "bibliography_results": _outage_bibliography(),
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    with pytest.raises(RuntimeError, match="bug"):
        await service._verify_claims_node(state)


@pytest.mark.asyncio
async def test_case21_chat_returns_draft_when_verifier_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 21 (C5.1) end to end: /chat returns the draft, its sources and the
    verification_error flag instead of failing when the verifier is down."""
    service = AgenticRAGService()
    bibliography = _outage_bibliography()

    async def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
        """Publish the draft as the final answer (no Neo4j/ES access).

        :param state: Pipeline state.
        :returns: State with final_answer set.
        :rtype: Dict[str, Any]
        """
        state["final_answer"] = state["draft_answer"]
        return state

    _patch_pipeline(monkeypatch, service, bibliography, _OUTAGE_DRAFT, finalize=finalize)
    verifier = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    monkeypatch.setattr(service, "_call_verifier", verifier)

    response = await service.chat("What does Arrant say about the leaves?")

    verifier.assert_awaited_once()
    assert response.success is True
    assert response.error_type is None
    assert response.answer == _OUTAGE_DRAFT
    assert response.bibliography_results == bibliography
    assert [flag.claim_type for flag in response.flagged_claims] == ["verification_error"]
    assert response.flagged_claims[0].reason == agent_module.VERIFICATION_UNAVAILABLE_REASON


@pytest.mark.asyncio
async def test_case22_truncated_verifier_output_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 22 (C5.4): finish_reason=length makes one verifier attempt, then
    the parse error becomes the visible verification_error flag."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value=('{"verified_claims": [{"type": "attribution", ', "length"))
    monkeypatch.setattr(service, "_chat_completion", llm_call)
    state = {
        "draft_answer": "Arrant established an unsupported conclusion.",
        "bibliography_results": [],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    assert llm_call.await_count == 1
    assert result["error_type"] is None
    assert result["verification_summary"]["parse_error"] == 1
    flag = result["soft_flagged_claims"][0]
    assert flag["type"] == "verification_error"
    assert "finish_reason=length" in flag["reason"]


@pytest.mark.asyncio
async def test_retry_prompt_lists_unsupported_attributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tell synthesis to remove an unsupported attribution on its retry."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value="Revised answer")
    monkeypatch.setattr(service, "_call_llm", llm_call)
    state = {
        "user_query": "Summarize Arrant's work",
        "bibliography_results": [],
        "graph_results": [],
        "synthesis_model_override": None,
        "excluded_claims": [{
            "type": "attribution",
            "text": "Arrant dates the codex to 1100",
            "reason": "No source supports this date",
        }],
        "retry_count": 1,
        "processing_steps": [],
        "error": "unsupported",
        "error_type": "FABRICATED_CLAIMS",
    }

    await service._synthesize_answer_node(state)

    prompt = llm_call.call_args.kwargs["messages"][0]["content"]
    assert "Arrant dates the codex to 1100" in prompt
    assert "Remove each item or rewrite it as a narrower claim" in prompt


@pytest.mark.asyncio
async def test_span_repair_rewrites_only_the_offending_paragraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revise the paragraph with the rejected claim; keep other paragraphs byte-identical."""
    service = AgenticRAGService()
    good_paragraph = (
        'Wallenstein writes that "the manuscript preserves a distinctive piyyut." '
        "(Wallenstein, p. 489)"
    )
    bad_paragraph = (
        "The piyyut has eighteen verses. He dates it to the seventh century."
    )
    revised_paragraph = "The piyyut has eighteen verses."
    llm_call = AsyncMock(return_value=(
        '{"revisions": [{"paragraph_index": 1, '
        f'"revised_text": "{revised_paragraph}"' '}]}'
    ))
    monkeypatch.setattr(service, "_call_llm", llm_call)
    state = {
        "draft_answer": f"{good_paragraph}\n\n{bad_paragraph}",
        "bibliography_results": [{
            "authors": ["Meir Wallenstein"],
            "title": "A Unique Kol-Nidre Piyyut",
            "extracted_page_number": 489,
            "full_text": (
                "Generated header. Main text: The manuscript preserves a distinctive piyyut."
            ),
            "description": "Generated catalog description.",
            "shelf_marks_mentioned": [],
        }],
        "graph_results": [],
        "synthesis_model_override": None,
        "excluded_claims": [{
            "type": "factual_claim",
            "text": "He dates it to the seventh century",
            "reason": "No numbered source supports this date.",
        }],
        "supported_evidence_units": [],
        "verification_feedback_history": [],
        "soft_flagged_claims": [],
        "retry_count": 1,
        "processing_steps": [],
        "error": "Unsupported date",
        "error_type": "FABRICATED_CLAIMS",
    }

    result = await service._repair_answer_node(state)

    prompt = llm_call.call_args.kwargs["messages"][0]["content"]
    assert "PARAGRAPH 1:" in prompt
    assert "PARAGRAPH 0:" not in prompt
    assert "He dates it to the seventh century" in prompt
    assert '"the manuscript preserves a distinctive piyyut."' in prompt
    assert llm_call.call_args.kwargs["temperature"] == 0.0
    # The verified paragraph is untouched; only the offending one changed.
    assert result["draft_answer"] == f"{good_paragraph}\n\n{revised_paragraph}"
    assert result["error_type"] is None
    # A repair that changed the draft goes back through the quote check.
    assert result["repair_noop"] is False
    assert agent_module.route_after_repair(result) == "check"


@pytest.mark.asyncio
async def test_span_repair_falls_back_to_deterministic_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delete the rejected sentence deterministically when the repair call fails."""
    service = AgenticRAGService()
    llm_call = AsyncMock(side_effect=RuntimeError("LM Studio unavailable"))
    monkeypatch.setattr(service, "_call_llm", llm_call)
    state = {
        "draft_answer": (
            "The piyyut has eighteen verses. He dates it to the seventh century."
        ),
        "bibliography_results": [],
        "graph_results": [],
        "synthesis_model_override": None,
        "excluded_claims": [{
            "type": "factual_claim",
            "text": "He dates it to the seventh century",
            "reason": "No numbered source supports this date.",
        }],
        "supported_evidence_units": [],
        "verification_feedback_history": [],
        "soft_flagged_claims": [],
        "retry_count": 1,
        "processing_steps": [],
        "error": "Unsupported date",
        "error_type": "FABRICATED_CLAIMS",
    }

    result = await service._repair_answer_node(state)

    assert "seventh century" not in result["draft_answer"]
    assert "eighteen verses" in result["draft_answer"]
    assert result["error_type"] is None


def test_locate_claim_sentence_matches_paraphrased_claims() -> None:
    """Anchor a verifier claim to the sentence expressing it, or return None."""
    answer = (
        "The piyyut has eighteen verses. Wallenstein dates the manuscript to the "
        "end of the eleventh century. It was found in the Gaster collection."
    )
    located = locate_claim_sentence(
        "The manuscript is dated to the end of the eleventh century", answer
    )
    assert located == (
        "Wallenstein dates the manuscript to the end of the eleventh century."
    )
    assert locate_claim_sentence("Maimonides wrote responsa in Fustat", answer) is None


def test_annotate_answer_with_flags_wraps_sentences_with_markers() -> None:
    """Wrap each anchored flag in ⟦flag:N⟧ markers and report unanchored flags."""
    answer = (
        "The piyyut has eighteen verses. The scribe likely worked in Fustat.\n\n"
        "A second paragraph follows."
    )
    flags = [
        {"type": "factual_claim", "text": "The scribe likely worked in Fustat", "reason": "No source"},
        {"type": "attribution", "text": "Goitein rejected the attribution", "reason": "Not found"},
    ]

    annotated, enriched = annotate_answer_with_flags(answer, flags)

    assert "⟦flag:1⟧The scribe likely worked in Fustat.⟦/flag⟧" in annotated
    assert enriched[0]["flag_id"] == 1
    assert enriched[0]["answer_span"] == "The scribe likely worked in Fustat."
    assert enriched[1]["answer_span"] is None
    assert strip_flag_markers(annotated) == answer


def test_topical_terms_exclude_collection_context() -> None:
    """Collection words never count as the subject of a query."""
    assert agent_module.extract_topical_terms(
        "Can you tell me about zemirot in the Cairo Genizah?"
    ) == ["zemirot"]
    # A query that is only collection context has no distinctive subject.
    assert agent_module.extract_topical_terms("Tell me about Genizah fragments") == []


def test_relevance_gate_detects_subject_absent_from_corpus() -> None:
    """Pages matching only 'Genizah' must not count as answering the query."""
    off_topic = [
        {
            "title": "A Cosmopolitan City: Muslims, Christians, and Jews in Old Cairo",
            "full_text": "Main text: The Genizah documents include medical glossaries.",
            "description": "",
        },
        {
            "title": "From Cairo to Manchester",
            "full_text": "Main text: Genizah fragments of magical texts and dowry lists.",
            "description": "",
        },
    ]
    addresses, terms = agent_module.evidence_addresses_query(
        "zemirot bentching in the Cairo Genizah", off_topic
    )
    assert addresses is False
    assert "zemirot" in terms


def test_relevance_gate_accepts_inflected_and_partial_subject_matches() -> None:
    """Real topical coverage must pass, including inflected forms."""
    on_topic = [{
        "title": "Jewish Marriage in Palestine: The Kettubba texts",
        "full_text": "Main text: Palestinian ketubbot differ from the Babylonian formulary.",
        "description": "",
    }]
    addresses, _ = agent_module.evidence_addresses_query(
        "Tell me about ketubba traditions in the Cairo Genizah", on_topic
    )
    assert addresses is True

    # Subject named in the title only, and via subject keywords.
    addresses_title, _ = agent_module.evidence_addresses_query(
        "What do scholars say about marriage contracts?",
        [{"title": "Jewish Marriage in Palestine", "full_text": "", "description": ""}],
    )
    assert addresses_title is True
    addresses_keywords, _ = agent_module.evidence_addresses_query(
        "Genizah piyyutim",
        [{
            "title": "Untitled study",
            "full_text": "",
            "description": "",
            "subject_keywords": ["piyyut", "liturgy"],
        }],
    )
    assert addresses_keywords is True


@pytest.mark.asyncio
async def test_model_resolution_uses_resident_instance_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LM Studio names extra instances 'model:2'; the bare id is 'not loaded'."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service, "_loaded_model_ids",
        AsyncMock(return_value=["qwen/qwen3.6-35b-a3b:2", "qwen3-vl-8b-heb-v17-step800"]),
    )

    resolved = await service.resolve_model("qwen/qwen3.6-35b-a3b")

    assert resolved == "qwen/qwen3.6-35b-a3b:2"


@pytest.mark.asyncio
async def test_model_resolution_refuses_cross_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable configured model must never use another resident model."""
    service = AgenticRAGService()
    service.synthesis_model = "qwen/qwen3.6-35b-a3b"
    monkeypatch.setattr(
        service, "_loaded_model_ids",
        AsyncMock(return_value=["qwen3-vl-8b-heb-v17-step800", "qwen/qwen3.6-35b-a3b:2"]),
    )
    load_model = AsyncMock(side_effect=RuntimeError("not loaded"))
    monkeypatch.setattr(service, "load_model", load_model)

    with pytest.raises(ModelUnavailableError):
        await service.resolve_model("qwen/qwen3-4b-2507")

    load_model.assert_awaited_once_with("qwen/qwen3-4b-2507")


@pytest.mark.asyncio
async def test_model_resolution_loads_configured_model_when_nothing_is_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty LM Studio process explicitly loads the configured model."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service, "_loaded_model_ids",
        AsyncMock(return_value=[]),
    )
    load_model = AsyncMock(return_value=None)
    monkeypatch.setattr(service, "load_model", load_model)

    resolved = await service.resolve_model("qwen/qwen3-4b-2507")

    assert resolved == "qwen/qwen3-4b-2507"
    load_model.assert_awaited_once_with("qwen/qwen3-4b-2507")


@pytest.mark.asyncio
async def test_model_resolution_never_jit_loads_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C5.3: with LM_STUDIO_ALLOW_JIT_LOAD=0 a non-resident model fails fast
    instead of being loaded inside the request; resident instances still resolve."""
    monkeypatch.setenv("LM_STUDIO_ALLOW_JIT_LOAD", "0")
    service = AgenticRAGService()
    assert service.allow_jit_load is False
    loaded_ids = AsyncMock(return_value=[])
    monkeypatch.setattr(service, "_loaded_model_ids", loaded_ids)
    load_model = AsyncMock(return_value=None)
    monkeypatch.setattr(service, "load_model", load_model)

    with pytest.raises(ModelUnavailableError) as excinfo:
        await service.resolve_model("qwen/qwen3.6-27b")

    load_model.assert_not_awaited()
    assert "qwen/qwen3.6-27b" in str(excinfo.value.__cause__)
    assert "LM_STUDIO_ALLOW_JIT_LOAD" in str(excinfo.value.__cause__)

    loaded_ids.return_value = ["qwen/qwen3.6-27b:2"]
    assert await service.resolve_model("qwen/qwen3.6-27b") == "qwen/qwen3.6-27b:2"
    load_model.assert_not_awaited()


def test_jit_load_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5.3: an unset LM_STUDIO_ALLOW_JIT_LOAD keeps today's JIT behaviour."""
    monkeypatch.delenv("LM_STUDIO_ALLOW_JIT_LOAD", raising=False)

    assert AgenticRAGService().allow_jit_load is True


@pytest.mark.asyncio
async def test_chat_completion_reports_finish_reason_and_effort_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C5.4: _chat_completion returns (content, finish_reason), sends the
    per-call effort override to any model and the global effort only to the
    synthesis/verification models; _call_llm still returns plain content."""
    service = AgenticRAGService()
    service.router_model = "router-m"
    service.synthesis_model = "synth-m"
    service.verification_model = "verify-m"
    service.reasoning_effort = "low"
    monkeypatch.setattr(
        service, "resolve_model",
        AsyncMock(side_effect=lambda model, force_refresh=False: model),
    )
    payloads: List[Dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """Answer every chat completion as truncated.

        :param request: Outgoing request to LM Studio.
        :returns: A canned chat-completion response.
        :rtype: httpx.Response
        """
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        agent_module.httpx, "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=httpx.MockTransport(handle), **kwargs),
    )
    messages = [{"role": "user", "content": "hi"}]

    assert await service._chat_completion(messages=messages, model="verify-m") == ("partial", "length")
    await service._chat_completion(messages=messages, model="router-m")
    await service._chat_completion(messages=messages, model="router-m", reasoning_effort="none")
    assert await service._call_llm(messages=messages, model="synth-m") == "partial"

    assert [payload.get("reasoning_effort") for payload in payloads] == ["low", None, "none", "low"]


def test_model_unavailable_error_is_recognised() -> None:
    """LM Studio's unloaded-model 400 must be treated as recoverable."""
    class _Response:
        def __init__(self, status_code: int, text: str) -> None:
            self.status_code = status_code
            self.text = text

    assert agent_module._is_model_unavailable_error(
        _Response(400, '{"error":"Model has not started loading/has been unloaded."}')
    )
    assert not agent_module._is_model_unavailable_error(_Response(400, '{"error":"context length"}'))
    assert not agent_module._is_model_unavailable_error(_Response(500, "server error"))


def test_sentence_split_keeps_citations_and_abbreviations_intact() -> None:
    """Flag spans must not cut a sentence at 'p. 45' or similar abbreviations."""
    text = (
        'The text concludes with a blessing: Levin, p. 45: "barukh atah". '
        "A separate sentence follows."
    )
    sentences = agent_module.split_sentences(text)

    assert len(sentences) == 2
    assert sentences[0].endswith('"barukh atah".')
    assert "p. 45" in sentences[0]
    # Sentences stay verbatim substrings so span wrapping can locate them.
    for sentence in sentences:
        assert sentence in text

    # Multi-abbreviation citation stays whole.
    cited = "See Goitein, vol. 2, pp. 12-14, ed. Smith. Then a new sentence."
    assert len(agent_module.split_sentences(cited)) == 2


def test_flag_wrapping_covers_whole_citation_sentence() -> None:
    """A flagged claim wraps its full sentence, citation included."""
    answer = 'The text concludes with a blessing: Levin, p. 45: "barukh atah".'
    annotated, enriched = agent_module.annotate_answer_with_flags(
        answer,
        [{"type": "factual_claim", "text": "The text concludes with a blessing", "reason": "x"}],
    )
    assert annotated.startswith("⟦flag:1⟧")
    assert annotated.endswith("⟦/flag⟧")
    assert enriched[0]["answer_span"] == answer
    assert agent_module.strip_flag_markers(annotated) == answer


def test_invented_shelfmark_searches_are_dropped() -> None:
    """A shelf mark nobody cited must never be searched (prompt-example leakage)."""
    plan = QueryPlan(
        actions=[SearchAction(search_type="primary_shelfmark", query="T-S 8.133")],
        needs_primary_secondary_linking=True,
        reasoning="follow-up",
    )
    state = {
        "user_query": "And what is the shelf mark of that piyyut?",
        "conversation_history": [
            {"role": "user", "content": "Tell me about zemirot"},
            {"role": "assistant", "content": "Levin published a piyyut in fragment WR IV. 329."},
        ],
        "processing_steps": [],
    }

    cleaned = AgenticRAGService._drop_ungrounded_shelfmark_actions(plan, state)

    assert all(action.search_type != "primary_shelfmark" for action in cleaned.actions)
    assert cleaned.actions[0].search_type == "bibliography_hybrid"
    assert any("Ignored invented shelf mark" in step for step in state["processing_steps"])


def test_shelfmark_cited_in_history_is_kept() -> None:
    """A shelf mark the previous answer actually mentioned stays searchable."""
    plan = QueryPlan(
        actions=[SearchAction(search_type="primary_shelfmark", query="T-S 12.388")],
        needs_primary_secondary_linking=True,
        reasoning="follow-up",
    )
    state = {
        "user_query": "Tell me more about that fragment",
        "conversation_history": [
            {"role": "assistant", "content": "Goitein discusses T-S 12.388 at length."},
        ],
        "processing_steps": [],
    }

    cleaned = AgenticRAGService._drop_ungrounded_shelfmark_actions(plan, state)

    assert cleaned.actions[0].search_type == "primary_shelfmark"
    assert cleaned.actions[0].query == "T-S 12.388"
    assert state["processing_steps"] == []


@pytest.mark.asyncio
async def test_synthesis_receives_conversation_history_for_reference_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-ups need the prior exchange to resolve 'that piyyut'."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value="answer")
    monkeypatch.setattr(service, "_call_llm", llm_call)
    state = {
        "user_query": "And what is the shelf mark of that piyyut?",
        "conversation_history": [
            {"role": "assistant", "content": "Levin published a piyyut in fragment WR IV. 329."},
        ],
        "bibliography_results": [],
        "graph_results": [],
        "synthesis_model_override": None,
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    await service._synthesize_answer_node(state)

    prompt = llm_call.call_args.kwargs["messages"][0]["content"]
    assert "EARLIER IN THIS CONVERSATION" in prompt
    assert "WR IV. 329" in prompt
    # History resolves references but must never become a citable source.
    assert "NOT a source" in prompt


def test_hebrew_quotes_and_pointed_text_verify() -> None:
    """Unpointed LLM quotes must match pointed (vocalized) source text, and
    gershayim-wrapped quotations must be extracted — but abbreviation marks
    like כ״י must not fake quote boundaries."""
    # Nikud-insensitive matching.
    assert agent_module._normalize_verification_text("קִינוֹת לְתִשְׁעָה בְּאָב") == (
        agent_module._normalize_verification_text("קינות לתשעה באב")
    )
    # Gershayim/geresh map onto ASCII quote marks consistently.
    assert agent_module._normalize_verification_text('בכ״י הגניזה') == (
        agent_module._normalize_verification_text('בכ"י הגניזה')
    )
    # Gershayim-wrapped quotation extracted; internal abbreviation ignored.
    text = 'לוין כותב: ״ברכת המזון המפויטת לשבת שנמצאה בגניזה״ ומוסיף כי בכ״י אחר הנוסח שונה.'
    quotes = agent_module.extract_direct_quotes(text)
    assert quotes == ["ברכת המזון המפויטת לשבת שנמצאה בגניזה"]
    # A pointed source still supports the unpointed quote.
    sources = [{"source_number": 1, "quoteable_text": "ברכת המזון המפויטת לשבת שנמצאה בגניזה", "evidence_text": ""}]
    assert agent_module.find_quote_source(quotes[0], sources) == 1


def test_relevance_gate_handles_crossscript_evidence() -> None:
    """A Hebrew page bridges via English metadata; an unjudgeable-only result
    set must not trigger a false 'subject absent' block."""
    hebrew_page_with_bridge = {
        "title": "In the Kingdom of Ishmael / במלכות ישמעאל",
        "full_text": "Main text: " + "הקהילה היהודית חיה תחת שלטון הפאטימים " * 5,
        "description": "Discusses Jewish communities under Fatimid rule in Egypt.",
        "subject_keywords": ["Jewish history"],
    }
    addresses, _ = agent_module.evidence_addresses_query(
        "Jewish communities under Fatimid rule", [hebrew_page_with_bridge]
    )
    assert addresses is True

    # Hebrew page with a Hebrew-only description: no bridge, so absence of
    # English terms proves nothing — gate must pass as inconclusive.
    unjudgeable = {
        "title": "גנזי קדם",
        "full_text": "Main text: " + "דיון בענייני מסחר בגניזה " * 8,
        "description": "דיון במסחר",
        "subject_keywords": [],
    }
    addresses2, _ = agent_module.evidence_addresses_query(
        "Radhanite merchant networks", [unjudgeable]
    )
    assert addresses2 is True

    # But an English page lacking the subject still blocks as before.
    english_offtopic = {
        "title": "A Cosmopolitan City",
        "full_text": "Main text: medical glossaries and pharmacology in Old Cairo.",
        "description": "Medical texts.",
        "subject_keywords": [],
    }
    addresses3, _ = agent_module.evidence_addresses_query(
        "Radhanite merchant networks", [english_offtopic]
    )
    assert addresses3 is False


def test_hebrew_sources_get_token_aware_budget() -> None:
    """Hebrew page text is capped tighter in characters (≈ equal in tokens)."""
    hebrew_text = "Main text: " + ("ברכת המזון המפויטת לשבת ומזכירה את קדושת השבת " * 80)
    english_text = "Main text: " + ("the piyyutic grace after meals for shabbat " * 80)
    sources = build_bibliography_source_context([
        {"authors": ["Levin"], "title": "Ginzei Kedem", "extracted_page_number": 44,
         "full_text": hebrew_text, "description": "", "shelf_marks_mentioned": []},
        {"authors": ["Scholar"], "title": "Survey", "extracted_page_number": 210,
         "full_text": english_text, "description": "", "shelf_marks_mentioned": []},
    ])
    hebrew_len = len(sources[0]["quoteable_text"])
    english_len = len(sources[1]["quoteable_text"])
    assert hebrew_len < english_len
    assert hebrew_len >= 500  # floor keeps the slice substantive


@pytest.mark.asyncio
async def test_crossscript_citation_not_downgraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An English claim supported by a Hebrew source must not be flagged for
    lacking English term overlap."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service,
        "_chat_completion",
        AsyncMock(return_value=((
            '{"verified_claims": [{"type": "factual_claim", '
            '"text": "Levin publishes a poetic Grace after Meals for Shabbat", '
            '"verdict": "supported", "source_number": 1, '
            '"reasoning": "The Hebrew source states this"}], "summary": "ok"}'
        ), "stop")),
    )
    state = {
        "draft_answer": "Levin publishes a poetic Grace after Meals for Shabbat.",
        "bibliography_results": [{
            "authors": ["Benyamin Menashe Levin"],
            "title": "גנזי קדם",
            "extracted_page_number": 44,
            "full_text": "Main text: " + "ברכת המזון של שבת המפויטת שנמצאה בגניזה " * 10,
            "description": "",
            "shelf_marks_mentioned": [],
        }],
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
    }

    result = await service._verify_claims_node(state)

    assert result["soft_flagged_claims"] == []
    assert result["error_type"] is None
    assert any(
        c.verification_status == "SUPPORTED" and "Grace after Meals" in c.claim
        for c in result["verified_claims"]
    )


@pytest.mark.asyncio
async def test_hebrew_search_variant_added_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Topical English plans gain a Hebrew search; scholar/Hebrew plans do not."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value="הקהילה היהודית בתקופה הפאטימית")
    monkeypatch.setattr(service, "_call_llm", llm_call)

    plan = QueryPlan(
        actions=[SearchAction(search_type="bibliography_hybrid", query="Fatimid Jews")],
        needs_primary_secondary_linking=True,
        reasoning="topic",
    )
    state = {"user_query": "Jewish communities under Fatimid rule", "processing_steps": []}
    augmented = await service._add_hebrew_search_variant(plan, state)

    assert len(augmented.actions) == 2
    assert agent_module.hebrew_char_ratio(augmented.actions[1].query) > 0.5
    assert any("Hebrew search variant" in s for s in state["processing_steps"])

    # Scholar-only plans are left alone (no pointless translation call).
    scholar_plan = QueryPlan(
        actions=[SearchAction(search_type="graph_scholar", query="Goitein")],
        needs_primary_secondary_linking=True,
        reasoning="scholar",
    )
    unchanged = await service._add_hebrew_search_variant(
        scholar_plan, {"user_query": "Who is Goitein?", "processing_steps": []}
    )
    assert len(unchanged.actions) == 1

    # Garbage translations are discarded rather than searched.
    monkeypatch.setattr(service, "_call_llm", AsyncMock(return_value="Sure! Here is the translation: The Jewish community"))
    state3 = {"user_query": "Jewish communities under Fatimid rule", "processing_steps": []}
    discarded = await service._add_hebrew_search_variant(plan, state3)
    assert len(discarded.actions) == 1


def test_alias_expansion_reserves_hebrew_slots() -> None:
    """Hebrew script forms survive the expansion cap despite being shortest."""
    forms = agent_module.expand_query_aliases("Tisha B'Av Kinnot", max_forms=6)
    assert any(agent_module.hebrew_char_ratio(f) > 0.5 for f in forms)


def test_alias_expansion_searches_corpus_spellings() -> None:
    """A query's own wording is not re-searched; corpus variants are added."""
    forms = agent_module.expand_query_aliases("Tish B'Av Kinnot")
    assert "ninth of av" in forms
    assert any(form in forms for form in ("qinot", "kinot", "lamentations"))
    assert "kinnot" not in forms  # already in the query
    assert agent_module.expand_query_aliases("shipping routes to Aden") == []


@pytest.mark.asyncio
async def test_irrelevant_retrieval_recovers_with_focused_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A focused retry on distinctive terms replaces context-only evidence."""
    service = AgenticRAGService()
    relevant = [{
        "title": "Studies in Genizah Liturgy",
        "full_text": "Main text: zemirot sung at the sabbath table appear in several fragments.",
        "description": "",
    }]
    calls: List[SearchAction] = []

    async def fake_search(action: SearchAction):
        calls.append(action)
        return relevant

    state = {"user_query": "zemirot in the Cairo Genizah", "processing_steps": []}
    recovered = await service._recover_irrelevant_retrieval(state, ["zemirot"], fake_search)

    assert recovered == relevant
    # Stage one drops collection context and goes keyword-heavy.
    assert calls[0].query == "zemirot"
    assert calls[0].keyword_weight == 85
    assert len(calls) == 1  # no re-plan needed when the focused retry works


@pytest.mark.asyncio
async def test_failed_recovery_escalates_to_planner_then_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When focused retry fails, the planner is re-invoked with failure context."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value='{"searches": [{"query": "table hymns", "keyword_weight": 80}]}')
    monkeypatch.setattr(service, "_call_llm", llm_call)
    still_irrelevant = [{
        "title": "A Cosmopolitan City",
        "full_text": "Main text: medical glossaries and trade letters.",
        "description": "",
    }]
    queries: List[str] = []

    async def fake_search(action: SearchAction):
        queries.append(action.query)
        return still_irrelevant

    state = {"user_query": "zemirot in the Cairo Genizah", "processing_steps": []}
    recovered = await service._recover_irrelevant_retrieval(state, ["zemirot"], fake_search)

    assert recovered == []  # nothing addressed the query, so no false evidence
    assert queries == ["zemirot", "table hymns"]  # focused retry, then re-plan
    prompt = llm_call.call_args.kwargs["messages"][0]["content"]
    assert "DISTINCTIVE TERMS NOT FOUND" in prompt
    assert "zemirot" in prompt
    assert "Do not repeat what was already tried" in prompt


@pytest.mark.asyncio
async def test_absent_subject_short_circuits_to_explicit_limitation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis must not run — and must not inventory the corpus — for an absent subject."""
    service = AgenticRAGService()
    llm_call = AsyncMock(return_value="should never be called")
    monkeypatch.setattr(service, "_call_llm", llm_call)
    state = {
        "user_query": "Do Genizah fragments discuss zemirot?",
        "error_type": "NO_RELEVANT_SOURCES",
        "subject_terms_not_found": ["zemirot"],
        "bibliography_results": [],
        "graph_results": [],
        "processing_steps": [],
    }

    result = await service._synthesize_answer_node(state)

    llm_call.assert_not_called()
    assert "zemirot" in result["draft_answer"]
    assert "wasn't able to find relevant information" in result["draft_answer"]


def test_author_variants_cover_initials_and_inverted_forms() -> None:
    """A canonical graph name must match pages catalogued under initials forms."""
    variants = ElasticsearchBibliographyService._person_name_variants("Shelomo Dov Goitein")
    assert "S. D. Goitein" in variants
    assert "S.D. Goitein" in variants
    assert "Goitein, S. D." in variants
    assert "Shelomo Goitein" in variants
    # Names already given as initials must not generate double-initial junk.
    initialed = ElasticsearchBibliographyService._person_name_variants("S. D. Goitein")
    assert "S. D. Goitein" in initialed


def test_page_numbers_accept_roman_and_range_labels() -> None:
    """Roman-numeral and range page labels must not crash result parsing."""
    assert BibliographySearchResult(doc_id="a", similarity_score=0.5,
                                    extracted_page_number="xxiv-xxv").extracted_page_number == "xxiv-xxv"
    assert BibliographySearchResult(doc_id="b", similarity_score=0.5,
                                    extracted_page_number="12").extracted_page_number == 12
    assert BibliographySearchResult(doc_id="c", similarity_score=0.5,
                                    extracted_page_number=[7]).extracted_page_number == 7
    assert BibliographySearchResult(doc_id="d", similarity_score=0.5,
                                    extracted_page_number=[]).extracted_page_number is None


def test_extract_json_object_ignores_thinking_prose_and_decoy_braces() -> None:
    """Find the real JSON object even inside reasoning-channel output."""
    noisy = (
        "Okay, the schema requires {type, text, verdict}. Let me think {a bit}.\n"
        'Here it is:\n{"verified_claims": [{"type": "factual_claim", "text": "x", '
        '"verdict": "supported", "source_number": 1, "reasoning": "brace } in string"}], '
        '"summary": "done"}\n'
    )
    parsed = agent_module.extract_json_object(noisy, anchor_key="verified_claims")
    assert parsed is not None
    assert parsed["summary"] == "done"
    assert parsed["verified_claims"][0]["reasoning"] == "brace } in string"
    assert agent_module.extract_json_object("no json here", anchor_key="revisions") is None


def test_remove_sentences_containing_deletes_only_offending_sentences() -> None:
    """Deterministic cleanup removes rejected sentences and keeps the rest."""
    answer = (
        "The piyyut has eighteen verses. He dates it to the seventh century. "
        "It survives in one fragment."
    )
    cleaned = remove_sentences_containing(answer, ["He dates it to the seventh century"])
    assert "seventh century" not in cleaned
    assert "eighteen verses" in cleaned
    assert "one fragment" in cleaned


def test_retry_exhaustion_retains_only_verified_short_quotes() -> None:
    """Prefer a citation-bearing verified excerpt over the generic failure text."""
    state = {
        "draft_answer": (
            'Wallenstein writes: "The manuscript preserves a distinctive piyyut for the '
            'Day of Atonement." An unsupported conclusion follows.'
        ),
        "bibliography_results": [{
            "authors": ["Meir Wallenstein"],
            "title": "A Unique Kol-Nidre Piyyut",
            "extracted_page_number": 489,
            "full_text": (
                "Generated header. Main text: The manuscript preserves a distinctive piyyut "
                "for the Day of Atonement."
            ),
            "description": "",
            "shelf_marks_mentioned": [],
        }],
        "graph_results": [],
    }

    fallback = AgenticRAGService._build_verified_evidence_fallback(state)

    assert fallback is not None
    assert "Meir Wallenstein, *A Unique Kol-Nidre Piyyut*, p. 489" in fallback
    assert "The manuscript preserves a distinctive piyyut for the Day of Atonement." in fallback
    assert "unsupported conclusion" not in fallback


def test_retry_policy_allows_three_targeted_repairs() -> None:
    """Allow three repairs after the initial synthesis, then stop."""
    assert should_retry_verification(1) is True
    assert should_retry_verification(2) is True
    assert should_retry_verification(3) is True
    assert should_retry_verification(4) is False


def test_terminal_fallback_groups_verified_facts_and_quotes_by_source() -> None:
    """Present surviving evidence as cited paragraphs rather than loose fragments."""
    citation = "Shelomo Dov Goitein, *A Mediterranean Society*, p. 12"
    state = {
        "draft_answer": "No surviving quotation in this draft.",
        "bibliography_results": [],
        "graph_results": [],
        "verification_feedback_history": [{
            "attempt": 1,
            "summary": "One rejected claim",
            "rejected_claims": [],
            "supported_claims": [{
                "type": "factual_claim",
                "text": "Goitein organized the discussion around documentary evidence",
                "source_number": 1,
                "citation": citation,
            }],
        }],
        "supported_evidence_units": [{
            "type": "attribution",
            "text": "He described the documents as evidence for everyday social history",
            "source_number": 1,
            "citation": citation,
        }],
    }

    fallback = AgenticRAGService._build_verified_evidence_fallback(state)

    assert fallback is not None
    assert fallback.count(citation) == 1
    assert "the verified evidence supports" in fallback
    assert "It also supports" in fallback
    assert "documentary evidence" in fallback
    assert "everyday social history" in fallback


def test_verified_quotes_are_limited_to_thirty_words() -> None:
    """Bound extractive output to a short one- or two-line passage."""
    quote = " ".join(f"word{number}" for number in range(35))

    bounded = bound_direct_quote(quote)

    assert len(bounded.rstrip("…").split()) == 30
    assert bounded.endswith("…")


# ============================================================================
# Deterministic quote check node (C2) and repair no-op guard (C4)
# ============================================================================

_QUOTES_REMOVED_LEAD = (
    "Several quoted statements in the drafted answer could not be found in the sources "
    "and were removed, leaving too little to present. The following source-linked "
    "account includes only claims and short quotations that were verified:"
)
_QUOTE_NOTE_EN = "_Some quoted statements were removed because they could not be found in the sources._"
_QUOTE_NOTE_HE = "_חלק מהציטוטים הוסרו משום שלא נמצאו במקורות._"
# Quote-free prose long enough (over 200 chars) to survive a deletion.
_FILLER = (
    " Goitein spent decades reading these letters. His volumes describe the merchants,"
    " their families, their partnerships and their journeys between Egypt and the"
    " Malabar coast in careful and unhurried detail, drawing on hundreds of documents."
)
_ALL_SUPPORTED_VERIFIER_JSON = '{"verified_claims": [], "summary": "ok"}'


def _goitein_bibliography() -> List[Dict[str, Any]]:
    """Build one bibliography result with quoteable text and a cited shelf mark.

    :returns: Fresh bibliography results for the quote-check tests.
    :rtype: List[Dict[str, Any]]
    """
    return [{
        "authors": ["S. D. Goitein"],
        "title": "Letters of Medieval Jewish Traders",
        "extracted_page_number": 7,
        "full_text": (
            "Main text: Goitein began collecting the documents of the India Book, which he "
            "called his life work. In Fustat the court met every Monday and Thursday in the "
            "synagogue. The letter T-S 12.243 was among the first he read."
        ),
        "description": "",
        "shelf_marks_mentioned": ["T-S 12.243"],
    }]


def _quote_check_state(draft: str, bibliography: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the state the quote check and verify nodes read.

    :param draft: Drafted answer text.
    :param bibliography: Bibliography results for this turn.
    :returns: A minimal pipeline state.
    :rtype: Dict[str, Any]
    """
    return {
        "user_query": "Who was S. D. Goitein?",
        "draft_answer": draft,
        "bibliography_results": bibliography,
        "graph_results": [],
        "excluded_claims": [],
        "retry_count": 0,
        "processing_steps": [],
        "error": None,
        "error_type": None,
        "quote_check": {},
        "soft_flagged_claims": [],
        "verification_feedback_history": [],
    }


def _initial_graph_state(user_query: str) -> Dict[str, Any]:
    """Build the initial graph state the way ``chat()`` does.

    :param user_query: The user's question.
    :returns: Initial pipeline state for ``graph.ainvoke``.
    :rtype: Dict[str, Any]
    """
    return {
        "user_query": user_query,
        "conversation_history": [],
        "resolved_query": None,
        "is_followup": False,
        "synthesis_model_override": None,
        "query_plan": None,
        "bibliography_results": [],
        "primary_source_results": [],
        "graph_results": [],
        "resolved_entities": [],
        "shelf_marks_to_fetch": [],
        "shelf_marks_in_bibliography": set(),
        "shelf_marks_from_search": set(),
        "shelf_mark_lookup": {},
        "draft_answer": None,
        "verified_claims": [],
        "verification_summary": {},
        "final_answer": None,
        "processing_steps": [],
        "error": None,
        "error_type": None,
        "retry_count": 0,
        "excluded_claims": [],
        "soft_flagged_claims": [],
        "subject_terms_not_found": [],
        "work_manuscripts": {},
        "supported_evidence_units": [],
        "verification_feedback_history": [],
        "stage_timings": {},
        "stage_calls": {},
        "quote_check": {},
        "repair_noop": False,
        "verification_unavailable": False,
    }


def _delete_mode_service(monkeypatch: pytest.MonkeyPatch) -> AgenticRAGService:
    """Build a service in delete mode with quote smoothing disabled.

    :param monkeypatch: Pytest monkeypatch fixture (sets the env vars).
    :returns: A service whose quote check deletes and never calls a model.
    :rtype: AgenticRAGService
    """
    monkeypatch.setenv("QUOTE_FAILURE_MODE", "delete")
    monkeypatch.setenv("QUOTE_SMOOTHING_MODEL", "")
    service = AgenticRAGService()
    assert service.quote_failure_mode == "delete"
    assert service.quote_smoothing_model == ""
    return service


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    service: AgenticRAGService,
    bibliography: List[Dict[str, Any]],
    draft: str,
    finalize: Any = None,
) -> None:
    """Replace retrieval and synthesis with fixed sources and a fixed draft.

    check_quotes, verify_claims and repair_answer stay real. The finalizer
    stays real too unless ``finalize`` is given; its Neo4j and Elasticsearch
    lookups are stubbed. The graph is rebuilt so it binds the patched methods.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param service: Service under test.
    :param bibliography: Bibliography results synthesis "retrieved".
    :param draft: The drafted answer synthesis "produced".
    :param finalize: Optional replacement finalizer coroutine function.
    :returns: Nothing.
    :rtype: None
    """
    async def passthrough(state: Dict[str, Any]) -> Dict[str, Any]:
        """Leave the state unchanged.

        :param state: Pipeline state.
        :returns: The same state.
        :rtype: Dict[str, Any]
        """
        return state

    async def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
        """Inject the retrieved sources and the drafted answer.

        :param state: Pipeline state.
        :returns: State with sources and draft.
        :rtype: Dict[str, Any]
        """
        state["bibliography_results"] = bibliography
        state["draft_answer"] = draft
        return state

    for node_name in (
        "_resolve_query_node", "_route_query_node",
        "_execute_searches_node", "_link_primary_secondary_node",
    ):
        monkeypatch.setattr(service, node_name, passthrough)
    monkeypatch.setattr(service, "_synthesize_answer_node", synthesize)
    if finalize is not None:
        monkeypatch.setattr(service, "_finalize_response_node", finalize)
    monkeypatch.setattr(service, "_build_works_cited", AsyncMock(return_value=[]))
    monkeypatch.setattr(service, "_resolve_unlinked_shelfmarks", AsyncMock(return_value=None))
    service.graph = service._build_graph()


def test_quote_failure_mode_reads_env_and_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """C4: QUOTE_FAILURE_MODE defaults to delete; unknown values warn and use delete."""
    monkeypatch.delenv("QUOTE_FAILURE_MODE", raising=False)
    assert AgenticRAGService().quote_failure_mode == "delete"
    monkeypatch.setenv("QUOTE_FAILURE_MODE", " Repair ")
    assert AgenticRAGService().quote_failure_mode == "repair"
    monkeypatch.setenv("QUOTE_FAILURE_MODE", "rewrite")
    with caplog.at_level("WARNING", logger=agent_module.logger.name):
        assert AgenticRAGService().quote_failure_mode == "delete"
    assert any("QUOTE_FAILURE_MODE" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_check_quotes_passes_through_guards_and_repair_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2 steps 1-2: guard states and repair mode pass through untouched
    (verify_claims, not check_quotes, records an empty draft)."""
    service = _delete_mode_service(monkeypatch)
    smoothing = AsyncMock(side_effect=AssertionError("nothing was deleted"))
    monkeypatch.setattr(service, "_smooth_after_deletion", smoothing)
    fabricated = 'He wrote "an invented quotation here" once.'

    for error_type in ("NO_RELEVANT_SOURCES", "NO_ANSWER_GENERATED"):
        state = _quote_check_state(fabricated, [])
        state["error_type"] = error_type
        result = await service._check_quotes_node(state)
        assert result["draft_answer"] == fabricated
        assert result["quote_check"] == {}
        assert result["processing_steps"] == []

    empty = await service._check_quotes_node(_quote_check_state("   ", []))
    assert empty["draft_answer"] == "   "
    assert empty["error_type"] is None
    assert empty["quote_check"] == {}

    verified_only = await service._check_quotes_node(_quote_check_state(
        'Goitein spoke of "the documents of the India Book."', _goitein_bibliography()
    ))
    assert verified_only["quote_check"]["sentences_deleted"] == 0
    smoothing.assert_not_awaited()

    service.quote_failure_mode = "repair"
    repair = await service._check_quotes_node(_quote_check_state(fabricated, []))
    assert repair["quote_check"] == {"mode": "repair"}
    assert repair["draft_answer"] == fabricated
    assert agent_module.route_after_quote_check(repair) == "verify"


@pytest.mark.asyncio
async def test_check_quotes_hands_gap_offsets_to_the_smoothing_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2 step 6: after a deletion the smoothing hook gets the post-deletion
    draft and the gap offsets of each affected paragraph; its draft and status
    are recorded."""
    service = _delete_mode_service(monkeypatch)
    first = "Goitein read the letters closely."
    fabricated = 'He wrote that "the traders never used written contracts at all" in Aden.'
    second_paragraph = "A second paragraph stays as it is."
    draft = f"{first} {fabricated}{_FILLER}\n\n{second_paragraph}"
    smoothing = AsyncMock(side_effect=lambda text, gaps: (
        text.replace("closely", "very closely"), "applied", {0: "applied"}
    ))
    monkeypatch.setattr(service, "_smooth_after_deletion", smoothing)

    result = await service._check_quotes_node(_quote_check_state(draft, _goitein_bibliography()))

    deleted = f"{first}{_FILLER}\n\n{second_paragraph}"
    smoothing.assert_awaited_once_with(deleted, {0: [len(first) + 1]})
    assert result["draft_answer"] == deleted.replace("closely", "very closely")
    assert result["quote_check"]["smoothing"] == "applied"
    assert result["quote_check"]["smoothing_paragraphs"] == {0: "applied"}
    assert result["quote_check"]["sentences_deleted"] == 1
    assert "⟦" not in result["draft_answer"]


@pytest.mark.asyncio
async def test_case18_deletion_leaving_too_little_returns_quote_removal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 18 (C2 step 5): deletion leaves under 200 chars, so the graph goes
    straight to the finalizer, which returns the verified fallback with the
    quote-removal lead; neither the verifier nor smoothing is called."""
    service = _delete_mode_service(monkeypatch)
    draft = (
        'Goitein notes that "the court met every Monday and Thursday" in Fustat. '
        'He adds that "judges were elected annually by the whole community" there. '
        'He also claims that "every merchant kept a personal scribe at home" then.'
    )
    _patch_pipeline(monkeypatch, service, _goitein_bibliography(), draft)
    verifier = AsyncMock(return_value=(_ALL_SUPPORTED_VERIFIER_JSON, "stop"))
    monkeypatch.setattr(service, "_chat_completion", verifier)
    model_27b = AsyncMock(side_effect=AssertionError("the 27B must not be called"))
    monkeypatch.setattr(service, "_call_llm", model_27b)
    smoothing = AsyncMock(side_effect=AssertionError("too short to smooth"))
    monkeypatch.setattr(service, "_smooth_after_deletion", smoothing)

    response = await service.chat("What did Goitein say about the court?")

    verifier.assert_not_awaited()
    model_27b.assert_not_awaited()
    smoothing.assert_not_awaited()
    assert response.answer.startswith(_QUOTES_REMOVED_LEAD + "\n\n")
    assert "the source states: “the court met every Monday and Thursday”" in response.answer
    assert "judges were elected" not in response.answer
    assert "personal scribe" not in response.answer
    assert "three targeted repairs" not in response.answer
    assert response.metrics["verification_cycles"] == 0
    assert response.metrics["stage_calls"]["check_quotes"] == 1
    assert any("too little of the draft" in step for step in response.processing_steps)


def test_case18_delete_mode_fallback_collects_quotes_with_the_quote_check_matcher() -> None:
    """Case 18 (C2): the delete-mode fallback finds quotes the legacy matcher
    misses on punctuation; the default keeps today's matcher and lead."""
    state = {
        "draft_answer": 'Goitein spoke of "the documents of the India Book." Nothing else survived.',
        "bibliography_results": _goitein_bibliography(),
        "graph_results": [],
    }

    assert AgenticRAGService._build_verified_evidence_fallback(state) is None
    fallback = AgenticRAGService._build_verified_evidence_fallback(
        state, quote_failure_mode="delete", quotes_removed=True
    )

    assert fallback is not None
    assert fallback.startswith(_QUOTES_REMOVED_LEAD)
    assert "“the documents of the India Book.”" in fallback
    exhausted = AgenticRAGService._build_verified_evidence_fallback(state, quote_failure_mode="delete")
    assert exhausted is not None and "after three targeted repairs" in exhausted


@pytest.mark.asyncio
async def test_case18_short_draft_with_nothing_deleted_still_goes_to_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 18 (second case): a short draft with nothing deleted is not 'too
    short' and is verified as today."""
    service = _delete_mode_service(monkeypatch)
    draft = (
        "Goitein described the rabbinical court of Fustat, which met twice a week in the "
        "synagogue and heard the cases brought by merchants and their families."
    )
    assert 100 < len(draft) < 200
    _patch_pipeline(monkeypatch, service, _goitein_bibliography(), draft)
    verifier = AsyncMock(return_value=(_ALL_SUPPORTED_VERIFIER_JSON, "stop"))
    monkeypatch.setattr(service, "_chat_completion", verifier)

    response = await service.chat("What did Goitein say about the court?")

    verifier.assert_awaited_once()
    assert response.answer == draft
    assert response.success is True
    assert response.metrics["verification_cycles"] == 1
    assert response.metrics["repair_attempts"] == 0


@pytest.mark.asyncio
async def test_case19_delete_mode_hands_verified_items_to_the_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 19 (C2): in delete mode the verify node records the quote check's
    verified quote and shelf mark as SUPPORTED, has no quote hard failures, and
    an all-supported 35B verdict means no repair."""
    service = _delete_mode_service(monkeypatch)
    verifier = AsyncMock(return_value=(_ALL_SUPPORTED_VERIFIER_JSON, "stop"))
    monkeypatch.setattr(service, "_chat_completion", verifier)
    draft = (
        'Goitein spoke of "the documents of the India Book." He cited T-S 12.243 often. '
        'He wrote that "the traders never used written contracts at all" in Aden.' + _FILLER
    )
    state = _quote_check_state(draft, _goitein_bibliography())
    # Premise: today's matcher would hard-fail this punctuation-only difference.
    sources = build_verification_sources(state["bibliography_results"], [])
    assert find_quote_source("the documents of the India Book.", sources) is None

    state = await service._check_quotes_node(state)
    assert agent_module.route_after_quote_check(state) == "verify"
    result = await service._verify_claims_node(state)

    statuses = {claim.claim: claim.verification_status for claim in result["verified_claims"]}
    assert statuses == {
        "quote: the documents of the India Book.": "SUPPORTED",
        "shelf_mark: T-S 12.243": "SUPPORTED",
    }
    assert result["verification_summary"]["CONTRADICTED_OR_FABRICATED"] == 0
    assert result["error_type"] is None
    assert result["excluded_claims"] == []
    # Deleted spans are gone from the answer, so they are never soft flags.
    assert result["soft_flagged_claims"] == []
    assert {unit["type"] for unit in result["supported_evidence_units"]} == {"quote", "shelf_mark"}
    verifier.assert_awaited_once()
    assert "written contracts" not in verifier.call_args.kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_case29_quote_check_verified_list_drops_quotes_from_deleted_sentences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 29 (C2 step 4): a verified quote sharing a sentence with an
    unverified one leaves quote_check["verified"]; the counts stay first-run."""
    service = _delete_mode_service(monkeypatch)
    draft = (
        'He wrote "the court met every Monday" and also "the judges were paid by the '
        'community" in one letter.' + _FILLER
    )

    result = await service._check_quotes_node(_quote_check_state(draft, _goitein_bibliography()))

    quote_check = result["quote_check"]
    assert quote_check["mode"] == "delete"
    assert quote_check["sentences_deleted"] == 1
    assert quote_check["counts"]["quotations"] == 2
    assert quote_check["counts"]["verified"] == 1
    assert quote_check["counts"]["unverified"] == 1
    assert quote_check["verified"] == []
    assert [span[2:] for span in quote_check["unverified_spans"]] == [
        ("the judges were paid by the community", "quote")
    ]
    assert quote_check["answer_too_short"] is False
    assert result["draft_answer"] == _FILLER.strip()
    assert len(result["processing_steps"]) == 1


@pytest.mark.asyncio
async def test_case20_unanchorable_rejections_make_a_noop_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 20 (C4, nothing anchorable): the error clears as today, the claim
    is soft-flagged, and repair_noop routes straight to the finalizer."""
    service = AgenticRAGService()
    model_27b = AsyncMock(side_effect=AssertionError("nothing to repair"))
    monkeypatch.setattr(service, "_call_llm", model_27b)
    state = _quote_check_state("The piyyut has eighteen verses.", [])
    state.update({
        "excluded_claims": [{
            "type": "factual_claim",
            "text": "Venetian glassmakers supplied the community with oil lamps",
            "reason": "Not in any source",
        }],
        "retry_count": 1,
        "error": "unsupported",
        "error_type": "FABRICATED_CLAIMS",
    })

    result = await service._repair_answer_node(state)

    model_27b.assert_not_awaited()
    assert result["repair_noop"] is True
    assert result["error_type"] is None
    assert result["draft_answer"] == "The piyyut has eighteen verses."
    assert result["soft_flagged_claims"][0]["text"].startswith("Venetian glassmakers")
    assert agent_module.route_after_repair(result) == "finalize"


@pytest.mark.asyncio
async def test_case20_unchanged_revision_keeps_error_and_narrows_excluded_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 20 (C4, draft unchanged): FABRICATED_CLAIMS stays and
    excluded_claims holds only the anchorable claims, so the unanchorable
    one is not soft-flagged twice."""
    service = AgenticRAGService()
    draft = "The piyyut was composed for Passover. It survives in a single fragment."
    monkeypatch.setattr(service, "_call_llm", AsyncMock(return_value=json.dumps(
        {"revisions": [{"paragraph_index": 0, "revised_text": draft}]}
    )))
    anchorable = {
        "type": "factual_claim",
        "text": "The piyyut was composed for Passover",
        "reason": "SOURCE 1 says Kol Nidre",
    }
    unanchorable = {
        "type": "factual_claim",
        "text": "Venetian glassmakers supplied the community with oil lamps",
        "reason": "Not in any source",
    }
    state = _quote_check_state(draft, [])
    state.update({
        "excluded_claims": [anchorable, unanchorable],
        "retry_count": 1,
        "error": "contradicted",
        "error_type": "FABRICATED_CLAIMS",
    })

    result = await service._repair_answer_node(state)

    assert result["repair_noop"] is True
    assert result["draft_answer"] == draft
    assert result["error_type"] == "FABRICATED_CLAIMS"
    assert result["excluded_claims"] == [anchorable]
    assert [flag["text"] for flag in result["soft_flagged_claims"]] == [unanchorable["text"]]
    assert agent_module.route_after_repair(result) == "finalize"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["delete", "repair"])
async def test_case20_unchanged_repair_ends_loop_and_exhaustion_removes_sentence(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Case 20 (C4) through the compiled graph: the 27B returns the paragraph
    unchanged, so repair_noop sends the graph to the finalizer, whose
    exhaustion branch removes the contradicted sentence; the verifier runs once."""
    monkeypatch.setenv("QUOTE_FAILURE_MODE", mode)
    monkeypatch.setenv("QUOTE_SMOOTHING_MODEL", "")
    service = AgenticRAGService()
    contradicted = "The piyyut was composed for Passover."
    remainder = (
        "Wallenstein describes the manuscript as a unique Kol Nidre composition preserved in "
        "the Cairo Genizah, and he discusses its liturgical setting, its poetic structure, and "
        "the community that used it at considerable length."
    )
    assert len(remainder) >= 200
    draft = f"{contradicted} {remainder}"
    bibliography = [{
        "authors": ["Meir Wallenstein"],
        "title": "A Unique Kol-Nidre Piyyut",
        "extracted_page_number": 489,
        "full_text": "Main text: This Kol Nidre piyyut is unique.",
        "description": "",
        "shelf_marks_mentioned": [],
    }]
    _patch_pipeline(monkeypatch, service, bibliography, draft)
    verifier = AsyncMock(return_value=(json.dumps({
        "verified_claims": [{
            "type": "factual_claim",
            "text": "The piyyut was composed for Passover",
            "verdict": "contradicted",
            "source_number": None,
            "reasoning": "SOURCE 1 identifies the piyyut as a Kol Nidre composition",
        }],
        "summary": "One contradicted claim",
    }), "stop"))
    monkeypatch.setattr(service, "_chat_completion", verifier)
    model_27b = AsyncMock(return_value=json.dumps(
        {"revisions": [{"paragraph_index": 0, "revised_text": draft}]}
    ))
    monkeypatch.setattr(service, "_call_llm", model_27b)

    response = await service.chat("What occasion was this piyyut written for?")

    verifier.assert_awaited_once()
    model_27b.assert_awaited_once()
    assert response.metrics["verification_cycles"] == 1
    assert response.metrics["repair_attempts"] == 1
    assert any("changed nothing" in step for step in response.processing_steps)
    assert "Passover" not in response.answer
    assert response.answer.startswith("Wallenstein describes the manuscript")
    assert response.success is True


@pytest.mark.asyncio
@pytest.mark.parametrize("draft, note", [
    ("Goitein read the Geniza letters of the India traders for decades.", _QUOTE_NOTE_EN),
    ("גויטיין קרא את מכתבי סוחרי הודו מן הגניזה במשך עשרות שנים.", _QUOTE_NOTE_HE),
], ids=["english", "hebrew"])
async def test_quote_removal_closing_note_follows_answer_language(draft: str, note: str) -> None:
    """C2 closing note: after a deletion one italic note in the answer's
    language closes the prose, before any appendix; none without a deletion."""
    service = AgenticRAGService()

    def finalize_state(sentences_deleted: int) -> Dict[str, Any]:
        """Build a finalizer state with one catalog entry.

        :param sentences_deleted: Sentences the quote check removed.
        :returns: Pipeline state for the finalizer.
        :rtype: Dict[str, Any]
        """
        return {
            "user_query": "Who was Goitein?",
            "draft_answer": draft,
            "error": None,
            "error_type": None,
            "quote_check": {"mode": "delete", "sentences_deleted": sentences_deleted},
            "soft_flagged_claims": [],
            "excluded_claims": [],
            "primary_source_results": [
                {"shelf_mark": "T-S 12.243", "doc_id": "doc-1", "title": "Letter from Aden"}
            ],
            "shelf_mark_lookup": {},
            "bibliography_results": [],
            "graph_results": [],
            "work_manuscripts": {},
            "shelf_marks_in_bibliography": set(),
            "processing_steps": [],
        }

    noted = (await service._finalize_response_node(finalize_state(1)))["final_answer"]
    plain = (await service._finalize_response_node(finalize_state(0)))["final_answer"]

    assert noted.startswith(f"{draft}\n\n{note}\n\n")
    assert noted.index(note) < noted.index("**Related catalog entries:**")
    assert noted.count(note) == 1
    assert note not in plain
    assert plain.startswith(f"{draft}\n\n---")


_ACCEPTANCE_DRAFTS = [
    (
        'Goitein spoke of "the documents of the India Book." He also wrote that "no court '
        'sat in Fustat on any weekday" at all.' + _FILLER,
        ["no court sat"],
    ),
    ("The letter T-S 99.999 describes a shipwreck in the Red Sea." + _FILLER, ["T-S 99.999"]),
    (
        "Goitein called the merchants “the most mobile people of their age” in one essay. "
        "They are listed in T-S 88.888 as well. He cited T-S 12.243 often." + _FILLER,
        ["most mobile people", "T-S 88.888"],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("draft, fabricated", _ACCEPTANCE_DRAFTS, ids=["quote", "shelf_mark", "both"])
async def test_acceptance_delete_mode_quote_misses_never_reach_repair(
    monkeypatch: pytest.MonkeyPatch,
    draft: str,
    fabricated: List[str],
) -> None:
    """Spec §7 #4 acceptance harness: with QUOTE_FAILURE_MODE=delete a quote or
    shelf-mark miss is deleted before the 35B, never sets FABRICATED_CLAIMS and
    never calls the 27B: one verification cycle, zero repairs."""
    service = _delete_mode_service(monkeypatch)
    verify_error_types: List[Any] = []
    real_verify = service._verify_claims_node

    async def recording_verify(state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the real verify node and record the error_type it leaves.

        :param state: Pipeline state.
        :returns: State after verification.
        :rtype: Dict[str, Any]
        """
        result = await real_verify(state)
        verify_error_types.append(result.get("error_type"))
        return result

    async def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
        """Publish the checked draft as the final answer.

        :param state: Pipeline state.
        :returns: State with final_answer set.
        :rtype: Dict[str, Any]
        """
        state["final_answer"] = state["draft_answer"]
        return state

    monkeypatch.setattr(service, "_verify_claims_node", recording_verify)
    _patch_pipeline(monkeypatch, service, _goitein_bibliography(), draft, finalize=finalize)
    verifier = AsyncMock(return_value=(_ALL_SUPPORTED_VERIFIER_JSON, "stop"))
    monkeypatch.setattr(service, "_chat_completion", verifier)
    model_27b = AsyncMock(side_effect=AssertionError("the 27B must not be called"))
    monkeypatch.setattr(service, "_call_llm", model_27b)

    started_at = agent_module.time.monotonic()
    final_state = await service.graph.ainvoke(_initial_graph_state("Who was S. D. Goitein?"))
    metrics = service._build_metrics(final_state, started_at, {"llm_calls": []})

    assert metrics["verification_cycles"] == 1
    assert metrics["repair_attempts"] == 0
    assert verify_error_types == [None]
    assert final_state["error_type"] is None
    model_27b.assert_not_awaited()
    verifier.assert_awaited_once()
    assert final_state["quote_check"]["sentences_deleted"] >= 1
    # QUOTE_SMOOTHING_MODEL="" really disables smoothing: the only model call
    # was the verifier.
    assert final_state["quote_check"]["smoothing"] == "disabled"
    for text in fabricated:
        assert text not in final_state["final_answer"]
    assert "Goitein spent decades reading these letters." in final_state["final_answer"]


# ----------------------------------------------------------------------------
# Case 17 / 17b: the guarded smoothing call after a quote deletion (C3, C3a)
# ----------------------------------------------------------------------------

_SMOOTHING_35B = "qwen/qwen3.6-35b-a3b"
_SMOOTH_FIRST = "Goitein read the letters closely."
_SMOOTH_FABRICATED = 'He wrote that "the traders never used written contracts at all" in Aden.'
_SMOOTH_NEXT = "He published the first volume later."
_SMOOTH_DRAFT = f"{_SMOOTH_FIRST} {_SMOOTH_FABRICATED} {_SMOOTH_NEXT}{_FILLER}"
_SMOOTH_DELETED = f"{_SMOOTH_FIRST} {_SMOOTH_NEXT}{_FILLER}"
_SMOOTH_ACCEPTED = f"{_SMOOTH_FIRST} However, he published the first volume later.{_FILLER}"


def _smoothing_service(monkeypatch: pytest.MonkeyPatch) -> AgenticRAGService:
    """Build a delete-mode service whose smoothing model defaults to the 35B verifier.

    The repo's ``.env`` is loaded on import, so every model variable the
    guard reads is set or cleared explicitly here.

    :param monkeypatch: Pytest monkeypatch fixture (sets the env vars).
    :returns: A service with smoothing enabled on the verification model.
    :rtype: AgenticRAGService
    """
    monkeypatch.setenv("QUOTE_FAILURE_MODE", "delete")
    monkeypatch.setenv("ROUTER_MODEL", "qwen/qwen3-4b-2507")
    monkeypatch.setenv("VERIFICATION_MODEL", _SMOOTHING_35B)
    monkeypatch.delenv("QUOTE_SMOOTHING_MODEL", raising=False)
    monkeypatch.delenv("QUOTE_SMOOTHING_REASONING_EFFORT", raising=False)
    service = AgenticRAGService()
    assert service.quote_smoothing_model == _SMOOTHING_35B
    assert service.quote_smoothing_reasoning_effort == "none"
    return service


def _revisions(*texts: Tuple[int, str]) -> str:
    """Serialise smoothing revisions the way the model returns them.

    :param texts: ``(paragraph_index, revised_text)`` pairs.
    :returns: The JSON response content.
    :rtype: str
    """
    return json.dumps({"revisions": [
        {"paragraph_index": index, "revised_text": text} for index, text in texts
    ]})


@pytest.mark.asyncio
async def test_case17_smoothing_applies_a_guarded_rewrite_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 17 acceptance: one smoothing call on the 35B (reasoning off, JSON
    schema, 1024 tokens) gets the paragraph with a gap marker and no evidence;
    a rewrite that passes the guard replaces the paragraph."""
    service = _smoothing_service(monkeypatch)
    chat_completion = AsyncMock(return_value=(_revisions((0, _SMOOTH_ACCEPTED)), "stop"))
    monkeypatch.setattr(service, "_chat_completion", chat_completion)

    result = await service._check_quotes_node(_quote_check_state(_SMOOTH_DRAFT, _goitein_bibliography()))

    assert result["draft_answer"] == _SMOOTH_ACCEPTED
    assert result["quote_check"]["smoothing"] == "applied"
    assert result["quote_check"]["smoothing_paragraphs"] == {0: "applied"}
    assert result["quote_check"]["sentences_deleted"] == 1
    chat_completion.assert_awaited_once()
    kwargs = chat_completion.call_args.kwargs
    assert kwargs["model"] == _SMOOTHING_35B
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 1024
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["recover_reasoning"] is True
    assert kwargs["response_format"] is AgenticRAGService._REPAIR_RESPONSE_FORMAT
    prompt = kwargs["messages"][0]["content"]
    assert f"PARAGRAPH 0:\n{_SMOOTH_FIRST} ⟦gap⟧ {_SMOOTH_NEXT}" in prompt
    assert agent_module.QUOTE_SMOOTHING_INSTRUCTIONS in prompt
    assert "add a new sentence only if nothing else works" in prompt
    assert "written contracts" not in prompt
    assert "India Book" not in prompt and "SOURCE" not in prompt
    assert "⟦" not in result["draft_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, reason", [
    ((_revisions((0, _SMOOTH_ACCEPTED)), "length"), "length"),
    (("I cannot produce JSON for this.", "stop"), "parse_error"),
    ((_revisions((5, _SMOOTH_ACCEPTED)), "stop"), "missing"),
    ((_revisions((0, _SMOOTH_ACCEPTED.replace("later", "in 1952"))), "stop"), "new_number"),
    ((_revisions((0, _SMOOTH_ACCEPTED.replace("However,", "As Mann noted,"))), "stop"), "new_proper_noun"),
    ((_revisions((0, _SMOOTH_ACCEPTED.replace("Goitein spent decades", "Goitein spent many decades"))), "stop"),
     "altered_sentence"),
    ((_revisions((0, _SMOOTH_DELETED.replace(_SMOOTH_NEXT, f"⟦gap⟧ {_SMOOTH_NEXT}"))), "stop"), "marker_left"),
], ids=["finish_reason_length", "parse_failure", "paragraph_missing", "c3a_in_1952", "c3a_mann",
        "c3a_non_adjacent_altered", "leftover_marker"])
async def test_case17_rejected_smoothing_keeps_the_unsmoothed_paragraph(
    monkeypatch: pytest.MonkeyPatch,
    response: Tuple[str, str],
    reason: str,
) -> None:
    """Case 17 rejections through the node: a cut-off, unparseable, missing or
    guard-violating rewrite leaves the post-deletion text and records why."""
    service = _smoothing_service(monkeypatch)
    chat_completion = AsyncMock(return_value=response)
    monkeypatch.setattr(service, "_chat_completion", chat_completion)

    result = await service._check_quotes_node(_quote_check_state(_SMOOTH_DRAFT, _goitein_bibliography()))

    chat_completion.assert_awaited_once()
    assert result["draft_answer"] == _SMOOTH_DELETED
    assert result["quote_check"]["smoothing"] == f"rejected:{reason}"
    assert result["quote_check"]["smoothing_paragraphs"] == {0: f"rejected:{reason}"}
    assert result["quote_check"]["sentences_deleted"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    httpx.ConnectError("connection refused"),
    ModelUnavailableError("model unavailable"),
], ids=["httpx_connect_error", "model_unavailable"])
async def test_case17_smoothing_call_failure_never_fails_the_answer(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """Case 17: a raised httpx.ConnectError (or ModelUnavailableError) from the
    smoothing call keeps the unsmoothed, post-deletion draft."""
    service = _smoothing_service(monkeypatch)
    chat_completion = AsyncMock(side_effect=error)
    monkeypatch.setattr(service, "_chat_completion", chat_completion)

    result = await service._check_quotes_node(_quote_check_state(_SMOOTH_DRAFT, _goitein_bibliography()))

    chat_completion.assert_awaited_once()
    assert result["draft_answer"] == _SMOOTH_DELETED
    assert result["quote_check"]["smoothing"] == "rejected:unavailable"
    assert result["quote_check"]["sentences_deleted"] == 1


@pytest.mark.asyncio
async def test_case17_smoothing_covers_three_paragraphs_and_judges_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 17: one call covers the first three affected paragraphs only; each
    rewrite is guarded on its own, so one rejection leaves the others applied
    ("applied" overall) and the per-paragraph outcomes say what happened."""
    service = _smoothing_service(monkeypatch)
    paragraphs = [
        f'Section {index} opens here. He wrote that "an invented line for section {index}" once. '
        "He moved on afterwards."
        for index in range(4)
    ]
    draft = "\n\n".join(paragraphs) + _FILLER
    kept = [f"Section {index} opens here. He moved on afterwards." for index in range(4)]
    chat_completion = AsyncMock(return_value=(_revisions(
        (0, "Section 0 opens here. Later, he moved on afterwards."),
        (1, "Section 1 opens here. As Mann noted, he moved on afterwards."),
        (2, "Section 2 opens here. Then he moved on afterwards."),
        (3, "Section 3 opens here. Then he moved on afterwards."),
    ), "stop"))
    monkeypatch.setattr(service, "_chat_completion", chat_completion)

    result = await service._check_quotes_node(_quote_check_state(draft, _goitein_bibliography()))

    chat_completion.assert_awaited_once()
    prompt = chat_completion.call_args.kwargs["messages"][0]["content"]
    assert "PARAGRAPH 2:" in prompt and "PARAGRAPH 3:" not in prompt
    assert result["draft_answer"].split("\n\n") == [
        "Section 0 opens here. Later, he moved on afterwards.",
        kept[1],
        "Section 2 opens here. Then he moved on afterwards.",
        kept[3] + _FILLER,
    ]
    assert result["quote_check"]["smoothing"] == "applied"
    assert result["quote_check"]["smoothing_paragraphs"] == {
        0: "applied", 1: "rejected:new_proper_noun", 2: "applied", 3: "skipped",
    }
    assert result["quote_check"]["sentences_deleted"] == 4


def test_case17_summarize_smoothing_status_rule() -> None:
    """Case 17 status rule: applied if any paragraph applied, else the first
    rejection in paragraph order, else skipped."""
    summarize = agent_module.summarize_smoothing
    assert summarize({0: "rejected:new_number", 1: "applied"}) == "applied"
    assert summarize({2: "rejected:too_long", 0: "rejected:new_number"}) == "rejected:new_number"
    assert summarize({0: "skipped"}) == "skipped"
    assert summarize({}) == "skipped"


def test_case17_smoothing_settings_distinguish_unset_from_empty(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Case 17 settings: unset QUOTE_SMOOTHING_MODEL means the verification
    model; an explicit empty value disables smoothing without a warning; an
    explicit model and reasoning effort are used as given."""
    monkeypatch.setenv("ROUTER_MODEL", "qwen/qwen3-4b-2507")
    monkeypatch.setenv("VERIFICATION_MODEL", _SMOOTHING_35B)
    monkeypatch.delenv("QUOTE_SMOOTHING_MODEL", raising=False)
    monkeypatch.delenv("QUOTE_SMOOTHING_REASONING_EFFORT", raising=False)
    default = AgenticRAGService()
    assert default.quote_smoothing_model == _SMOOTHING_35B
    assert default.quote_smoothing_reasoning_effort == "none"

    monkeypatch.setenv("QUOTE_SMOOTHING_MODEL", "")
    with caplog.at_level("WARNING", logger=agent_module.logger.name):
        disabled = AgenticRAGService()
    assert disabled.quote_smoothing_model == ""
    assert not [record for record in caplog.records if "Quote smoothing" in record.getMessage()]

    monkeypatch.setenv("QUOTE_SMOOTHING_MODEL", " lab/smoother-27b ")
    monkeypatch.setenv("QUOTE_SMOOTHING_REASONING_EFFORT", "Low")
    explicit = AgenticRAGService()
    assert explicit.quote_smoothing_model == "lab/smoother-27b"
    assert explicit.quote_smoothing_reasoning_effort == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize("env", [
    {"VERIFICATION_MODEL": "qwen/qwen3-4b-2507", "ROUTER_MODEL": "lab/router-model"},
    {"VERIFICATION_MODEL": "lab/shared-model", "ROUTER_MODEL": "lab/shared-model"},
    {"VERIFICATION_MODEL": _SMOOTHING_35B, "ROUTER_MODEL": "lab/router-model",
     "QUOTE_SMOOTHING_MODEL": "lab/router-model"},
    {"VERIFICATION_MODEL": _SMOOTHING_35B, "ROUTER_MODEL": "lab/router-model",
     "QUOTE_SMOOTHING_MODEL": "Qwen/Qwen3-4B-Instruct-2507"},
], ids=["verification_is_4b_and_smoothing_unset", "smoothing_default_equals_router",
        "smoothing_set_to_router", "smoothing_set_to_a_4b"])
async def test_case17b_4b_guard_disables_smoothing_but_deletion_still_applies(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env: Dict[str, str],
) -> None:
    """Case 17b: a smoothing model that is a 4B or the router model is refused
    at init with one warning; smoothing is "disabled", no model is called, and
    the unverifiable sentence is still deleted."""
    for name in ("VERIFICATION_MODEL", "ROUTER_MODEL", "QUOTE_SMOOTHING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("QUOTE_FAILURE_MODE", "delete")
    with caplog.at_level("WARNING", logger=agent_module.logger.name):
        service = AgenticRAGService()
    assert service.quote_smoothing_model == ""
    warnings = [record for record in caplog.records if "Quote smoothing disabled" in record.getMessage()]
    assert len(warnings) == 1
    chat_completion = AsyncMock(side_effect=AssertionError("smoothing must not call any model"))
    monkeypatch.setattr(service, "_chat_completion", chat_completion)

    result = await service._check_quotes_node(_quote_check_state(_SMOOTH_DRAFT, _goitein_bibliography()))

    chat_completion.assert_not_awaited()
    assert result["quote_check"]["smoothing"] == "disabled"
    assert result["quote_check"]["sentences_deleted"] == 1
    assert result["draft_answer"] == _SMOOTH_DELETED


# ----------------------------------------------------------------------------
# Case 30: chat_stream backstop for a late pipeline failure (C5.2)
# ----------------------------------------------------------------------------

_BACKSTOP_DRAFT = (
    'Goitein began collecting "the documents of the India Book", which he called his '
    "life work." + _FILLER
)
_BACKSTOP_CONTRADICTED = "The piyyut was composed for Passover."
_BACKSTOP_REMAINDER = (
    "Wallenstein describes the manuscript as a unique Kol Nidre composition preserved in "
    "the Cairo Genizah, and he discusses its liturgical setting, its poetic structure, and "
    "the community that used it at considerable length."
)


def _wallenstein_bibliography() -> List[Dict[str, Any]]:
    """Build one bibliography result that contradicts the Passover claim.

    :returns: Fresh bibliography results for the repair-crash test.
    :rtype: List[Dict[str, Any]]
    """
    return [{
        "authors": ["Meir Wallenstein"],
        "title": "A Unique Kol-Nidre Piyyut",
        "extracted_page_number": 489,
        "full_text": "Main text: This Kol Nidre piyyut is unique.",
        "description": "",
        "shelf_marks_mentioned": [],
    }]


async def _drain_stream(service: AgenticRAGService, question: str) -> List[Dict[str, Any]]:
    """Collect every event ``chat_stream`` yields for one question.

    :param service: Service under test.
    :param question: The user's question.
    :returns: The streamed events in order.
    :rtype: List[Dict[str, Any]]
    """
    return [event async for event in service.chat_stream(question)]


def _verification_error_flags(final: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Select the verification_error flags of a streamed final payload.

    :param final: The ``data`` of the stream's final event.
    :returns: Flag dictionaries whose claim_type is ``verification_error``.
    :rtype: List[Dict[str, Any]]
    """
    return [flag for flag in final["flagged_claims"] if flag["claim_type"] == "verification_error"]


@pytest.mark.asyncio
async def test_case30a_stream_backstop_keeps_draft_when_verify_claims_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 30a (C5.2): verify_claims raises a non-outage error (a bug) after
    the draft exists; chat_stream still ends with a final carrying the draft,
    its sources and the verification_error flag, reported as
    VERIFICATION_INCOMPLETE with success=False."""
    service = _delete_mode_service(monkeypatch)
    bibliography = _goitein_bibliography()
    verifier = AsyncMock(side_effect=RuntimeError("verifier bug"))
    monkeypatch.setattr(service, "_call_verifier", verifier)
    _patch_pipeline(monkeypatch, service, bibliography, _BACKSTOP_DRAFT)

    events = await _drain_stream(service, "Who was S. D. Goitein?")

    verifier.assert_awaited_once()
    assert events[-1]["type"] == "final"
    assert [event["type"] for event in events].count("final") == 1
    final = events[-1]["data"]
    assert final["success"] is False
    assert final["error_type"] == "VERIFICATION_INCOMPLETE"
    assert strip_flag_markers(final["answer"]) == _BACKSTOP_DRAFT
    assert final["bibliography_results"] == bibliography
    flags = _verification_error_flags(final)
    assert len(flags) == 1
    assert flags[0]["reason"] == agent_module.VERIFICATION_UNAVAILABLE_REASON
    assert final["metrics"]["verification_unavailable"] is True
    assert final["metrics"]["quote_check"]["mode"] == "delete"
    assert final["metrics"]["stage_calls"]["finalize_response"] == 1
    assert any("RuntimeError in verify_claims" in step for step in final["processing_steps"])
    assert agent_module._REQUEST_METRICS.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", [
    f"{_BACKSTOP_CONTRADICTED} {_BACKSTOP_REMAINDER}",
    f"{_BACKSTOP_CONTRADICTED} Wallenstein describes it as a Kol Nidre piyyut.",
], ids=["long_draft", "short_draft"])
async def test_case30b_stream_backstop_after_repair_crash_neither_deletes_nor_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    draft: str,
) -> None:
    """Case 30b (C5.2): an earlier verify pass left FABRICATED_CLAIMS and
    excluded_claims, then repair_answer raises; the backstop clears them
    before finalizing, so no sentence is deleted and the extractive fallback
    is not used. The short draft (under 200 chars) would get the fallback if
    only excluded_claims were cleared and FABRICATED_CLAIMS kept."""
    service = _delete_mode_service(monkeypatch)
    verifier = AsyncMock(return_value=(json.dumps({
        "verified_claims": [{
            "type": "factual_claim",
            "text": "The piyyut was composed for Passover",
            "verdict": "contradicted",
            "source_number": None,
            "reasoning": "SOURCE 1 identifies the piyyut as a Kol Nidre composition",
        }],
        "summary": "One contradicted claim",
    }), "stop"))
    monkeypatch.setattr(service, "_chat_completion", verifier)
    repair = AsyncMock(side_effect=RuntimeError("repair crashed"))
    monkeypatch.setattr(service, "_repair_answer_node", repair)
    _patch_pipeline(monkeypatch, service, _wallenstein_bibliography(), draft)

    events = await _drain_stream(service, "What occasion was this piyyut written for?")

    verifier.assert_awaited_once()
    repair.assert_awaited_once()
    # Precondition: the verify pass really handed repair a fabricated claim.
    repair_input = repair.await_args.args[0]
    assert repair_input["error_type"] == "FABRICATED_CLAIMS"
    assert repair_input["excluded_claims"]
    final = events[-1]["data"]
    assert events[-1]["type"] == "final"
    assert final["success"] is False
    assert final["error_type"] == "VERIFICATION_INCOMPLETE"
    answer = strip_flag_markers(final["answer"])
    assert answer == draft
    assert not answer.startswith(AgenticRAGService._FALLBACK_LEAD_AFTER_REPAIRS)
    assert len(_verification_error_flags(final)) == 1
    assert final["metrics"]["verification_unavailable"] is True
    assert any("RuntimeError in repair_answer" in step for step in final["processing_steps"])


@pytest.mark.asyncio
async def test_case30c_finalizer_failure_is_reraised_without_a_second_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 30c (C5.2): when finalize_response itself raises, the backstop
    does not run it again; the error propagates and no final is sent."""
    service = _delete_mode_service(monkeypatch)
    monkeypatch.setattr(
        service, "_chat_completion", AsyncMock(return_value=(_ALL_SUPPORTED_VERIFIER_JSON, "stop"))
    )
    finalizer = AsyncMock(side_effect=RuntimeError("finalizer bug"))
    _patch_pipeline(monkeypatch, service, _goitein_bibliography(), _BACKSTOP_DRAFT, finalize=finalizer)

    events: List[Dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="finalizer bug"):
        async for event in service.chat_stream("Who was S. D. Goitein?"):
            events.append(event)

    finalizer.assert_awaited_once()
    assert all(event["type"] != "final" for event in events)
    assert agent_module._REQUEST_METRICS.get() is None


@pytest.mark.asyncio
async def test_case30d_stream_failure_before_the_draft_is_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 30 (C5.2): a failure before any draft exists has nothing to keep,
    so chat_stream re-raises it as before and never finalizes."""
    service = _delete_mode_service(monkeypatch)
    finalizer = AsyncMock(side_effect=AssertionError("must not finalize"))
    _patch_pipeline(monkeypatch, service, _goitein_bibliography(), _BACKSTOP_DRAFT, finalize=finalizer)
    monkeypatch.setattr(
        service, "_synthesize_answer_node", AsyncMock(side_effect=RuntimeError("synthesis bug"))
    )
    service.graph = service._build_graph()

    with pytest.raises(RuntimeError, match="synthesis bug"):
        await _drain_stream(service, "Who was S. D. Goitein?")

    finalizer.assert_not_awaited()


@pytest.mark.asyncio
async def test_case30_chat_has_no_backstop_and_reraises_a_verifier_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C5.2 / §7 #6: chat() gets no backstop; a non-outage verifier error
    (a bug, not ModelUnavailableError or httpx.HTTPError) still propagates."""
    service = _delete_mode_service(monkeypatch)
    finalizer = AsyncMock(side_effect=AssertionError("must not finalize"))
    verifier = AsyncMock(side_effect=RuntimeError("verifier bug"))
    monkeypatch.setattr(service, "_call_verifier", verifier)
    _patch_pipeline(monkeypatch, service, _goitein_bibliography(), _BACKSTOP_DRAFT, finalize=finalizer)

    with pytest.raises(RuntimeError, match="verifier bug"):
        await service.chat("Who was S. D. Goitein?")

    verifier.assert_awaited_once()
    finalizer.assert_not_awaited()


# ----------------------------------------------------------------------------
# Review fixes: repair -> check_quotes loop, rollback gate, metrics, outages
# ----------------------------------------------------------------------------

_LOOP_FABRICATED = 'Wallenstein wrote that "the poem was recited only at dawn" in Fustat.'
_LOOP_DRAFT = f"{_BACKSTOP_CONTRADICTED} {_LOOP_FABRICATED} {_BACKSTOP_REMAINDER}"
_CONTRADICTED_VERIFIER_JSON = json.dumps({
    "verified_claims": [{
        "type": "factual_claim",
        "text": "The piyyut was composed for Passover",
        "verdict": "contradicted",
        "source_number": None,
        "reasoning": "SOURCE 1 identifies the piyyut as a Kol Nidre composition",
    }],
    "summary": "One contradicted claim",
})


@pytest.mark.asyncio
@pytest.mark.parametrize("revision, deleted_total, quotations_total, unverified_total", [
    (
        "The piyyut was composed for Kol Nidre, and Wallenstein wrote that \"it was sung only in "
        f"Aleppo and Damascus by the cantors\". {_BACKSTOP_REMAINDER}",
        2,
        2,
        2,
    ),
    (f"The piyyut was composed for Kol Nidre. {_BACKSTOP_REMAINDER}", 1, 1, 1),
], ids=["repair_adds_fabricated_quote", "clean_repair"])
async def test_repaired_draft_goes_back_through_the_quote_check(
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
    deleted_total: int,
    quotations_total: int,
    unverified_total: int,
) -> None:
    """Delete mode through the compiled graph: the first check deletes a
    fabricated quote, the 35B contradicts a claim, and the 27B repair goes
    back through check_quotes, which deletes any quotation the repair
    invented. sentences_deleted totals both checks, so the closing note
    survives even a clean repair. A quotation the repair invented counts in
    both quotations and unverified, so the quote-level partition holds."""
    service = _delete_mode_service(monkeypatch)
    _patch_pipeline(monkeypatch, service, _wallenstein_bibliography(), _LOOP_DRAFT)
    verifier = AsyncMock(side_effect=[
        (_CONTRADICTED_VERIFIER_JSON, "stop"),
        (_ALL_SUPPORTED_VERIFIER_JSON, "stop"),
    ])
    monkeypatch.setattr(service, "_chat_completion", verifier)
    model_27b = AsyncMock(return_value=json.dumps(
        {"revisions": [{"paragraph_index": 0, "revised_text": revision}]}
    ))
    monkeypatch.setattr(service, "_call_llm", model_27b)

    response = await service.chat("What occasion was this piyyut written for?")

    model_27b.assert_awaited_once()
    assert verifier.await_count == 2
    assert response.metrics["stage_calls"]["check_quotes"] == 2
    assert response.metrics["repair_attempts"] == 1
    for text in ("Passover", "recited only at dawn", "Aleppo"):
        assert text not in response.answer
    assert _BACKSTOP_REMAINDER in response.answer
    assert response.answer.count(_QUOTE_NOTE_EN) == 1
    quote_metrics = response.metrics["quote_check"]
    assert quote_metrics["sentences_deleted"] == deleted_total
    assert quote_metrics["quotations"] == quotations_total
    assert quote_metrics["unverified"] == unverified_total
    assert quote_metrics["quotations"] == (
        quote_metrics["verified"] + quote_metrics["exempt_title"]
        + quote_metrics["exempt_question"] + quote_metrics["unverified"]
    )
    assert quote_metrics["smoothing"] == "disabled"


@pytest.mark.asyncio
async def test_repair_mode_keeps_the_verifiers_deterministic_quote_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QUOTE_FAILURE_MODE=repair rollback: check_quotes only records the mode,
    so the verify node's legacy quote gate must still hard-fail a fabricated
    quotation."""
    monkeypatch.setenv("QUOTE_FAILURE_MODE", "repair")
    monkeypatch.setenv("QUOTE_SMOOTHING_MODEL", "")
    service = AgenticRAGService()
    monkeypatch.setattr(
        service, "_chat_completion", AsyncMock(return_value=(_ALL_SUPPORTED_VERIFIER_JSON, "stop"))
    )
    fabricated = "the traders never used written contracts at all"
    state = _quote_check_state(f'He wrote that "{fabricated}" in Aden.{_FILLER}', _goitein_bibliography())

    state = await service._check_quotes_node(state)
    assert state["quote_check"] == {"mode": "repair"}
    result = await service._verify_claims_node(state)

    assert result["error_type"] == "FABRICATED_CLAIMS"
    assert [claim["text"] for claim in result["excluded_claims"]] == [fabricated]


@pytest.mark.asyncio
async def test_check_quotes_matches_only_this_turns_question_never_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec C1: an earlier answer (or a query rewritten from it) cannot
    launder an unverifiable quotation as an echo of the question."""
    service = _delete_mode_service(monkeypatch)
    fabricated = "the traders never used written contracts at all"
    draft = f'Goitein read the letters closely. He wrote that "{fabricated}" in Aden.{_FILLER}'
    state = _quote_check_state(draft, _goitein_bibliography())
    state["user_query"] = "And what did he say about contracts?"
    state["conversation_history"] = [
        {"role": "user", "content": "What did Goitein say about Aden?"},
        {"role": "assistant", "content": f'He wrote that "{fabricated}" in Aden.'},
    ]
    state["resolved_query"] = f'What did Goitein mean by "{fabricated}"?'

    result = await service._check_quotes_node(state)

    assert fabricated not in result["draft_answer"]
    assert result["quote_check"]["counts"]["exempt_question"] == 0
    assert result["quote_check"]["sentences_deleted"] == 1


@pytest.mark.asyncio
async def test_repair_protects_quotes_the_quote_check_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In delete mode the repair's protected-quote list uses the quote check's
    matcher, so a punctuation-variant quote it verified survives a 27B
    revision that drops it (deterministic removal of only the contradicted
    sentence is used instead)."""
    service = _delete_mode_service(monkeypatch)
    quote_sentence = 'Goitein spoke of "the documents of the India Book."'
    contradicted = "The court met on Sundays in Fustat."
    model_27b = AsyncMock(return_value=json.dumps(
        {"revisions": [{"paragraph_index": 0, "revised_text": _FILLER.strip()}]}
    ))
    monkeypatch.setattr(service, "_call_llm", model_27b)
    state = _quote_check_state(f"{quote_sentence} {contradicted}{_FILLER}", _goitein_bibliography())
    state.update({
        "excluded_claims": [{
            "type": "factual_claim",
            "text": "The court met on Sundays in Fustat",
            "reason": "SOURCE 1 says Monday and Thursday",
        }],
        "retry_count": 1,
        "error": "contradicted",
        "error_type": "FABRICATED_CLAIMS",
    })

    result = await service._repair_answer_node(state)

    prompt = model_27b.call_args.kwargs["messages"][0]["content"]
    assert '- "the documents of the India Book."' in prompt
    assert quote_sentence in result["draft_answer"]
    assert contradicted not in result["draft_answer"]


def test_exhaustion_fallback_after_a_noop_repair_does_not_claim_three_repairs() -> None:
    """A no-op repair ends the loop after one round, so the fallback lead
    leaves out the repair count."""
    state = {
        "draft_answer": 'Goitein spoke of "the documents of the India Book." Nothing else survived.',
        "bibliography_results": _goitein_bibliography(),
        "graph_results": [],
        "repair_noop": True,
    }

    fallback = AgenticRAGService._build_verified_evidence_fallback(state, quote_failure_mode="delete")

    assert fallback is not None
    assert fallback.startswith(AgenticRAGService._FALLBACK_LEAD_AFTER_NOOP_REPAIR + "\n\n")
    assert "three" not in fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", [
    "The piyyut was composed for Passover. It survives in a single fragment.\n",
    "The piyyut was composed for Passover. It survives in a single fragment.\n\n\n\nA second paragraph stays.",
], ids=["trailing_newline", "blank_line_run"])
async def test_case20_revision_differing_only_in_whitespace_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
    draft: str,
) -> None:
    """C4: rejoining the paragraphs may change blank lines only; an echoed
    paragraph is still a no-op, so no second verify and repair round runs."""
    service = AgenticRAGService()
    echoed = "The piyyut was composed for Passover. It survives in a single fragment."
    monkeypatch.setattr(service, "_call_llm", AsyncMock(return_value=json.dumps(
        {"revisions": [{"paragraph_index": 0, "revised_text": echoed}]}
    )))
    state = _quote_check_state(draft, [])
    state.update({
        "excluded_claims": [{
            "type": "factual_claim",
            "text": "The piyyut was composed for Passover",
            "reason": "SOURCE 1 says Kol Nidre",
        }],
        "retry_count": 1,
        "error": "contradicted",
        "error_type": "FABRICATED_CLAIMS",
    })

    result = await service._repair_answer_node(state)

    assert result["repair_noop"] is True
    assert result["error_type"] == "FABRICATED_CLAIMS"
    assert agent_module.route_after_repair(result) == "finalize"


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"json": {"error": "model crashed"}},
    {"text": "<html>Bad gateway</html>"},
    {"json": {"choices": []}},
], ids=["error_object", "html", "no_choices"])
async def test_malformed_2xx_completion_is_an_outage_so_smoothing_keeps_the_deletion(
    monkeypatch: pytest.MonkeyPatch,
    body: Dict[str, Any],
) -> None:
    """A 2xx body that is not a chat completion raises ModelUnavailableError,
    so the optional smoothing call is 'unavailable' and the quote deletion
    still applies (instead of a KeyError discarding it)."""
    service = _smoothing_service(monkeypatch)
    monkeypatch.setattr(
        service, "resolve_model",
        AsyncMock(side_effect=lambda model, force_refresh=False: model),
    )

    def handle(request: httpx.Request) -> httpx.Response:
        """Answer every chat completion with the malformed body.

        :param request: Outgoing request to LM Studio.
        :returns: A 200 response that is not a chat completion.
        :rtype: httpx.Response
        """
        return httpx.Response(200, **body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        agent_module.httpx, "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=httpx.MockTransport(handle), **kwargs),
    )

    with pytest.raises(ModelUnavailableError):
        await service._chat_completion(messages=[{"role": "user", "content": "hi"}], model=_SMOOTHING_35B)
    result = await service._check_quotes_node(_quote_check_state(_SMOOTH_DRAFT, _goitein_bibliography()))

    assert result["draft_answer"] == _SMOOTH_DELETED
    assert result["quote_check"]["smoothing"] == "rejected:unavailable"
    assert result["quote_check"]["sentences_deleted"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status, body, degrades", [
    (200, {"json": {"error": "model crashed"}}, True),
    (200, {"json": {"choices": []}}, True),
    (500, {"text": "server error"}, False),
], ids=["malformed_error_object", "malformed_no_choices", "server_error"])
async def test_follow_up_resolver_degrades_on_a_malformed_reply_but_not_on_an_outage(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: Dict[str, Any],
    degrades: bool,
) -> None:
    """A malformed 2xx router reply makes the optional follow-up resolver fall
    back to the literal message (None), as before the malformed-body change;
    a genuine outage still raises ModelUnavailableError and aborts."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service, "resolve_model",
        AsyncMock(side_effect=lambda model, force_refresh=False: model),
    )

    def handle(request: httpx.Request) -> httpx.Response:
        """Answer every chat completion with the canned reply.

        :param request: Outgoing request to LM Studio.
        :returns: The canned response.
        :rtype: httpx.Response
        """
        return httpx.Response(status, **body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        agent_module.httpx, "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=httpx.MockTransport(handle), **kwargs),
    )

    if degrades:
        with pytest.raises(agent_module.MalformedCompletionError):
            await service._chat_completion(messages=[{"role": "user", "content": "hi"}], model="router-m")
        assert await service._call_resolver("system", "user") is None
    else:
        with pytest.raises(ModelUnavailableError):
            await service._call_resolver("system", "user")


@pytest.mark.asyncio
async def test_quote_check_rerun_after_repair_keeps_request_level_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-check that deletes nothing keeps the first run's counts and the
    smoothing status of the run that deleted, so the metrics do not report a
    deletion with zero unverified quotations and smoothing 'skipped'."""
    service = _smoothing_service(monkeypatch)
    monkeypatch.setattr(service, "_chat_completion", AsyncMock(
        return_value=(_revisions((0, _SMOOTH_ACCEPTED)), "stop")
    ))
    first = await service._check_quotes_node(_quote_check_state(_SMOOTH_DRAFT, _goitein_bibliography()))
    monkeypatch.setattr(service, "_chat_completion", AsyncMock(
        side_effect=AssertionError("nothing was deleted, so nothing is smoothed")
    ))

    second = await service._check_quotes_node(first)

    quote_check = second["quote_check"]
    assert quote_check["counts"]["quotations"] == 1
    assert quote_check["counts"]["unverified"] == 1
    assert quote_check["sentences_deleted"] == 1
    assert quote_check["smoothing"] == "applied"
    assert quote_check["smoothing_paragraphs"] == {0: "applied"}
    metrics = agent_module.quote_check_metrics(quote_check)
    assert (metrics["unverified"], metrics["smoothing"]) == (1, "applied")
    # The processing step still reports what this run found.
    assert second["processing_steps"][-1].startswith("Quote check: 0/0 quotations verified")
