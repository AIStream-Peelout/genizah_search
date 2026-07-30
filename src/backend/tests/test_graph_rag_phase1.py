"""Focused Phase 1 regression tests for graph-first scholar retrieval."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend import lms_agentic_search as agent_module
from src.backend.lms_agentic_search import (
    AgenticRAGService,
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
    monkeypatch.setattr(
        service,
        "_call_llm",
        AsyncMock(return_value=(
            '{"verified_claims": [{"type": "attribution", '
            '"text": "Arrant argues that this codex was copied in 1100", '
            '"verdict": "unsupported", "source_number": null, '
            '"reasoning": "SOURCE 1 names Arrant but gives no such date"}], '
            '"summary": "Unsupported attribution"}'
        )),
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
async def test_contradicted_claim_triggers_targeted_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim the sources contradict is a hard failure that forces repair."""
    service = AgenticRAGService()
    monkeypatch.setattr(
        service,
        "_call_llm",
        AsyncMock(return_value=(
            '{"verified_claims": [{"type": "factual_claim", '
            '"text": "The piyyut was composed for Passover", '
            '"verdict": "contradicted", "source_number": null, '
            '"reasoning": "SOURCE 1 identifies the piyyut as a Kol Nidre composition"}], '
            '"summary": "One contradicted claim"}'
        )),
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
    llm_call = AsyncMock(return_value='{"verified_claims": [], "summary": "ok"}')
    monkeypatch.setattr(service, "_call_llm", llm_call)
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
    llm_call = AsyncMock(return_value="not json")
    monkeypatch.setattr(service, "_call_llm", llm_call)
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
