import pytest
from src.backend.lms_agentic_search import AgenticRAGService, AgenticRAGState

class TestShelfmarkLinkification:
    @pytest.mark.asyncio
    async def test_basic_linkification(self):
        service = AgenticRAGService()
        state: AgenticRAGState = {
            "shelf_mark_lookup": {
                "T-S 8J22.22": "doc1"
            }
        }
        text = "This document is T-S 8J22.22."
        result = service._linkify_all_shelfmarks(text, state)
        assert result == "This document is [T-S 8J22.22](doc:doc1)."

    @pytest.mark.asyncio
    async def test_longest_match_priority(self):
        service = AgenticRAGService()
        state: AgenticRAGState = {
            "shelf_mark_lookup": {
                "T-S 8J22": "short",
                "T-S 8J22.22": "long"
            }
        }
        text = "Checking T-S 8J22.22 and T-S 8J22."
        result = service._linkify_all_shelfmarks(text, state)
        # Should match long one first, then short one
        assert "[T-S 8J22.22](doc:long)" in result
        assert "[T-S 8J22](doc:short)" in result

    @pytest.mark.asyncio
    async def test_prevent_double_linking(self):
        service = AgenticRAGService()
        state: AgenticRAGState = {
            "shelf_mark_lookup": {
                "T-S 8J22.22": "doc1"
            }
        }
        # Text already contains a link
        text = "Already linked: [T-S 8J22.22](doc:doc1). Also mention T-S 8J22.22 again."
        result = service._linkify_all_shelfmarks(text, state)
        # Should only link the second mention
        assert "Already linked: [T-S 8J22.22](doc:doc1)" in result
        assert "mention [T-S 8J22.22](doc:doc1) again" in result
        assert result.count("[T-S 8J22.22](doc:doc1)") == 2

    @pytest.mark.asyncio
    async def test_case_insensitivity(self):
        service = AgenticRAGService()
        state: AgenticRAGState = {
            "shelf_mark_lookup": {
                "T-S 8J22.22": "doc1"
            }
        }
        text = "Lower case: t-s 8j22.22"
        result = service._linkify_all_shelfmarks(text, state)
        assert result == "Lower case: [t-s 8j22.22](doc:doc1)"

    @pytest.mark.asyncio
    async def test_empty_map(self):
        service = AgenticRAGService()
        state: AgenticRAGState = {}
        text = "No map T-S 8J22.22"
        result = service._linkify_all_shelfmarks(text, state)
        assert result == text

    @pytest.mark.asyncio
    async def test_finalize_node_integration(self):
        service = AgenticRAGService()
        state: AgenticRAGState = {
            "draft_answer": "According to T-S 8J22.22...",
            "verification_summary": {},
            "shelf_mark_lookup": {"T-S 8J22.22": "doc1"},
            "primary_source_results": [],
            "error_type": None,
            "processing_steps": []
        }
        result = await service._finalize_response_node(state)
        assert "[T-S 8J22.22](doc:doc1)" in result["final_answer"]


class TestShelfmarkResolutionGuarantees:
    """Regression guarantees: cited fragments that exist in the index must be
    linked; near-miss fuzzy hits must never be; graph fragments with known ES
    document ids must link without a search round-trip."""

    def test_institution_prefix_variants_are_equivalent(self):
        from src.backend.lms_agentic_search import shelfmarks_equivalent

        # Same fragment, index stores the institution-prefixed form.
        assert shelfmarks_equivalent(
            "Rylands Genizah Fragment 1", "Manchester: Rylands Genizah fragment 1"
        )
        assert shelfmarks_equivalent("B 3989", "Manchester: B 3989")
        assert shelfmarks_equivalent("T-S Misc. 35.45", "T-S Misc. 35.45")
        # Different fragments must never be treated as the same document.
        assert not shelfmarks_equivalent("T-S H3.111", "T-S H3.101")
        assert not shelfmarks_equivalent("ENA NS 12.5", "ENA NS 18.5")
        assert not shelfmarks_equivalent("T-S 12.388", "T-S 12.38")

    @pytest.mark.asyncio
    async def test_resolver_links_institution_prefixed_index_hit(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock
        from src.backend import lms_agentic_search as agent_module

        service = AgenticRAGService()
        hit = MagicMock()
        hit.doc_id = "Manchester_JRL_Genizah_fragment_1"
        hit.metadata.shelf_mark = "Manchester: Rylands Genizah fragment 1"
        hit.similarity_score = 0.9
        response = MagicMock()
        response.results = [hit]
        monkeypatch.setattr(
            agent_module.search_service, "search_by_shelfmark", AsyncMock(return_value=response)
        )
        record_calls = []
        monkeypatch.setattr(
            agent_module.missing_fragment_tracker, "record",
            lambda **kwargs: record_calls.append(kwargs),
        )

        state = {"shelf_mark_lookup": {}, "primary_source_results": [], "user_query": "q"}
        await service._resolve_unlinked_shelfmarks(
            "The manuscript Rylands Genizah Fragment 1 is worn.", state
        )

        assert state["shelf_mark_lookup"]["Rylands Genizah Fragment 1"] == (
            "Manchester_JRL_Genizah_fragment_1"
        )
        assert record_calls == []
        linkified = service._linkify_all_shelfmarks(
            "The manuscript Rylands Genizah Fragment 1 is worn.", state
        )
        assert "[Rylands Genizah Fragment 1](doc:Manchester_JRL_Genizah_fragment_1)" in linkified

    @pytest.mark.asyncio
    async def test_resolver_rejects_near_miss_and_records_missing(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock
        from src.backend import lms_agentic_search as agent_module

        service = AgenticRAGService()
        hit = MagicMock()
        hit.doc_id = "Cambridge_CUL_T_S_H3_101"
        hit.metadata.shelf_mark = "T-S H3.101"
        hit.similarity_score = 0.9
        response = MagicMock()
        response.results = [hit]
        monkeypatch.setattr(
            agent_module.search_service, "search_by_shelfmark", AsyncMock(return_value=response)
        )
        record_calls = []
        monkeypatch.setattr(
            agent_module.missing_fragment_tracker, "record",
            lambda **kwargs: record_calls.append(kwargs),
        )

        state = {"shelf_mark_lookup": {}, "primary_source_results": [], "user_query": "q"}
        await service._resolve_unlinked_shelfmarks("A reshut in T-S H3.111.", state)

        assert state["shelf_mark_lookup"] == {}
        assert len(record_calls) == 1
        assert record_calls[0]["shelf_mark"] == "T-S H3.111"
        assert record_calls[0]["nearest_match"] == "T-S H3.101"

    @pytest.mark.asyncio
    async def test_graph_fragment_samples_link_without_search(self, monkeypatch):
        from unittest.mock import AsyncMock
        from src.backend import lms_agentic_search as agent_module
        from src.backend.lms_agentic_search import QueryPlan, SearchAction
        from src.backend.search_bibliography import BibliographySearchResponse

        service = AgenticRAGService()
        monkeypatch.setattr(
            agent_module.neo4j_service, "find_scholars",
            AsyncMock(return_value=[{"name": "Test Scholar", "match_score": 1.0}]),
        )
        monkeypatch.setattr(
            agent_module.neo4j_service, "get_scholar_rag_evidence",
            AsyncMock(return_value={
                "scholar": {"name": "Test Scholar"},
                "works": [{
                    "title": "A Work", "year": 1970, "article_id": "x",
                    "referenced_fragment_count": 3,
                    "referenced_fragment_samples": [
                        {"shelfmark": "T_S_12_388", "es_doc_id": "Cambridge_CUL_T_S_12_388"},
                        {"shelfmark": "1024", "es_doc_id": "bogus_numeric"},
                        {"shelfmark": "XXII", "es_doc_id": "bogus_roman"},
                    ],
                }],
                "studied_fragment_count": 1,
                "studied_fragment_samples": [
                    {"shelfmark": "p_Heid_Hebr_12", "es_doc_id": "Heidelberg_p_Heid_Hebr_12"},
                ],
                "relationships": [],
            }),
        )
        monkeypatch.setattr(
            agent_module.bibliography_search_service, "search_by_author",
            AsyncMock(return_value=BibliographySearchResponse(
                results=[], count=0, processing_time_ms=0.0,
            )),
        )

        state = {
            "user_query": "Who is Test Scholar?",
            "query_plan": QueryPlan(
                actions=[SearchAction(search_type="graph_scholar", query="Test Scholar")],
                needs_primary_secondary_linking=True,
                reasoning="test",
            ),
            "processing_steps": [],
            "error": None,
            "error_type": None,
        }
        result = await service._execute_searches_node(state)

        lookup = result["shelf_mark_lookup"]
        # Real fragment ids from the graph are registered for direct linking …
        assert lookup["T_S_12_388"] == "Cambridge_CUL_T_S_12_388"
        assert lookup["p_Heid_Hebr_12"] == "Heidelberg_p_Heid_Hebr_12"
        # … while junk sample identifiers (bare numbers, roman numerals) that
        # would corrupt answer text if linkified are excluded.
        assert "1024" not in lookup
        assert "XXII" not in lookup

        linkified = service._linkify_all_shelfmarks(
            "Fragments include T_S_12_388 and p_Heid_Hebr_12; also 1024 pages.", result
        )
        assert "[T_S_12_388](doc:Cambridge_CUL_T_S_12_388)" in linkified
        assert "[p_Heid_Hebr_12](doc:Heidelberg_p_Heid_Hebr_12)" in linkified
        assert "[1024]" not in linkified
