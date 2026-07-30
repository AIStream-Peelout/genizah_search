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
    async def test_work_manuscripts_are_collected_and_linked(self, monkeypatch):
        """A work's underlying fragments must be linkable even when its pages print none."""
        from unittest.mock import AsyncMock
        from src.backend import lms_agentic_search as agent_module

        service = AgenticRAGService()
        monkeypatch.setattr(
            agent_module.neo4j_service, "get_fragments_for_works",
            AsyncMock(return_value={
                "Jewish Marriage in Palestine: The Kettubba texts": [
                    {"es_doc_id": "Cambridge_CUL_T_S_NS_J48", "canonical_shelfmark": "T_S_NS_J48"},
                    {"es_doc_id": "Not_In_Collection_999", "canonical_shelfmark": "X_999"},
                ],
            }),
        )
        monkeypatch.setattr(
            AgenticRAGService, "_fetch_display_shelfmarks",
            AsyncMock(return_value={"Cambridge_CUL_T_S_NS_J48": "T-S NS J48"}),
        )

        state = {
            "bibliography_results": [{"title": "Jewish Marriage in Palestine: The Kettubba texts"}],
            "shelf_mark_lookup": {},
            "processing_steps": [],
        }
        await service._collect_work_manuscripts(state)

        entries = state["work_manuscripts"]["Jewish Marriage in Palestine: The Kettubba texts"]
        assert entries == [{"shelf_mark": "T-S NS J48", "doc_id": "Cambridge_CUL_T_S_NS_J48"}]
        # Fragments the collection does not hold are dropped, not shown unopenable.
        assert all(e["doc_id"] != "Not_In_Collection_999" for e in entries)
        # Registered for linkification of the answer body too.
        assert state["shelf_mark_lookup"]["T-S NS J48"] == "Cambridge_CUL_T_S_NS_J48"

    @pytest.mark.asyncio
    async def test_finalize_renders_manuscripts_behind_works(self):
        """The primary-source bridge appears as clickable links in the answer."""
        service = AgenticRAGService()
        state: AgenticRAGState = {
            "draft_answer": "Friedman traced the evolution of the ketubba.",
            "verification_summary": {},
            "shelf_mark_lookup": {},
            "shelf_marks_in_bibliography": set(),
            "primary_source_results": [],
            "work_manuscripts": {
                "Jewish Marriage in Palestine": [
                    {"shelf_mark": "T-S NS J48", "doc_id": "Cambridge_CUL_T_S_NS_J48"},
                ]
            },
            "error_type": None,
            "processing_steps": [],
        }

        result = await service._finalize_response_node(state)

        assert "Manuscripts these works are based on" in result["final_answer"]
        assert "[T-S NS J48](doc:Cambridge_CUL_T_S_NS_J48)" in result["final_answer"]

    def test_publication_references_are_not_treated_as_shelfmarks(self):
        """Index metadata mixes real marks with publication items; keep only marks."""
        from src.backend.lms_agentic_search import filter_manuscript_shelfmarks

        raw = [
            "Halper 331", "T-S Ar. 38.11", "DJD II, no. 20", "DJD II, no. 21",
            "Babata's Ketubba", "TS", "Or. 1080 J291", "Bodl. MS heb. d. 66",
        ]
        kept = filter_manuscript_shelfmarks(raw)

        assert kept == [
            "Halper 331", "T-S Ar. 38.11", "Or. 1080 J291", "Bodl. MS heb. d. 66",
        ]
        assert filter_manuscript_shelfmarks(None) == []

    @pytest.mark.asyncio
    async def test_cited_shelfmarks_absent_from_collection_are_still_listed(self):
        """Cited marks we cannot link must still be visible to the reader."""
        service = AgenticRAGService()
        state: AgenticRAGState = {
            "draft_answer": "Levin published a piyyutic Grace After Meals.",
            "verification_summary": {},
            "shelf_mark_lookup": {"T-S 12.388": "Cambridge_CUL_T_S_12_388"},
            "shelf_marks_in_bibliography": {"T-S 12.388", "WR IV. 329"},
            "primary_source_results": [{
                "doc_id": "Cambridge_CUL_T_S_12_388",
                "shelf_mark": "T-S 12.388",
                "title": "A fragment",
            }],
            "error_type": None,
            "processing_steps": [],
        }

        result = await service._finalize_response_node(state)
        answer = result["final_answer"]

        # Held fragment: clickable catalog entry.
        assert "[T-S 12.388](doc:Cambridge_CUL_T_S_12_388)" in answer
        # Cited but not held: listed plainly, never as a broken link.
        assert "not in this collection" in answer
        assert "WR IV. 329" in answer
        assert "(doc:" not in answer.split("not in this collection")[1]

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
