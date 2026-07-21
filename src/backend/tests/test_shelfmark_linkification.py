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
