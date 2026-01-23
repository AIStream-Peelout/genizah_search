# test_agentic_rag.py
"""
Test suite for the agentic RAG service with conversation history support.
Tests both individual components and full end-to-end workflows.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import json

from src.backend.lms_agentic_search import (
    agentic_rag_service,
    AgenticRAGService,

    QueryPlan,
    SearchAction
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing follow-ups"""
    return [
        {
            "timestamp": "2024-01-15T10:00:00",
            "user_query": "Tell me about T-S 8J22.22",
            "answer": "T-S 8J22.22 is a business letter from the Cairo Genizah collection, written in Judeo-Arabic. According to **'A Mediterranean Society' by S.D. Goitein, p. 245**, this document discusses trade between Egypt and India in the 11th century.",
            "query_plan": {
                "actions": [
                    {"search_type": "primary_shelfmark", "query": "T-S 8J22.22", "exact_match": True},
                    {"search_type": "bibliography_hybrid", "query": "T-S 8J22.22"}
                ],
                "needs_primary_secondary_linking": True,
                "is_followup": False,
                "references_previous_results": False,
                "reasoning": "User asked about specific shelf mark"
            },
            "bibliography_results": [
                {
                    "doc_id": "goitein_med_soc_p245",
                    "title": "A Mediterranean Society",
                    "authors": ["S.D. Goitein"],
                    "author": "S.D. Goitein",
                    "extracted_page_number": "245",
                    "full_text": "T-S 8J22.22 is a fascinating example of the India trade documents...",
                    "shelf_marks_mentioned": ["T-S 8J22.22", "T-S 8J23.1"],
                    "similarity_score": 0.95
                }
            ],
            "primary_source_results": [
                {
                    "doc_id": "TS-8J22-22",
                    "shelf_mark": "T-S 8J22.22",
                    "title": "Business Letter (Judeo-Arabic)",
                    "description": "11th century business letter discussing trade",
                    "transcription": "בשם רחמנא...",
                    "translation": "In the name of the Merciful...",
                    "similarity_score": 1.0
                }
            ]
        }
    ]


@pytest.fixture
def complex_conversation_history():
    """Complex multi-turn conversation for advanced testing"""
    return [
        {
            "timestamp": "2024-01-15T10:00:00",
            "user_query": "What documents discuss the India trade?",
            "answer": "Several documents in the Cairo Genizah collection discuss the India trade. According to **'A Mediterranean Society' by S.D. Goitein**, key documents include T-S 8J22.22, T-S 10J12.9, and T-S NS 320.55.",
            "bibliography_results": [
                {
                    "doc_id": "goitein_india_trade",
                    "title": "A Mediterranean Society",
                    "authors": ["S.D. Goitein"],
                    "shelf_marks_mentioned": ["T-S 8J22.22", "T-S 10J12.9", "T-S NS 320.55"],
                    "similarity_score": 0.92
                }
            ],
            "primary_source_results": []
        },
        {
            "timestamp": "2024-01-15T10:01:00",
            "user_query": "Tell me more about the first one",
            "answer": "T-S 8J22.22 is a business letter from the 11th century written in Judeo-Arabic...",
            "bibliography_results": [],
            "primary_source_results": [
                {
                    "doc_id": "TS-8J22-22",
                    "shelf_mark": "T-S 8J22.22",
                    "description": "11th century business letter",
                    "similarity_score": 1.0
                }
            ]
        }
    ]


# ============================================================================
# Unit Tests - Reference Resolution
# ============================================================================

class TestReferenceResolution:
    """Test the reference resolution node"""

    @pytest.mark.asyncio
    async def test_resolve_simple_pronoun(self, sample_conversation_history):
        """Test resolving 'it' to specific document"""
        service = AgenticRAGService()

        state = {
            "user_query": "What else do we know about it?",
            "conversation_history": [ConversationTurn(**h) for h in sample_conversation_history],
            "processing_steps": []
        }

        result = await service._resolve_references_node(state)

        # Should resolve "it" to T-S 8J22.22
        assert "T-S 8J22.22" in result["resolved_query"] or "document" in result["resolved_query"]
        assert result["resolved_query"] != state["user_query"]
        print(f"✓ Resolved: '{state['user_query']}' → '{result['resolved_query']}'")

    @pytest.mark.asyncio
    async def test_resolve_that_document(self, sample_conversation_history):
        """Test resolving 'that document'"""
        service = AgenticRAGService()

        state = {
            "user_query": "What did scholars say about that document?",
            "conversation_history": [ConversationTurn(**h) for h in sample_conversation_history],
            "processing_steps": []
        }

        result = await service._resolve_references_node(state)

        assert "T-S 8J22.22" in result["resolved_query"]
        print(f"✓ Resolved: '{state['user_query']}' → '{result['resolved_query']}'")

    @pytest.mark.asyncio
    async def test_no_resolution_needed_standalone_query(self):
        """Test that standalone queries pass through unchanged"""
        service = AgenticRAGService()

        state = {
            "user_query": "What documents discuss trade in the Genizah?",
            "conversation_history": [],
            "processing_steps": []
        }

        result = await service._resolve_references_node(state)

        assert result["resolved_query"] == state["user_query"]
        print(f"✓ Standalone query unchanged: '{result['resolved_query']}'")


# ============================================================================
# Unit Tests - Query Routing
# ============================================================================

class TestQueryRouting:
    """Test the query routing and planning"""

    @pytest.mark.asyncio
    async def test_route_shelfmark_query(self):
        """Test routing a shelf mark query"""
        service = AgenticRAGService()

        state = {
            "user_query": "Tell me about T-S 8J22.22",
            "resolved_query": "Tell me about T-S 8J22.22",
            "conversation_history": [],
            "context_entities": {"documents": [], "authors": [], "topics": []},
            "processing_steps": []
        }

        result = await service._route_query_node(state)

        plan = result["query_plan"]
        assert plan is not None
        assert len(plan.actions) >= 1

        # Should include shelfmark search
        has_shelfmark_search = any(
            action.search_type == "primary_shelfmark"
            for action in plan.actions
        )
        assert has_shelfmark_search
        print(f"✓ Routed shelf mark query: {len(plan.actions)} actions")
        print(f"  Actions: {[a.search_type for a in plan.actions]}")

    @pytest.mark.asyncio
    async def test_route_conceptual_query(self):
        """Test routing a conceptual query"""
        service = AgenticRAGService()

        state = {
            "user_query": "What did Goitein say about medieval trade?",
            "resolved_query": "What did Goitein say about medieval trade?",
            "conversation_history": [],
            "context_entities": {"documents": [], "authors": [], "topics": []},
            "processing_steps": []
        }

        result = await service._route_query_node(state)

        plan = result["query_plan"]
        assert plan is not None

        # Should use bibliography search
        has_bib_search = any(
            action.search_type.startswith("bibliography")
            for action in plan.actions
        )
        assert has_bib_search
        print(f"✓ Routed conceptual query: {len(plan.actions)} actions")

    @pytest.mark.asyncio
    async def test_route_followup_no_new_searches(self, sample_conversation_history):
        """Test that simple follow-ups might not need new searches"""
        service = AgenticRAGService()

        state = {
            "user_query": "Who wrote that?",
            "resolved_query": "Who wrote T-S 8J22.22?",
            "conversation_history": [ConversationTurn(**h) for h in sample_conversation_history],
            "context_entities": {
                "documents": ["T-S 8J22.22"],
                "authors": ["S.D. Goitein"],
                "topics": []
            },
            "processing_steps": []
        }

        result = await service._route_query_node(state)

        plan = result["query_plan"]
        assert plan.is_followup
        assert plan.references_previous_results

        # May have zero actions if it can use previous results
        print(f"✓ Follow-up query: {len(plan.actions)} actions, is_followup={plan.is_followup}")


# ============================================================================
# Integration Tests - Full Workflows
# ============================================================================

class TestFullWorkflows:
    """Test complete end-to-end workflows"""

    @pytest.mark.asyncio
    async def test_simple_standalone_query(self):
        """Test a simple standalone query"""
        print("\n" + "=" * 80)
        print("TEST: Simple Standalone Query")
        print("=" * 80)

        response = await agentic_rag_service.chat(
            user_query="What documents in the Genizah discuss business letters?",
            conversation_history=None
        )

        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.query_plan is not None

        print(f"\n✓ Query: What documents in the Genizah discuss business letters?")
        print(f"✓ Plan: {response.query_plan.reasoning}")
        print(f"✓ Actions: {[a.search_type for a in response.query_plan.actions]}")
        print(f"✓ Bibliography results: {len(response.bibliography_results)}")
        print(f"✓ Primary results: {len(response.primary_source_results)}")
        print(f"✓ Answer length: {len(response.answer)} chars")
        print(f"\nFirst 200 chars of answer:\n{response.answer[:200]}...")

    @pytest.mark.asyncio
    async def test_shelfmark_query(self):
        """Test querying a specific shelf mark"""
        print("\n" + "=" * 80)
        print("TEST: Shelf Mark Query")
        print("=" * 80)

        response = await agentic_rag_service.chat(
            user_query="Tell me about T-S 10J12.9",
            conversation_history=None
        )

        assert response.answer is not None
        assert "T-S 10J12.9" in response.answer or "10J12.9" in response.answer

        # Should use shelfmark search
        has_shelfmark = any(
            a.search_type == "primary_shelfmark"
            for a in response.query_plan.actions
        )
        assert has_shelfmark

        print(f"\n✓ Query: Tell me about T-S 10J12.9")
        print(f"✓ Plan: {response.query_plan.reasoning}")
        print(f"✓ Used shelfmark search: {has_shelfmark}")
        print(f"\nAnswer:\n{response.answer[:300]}...")

    @pytest.mark.asyncio
    async def test_conversation_with_followup(self, sample_conversation_history):
        """Test a follow-up question in conversation"""
        print("\n" + "=" * 80)
        print("TEST: Conversation with Follow-up")
        print("=" * 80)

        print("\nPrevious conversation:")
        print(f"User: {sample_conversation_history[0]['user_query']}")
        print(f"Assistant: {sample_conversation_history[0]['answer'][:150]}...")

        print("\nFollow-up query:")
        response = await agentic_rag_service.chat(
            user_query="What about the other documents mentioned?",
            conversation_history=sample_conversation_history
        )

        assert response.answer is not None
        assert response.resolved_query is not None
        assert response.resolved_query != "What about the other documents mentioned?"

        print(f"\nOriginal: What about the other documents mentioned?")
        print(f"Resolved: {response.resolved_query}")
        print(f"Is follow-up: {response.query_plan.is_followup}")
        print(f"Actions: {[a.search_type for a in response.query_plan.actions]}")
        print(f"\nAnswer:\n{response.answer[:300]}...")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, complex_conversation_history):
        """Test multi-turn conversation building context"""
        print("\n" + "=" * 80)
        print("TEST: Multi-turn Conversation")
        print("=" * 80)

        print("\nConversation history:")
        for i, turn in enumerate(complex_conversation_history, 1):
            print(f"\nTurn {i}:")
            print(f"  User: {turn['user_query']}")
            print(f"  Assistant: {turn['answer'][:100]}...")

        print("\nNew follow-up query:")
        response = await agentic_rag_service.chat(
            user_query="And what about trade routes?",
            conversation_history=complex_conversation_history
        )

        assert response.answer is not None
        assert response.query_plan.is_followup

        print(f"\nOriginal: And what about trade routes?")
        print(f"Resolved: {response.resolved_query}")
        print(f"Is follow-up: {response.query_plan.is_followup}")
        print(f"Processing steps:")
        for step in response.processing_steps:
            print(f"  - {step}")


# ============================================================================
# Test Scenarios - Real-world Use Cases
# ============================================================================

class TestRealWorldScenarios:
    """Test realistic conversation scenarios"""

    @pytest.mark.asyncio
    async def test_researcher_workflow(self):
        """Simulate a researcher exploring documents"""
        print("\n" + "=" * 80)
        print("SCENARIO: Researcher Workflow")
        print("=" * 80)

        conversation = []

        # Turn 1: Initial broad query
        print("\n--- Turn 1: Initial Query ---")
        response1 = await agentic_rag_service.chat(
            user_query="What documents discuss trade between Egypt and India in the medieval period?",
            conversation_history=conversation
        )

        print(f"Query: What documents discuss trade between Egypt and India?")
        print(f"Answer: {response1.answer[:200]}...")

        # Add to history
        conversation.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": "What documents discuss trade between Egypt and India in the medieval period?",
            "answer": response1.answer,
            "query_plan": response1.query_plan.dict(),
            "bibliography_results": response1.bibliography_results,
            "primary_source_results": response1.primary_source_results
        })

        # Turn 2: Ask about specific document
        print("\n--- Turn 2: Follow-up on Specific Document ---")
        response2 = await agentic_rag_service.chat(
            user_query="Tell me more about the first document you mentioned",
            conversation_history=conversation
        )

        print(f"Query: Tell me more about the first document")
        print(f"Resolved: {response2.resolved_query}")
        print(f"Answer: {response2.answer[:200]}...")

        conversation.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": "Tell me more about the first document you mentioned",
            "answer": response2.answer,
            "query_plan": response2.query_plan.dict(),
            "bibliography_results": response2.bibliography_results,
            "primary_source_results": response2.primary_source_results
        })

        # Turn 3: Ask about scholarship
        print("\n--- Turn 3: Ask About Scholarship ---")
        response3 = await agentic_rag_service.chat(
            user_query="What have scholars said about it?",
            conversation_history=conversation
        )

        print(f"Query: What have scholars said about it?")
        print(f"Resolved: {response3.resolved_query}")
        print(f"Answer: {response3.answer[:200]}...")

        # Verify conversation continuity
        assert response2.query_plan.is_followup
        assert response3.query_plan.is_followup
        assert response3.resolved_query != "What have scholars said about it?"

        print("\n✓ Researcher workflow completed successfully")

    @pytest.mark.asyncio
    async def test_clarification_question(self):
        """Test handling clarification questions"""
        print("\n" + "=" * 80)
        print("SCENARIO: Clarification Questions")
        print("=" * 80)

        conversation = []

        # Turn 1
        response1 = await agentic_rag_service.chat(
            user_query="Tell me about business letters in the Genizah",
            conversation_history=None
        )

        conversation.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": "Tell me about business letters in the Genizah",
            "answer": response1.answer,
            "query_plan": response1.query_plan.dict(),
            "bibliography_results": response1.bibliography_results,
            "primary_source_results": response1.primary_source_results
        })

        # Turn 2: Clarification
        response2 = await agentic_rag_service.chat(
            user_query="Who wrote those?",
            conversation_history=conversation
        )

        print(f"\nOriginal: Who wrote those?")
        print(f"Resolved: {response2.resolved_query}")
        print(f"May skip searches: {response2.query_plan.references_previous_results}")

        assert response2.query_plan.is_followup
        assert response2.query_plan.references_previous_results


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_conversation_history(self):
        """Test with empty conversation history"""
        response = await agentic_rag_service.chat(
            user_query="What is the Cairo Genizah?",
            conversation_history=[]
        )

        assert response.answer is not None
        assert not response.query_plan.is_followup
        print("✓ Handled empty conversation history")

    @pytest.mark.asyncio
    async def test_very_long_conversation(self):
        """Test with very long conversation history (should handle gracefully)"""
        long_history = [
            {
                "timestamp": f"2024-01-15T10:{i:02d}:00",
                "user_query": f"Question {i}",
                "answer": f"Answer {i}",
                "bibliography_results": [],
                "primary_source_results": []
            }
            for i in range(20)
        ]

        response = await agentic_rag_service.chat(
            user_query="What about that?",
            conversation_history=long_history
        )

        assert response.answer is not None
        # Should only use recent context, not all 20 turns
        print(f"✓ Handled long conversation ({len(long_history)} turns)")

    @pytest.mark.asyncio
    async def test_ambiguous_reference(self):
        """Test handling ambiguous references"""
        history = [
            {
                "timestamp": "2024-01-15T10:00:00",
                "user_query": "Tell me about T-S 8J22.22 and T-S 10J12.9",
                "answer": "Both are business letters...",
                "bibliography_results": [],
                "primary_source_results": [
                    {"doc_id": "TS-8J22-22", "shelf_mark": "T-S 8J22.22"},
                    {"doc_id": "TS-10J12-9", "shelf_mark": "T-S 10J12.9"}
                ]
            }
        ]

        response = await agentic_rag_service.chat(
            user_query="Tell me more about it",  # Ambiguous!
            conversation_history=history
        )

        # Should handle gracefully, possibly asking for clarification or picking one
        assert response.answer is not None
        print(f"✓ Handled ambiguous reference")
        print(f"  Resolved to: {response.resolved_query}")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_interactive_tests():
    """Run tests interactively with user prompts"""
    print("\n" + "=" * 80)
    print("INTERACTIVE TEST MODE")
    print("=" * 80)
    print("\nThis will run several test conversations.")
    print("Press Enter to continue through each test...\n")

    async def run():
        conversation = []

        queries = [
            "What documents in the Genizah discuss the India trade?",
            "Tell me more about the first one",
            "Who is the author mentioned?",
            "What other documents did they write about?"
        ]

        for i, query in enumerate(queries, 1):
            input(f"\nPress Enter to run query {i}/{len(queries)}...")

            print(f"\n{'=' * 80}")
            print(f"QUERY {i}: {query}")
            print('=' * 80)

            if conversation:
                print("\nConversation so far:")
                for j, turn in enumerate(conversation, 1):
                    print(f"  Turn {j}: {turn['user_query']}")

            response = await agentic_rag_service.chat(
                user_query=query,
                conversation_history=conversation
            )

            print(f"\n{'─' * 80}")
            print("RESPONSE:")
            print('─' * 80)
            print(f"\nResolved Query: {response.resolved_query}")
            print(f"Is Follow-up: {response.query_plan.is_followup}")
            print(f"Search Actions: {[a.search_type for a in response.query_plan.actions]}")
            print(f"\nPlan Reasoning:\n{response.query_plan.reasoning}")
            print(f"\nProcessing Steps:")
            for step in response.processing_steps:
                print(f"  • {step}")
            print(f"\nAnswer:\n{response.answer}")
            print(f"\nVerification Summary: {response.verification_summary}")

            # Add to history
            conversation.append({
                "timestamp": datetime.now().isoformat(),
                "user_query": query,
                "answer": response.answer,
                "query_plan": response.query_plan.dict(),
                "bibliography_results": response.bibliography_results,
                "primary_source_results": response.primary_source_results
            })

        print("\n" + "=" * 80)
        print("INTERACTIVE TEST COMPLETED")
        print("=" * 80)

    asyncio.run(run())


if __name__ == "__main__":
    # Run with pytest for automated tests:
    # pytest test_agentic_rag.py -v -s

    # Or run interactively:
    import sys

    if "--interactive" in sys.argv:
        run_interactive_tests()
    else:
        print("Run with: pytest test_agentic_rag.py -v -s")
        print("Or run with: python test_agentic_rag.py --interactive")