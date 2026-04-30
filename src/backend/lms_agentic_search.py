# agentic_rag_service.py
"""
LangGraph-based agentic RAG service for Cairo Genizah collection.

CORE PRINCIPLE: Scholarly synthesis first, with embedded manuscript references.
Primary sources are supplementary evidence, not the main content.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Literal, TypedDict, Set
from pydantic import BaseModel, Field
import json

from langgraph.graph import StateGraph, END
import weave

from src.backend.search_service import search_service, SearchRequest, DocumentMetadata
from src.backend.search_bibliography import bibliography_search_service, BibliographyHybridSearchRequest
from src.backend.ollama_rag_service import llm_studio_rag_service, ShelfMarkSearchRequest
import dotenv

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

weave.init(os.getenv("WANDB_PROJECT", "cairo-genizah-agentic-rag"))

# ============================================================================
# Constants
# ============================================================================

# If ALL bibliography results score below this threshold, retrieval has likely
# failed (nearest-neighbor garbage rather than genuine matches). Trigger retry.
SIMILARITY_THRESHOLD = 0.4


# ============================================================================
# Pydantic Models
# ============================================================================

class SearchAction(BaseModel):
    """Action to perform a search"""
    search_type: Literal[
        "bibliography_semantic",
        "bibliography_hybrid",
        "primary_semantic",
        "primary_keyword",
        "primary_hybrid",
        "primary_shelfmark"
    ] = Field(..., description="Type of search to perform")
    query: str = Field(..., description="Search query or shelf mark")
    num_results: int = Field(default=5, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    semantic_weight: Optional[int] = Field(default=50, description="Semantic weight (0-100)")
    keyword_weight: Optional[int] = Field(default=50, description="Keyword weight (0-100)")
    exact_match: Optional[bool] = Field(default=False, description="Exact shelf mark match")


class QueryPlan(BaseModel):
    """Plan for answering the query"""
    actions: List[SearchAction] = Field(..., description="Searches to perform in order")
    needs_primary_secondary_linking: bool = Field(
        default=True,
        description="Whether to extract shelf marks from bibliography and link to primary sources"
    )
    is_followup: bool = Field(default=False)
    reasoning: str = Field(..., description="Explanation of the search strategy")


class ScholarlyReference(BaseModel):
    """A reference to a scholarly work with embedded shelf marks"""
    citation: str = Field(..., description="Full citation (Author, Title, Page)")
    quoted_text: Optional[str] = Field(None, description="Direct quote from the scholar")
    shelf_marks_mentioned: List[str] = Field(default_factory=list,
                                             description="Shelf marks mentioned in this reference")


class StructuredAnswer(BaseModel):
    """Structured answer with scholarly synthesis"""
    scholarly_synthesis: str = Field(...,
                                     description="Answer based on scholarly sources with embedded shelf mark references")
    scholarly_references: List[ScholarlyReference] = Field(default_factory=list)
    supplementary_manuscripts: List[str] = Field(
        default_factory=list,
        description="Additional manuscripts not mentioned in scholarly sources"
    )


class ConversationTurn(BaseModel):
    """A single turn in the conversation"""
    timestamp: str
    user_query: str
    query_plan: Optional[Dict[str, Any]] = None
    answer: str
    bibliography_results: List[Dict[str, Any]] = Field(default_factory=list)
    primary_source_results: List[Dict[str, Any]] = Field(default_factory=list)


class VerifiedClaim(BaseModel):
    """A claim with verification status"""
    claim: str = Field(..., description="The factual claim made")
    source_citation: str = Field(..., description="Full citation")
    verification_status: Literal["SUPPORTED", "NOT_SUPPORTED"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Verification reasoning")


class AgenticRAGResponse(BaseModel):
    """Final response from the agentic RAG system"""
    answer: str = Field(..., description="The synthesized answer or error message")
    success: bool = Field(..., description="Whether answer generation succeeded")
    error_type: Optional[str] = Field(None, description="Type of error if failed")
    query_plan: Optional[QueryPlan] = Field(None)
    bibliography_results: List[Dict[str, Any]] = Field(default_factory=list)
    primary_source_results: List[Dict[str, Any]] = Field(default_factory=list)
    verified_claims: List[VerifiedClaim] = Field(default_factory=list)
    verification_summary: Dict[str, int] = Field(default_factory=dict)
    processing_steps: List[str] = Field(default_factory=list)


# ============================================================================
# LangGraph State
# ============================================================================

class AgenticRAGState(TypedDict):
    """State for the agentic RAG graph"""
    user_query: str
    conversation_history: Optional[List[Dict[str, str]]]
    query_plan: Optional[QueryPlan]
    bibliography_results: List[Dict[str, Any]]
    primary_source_results: List[Dict[str, Any]]
    shelf_marks_to_fetch: List[str]

    # Shelf mark tracking
    shelf_marks_in_bibliography: Set[str]  # Mentioned by scholars
    shelf_marks_from_search: Set[str]  # From primary source searches
    shelf_mark_lookup: Dict[str, str]  # shelf_mark -> doc_id mapping

    draft_answer: Optional[str]
    verified_claims: List[VerifiedClaim]
    verification_summary: Dict[str, int]
    final_answer: Optional[str]
    processing_steps: List[str]
    error: Optional[str]
    error_type: Optional[str]

    # Retry tracking
    retry_count: int
    excluded_claims: List[Dict[str, str]]  # Claims flagged as unverifiable by verification agent
                                           # Each entry: {"type": "shelf_mark"|"quote", "text": "...", "reason": "..."}


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_weights(semantic_weight: Optional[int], keyword_weight: Optional[int]) -> tuple[int, int]:
    """Ensure semantic and keyword weights sum to 100"""
    if semantic_weight is None and keyword_weight is None:
        return (50, 50)
    if semantic_weight is None:
        return (100 - keyword_weight, keyword_weight)
    if keyword_weight is None:
        return (semantic_weight, 100 - semantic_weight)

    total = semantic_weight + keyword_weight
    if total == 100:
        return (semantic_weight, keyword_weight)
    if total == 0:
        return (50, 50)

    normalized_semantic = int((semantic_weight / total) * 100)
    normalized_keyword = 100 - normalized_semantic
    return (normalized_semantic, normalized_keyword)


def all_results_below_threshold(results: List[Dict[str, Any]], threshold: float) -> bool:
    """Return True if every result's similarity_score is below the threshold."""
    if not results:
        return True
    return all((r.get("similarity_score") or 0.0) < threshold for r in results)


# ============================================================================
# Agentic RAG Service
# ============================================================================

class AgenticRAGService:
    """LangGraph-based agentic RAG with scholarly synthesis"""

    def __init__(self):
        self.llm_studio_base_url = os.getenv("LLM_STUDIO_URL", "http://127.0.0.1:1234")
        self.router_model = os.getenv("ROUTER_MODEL", "qwen/qwen3-4b-2507")
        self.synthesis_model = os.getenv("SYNTHESIS_MODEL", "c4ai-command-r-v01")
        self.verification_model = os.getenv("VERIFICATION_MODEL", "qwen/qwen3-4b-2507")

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("route_query", self._route_query_node)
        workflow.add_node("execute_searches", self._execute_searches_node)
        workflow.add_node("link_primary_secondary", self._link_primary_secondary_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("verify_claims", self._verify_claims_node)
        workflow.add_node("finalize_response", self._finalize_response_node)

        workflow.set_entry_point("route_query")
        workflow.add_edge("route_query", "execute_searches")
        workflow.add_edge("execute_searches", "link_primary_secondary")
        workflow.add_edge("link_primary_secondary", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "verify_claims")

        def _route_after_verify(state: AgenticRAGState) -> str:
            error_type = state.get("error_type")
            if error_type == "FABRICATED_CLAIMS":
                if state.get("retry_count", 0) < 2:
                    return "retry"
                return "abort"
            return "continue"

        workflow.add_conditional_edges(
            "verify_claims",
            _route_after_verify,
            {
                "retry": "synthesize_answer",
                "continue": "finalize_response",
                "abort": "finalize_response"
            }
        )

        workflow.add_edge("finalize_response", END)

        return workflow.compile()

    @weave.op()
    async def _call_llm_with_tools(
            self,
            messages: List[Dict[str, str]],
            tools: List[Dict[str, Any]],
            model: str,
            tool_choice: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call LLM Studio with function calling"""
        url = f"{self.llm_studio_base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        if tool_choice:
            payload["tool_choice"] = tool_choice

        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @weave.op()
    async def _call_llm(
            self,
            messages: List[Dict[str, str]],
            model: str,
            temperature: float = 0.7
    ) -> str:
        """Call LLM Studio without tools"""
        url = f"{self.llm_studio_base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096
        }

        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

        return result["choices"][0]["message"]["content"]

    @weave.op()
    async def _route_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Route query to appropriate search strategy."""
        logger.info(f"Routing query: {state['user_query']}")

        system_prompt = """You are a query router for the Cairo Genizah SCHOLARLY chat assistant.

**CORE PHILOSOPHY: SCHOLARSHIP OVER MANUSCRIPTS**

This system's value is connecting users to ACADEMIC RESEARCH and SCHOLARLY INTERPRETATION, not just raw manuscript catalogs.

**WHY BIBLIOGRAPHY-FIRST:**
1. Scholars provide interpretation and context that manuscripts lack
2. Most manuscripts lack transcriptions, making direct analysis impossible
3. Users can search manuscripts directly in the main search interface
4. Scholarly sources cite manuscripts WITH interpretation already attached

**DEFAULT STRATEGY: BIBLIOGRAPHY ONLY**

For 90% of queries, you should ONLY search bibliography. Let the linking system automatically fetch manuscripts mentioned by scholars.

**NAMED SCHOLAR QUERIES — CRITICAL RULE:**

When the query is about a specific named person (e.g. "Tell me about Estara Arrant",
"What has Goitein written", "Friedman's work on marriage"), you MUST use TWO searches:
1. `bibliography_hybrid` keyword_weight=90, query=the person's name alone
   — finds documents they authored (author field match)
2. `bibliography_hybrid` keyword_weight=85, query=the full original query
   — finds documents that discuss or cite them

Reason: proper names have no semantic neighbourhood. A semantic search for "Estara Arrant"
will return whatever is nearest in embedding space — completely unrelated material —
and the synthesizer will confabulate. Keyword search is mandatory for names.

If both searches return low similarity scores, the scholar is likely absent from the corpus.
In that case the answer must say so explicitly rather than synthesising from unrelated sources.

**Query Type → Strategy:**

"Tell me about ketubbot in the Genizah"
→ `bibliography_hybrid` (keyword_weight: 70) ONLY
→ Reasoning: topical query, semantic+keyword blend appropriate

"Tell me about Estara Arrant" / "What has Goitein written" / "Friedman's work"
→ `bibliography_hybrid` (keyword_weight: 90) query="[name only]"
  + `bibliography_hybrid` (keyword_weight: 85) query="[full original query]"
→ Reasoning: named scholar, keyword search mandatory

"Show me Purim fragments"
→ `primary_keyword: "Purim"` + `bibliography_hybrid: "Purim Genizah"`
→ Reasoning: user explicitly wants to see manuscripts

"T-S 8.133"
→ `primary_shelfmark: "T-S 8.133"` + `bibliography_hybrid: "T-S 8.133"`
→ Reasoning: specific shelf mark lookup

"What do we know about Yom Kippur liturgy"
→ `bibliography_hybrid` (keyword_weight: 70) ONLY
→ Reasoning: broad topical query

**Critical Rules:**
- Default to 1-2 bibliography searches
- Named persons → always keyword-heavy, always two searches
- Only add primary searches when user explicitly asks to see manuscripts
- When in doubt, bibliography only

**Available Search Types:**
- `bibliography_semantic`: Broad conceptual queries only
- `bibliography_hybrid`: Most queries (set keyword/semantic weights appropriately)
- `primary_shelfmark`: Specific shelf mark lookup
- `primary_keyword`: Keyword search in manuscripts (rare)
- `primary_hybrid`: Balanced manuscript search (rare)"""

        history_context = ""
        if state.get("conversation_history"):
            history_context = "\n\n**Conversation History:**\n"
            for turn in state["conversation_history"]:
                if isinstance(turn, dict):
                    content = turn.get("user_query", "")
                    answer = turn.get("answer", "")
                else:
                    content = getattr(turn, "user_query", "")
                    answer = getattr(turn, "answer", "")

                history_context += f"User: {content}\n"
                history_context += f"Assistant: {answer[:200]}...\n"

        messages = [
            {"role": "system", "content": system_prompt + history_context},
            {"role": "user", "content": state["user_query"]}
        ]

        tools = [{
            "type": "function",
            "function": {
                "name": "create_search_plan",
                "description": "Create a search plan for the Cairo Genizah collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "search_type": {
                                        "type": "string",
                                        "enum": [
                                            "bibliography_semantic",
                                            "bibliography_hybrid",
                                            "primary_semantic",
                                            "primary_keyword",
                                            "primary_hybrid",
                                            "primary_shelfmark"
                                        ]
                                    },
                                    "query": {"type": "string"},
                                    "num_results": {"type": "integer", "default": 5},
                                    "semantic_weight": {"type": "integer", "default": 50},
                                    "keyword_weight": {"type": "integer", "default": 50},
                                    "exact_match": {"type": "boolean", "default": False}
                                },
                                "required": ["search_type", "query"]
                            }
                        },
                        "needs_primary_secondary_linking": {"type": "boolean"},
                        "is_followup": {"type": "boolean"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["actions", "needs_primary_secondary_linking", "reasoning"]
                }
            }
        }]

        response = await self._call_llm_with_tools(
            messages=messages,
            tools=tools,
            model=self.router_model,
            tool_choice=None
        )

        if response["choices"][0]["message"].get("tool_calls"):
            tool_call = response["choices"][0]["message"]["tool_calls"][0]
            arguments = json.loads(tool_call["function"]["arguments"])
            query_plan = QueryPlan(**arguments)
        else:
            logger.warning("No tool call, using bibliography-only fallback")
            query_plan = QueryPlan(
                actions=[
                    SearchAction(
                        search_type="bibliography_hybrid",
                        query=state["user_query"],
                        keyword_weight=70,
                        semantic_weight=30,
                        num_results=5
                    )
                ],
                needs_primary_secondary_linking=True,
                is_followup=False,
                reasoning="Fallback: bibliography-only search (router produced no tool call)"
            )

        state["query_plan"] = query_plan
        state["processing_steps"].append(f"Search plan: {query_plan.reasoning}")

        return state

    @weave.op()
    async def _execute_searches_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Execute searches, track shelf mark sources, and retry on low similarity.

        Retry logic:
        - After executing planned searches, check whether ALL bibliography results
          fall below SIMILARITY_THRESHOLD (indicating the retriever found nothing
          genuinely relevant — nearest-neighbour garbage).
        - If so, attempt one complementary fallback search:
            * If original searches were semantic-heavy → retry keyword-heavy
            * If original searches were keyword-heavy → retry semantic
          This handles the case where a name search found nothing and we want to
          confirm absence rather than silently accept bad results.
        """
        query_plan = state["query_plan"]
        bibliography_results = []
        primary_source_results = []
        shelf_marks_in_bibliography = set()
        shelf_marks_from_search = set()
        shelf_mark_lookup = {}

        async def _run_bib_search(action: SearchAction) -> List[Dict[str, Any]]:
            """Execute a single bibliography search action, return result dicts."""
            results = []
            try:
                sem_weight, kw_weight = normalize_weights(
                    action.semantic_weight, action.keyword_weight
                )
                # Force pure semantic if action type demands it
                if action.search_type == "bibliography_semantic":
                    sem_weight, kw_weight = 100, 0

                search_request = BibliographyHybridSearchRequest(
                    query=action.query,
                    semanticWeight=sem_weight,
                    keywordWeight=kw_weight,
                    num_results=action.num_results,
                    page=1
                )
                response = await bibliography_search_service.search_hybrid(search_request)

                for r in response.results:
                    results.append({
                        "doc_id": r.doc_id,
                        "title": r.title,
                        "authors": r.authors,
                        "author": r.author,
                        "description": r.description,
                        "full_text": r.full_text,
                        "extracted_page_number": r.extracted_page_number,
                        "shelf_marks_mentioned": r.shelf_marks_mentioned,
                        "subject_keywords": r.subject_keywords,
                        "similarity_score": r.similarity_score
                    })
            except Exception as e:
                logger.error(f"Bib search failed ({action.search_type} / '{action.query}'): {e}")
            return results

        for action in query_plan.actions:
            logger.info(f"Executing {action.search_type}: {action.query}")

            try:
                if action.search_type in ("bibliography_semantic", "bibliography_hybrid"):
                    new_results = await _run_bib_search(action)
                    bibliography_results.extend(new_results)

                    for r_dict in new_results:
                        if r_dict.get("shelf_marks_mentioned"):
                            shelf_marks_in_bibliography.update(r_dict["shelf_marks_mentioned"])

                elif action.search_type in ["primary_shelfmark", "primary_keyword", "primary_hybrid",
                                            "primary_semantic"]:
                    if action.search_type == "primary_shelfmark":
                        search_request = ShelfMarkSearchRequest(
                            shelf_mark=action.query,
                            exact_match=action.exact_match or False,
                            num_results=action.num_results
                        )
                        response = await search_service.search_by_shelfmark(search_request)
                    elif action.search_type == "primary_keyword":
                        search_request = SearchRequest(
                            query=action.query,
                            filters=action.filters,
                            num_results=action.num_results
                        )
                        response = await search_service.search_by_keyword(search_request)
                    elif action.search_type == "primary_hybrid":
                        sem_weight, kw_weight = normalize_weights(
                            action.semantic_weight,
                            action.keyword_weight
                        )
                        search_request = SearchRequest(
                            query=action.query,
                            semantic_weight=sem_weight,
                            keyword_weight=kw_weight,
                            filters=action.filters,
                            num_results=action.num_results,
                            page=1
                        )
                        response = await search_service.search_hybrid(search_request)
                    else:  # primary_semantic
                        search_request = SearchRequest(
                            query=action.query,
                            filters=action.filters,
                            num_results=action.num_results
                        )
                        response = await search_service.search(search_request)

                    for r in response.results:
                        ps_dict = {
                            "doc_id": r.doc_id,
                            "shelf_mark": r.metadata.shelf_mark if r.metadata else None,
                            "title": r.metadata.title if r.metadata else None,
                            "description": r.metadata.description if r.metadata else None,
                            "transcription": r.metadata.transcription_full_text if r.metadata else None,
                            "translation": r.metadata.translation_full_text if r.metadata else None,
                            "image_urls": r.metadata.image_urls if r.metadata else None,
                            "similarity_score": r.similarity_score
                        }
                        primary_source_results.append(ps_dict)

                        if r.metadata and r.metadata.shelf_mark:
                            sm = r.metadata.shelf_mark
                            shelf_marks_from_search.add(sm)
                            shelf_mark_lookup[sm] = r.doc_id

            except Exception as e:
                logger.error(f"Search failed for {action.search_type}: {e}")
                state["processing_steps"].append(f"Search failed: {action.search_type} - {str(e)}")

        # ------------------------------------------------------------------
        # Low-similarity fallback: if all bibliography results are below
        # threshold, the retriever found nothing relevant. Attempt one
        # complementary search to confirm absence before proceeding.
        # ------------------------------------------------------------------
        if bibliography_results and all_results_below_threshold(bibliography_results, SIMILARITY_THRESHOLD):
            logger.warning(
                f"All {len(bibliography_results)} bib results below similarity threshold "
                f"({SIMILARITY_THRESHOLD}). Attempting fallback search."
            )
            state["processing_steps"].append(
                f"Low-similarity results (all < {SIMILARITY_THRESHOLD}). Attempting fallback search."
            )

            # Determine fallback strategy: flip keyword ↔ semantic bias
            original_actions = query_plan.actions
            original_was_keyword_heavy = any(
                (a.keyword_weight or 50) > 60
                for a in original_actions
                if a.search_type in ("bibliography_hybrid", "bibliography_semantic")
            )

            fallback_query = state["user_query"]
            if original_was_keyword_heavy:
                # Keyword search found nothing → try broader semantic search
                fallback_action = SearchAction(
                    search_type="bibliography_semantic",
                    query=fallback_query,
                    semantic_weight=100,
                    keyword_weight=0,
                    num_results=5
                )
                fallback_label = "semantic fallback"
            else:
                # Semantic search found nothing → try tighter keyword search
                fallback_action = SearchAction(
                    search_type="bibliography_hybrid",
                    query=fallback_query,
                    semantic_weight=10,
                    keyword_weight=90,
                    num_results=5
                )
                fallback_label = "keyword fallback"

            fallback_results = await _run_bib_search(fallback_action)

            if fallback_results and not all_results_below_threshold(fallback_results, SIMILARITY_THRESHOLD):
                logger.info(f"Fallback search ({fallback_label}) returned relevant results.")
                bibliography_results.extend(fallback_results)
                for r_dict in fallback_results:
                    if r_dict.get("shelf_marks_mentioned"):
                        shelf_marks_in_bibliography.update(r_dict["shelf_marks_mentioned"])
                state["processing_steps"].append(
                    f"Fallback ({fallback_label}) returned {len(fallback_results)} results above threshold."
                )
            else:
                # Both searches failed — mark state so synthesis returns IDK
                logger.warning("Fallback search also returned only low-similarity results.")
                state["processing_steps"].append(
                    "Fallback search also below threshold. Corpus likely has no relevant material."
                )
                # Set a soft flag that synthesis will read; not a hard error yet —
                # synthesis should produce an explicit "not in corpus" response.
                state["error_type"] = "NO_RELEVANT_SOURCES"
                state["error"] = (
                    f"All searches returned similarity scores below {SIMILARITY_THRESHOLD}. "
                    "The corpus likely does not contain relevant information for this query."
                )

        state["bibliography_results"] = bibliography_results
        state["primary_source_results"] = primary_source_results
        state["shelf_marks_in_bibliography"] = shelf_marks_in_bibliography
        state["shelf_marks_from_search"] = shelf_marks_from_search
        state["shelf_mark_lookup"] = shelf_mark_lookup
        state["processing_steps"].append(
            f"Executed searches: {len(bibliography_results)} bib, {len(primary_source_results)} primary. "
            f"Scholars mentioned {len(shelf_marks_in_bibliography)} shelf marks."
        )

        return state

    @weave.op()
    async def _link_primary_secondary_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Fetch manuscripts mentioned by scholars"""
        # Skip if we already know there are no relevant sources
        if state.get("error_type") == "NO_RELEVANT_SOURCES":
            state["processing_steps"].append("Skipping linking — no relevant sources found")
            return state

        if not state["query_plan"].needs_primary_secondary_linking:
            state["processing_steps"].append("Skipping linking")
            return state

        shelf_marks_to_fetch = state["shelf_marks_in_bibliography"]

        if not shelf_marks_to_fetch:
            state["processing_steps"].append("No shelf marks in bibliography to fetch")
            return state

        logger.info(f"Fetching {len(shelf_marks_to_fetch)} manuscripts mentioned by scholars")

        fetched_count = 0
        for shelf_mark in list(shelf_marks_to_fetch)[:30]:
            if shelf_mark in state["shelf_mark_lookup"]:
                continue

            try:
                search_request = ShelfMarkSearchRequest(
                    shelf_mark=shelf_mark,
                    exact_match=False,
                    num_results=1
                )
                response = await search_service.search_by_shelfmark(search_request)

                if response.results:
                    r = response.results[0]
                    doc_id = r.doc_id

                    if not any(ps["doc_id"] == doc_id for ps in state["primary_source_results"]):
                        ps_dict = {
                            "doc_id": doc_id,
                            "shelf_mark": r.metadata.shelf_mark if r.metadata else None,
                            "title": r.metadata.title if r.metadata else None,
                            "description": r.metadata.description if r.metadata else None,
                            "transcription": r.metadata.transcription_full_text if r.metadata else None,
                            "translation": r.metadata.translation_full_text if r.metadata else None,
                            "image_urls": r.metadata.image_urls if r.metadata else None,
                            "similarity_score": r.similarity_score,
                            "linked_from_bibliography": True
                        }
                        state["primary_source_results"].append(ps_dict)
                        fetched_count += 1

                    actual_sm = r.metadata.shelf_mark if r.metadata else None
                    if actual_sm:
                        state["shelf_mark_lookup"][actual_sm] = doc_id
                        if shelf_mark != actual_sm:
                            state["shelf_mark_lookup"][shelf_mark] = doc_id

            except Exception as e:
                logger.error(f"Failed to fetch {shelf_mark}: {e}")

        state["processing_steps"].append(
            f"Fetched {fetched_count} manuscripts mentioned in scholarly sources"
        )

        return state

    @weave.op()
    async def _synthesize_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Synthesize scholarly answer from secondary sources only.

        If error_type is NO_RELEVANT_SOURCES, returns an explicit IDK response
        without invoking the LLM (prevents confabulation from empty context).
        """
        logger.info("Synthesizing scholarly answer")

        # Clear synthesis-level error state (handles retry) but preserve
        # NO_RELEVANT_SOURCES — that must short-circuit synthesis entirely.
        if state.get("error_type") == "NO_RELEVANT_SOURCES":
            state["draft_answer"] = (
                "I wasn't able to find relevant information about this topic in the "
                "scholarly bibliography. The corpus may not yet include work on this "
                "specific subject or scholar. You may want to search external databases "
                "such as the Princeton Geniza Project catalog or JSTOR directly."
            )
            state["processing_steps"].append(
                "Short-circuited synthesis: no relevant sources above similarity threshold."
            )
            return state

        state["error"] = None
        state["error_type"] = None

        bib_context = []
        for bib in state["bibliography_results"][:8]:
            authors = bib.get("authors") or ([bib.get("author")] if bib.get("author") else ["Unknown"])
            author_str = ", ".join(authors)
            title = bib.get("title") or "Untitled"
            page = bib.get("extracted_page_number")

            parts = [f"Source: {title} by {author_str}" + (f", p. {page}" if page else "")]

            if bib.get("full_text"):
                parts.append(f"Text: {bib['full_text'][:600]}")
            if bib.get("description"):
                parts.append(f"Summary: {bib['description']}")
            if bib.get("shelf_marks_mentioned"):
                parts.append(f"Shelf marks cited in this source: {', '.join(bib['shelf_marks_mentioned'][:10])}")

            bib_context.append("\n".join(parts))

        system_prompt = """You are a scholarly research assistant specializing in Cairo Genizah studies.

Your inputs are chunks retrieved from academic secondary sources (books and articles about the Genizah). Your job is to synthesize these sources into a coherent scholarly response with precise citations.

Rules:
1. Lead with what scholars have written. Quote directly where it strengthens the response.
   Format quotes as: Author (Year), p. X: "quote text"
2. Every factual claim must cite a specific retrieved source with page number.
   Do not draw on background knowledge — only the retrieved chunks.
3. When a shelf-mark appears in a retrieved source, include it exactly as written.
   Treat it as a reference to be cited, not a document to analyze or describe.
   Do not add any information about what the fragment contains beyond what the source text says.
4. Do not invent shelf-marks, page numbers, or citations. If the retrieved sources
   don't cover an aspect of the query, say so explicitly rather than filling the gap.
5. If retrieved sources are sparse, return what you have with honest attribution
   rather than padding with general knowledge.
6. CRITICAL — QUOTES: Only use text in quotation marks if it appears verbatim (or near-verbatim)
   in the retrieved source chunks above. Do not construct plausible-sounding quotes.
   If you want to represent what a scholar argued, paraphrase with attribution instead."""

        exclusion_note = ""
        excluded_claims = state.get("excluded_claims", [])

        if excluded_claims:
            excluded_sms = [c["text"] for c in excluded_claims if c["type"] == "shelf_mark"]
            excluded_quotes = [c["text"] for c in excluded_claims if c["type"] == "quote"]

            if excluded_sms:
                exclusion_note += (
                    f"\n\nThe following shelf marks were flagged as unverifiable by the "
                    f"verification agent and must be excluded from your response:\n"
                    + "\n".join(f"  - {sm}" for sm in excluded_sms)
                )
            if excluded_quotes:
                exclusion_note += (
                    f"\n\nThe following direct quotes could not be verified in the source text. "
                    f"Remove them and use paraphrased attribution instead:\n"
                    + "\n".join(f'  - "{q[:80]}{"…" if len(q) > 80 else ""}"' for q in excluded_quotes)
                )

        user_message = f"""{system_prompt}{exclusion_note}

RETRIEVED SCHOLARLY SOURCES:

{chr(10).join(bib_context) if bib_context else "No scholarly sources retrieved."}

USER QUERY:
{state['user_query']}

Provide your scholarly synthesis. Cite only what appears in the retrieved sources above."""

        messages = [{"role": "user", "content": user_message}]

        draft_answer = await self._call_llm(
            messages=messages,
            model=self.synthesis_model,
            temperature=0.2
        )

        state["draft_answer"] = draft_answer
        retry = state.get("retry_count", 0)
        state["processing_steps"].append(
            "Synthesized scholarly answer" + (f" (retry {retry})" if retry else "")
        )

        return state

    @weave.op()
    async def _verify_claims_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: LLM-based verification agent.

        Uses the verification_model (small, fast) to check whether shelf marks
        and direct quotes in the draft answer are genuinely present in the
        retrieved source chunks. No regex — the LLM handles format variations
        in shelf marks and near-verbatim matching for quotes.

        On failure: populates excluded_claims and sets error_type=FABRICATED_CLAIMS,
        triggering a retry in synthesize_answer with exclusion instructions.
        """
        logger.info("Verifying claims")

        # If synthesis was short-circuited (no relevant sources), nothing to verify.
        if state.get("error_type") == "NO_RELEVANT_SOURCES":
            return state
        draft = state["draft_answer"]
        bib_results = state["bibliography_results"]

        # Build source context for the verifier — full text of each retrieved chunk
        source_chunks = []
        for i, bib in enumerate(bib_results):
            authors = bib.get("authors") or ([bib.get("author")] if bib.get("author") else ["Unknown"])
            chunk_text = " ".join(filter(None, [
                bib.get("full_text", ""),
                bib.get("description", ""),
            ]))
            source_chunks.append(f"[SOURCE {i+1}] {', '.join(authors)}: {chunk_text[:800]}")

        sources_text = "\n\n".join(source_chunks) if source_chunks else "No sources retrieved."

        # Build exclusion context so the verifier knows what was already flagged
        prior_excluded = state.get("excluded_claims", [])
        exclusion_note = ""
        if prior_excluded:
            exclusion_note = (
                "\n\nThe following were flagged as unverifiable in a previous attempt "
                "and must still be treated as NOT_SUPPORTED:\n"
                + "\n".join(f'- {c["type"]}: {c["text"][:80]}' for c in prior_excluded)
            )

        verifier_prompt = f"""You are a verification agent for a scholarly Cairo Genizah research assistant.

Your job is narrow and specific: check whether shelf marks and direct quotes in the DRAFT ANSWER
are genuinely present in the SOURCE CHUNKS below. You are NOT judging whether the answer is
broadly correct — only whether these specific textual elements can be traced to the sources.

RULES:
- A shelf mark is verified if it appears in any source chunk in any reasonable form
  (spacing and punctuation may vary, e.g. "T-S 8.133" and "TS 8.133" are the same mark).
- A direct quote is verified if the quoted text appears verbatim or near-verbatim in a source chunk.
  Minor OCR differences are acceptable. Wholly invented text is not.
- If a shelf mark or quote does NOT appear in any source chunk, mark it NOT_SUPPORTED.
- Paraphrased content (not in quotation marks) does NOT need verification — ignore it.
{exclusion_note}

SOURCE CHUNKS:
{sources_text}

DRAFT ANSWER:
{draft}

Respond with ONLY valid JSON in this exact format — no markdown, no explanation:
{{
  "verified_claims": [
    {{"type": "shelf_mark", "text": "T-S 8.133", "supported": true, "reasoning": "Appears in SOURCE 2"}},
    {{"type": "quote", "text": "first ten words of quote...", "supported": false, "reasoning": "Not found in any source chunk"}}
  ],
  "overall": "PASS",
  "summary": "2 shelf marks verified, 1 fabricated quote removed"
}}

Set "overall" to "FAIL" if any claim is supported=false, otherwise "PASS".
If there are no shelf marks or quotes in the draft, return {{"verified_claims": [], "overall": "PASS", "summary": "No shelf marks or quotes to verify"}}.
"""

        raw_response = await self._call_llm(
            messages=[{"role": "user", "content": verifier_prompt}],
            model=self.verification_model,
            temperature=0.0  # Deterministic — this is a factual check
        )

        # Parse verifier response
        try:
            # Strip markdown fences if present
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            verification_result = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Verification agent returned unparseable response: {e}\n{raw_response[:200]}")
            # If we can't parse the verifier output, pass through rather than
            # silently failing — log it and continue. Better to show an
            # unverified answer than to crash.
            state["processing_steps"].append(
                f"Verification agent parse error — passing through unverified. Error: {e}"
            )
            state["verified_claims"] = []
            state["verification_summary"] = {"SUPPORTED": 0, "NOT_SUPPORTED": 0, "parse_error": 1}
            return state

        claims = verification_result.get("verified_claims", [])
        overall = verification_result.get("overall", "PASS")
        summary = verification_result.get("summary", "")

        # Build VerifiedClaim objects from agent output
        verified_claim_objects = []
        new_excluded = list(prior_excluded)

        for claim in claims:
            status = "SUPPORTED" if claim.get("supported") else "NOT_SUPPORTED"
            verified_claim_objects.append(VerifiedClaim(
                claim=f'{claim.get("type", "claim")}: {claim.get("text", "")[:80]}',
                source_citation="Retrieved bibliography chunks",
                verification_status=status,
                confidence=1.0 if claim.get("supported") else 0.0,
                reasoning=claim.get("reasoning", "")
            ))
            if not claim.get("supported"):
                new_excluded.append({
                    "type": claim.get("type", "unknown"),
                    "text": claim.get("text", ""),
                    "reason": claim.get("reasoning", "")
                })

        state["verified_claims"] = verified_claim_objects
        supported_count = sum(1 for c in claims if c.get("supported"))
        unsupported_count = sum(1 for c in claims if not c.get("supported"))
        state["verification_summary"] = {
            "SUPPORTED": supported_count,
            "NOT_SUPPORTED": unsupported_count
        }

        if overall == "FAIL":
            retry_count = state.get("retry_count", 0) + 1
            logger.warning(
                f"Verification failed (attempt {retry_count}): {unsupported_count} unverifiable claims. "
                f"{summary}"
            )
            state["excluded_claims"] = new_excluded
            state["retry_count"] = retry_count
            state["error_type"] = "FABRICATED_CLAIMS"
            state["error"] = summary
            state["processing_steps"].append(
                f"Verification FAILED (attempt {retry_count}/2): {summary}"
            )
        else:
            state["excluded_claims"] = []
            state["processing_steps"].append(f"Verification PASSED: {summary}")

        return state

    _GRACEFUL_FALLBACK = (
        "I wasn't able to construct a fully verified response for this query. "
        "Please try rephrasing or narrowing your question, or use the search "
        "panel directly to explore relevant sources."
    )

    @weave.op()
    async def _finalize_response_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Finalize with shelf mark linking; append primary sources as catalog only"""
        logger.info("Finalizing response")

        error_type = state.get("error_type")

        if error_type == "NO_RELEVANT_SOURCES":
            state["final_answer"] = state["draft_answer"]
            state["processing_steps"].append("Returned IDK response — no relevant sources.")
            return state

        if error_type == "FABRICATED_CLAIMS":
            state["final_answer"] = self._GRACEFUL_FALLBACK
            state["processing_steps"].append(
                f"Returned graceful fallback after retry exhaustion. "
                f"Error: {state.get('error', '')}"
            )
            return state

        final_answer = self._linkify_all_shelfmarks(state["draft_answer"], state)

        catalog_entries = []
        for ps in state["primary_source_results"]:
            sm = ps.get("shelf_mark")
            if not sm:

                continue
            doc_id = ps.get("doc_id") or state["shelf_mark_lookup"].get(sm)
            title = ps.get("title") or ""
            description = ps.get("description") or ""
            entry_parts = [f"- **{sm}**" + (f": {title}" if title else "")]
            if description:
                entry_parts.append(f"  {description[:120]}")
            if doc_id:
                entry_parts[0] = f"- **[{sm}](doc:{doc_id})**" + (f": {title}" if title else "")
            catalog_entries.append("\n".join(entry_parts))

        if catalog_entries:
            final_answer = (
                final_answer.rstrip()
                + "\n\n---\n**Related catalog entries:**\n\n"
                + "\n".join(catalog_entries)
            )

        state["final_answer"] = final_answer
        state["processing_steps"].append("Linked shelf marks and appended catalog entries")

        return state

    def _linkify_all_shelfmarks(self, text: str, state: AgenticRAGState) -> str:
        """Link all shelf marks in text to their doc_ids"""
        if not text:
            return text

        shelf_mark_lookup = state.get("shelf_mark_lookup", {})
        all_shelf_marks = sorted(shelf_mark_lookup.keys(), key=len, reverse=True)

        for sm in all_shelf_marks:
            doc_id = shelf_mark_lookup[sm]
            escaped_sm = re.escape(sm)
            pattern = rf'(?<!\[)(?<!\(doc:){escaped_sm}(?!\]\(doc:)(?!\])'
            replacement = f"[{sm}](doc:{doc_id})"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    @weave.op()
    async def chat(
            self,
            user_query: str,
            conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AgenticRAGResponse:
        """Main entry point for agentic RAG chat"""

        initial_state: AgenticRAGState = {
            "user_query": user_query,
            "conversation_history": conversation_history,
            "query_plan": None,
            "bibliography_results": [],
            "primary_source_results": [],
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
            "excluded_claims": []
        }

        final_state = await self.graph.ainvoke(initial_state)

        return AgenticRAGResponse(
            answer=final_state["final_answer"] or "Unable to generate answer",
            success=final_state["error_type"] is None,
            error_type=final_state.get("error_type"),
            query_plan=final_state.get("query_plan"),
            bibliography_results=final_state["bibliography_results"],
            primary_source_results=final_state["primary_source_results"],
            verified_claims=final_state["verified_claims"],
            verification_summary=final_state["verification_summary"],
            processing_steps=final_state["processing_steps"]
        )

    @weave.op()
    async def chat_stream(
            self,
            user_query: str,
            conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """Streaming entry point"""
        initial_state: AgenticRAGState = {
            "user_query": user_query,
            "conversation_history": conversation_history,
            "query_plan": None,
            "bibliography_results": [],
            "primary_source_results": [],
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
            "excluded_claims": []
        }

        node_status_map = {
            "route_query": "Planning search strategy...",
            "execute_searches": "Searching scholarly sources...",
            "link_primary_secondary": "Fetching manuscripts mentioned by scholars...",
            "synthesize_answer": "Synthesizing scholarly analysis...",
            "verify_claims": "Verifying claims...",
            "finalize_response": "Finalizing response..."
        }

        async for event in self.graph.astream(initial_state, stream_mode="updates"):
            for node_name, updates in event.items():
                status = node_status_map.get(node_name, f"Processing {node_name}...")

                query_plan = updates.get("query_plan")
                yield {
                    "type": "status",
                    "status": status,
                    "node": node_name,
                    "query_plan": query_plan.dict() if query_plan and hasattr(query_plan, 'dict') else query_plan,
                    "bibliography_count": len(updates.get("bibliography_results", [])),
                    "primary_count": len(updates.get("primary_source_results", [])),
                    "verified_claims_count": len(updates.get("verified_claims", []))
                }

        final_result = await self.chat(user_query, conversation_history)
        yield {
            "type": "final",
            "data": final_result.dict()
        }


# Global service instance
agentic_rag_service = AgenticRAGService()