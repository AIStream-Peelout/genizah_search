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


def extract_shelf_marks(text: str) -> Set[str]:
    """Extract all shelf mark patterns from text"""
    patterns = [
        r'T-S\s+[\w.]+',
        r'CUL\s+[\w.]+',
        r'ENA\s+[\w.]+',
        r'Or\.\s*\d+',
        r'MS\s+[\w.]+',
        r'Cambridge\s+(?:CUL|Lewis-Gibson):\s*[\w.-]+',
        r'New York\s+JTS:\s*[\w.\s]+',
        r'Philadelphia\s+Penn\s+CAJS:\s*[\w.\s]+',
        r'Geneva:\s*[\w.\s]+',
        r'Paris\s+AIU:\s*[\w.\s]+',
    ]

    shelf_marks = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        shelf_marks.update(m.strip() for m in matches)

    return shelf_marks


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

        workflow.add_conditional_edges(
            "verify_claims",
            lambda s: "abort" if s.get("error_type") else "continue",
            {
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
        """Node: Route with BIBLIOGRAPHY-FIRST strategy"""
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

For 90% of queries, you should ONLY search bibliography:
- `bibliography_hybrid` (keyword_weight: 60-70) for most queries
- `bibliography_semantic` for very broad conceptual queries

Let the linking system automatically fetch manuscripts mentioned by scholars.

**WHEN TO ADD PRIMARY SOURCE SEARCHES:**

ONLY add primary source searches if:
1. User explicitly asks: "show me manuscripts", "find fragments", "which manuscripts mention X"
2. User provides specific shelf mark: "tell me about T-S 8.133"
3. Query is ONLY about manuscript features with no scholarly angle: "what materials were used"

**Query Type → Strategy:**

"Tell me about ketubbot in the Genizah"
→ `bibliography_hybrid` (keyword_weight: 70) ONLY
→ Reasoning: General knowledge, scholars have written extensively about this

"What has Friedman said about marriage contracts"  
→ `bibliography_semantic` or `bibliography_hybrid` ONLY
→ Reasoning: Asking about specific scholar

"Show me Purim fragments"
→ `primary_keyword: "Purim"` + `bibliography_hybrid: "Purim Genizah"`
→ Reasoning: Explicitly asks to SEE manuscripts

"T-S 8.133"
→ `primary_shelfmark: "T-S 8.133"` + `bibliography_hybrid: "T-S 8.133"`  
→ Reasoning: Specific shelf mark query

"What do we know about Yom Kippur liturgy"
→ `bibliography_hybrid` (keyword_weight: 70) ONLY
→ Reasoning: Scholarly knowledge question

**Critical Rules:**
- Default to 1-2 bibliography searches
- Only add primary searches when explicitly needed
- Trust the linking system to fetch manuscripts
- When in doubt, bibliography only

**Available Search Types:**
- `bibliography_semantic`: Broad conceptual queries
- `bibliography_hybrid`: Most bibliography searches (default)
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
                reasoning="Fallback: bibliography-only search (scholarly focus)"
            )

        state["query_plan"] = query_plan
        state["processing_steps"].append(f"Search plan: {query_plan.reasoning}")

        return state

    @weave.op()
    async def _execute_searches_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Execute searches and track shelf mark sources"""
        query_plan = state["query_plan"]
        bibliography_results = []
        primary_source_results = []
        shelf_marks_in_bibliography = set()
        shelf_marks_from_search = set()
        shelf_mark_lookup = {}

        for action in query_plan.actions:
            logger.info(f"Executing {action.search_type}: {action.query}")

            try:
                if action.search_type == "bibliography_semantic":
                    search_request = BibliographyHybridSearchRequest(
                        query=action.query,
                        semanticWeight=100,
                        keywordWeight=0,
                        num_results=action.num_results,
                        page=1
                    )
                    response = await bibliography_search_service.search_hybrid(search_request)

                    for r in response.results:
                        bib_dict = {
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
                        }
                        bibliography_results.append(bib_dict)

                        # Track shelf marks mentioned by scholars
                        if r.shelf_marks_mentioned:
                            shelf_marks_in_bibliography.update(r.shelf_marks_mentioned)

                elif action.search_type == "bibliography_hybrid":
                    sem_weight, kw_weight = normalize_weights(
                        action.semantic_weight,
                        action.keyword_weight
                    )

                    search_request = BibliographyHybridSearchRequest(
                        query=action.query,
                        semanticWeight=sem_weight,
                        keywordWeight=kw_weight,
                        num_results=action.num_results,
                        page=1
                    )
                    response = await bibliography_search_service.search_hybrid(search_request)

                    for r in response.results:
                        bib_dict = {
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
                        }
                        bibliography_results.append(bib_dict)

                        if r.shelf_marks_mentioned:
                            shelf_marks_in_bibliography.update(r.shelf_marks_mentioned)

                elif action.search_type in ["primary_shelfmark", "primary_keyword", "primary_hybrid",
                                            "primary_semantic"]:
                    # Primary source searches - these are supplementary

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
            # Skip if already fetched
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

                    # CRITICAL: Map ALL variations of shelf mark
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
        """Node: Synthesize scholarly answer with embedded manuscript references"""
        logger.info("Synthesizing scholarly synthesis")

        # Build rich bibliography context
        bib_context = []
        for bib in state["bibliography_results"][:8]:
            authors = bib.get("authors") or [bib.get("author")] if bib.get("author") else ["Unknown"]
            author_str = ", ".join(authors)
            title = bib.get("title") or "Untitled"
            page = bib.get("extracted_page_number")

            parts = [f"**Source: {title} by {author_str}" + (f", p. {page}**" if page else "**")]

            if bib.get("full_text"):
                # Full text is CRITICAL - preserve shelf mark mentions
                parts.append(f"Text: {bib['full_text'][:600]}")
            if bib.get("description"):
                parts.append(f"Summary: {bib['description']}")
            if bib.get("shelf_marks_mentioned"):
                parts.append(f"Manuscripts mentioned: {', '.join(bib['shelf_marks_mentioned'][:5])}")

            bib_context.append("\n".join(parts))

        # List primary sources available for supplementary section
        supplementary_shelf_marks = []
        for ps in state["primary_source_results"]:
            sm = ps.get("shelf_mark")
            if sm and sm not in state["shelf_marks_in_bibliography"]:
                supplementary_shelf_marks.append({
                    "shelf_mark": sm,
                    "description": ps.get("description", "No description")[:100]
                })

        system_prompt = f"""You are synthesizing a SCHOLARLY answer about the Cairo Genizah.

**YOUR ROLE:** Present what SCHOLARS have written, preserving their manuscript references.

**CRITICAL STRUCTURE:**

Your answer should have TWO sections:

**SECTION 1: Scholarly Synthesis (PRIMARY - 80% of answer)**
- Quote or paraphrase what scholars have written
- When scholars mention manuscripts, PRESERVE those references inline
- Format: "Friedman in *Jewish Marriage* (p. 104) notes that T-S 8.133 shows..."
- Keep manuscript references in the flow of scholarly argument

**SECTION 2: Additional Manuscripts (SUPPLEMENTARY - 20% if applicable)**
- ONLY list manuscripts NOT already mentioned in scholarly sources
- Brief format: "- Shelf Mark: Brief description"
- These are bonus materials for exploration

**EXAMPLE (GOOD):**

Scholars including M.A. Friedman, in *Jewish Marriage in Palestine: The Ketubba Texts*, have studied Genizah ketubbot extensively. Friedman explores the historical development of these contracts, noting that manuscripts such as T-S 8.133, a late tenth-century fragment from Tinnis, and T-S 16.198, dating to the tenth-twelfth century from Tyre, reveal diverse traditions challenging previous understanding of a uniform Babylonian model.

Goitein in *A Mediterranean Society* observes that T-S 16.107, from Aleppo (1107/08 CE), provides an unprecedented window into...

**Additional Related Manuscripts:**
- T-S 20.65: Replacement ketubba from Safed, dated 1539 CE
- Paris AIU: IV.B.9: Customs for the month of Av

**RULES:**

1. **Lead with scholarship** - What have scholars discovered?
2. **Preserve inline manuscript references** - Don't separate them from scholarly context
3. **Use exact shelf marks** from the sources (scholar may use abbreviated forms)
4. **Acknowledge transcription limits** - "Many manuscripts lack full transcriptions"
5. **Supplementary section ONLY** for manuscripts not in scholarly sources
6. **If no scholarship** - Honestly say "I found manuscripts but limited scholarly analysis"

**AVAILABLE SCHOLARLY SOURCES:**

{chr(10).join(bib_context)}

**SUPPLEMENTARY MANUSCRIPTS (not mentioned by scholars):**
{json.dumps(supplementary_shelf_marks, indent=2) if supplementary_shelf_marks else "None"}

**USER QUERY:**
{state['user_query']}

Provide your scholarly synthesis focusing on what researchers have written."""

        messages = [{"role": "user", "content": system_prompt}]

        draft_answer = await self._call_llm(
            messages=messages,
            model=self.synthesis_model,
            temperature=0.2
        )

        state["draft_answer"] = draft_answer
        state["processing_steps"].append("Synthesized scholarly answer")

        return state

    @weave.op()
    async def _verify_claims_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Verify shelf marks in answer against available sources"""
        logger.info("Verifying shelf marks")

        draft = state["draft_answer"]

        # Extract all shelf marks mentioned in the answer
        mentioned_shelf_marks = extract_shelf_marks(draft)

        # Check which are available
        available = state["shelf_marks_in_bibliography"] | state["shelf_marks_from_search"]

        verified = []
        hallucinations = []

        for sm in mentioned_shelf_marks:
            # Check if shelf mark is in our available set (exact or fuzzy)
            if sm in available or sm in state["shelf_mark_lookup"]:
                verified.append(sm)
            else:
                # Check fuzzy match (bibliography might abbreviate)
                found = False
                for available_sm in available:
                    if sm in available_sm or available_sm in sm:
                        verified.append(sm)
                        found = True
                        break

                if not found:
                    hallucinations.append(sm)
                    logger.critical(f"🚨 FABRICATED SHELF MARK: {sm}")

        if hallucinations:
            logger.critical(f"BLOCKING: {len(hallucinations)} fabricated shelf marks")
            state["error_type"] = "FABRICATED_SHELFMARKS"
            state["error"] = (
                f"I cannot provide this answer. Referenced shelf marks not in sources:\n"
                f"{', '.join(hallucinations)}\n\n"
                f"Let me search again."
            )
            return state

        state["verification_summary"] = {
            "SUPPORTED": len(verified),
            "NOT_SUPPORTED": 0
        }

        state["verified_claims"] = [
            VerifiedClaim(
                claim=f"References {sm}",
                source_citation=f"Manuscript {sm}",
                verification_status="SUPPORTED",
                confidence=1.0,
                reasoning="Shelf mark from scholarly or search sources"
            )
            for sm in verified
        ]

        state["processing_steps"].append(f"Verified {len(verified)} shelf marks")

        return state

    @weave.op()
    async def _finalize_response_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Finalize with shelf mark linking"""
        logger.info("Finalizing response")

        if state.get("error_type"):
            state["final_answer"] = state["error"]
            return state

        final_answer = self._linkify_all_shelfmarks(state["draft_answer"], state)

        state["final_answer"] = final_answer
        state["processing_steps"].append("Linked all shelf marks")

        return state

    def _linkify_all_shelfmarks(self, text: str, state: AgenticRAGState) -> str:
        """Link all shelf marks in text to their doc_ids"""
        if not text:
            return text

        shelf_mark_lookup = state.get("shelf_mark_lookup", {})

        # Get all shelf marks (sorted by length to avoid partial matches)
        all_shelf_marks = sorted(shelf_mark_lookup.keys(), key=len, reverse=True)

        for sm in all_shelf_marks:
            doc_id = shelf_mark_lookup[sm]

            # Escape for regex
            escaped_sm = re.escape(sm)

            # Pattern: Match shelf mark not already in a link
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
            "error_type": None
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
            "error_type": None
        }

        node_status_map = {
            "route_query": "Planning search strategy...",
            "execute_searches": "Searching scholarly sources...",
            "link_primary_secondary": "Fetching manuscripts mentioned by scholars...",
            "synthesize_answer": "Synthesizing scholarly analysis...",
            "verify_claims": "Verifying shelf marks...",
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
