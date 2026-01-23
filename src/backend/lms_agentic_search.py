# agentic_rag_service.py
"""
LangGraph-based agentic RAG service for Cairo Genizah collection.
Routes queries intelligently, executes searches, verifies claims, and synthesizes answers.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
import json

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# W&B Weave for monitoring
import weave

# Your existing services
from src.backend.search_service import search_service, SearchRequest
from src.backend.search_bibliography import bibliography_search_service, BibliographyHybridSearchRequest
from src.backend.ollama_rag_service import llm_studio_rag_service, ShelfMarkSearchRequest

logger = logging.getLogger(__name__)

# Initialize Weave
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
    # For hybrid searches
    semantic_weight: Optional[int] = Field(default=50, description="Semantic weight (0-100)")
    keyword_weight: Optional[int] = Field(default=50, description="Keyword weight (0-100)")
    # For shelfmark searches
    exact_match: Optional[bool] = Field(default=False, description="Exact shelf mark match")


class QueryPlan(BaseModel):
    """Plan for answering the query"""
    actions: List[SearchAction] = Field(..., description="Searches to perform in order")
    needs_primary_secondary_linking: bool = Field(
        default=True,
        description="Whether to extract shelf marks from bibliography and link to primary sources"
    )
    reasoning: str = Field(..., description="Explanation of the search strategy")


class VerifiedClaim(BaseModel):
    """A claim with verification status"""
    claim: str = Field(..., description="The factual claim made")
    source_citation: str = Field(..., description="Full citation (Author, Title, Page)")
    quote: Optional[str] = Field(None, description="Direct quote supporting the claim")
    verification_status: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "NOT_FOUND"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Why this verification status was assigned")


class AgenticRAGResponse(BaseModel):
    """Final response from the agentic RAG system"""
    answer: str = Field(..., description="The synthesized answer")
    query_plan: QueryPlan = Field(..., description="The search plan that was executed")
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
    # Input
    user_query: str
    conversation_history: Optional[List[Dict[str, str]]]

    # Planning
    query_plan: Optional[QueryPlan]

    # Search results
    bibliography_results: List[Dict[str, Any]]
    primary_source_results: List[Dict[str, Any]]
    shelf_marks_to_fetch: List[str]

    # Generation
    draft_answer: Optional[str]

    # Verification
    verified_claims: List[VerifiedClaim]
    verification_summary: Dict[str, int]

    # Final output
    final_answer: Optional[str]

    # Metadata
    processing_steps: List[str]
    error: Optional[str]


# ============================================================================
# Agentic RAG Service
# ============================================================================

class AgenticRAGService:
    """LangGraph-based agentic RAG service with intelligent routing and verification"""

    def __init__(self):
        self.llm_studio_base_url = os.getenv("LLM_STUDIO_URL", "http://localhost:1234")
        self.router_model = os.getenv("ROUTER_MODEL", "qwen3")  # or nanbeige4-3b-thinking-2511
        self.synthesis_model = os.getenv("SYNTHESIS_MODEL", "command-r")
        self.verification_model = os.getenv("VERIFICATION_MODEL", "qwen3")

        # Build the LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create graph
        workflow = StateGraph(AgenticRAGState)

        # Add nodes
        workflow.add_node("route_query", self._route_query_node)
        workflow.add_node("execute_searches", self._execute_searches_node)
        workflow.add_node("link_primary_secondary", self._link_primary_secondary_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("verify_claims", self._verify_claims_node)
        workflow.add_node("finalize_response", self._finalize_response_node)

        # Define edges
        workflow.set_entry_point("route_query")
        workflow.add_edge("route_query", "execute_searches")
        workflow.add_edge("execute_searches", "link_primary_secondary")
        workflow.add_edge("link_primary_secondary", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "verify_claims")
        workflow.add_edge("verify_claims", "finalize_response")
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
        """Node: Route the query and create a search plan"""
        logger.info(f"Routing query: {state['user_query']}")

        # Define the search planning tool
        tools = [{
            "type": "function",
            "function": {
                "name": "create_search_plan",
                "description": "Create a comprehensive search plan for answering a query about the Cairo Genizah collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "description": "List of search actions to perform",
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
                                        ],
                                        "description": "Type of search: bibliography_* for secondary literature, primary_* for manuscript searches"
                                    },
                                    "query": {
                                        "type": "string",
                                        "description": "Search query or shelf mark"
                                    },
                                    "num_results": {
                                        "type": "integer",
                                        "default": 5
                                    },
                                    "semantic_weight": {
                                        "type": "integer",
                                        "default": 50,
                                        "description": "For hybrid searches: semantic weight 0-100"
                                    },
                                    "keyword_weight": {
                                        "type": "integer",
                                        "default": 50,
                                        "description": "For hybrid searches: keyword weight 0-100"
                                    },
                                    "exact_match": {
                                        "type": "boolean",
                                        "default": False,
                                        "description": "For shelfmark searches: exact match?"
                                    }
                                },
                                "required": ["search_type", "query"]
                            }
                        },
                        "needs_primary_secondary_linking": {
                            "type": "boolean",
                            "description": "Extract shelf marks from bibliography and fetch related primary sources"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of the search strategy"
                        }
                    },
                    "required": ["actions", "needs_primary_secondary_linking", "reasoning"]
                }
            }
        }]

        system_prompt = """You are a query router for the Cairo Genizah search system. Your job is to analyze queries and create optimal search plans.

**Available Search Types:**

**Bibliography Searches** (secondary scholarly literature):
- `bibliography_semantic`: Semantic search using embeddings (best for conceptual queries)
- `bibliography_hybrid`: Combines semantic + keyword (best for most queries)

**Primary Source Searches** (manuscripts):
- `primary_shelfmark`: Search by shelf mark (e.g., "T-S 8J22.22")
  - Use `exact_match: true` for precise shelf marks
  - Use `exact_match: false` for partial/fuzzy matching
- `primary_semantic`: Semantic search over manuscripts (for conceptual/content queries)
- `primary_keyword`: Keyword search in transcriptions/translations (for specific terms)
- `primary_hybrid`: Combines semantic + keyword (balanced approach)

**Query Patterns & Strategy:**

1. **Shelf marks mentioned** (T-S, CUL, MS, etc.):
   - Primary: `primary_shelfmark` with exact_match based on precision
   - Secondary: `bibliography_semantic` or `bibliography_hybrid` to find scholarship about it

2. **"What have scholars said about X?"**:
   - Focus on `bibliography_hybrid` or `bibliography_semantic`
   - Set `needs_primary_secondary_linking: true` to connect to manuscripts

3. **Content questions** ("find letters about trade"):
   - Primary: `primary_semantic` or `primary_hybrid`
   - Consider `bibliography_hybrid` for scholarly context

4. **Specific terms/names** ("documents mentioning 'Abraham ben Yiju'"):
   - Primary: `primary_keyword` or `primary_hybrid` (keyword_weight: 70)
   - Bibliography: `bibliography_hybrid` for scholarship

5. **Compound queries** ("T-S 8J22.22 and what scholars say"):
   - Multiple actions: `primary_shelfmark` + `bibliography_hybrid`

**Guidelines:**
- Simple single-fact queries: 1 action
- Medium complexity: 2-3 actions
- Complex research: 3-5 actions
- Always explain your reasoning
- Use `needs_primary_secondary_linking: true` unless query is ONLY about bibliography
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_query"]}
        ]

        # Call router model with tool forcing
        response = await self._call_llm_with_tools(
            messages=messages,
            tools=tools,
            model=self.router_model,
            tool_choice={"type": "function", "function": {"name": "create_search_plan"}}
        )

        # Parse the function call
        tool_call = response["choices"][0]["message"]["tool_calls"][0]
        arguments = json.loads(tool_call["function"]["arguments"])

        query_plan = QueryPlan(**arguments)

        state["query_plan"] = query_plan
        state["processing_steps"].append(f"Created search plan: {query_plan.reasoning}")

        logger.info(f"Query plan: {query_plan.reasoning}")
        logger.info(f"Actions: {len(query_plan.actions)}")

        return state

    @weave.op()
    async def _execute_searches_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Execute all planned searches"""
        query_plan = state["query_plan"]
        bibliography_results = []
        primary_source_results = []

        for action in query_plan.actions:
            logger.info(f"Executing {action.search_type}: {action.query}")

            try:
                if action.search_type == "bibliography_semantic":
                    # Semantic search in bibliography
                    search_request = BibliographyHybridSearchRequest(
                        query=action.query,
                        semanticWeight=100,
                        keywordWeight=0,
                        num_results=action.num_results,
                        page=1
                    )
                    response = await bibliography_search_service.search_hybrid(search_request)
                    bibliography_results.extend([{
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
                    } for r in response.results])

                elif action.search_type == "bibliography_hybrid":
                    # Hybrid search in bibliography
                    search_request = BibliographyHybridSearchRequest(
                        query=action.query,
                        semanticWeight=action.semantic_weight or 50,
                        keywordWeight=action.keyword_weight or 50,
                        num_results=action.num_results,
                        page=1
                    )
                    response = await bibliography_search_service.search_hybrid(search_request)
                    bibliography_results.extend([{
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
                    } for r in response.results])

                elif action.search_type == "primary_shelfmark":
                    # Shelf mark search
                    search_request = ShelfMarkSearchRequest(
                        shelf_mark=action.query,
                        exact_match=action.exact_match or False,
                        num_results=action.num_results
                    )
                    response = await search_service.search_by_shelfmark(search_request)
                    primary_source_results.extend([{
                        "doc_id": r.doc_id,
                        "shelf_mark": r.metadata.shelf_mark if r.metadata else None,
                        "title": r.metadata.title if r.metadata else None,
                        "description": r.metadata.description if r.metadata else None,
                        "transcription": r.metadata.transcription_full_text if r.metadata else None,
                        "translation": r.metadata.translation_full_text if r.metadata else None,
                        "image_urls": r.metadata.image_urls if r.metadata else None,
                        "similarity_score": r.similarity_score
                    } for r in response.results])

                elif action.search_type == "primary_semantic":
                    # Semantic search in primary sources
                    search_request = SearchRequest(
                        query=action.query,
                        filters=action.filters,
                        num_results=action.num_results
                    )
                    response = await search_service.search(search_request)
                    primary_source_results.extend([{
                        "doc_id": r.doc_id,
                        "shelf_mark": r.metadata.shelf_mark if r.metadata else None,
                        "title": r.metadata.title if r.metadata else None,
                        "description": r.metadata.description if r.metadata else None,
                        "transcription": r.metadata.transcription_full_text if r.metadata else None,
                        "translation": r.metadata.translation_full_text if r.metadata else None,
                        "image_urls": r.metadata.image_urls if r.metadata else None,
                        "similarity_score": r.similarity_score
                    } for r in response.results])

                elif action.search_type == "primary_keyword":
                    # Keyword search in primary sources
                    search_request = SearchRequest(
                        query=action.query,
                        filters=action.filters,
                        num_results=action.num_results
                    )
                    response = await search_service.search_by_keyword(search_request)
                    primary_source_results.extend([{
                        "doc_id": r.doc_id,
                        "shelf_mark": r.metadata.shelf_mark if r.metadata else None,
                        "title": r.metadata.title if r.metadata else None,
                        "description": r.metadata.description if r.metadata else None,
                        "transcription": r.metadata.transcription_full_text if r.metadata else None,
                        "translation": r.metadata.translation_full_text if r.metadata else None,
                        "image_urls": r.metadata.image_urls if r.metadata else None,
                        "similarity_score": r.similarity_score
                    } for r in response.results])

                elif action.search_type == "primary_hybrid":
                    # Hybrid search in primary sources
                    from search_service import SearchRequest as HybridSearchRequest
                    search_request = HybridSearchRequest(
                        query=action.query,
                        semanticWeight=action.semantic_weight or 50,
                        keywordWeight=action.keyword_weight or 50,
                        filters=action.filters,
                        num_results=action.num_results,
                        page=1
                    )
                    response = await search_service.search_hybrid(search_request)
                    primary_source_results.extend([{
                        "doc_id": r.doc_id,
                        "shelf_mark": r.metadata.shelf_mark if r.metadata else None,
                        "title": r.metadata.title if r.metadata else None,
                        "description": r.metadata.description if r.metadata else None,
                        "transcription": r.metadata.transcription_full_text if r.metadata else None,
                        "translation": r.metadata.translation_full_text if r.metadata else None,
                        "image_urls": r.metadata.image_urls if r.metadata else None,
                        "similarity_score": r.similarity_score
                    } for r in response.results])

            except Exception as e:
                logger.error(f"Search failed for {action.search_type}: {e}")
                state["processing_steps"].append(f"Search failed: {action.search_type} - {str(e)}")

        state["bibliography_results"] = bibliography_results
        state["primary_source_results"] = primary_source_results
        state["processing_steps"].append(
            f"Executed {len(query_plan.actions)} searches: "
            f"{len(bibliography_results)} bib results, {len(primary_source_results)} primary results"
        )

        return state

    @weave.op()
    async def _link_primary_secondary_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Extract shelf marks from bibliography and fetch related primary sources"""
        if not state["query_plan"].needs_primary_secondary_linking:
            state["processing_steps"].append("Skipping primary-secondary linking")
            return state

        # Extract all shelf marks mentioned in bibliography
        shelf_marks = set()
        for bib in state["bibliography_results"]:
            if bib.get("shelf_marks_mentioned"):
                shelf_marks.update(bib["shelf_marks_mentioned"])

        state["shelf_marks_to_fetch"] = list(shelf_marks)

        if not shelf_marks:
            state["processing_steps"].append("No shelf marks found in bibliography")
            return state

        logger.info(f"Fetching {len(shelf_marks)} primary sources from bibliography mentions")

        # Fetch primary sources for each shelf mark
        for shelf_mark in list(shelf_marks)[:20]:  # Limit to 20 to avoid overload
            try:
                search_request = ShelfMarkSearchRequest(
                    shelf_mark=shelf_mark,
                    exact_match=False,  # Liberal matching
                    num_results=1
                )
                response = await search_service.search_by_shelfmark(search_request)

                if response.results:
                    r = response.results[0]
                    # Only add if not already in results
                    if not any(ps["doc_id"] == r.doc_id for ps in state["primary_source_results"]):
                        state["primary_source_results"].append({
                            "doc_id": r.doc_id,
                            "shelf_mark": r.metadata.shelf_mark if r.metadata else None,
                            "title": r.metadata.title if r.metadata else None,
                            "description": r.metadata.description if r.metadata else None,
                            "transcription": r.metadata.transcription_full_text if r.metadata else None,
                            "translation": r.metadata.translation_full_text if r.metadata else None,
                            "image_urls": r.metadata.image_urls if r.metadata else None,
                            "similarity_score": r.similarity_score,
                            "linked_from_bibliography": True
                        })
            except Exception as e:
                logger.error(f"Failed to fetch shelf mark {shelf_mark}: {e}")

        state["processing_steps"].append(f"Linked {len(shelf_marks)} shelf marks to primary sources")

        return state

    @weave.op()
    async def _synthesize_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Synthesize answer from all retrieved context"""
        logger.info("Synthesizing answer with Command-R")

        # Build context from bibliography
        bib_context = []
        for i, bib in enumerate(state["bibliography_results"][:10], 1):  # Top 10
            authors = bib.get("authors") or [bib.get("author")] if bib.get("author") else ["Unknown"]
            author_str = ", ".join(authors)
            title = bib.get("title") or "Untitled"
            page = bib.get("extracted_page_number")
            page_str = f", p. {page}" if page else ""

            citation = f"[{title} by {author_str}{page_str}]"

            parts = [f"Citation: {citation}"]
            if bib.get("full_text"):
                text = bib["full_text"][:500]  # Truncate
                parts.append(f"Text: {text}")
            if bib.get("description"):
                parts.append(f"Description: {bib['description']}")

            bib_context.append("\n".join(parts))

        # Build context from primary sources
        primary_context = []
        for i, ps in enumerate(state["primary_source_results"][:10], 1):  # Top 10
            shelf_mark = ps.get("shelf_mark") or ps.get("doc_id")
            parts = [f"[Manuscript {shelf_mark}]"]

            if ps.get("description"):
                parts.append(f"Description: {ps['description']}")
            if ps.get("transcription"):
                trans = ps["transcription"][:300]
                parts.append(f"Transcription: {trans}")
            if ps.get("translation"):
                transl = ps["translation"][:300]
                parts.append(f"Translation: {transl}")

            primary_context.append("\n".join(parts))

        # Build system prompt
        system_prompt = """You are a Judaic Studies AI assistant specialized in the Cairo Genizah collection.

**CRITICAL CITATION REQUIREMENTS:**
1. ALWAYS cite sources using: **"Title by Author, p. Page"** in bold
2. Use *italics* for direct quotes: *"quoted text"*
3. NEVER make claims without citing a source
4. If you quote, it must be a SHORT excerpt (<15 words) from the provided text
5. ONE quote per source maximum
6. Default to paraphrasing - quotes should be rare exceptions

**FORBIDDEN:**
- Do NOT reproduce long quotes (15+ words is a violation)
- Do NOT use multiple quotes from the same source
- Do NOT make unsupported claims
- Do NOT cite sources not in the context

If the context doesn't answer the question, say so clearly."""

        # Build messages
        context_message = "**Bibliography Context:**\n\n" + "\n\n".join(bib_context)
        if primary_context:
            context_message += "\n\n**Primary Source Context:**\n\n" + "\n\n".join(primary_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
            {"role": "user", "content": state["user_query"]}
        ]

        # Generate answer
        draft_answer = await self._call_llm(
            messages=messages,
            model=self.synthesis_model,
            temperature=0.3  # Lower temperature for factual accuracy
        )

        state["draft_answer"] = draft_answer
        state["processing_steps"].append("Synthesized draft answer")

        return state

    @weave.op()
    async def _verify_claims_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Verify claims in the draft answer against retrieved context"""
        logger.info("Verifying claims")

        # Extract claims from draft answer using verification model
        extract_prompt = f"""Extract all factual claims from this answer. For each claim, identify:
1. The claim itself
2. The citation given (if any)
3. Any direct quote used

Answer to analyze:
{state['draft_answer']}

Return a JSON array of claims in this format:
[
  {{
    "claim": "the factual assertion",
    "citation": "Author, Title, Page" or null,
    "quote": "exact quote" or null
  }}
]"""

        messages = [{"role": "user", "content": extract_prompt}]
        claims_json = await self._call_llm(messages, self.verification_model, temperature=0.1)

        # Parse claims
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in claims_json:
                claims_json = claims_json.split("```json")[1].split("```")[0]
            elif "```" in claims_json:
                claims_json = claims_json.split("```")[1].split("```")[0]

            claims = json.loads(claims_json.strip())
        except Exception as e:
            logger.error(f"Failed to parse claims: {e}")
            claims = []

        verified_claims = []

        # Verify each claim
        for claim_data in claims:
            claim_text = claim_data.get("claim", "")
            citation = claim_data.get("citation")
            quote = claim_data.get("quote")

            # Find the source in our context
            found_source = None
            for bib in state["bibliography_results"]:
                title = bib.get("title", "")
                authors = bib.get("authors", []) or [bib.get("author")]
                page = bib.get("extracted_page_number")

                # Check if citation matches
                if citation and any(author in citation for author in authors if author):
                    found_source = bib
                    break

            if not found_source:
                # Check primary sources
                for ps in state["primary_source_results"]:
                    shelf_mark = ps.get("shelf_mark")
                    if citation and shelf_mark and shelf_mark in citation:
                        found_source = ps
                        break

            # Verify the claim
            if found_source:
                # Simple verification: check if claim concepts appear in source
                source_text = (found_source.get("full_text") or
                               found_source.get("description") or
                               found_source.get("transcription") or
                               found_source.get("translation") or "")

                # Check quote length if present
                if quote and len(quote.split()) >= 15:
                    verification_status = "NOT_SUPPORTED"
                    reasoning = f"Quote is too long ({len(quote.split())} words, max 15)"
                    confidence = 0.0
                else:
                    # Basic keyword overlap check
                    claim_words = set(claim_text.lower().split())
                    source_words = set(source_text.lower().split())
                    overlap = len(claim_words & source_words)

                    if overlap > len(claim_words) * 0.5:
                        verification_status = "SUPPORTED"
                        confidence = 0.8
                        reasoning = "Claim content found in source"
                    elif overlap > len(claim_words) * 0.3:
                        verification_status = "PARTIALLY_SUPPORTED"
                        confidence = 0.5
                        reasoning = "Some claim content found in source"
                    else:
                        verification_status = "NOT_SUPPORTED"
                        confidence = 0.2
                        reasoning = "Minimal overlap with source"
            else:
                verification_status = "NOT_FOUND"
                confidence = 0.0
                reasoning = "Source not found in retrieved context"

            verified_claims.append(VerifiedClaim(
                claim=claim_text,
                source_citation=citation or "No citation",
                quote=quote,
                verification_status=verification_status,
                confidence=confidence,
                reasoning=reasoning
            ))

        # Build verification summary
        summary = {
            "SUPPORTED": len([c for c in verified_claims if c.verification_status == "SUPPORTED"]),
            "PARTIALLY_SUPPORTED": len([c for c in verified_claims if c.verification_status == "PARTIALLY_SUPPORTED"]),
            "NOT_SUPPORTED": len([c for c in verified_claims if c.verification_status == "NOT_SUPPORTED"]),
            "NOT_FOUND": len([c for c in verified_claims if c.verification_status == "NOT_FOUND"])
        }

        state["verified_claims"] = verified_claims
        state["verification_summary"] = summary
        state["processing_steps"].append(f"Verified {len(verified_claims)} claims")

        return state

    @weave.op()
    async def _finalize_response_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Finalize the response, potentially revising based on verification"""
        # Check verification results
        unsupported_count = state["verification_summary"].get("NOT_SUPPORTED", 0)
        not_found_count = state["verification_summary"].get("NOT_FOUND", 0)

        if unsupported_count > 0 or not_found_count > 0:
            # Flag issues but don't regenerate for now
            warning = f"\n\n**Note:** {unsupported_count} claims could not be verified, {not_found_count} citations not found in sources."
            state["final_answer"] = state["draft_answer"] + warning
            state["processing_steps"].append(f"Added verification warning ({unsupported_count} unsupported)")
        else:
            state["final_answer"] = state["draft_answer"]
            state["processing_steps"].append("All claims verified, no revision needed")

        return state

    @weave.op()
    async def chat(
            self,
            user_query: str,
            conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AgenticRAGResponse:
        """Main entry point for agentic RAG chat"""

        # Initialize state
        initial_state: AgenticRAGState = {
            "user_query": user_query,
            "conversation_history": conversation_history,
            "query_plan": None,
            "bibliography_results": [],
            "primary_source_results": [],
            "shelf_marks_to_fetch": [],
            "draft_answer": None,
            "verified_claims": [],
            "verification_summary": {},
            "final_answer": None,
            "processing_steps": [],
            "error": None
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        # Build response
        return AgenticRAGResponse(
            answer=final_state["final_answer"] or "Unable to generate answer",
            query_plan=final_state["query_plan"],
            bibliography_results=final_state["bibliography_results"],
            primary_source_results=final_state["primary_source_results"],
            verified_claims=final_state["verified_claims"],
            verification_summary=final_state["verification_summary"],
            processing_steps=final_state["processing_steps"]
        )


# Global service instance
agentic_rag_service = AgenticRAGService()