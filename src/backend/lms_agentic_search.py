# agentic_rag_service.py
"""
LangGraph-based agentic RAG service for Cairo Genizah collection.

CORE PRINCIPLE: Scholarly synthesis first, with embedded manuscript references.
Primary sources are supplementary evidence, not the main content.
"""

import os
import re
import logging
import time
import unicodedata
from contextlib import asynccontextmanager
from contextvars import ContextVar
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Literal, TypedDict, Set, Callable, TypeVar
from pydantic import BaseModel, Field, ValidationError
import json
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from src.backend.shelfmark_normalizer import detect_shelfmarks, ShelfmarkNormalizer
from src.backend.genizah_terminology import (
    aliases_for_concept,
    expand_query_aliases,
    find_concepts,
    hebrew_char_ratio,
)

from langgraph.graph import StateGraph, END

from src.backend.search_service import search_service, SearchRequest, DocumentMetadata
from src.backend.search_bibliography import bibliography_search_service, BibliographyHybridSearchRequest
from src.backend.neo4j_service import neo4j_service, build_direct_work_link
from src.backend.missing_fragments import missing_fragment_tracker
logger = logging.getLogger(__name__)

Operation = TypeVar("Operation", bound=Callable[..., Any])

# Per-request metrics sink and the pipeline stage currently executing. Set by
# the chat entry points / graph node wrapper; read by the LLM call helpers so
# each model call is attributed to its stage. Context variables propagate into
# the tasks LangGraph and the streaming producer create.
_REQUEST_METRICS: ContextVar[Optional[Dict[str, Any]]] = ContextVar("_REQUEST_METRICS", default=None)
_CURRENT_STAGE: ContextVar[str] = ContextVar("_CURRENT_STAGE", default="")


def record_llm_call(model: str, result: Dict[str, Any], seconds: float, with_tools: bool = False) -> None:
    """Record one LM Studio call in the current request's metrics, if any.

    :param model: Model id the request was served by.
    :param result: Raw chat-completion response (its ``usage`` block is read).
    :param seconds: Wall time of the HTTP call.
    :param with_tools: Whether the call used function calling.
    """
    sink = _REQUEST_METRICS.get()
    if sink is None:
        return
    usage = result.get("usage") or {} if isinstance(result, dict) else {}
    details = usage.get("completion_tokens_details") or {}
    sink.setdefault("llm_calls", []).append({
        "stage": _CURRENT_STAGE.get(),
        "model": model,
        "with_tools": with_tools,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        # Thinking models spend output budget reasoning before answering;
        # LM Studio reports that share separately when it can.
        "reasoning_tokens": details.get("reasoning_tokens"),
        "seconds": round(seconds, 3),
    })


class LMStudioGateway:
    """Serializes access to the single local inference server.

    LM Studio holds one model resident and processes requests one at a time, so
    unbounded concurrency does not add throughput — it multiplies every user's
    latency (two simultaneous chats each make up to nine model calls) and pushes
    requests past proxy timeouts. Admitting a bounded number of calls at a time
    keeps latency predictable and lets the UI tell a waiting user where they
    stand.
    """

    def __init__(self) -> None:
        self._max_concurrency = max(1, int(os.getenv("LM_STUDIO_MAX_CONCURRENCY", "1")))
        self._semaphore: Optional[Any] = None
        self.waiting = 0
        self.active = 0
        self.completed = 0

    def _ensure_semaphore(self) -> Any:
        """Create the semaphore lazily, inside the running event loop.

        :returns: The shared concurrency semaphore.
        :rtype: asyncio.Semaphore
        """
        import asyncio

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    def snapshot(self) -> Dict[str, int]:
        """Report queue state for progress messages.

        :returns: Waiting, active, and completed model-call counts.
        :rtype: Dict[str, int]
        """
        return {"waiting": self.waiting, "active": self.active, "completed": self.completed}

    @asynccontextmanager
    async def slot(self, label: str):
        """Acquire a model-call slot, waiting behind other in-flight calls.

        :param label: Short description of the call, for logging.
        :yields: Control once a slot is available.
        """
        semaphore = self._ensure_semaphore()
        if semaphore.locked():
            self.waiting += 1
            logger.info(
                "Model call '%s' queued (%s waiting, %s active)",
                label, self.waiting, self.active,
            )
            try:
                await semaphore.acquire()
            finally:
                self.waiting -= 1
        else:
            await semaphore.acquire()
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1
            self.completed += 1
            semaphore.release()


lm_studio_gateway = LMStudioGateway()


class ModelUnavailableError(RuntimeError):
    """The local inference server could not serve a request.

    Raised for conditions the user can understand and act on — the model is
    unloaded, the server is out of memory, or it is busy with other work —
    rather than surfacing an HTTP stack trace as a blank answer.
    """


LOCAL_MODEL_CAPACITY_MESSAGE = (
    "**The AI Assistant is temporarily unavailable** — the language model is out of memory "
    "or busy with another workload on the same machine.\n\n"
    "This project runs entirely on a single Mac Studio that also does the research training "
    "work, so the Assistant can briefly go offline when both compete for resources. Search, "
    "browsing, and the collection explorer all still work normally.\n\n"
    "Please try again in a few minutes. "
    "[Why does this happen?](/faq#hardware) — with funding for a dedicated serving machine, "
    "this would not occur."
)


def _is_model_unavailable_error(response: Any) -> bool:
    """Detect LM Studio's "model is not loaded" rejection.

    LM Studio answers 400 with this when the requested model was evicted or
    was never loaded and just-in-time loading did not engage — a recoverable
    condition on a machine where models come and go, not a bad request.

    :param response: httpx response from a chat-completions call.
    :returns: Whether the error indicates an unloaded/unavailable model.
    :rtype: bool
    """
    if getattr(response, "status_code", None) != 400:
        return False
    try:
        text = response.text.lower()
    except Exception:
        return False
    return (
        "not started loading" in text
        or "has been unloaded" in text
        # Newer LM Studio builds phrase eviction as {"error":"Model unloaded."}
        or "model unloaded" in text
        or "model_not_found" in text
    )


class _NoOpWeave:
    """Provide a decorator-compatible no-op when tracing is disabled."""

    @staticmethod
    def op() -> Callable[[Operation], Operation]:
        """Return an identity decorator compatible with ``weave.op``.

        :returns: A decorator that returns the original callable.
        :rtype: Callable[[Operation], Operation]
        """
        def decorator(function: Operation) -> Operation:
            """Return the callable without wrapping it.

            :param function: Callable that would otherwise be traced.
            :returns: The unmodified callable.
            :rtype: Operation
            """
            return function

        return decorator


if os.getenv("WEAVE_ENABLED", "false").lower() in {"1", "true", "yes"}:
    import weave

    try:
        weave.init(os.getenv("WANDB_PROJECT", "cairo-genizah-agentic-rag"))
    except Exception as exc:
        logger.warning("Weave initialization failed; continuing without tracing: %s", exc)
else:
    weave = _NoOpWeave()

# ============================================================================
# Constants
# ============================================================================

# If ALL bibliography results score below this threshold, retrieval has likely
# failed (nearest-neighbor garbage rather than genuine matches). Trigger retry.
SIMILARITY_THRESHOLD = 0.4
MAX_VERIFICATION_REPAIR_ATTEMPTS = 3
# Total character budget for bibliography evidence in LLM prompts. Keeps the
# synthesis/verification prompt within the loaded model context (Hebrew-heavy
# pages tokenize near one token per character, so budget conservatively).
EVIDENCE_CHAR_BUDGET = int(os.getenv("EVIDENCE_CHAR_BUDGET", "15000"))
# Output budget for the synthesis call. Thinking models routinely spend 90%+ of
# output tokens reasoning (observed up to the full 8192), so synthesis gets a
# larger ceiling than utility calls.
SYNTHESIS_MAX_TOKENS = int(os.getenv("SYNTHESIS_MAX_TOKENS", "16384"))
DIRECT_SEARCH_ACTION_TYPES = {
    "bibliography_semantic",
    "bibliography_hybrid",
    "primary_semantic",
    "primary_keyword",
    "primary_hybrid",
    "primary_shelfmark",
    "graph_scholar",
}


# ============================================================================
# Pydantic Models
# ============================================================================

class ShelfMarkSearchRequest(BaseModel):
    """Request model for resolving a manuscript shelf mark.

    Kept with the agentic service so importing the active RAG pipeline does not
    initialize the deprecated RAG service and its observability side effects.
    """

    shelf_mark: str = Field(..., min_length=1, max_length=100)
    exact_match: bool = False
    num_results: int = Field(default=10, ge=1, le=50)
    include_embeddings: bool = False
    index_name: Optional[str] = None


class SearchAction(BaseModel):
    """Action to perform a search"""
    search_type: Literal[
        "bibliography_semantic",
        "bibliography_hybrid",
        "primary_semantic",
        "primary_keyword",
        "primary_hybrid",
        "primary_shelfmark",
        "graph_scholar",
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
    verification_status: Literal["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Verification reasoning")


class FlaggedClaim(BaseModel):
    """An unsupported-but-not-contradicted claim surfaced to the user.

    Soft failures no longer trigger destructive re-synthesis. Instead the
    claim is kept in the answer, wrapped in inline flag markers, and returned
    here so the UI can highlight it and show the verifier's exact reasoning.
    """

    flag_id: int = Field(..., description="Matches the inline ⟦flag:N⟧ marker in the answer")
    claim_type: str = Field(..., description="attribution, factual_claim, graph_claim, citation_claim, or verification_error")
    text: str = Field(..., description="The claim as extracted by the verifier")
    answer_span: Optional[str] = Field(
        None, description="Sentence in the answer the claim was anchored to, if located"
    )
    reason: str = Field(..., description="The verifier's reasoning for not supporting the claim")
    source_citation: Optional[str] = Field(None, description="Citation the claim referenced, if any")


class AgenticRAGResponse(BaseModel):
    """Final response from the agentic RAG system"""
    answer: str = Field(..., description="The synthesized answer or error message")
    success: bool = Field(..., description="Whether answer generation succeeded")
    error_type: Optional[str] = Field(None, description="Type of error if failed")
    resolved_query: Optional[str] = Field(
        None,
        description="The user's message restated as a standalone question when it was a "
                    "follow-up that needed context; None when the message was searched as written",
    )
    is_followup: bool = Field(
        default=False,
        description="Whether the message was interpreted in the context of earlier turns",
    )
    query_plan: Optional[QueryPlan] = Field(None)
    bibliography_results: List[Dict[str, Any]] = Field(default_factory=list)
    primary_source_results: List[Dict[str, Any]] = Field(default_factory=list)
    graph_results: List[Dict[str, Any]] = Field(default_factory=list)
    verified_claims: List[VerifiedClaim] = Field(default_factory=list)
    verification_summary: Dict[str, int] = Field(default_factory=dict)
    flagged_claims: List[FlaggedClaim] = Field(
        default_factory=list,
        description="Unsupported claims kept in the answer and flagged for user review",
    )
    processing_steps: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Run metrics: total_seconds, stage_timings, stage_calls, "
                    "verification_cycles, repair_attempts, llm_calls (per-call model, "
                    "stage, tokens, seconds), synthesis_model",
    )


# ============================================================================
# LangGraph State
# ============================================================================

class AgenticRAGState(TypedDict):
    """State for the agentic RAG graph"""
    user_query: str
    conversation_history: Optional[List[Dict[str, str]]]
    # The message restated as a standalone question when it is a follow-up
    # ("the verses" -> "the verses of the Kol Nidre piyyut ..."); retrieval
    # and relevance checks use this, never the bare follow-up wording.
    resolved_query: Optional[str]
    is_followup: bool
    synthesis_model_override: Optional[str]  # Private eval override; public chat uses the default.
    query_plan: Optional[QueryPlan]
    bibliography_results: List[Dict[str, Any]]
    primary_source_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    resolved_entities: List[Dict[str, Any]]
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
    excluded_claims: List[Dict[str, str]]  # HARD failures only (fabricated quotes/shelf marks,
                                           # contradicted claims); these trigger targeted repair.
                                           # Each entry: {"type": ..., "text": ..., "reason": ...}
    soft_flagged_claims: List[Dict[str, Any]]  # Unsupported-but-not-contradicted claims kept in
                                               # the answer and surfaced to the UI as flags.
    subject_terms_not_found: List[str]  # Distinctive query terms absent from all retrieved pages
    work_manuscripts: Dict[str, List[Dict[str, str]]]  # Cited work -> fragments it is based on
    supported_evidence_units: List[Dict[str, Any]]
    verification_feedback_history: List[Dict[str, Any]]
    # Per-node wall time and call counts, filled by the graph wrapper so
    # responses can report where time went and how many verify/repair
    # cycles ran (evaluation of synthesis models depends on both).
    stage_timings: Dict[str, float]
    stage_calls: Dict[str, int]


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


def should_retry_verification(failure_count: int) -> bool:
    """Return whether another targeted synthesis repair is permitted.

    The initial synthesis is followed by at most three repairs, giving the
    verifier four total opportunities to accept an answer.

    :param failure_count: Number of failed verification passes so far.
    :returns: ``True`` while a targeted repair remains available.
    :rtype: bool
    """
    return 0 < failure_count <= MAX_VERIFICATION_REPAIR_ATTEMPTS


def collect_supported_evidence_units(state: AgenticRAGState) -> List[Dict[str, Any]]:
    """Collect deduplicated claims supported in any verification pass.

    Carrying successful claim decisions forward prevents a later repair from
    silently dropping useful evidence that an earlier pass had already tied to
    a numbered source.

    :param state: Current RAG state including verification feedback history.
    :returns: Supported evidence units in first-seen order.
    :rtype: List[Dict[str, Any]]
    """
    candidates: List[Dict[str, Any]] = []
    for feedback in state.get("verification_feedback_history", []):
        candidates.extend(feedback.get("supported_claims", []))
    candidates.extend(state.get("supported_evidence_units", []))

    collected: List[Dict[str, Any]] = []
    seen: Set[tuple[str, str, str]] = set()
    for unit in candidates:
        citation = str(unit.get("citation") or "")
        unit_type = str(unit.get("type") or "factual_claim")
        unit_text = str(unit.get("text") or "")
        key = (citation, unit_type, _normalize_verification_text(unit_text))
        if not unit_text or key in seen:
            continue
        seen.add(key)
        collected.append(dict(unit))
    return collected


def all_results_below_threshold(results: List[Dict[str, Any]], threshold: float) -> bool:
    """Return True if every result's similarity_score is below the threshold."""
    if not results:
        return True
    return all((r.get("similarity_score") or 0.0) < threshold for r in results)


# Words that situate a query in this collection rather than describing its
# topic. A page matching only these has matched the corpus, not the question.
COLLECTION_CONTEXT_TERMS: Set[str] = {
    "cairo", "genizah", "geniza", "genizot", "fragment", "fragments", "manuscript",
    "manuscripts", "document", "documents", "text", "texts", "collection", "scholarship",
    "literature", "study", "studies", "research", "jewish", "hebrew", "medieval",
    "about", "regarding", "concerning", "tell", "know", "what", "which", "who", "whom",
    "when", "where", "why", "how", "does", "did", "do", "can", "could", "would", "there",
    "any", "some", "the", "and", "for", "from", "with", "was", "were", "are", "have",
    "has", "that", "this", "these", "those", "into", "made", "their", "them", "they",
    "you", "your", "our", "his", "her", "its", "been", "being", "other", "more", "most",
    "such", "than", "then", "also", "only", "over", "under", "between", "during",
}


def extract_topical_terms(query: str, min_length: int = 4) -> List[str]:
    """Extract the distinctive content terms a query is actually asking about.

    Collection-context words ("Cairo Genizah", "fragments") are removed: they
    match nearly every document in the corpus and so carry no information
    about whether retrieval found the requested subject.

    :param query: Raw user query.
    :param min_length: Minimum term length to consider distinctive.
    :returns: Lowercased distinctive terms in first-seen order.
    :rtype: List[str]
    """
    tokens = re.findall(r"[\w֐-׿']+", _normalize_verification_text(query))
    terms: List[str] = []
    for token in tokens:
        cleaned = token.strip("'")
        if len(cleaned) < min_length or cleaned in COLLECTION_CONTEXT_TERMS:
            continue
        if cleaned.isdigit():
            continue
        if cleaned not in terms:
            terms.append(cleaned)
    return terms


def term_appears_in_text(term: str, text: str) -> bool:
    """Check for a term in text, tolerating inflection via prefix matching.

    Longer terms match on a truncated stem so that "ketubba" also matches
    "ketubbot" and "zemirot" matches "zemirah".

    :param term: Distinctive query term, already lowercased.
    :param text: Lowercased document text to search.
    :returns: Whether the term (or its stem) occurs in the text.
    :rtype: bool
    """
    if term in text:
        return True
    if len(term) >= 6:
        return term[: max(5, len(term) - 2)] in text
    return False


def evidence_addresses_query(
    query: str,
    results: List[Dict[str, Any]],
) -> tuple[bool, List[str]]:
    """Report whether retrieved evidence mentions any distinctive query term.

    Dense retrieval always returns its nearest neighbours, and for a query
    naming a subject absent from the corpus those neighbours are pages that
    merely share the collection context — which then get summarized as though
    they answered the question. When no retrieved page contains any
    distinctive term, retrieval has not found the requested subject.

    :param query: Raw user query.
    :param results: Retrieved bibliography result dictionaries.
    :returns: Whether the evidence addresses the query, and the terms checked.
    :rtype: tuple[bool, List[str]]
    """
    topical_terms = extract_topical_terms(query)
    normalized_query = _normalize_verification_text(query)
    # Domain concepts let a query's spelling match the corpus's spelling:
    # "kinnot" and the corpus's "qinot" are the same subject.
    query_concepts = find_concepts(normalized_query)
    concept_forms = [
        form for concept in query_concepts for form in aliases_for_concept(concept)
    ]
    if not topical_terms and not concept_forms:
        return True, topical_terms
    if not results:
        return True, topical_terms
    query_is_hebrew = hebrew_char_ratio(query) > 0.5
    judgeable_results = 0
    for result in results:
        haystack = " ".join(
            str(result.get(field) or "")
            for field in ("title", "full_text", "description", "author")
        )
        haystack = _normalize_verification_text(haystack)
        subject_keywords = result.get("subject_keywords") or []
        if isinstance(subject_keywords, list):
            haystack += " " + _normalize_verification_text(" ".join(map(str, subject_keywords)))
        # A page written in the other script can only be judged through its
        # metadata bridges (title/description/keywords); count it judgeable
        # only when those carry enough query-script text to test against.
        page_is_hebrew = hebrew_char_ratio(str(result.get("full_text") or "")) > 0.5
        if page_is_hebrew != query_is_hebrew:
            bridge = " ".join(
                str(result.get(field) or "") for field in ("title", "description")
            )
            if isinstance(subject_keywords, list):
                bridge += " " + " ".join(map(str, subject_keywords))
            bridge_judgeable = (
                hebrew_char_ratio(bridge) > 0.5 if query_is_hebrew
                else sum(1 for c in bridge if c.isascii() and c.isalpha()) >= 40
            )
            if bridge_judgeable:
                judgeable_results += 1
        else:
            judgeable_results += 1
        if any(term_appears_in_text(term, haystack) for term in topical_terms):
            return True, topical_terms
        if any(form in haystack for form in concept_forms):
            return True, topical_terms
    if judgeable_results == 0:
        # Every retrieved page is cross-script with no usable metadata bridge:
        # absence of the query's terms proves nothing, so do not block.
        return True, topical_terms
    return False, topical_terms


def bibliography_result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a bibliography result model into agent state data.

    :param result: BibliographySearchResult-like object.
    :returns: Serializable bibliography evidence dictionary.
    :rtype: Dict[str, Any]
    """
    return {
        "doc_id": result.doc_id,
        "title": result.title,
        "authors": result.authors,
        "author": result.author,
        "description": result.description,
        "full_text": result.full_text,
        "extracted_page_number": result.extracted_page_number,
        "shelf_marks_mentioned": result.shelf_marks_mentioned,
        "subject_keywords": result.subject_keywords,
        "similarity_score": result.similarity_score,
        "retrieval_details": getattr(result, "retrieval_details", {}) or {},
        "metadata": getattr(result, "metadata", {}) or {},
    }


def deduplicate_bibliography_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate bibliography chunks and keep the best-ranked occurrence.

    :param results: Possibly overlapping results from multiple retrieval actions.
    :returns: Unique results ordered by descending normalized retrieval score.
    :rtype: List[Dict[str, Any]]
    """
    by_doc_id: Dict[str, Dict[str, Any]] = {}
    for result in results:
        # Prefer the immutable ES _id: the doc_id field can be duplicated
        # across distinct pages, and deduplicating on it drops real evidence.
        doc_id = (result.get("metadata") or {}).get("_es_id") or result.get("doc_id")
        if not doc_id:
            continue
        existing = by_doc_id.get(doc_id)
        if existing is None or (result.get("similarity_score") or 0.0) > (
            existing.get("similarity_score") or 0.0
        ):
            by_doc_id[doc_id] = result
    return sorted(
        by_doc_id.values(),
        key=lambda result: result.get("similarity_score") or 0.0,
        reverse=True,
    )


def build_graph_prompt_context(graph_results: List[Dict[str, Any]]) -> str:
    """Format bounded Neo4j evidence while preserving graph provenance.

    :param graph_results: Structured neighborhoods returned by Neo4jService.
    :returns: Human-readable graph evidence for the synthesis prompt.
    :rtype: str
    """
    sections: List[str] = []
    for evidence in graph_results:
        scholar = evidence.get("scholar") or {}
        scholar_name = scholar.get("name") or "Unknown scholar"
        lines = [
            f"Scholar node: {scholar_name}",
            f"Graph data sources: {', '.join(scholar.get('data_sources') or []) or 'unspecified'}",
            "BookArticle nodes connected by WROTE:",
        ]
        for work in (evidence.get("works") or [])[:15]:
            title = work.get("title") or "Untitled"
            year = f" ({work['year']})" if work.get("year") else ""
            count = work.get("referenced_fragment_count") or 0
            samples = [
                sample.get("shelfmark")
                for sample in (work.get("referenced_fragment_samples") or [])[:3]
                if sample.get("shelfmark")
            ]
            sample_text = f"; sample shelf marks: {', '.join(samples)}" if samples else ""
            lines.append(
                f"- {title}{year}; article_id={work.get('article_id')}; "
                f"references {count} distinct Fragment nodes{sample_text}"
            )

        studied_samples = [
            sample.get("shelfmark")
            for sample in (evidence.get("studied_fragment_samples") or [])[:5]
            if sample.get("shelfmark")
        ]
        lines.append(
            f"STUDIED relationships: {evidence.get('studied_fragment_count', 0)} distinct fragments"
            + (f"; sample shelf marks: {', '.join(studied_samples)}" if studied_samples else "")
        )
        relationships = evidence.get("relationships") or []
        if relationships:
            lines.append("Other graph relationships:")
            for relationship in relationships[:20]:
                lines.append(
                    f"- {relationship.get('relationship')}: {relationship.get('name')} "
                    f"({', '.join(relationship.get('labels') or [])})"
                )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def extract_quoteable_main_text(full_text: str) -> str:
    """Extract original page text from an enriched Elasticsearch text blob.

    Bibliography pages may prefix the OCR/transcription with generated metadata
    and a generated summary. Only the content after ``Main text:`` is safe to
    present to the model as verbatim, quoteable scholarship. Older records that
    do not contain the marker are treated as already-original page text.

    :param full_text: Elasticsearch ``full_text_content`` value.
    :returns: Original page text suitable for short direct quotations.
    :rtype: str
    """
    text = str(full_text or "").strip()
    marker_match = re.search(r"\bMain text:\s*", text, flags=re.IGNORECASE)
    if marker_match:
        return text[marker_match.end():].strip()
    return text


def build_bibliography_source_context(
    bibliography_results: List[Dict[str, Any]],
    limit: int = 8,
    total_char_budget: int = EVIDENCE_CHAR_BUDGET,
) -> List[Dict[str, Any]]:
    """Build one bounded evidence representation shared by synthesis and verification.

    Keeping the prompt text identical prevents the synthesizer from seeing text
    that is later truncated away before verification. The per-source page-text
    cap adapts to ``total_char_budget`` so evidence-heavy retrievals (e.g.
    graph-scholar queries adding author-constrained pages) cannot push the
    prompt past the loaded model context.

    :param bibliography_results: Retrieved bibliography result dictionaries.
    :param limit: Maximum number of bibliography chunks to include.
    :param total_char_budget: Approximate cap on total evidence characters.
    :returns: Source dictionaries containing citation and prompt text.
    :rtype: List[Dict[str, Any]]
    """
    selected = bibliography_results[:limit]
    if not selected:
        return []
    # Citation line, truncated description, and shelf-mark list cost roughly
    # this much per source; the remaining budget goes to quoteable page text.
    per_source_fixed_chars = 700
    quote_budget = max(2000, total_char_budget - per_source_fixed_chars * len(selected))
    quote_cap = min(1800, max(400, quote_budget // len(selected)))
    # Hebrew tokenizes near one token per character (vs ~4 chars/token for
    # English), so an equal character cap makes a Hebrew page cost ~4x the
    # prompt tokens. Cap Hebrew-dominant page text by equivalent token cost,
    # with a floor that keeps the slice substantive.
    hebrew_quote_cap = max(500, quote_cap // 3)

    sources: List[Dict[str, Any]] = []
    for source_number, bibliography in enumerate(selected, start=1):
        authors = bibliography.get("authors") or (
            [bibliography.get("author")] if bibliography.get("author") else ["Unknown"]
        )
        author_text = ", ".join(str(author) for author in authors if author) or "Unknown"
        title = bibliography.get("title") or "Untitled"
        page = bibliography.get("extracted_page_number")
        citation = f"{author_text}, *{title}*" + (f", p. {page}" if page else "")
        parts = [f"[SOURCE {source_number}] {citation}"]
        full_text = str(bibliography.get("full_text") or "").strip()
        quoteable_text = extract_quoteable_main_text(full_text)
        description = str(bibliography.get("description") or "").strip()
        shelf_marks = filter_manuscript_shelfmarks(bibliography.get("shelf_marks_mentioned"))
        if description:
            parts.append(
                "Generated catalog summary (orientation only; never quote): "
                f"{description[:500]}"
            )
        effective_cap = (
            hebrew_quote_cap if hebrew_char_ratio(quoteable_text) > 0.5 else quote_cap
        )
        if quoteable_text:
            parts.append(f"Original page text (quoteable): {quoteable_text[:effective_cap]}")
        if shelf_marks:
            parts.append(
                "Shelf marks cited in this source: "
                + ", ".join(str(shelf_mark) for shelf_mark in shelf_marks[:10])
            )
        prompt_text = "\n".join(parts)
        sources.append({
            "source_number": source_number,
            "citation": citation,
            "prompt_text": prompt_text,
            "evidence_text": "\n".join(parts[1:]),
            "quoteable_text": quoteable_text[:effective_cap],
        })
    return sources


def build_verification_sources(
    bibliography_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build numbered bibliography and Neo4j evidence for the verifier.

    :param bibliography_results: Retrieved bibliography result dictionaries.
    :param graph_results: Structured Neo4j evidence dictionaries.
    :returns: Uniform source dictionaries with stable source numbers.
    :rtype: List[Dict[str, Any]]
    """
    sources = build_bibliography_source_context(bibliography_results)
    for graph_evidence in graph_results:
        source_number = len(sources) + 1
        scholar_name = (graph_evidence.get("scholar") or {}).get("name") or "Unknown scholar"
        citation = f"Neo4j knowledge graph record for {scholar_name}"
        evidence_text = build_graph_prompt_context([graph_evidence])[:2400]
        sources.append({
            "source_number": source_number,
            "citation": citation,
            "prompt_text": f"[SOURCE {source_number}] {citation}\n{evidence_text}",
            "evidence_text": evidence_text,
        })
    return sources


def _normalize_verification_text(text: str) -> str:
    """Normalize Unicode, punctuation, and whitespace for evidence matching.

    :param text: Text to normalize.
    :returns: Conservatively normalized lowercase text.
    :rtype: str
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.translate(str.maketrans({
        "“": '"', "”": '"', "’": "'", "–": "-", "—": "-",
        # Hebrew punctuation: gershayim/geresh quote/abbreviation marks and
        # maqaf, so a draft quoting with ASCII marks matches evidence that
        # uses the Hebrew forms (and vice versa).
        "״": '"', "׳": "'", "־": "-",
    }))
    # Strip Hebrew vocalization and cantillation: LLM output is typically
    # unpointed, while piyyut/Bible transcriptions in the corpus are pointed —
    # without this, a correct quote never matches its own source.
    normalized = re.sub(r"[֑-ׇֽֿׁׂׅׄ]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_direct_quotes(text: str) -> List[str]:
    """Extract substantive text enclosed in straight or curly double quotes.

    :param text: Draft answer text.
    :returns: Deduplicated quote bodies in appearance order.
    :rtype: List[str]
    """
    matches = re.findall(r'"([^"\n]{8,})"|“([^”\n]{8,})”', text or "")
    candidates: List[str] = [straight or curly for straight, curly in matches]
    # Hebrew quotations wrapped in gershayim (״…״). Word-internal gershayim
    # mark abbreviations (כ״י), so a quotation's marks must sit at word
    # boundaries — otherwise abbreviation marks would masquerade as quotes.
    candidates.extend(re.findall(r"(?:^|(?<=[\s(:—–\-]))״([^״\n]{8,})״(?=[\s).,;:!?]|$)", text or ""))
    quotes: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        quote = candidate.strip()
        normalized = _normalize_verification_text(quote)
        if normalized and normalized not in seen:
            seen.add(normalized)
            quotes.append(quote)
    return quotes


def bound_direct_quote(quote: str, max_words: int = 30) -> str:
    """Limit a direct quotation to a short copyright-conscious excerpt.

    :param quote: Verified source wording to bound.
    :param max_words: Maximum number of whitespace-delimited words to retain.
    :returns: The original quote or a shortened leading excerpt.
    :rtype: str
    :raises ValueError: If ``max_words`` is less than one.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least one")
    words = quote.split()
    return quote if len(words) <= max_words else " ".join(words[:max_words]) + "…"


def find_quote_source(quote: str, sources: List[Dict[str, Any]]) -> Optional[int]:
    """Find a source containing an exact or tightly near-verbatim quote.

    Short quotations require an exact normalized match. For longer quotations,
    a high similarity threshold permits minor OCR noise without accepting loose
    paraphrases as direct quotations.

    :param quote: Quoted text from the draft answer.
    :param sources: Uniform verification source dictionaries.
    :returns: Matching source number, or ``None`` when unsupported.
    :rtype: Optional[int]
    """
    normalized_quote = _normalize_verification_text(quote)
    for source in sources:
        evidence = _normalize_verification_text(str(source.get("quoteable_text") or ""))
        if normalized_quote and normalized_quote in evidence:
            return int(source["source_number"])
        if len(normalized_quote) < 40 or not evidence:
            continue
        quote_words = normalized_quote.split()
        evidence_words = evidence.split()
        window_size = len(quote_words)
        for start in range(0, max(1, len(evidence_words) - window_size + 1)):
            candidate = " ".join(evidence_words[start:start + window_size])
            if SequenceMatcher(None, normalized_quote, candidate).ratio() >= 0.93:
                return int(source["source_number"])
    return None


FLAG_MARKER_OPEN_TEMPLATE = "⟦flag:{flag_id}⟧"
FLAG_MARKER_CLOSE = "⟦/flag⟧"
FLAG_MARKER_REGEX = re.compile(r"⟦flag:\d+⟧|⟦/flag⟧")

_SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9א-ת])")

# Citation and honorific abbreviations whose period does not end a sentence.
# Without these, "Levin, p. 45" splits after "p.", so a flagged claim gets
# wrapped mid-citation.
_ABBREVIATION_REGEX = re.compile(
    r"\b(?:pp|p|vol|vols|no|nos|ed|eds|trans|cf|ch|chap|fig|figs|ff|n|esp|et al|e\.g|i\.e"
    r"|etc|st|mt|mr|mrs|ms|dr|prof|fol|fols|ms|mss|r|v)\.\s",
    flags=re.IGNORECASE,
)
_ABBREVIATION_SENTINEL = "\x00"


def split_sentences(paragraph: str) -> List[str]:
    """Split a paragraph into sentences with a conservative boundary rule.

    Abbreviation periods are masked before splitting so that citations such as
    "Levin, p. 45" stay intact; the returned sentences are verbatim substrings
    of the input, which callers rely on for locating and wrapping spans.

    :param paragraph: Single paragraph of answer text (no blank lines).
    :returns: Sentences in order; the original text is recoverable modulo
        inter-sentence whitespace.
    :rtype: List[str]
    """
    stripped = paragraph.strip()
    if not stripped:
        return []
    masked = _ABBREVIATION_REGEX.sub(
        lambda match: match.group(0).replace(".", _ABBREVIATION_SENTINEL),
        stripped,
    )
    return [
        sentence.replace(_ABBREVIATION_SENTINEL, ".")
        for sentence in _SENTENCE_BOUNDARY_REGEX.split(masked)
        if sentence.strip()
    ]


def locate_claim_sentence(claim_text: str, answer_text: str) -> Optional[str]:
    """Find the answer sentence that most plausibly expresses a verifier claim.

    Verifier claim texts are usually near-verbatim extractions, so a normalized
    substring test is tried first, then a similarity match over sentences. A
    conservative threshold avoids anchoring a claim to an unrelated sentence.

    :param claim_text: Claim text reported by the verifier or a quote body.
    :param answer_text: Current draft or final answer text.
    :returns: The matching sentence exactly as it appears, or ``None``.
    :rtype: Optional[str]
    """
    normalized_claim = _normalize_verification_text(claim_text)
    if not normalized_claim:
        return None
    best_sentence: Optional[str] = None
    best_ratio = 0.0
    for paragraph in answer_text.split("\n"):
        for sentence in split_sentences(paragraph):
            normalized_sentence = _normalize_verification_text(sentence)
            if not normalized_sentence:
                continue
            if normalized_claim in normalized_sentence or normalized_sentence in normalized_claim:
                return sentence
            ratio = SequenceMatcher(None, normalized_claim, normalized_sentence).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_sentence = sentence
    if best_ratio >= 0.55:
        return best_sentence
    return None


def extract_json_object(text: str, anchor_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Parse the first valid JSON object embedded in noisy model output.

    Reasoning-model output can surround (or contain) the JSON with thinking
    prose, including stray braces, so naive first-``{``/last-``}`` slicing is
    unsafe. This scans candidate ``{`` positions with ``raw_decode``, which
    handles braces inside JSON strings correctly.

    :param text: Raw model output expected to contain one JSON object.
    :param anchor_key: When given, only objects containing this top-level key
        qualify — skips decoy objects inside thinking prose.
    :returns: The parsed object, or ``None`` when no qualifying object exists.
    :rtype: Optional[Dict[str, Any]]
    """
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
    decoder = json.JSONDecoder()
    search_from = 0
    for _ in range(200):
        start = clean.find("{", search_from)
        if start == -1:
            return None
        try:
            candidate, _end = decoder.raw_decode(clean, start)
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(candidate, dict) and (anchor_key is None or anchor_key in candidate):
            return candidate
        search_from = start + 1
    return None


def strip_flag_markers(text: str) -> str:
    """Remove inline flag markers, leaving the visible answer text.

    :param text: Answer text possibly containing ⟦flag:N⟧ … ⟦/flag⟧ markers.
    :returns: Marker-free text.
    :rtype: str
    """
    return FLAG_MARKER_REGEX.sub("", text or "")


def annotate_answer_with_flags(
    answer: str,
    flags: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """Wrap flagged sentences in inline markers the chat UI can render.

    Each flag is anchored to at most one sentence; a sentence is wrapped at
    most once. Flags whose text can no longer be located (for example after a
    repair rewrote the sentence) are returned unanchored so the UI can still
    list them below the answer.

    :param answer: Final answer text (after shelf-mark linkification).
    :param flags: Soft-flag dictionaries with ``text``, ``type``, ``reason``.
    :returns: The annotated answer and flag dictionaries with ``flag_id`` and
        ``answer_span`` populated.
    :rtype: tuple[str, List[Dict[str, Any]]]
    """
    annotated = answer
    enriched: List[Dict[str, Any]] = []
    wrapped_spans: Set[str] = set()
    for flag_id, flag in enumerate(flags, start=1):
        entry = dict(flag)
        entry["flag_id"] = flag_id
        span = locate_claim_sentence(str(flag.get("text") or ""), strip_flag_markers(annotated))
        if span and span not in wrapped_spans and "\n" not in span:
            open_marker = FLAG_MARKER_OPEN_TEMPLATE.format(flag_id=flag_id)
            index = annotated.find(span)
            if index != -1:
                annotated = (
                    annotated[:index]
                    + open_marker
                    + span
                    + FLAG_MARKER_CLOSE
                    + annotated[index + len(span):]
                )
                wrapped_spans.add(span)
                entry["answer_span"] = span
            else:
                entry["answer_span"] = None
        else:
            entry["answer_span"] = None
        enriched.append(entry)
    return annotated, enriched


def remove_sentences_containing(answer: str, claim_texts: List[str]) -> str:
    """Deterministically delete the sentences expressing the given claims.

    Used when repair attempts are exhausted: rather than discarding the whole
    answer, only the sentences carrying still-rejected claims are removed.

    :param answer: Draft answer text.
    :param claim_texts: Claim texts whose sentences must be removed.
    :returns: The answer with offending sentences deleted.
    :rtype: str
    """
    result = answer
    for claim_text in claim_texts:
        sentence = locate_claim_sentence(claim_text, result)
        if sentence is None:
            continue
        result = result.replace(sentence, "", 1)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def filter_manuscript_shelfmarks(candidates: Any) -> List[str]:
    """Keep only candidates that are genuinely manuscript shelf marks.

    The index's ``shelf_marks_mentioned`` field is extracted upstream and
    contains publication references alongside real shelf marks — "DJD II,
    no. 20" (a Discoveries in the Judaean Desert item), "Babata's Ketubba",
    or a bare collection abbreviation. Surfacing those as manuscripts of this
    collection misleads readers and pollutes the missing-fragment worklist.

    :param candidates: Raw ``shelf_marks_mentioned`` value from the index.
    :returns: Candidates that parse as manuscript shelf marks, in order.
    :rtype: List[str]
    """
    if not isinstance(candidates, (list, tuple, set)):
        return []
    kept: List[str] = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and detect_shelfmarks(text) and text not in kept:
            kept.append(text)
    return kept


def shelfmarks_equivalent(first: str, second: str) -> bool:
    """Return whether two shelf marks normalize to the same canonical id.

    Used to gate document linking: a fuzzy search hit that is merely *near* a
    cited shelf mark (T-S H3.101 for T-S H3.111) must never be linked, since a
    wrong clickable citation is worse than no link.

    An institution prefix is the one tolerated difference: "Rylands Genizah
    Fragment 1" and "Manchester: Rylands Genizah Fragment 1" are the same
    fragment, so the shorter canonical may match as a whole-token suffix of
    the longer one. Numeric tails must always match exactly.

    :param first: First shelf mark string.
    :param second: Second shelf mark string.
    :returns: ``True`` when both canonicalize identically, or differ only by
        a leading institution prefix.
    :rtype: bool
    """
    canonical_first = ShelfmarkNormalizer.to_canonical_id(first or "").lower()
    canonical_second = ShelfmarkNormalizer.to_canonical_id(second or "").lower()
    if not canonical_first or not canonical_second:
        return False
    if canonical_first == canonical_second:
        return True
    shorter, longer = sorted((canonical_first, canonical_second), key=len)
    return len(shorter) >= 3 and longer.endswith("_" + shorter)


def find_shelfmark_source(shelf_mark: str, sources: List[Dict[str, Any]]) -> Optional[int]:
    """Find a source containing an equivalent detected shelf mark.

    :param shelf_mark: Shelf mark extracted from the draft answer.
    :param sources: Uniform verification source dictionaries.
    :returns: Matching source number, or ``None`` when unsupported.
    :rtype: Optional[int]
    """
    target = ShelfmarkNormalizer.to_canonical_id(shelf_mark).lower()
    for source in sources:
        evidence = str(source.get("evidence_text") or "")
        if target and target in evidence.lower():
            return int(source["source_number"])
        candidates = detect_shelfmarks(evidence)
        if any(ShelfmarkNormalizer.to_canonical_id(candidate).lower() == target for candidate in candidates):
            return int(source["source_number"])
    return None


# ============================================================================
# Conversation history and follow-up resolution
# ============================================================================

# Sections the finalizer appends below the prose answer ("---\n**Works cited:**"
# etc.). The frontend round-trips the whole answer as history, so prior turns
# must be cut back to their prose before reaching any prompt.
ANSWER_APPENDIX_REGEX = re.compile(r"\n\s*---\s*\n\s*\*\*")
MARKDOWN_LINK_REGEX = re.compile(r"\[([^\]]+)\]\((?:doc:)?[^)]+\)")
PARENTHETICAL_CITATION_REGEX = re.compile(
    r"\s*\((?:[^()]{1,80}?,\s*)?(?:p|pp)\.\s*\d+[^()]{0,20}\)"
)
# Deterministic signals that a message depends on earlier turns. Only cues; the
# resolver model makes the final call so that "Is it true that..." questions are
# not forced into a rewrite.
PRONOUN_CUE_REGEX = re.compile(
    r"\b(it|its|itself|he|she|him|his|her|hers|they|them|their|theirs)\b", re.IGNORECASE
)
BARE_DEMONSTRATIVE_REGEX = re.compile(r"\b(that|this|these|those)\s*(?=[?.,!]|$)", re.IGNORECASE)
DEMONSTRATIVE_PHRASE_REGEX = re.compile(r"\b(?:that|this|these|those)\s+([\w'’-]+)", re.IGNORECASE)
DEFINITE_PHRASE_REGEX = re.compile(r"\bthe\s+([\w'’-]+)", re.IGNORECASE)
CONTINUATION_OPENER_REGEX = re.compile(
    r"^\s*(and|but|so|also|what about|how about|more|any more|anything else)\b", re.IGNORECASE
)
RESOLUTION_FOLLOWUP_REGEX = re.compile(r"FOLLOW-?UP\s*:\s*(yes|no|true|false)", re.IGNORECASE)
RESOLUTION_QUESTION_REGEX = re.compile(r"QUESTION\s*:\s*(.+)", re.IGNORECASE)
# A resolved query longer than this is almost always leaked answer text.
MAX_RESOLVED_QUERY_CHARS = 220


def normalize_conversation_history(history: Any) -> List[Dict[str, str]]:
    """Coerce every accepted history shape into ``[{"role", "content"}]`` dicts.

    The API delivers pydantic ``ChatMessage`` objects, tests use plain dicts,
    and older callers pass ``{"user_query", "answer"}`` turn records. Consumers
    that checked ``isinstance(turn, dict)`` silently saw an empty conversation
    in production, so everything is normalized here once.

    :param history: Raw conversation history in any supported shape, or ``None``.
    :returns: Ordered user/assistant turns with non-empty content.
    :rtype: List[Dict[str, str]]
    """
    turns: List[Dict[str, str]] = []
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            user_query = turn.get("user_query")
            answer = turn.get("answer")
        else:
            role = getattr(turn, "role", None)
            content = getattr(turn, "content", None)
            user_query = getattr(turn, "user_query", None)
            answer = getattr(turn, "answer", None)

        if role is not None and content is not None:
            role_name = str(role).strip().lower()
            if role_name in ("user", "assistant") and str(content).strip():
                turns.append({"role": role_name, "content": str(content)})
            continue
        if user_query and str(user_query).strip():
            turns.append({"role": "user", "content": str(user_query)})
        if answer and str(answer).strip():
            turns.append({"role": "assistant", "content": str(answer)})
    return turns


def conversation_turns(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return the state's conversation history as normalized turns.

    :param state: Current RAG state (any accepted history shape).
    :returns: Normalized turns; empty when there is no history.
    :rtype: List[Dict[str, str]]
    """
    return normalize_conversation_history(state.get("conversation_history"))


def answer_prose(text: str) -> str:
    """Reduce a prior assistant answer to its prose for use in prompts.

    Drops the appended catalog/works-cited sections, flag markers, markdown
    link targets, and parenthetical page citations — none of which help a
    model resolve what "the verses" refers to, and all of which tempt a small
    model into copying citations into a search query.

    :param text: Full answer text as returned to the client.
    :returns: Whitespace-normalized prose.
    :rtype: str
    """
    prose = ANSWER_APPENDIX_REGEX.split(text or "", 1)[0]
    prose = strip_flag_markers(prose)
    prose = MARKDOWN_LINK_REGEX.sub(r"\1", prose)
    prose = PARENTHETICAL_CITATION_REGEX.sub("", prose)
    return " ".join(prose.split())


def render_conversation(
    turns: List[Dict[str, str]],
    max_turns: int,
    user_chars: int,
    assistant_chars: int,
) -> str:
    """Render recent turns as ``Role: text`` lines for a prompt.

    :param turns: Normalized conversation turns.
    :param max_turns: How many of the most recent turns to include.
    :param user_chars: Character cap per user turn.
    :param assistant_chars: Character cap per assistant turn (prose only).
    :returns: Rendered transcript, empty when there are no turns.
    :rtype: str
    """
    lines: List[str] = []
    for turn in turns[-max_turns:]:
        if turn["role"] == "assistant":
            lines.append(f"Assistant: {answer_prose(turn['content'])[:assistant_chars]}")
        else:
            lines.append(f"User: {' '.join(turn['content'].split())[:user_chars]}")
    return "\n".join(lines)


def conversation_grounding_text(state: Dict[str, Any]) -> str:
    """Concatenate everything anyone in the conversation actually said.

    Used to decide whether a shelf mark in a plan or rewrite is grounded: only
    marks the user typed or a previous answer printed (prose or appendix) count.

    :param state: Current RAG state.
    :returns: Current message, resolved query, and all prior turn text.
    :rtype: str
    """
    parts = [str(state.get("user_query") or ""), str(state.get("resolved_query") or "")]
    parts.extend(turn["content"] for turn in conversation_turns(state))
    return " ".join(parts)


def reference_cues(message: str, turns: List[Dict[str, str]]) -> List[str]:
    """Find phrases in a message that likely point back to earlier turns.

    Cues are pronouns, bare demonstratives ("who wrote that?"), continuation
    openers ("and what about..."), and definite phrases whose noun appeared in
    the conversation ("the verses" after an answer that discussed verses).
    Collection-context nouns ("the Genizah") never count.

    :param message: The latest user message.
    :param turns: Normalized prior turns.
    :returns: Cue phrases in order of detection; empty when none.
    :rtype: List[str]
    """
    if not turns:
        return []
    prior_text = " ".join(
        answer_prose(turn["content"]) if turn["role"] == "assistant" else turn["content"]
        for turn in turns
    ).lower()
    cues: List[str] = []
    for match in DEFINITE_PHRASE_REGEX.finditer(message):
        noun = match.group(1).lower().strip("'’")
        if len(noun) < 4 or noun in COLLECTION_CONTEXT_TERMS:
            continue
        if term_appears_in_text(noun, prior_text):
            cues.append(match.group(0))
    for match in DEMONSTRATIVE_PHRASE_REGEX.finditer(message):
        noun = match.group(1).lower().strip("'’")
        if len(noun) >= 4 and term_appears_in_text(noun, prior_text):
            cues.append(match.group(0))
    bare = BARE_DEMONSTRATIVE_REGEX.search(message)
    if bare:
        cues.append(bare.group(1))
    pronoun = PRONOUN_CUE_REGEX.search(message)
    if pronoun:
        cues.append(pronoun.group(1))
    opener = CONTINUATION_OPENER_REGEX.match(message)
    if opener:
        cues.append(opener.group(1))
    return cues


def parse_resolution_reply(reply: str, fallback_question: str) -> tuple[bool, str]:
    """Parse the resolver model's ``FOLLOWUP:`` / ``QUESTION:`` lines.

    A labeled two-line format is used instead of JSON because the small
    router model under LM Studio's schema mode emitted malformed objects
    (``{is_followup: true}`` followed by prose) on real inputs.

    :param reply: Raw model reply.
    :param fallback_question: Value to return when no question line is found.
    :returns: Whether the model judged the message a follow-up, and the
        standalone question it proposed.
    :rtype: tuple[bool, str]
    """
    followup_match = RESOLUTION_FOLLOWUP_REGEX.search(reply or "")
    question_match = RESOLUTION_QUESTION_REGEX.search(reply or "")
    if not followup_match:
        return False, fallback_question
    is_followup = followup_match.group(1).lower() in ("yes", "true")
    question = question_match.group(1).strip() if question_match else fallback_question
    return is_followup, question or fallback_question


def accept_resolved_query(
    original: str,
    rewrite: str,
    grounding_text: str,
) -> tuple[Optional[str], str]:
    """Decide whether a proposed standalone question may replace the message.

    Small models over-contextualize: they append the previous answer, copy
    citations, or invent specifics. Every guard here was hit by a real rewrite
    from the production router model, so a rejected rewrite falls back to the
    user's own words rather than searching for something they never asked.

    :param original: The user's literal message.
    :param rewrite: The model's proposed standalone question.
    :param grounding_text: Everything said in the conversation, including the
        current message.
    :returns: The cleaned rewrite when accepted (else ``None``), and a short
        reason suitable for logging.
    :rtype: tuple[Optional[str], str]
    """
    cleaned = answer_prose(rewrite)
    if not cleaned or cleaned.lower() == " ".join(original.split()).lower():
        return None, "rewrite unchanged"
    if len(cleaned) > MAX_RESOLVED_QUERY_CHARS:
        return None, f"rewrite too long ({len(cleaned)} chars)"
    if cleaned.count('"') >= 2 or "“" in cleaned or "”" in cleaned:
        return None, "rewrite contains quoted text"

    grounded_marks = {
        ShelfmarkNormalizer.to_canonical_id(mark).lower()
        for mark in detect_shelfmarks(grounding_text)
    }
    for mark in detect_shelfmarks(cleaned):
        if ShelfmarkNormalizer.to_canonical_id(mark).lower() not in grounded_marks:
            return None, f"rewrite introduces shelf mark {mark!r}"

    grounding_lower = grounding_text.lower()
    rewrite_terms = extract_topical_terms(cleaned)
    novel_terms = [term for term in rewrite_terms if not term_appears_in_text(term, grounding_lower)]
    if rewrite_terms and (len(novel_terms) > 3 or len(novel_terms) / len(rewrite_terms) > 0.25):
        return None, f"rewrite introduces terms {novel_terms}"

    original_terms = extract_topical_terms(original)
    lost_terms = [
        term for term in original_terms if not term_appears_in_text(term, cleaned.lower())
    ]
    if original_terms and len(lost_terms) / len(original_terms) > 0.5:
        return None, f"rewrite drops the user's terms {lost_terms}"
    return cleaned, "accepted"


def search_query(state: Dict[str, Any]) -> str:
    """Return the query retrieval should use: the resolved follow-up if any.

    :param state: Current RAG state.
    :returns: The standalone question, or the literal message when none.
    :rtype: str
    """
    return str(state.get("resolved_query") or state.get("user_query") or "")


# ============================================================================
# Agentic RAG Service
# ============================================================================

class AgenticRAGService:
    """LangGraph-based agentic RAG with scholarly synthesis."""

    def __init__(self):
        self.llm_studio_base_url = os.getenv("LLM_STUDIO_URL", "http://127.0.0.1:1234")
        self.router_model = os.getenv("ROUTER_MODEL", "qwen/qwen3-4b-2507")
        self.synthesis_model = os.getenv("SYNTHESIS_MODEL", "c4ai-command-r-v01")
        self.verification_model = os.getenv("VERIFICATION_MODEL", "qwen/qwen3-4b-2507")
        # Idle TTL (seconds) sent with every LM Studio request so JIT-loaded
        # models auto-unload when idle, bounding memory use. 0 disables it.
        self.model_ttl_seconds = int(os.getenv("LM_STUDIO_MODEL_TTL", "3600"))
        # Per-request timeout must cover a cold JIT model load, which can take
        # minutes for a large model; 120s was a recurring cause of failures.
        self.request_timeout_seconds = float(os.getenv("LM_STUDIO_REQUEST_TIMEOUT", "300"))
        # (timestamp, loaded model ids) — LM Studio's resident set changes
        # outside this app (training jobs, manual ejects, idle unloads).
        self._loaded_models_cache: tuple[float, List[str]] = (0.0, [])

        self.graph = self._build_graph()

    async def _loaded_model_ids(self, force_refresh: bool = False) -> List[str]:
        """List model ids LM Studio currently holds in memory.

        Cached briefly: this is consulted before every model call, but the set
        only changes when a model is loaded, evicted, or unloaded.

        :param force_refresh: Bypass the cache (used after a stale-state error).
        :returns: Loaded model identifiers, or an empty list if unknown.
        :rtype: List[str]
        """
        now = time.monotonic()
        cached_at, cached_ids = self._loaded_models_cache
        if not force_refresh and now - cached_at < 10.0:
            return cached_ids

        import httpx
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(f"{self.llm_studio_base_url}/api/v0/models")
                response.raise_for_status()
                ids = [
                    entry["id"]
                    for entry in response.json().get("data", [])
                    if entry.get("id") and entry.get("state") == "loaded"
                ]
        except Exception as exc:
            logger.warning("Could not read LM Studio loaded-model list: %s", exc)
            return cached_ids
        self._loaded_models_cache = (now, ids)
        return ids

    async def resolve_model(self, configured_model: str, force_refresh: bool = False) -> str:
        """Map a configured model id onto one LM Studio can actually serve.

        This machine also runs training jobs, so models get evicted, reloaded,
        and duplicated as numbered instances ("model:2") outside this app's
        control. Requesting a configured id blindly fails with "Model has not
        started loading/has been unloaded" even when the same weights are
        resident under an instance id.

        Resolution order: the exact id, then another instance of the same
        model, then explicit JIT loading of the configured model. It never
        substitutes a different model because doing so makes behavior
        unpredictable and can expose models not approved for that role.

        :param configured_model: Model id from configuration or a request.
        :param force_refresh: Re-read LM Studio state before resolving.
        :returns: A model id believed to be servable right now.
        :rtype: str
        :raises ModelUnavailableError: If the configured model cannot be loaded.
        """
        loaded = await self._loaded_model_ids(force_refresh=force_refresh)
        if configured_model in loaded:
            return configured_model

        base = configured_model.split(":")[0]
        instances = [model for model in loaded if model.split(":")[0] == base]
        if instances:
            logger.info(
                "Model %r is not loaded; using resident instance %r",
                configured_model, instances[0],
            )
            return instances[0]

        try:
            await self.load_model(configured_model)
            self._loaded_models_cache = (0.0, [])
            return configured_model
        except Exception as exc:
            logger.warning(
                "JIT load of configured model %r failed (%s); refusing cross-model fallback",
                configured_model, exc,
            )
            raise ModelUnavailableError(LOCAL_MODEL_CAPACITY_MESSAGE) from exc

    async def is_model_allowed(self, model_id: str) -> bool:
        """Check whether a model id is one LM Studio actually has downloaded.

        Used to bound any externally-supplied model string to the set of known
        local models before it is forwarded to LM Studio, preventing arbitrary
        strings from being pushed at the local inference server.

        :param model_id: The model identifier to validate.
        :returns: True if the model is present in LM Studio, else False.
        :rtype: bool
        """
        try:
            available = await self.list_available_models()
        except Exception as e:
            logger.error(f"Could not verify model allowlist: {e}")
            return False
        return model_id in set(available.get("models", []))

    @staticmethod
    def _timed_node(
        name: str,
        node: Callable[[AgenticRAGState], Any],
    ) -> Callable[[AgenticRAGState], Any]:
        """Wrap a graph node to accumulate its wall time and call count in state.

        Also marks the node as the current stage so LLM calls made inside it
        are attributed to it in the request metrics.

        :param name: Graph node name.
        :param node: The node coroutine function.
        :returns: A coroutine function with the same signature.
        :rtype: Callable[[AgenticRAGState], Any]
        """
        async def timed(state: AgenticRAGState) -> AgenticRAGState:
            stage_token = _CURRENT_STAGE.set(name)
            started = time.monotonic()
            try:
                return await node(state)
            finally:
                _CURRENT_STAGE.reset(stage_token)
                elapsed = time.monotonic() - started
                timings = state.setdefault("stage_timings", {})
                timings[name] = round(timings.get(name, 0.0) + elapsed, 3)
                calls = state.setdefault("stage_calls", {})
                calls[name] = calls.get(name, 0) + 1

        timed.__name__ = getattr(node, "__name__", name)
        return timed

    def _build_metrics(
        self,
        state: Dict[str, Any],
        started_at: float,
        request_metrics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the metrics block reported on a response.

        :param state: Final pipeline state.
        :param started_at: ``time.monotonic()`` when the request began.
        :param request_metrics: Sink populated by ``record_llm_call`` during the run.
        :returns: Metrics dictionary (see ``AgenticRAGResponse.metrics``).
        :rtype: Dict[str, Any]
        """
        stage_calls = dict(state.get("stage_calls") or {})
        llm_calls = list((request_metrics or {}).get("llm_calls") or [])
        by_stage: Dict[str, Dict[str, Any]] = {}
        for call in llm_calls:
            bucket = by_stage.setdefault(call["stage"] or "unattributed", {
                "calls": 0, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_tokens": 0,
            })
            bucket["calls"] += 1
            bucket["seconds"] = round(bucket["seconds"] + (call.get("seconds") or 0.0), 3)
            bucket["prompt_tokens"] += call.get("prompt_tokens") or 0
            bucket["completion_tokens"] += call.get("completion_tokens") or 0
            bucket["reasoning_tokens"] += call.get("reasoning_tokens") or 0
        return {
            "total_seconds": round(time.monotonic() - started_at, 3),
            "synthesis_model": state.get("synthesis_model_override") or self.synthesis_model,
            "router_model": self.router_model,
            "verification_model": self.verification_model,
            "stage_timings": dict(state.get("stage_timings") or {}),
            "stage_calls": stage_calls,
            "verification_cycles": stage_calls.get("verify_claims", 0),
            "repair_attempts": stage_calls.get("repair_answer", 0),
            "retry_count": state.get("retry_count", 0),
            "llm_calls": llm_calls,
            "llm_by_stage": by_stage,
        }

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgenticRAGState)

        for name, node in [
            ("resolve_query", self._resolve_query_node),
            ("route_query", self._route_query_node),
            ("execute_searches", self._execute_searches_node),
            ("link_primary_secondary", self._link_primary_secondary_node),
            ("synthesize_answer", self._synthesize_answer_node),
            ("repair_answer", self._repair_answer_node),
            ("verify_claims", self._verify_claims_node),
            ("finalize_response", self._finalize_response_node),
        ]:
            workflow.add_node(name, self._timed_node(name, node))

        workflow.set_entry_point("resolve_query")
        workflow.add_edge("resolve_query", "route_query")
        workflow.add_edge("route_query", "execute_searches")
        workflow.add_edge("execute_searches", "link_primary_secondary")
        workflow.add_edge("link_primary_secondary", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "verify_claims")

        def _route_after_verify(state: AgenticRAGState) -> str:
            error_type = state.get("error_type")
            if error_type == "FABRICATED_CLAIMS":
                if should_retry_verification(state.get("retry_count", 0)):
                    return "retry"
                return "abort"
            return "continue"

        workflow.add_conditional_edges(
            "verify_claims",
            _route_after_verify,
            {
                "retry": "repair_answer",
                "continue": "finalize_response",
                "abort": "finalize_response"
            }
        )

        workflow.add_edge("repair_answer", "verify_claims")
        workflow.add_edge("finalize_response", END)

        return workflow.compile()

    @weave.op()
    async def _call_llm_with_tools(
            self,
            messages: List[Dict[str, str]],
            tools: List[Dict[str, Any]],
            model: str,
            tool_choice: Optional[Dict[str, Any] | str] = None
    ) -> Dict[str, Any]:
        """Call LLM Studio with function calling"""
        url = f"{self.llm_studio_base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.0,
            "max_tokens": 2048
        }
        if self.model_ttl_seconds > 0:
            payload["ttl"] = self.model_ttl_seconds

        if tool_choice:
            payload["tool_choice"] = tool_choice

        import httpx
        async with lm_studio_gateway.slot(f"tools:{model}"):
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                for attempt in range(2):
                    payload["model"] = await self.resolve_model(model, force_refresh=attempt > 0)
                    started = time.monotonic()
                    try:
                        response = await client.post(url, json=payload)
                    except httpx.TimeoutException as exc:
                        # A timeout against the local server means it is
                        # saturated (e.g. competing training workload), which
                        # is a capacity condition, not a bug.
                        raise ModelUnavailableError(LOCAL_MODEL_CAPACITY_MESSAGE) from exc
                    if not response.is_error:
                        result = response.json()
                        record_llm_call(payload["model"], result, time.monotonic() - started, with_tools=True)
                        return result
                    if attempt == 0 and _is_model_unavailable_error(response):
                        logger.warning(
                            "LM Studio reports %r unavailable; re-resolving and retrying",
                            payload["model"],
                        )
                        continue
                    logger.error(
                        "LM Studio returned %s for model '%s': %s",
                        response.status_code, payload["model"], response.text
                    )
                    if response.status_code >= 500 or _is_model_unavailable_error(response):
                        raise ModelUnavailableError(LOCAL_MODEL_CAPACITY_MESSAGE)
                    response.raise_for_status()
                raise RuntimeError("LM Studio could not serve the request")

    @weave.op()
    async def _call_llm(
            self,
            messages: List[Dict[str, str]],
            model: str,
            temperature: float = 0.7,
            response_format: Optional[Dict[str, Any]] = None,
            recover_reasoning: bool = False,
            max_tokens: int = 8192,
    ) -> str:
        """Call LM Studio without tools, auto-loading the model if not yet loaded.

        :param messages: Chat messages to send.
        :param model: LM Studio model identifier.
        :param temperature: Sampling temperature.
        :param response_format: Optional OpenAI-style ``response_format`` payload
            (e.g. a ``json_schema`` spec) enforcing structured output.
        :param recover_reasoning: Return the reasoning channel when content is
            empty. Safe only for JSON calls whose parser isolates the object;
            for prose calls it would leak chain-of-thought into the answer.
        :returns: The model's response content string.
        :rtype: str
        """
        url = f"{self.llm_studio_base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # Reasoning models spend output budget thinking before the final
            # answer; a tight cap can exhaust it mid-thought and yield empty
            # content.
            "max_tokens": max_tokens
        }
        if self.model_ttl_seconds > 0:
            payload["ttl"] = self.model_ttl_seconds
        if response_format is not None:
            payload["response_format"] = response_format

        import httpx
        async with lm_studio_gateway.slot(f"chat:{model}"):
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                result = None
                for attempt in range(2):
                    payload["model"] = await self.resolve_model(model, force_refresh=attempt > 0)
                    started = time.monotonic()
                    try:
                        response = await client.post(url, json=payload)
                    except httpx.TimeoutException as exc:
                        # Same capacity semantics as the tools variant above.
                        raise ModelUnavailableError(LOCAL_MODEL_CAPACITY_MESSAGE) from exc
                    if not response.is_error:
                        result = response.json()
                        record_llm_call(payload["model"], result, time.monotonic() - started)
                        break
                    if attempt == 0 and _is_model_unavailable_error(response):
                        logger.warning(
                            "LM Studio reports %r unavailable; re-resolving and retrying",
                            payload["model"],
                        )
                        continue
                    logger.error(
                        "LM Studio returned %s for model '%s': %s",
                        response.status_code, payload["model"], response.text
                    )
                    if response.status_code >= 500 or _is_model_unavailable_error(response):
                        raise ModelUnavailableError(LOCAL_MODEL_CAPACITY_MESSAGE)
                    response.raise_for_status()
                if result is None:
                    raise RuntimeError("LM Studio could not serve the request")

        message = result["choices"][0]["message"]
        content = str(message.get("content") or "")
        if not content.strip():
            # LM Studio quirk: with reasoning models under response_format,
            # the schema-constrained output can land entirely in the
            # reasoning channel while content comes back empty.
            fallback = str(message.get("reasoning_content") or message.get("reasoning") or "")
            if recover_reasoning and fallback.strip():
                logger.warning(
                    "LM Studio returned empty content for '%s'; recovering output "
                    "from the reasoning channel", model,
                )
                content = fallback
            elif fallback.strip():
                logger.error(
                    "LM Studio returned empty content for '%s' with %s reasoning chars; "
                    "not recovering for a prose call", model, len(fallback),
                )
        return content

    async def load_model(self, model_id: str) -> None:
        """Warm-load a model in LM Studio via Just-In-Time loading.

        LM Studio's HTTP API has no explicit load endpoint; instead, sending an
        inference request for an unloaded model triggers JIT loading (requires
        "Just-In-Time Model Loading" to be enabled in LM Studio server settings).
        This issues a minimal completion so the model is resident before the user
        sends their real query, and passes ``ttl`` so it auto-unloads when idle.

        Enable "Only keep last JIT loaded model" in LM Studio to enforce a single
        resident model and bound memory use.

        The caller is responsible for validating ``model_id`` against
        :meth:`is_model_allowed` first; this method does not accept arbitrary
        strings from untrusted callers without that check.

        :param model_id: The LM Studio model identifier to warm-load.
        :raises ValueError: If the model is not in LM Studio's downloaded set.
        :raises httpx.HTTPStatusError: If LM Studio returns an error response.
        """
        if not await self.is_model_allowed(model_id):
            raise ValueError(f"Unknown model: {model_id!r}")

        import httpx
        url = f"{self.llm_studio_base_url}/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        if self.model_ttl_seconds > 0:
            payload["ttl"] = self.model_ttl_seconds

        logger.info(f"Warm-loading model in LM Studio: {model_id}")
        # Generous timeout: a cold large model can take minutes to load.
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            if response.is_error:
                logger.error(f"LM Studio load failed ({response.status_code}): {response.text}")
            response.raise_for_status()
        logger.info(f"Model loaded: {model_id}")

    _RESOLVE_QUERY_SYSTEM_PROMPT = """You prepare the latest user message in a scholarly chat about the Cairo Genizah for a search engine.

First decide whether the latest message is a FOLLOW-UP: it is a follow-up if a reader who had NOT seen the earlier conversation could not tell what it is asking about — because it uses "it", "that", "he", or a phrase like "the verses", "the poem", "the article", "the shelf mark" that points to something already discussed. A message that names its own topic (e.g. "Tell me about ketubbot", "Show me Purim fragments") is NOT a follow-up even if the topic is related.

If it is a follow-up, restate the latest message as ONE self-contained question of at most 25 words: keep the user's request itself (samples, date, author, etc.) and replace each reference with the specific thing it points to — a name, title, term, or shelf mark copied exactly from the conversation. Do not add other details from the conversation, and never add names, shelf marks, dates, page numbers, or facts that are not in it. Never answer the question. If it is not a follow-up, repeat the latest message unchanged.

Reply with exactly two lines and nothing else:
FOLLOWUP: yes or no
QUESTION: the standalone question (when FOLLOWUP is yes it must name the subject explicitly, not repeat the message)"""

    _REWRITE_QUERY_SYSTEM_PROMPT = """You rewrite the latest user message in a scholarly chat about the Cairo Genizah into ONE self-contained search question of at most 25 words.

Rules:
- Resolve pronouns and references ("it", "that fragment", "the verses", "he", "the article") using ONLY the conversation.
- Replace each reference with the specific name, term, title, or shelf mark it points to, copied exactly from the conversation. Do not add other details from the conversation.
- Never introduce a name, shelf mark, date, page number, or fact that does not appear in the conversation.
- Preserve the user's actual request (samples, date, author, shelf mark, comparison...). Never answer it.
- Output the rewritten question only: one line, at most 25 words, no quotes, no label, no explanation."""

    @weave.op()
    async def _resolve_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: restate a follow-up as a standalone question before routing.

        A follow-up such as "what do the verses actually say?" carries none of
        the topic; searched literally it retrieves whatever mentions verses.
        The message is classified (with deterministic reference cues as a
        hint) and, when it depends on earlier turns, rewritten by the router
        model into a self-contained question that every later stage searches
        and judges relevance against. Rewrites are accepted only when they
        pass the grounding guards, so a bad rewrite degrades to the user's own
        wording rather than a hallucinated topic.

        :param state: Current LangGraph RAG state.
        :returns: State with ``conversation_history`` normalized and
            ``resolved_query`` / ``is_followup`` populated.
        :rtype: AgenticRAGState
        """
        turns = conversation_turns(state)
        state["conversation_history"] = turns
        message = str(state["user_query"])
        state["resolved_query"] = message
        state["is_followup"] = False
        if not turns:
            return state

        cues = reference_cues(message, turns)
        transcript = render_conversation(turns, max_turns=6, user_chars=400, assistant_chars=1200)
        hint = ""
        if cues:
            hint = (
                "\n\nNOTE: "
                + ", ".join(repr(cue) for cue in cues[:3])
                + " in the latest message may refer back to something in the conversation; "
                "if so, this is a follow-up."
            )
        user_message = (
            f"CONVERSATION SO FAR:\n{transcript}\n\nLATEST USER MESSAGE:\n{message}{hint}"
        )

        reply = await self._call_resolver(self._RESOLVE_QUERY_SYSTEM_PROMPT, user_message)
        if reply is None:
            state["processing_steps"].append(
                "Could not resolve follow-up context; searching the message as written"
            )
            return state

        is_followup, question = parse_resolution_reply(reply, message)
        if not is_followup:
            if cues:
                state["processing_steps"].append("Message judged self-contained despite referring phrases")
            return state

        state["is_followup"] = True
        grounding = conversation_grounding_text(state)
        accepted, reason = accept_resolved_query(message, question, grounding)
        if accepted is None:
            # The model agreed it is a follow-up but did not (usably) rewrite
            # it; a dedicated rewrite pass succeeds where the combined
            # classify-and-rewrite reply repeated the message.
            logger.info("Follow-up rewrite rejected (%s); running dedicated rewrite pass", reason)
            rewrite_reply = await self._call_resolver(self._REWRITE_QUERY_SYSTEM_PROMPT, user_message) or ""
            candidate = next(
                (line for line in rewrite_reply.splitlines() if line.strip()), ""
            )
            candidate = re.sub(
                r"^\s*(?:self-contained |standalone |rewritten )?question\s*:\s*",
                "", candidate, flags=re.IGNORECASE,
            )
            accepted, reason = accept_resolved_query(message, candidate, grounding)

        if accepted is not None:
            state["resolved_query"] = accepted
            state["processing_steps"].append(f"Interpreted follow-up as: {accepted}")
            logger.info("Resolved follow-up %r -> %r", message, accepted)
        else:
            state["processing_steps"].append(f"Follow-up kept as written ({reason})")
            logger.info("Follow-up %r kept as written: %s", message, reason)
        return state

    async def _call_resolver(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Run one follow-up-resolution call on the router model.

        The resolver is an enhancement: on any failure other than model
        capacity the pipeline continues on the literal message, so errors are
        logged and reported as ``None`` rather than raised.

        :param system_prompt: Resolver or rewrite instructions.
        :param user_message: Rendered transcript plus the latest message.
        :returns: The model reply, or ``None`` when the call failed.
        :rtype: Optional[str]
        :raises ModelUnavailableError: When the local model cannot serve requests.
        """
        try:
            return await self._call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                model=self.router_model,
                temperature=0.0,
                recover_reasoning=True,
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Follow-up resolution call failed; using the literal message: %s", exc)
            return None

    @weave.op()
    async def _route_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Route a query using the LLM planner without deterministic search overrides.

        Neo4j entity resolution happens only when the planner selects a graph
        action. This keeps topic and collection queries in the bibliography or
        primary-source retrieval paths chosen by the planner.

        :param state: Current LangGraph RAG state.
        :returns: State populated with a validated or fallback query plan.
        :rtype: AgenticRAGState
        """
        logger.info(f"Routing query: {state['user_query']}")
        state["resolved_entities"] = []

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

**NEO4J GRAPH — USE SELECTIVELY FOR NAMED SCHOLARS AND RELATIONSHIPS:**

Neo4j contains Scholar, Person, BookArticle, Fragment, Place, and Institution relationships.
Use `graph_scholar` when the query names a specific human scholar/author and asks about
that person's work, publications, collaborators, institutions, studied fragments, or
referenced manuscripts. Supply only the person's name as the graph query. The graph action
will resolve the identity and automatically perform author-constrained bibliography retrieval.

Do NOT use Neo4j merely because a query contains "Cairo Genizah", "Genizah", a document
type, a historical topic, a religious concept, or another collection/topic phrase. Those
are not scholar names. In particular, "Cairo Genizah" is the collection/context, not an
author. Topic questions belong in semantic, keyword, or hybrid retrieval.

**NAMED SCHOLAR QUERIES — CRITICAL RULE:**

When the query is about a specific named scholar (e.g. "Tell me about Estara Arrant",
"What has Goitein written", "Friedman's work on marriage"), use `graph_scholar`. Its executor
combines Neo4j evidence with author-constrained bibliography retrieval, so do not duplicate
that work with a name-only bibliography action. You may add a topical bibliography search
only when the user separately asks about a subject that may include discussion by others.

**CHOOSING AMONG BIBLIOGRAPHY SEARCH MODES:**

- `bibliography_semantic`: conceptual or synonym-rich questions where exact wording may vary.
- `bibliography_hybrid`: the default for topics; combines concepts with important terms.
- Keyword-heavy `bibliography_hybrid`: exact terminology, titles, names mentioned in prose,
  transliterations, or distinctive phrases.
- `graph_scholar`: only a specific human scholar/author or an explicit scholar relationship.

**Query Type → Strategy:**

"Tell me about ketubbot in the Genizah"
→ `bibliography_hybrid` (keyword_weight: 70) ONLY
→ Reasoning: ketubbot are the topic and the Genizah is the collection context; neither is a scholar

"What ketubah fragments are in the Cairo Genizah?"
→ `primary_hybrid` query="ketubah" + `bibliography_hybrid` query="ketubah Cairo Genizah"
→ Reasoning: the user requests manuscripts, plus scholarship that contextualizes them; never graph_scholar

"Tell me about Estara Arrant" / "What has Goitein written" / "Friedman's work"
→ `graph_scholar` query="[name only]"
→ Reasoning: a specific human scholar is the subject; graph execution also retrieves their indexed scholarship

"Show me Purim fragments"
→ `primary_keyword: "Purim"` + `bibliography_hybrid: "Purim Genizah"`
→ Reasoning: user explicitly wants to see manuscripts

A message that is itself a shelf mark (the pattern <collection> <number>, e.g. one the user
typed or one quoted from earlier in this conversation)
→ `primary_shelfmark` + `bibliography_hybrid`, both using THAT EXACT shelf mark
→ Reasoning: specific shelf mark lookup

**NEVER INVENT A SHELF MARK.** Only use a shelf mark that appears verbatim in the user's
message or in the conversation history above. Shelf marks written in these instructions are
formatting illustrations, not real citations — never copy one into a search. If a follow-up
question asks about "that fragment" or "the shelf mark" and no shelf mark appears in the
conversation, do NOT guess one: search the bibliography for the topic under discussion
instead, using distinctive terms from the previous exchange.

"What do we know about Yom Kippur liturgy"
→ `bibliography_hybrid` (keyword_weight: 70) ONLY
→ Reasoning: broad topical query

**HEBREW-LANGUAGE SCHOLARSHIP — ADD A HEBREW SEARCH VARIANT:**

A substantial share of the indexed scholarship is written in Hebrew (e.g. Gil's במלכות
ישמעאל on Jews under medieval Islam, Levin's גנזי קדם on Geonic material). English queries
cannot reach Hebrew page text lexically. For topical queries (history, communities, trade,
liturgy, law — NOT named-scholar or shelf-mark lookups), ADD one extra `bibliography_hybrid`
action whose query is a concise Hebrew rendering of the distinctive terms (2-5 words,
keyword_weight 70). Examples:
- "Jewish communities under Fatimid rule" → add query "הקהילה היהודית בתקופה הפאטימית"
- "Geonic responsa about trade" → add query "תשובות הגאונים מסחר"
Translate only the topical content words; do not translate scholar names or shelf marks.

**Critical Rules:**
- Default to 1-2 bibliography searches (plus the Hebrew variant when applicable)
- Named human scholars/authors → graph_scholar
- Collections, document types, and research topics → never graph_scholar
- Let the requested information determine semantic, keyword-heavy, or hybrid retrieval
- Only add primary searches when user explicitly asks to see manuscripts
- When in doubt, bibliography only

**Available Search Types:**
- `bibliography_semantic`: Broad conceptual queries only
- `bibliography_hybrid`: Most queries (set keyword/semantic weights appropriately)
- `primary_shelfmark`: Specific shelf mark lookup
- `primary_keyword`: Keyword search in manuscripts (rare)
- `primary_hybrid`: Balanced manuscript search (rare)
- `graph_scholar`: Named scholar identity, works, relationships, and fragment connections"""

        history_context = ""
        turns = conversation_turns(state)
        if turns:
            history_context = "\n\n**Conversation History:**\n" + render_conversation(
                turns, max_turns=6, user_chars=400, assistant_chars=500
            ) + "\n"

        # Route on the standalone question. A follow-up's literal wording
        # ("what do the verses say?") names no topic; the resolved form does.
        routed_query = search_query(state)
        router_input = routed_query
        if state.get("is_followup") and routed_query != state["user_query"]:
            router_input += (
                f'\n\n(The user\'s literal message was: "{state["user_query"]}" — a follow-up '
                "to the conversation above, restated as a standalone question.)"
            )

        messages = [
            {"role": "system", "content": system_prompt + history_context},
            {"role": "user", "content": router_input}
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
                                            "primary_shelfmark",
                                            "graph_scholar"
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

        query_plan: Optional[QueryPlan] = None
        for attempt in range(2):
            response = await self._call_llm_with_tools(
                messages=messages,
                tools=tools,
                model=self.router_model,
                tool_choice="required"
            )
            query_plan = self._parse_router_plan(response, state)
            if query_plan is not None:
                break
            # The local server sometimes returns the plan as prose (parsed
            # above) or nothing usable at all; at temperature 0 an identical
            # retry would repeat that, so the retry asks for the tool call
            # explicitly instead of falling straight back to a blind search.
            logger.warning("Router produced no usable plan (attempt %d)", attempt + 1)
            messages = [
                messages[0],
                {
                    "role": "user",
                    "content": router_input + (
                        "\n\nRespond ONLY by calling the create_search_plan tool with a "
                        "complete plan for this question."
                    ),
                },
            ]

        if query_plan is None:
            logger.warning("No tool call, using bibliography-only fallback")
            query_plan = QueryPlan(
                actions=[
                    SearchAction(
                        search_type="bibliography_hybrid",
                        query=routed_query,
                        keyword_weight=70,
                        semantic_weight=30,
                        num_results=5
                    )
                ],
                needs_primary_secondary_linking=True,
                is_followup=bool(state.get("is_followup")),
                reasoning="Fallback: bibliography-only search (router produced no tool call)"
            )

        query_plan = self._drop_ungrounded_shelfmark_actions(query_plan, state)
        query_plan = self._ensure_bibliography_action(query_plan, state)
        query_plan = await self._add_hebrew_search_variant(query_plan, state)
        state["query_plan"] = query_plan
        state["processing_steps"].append(f"Search plan: {query_plan.reasoning}")

        return state

    @staticmethod
    def _parse_router_plan(response: Dict[str, Any], state: AgenticRAGState) -> Optional[QueryPlan]:
        """Extract a validated plan from a router response, if it holds one.

        Accepts the ``create_search_plan`` tool call, a direct search-type
        tool call, or — when the server returned no tool call — a plan object
        embedded in the assistant text, which is how LM Studio surfaces a
        tool call it failed to parse.

        :param response: Raw chat-completion response.
        :param state: Current RAG state (for follow-up flags).
        :returns: The plan, or ``None`` when nothing usable was returned.
        :rtype: Optional[QueryPlan]
        """
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            logger.warning("Router response had no message")
            return None

        for tool_call in message.get("tool_calls") or []:
            try:
                function_name = tool_call["function"].get("name")
                arguments = json.loads(tool_call["function"]["arguments"])
                if function_name == "create_search_plan":
                    return QueryPlan(**arguments)
                if function_name in DIRECT_SEARCH_ACTION_TYPES:
                    direct_arguments = dict(arguments)
                    direct_arguments["search_type"] = function_name
                    return QueryPlan(
                        actions=[SearchAction(**direct_arguments)],
                        needs_primary_secondary_linking=True,
                        is_followup=bool(state.get("is_followup")),
                        reasoning=(
                            f"Normalized the planner's direct {function_name} action "
                            "into a complete query plan"
                        ),
                    )
                logger.warning("Router returned unexpected tool function: %s", function_name)
            except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                logger.warning("Router returned an invalid search plan: %s", exc)

        content = str(message.get("content") or "")
        salvaged = extract_json_object(content, anchor_key="actions") if content else None
        if salvaged:
            try:
                plan = QueryPlan(**salvaged)
                logger.info("Salvaged search plan from router prose output")
                return plan
            except (TypeError, ValidationError) as exc:
                logger.warning("Plan-like JSON in router prose was invalid: %s", exc)
        return None

    @staticmethod
    def _ensure_bibliography_action(query_plan: QueryPlan, state: AgenticRAGState) -> QueryPlan:
        """Guarantee every plan retrieves scholarship, not just manuscripts.

        A shelf-mark-only plan (the router's habit for "what do the verses
        say?" once a shelf mark is in play) finds the catalog entry and no
        text, then trips the no-scholarship gate. The scholarship on that
        manuscript is what the answer needs, so a topical bibliography search
        on the standalone question is added.

        :param query_plan: Plan after grounding checks.
        :param state: Current RAG state.
        :returns: The plan with a bibliography search appended when it had none.
        :rtype: QueryPlan
        """
        if any(
            action.search_type.startswith("bibliography") or action.search_type == "graph_scholar"
            for action in query_plan.actions
        ):
            return query_plan
        state["processing_steps"].append(
            "Plan searched only manuscripts; added a bibliography search for the scholarship on them"
        )
        return QueryPlan(
            actions=list(query_plan.actions) + [SearchAction(
                search_type="bibliography_hybrid",
                query=search_query(state),
                keyword_weight=70,
                semantic_weight=30,
                num_results=5,
            )],
            needs_primary_secondary_linking=query_plan.needs_primary_secondary_linking,
            is_followup=query_plan.is_followup,
            reasoning=query_plan.reasoning,
        )

    async def _add_hebrew_search_variant(
        self,
        query_plan: QueryPlan,
        state: AgenticRAGState,
    ) -> QueryPlan:
        """Guarantee topical English queries also search in Hebrew.

        523 bibliography pages (including two-thirds of Gil's במלכות ישמעאל)
        are Hebrew-dominant and lexically unreachable from English wording.
        Planner prompting alone proved unreliable for this, so the variant is
        added deterministically: one short translation call, one extra search.

        :param query_plan: Validated plan from the router.
        :param state: Current LangGraph RAG state.
        :returns: The plan, with a Hebrew bibliography search appended when
            applicable; unchanged on any failure.
        :rtype: QueryPlan
        """
        has_bibliography_action = any(
            action.search_type.startswith("bibliography") for action in query_plan.actions
        )
        topical_query = search_query(state)
        already_hebrew = hebrew_char_ratio(topical_query) > 0.2 or any(
            hebrew_char_ratio(action.query) > 0.2 for action in query_plan.actions
        )
        only_scholar_lookup = all(
            action.search_type == "graph_scholar" for action in query_plan.actions
        )
        if not has_bibliography_action or already_hebrew or only_scholar_lookup:
            return query_plan

        try:
            translation = await self._call_llm(
                messages=[{
                    "role": "user",
                    "content": (
                        "Translate the topical content of this English search query into a "
                        "concise Hebrew search phrase (2-5 words). Return ONLY the Hebrew "
                        "phrase — no explanation, no transliteration. Keep modern scholars' "
                        "names and manuscript shelf marks untranslated (or omit them).\n\n"
                        f"Query: {topical_query}"
                    ),
                }],
                model=self.router_model,
                temperature=0.0,
                recover_reasoning=True,
            )
        except Exception as exc:
            logger.warning("Hebrew search-variant translation failed: %s", exc)
            return query_plan

        hebrew_query = translation.strip().strip('"״.').splitlines()[-1].strip()
        if not hebrew_query or len(hebrew_query) > 60 or hebrew_char_ratio(hebrew_query) < 0.5:
            logger.info("Discarded unusable Hebrew variant %r", translation[:60])
            return query_plan

        state["processing_steps"].append(f"Added Hebrew search variant: {hebrew_query}")
        return QueryPlan(
            actions=list(query_plan.actions) + [SearchAction(
                search_type="bibliography_hybrid",
                query=hebrew_query,
                keyword_weight=70,
                semantic_weight=30,
                num_results=6,
            )],
            needs_primary_secondary_linking=query_plan.needs_primary_secondary_linking,
            is_followup=query_plan.is_followup,
            reasoning=query_plan.reasoning,
        )

    @staticmethod
    def _drop_ungrounded_shelfmark_actions(
        query_plan: QueryPlan,
        state: AgenticRAGState,
    ) -> QueryPlan:
        """Remove shelf-mark searches for marks nobody in the conversation cited.

        Planners can emit a plausible-looking shelf mark copied from their own
        instructions or invented outright; searching it wastes the turn and, in
        a follow-up, silently answers about the wrong fragment. Only shelf marks
        the user or a previous answer actually mentioned may be searched.

        :param query_plan: Plan returned by the router.
        :param state: Current RAG state, used for the conversation transcript.
        :returns: The plan with ungrounded shelf-mark actions removed or, if
            that empties the plan, replaced by a topical bibliography search.
        :rtype: QueryPlan
        """
        grounded = {
            ShelfmarkNormalizer.to_canonical_id(mark).lower()
            for mark in detect_shelfmarks(conversation_grounding_text(state))
        }

        kept: List[SearchAction] = []
        dropped: List[str] = []
        for action in query_plan.actions:
            if action.search_type == "primary_shelfmark":
                canonical = ShelfmarkNormalizer.to_canonical_id(action.query).lower()
                if canonical not in grounded:
                    dropped.append(action.query)
                    continue
            kept.append(action)

        if not dropped:
            return query_plan

        logger.warning(
            "Dropped ungrounded shelf-mark search(es) %s: not mentioned by the user "
            "or any previous answer",
            dropped,
        )
        state["processing_steps"].append(
            f"Ignored invented shelf mark(s) {', '.join(dropped)}; they appear nowhere in "
            "the conversation."
        )
        if not kept:
            kept = [SearchAction(
                search_type="bibliography_hybrid",
                query=search_query(state),
                keyword_weight=70,
                semantic_weight=30,
                num_results=5,
            )]
        return QueryPlan(
            actions=kept,
            needs_primary_secondary_linking=query_plan.needs_primary_secondary_linking,
            is_followup=query_plan.is_followup,
            reasoning=query_plan.reasoning,
        )

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
        graph_results = []
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
                    results.append(bibliography_result_to_dict(r))
            except Exception as e:
                logger.error(f"Bib search failed ({action.search_type} / '{action.query}'): {e}")
            return results

        graph_actions = [
            action for action in query_plan.actions
            if action.search_type == "graph_scholar"
        ]
        for action in graph_actions:
            try:
                candidates = await neo4j_service.find_scholars(action.query, limit=3)
                if not candidates:
                    state["processing_steps"].append(
                        f"Neo4j found no Scholar matching '{action.query}'"
                    )
                    continue
                canonical_name = candidates[0]["name"]
                evidence = await neo4j_service.get_scholar_rag_evidence(canonical_name)
                if not evidence:
                    continue
                evidence["resolution"] = candidates[0]
                graph_results.append(evidence)
                state["processing_steps"].append(
                    f"Neo4j resolved {canonical_name}: {len(evidence.get('works', []))} works, "
                    f"{evidence.get('studied_fragment_count', 0)} studied fragments"
                )

                # Graph fragment samples carry their ES document ids, so any
                # of their shelf marks that reach the answer can be linked
                # directly — no shelf-mark search round-trip needed.
                graph_fragment_samples = list(evidence.get("studied_fragment_samples") or [])
                for work in evidence.get("works") or []:
                    graph_fragment_samples.extend(work.get("referenced_fragment_samples") or [])
                for sample in graph_fragment_samples:
                    sample_mark = str(sample.get("shelfmark") or "")
                    sample_doc_id = sample.get("es_doc_id")
                    if (
                        sample_mark and sample_doc_id
                        and "_" in sample_mark
                        and any(c.isalpha() for c in sample_mark)
                        and any(c.isdigit() for c in sample_mark)
                    ):
                        shelf_mark_lookup.setdefault(sample_mark, sample_doc_id)

                try:
                    author_response = await bibliography_search_service.search_by_author(
                        author_name=canonical_name,
                        query=search_query(state),
                        num_results=8,
                    )
                    author_results = [
                        bibliography_result_to_dict(result)
                        for result in author_response.results
                    ]
                    bibliography_results.extend(author_results)
                    for result in author_results:
                        if result.get("shelf_marks_mentioned"):
                            shelf_marks_in_bibliography.update(
                                filter_manuscript_shelfmarks(result["shelf_marks_mentioned"])
                            )
                    state["processing_steps"].append(
                        f"Retrieved {len(author_results)} bibliography chunks constrained to {canonical_name}"
                    )
                except Exception as exc:
                    logger.error("Author-constrained retrieval failed for %r: %s", canonical_name, exc)
                    state["processing_steps"].append(
                        f"Author-constrained bibliography retrieval failed for '{canonical_name}'"
                    )
            except Exception as exc:
                logger.error("Graph scholar retrieval failed for %r: %s", action.query, exc)
                state["processing_steps"].append(
                    f"Graph scholar retrieval failed for '{action.query}': {exc}"
                )

        for action in query_plan.actions:
            if action.search_type == "graph_scholar":
                continue
            logger.info(f"Executing {action.search_type}: {action.query}")

            try:
                if action.search_type in ("bibliography_semantic", "bibliography_hybrid"):
                    new_results = await _run_bib_search(action)
                    bibliography_results.extend(new_results)

                    for r_dict in new_results:
                        if r_dict.get("shelf_marks_mentioned"):
                            shelf_marks_in_bibliography.update(
                                filter_manuscript_shelfmarks(r_dict["shelf_marks_mentioned"])
                            )

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

        bibliography_results = deduplicate_bibliography_results(bibliography_results)

        # ------------------------------------------------------------------
        # Low-similarity fallback: if all bibliography results are below
        # threshold, the retriever found nothing relevant. Attempt one
        # complementary search to confirm absence before proceeding.
        # ------------------------------------------------------------------
        if (
            bibliography_results
            and not graph_results
            and all_results_below_threshold(bibliography_results, SIMILARITY_THRESHOLD)
        ):
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

            fallback_query = search_query(state)
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
                bibliography_results = deduplicate_bibliography_results(fallback_results)
                for r_dict in fallback_results:
                    if r_dict.get("shelf_marks_mentioned"):
                        shelf_marks_in_bibliography.update(
                                filter_manuscript_shelfmarks(r_dict["shelf_marks_mentioned"])
                            )
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

        if not bibliography_results and not graph_results:
            state["error_type"] = "NO_RELEVANT_SOURCES"
            state["error"] = "Neither bibliography nor graph retrieval returned evidence."
            state["processing_steps"].append(
                "No bibliography or graph evidence found; synthesis will return an explicit limitation."
            )

        # Subject-presence gate: nearest-neighbour retrieval always returns
        # something, so a query naming a subject absent from the corpus comes
        # back with pages that share only the collection context. Rather than
        # letting synthesis summarize those as an answer, retry retrieval with
        # a differently-shaped search before concluding the corpus lacks it.
        if (
            state.get("error_type") != "NO_RELEVANT_SOURCES"
            and bibliography_results
            and not graph_results
            and not primary_source_results
        ):
            addresses_query, topical_terms = evidence_addresses_query(
                search_query(state), bibliography_results
            )
            if not addresses_query:
                logger.warning(
                    "Retrieved evidence mentions none of the query's distinctive terms %s; "
                    "attempting focused retrieval recovery",
                    topical_terms,
                )
                state["processing_steps"].append(
                    "First-pass retrieval matched only collection context "
                    f"({', '.join(topical_terms[:6])}); retrying with focused search."
                )
                recovered = await self._recover_irrelevant_retrieval(
                    state, topical_terms, _run_bib_search
                )
                if recovered:
                    bibliography_results = deduplicate_bibliography_results(recovered)
                    for result in recovered:
                        if result.get("shelf_marks_mentioned"):
                            shelf_marks_in_bibliography.update(
                                filter_manuscript_shelfmarks(result["shelf_marks_mentioned"])
                            )
                else:
                    state["error_type"] = "NO_RELEVANT_SOURCES"
                    state["error"] = (
                        "No retrieved source mentions "
                        + ", ".join(topical_terms[:6])
                        + "; the corpus does not appear to cover this subject."
                    )
                    state["subject_terms_not_found"] = topical_terms[:6]
                    state["processing_steps"].append(
                        "Focused retry also found no page discussing the query subject; "
                        "returning an explicit limitation."
                    )

        state["bibliography_results"] = bibliography_results
        state["primary_source_results"] = primary_source_results
        state["graph_results"] = graph_results
        state["shelf_marks_in_bibliography"] = shelf_marks_in_bibliography
        state["shelf_marks_from_search"] = shelf_marks_from_search
        state["shelf_mark_lookup"] = shelf_mark_lookup
        state["processing_steps"].append(
            f"Executed searches: {len(bibliography_results)} bib, {len(primary_source_results)} primary, "
            f"{len(graph_results)} graph neighborhoods. "
            f"Scholars mentioned {len(shelf_marks_in_bibliography)} shelf marks."
        )

        return state

    @weave.op()
    async def _recover_irrelevant_retrieval(
        self,
        state: AgenticRAGState,
        topical_terms: List[str],
        run_bib_search: Callable[[SearchAction], Any],
    ) -> List[Dict[str, Any]]:
        """Retry retrieval when the first pass missed the query's subject.

        Two escalating stages, cheapest first:

        1. A deterministic focused search on the query's distinctive terms
           alone, keyword-heavy. Dropping collection context ("Cairo Genizah
           fragments") stops those words from dominating the match, which is
           the usual reason a first pass returns topically unrelated pages.
        2. A planner re-plan that is told what was searched and why it failed,
           so it can reformulate (different terminology, different search
           type) instead of repeating the original strategy.

        :param state: Current LangGraph RAG state.
        :param topical_terms: Distinctive query terms absent from pass-one evidence.
        :param run_bib_search: Bound bibliography-search executor from the node.
        :returns: Results that address the query, or an empty list.
        :rtype: List[Dict[str, Any]]
        """
        focused_query = " ".join(topical_terms[:6]) or search_query(state)
        stage_one = await run_bib_search(SearchAction(
            search_type="bibliography_hybrid",
            query=focused_query,
            keyword_weight=85,
            semantic_weight=15,
            num_results=6,
        ))
        if stage_one:
            addresses, _ = evidence_addresses_query(search_query(state), stage_one)
            if addresses:
                state["processing_steps"].append(
                    f"Focused retry on '{focused_query}' found pages discussing the subject."
                )
                return stage_one

        replanned_actions = await self._replan_after_failed_retrieval(
            state, topical_terms, focused_query
        )
        recovered: List[Dict[str, Any]] = []
        for action in replanned_actions:
            recovered.extend(await run_bib_search(action))
        if recovered:
            addresses, _ = evidence_addresses_query(search_query(state), recovered)
            if addresses:
                state["processing_steps"].append(
                    "Planner reformulation recovered relevant evidence: "
                    + "; ".join(action.query for action in replanned_actions)
                )
                return recovered
        return []

    async def _replan_after_failed_retrieval(
        self,
        state: AgenticRAGState,
        topical_terms: List[str],
        attempted_query: str,
    ) -> List[SearchAction]:
        """Ask the planner to reformulate after retrieval missed the subject.

        :param state: Current LangGraph RAG state.
        :param topical_terms: Distinctive terms that no retrieved page contained.
        :param attempted_query: The focused query already tried.
        :returns: Up to two replacement bibliography search actions.
        :rtype: List[SearchAction]
        """
        alias_hint = ", ".join(expand_query_aliases(search_query(state))[:10]) or "none known"
        prompt = f"""Retrieval failed to find material on a user's question about the Cairo Genizah
bibliography, and you must reformulate the search.

USER QUESTION: {search_query(state)}
DISTINCTIVE TERMS NOT FOUND IN ANY RETRIEVED PAGE: {', '.join(topical_terms) or 'none'}
ALREADY TRIED: the full question, and a focused search for "{attempted_query}"
KNOWN ALTERNATE SPELLINGS FOR THIS DOMAIN: {alias_hint}

Scholarship on the Genizah uses varied transliterations, Hebrew script, and technical
genre names. Propose one or two DIFFERENT searches likely to surface relevant scholarship:
use alternate transliterations, the Hebrew term, a broader scholarly category the subject
belongs to (for example a specific song genre → "liturgical poetry" or "domestic ritual"),
or a related well-studied topic. Do not repeat what was already tried.

Return ONLY valid JSON:
{{"searches": [{{"query": "...", "keyword_weight": 70}}]}}"""

        try:
            raw = await self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.router_model,
                temperature=0.0,
                recover_reasoning=True,
            )
            parsed = extract_json_object(raw, anchor_key="searches") or {}
        except Exception as exc:
            logger.warning("Retrieval re-plan failed: %s", exc)
            return []

        actions: List[SearchAction] = []
        for entry in (parsed.get("searches") or [])[:2]:
            query = str((entry or {}).get("query") or "").strip()
            if not query:
                continue
            keyword_weight = entry.get("keyword_weight")
            keyword_weight = keyword_weight if isinstance(keyword_weight, int) else 70
            keyword_weight = max(0, min(100, keyword_weight))
            actions.append(SearchAction(
                search_type="bibliography_hybrid",
                query=query,
                keyword_weight=keyword_weight,
                semantic_weight=100 - keyword_weight,
                num_results=6,
            ))
        return actions

    @weave.op()
    async def _link_primary_secondary_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Fetch manuscripts mentioned by scholars"""
        # Skip if we already know there are no relevant sources
        if state.get("error_type") == "NO_RELEVANT_SOURCES":
            state["processing_steps"].append("Skipping linking — no relevant sources found")
            return state

        shelf_marks_to_fetch = state["shelf_marks_in_bibliography"]

        if not shelf_marks_to_fetch:
            # No marks printed on the retrieved pages is the NORMAL case for
            # secondary scholarship, and exactly when the work-level bridge
            # matters most — collect it before returning.
            state["processing_steps"].append("No shelf marks printed on the retrieved pages")
            await self._collect_work_manuscripts(state)
            return state

        # Linking is always attempted when scholars cited shelf marks: it is a
        # cheap lookup that produces the clickable catalog of manuscripts behind
        # an answer, and the planner's flag is not a reliable judge of whether
        # the user would want it.
        if not state["query_plan"].needs_primary_secondary_linking:
            state["processing_steps"].append(
                "Planner suggested skipping linking; fetching cited manuscripts anyway "
                "so they can be linked and listed."
            )

        logger.info(f"Fetching {len(shelf_marks_to_fetch)} manuscripts mentioned by scholars")

        # Which scholarly works cited each shelf mark — recorded with any
        # unresolved mark so missing fragments can be prioritized for scraping.
        citations_by_canonical: Dict[str, List[str]] = {}
        for bibliography in state["bibliography_results"]:
            authors = bibliography.get("authors") or (
                [bibliography.get("author")] if bibliography.get("author") else []
            )
            citation = ", ".join(str(a) for a in authors if a) or "Unknown"
            citation += f", *{bibliography.get('title') or 'Untitled'}*"
            for mentioned in bibliography.get("shelf_marks_mentioned") or []:
                canonical = ShelfmarkNormalizer.to_canonical_id(str(mentioned)).lower()
                if canonical and citation not in citations_by_canonical.setdefault(canonical, []):
                    citations_by_canonical[canonical].append(citation)

        def _record_missing(shelf_mark: str, nearest: Optional[str]) -> None:
            missing_fragment_tracker.record(
                shelf_mark=shelf_mark,
                origin="bibliography_mention",
                citations=citations_by_canonical.get(
                    ShelfmarkNormalizer.to_canonical_id(shelf_mark).lower(), []
                ),
                user_query=state.get("user_query"),
                nearest_match=nearest,
            )

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

                if not response.results:
                    _record_missing(shelf_mark, nearest=None)
                    continue

                if response.results:
                    r = response.results[0]
                    doc_id = r.doc_id
                    fetched_sm = (r.metadata.shelf_mark if r.metadata else None) or ""
                    if not shelfmarks_equivalent(shelf_mark, fetched_sm):
                        # Near-miss fuzzy hit for a fragment not in the index;
                        # linking it would attach the wrong manuscript.
                        _record_missing(shelf_mark, nearest=fetched_sm)
                        continue

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

        await self._collect_work_manuscripts(state)

        return state

    @weave.op()
    async def _collect_work_manuscripts(self, state: AgenticRAGState) -> None:
        """Find the manuscripts behind each work in the retrieved evidence.

        Core to this system's purpose (see docs/DESIGN_PRECEPTS.md): a reader
        who sees that Friedman traced the ketubba's evolution should be able to
        open the ketubbot he worked from, even when the retrieved page itself
        prints no shelf mark. The work — not the page — is the unit of linkage.

        Populates ``state["work_manuscripts"]`` and registers every resolved
        fragment in the shelf-mark lookup so answer text linkifies too.

        :param state: Current LangGraph RAG state.
        """
        titles: List[str] = []
        for bibliography in state.get("bibliography_results", []):
            title = str(bibliography.get("title") or "").strip()
            if title and title not in titles:
                titles.append(title)
        titles = titles[:5]
        if not titles:
            return

        try:
            by_title = await neo4j_service.get_fragments_for_works(titles, per_work_limit=12)
        except Exception as exc:
            logger.warning("Work-to-fragment lookup failed: %s", exc)
            return

        doc_ids = [
            str(fragment.get("es_doc_id"))
            for fragments in by_title.values()
            for fragment in fragments
            if fragment.get("es_doc_id")
        ]
        display_by_doc_id = await self._fetch_display_shelfmarks(doc_ids)

        work_manuscripts: Dict[str, List[Dict[str, str]]] = {}
        lookup = state.setdefault("shelf_mark_lookup", {})
        for title, fragments in by_title.items():
            entries: List[Dict[str, str]] = []
            for fragment in fragments:
                doc_id = str(fragment.get("es_doc_id") or "")
                if not doc_id:
                    continue
                # Only surface fragments this collection actually holds: an
                # unopenable link is worse than no link.
                display = display_by_doc_id.get(doc_id)
                if not display:
                    continue
                entries.append({"shelf_mark": display, "doc_id": doc_id})
                lookup.setdefault(display, doc_id)
            if entries:
                work_manuscripts[title] = entries

        if work_manuscripts:
            state["work_manuscripts"] = work_manuscripts
            total = sum(len(entries) for entries in work_manuscripts.values())
            state["processing_steps"].append(
                f"Linked {total} manuscripts underlying {len(work_manuscripts)} cited works"
            )

    @staticmethod
    async def _build_works_cited(state: AgenticRAGState) -> List[str]:
        """List the works behind an answer, linked to where they can be obtained.

        :param state: Current LangGraph RAG state.
        :returns: Markdown bullet lines, one per distinct cited work.
        :rtype: List[str]
        """
        seen: Set[str] = set()
        works: List[tuple[str, str]] = []  # (title, author display)
        for bibliography in state.get("bibliography_results", []):
            title = str(bibliography.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            authors = bibliography.get("authors") or (
                [bibliography.get("author")] if bibliography.get("author") else []
            )
            author_text = ", ".join(str(a) for a in authors if a)
            works.append((title, author_text))
        if not works:
            return []

        lines: List[str] = []
        for title, author_text in works[:6]:
            link = None
            try:
                link = build_direct_work_link(await neo4j_service.find_book_article(title))
            except Exception as exc:
                logger.warning("Work-link lookup failed for %r: %s", title, exc)
            prefix = f"**{author_text}**, " if author_text else ""
            lines.append(
                f"- {prefix}[{title}]({link})" if link else f"- {prefix}{title}"
            )
        return lines

    @staticmethod
    async def _fetch_display_shelfmarks(doc_ids: List[str]) -> Dict[str, str]:
        """Resolve primary-index document ids to their display shelf marks.

        :param doc_ids: Elasticsearch document ids from graph fragment records.
        :returns: Mapping of document id to display-form shelf mark, omitting
            ids the primary index does not hold.
        :rtype: Dict[str, str]
        """
        unique_ids = list(dict.fromkeys(doc_id for doc_id in doc_ids if doc_id))[:60]
        if not unique_ids:
            return {}
        try:
            response = search_service.es.mget(
                index=search_service.index_name,
                ids=unique_ids,
                _source=["shelf_mark"],
            )
        except Exception as exc:
            logger.warning("Display shelf-mark lookup failed: %s", exc)
            return {}
        resolved: Dict[str, str] = {}
        for doc in response.get("docs", []):
            if not doc.get("found"):
                continue
            shelf_mark = str((doc.get("_source") or {}).get("shelf_mark") or "").strip()
            if shelf_mark:
                resolved[str(doc.get("_id"))] = shelf_mark
        return resolved

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
            missing_terms = state.get("subject_terms_not_found") or []
            subject_note = (
                f" No indexed page mentions {', '.join(missing_terms)}."
                if missing_terms
                else ""
            )
            state["draft_answer"] = (
                "I wasn't able to find relevant information about this topic in the "
                f"scholarly bibliography.{subject_note} The corpus may not yet include work "
                "on this specific subject or scholar. You may want to search external "
                "databases such as the Princeton Geniza Project catalog or JSTOR directly."
            )
            state["processing_steps"].append(
                "Short-circuited synthesis: no relevant sources above similarity threshold."
            )
            return state

        state["error"] = None
        state["error_type"] = None

        bibliography_sources = build_bibliography_source_context(state["bibliography_results"])
        all_sources = build_verification_sources(
            state["bibliography_results"],
            state.get("graph_results", []),
        )
        bib_context = [source["prompt_text"] for source in bibliography_sources]
        graph_context = "\n\n".join(
            source["prompt_text"] for source in all_sources[len(bibliography_sources):]
        )

        system_prompt = """You are a scholarly research assistant specializing in Cairo Genizah studies.

Your inputs are chunks retrieved from academic secondary sources (books and articles about the Genizah). Your job is to synthesize these sources into a coherent scholarly response with precise citations.

Rules:
1. Lead with what scholars have written. Prefer short direct quotations where they strengthen
   the response. A quotation must be at most 30 words (roughly one or two lines) and copied
   only from a field labeled "Original page text (quoteable)."
   Format quotes as: **Author**, p. X: "quote text"
   Always wrap a cited author's name in ** so it renders bold, in quoted and
   unquoted citations alike (e.g. **Goitein**, p. 112).
2. Every factual claim must cite a specific retrieved source with page number.
   Do not draw on background knowledge — only the retrieved chunks.
3. When a retrieved source identifies a manuscript by shelf mark, cite that shelf mark
   exactly as written — readers can open those manuscripts directly, so specific shelf marks
   are among the most valuable things an answer carries. Prefer a claim anchored to a named
   fragment over the same claim stated generically. Treat a shelf mark as a reference to be
   cited, not a document to analyze: add no information about what the fragment contains
   beyond what the source text says, and never invent or adapt a shelf mark.
4. Do not invent shelf-marks, page numbers, or citations. If the retrieved sources
   don't cover an aspect of the query, say so explicitly rather than filling the gap.
5. If retrieved sources are sparse, return what you have with honest attribution
   rather than padding with general knowledge.
5a. CRITICAL — RELEVANCE: the retrieved sources are nearest matches, not guaranteed answers.
   If they do not discuss the specific subject the user asked about, say so plainly in one or
   two sentences and STOP. Never follow such a statement with a survey of what the sources do
   cover ("the available texts focus on..."), and never summarize a source merely because it
   was retrieved. An unrelated page about medicine, trade, or magic is not a partial answer to
   a question about liturgy — it is silence. Only material that bears on the user's actual
   question belongs in the response.
6. CRITICAL — QUOTES: Only use text in quotation marks if it appears verbatim (or near-verbatim)
   in "Original page text (quoteable)." Never quote a generated catalog summary. Do not
   construct plausible-sounding quotes.
   If you want to represent what a scholar argued, paraphrase with attribution instead.
7. Neo4j evidence is structured catalog and relationship metadata, not prose scholarship.
   You may report graph relationships and counts using wording such as "the knowledge graph
   records" or "the graph associates." Do not infer a work's argument or subject solely from
   a WROTE, STUDIED, or REFERENCES edge. Claims about what a scholar argues must come from
   retrieved scholarly source text.
8. If Neo4j lists works for which no indexed text was retrieved, distinguish those graph
   associations from works whose text is available in the bibliography evidence.
9. When graph evidence includes sample fragment identifiers, mention at most two or three
   representative ones per work where they add value. Never reproduce long lists of raw
   identifiers — aggregate counts ("references 80 fragments") communicate scale better.
10. DO state plainly when the retrieved scholarship does not record something the user asked
   about — a shelf mark, a date, a scribe, a provenance. Honest limitation notes are wanted,
   required behavior. Phrase them in terms of the scholarship ("the retrieved scholarship
   does not record this fragment's shelf mark"), never in terms of system internals: do not
   mention excerpts, chunks, prompts, retrieval, the pipeline, or knowledge-graph coverage.
11. Sources may be written in Hebrew, Aramaic, or Judeo-Arabic. They are FIRST-CLASS
   evidence: read them, cite them with page numbers, and prefer their specific content over
   generic English material. When quoting, copy the original script verbatim inside straight
   double quotes and give a rendering in the answer's language OUTSIDE the quotation marks —
   never present your own translation as though it were a quotation.
12. ANSWER IN THE USER'S LANGUAGE: write the prose of your answer in the language of the
   user's question — a Hebrew question receives a Hebrew answer, an English question an
   English answer — regardless of the language of the retrieved sources. Quotations keep
   their original script (rule 11), and shelf marks, scholars' names, and work titles stay
   exactly as written in the sources."""

        exclusion_note = ""
        excluded_claims = state.get("excluded_claims", [])

        if excluded_claims:
            exclusion_note = (
                "\n\nThe verifier found the following unsupported items in the prior draft. "
                "Remove each item or rewrite it as a narrower claim that is directly supported "
                "by a numbered source. Do not repeat it merely with softer wording:\n"
                + "\n".join(
                    f"  - {claim.get('type', 'claim')}: "
                    f"{str(claim.get('text') or '')[:160]} "
                    f"(reason: {str(claim.get('reason') or 'not found in evidence')[:160]})"
                    for claim in excluded_claims
                )
            )

        graph_section = (
            f"\n\nNEO4J GRAPH EVIDENCE:\n\n{graph_context}"
            if graph_context
            else ""
        )

        # A follow-up ("what is the shelf mark of that piyyut?") is meaningless
        # without the exchange it refers to; supply it as context for resolving
        # references, never as a source of factual claims.
        conversation_section = ""
        rendered_history = render_conversation(
            conversation_turns(state), max_turns=4, user_chars=600, assistant_chars=1200
        )
        if rendered_history:
            conversation_section = (
                "\n\nEARLIER IN THIS CONVERSATION (for resolving what the user means by "
                "'that fragment', 'it', or 'the piyyut' — NOT a source: every factual "
                "claim and citation must still come from the retrieved sources below):\n"
                + rendered_history
            )

        # Show the standalone reading next to the literal message so the
        # synthesis model answers the question actually being asked.
        resolved = search_query(state)
        query_section = str(state["user_query"])
        if resolved and resolved != state["user_query"]:
            query_section += f"\n(In the context of the conversation, this asks: {resolved})"

        user_message = f"""{system_prompt}{exclusion_note}{conversation_section}

RETRIEVED SCHOLARLY SOURCES:

{chr(10).join(bib_context) if bib_context else "No scholarly sources retrieved."}
{graph_section}

USER QUERY:
{query_section}

Provide your scholarly synthesis. Use only the retrieved source text and structured graph
evidence above, following their distinct provenance rules."""

        messages = [{"role": "user", "content": user_message}]

        # Allow a per-request synthesis model override (chosen in the UI for testing);
        # fall back to the configured default when none is supplied.
        synthesis_model = state.get("synthesis_model_override") or self.synthesis_model
        logger.info(f"Synthesizing with model: {synthesis_model}")
        # Thinking-heavy models can spend the entire output budget reasoning
        # and emit no answer at all (observed: 8191/8191 reasoning tokens).
        # Give synthesis a large budget, then verify prose actually arrived.
        draft_answer = await self._call_llm(
            messages=messages,
            model=synthesis_model,
            temperature=0.2,
            max_tokens=SYNTHESIS_MAX_TOKENS,
        )
        if not draft_answer.strip():
            logger.warning(
                "Synthesis returned no answer text (model %s); retrying once",
                synthesis_model,
            )
            state["processing_steps"].append(
                "Synthesis produced no answer text; retrying once"
            )
            draft_answer = await self._call_llm(
                messages=messages,
                model=synthesis_model,
                temperature=0.2,
                max_tokens=SYNTHESIS_MAX_TOKENS,
            )
        if not draft_answer.strip():
            # Never let an empty draft flow onward: the verifier would extract
            # "claims" from the evidence and pass it, and the finalizer would
            # append catalog sections to nothing, shipping an appendix-only
            # answer. Fail honestly instead.
            logger.error(
                "Synthesis produced no answer text after retry (model %s)", synthesis_model
            )
            state["draft_answer"] = (
                "The assistant could not compose an answer to this question just now. "
                "Please try again in a moment."
            )
            state["error_type"] = "NO_ANSWER_GENERATED"
            state["error"] = "Synthesis returned no answer text after one retry."
            state["processing_steps"].append(
                "Synthesis produced no answer text after retry; returning an honest failure."
            )
            return state

        state["draft_answer"] = draft_answer
        retry = state.get("retry_count", 0)
        state["processing_steps"].append(
            "Synthesized scholarly answer" + (f" (retry {retry})" if retry else "")
        )

        return state

    _REPAIR_RESPONSE_FORMAT: Dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "paragraph_revisions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "revisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paragraph_index": {"type": "integer"},
                                "revised_text": {"type": "string"},
                            },
                            "required": ["paragraph_index", "revised_text"],
                        },
                    },
                },
                "required": ["revisions"],
            },
        },
    }

    @weave.op()
    async def _repair_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Surgically revise only the paragraphs containing hard-rejected claims.

        Verified text must never be re-generated: paragraphs without rejected
        claims are spliced back byte-identical, so repeated repairs converge on
        the error sites instead of re-rolling good prose. Rejected claims that
        cannot be located in the draft are downgraded to user-visible soft
        flags rather than risking a whole-answer rewrite.

        :param state: Current LangGraph RAG state after failed verification.
        :returns: State containing a minimally revised draft answer.
        :rtype: AgenticRAGState
        """
        rejected_claims = state.get("excluded_claims", [])
        if not rejected_claims:
            return state

        draft = str(state.get("draft_answer") or "")
        paragraphs = draft.split("\n\n")
        sources = build_verification_sources(
            state.get("bibliography_results", []),
            state.get("graph_results", []),
        )
        supported_quotes = [
            quote
            for quote in extract_direct_quotes(draft)
            if find_quote_source(quote, sources) is not None
        ]

        def _paragraph_index_of(sentence: str) -> Optional[int]:
            for index, paragraph in enumerate(paragraphs):
                if sentence in paragraph:
                    return index
            return None

        claims_by_paragraph: Dict[int, List[Dict[str, str]]] = {}
        unlocatable: List[Dict[str, str]] = []
        for claim in rejected_claims:
            sentence = locate_claim_sentence(str(claim.get("text") or ""), draft)
            paragraph_index = _paragraph_index_of(sentence) if sentence else None
            if paragraph_index is None:
                unlocatable.append(claim)
                continue
            claims_by_paragraph.setdefault(paragraph_index, []).append(dict(claim, sentence=sentence))

        if unlocatable:
            # A rejection that cannot be anchored to a sentence cannot be
            # surgically repaired; surface it to the user instead of gambling
            # the whole answer on another rewrite.
            state.setdefault("soft_flagged_claims", []).extend(
                {
                    "type": claim.get("type") or "factual_claim",
                    "text": str(claim.get("text") or ""),
                    "reason": (
                        "Rejected by verification but could not be located for targeted "
                        f"repair: {str(claim.get('reason') or 'no reason supplied')[:200]}"
                    ),
                }
                for claim in unlocatable
            )
            state["processing_steps"].append(
                f"{len(unlocatable)} rejected claims could not be anchored to the draft; "
                "flagged for user review instead of repair"
            )

        repair_number = state.get("retry_count", 1)
        if not claims_by_paragraph:
            state["error"] = None
            state["error_type"] = None
            return state

        sources_text = "\n\n".join(
            str(source["prompt_text"]) for source in sources
        ) or "No sources retrieved."
        paragraph_sections: List[str] = []
        for index in sorted(claims_by_paragraph):
            problems = "\n".join(
                f"  - Sentence: {claim['sentence']}\n"
                f"    Rejected claim: {str(claim.get('text') or '')[:200]}\n"
                f"    Reason: {str(claim.get('reason') or 'Not supported')[:200]}"
                for claim in claims_by_paragraph[index]
            )
            paragraph_sections.append(
                f"PARAGRAPH {index}:\n{paragraphs[index]}\n\nPROBLEMS IN PARAGRAPH {index}:\n{problems}"
            )
        protected_quote_text = (
            "\n".join(f'- "{quote}"' for quote in supported_quotes) if supported_quotes else "None"
        )

        repair_prompt = f"""You are a conservative copy editor. Revise ONLY the paragraphs below from a scholarly
answer; every other paragraph of the answer is verified and will be kept unchanged
automatically. This is targeted repair {repair_number} of {MAX_VERIFICATION_REPAIR_ATTEMPTS}.

For each listed paragraph, fix exactly the listed problems and nothing else:
- Delete or correct only the offending sentence or clause; a correction must be directly
  supported by a numbered source below.
- Keep every other sentence of the paragraph verbatim.
- Do not add new factual claims, quotations, shelf marks, or citations.
- If nothing in a paragraph can be salvaged, return an empty revised_text to delete it.
- Keep transitions natural: if removing a sentence breaks the flow, you may minimally adjust
  the adjacent connective wording without changing any factual content.

These verified quotations must remain verbatim wherever they occur:
{protected_quote_text}

NUMBERED EVIDENCE:
{sources_text}

{chr(10).join(paragraph_sections)}

Return revisions for every listed paragraph index."""

        revised_by_index: Dict[int, str] = {}
        try:
            raw_response = await self._call_llm(
                messages=[{"role": "user", "content": repair_prompt}],
                model=state.get("synthesis_model_override") or self.synthesis_model,
                temperature=0.0,
                response_format=self._REPAIR_RESPONSE_FORMAT,
                recover_reasoning=True,
            )
            parsed_revisions = extract_json_object(raw_response, anchor_key="revisions") or {}
            for revision in parsed_revisions.get("revisions", []):
                index = revision.get("paragraph_index")
                if isinstance(index, int) and index in claims_by_paragraph:
                    revised_by_index[index] = str(revision.get("revised_text") or "")
        except Exception as exc:
            logger.error("Paragraph repair call failed; using deterministic removal: %s", exc)

        removed_or_revised = 0
        for index, claims in claims_by_paragraph.items():
            original_paragraph = paragraphs[index]
            revised = revised_by_index.get(index)
            if revised is None:
                revised = remove_sentences_containing(
                    original_paragraph,
                    [str(claim.get("text") or "") for claim in claims],
                )
            # A revision may not drop a protected quote that the original
            # paragraph contained; fall back to deterministic removal.
            for quote in supported_quotes:
                normalized_quote = _normalize_verification_text(quote)
                if (
                    normalized_quote in _normalize_verification_text(original_paragraph)
                    and normalized_quote not in _normalize_verification_text(revised)
                ):
                    revised = remove_sentences_containing(
                        original_paragraph,
                        [str(claim.get("text") or "") for claim in claims],
                    )
                    break
            paragraphs[index] = revised
            removed_or_revised += len(claims)

        state["draft_answer"] = "\n\n".join(
            paragraph for paragraph in paragraphs if paragraph.strip()
        )
        state["error"] = None
        state["error_type"] = None
        state["processing_steps"].append(
            f"Targeted repair {repair_number}/{MAX_VERIFICATION_REPAIR_ATTEMPTS}: revised "
            f"{len(claims_by_paragraph)} paragraphs covering {removed_or_revised} rejected claims"
        )
        return state

    @weave.op()
    async def _verify_claims_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Verify quotes, shelf marks, and atomic claims against identical evidence.

        Quotes and shelf marks are checked deterministically. The verification
        model checks the complete proposition for every remaining factual claim,
        including attributed, hedged, and graph-derived claims. Any unsupported
        item triggers a constrained synthesis retry.

        :param state: Current LangGraph RAG state.
        :returns: State populated with claim-level verification results.
        :rtype: AgenticRAGState
        """
        logger.info("Running claim and quote verification")
        if state.get("error_type") in ("NO_RELEVANT_SOURCES", "NO_ANSWER_GENERATED"):
            return state
        if not str(state.get("draft_answer") or "").strip():
            # An empty draft has nothing to verify; running the verifier on it
            # makes the model invent claims from the evidence and "pass" them.
            logger.error("Verification skipped: draft answer is empty")
            state["error_type"] = "NO_ANSWER_GENERATED"
            state["error"] = "Draft answer was empty at verification time."
            state["draft_answer"] = (
                "The assistant could not compose an answer to this question just now. "
                "Please try again in a moment."
            )
            state["processing_steps"].append("Verification skipped — empty draft answer.")
            return state

        draft = str(state.get("draft_answer") or "")
        sources = build_verification_sources(
            state.get("bibliography_results", []),
            state.get("graph_results", []),
        )
        source_citations = {
            int(source["source_number"]): str(source["citation"])
            for source in sources
        }
        sources_text = "\n\n".join(
            str(source["prompt_text"]) for source in sources
        ) or "No sources retrieved."

        deterministic_claims: List[Dict[str, Any]] = []
        for quote in extract_direct_quotes(draft):
            source_number = find_quote_source(quote, sources)
            deterministic_claims.append({
                "type": "quote",
                "text": quote,
                "supported": source_number is not None,
                "source_number": source_number,
                "reasoning": (
                    f"Near-verbatim text found in SOURCE {source_number}"
                    if source_number is not None
                    else "Quoted wording was not found in the supplied evidence"
                ),
            })
        for shelf_mark in detect_shelfmarks(draft):
            source_number = find_shelfmark_source(shelf_mark, sources)
            deterministic_claims.append({
                "type": "shelf_mark",
                "text": shelf_mark,
                "supported": source_number is not None,
                "source_number": source_number,
                "reasoning": (
                    f"Equivalent shelf mark found in SOURCE {source_number}"
                    if source_number is not None
                    else "Shelf mark was not found in the supplied evidence"
                ),
            })

        prior_excluded: List[Dict[str, str]] = []
        seen_prior: Set[tuple[str, str]] = set()
        for feedback in state.get("verification_feedback_history", []):
            for claim in feedback.get("rejected_claims", []):
                claim_type = str(claim.get("type") or "claim")
                claim_text = str(claim.get("text") or "")
                key = (claim_type, _normalize_verification_text(claim_text))
                if key in seen_prior:
                    continue
                seen_prior.add(key)
                prior_excluded.append({
                    "type": claim_type,
                    "text": claim_text,
                    "reason": str(claim.get("reason") or "Not supported"),
                })
        exclusion_note = ""
        if prior_excluded:
            exclusion_note = (
                "\nFor context only: earlier drafts of this answer contained the rejected items "
                "below, and the current draft may state corrected versions of them. A prior "
                "rejection NEVER overrides current source evidence. Judge each claim in the "
                "current draft strictly against the SOURCE CHUNKS: if a source directly supports "
                "the claim as now written, it is supported regardless of this list; only mark it "
                "otherwise if the current wording still lacks support or contradicts a source. "
                "Base every reasoning field on source content, never on these instructions:\n"
                + "\n".join(
                    f"- {claim.get('type', 'claim')}: {str(claim.get('text') or '')[:160]}"
                    for claim in prior_excluded
                )
            )

        verifier_prompt = f"""You verify a scholarly answer against numbered evidence.

Split the DRAFT ANSWER into atomic, substantive factual claims. Check every claim against
the actual proposition expressed by a SOURCE, not merely whether an author's name or a
related keyword occurs. This includes:
- claims attributed to a scholar: the source must support what the draft says they argued;
- un-attributed background and descriptive claims;
- numerical, publication, relationship, and manuscript-content claims;
- Neo4j claims, which must be limited to relationships and counts explicitly in graph evidence.

Give each claim exactly one verdict:
- "supported": one numbered source directly supports the whole proposition.
- "contradicted": a numbered source states the OPPOSITE of the claim. Reserve this for real
  conflicts (wrong holiday, wrong person, wrong date where the source gives a different one).
- "unsupported": the claim is neither supported nor contradicted by the evidence.

Calibration rules — apply these before assigning a verdict:
1. HEDGING: if the draft hedges a claim (seems, may, perhaps, likely, suggests) and a source
   asserts the same content with equivalent hedging or speculation, the claim is SUPPORTED.
   Faithfully reporting a scholar's surmise as a surmise is correct scholarship, not error.
2. TITLES COUNT: a source's own title, headings, and citation line are evidence. A claim that
   restates what the title asserts is SUPPORTED by that source.
3. Judge the proposition the draft actually makes, not a stronger version of it.
4. If evidence supports only part of a sentence, return the unsupported portion as its own
   claim rather than failing the whole sentence.
5. EVIDENCE-LIMITATION CLAIMS: a statement about what the retrieved sources themselves do or
   do not contain ("the sources do not identify the scribe", "no shelf mark is recorded") is
   judged against the SOURCE CHUNKS as the complete universe. If the sources indeed lack what
   the draft says is missing, mark it SUPPORTED with source_number null — an accurate
   limitation note is honest scholarship, never fabrication. If a source plainly contains the
   supposedly missing item, mark it CONTRADICTED. Use type "evidence_limitation" for these.

For bibliography sources, generated catalog summaries are orientation aids and are not
sufficient evidence by themselves. Factual support must be traceable to text labeled
"Original page text (quoteable)" or to the source's citation line and title. Neo4j records
may support graph claims only. Sources and quotations may be in Hebrew, Aramaic, or
Judeo-Arabic: apply every rule identically across languages, and treat an English draft
claim as supported when a Hebrew source states it in Hebrew.

Do not separately return direct quotes or shelf-mark identifiers; those are checked by
deterministic matchers. You must still verify the surrounding proposition containing them.
{exclusion_note}

SOURCE CHUNKS:
{sources_text}

DRAFT ANSWER:
{draft}

The type must be attribution, factual_claim, graph_claim, citation_claim, or
evidence_limitation.
source_number must be an integer for supported claims (except supported
evidence_limitation claims, where it is null) and null otherwise.
If the draft contains no substantive factual claims, return an empty verified_claims list.
"""

        model_claims, model_summary, parse_error = await self._call_verifier(verifier_prompt)

        verified_claim_objects: List[VerifiedClaim] = []
        hard_failures: List[Dict[str, str]] = []
        soft_flags: List[Dict[str, Any]] = []
        supported_evidence_units: List[Dict[str, Any]] = []

        def _record(claim_type: str, claim_text: str, status: str, citation: str, reasoning: str) -> None:
            verified_claim_objects.append(VerifiedClaim(
                claim=f"{claim_type}: {claim_text[:160]}",
                source_citation=citation,
                verification_status=status,
                confidence=1.0 if status == "SUPPORTED" else 0.0,
                reasoning=reasoning,
            ))

        # Deterministic quote and shelf-mark checks are hard gates: a quotation
        # or shelf mark absent from the evidence is a fabrication, never a flag.
        for claim in deterministic_claims:
            claim_type = str(claim.get("type") or "claim")
            claim_text = str(claim.get("text") or "")
            reasoning = str(claim.get("reasoning") or "")
            source_number = claim.get("source_number")
            if claim.get("supported") and source_number in source_citations:
                citation = source_citations[source_number]
                _record(claim_type, claim_text, "SUPPORTED", citation, reasoning)
                supported_evidence_units.append({
                    "type": claim_type,
                    "text": claim_text,
                    "source_number": source_number,
                    "citation": citation,
                    "reason": reasoning,
                })
            else:
                _record(claim_type, claim_text, "NOT_SUPPORTED", "Not found in sources", reasoning)
                hard_failures.append({"type": claim_type, "text": claim_text, "reason": reasoning})

        for claim in model_claims:
            claim_type = str(claim.get("type") or "factual_claim")
            claim_text = str(claim.get("text") or "")
            reasoning = str(claim.get("reasoning") or "No verification reasoning supplied")
            verdict = str(claim.get("verdict") or "unsupported")
            source_number = claim.get("source_number")
            has_valid_source = (
                not isinstance(source_number, bool)
                and isinstance(source_number, int)
                and source_number in source_citations
            )
            if verdict == "supported" and not has_valid_source:
                if claim_type == "evidence_limitation":
                    # An accurate statement about what the evidence LACKS spans
                    # the whole bundle, so no single source number applies.
                    citation = "Checked against all retrieved sources"
                    _record(claim_type, claim_text, "SUPPORTED", citation, reasoning)
                    supported_evidence_units.append({
                        "type": claim_type,
                        "text": claim_text,
                        "source_number": None,
                        "citation": citation,
                        "reason": reasoning,
                    })
                    continue
                verdict = "unsupported"
                reasoning = "Verifier marked this supported but supplied no valid source number"

            if verdict == "supported":
                # Trust-but-verify the citation itself: a hallucinated
                # source_number is the one way an unsupported claim could
                # publish as verified. A genuinely supporting source shares at
                # least one distinctive term with the claim, even paraphrased.
                cited_source = next(
                    (s for s in sources if int(s["source_number"]) == source_number), None
                )
                claim_terms = extract_topical_terms(claim_text)
                if cited_source is not None and claim_terms:
                    source_raw = str(cited_source.get("prompt_text") or "")
                    source_text = _normalize_verification_text(source_raw)
                    # Cross-script pairs (an English claim supported by a
                    # Hebrew source, or vice versa) can share zero surface
                    # terms while being genuinely supported; term overlap is
                    # meaningless there. Corroborate via domain concepts when
                    # possible, otherwise trust the verifier's verdict.
                    cross_script = (
                        (hebrew_char_ratio(claim_text) > 0.5)
                        != (hebrew_char_ratio(source_raw) > 0.5)
                    )
                    if cross_script:
                        claim_concepts = find_concepts(_normalize_verification_text(claim_text))
                        if claim_concepts and not (claim_concepts & find_concepts(source_text)):
                            logger.info(
                                "Cross-script citation lacks concept overlap; trusting "
                                "verifier verdict for %r", claim_text[:80],
                            )
                    elif not any(term_appears_in_text(t, source_text) for t in claim_terms):
                        _record(
                            claim_type, claim_text, "NOT_SUPPORTED",
                            "Citation could not be corroborated",
                            "Verifier cited a source that shares no distinctive terms "
                            "with this claim; the citation could not be corroborated",
                        )
                        soft_flags.append({
                            "type": claim_type,
                            "text": claim_text,
                            "reason": (
                                "The verifier attributed this claim to a source that does "
                                "not appear to mention its subject at all."
                            ),
                        })
                        continue
                citation = source_citations[source_number]
                _record(claim_type, claim_text, "SUPPORTED", citation, reasoning)
                supported_evidence_units.append({
                    "type": claim_type,
                    "text": claim_text,
                    "source_number": source_number,
                    "citation": citation,
                    "reason": reasoning,
                })
            elif verdict == "contradicted":
                _record(claim_type, claim_text, "CONTRADICTED", "Contradicted by sources", reasoning)
                hard_failures.append({"type": claim_type, "text": claim_text, "reason": reasoning})
            else:
                _record(claim_type, claim_text, "NOT_SUPPORTED", "Not found in sources", reasoning)
                soft_flags.append({
                    "type": claim_type,
                    "text": claim_text,
                    "reason": reasoning,
                })

        if parse_error:
            soft_flags.append({
                "type": "verification_error",
                "text": "Automatic claim verification did not complete for this answer",
                "reason": f"The verifier returned invalid structured output: {parse_error[:200]}",
            })

        for rejected in hard_failures:
            logger.warning(
                "Hard-rejected claim: type=%s text=%r reason=%s",
                rejected.get("type"), rejected.get("text"), rejected.get("reason"),
            )
        for flagged in soft_flags:
            logger.info(
                "Soft-flagged claim: type=%s text=%r",
                flagged.get("type"), flagged.get("text"),
            )

        supported_count = sum(
            claim.verification_status == "SUPPORTED" for claim in verified_claim_objects
        )
        state["verified_claims"] = verified_claim_objects
        state["supported_evidence_units"] = supported_evidence_units
        state["soft_flagged_claims"] = soft_flags
        state["verification_summary"] = {
            "SUPPORTED": supported_count,
            "NOT_SUPPORTED": len(soft_flags),
            "CONTRADICTED_OR_FABRICATED": len(hard_failures),
            **({"parse_error": 1} if parse_error else {}),
        }

        if hard_failures:
            retry_count = state.get("retry_count", 0) + 1
            summary = model_summary or f"{len(hard_failures)} contradicted or fabricated claims"
            logger.warning(
                "Verification failed (attempt %s): %s hard failures. %s",
                retry_count, len(hard_failures), summary,
            )
            state["excluded_claims"] = hard_failures
            state["retry_count"] = retry_count
            feedback_history = state.setdefault("verification_feedback_history", [])
            feedback_history.append({
                "attempt": retry_count,
                "summary": summary,
                "rejected_claims": hard_failures,
                "supported_claims": supported_evidence_units,
            })
            state["error_type"] = "FABRICATED_CLAIMS"
            state["error"] = summary
            state["processing_steps"].append(
                f"Verification FAILED (attempt {retry_count}/"
                f"{MAX_VERIFICATION_REPAIR_ATTEMPTS + 1}): {summary}"
            )
        else:
            state["excluded_claims"] = []
            state["error_type"] = None
            state["error"] = None
            summary = model_summary or f"Verified {supported_count} claims"
            flag_note = f"; {len(soft_flags)} claims flagged for user review" if soft_flags else ""
            # Also log (not only processing_steps) so long-horizon log analysis
            # can measure pass rates and flag composition.
            logger.info(
                "Verification PASSED: %s supported, %s flagged", supported_count, len(soft_flags)
            )
            state["processing_steps"].append(f"Verification PASSED: {summary}{flag_note}")

        return state

    _VERIFIER_RESPONSE_FORMAT: Dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "claim_verification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verified_claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "attribution",
                                        "factual_claim",
                                        "graph_claim",
                                        "citation_claim",
                                        "evidence_limitation",
                                    ],
                                },
                                "text": {"type": "string"},
                                "verdict": {
                                    "type": "string",
                                    "enum": ["supported", "unsupported", "contradicted"],
                                },
                                "source_number": {"type": ["integer", "null"]},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["type", "text", "verdict", "source_number", "reasoning"],
                        },
                    },
                    "summary": {"type": "string"},
                },
                "required": ["verified_claims", "summary"],
            },
        },
    }

    async def _call_verifier(
        self,
        verifier_prompt: str,
    ) -> tuple[List[Dict[str, Any]], str, Optional[str]]:
        """Run the claim verifier with schema-enforced output and one retry.

        :param verifier_prompt: Full verification prompt including evidence.
        :returns: Model claim dictionaries, the verifier summary, and a parse
            error message when both attempts returned invalid structure.
        :rtype: tuple[List[Dict[str, Any]], str, Optional[str]]
        """
        parse_error: Optional[str] = None
        for attempt in range(2):
            raw_response = await self._call_llm(
                messages=[{"role": "user", "content": verifier_prompt}],
                model=self.verification_model,
                temperature=0.0,
                response_format=self._VERIFIER_RESPONSE_FORMAT,
                recover_reasoning=True,
            )
            try:
                verification_result = extract_json_object(raw_response, anchor_key="verified_claims")
                if verification_result is None:
                    raise ValueError("no JSON object with verified_claims found")
                raw_claims = verification_result.get("verified_claims", [])
                if not isinstance(raw_claims, list) or not all(
                    isinstance(claim, dict) for claim in raw_claims
                ):
                    raise ValueError("verified_claims must be a list of objects")
                return raw_claims, str(verification_result.get("summary") or ""), None
            except (AttributeError, TypeError, ValueError) as exc:
                parse_error = str(exc)
                logger.error(
                    "Verifier returned invalid structured output (attempt %s): %s; response=%s",
                    attempt + 1, exc, raw_response[:200],
                )
        return [], "Claim verification did not complete", parse_error

    @staticmethod
    def _flags_to_models(flags: List[Dict[str, Any]]) -> List[FlaggedClaim]:
        """Convert state flag dictionaries into response models.

        :param flags: Soft-flag dictionaries from the final pipeline state.
        :returns: FlaggedClaim models with stable ids for UI anchoring.
        :rtype: List[FlaggedClaim]
        """
        models: List[FlaggedClaim] = []
        for index, flag in enumerate(flags, start=1):
            models.append(FlaggedClaim(
                flag_id=int(flag.get("flag_id") or index),
                claim_type=str(flag.get("type") or "factual_claim"),
                text=str(flag.get("text") or ""),
                answer_span=flag.get("answer_span"),
                reason=str(flag.get("reason") or ""),
                source_citation=flag.get("citation"),
            ))
        return models

    _GRACEFUL_FALLBACK = (
        "I wasn't able to construct a fully verified response for this query. "
        "Please try rephrasing or narrowing your question, or use the search "
        "panel directly to explore relevant primary sources."
    )

    @staticmethod
    def _build_verified_evidence_fallback(state: AgenticRAGState) -> Optional[str]:
        """Build coherent source-grouped prose from verified facts and quotes.

        :param state: Current RAG state after verification retry exhaustion.
        :returns: Citation-bearing evidence paragraphs, or ``None`` if nothing survived.
        :rtype: Optional[str]
        """
        sources = build_verification_sources(
            state.get("bibliography_results", []),
            state.get("graph_results", []),
        )
        citations = {
            int(source["source_number"]): str(source["citation"])
            for source in sources
        }
        evidence_units = collect_supported_evidence_units(state)
        for quote in extract_direct_quotes(str(state.get("draft_answer") or "")):
            source_number = find_quote_source(quote, sources)
            if source_number is None:
                continue
            evidence_units.append({
                "type": "quote",
                "text": bound_direct_quote(quote),
                "source_number": source_number,
                "citation": citations[source_number],
            })

        grouped: Dict[str, Dict[str, List[str]]] = {}
        seen: Set[tuple[str, str, str]] = set()
        for unit in evidence_units:
            citation = str(unit.get("citation") or "").strip()
            text = str(unit.get("text") or "").strip()
            unit_type = str(unit.get("type") or "factual_claim")
            if not citation or not text:
                continue
            if unit_type == "quote":
                text = bound_direct_quote(text)
            key = (citation, unit_type, _normalize_verification_text(text.rstrip("…")))
            if key in seen:
                continue
            seen.add(key)
            group = grouped.setdefault(citation, {"facts": [], "quotes": []})
            destination = "quotes" if unit_type == "quote" else "facts"
            group[destination].append(text)

        if not grouped:
            return None

        paragraphs: List[str] = []
        for citation, group in grouped.items():
            sentences: List[str] = []
            for fact in group["facts"]:
                bounded_fact = fact.rstrip()
                if bounded_fact and bounded_fact[-1] not in ".?!":
                    bounded_fact += "."
                lead = "the verified evidence supports: " if not sentences else "It also supports: "
                sentences.append(lead + bounded_fact)
            for quote in group["quotes"]:
                lead = "the source states" if not sentences else "It also states"
                sentences.append(f'{lead}: “{quote}”')
            paragraphs.append(f"In {citation}, " + " ".join(sentences))

        return (
            "A fully synthesized response did not pass verification after three targeted "
            "repairs. The following source-linked account includes only claims and short "
            "quotations that did pass verification:\n\n"
            + "\n\n".join(paragraphs)
        )

    @weave.op()
    async def _finalize_response_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Finalize with shelf mark linking; append primary sources as catalog only"""
        logger.info("Finalizing response")

        error_type = state.get("error_type")

        if error_type == "NO_ANSWER_GENERATED":
            state["final_answer"] = state["draft_answer"]
            state["processing_steps"].append("Returned honest failure — no answer was generated.")
            return state

        if error_type == "NO_RELEVANT_SOURCES":
            final_answer = str(state["draft_answer"] or "")
            # A shelf-mark lookup may still have found the manuscript itself;
            # the reader should get its catalog entry even when no scholarship
            # on it was retrieved.
            catalog_entries = self._render_catalog_entries(state)
            if catalog_entries:
                final_answer = (
                    final_answer.rstrip()
                    + "\n\n---\n**Related catalog entries:**\n\n"
                    + "\n".join(catalog_entries)
                )
            state["final_answer"] = final_answer
            state["processing_steps"].append("Returned IDK response — no relevant sources.")
            return state

        if error_type == "FABRICATED_CLAIMS":
            # Repairs are exhausted. Instead of discarding the mostly verified
            # answer, deterministically remove the sentences carrying the
            # remaining hard-rejected claims and flag anything unlocatable.
            draft = str(state.get("draft_answer") or "")
            remaining = state.get("excluded_claims", [])
            locatable = [
                claim for claim in remaining
                if locate_claim_sentence(str(claim.get("text") or ""), draft) is not None
            ]
            cleaned = remove_sentences_containing(
                draft, [str(claim.get("text") or "") for claim in locatable]
            )
            unlocatable = [claim for claim in remaining if claim not in locatable]
            if unlocatable:
                state.setdefault("soft_flagged_claims", []).extend(
                    {
                        "type": claim.get("type") or "factual_claim",
                        "text": str(claim.get("text") or ""),
                        "reason": (
                            "Failed verification after all repair attempts: "
                            f"{str(claim.get('reason') or 'not supported')[:200]}"
                        ),
                    }
                    for claim in unlocatable
                )
            if len(cleaned) >= 200:
                state["draft_answer"] = cleaned
                state["error_type"] = None
                state["error"] = None
                state["processing_steps"].append(
                    f"Repair attempts exhausted; deterministically removed {len(locatable)} "
                    f"rejected sentences and kept the verified remainder"
                )
            else:
                state["final_answer"] = (
                    self._build_verified_evidence_fallback(state)
                    or self._GRACEFUL_FALLBACK
                )
                state["processing_steps"].append(
                    f"Returned verified extractive fallback after targeted repair exhaustion. "
                    f"Error: {state.get('error', '')}"
                )
                return state

        await self._resolve_unlinked_shelfmarks(state["draft_answer"], state)

        # Anchor flags BEFORE linkification: claim sentences must be matched
        # against the prose the verifier saw, not markdown-linkified text.
        final_answer = state["draft_answer"]
        soft_flags = state.get("soft_flagged_claims", [])
        if soft_flags:
            final_answer, enriched_flags = annotate_answer_with_flags(final_answer, soft_flags)
            state["soft_flagged_claims"] = enriched_flags
            anchored = sum(1 for flag in enriched_flags if flag.get("answer_span"))
            state["processing_steps"].append(
                f"Flagged {len(enriched_flags)} unverified claims for user review "
                f"({anchored} highlighted inline)"
            )

        final_answer = self._linkify_all_shelfmarks(final_answer, state)

        catalog_entries = self._render_catalog_entries(state)
        if catalog_entries:
            final_answer = (
                final_answer.rstrip()
                + "\n\n---\n**Related catalog entries:**\n\n"
                + "\n".join(catalog_entries)
            )

        # Works cited, each linked to where the reader can actually obtain it.
        # This does not depend on the synthesis model italicizing titles (it
        # frequently cites by surname alone, e.g. "(Friedman, p. 7)"), so the
        # scholarship stays reachable whatever prose style the model chooses.
        works_cited = await self._build_works_cited(state)
        if works_cited:
            final_answer = (
                final_answer.rstrip()
                + "\n\n---\n**Works cited:**\n\n"
                + "\n".join(works_cited)
            )

        # The manuscripts each cited work was built on — the primary→secondary
        # bridge that distinguishes this system (docs/DESIGN_PRECEPTS.md §1).
        work_manuscripts = state.get("work_manuscripts") or {}
        if work_manuscripts:
            sections: List[str] = []
            for title, entries in list(work_manuscripts.items())[:4]:
                links = ", ".join(
                    f"[{entry['shelf_mark']}](doc:{entry['doc_id']})"
                    for entry in entries[:10]
                )
                sections.append(f"- **{title}** — {links}")
            final_answer = (
                final_answer.rstrip()
                + "\n\n---\n**Manuscripts these works are based on:**\n\n"
                + "\n".join(sections)
            )

        # Shelf marks the scholarship cites that this collection does not hold
        # (other collections, or not yet ingested). Listing them is still useful:
        # the reader learns what to look up elsewhere, and they are already
        # recorded in the missing-fragments worklist.
        linked_canonical = {
            ShelfmarkNormalizer.to_canonical_id(mark).lower()
            for mark in state.get("shelf_mark_lookup", {})
        }
        unlinked = [
            mark for mark in sorted(state.get("shelf_marks_in_bibliography") or [])
            if ShelfmarkNormalizer.to_canonical_id(mark).lower() not in linked_canonical
        ]
        if unlinked:
            final_answer = (
                final_answer.rstrip()
                + "\n\n---\n**Also cited in these sources (not in this collection):**\n\n"
                + "\n".join(f"- {mark}" for mark in unlinked[:20])
            )
            state["processing_steps"].append(
                f"Listed {len(unlinked[:20])} cited shelf marks not held in this collection"
            )

        state["final_answer"] = final_answer
        state["processing_steps"].append("Linked shelf marks and appended catalog entries")

        return state

    @staticmethod
    def _render_catalog_entries(state: AgenticRAGState) -> List[str]:
        """Render retrieved manuscripts as linked markdown catalog entries.

        :param state: Current RAG state.
        :returns: One markdown bullet per manuscript with a shelf mark.
        :rtype: List[str]
        """
        catalog_entries: List[str] = []
        lookup = state.get("shelf_mark_lookup") or {}
        for ps in state.get("primary_source_results") or []:
            sm = ps.get("shelf_mark")
            if not sm:
                continue
            doc_id = ps.get("doc_id") or lookup.get(sm)
            title = ps.get("title") or ""
            description = ps.get("description") or ""
            entry_parts = [f"- **{sm}**" + (f": {title}" if title else "")]
            if description:
                entry_parts.append(f"  {description[:120]}")
            if doc_id:
                entry_parts[0] = f"- **[{sm}](doc:{doc_id})**" + (f": {title}" if title else "")
            catalog_entries.append("\n".join(entry_parts))
        return catalog_entries

    async def _resolve_unlinked_shelfmarks(self, text: str, state: AgenticRAGState) -> None:
        """Detect shelf marks in text that aren't yet in the lookup and search for them."""
        candidates = detect_shelfmarks(text)
        lookup = state.setdefault("shelf_mark_lookup", {})

        already_known_lower = {k.lower() for k in lookup}
        unresolved = [sm for sm in candidates if sm.lower() not in already_known_lower]

        if not unresolved:
            return

        logger.info(f"Resolving {len(unresolved)} shelf marks found in answer text: {unresolved}")

        for sm in unresolved:
            try:
                search_request = ShelfMarkSearchRequest(
                    shelf_mark=sm,
                    exact_match=False,
                    num_results=1,
                    include_embeddings=False,
                )
                response = await search_service.search_by_shelfmark(search_request)
                if not response.results:
                    missing_fragment_tracker.record(
                        shelf_mark=sm,
                        origin="answer_mention",
                        user_query=state.get("user_query"),
                    )
                    continue

                best = response.results[0]
                doc_id = best.doc_id
                actual_sm = (best.metadata.shelf_mark if best.metadata else None) or doc_id

                # Link only on canonical equivalence: fuzzy shelf-mark search
                # returns near neighbors (H3.101 for H3.111) when the cited
                # fragment is not in the index, and score-based acceptance
                # produced wrong clickable citations.
                if not shelfmarks_equivalent(sm, actual_sm):
                    missing_fragment_tracker.record(
                        shelf_mark=sm,
                        origin="answer_mention",
                        user_query=state.get("user_query"),
                        nearest_match=actual_sm,
                    )
                    continue

                lookup[sm] = doc_id
                if actual_sm and actual_sm != sm:
                    lookup[actual_sm] = doc_id

                # Also add to primary_source_results if not already present
                if not any(ps.get("doc_id") == doc_id for ps in state["primary_source_results"]):
                    meta = best.metadata
                    state["primary_source_results"].append({
                        "doc_id": doc_id,
                        "shelf_mark": actual_sm or sm,
                        "title": meta.title if meta else None,
                        "description": meta.description if meta else None,
                        "similarity_score": best.similarity_score or 0,
                        "linked_from_answer": True,
                    })
                logger.info(f"Resolved shelf mark '{sm}' → doc_id '{doc_id}'")
            except Exception as exc:
                logger.warning(f"Could not resolve shelf mark '{sm}': {exc}")

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
            text = re.sub(
                pattern,
                lambda match: f"[{match.group(0)}](doc:{doc_id})",
                text,
                flags=re.IGNORECASE,
            )

        return text

    @weave.op()
    async def list_available_models(self) -> Dict[str, Any]:
        """Fetch all downloaded models from LM Studio (loaded and unloaded).

        Uses the v0 management API (``/api/v0/models``) which lists every model
        present on disk, not only those currently loaded into memory.  Falls back
        to the OpenAI-compatible ``/v1/models`` endpoint if the v0 API is
        unavailable (older LM Studio builds).

        :returns: A dict with ``models`` (list of model id strings) and
            ``default`` (the configured synthesis model used when no override is
            selected).
        :rtype: Dict[str, Any]
        """
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Prefer v0 API which includes models not yet loaded.
            try:
                response = await client.get(f"{self.llm_studio_base_url}/api/v0/models")
                response.raise_for_status()
                data = response.json()
                models = [m["id"] for m in data.get("data", []) if m.get("id")]
                if models:
                    return {"models": models, "default": self.synthesis_model}
            except Exception:
                pass

            # Fallback: only loaded models.
            response = await client.get(f"{self.llm_studio_base_url}/v1/models")
            response.raise_for_status()
            data = response.json()

        models = [m["id"] for m in data.get("data", []) if m.get("id")]
        return {"models": models, "default": self.synthesis_model}

    @staticmethod
    def _reported_resolved_query(state: Dict[str, Any], user_query: str) -> Optional[str]:
        """Return the resolved query for the response only when it differs from the message.

        The chat UI shows a resolved query as an "Understanding:" line; a copy
        of the user's own words there is noise, so it is reported only when the
        follow-up was actually reinterpreted.

        :param state: Final pipeline state.
        :param user_query: The user's literal message.
        :returns: The standalone question, or ``None`` when unchanged.
        :rtype: Optional[str]
        """
        resolved = state.get("resolved_query")
        if resolved and resolved != user_query:
            return str(resolved)
        return None

    async def chat(
            self,
            user_query: str,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            synthesis_model: Optional[str] = None
    ) -> AgenticRAGResponse:
        """Main entry point for agentic RAG chat.

        :param user_query: The user's question.
        :param conversation_history: Prior turns for context, if any.
        :param synthesis_model: Optional LM Studio model id to use for the
            synthesis step only; ``None`` uses the configured default.
        :returns: The agentic RAG response.
        :rtype: AgenticRAGResponse
        """

        initial_state: AgenticRAGState = {
            "user_query": user_query,
            "conversation_history": normalize_conversation_history(conversation_history),
            "resolved_query": None,
            "is_followup": False,
            "synthesis_model_override": synthesis_model,
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
        }

        started_at = time.monotonic()
        request_metrics: Dict[str, Any] = {"llm_calls": []}
        metrics_token = _REQUEST_METRICS.set(request_metrics)
        try:
            final_state = await self.graph.ainvoke(initial_state)
        finally:
            _REQUEST_METRICS.reset(metrics_token)

        return AgenticRAGResponse(
            answer=final_state["final_answer"] or "Unable to generate answer",
            success=final_state["error_type"] is None,
            error_type=final_state.get("error_type"),
            resolved_query=self._reported_resolved_query(final_state, user_query),
            is_followup=bool(final_state.get("is_followup")),
            query_plan=final_state.get("query_plan"),
            bibliography_results=final_state["bibliography_results"],
            primary_source_results=final_state["primary_source_results"],
            graph_results=final_state.get("graph_results", []),
            verified_claims=final_state["verified_claims"],
            verification_summary=final_state["verification_summary"],
            flagged_claims=self._flags_to_models(final_state.get("soft_flagged_claims", [])),
            processing_steps=final_state["processing_steps"],
            metrics=self._build_metrics(final_state, started_at, request_metrics),
        )

    async def chat_stream(
            self,
            user_query: str,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            synthesis_model: Optional[str] = None
    ):
        """Streaming entry point.

        Runs the LangGraph pipeline once, yielding intermediate status events
        as each node completes, then a single ``"final"`` event with the full
        result.  The ``@weave.op()`` decorator is intentionally omitted here
        because Weave does not support async generator functions and leaks a
        ``StopAsyncIteration`` (empty message) that kills the SSE stream.

        :param user_query: The user's question.
        :param conversation_history: Prior turns for context, if any.
        :param synthesis_model: Optional LM Studio model id to use for the
            synthesis step only; ``None`` uses the configured default.
        :yields: Server-sent-event payload dicts describing pipeline progress.
        """
        initial_state: AgenticRAGState = {
            "user_query": user_query,
            "conversation_history": normalize_conversation_history(conversation_history),
            "resolved_query": None,
            "is_followup": False,
            "synthesis_model_override": synthesis_model,
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
        }

        # LangGraph's update stream fires when a node FINISHES, so the label a
        # user should see is the work that starts next — otherwise the UI reads
        # "Fetching manuscripts" for the minutes that synthesis is running.
        node_status_map = {
            "resolve_query": "Planning search strategy...",
            "route_query": "Searching scholarly sources...",
            "execute_searches": "Fetching manuscripts mentioned by scholars...",
            "link_primary_secondary": "Synthesizing scholarly analysis...",
            "synthesize_answer": "Verifying claims...",
            "verify_claims": "Reviewing verification results...",
            "repair_answer": "Verifying revised claims...",
            "finalize_response": "Finalizing response..."
        }
        initial_status = "Understanding your question..."

        # Accumulate state deltas so we can build the final response without
        # a second pipeline invocation.
        accumulated: dict = dict(initial_state)
        request_metrics: Dict[str, Any] = {"llm_calls": []}
        # Set before the producer task is created so the task inherits it.
        metrics_token = _REQUEST_METRICS.set(request_metrics)

        import asyncio

        # A pipeline stage can run for minutes (a large model generating a
        # synthesis). Without traffic in between, intermediaries treat the
        # stream as idle and close it, and the user sees nothing happening.
        # A producer task lets this generator emit heartbeats while waiting.
        event_queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        async def _produce() -> None:
            try:
                async for event in self.graph.astream(initial_state, stream_mode="updates"):
                    await event_queue.put(event)
            except Exception as exc:  # surfaced to the client below
                await event_queue.put(exc)
            finally:
                await event_queue.put(_DONE)

        producer = asyncio.create_task(_produce())
        started_at = time.monotonic()
        current_stage = initial_status
        heartbeat_seconds = float(os.getenv("CHAT_STREAM_HEARTBEAT_SECONDS", "8"))

        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=heartbeat_seconds)
                except asyncio.TimeoutError:
                    queue_state = lm_studio_gateway.snapshot()
                    yield {
                        "type": "progress",
                        "status": current_stage,
                        "elapsed_seconds": round(time.monotonic() - started_at, 1),
                        "model_queue": queue_state,
                        # Any request ahead of this one delays it; say so rather
                        # than letting the UI look stalled.
                        "queued_behind": queue_state["waiting"],
                    }
                    continue

                if event is _DONE:
                    break
                if isinstance(event, Exception):
                    raise event

                for node_name, updates in event.items():
                    accumulated.update(updates)
                    status = node_status_map.get(node_name, f"Processing {node_name}...")
                    current_stage = status

                    query_plan = updates.get("query_plan")
                    if query_plan and hasattr(query_plan, 'dict'):
                        query_plan_data = query_plan.dict()
                    elif query_plan and hasattr(query_plan, 'model_dump'):
                        query_plan_data = query_plan.model_dump()
                    else:
                        query_plan_data = query_plan

                    yield {
                        "type": "status",
                        "status": status,
                        "node": node_name,
                        "elapsed_seconds": round(time.monotonic() - started_at, 1),
                        "query_plan": query_plan_data,
                        "processing_steps": list(updates.get("processing_steps") or [])[-3:],
                        "bibliography_count": len(updates.get("bibliography_results", [])),
                        "primary_count": len(updates.get("primary_source_results", [])),
                        "graph_count": len(updates.get("graph_results", [])),
                        "verified_claims_count": len(updates.get("verified_claims", []))
                    }
        finally:
            if not producer.done():
                producer.cancel()
            _REQUEST_METRICS.reset(metrics_token)

        final_result = AgenticRAGResponse(
            answer=accumulated.get("final_answer") or "Unable to generate answer",
            success=accumulated.get("error_type") is None,
            error_type=accumulated.get("error_type"),
            resolved_query=self._reported_resolved_query(accumulated, user_query),
            is_followup=bool(accumulated.get("is_followup")),
            query_plan=accumulated.get("query_plan"),
            bibliography_results=accumulated.get("bibliography_results", []),
            primary_source_results=accumulated.get("primary_source_results", []),
            graph_results=accumulated.get("graph_results", []),
            verified_claims=accumulated.get("verified_claims", []),
            verification_summary=accumulated.get("verification_summary", {}),
            flagged_claims=self._flags_to_models(accumulated.get("soft_flagged_claims", [])),
            processing_steps=accumulated.get("processing_steps", []),
            metrics=self._build_metrics(accumulated, started_at, request_metrics),
        )
        if hasattr(final_result, 'model_dump'):
            result_data = final_result.model_dump()
        else:
            result_data = final_result.dict()
        yield {"type": "final", "data": result_data}


# Global service instance
agentic_rag_service = AgenticRAGService()
