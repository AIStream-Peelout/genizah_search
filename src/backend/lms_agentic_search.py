# agentic_rag_service.py
"""
LangGraph-based agentic RAG service for Cairo Genizah collection.

CORE PRINCIPLE: Scholarly synthesis first, with embedded manuscript references.
Primary sources are supplementary evidence, not the main content.
"""

import os
import re
import logging
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Literal, TypedDict, Set, Callable, TypeVar
from pydantic import BaseModel, Field, ValidationError
import json
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from src.backend.shelfmark_normalizer import detect_shelfmarks, ShelfmarkNormalizer

from langgraph.graph import StateGraph, END

from src.backend.search_service import search_service, SearchRequest, DocumentMetadata
from src.backend.search_bibliography import bibliography_search_service, BibliographyHybridSearchRequest
from src.backend.neo4j_service import neo4j_service
logger = logging.getLogger(__name__)

Operation = TypeVar("Operation", bound=Callable[..., Any])


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


# ============================================================================
# LangGraph State
# ============================================================================

class AgenticRAGState(TypedDict):
    """State for the agentic RAG graph"""
    user_query: str
    conversation_history: Optional[List[Dict[str, str]]]
    synthesis_model_override: Optional[str]  # Per-request synthesis model chosen in the UI (None = use default)
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
    supported_evidence_units: List[Dict[str, Any]]
    verification_feedback_history: List[Dict[str, Any]]


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
        doc_id = result.get("doc_id")
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
                for sample in (work.get("referenced_fragment_samples") or [])[:5]
                if sample.get("shelfmark")
            ]
            sample_text = f"; sample shelf marks: {', '.join(samples)}" if samples else ""
            lines.append(
                f"- {title}{year}; article_id={work.get('article_id')}; "
                f"references {count} distinct Fragment nodes{sample_text}"
            )

        studied_samples = [
            sample.get("shelfmark")
            for sample in (evidence.get("studied_fragment_samples") or [])[:10]
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
) -> List[Dict[str, Any]]:
    """Build one bounded evidence representation shared by synthesis and verification.

    Keeping the prompt text identical prevents the synthesizer from seeing text
    that is later truncated away before verification.

    :param bibliography_results: Retrieved bibliography result dictionaries.
    :param limit: Maximum number of bibliography chunks to include.
    :returns: Source dictionaries containing citation and prompt text.
    :rtype: List[Dict[str, Any]]
    """
    sources: List[Dict[str, Any]] = []
    for source_number, bibliography in enumerate(bibliography_results[:limit], start=1):
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
        shelf_marks = bibliography.get("shelf_marks_mentioned") or []
        if description:
            parts.append(
                "Generated catalog summary (orientation only; never quote): "
                f"{description[:500]}"
            )
        if quoteable_text:
            parts.append(f"Original page text (quoteable): {quoteable_text[:1800]}")
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
            "quoteable_text": quoteable_text[:1800],
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
    normalized = normalized.translate(str.maketrans({"“": '"', "”": '"', "’": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", normalized).strip()


def extract_direct_quotes(text: str) -> List[str]:
    """Extract substantive text enclosed in straight or curly double quotes.

    :param text: Draft answer text.
    :returns: Deduplicated quote bodies in appearance order.
    :rtype: List[str]
    """
    matches = re.findall(r'"([^"\n]{8,})"|“([^”\n]{8,})”', text or "")
    quotes: List[str] = []
    seen: Set[str] = set()
    for straight_quote, curly_quote in matches:
        quote = (straight_quote or curly_quote).strip()
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


def split_sentences(paragraph: str) -> List[str]:
    """Split a paragraph into sentences with a conservative boundary rule.

    :param paragraph: Single paragraph of answer text (no blank lines).
    :returns: Sentences in order; the original text is recoverable modulo
        inter-sentence whitespace.
    :rtype: List[str]
    """
    stripped = paragraph.strip()
    if not stripped:
        return []
    return [sentence for sentence in _SENTENCE_BOUNDARY_REGEX.split(stripped) if sentence.strip()]


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

        self.graph = self._build_graph()

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

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("route_query", self._route_query_node)
        workflow.add_node("execute_searches", self._execute_searches_node)
        workflow.add_node("link_primary_secondary", self._link_primary_secondary_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("repair_answer", self._repair_answer_node)
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
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            if response.is_error:
                logger.error(
                    "LM Studio returned %s for model '%s': %s",
                    response.status_code, model, response.text
                )
            response.raise_for_status()
            return response.json()

    @weave.op()
    async def _call_llm(
            self,
            messages: List[Dict[str, str]],
            model: str,
            temperature: float = 0.7,
            response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call LM Studio without tools, auto-loading the model if not yet loaded.

        :param messages: Chat messages to send.
        :param model: LM Studio model identifier.
        :param temperature: Sampling temperature.
        :param response_format: Optional OpenAI-style ``response_format`` payload
            (e.g. a ``json_schema`` spec) enforcing structured output.
        :returns: The model's response content string.
        :rtype: str
        """
        url = f"{self.llm_studio_base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096
        }
        if self.model_ttl_seconds > 0:
            payload["ttl"] = self.model_ttl_seconds
        if response_format is not None:
            payload["response_format"] = response_format

        import httpx
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            if response.is_error:
                logger.error(
                    "LM Studio returned %s for model '%s': %s",
                    response.status_code, model, response.text
                )
            response.raise_for_status()
            result = response.json()

        return result["choices"][0]["message"]["content"]

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

"T-S 8.133"
→ `primary_shelfmark: "T-S 8.133"` + `bibliography_hybrid: "T-S 8.133"`
→ Reasoning: specific shelf mark lookup

"What do we know about Yom Kippur liturgy"
→ `bibliography_hybrid` (keyword_weight: 70) ONLY
→ Reasoning: broad topical query

**Critical Rules:**
- Default to 1-2 bibliography searches
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
        if state.get("conversation_history"):
            history_context = "\n\n**Conversation History:**\n"
            for turn in state["conversation_history"][-8:]:
                if isinstance(turn, dict):
                    if "role" in turn and "content" in turn:
                        role = str(turn.get("role") or "user").capitalize()
                        history_context += f"{role}: {str(turn.get('content') or '')[:500]}\n"
                        continue
                    content = turn.get("user_query", "")
                    answer = turn.get("answer", "")
                else:
                    role = getattr(turn, "role", None)
                    message_content = getattr(turn, "content", None)
                    if role is not None and message_content is not None:
                        history_context += f"{str(role).capitalize()}: {str(message_content)[:500]}\n"
                        continue
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

        response = await self._call_llm_with_tools(
            messages=messages,
            tools=tools,
            model=self.router_model,
            tool_choice="required"
        )

        query_plan: Optional[QueryPlan] = None
        try:
            tool_calls = response["choices"][0]["message"].get("tool_calls") or []
            if tool_calls:
                tool_call = tool_calls[0]
                function_name = tool_call["function"].get("name")
                arguments = json.loads(tool_call["function"]["arguments"])
                if function_name == "create_search_plan":
                    query_plan = QueryPlan(**arguments)
                elif function_name in DIRECT_SEARCH_ACTION_TYPES:
                    direct_arguments = dict(arguments)
                    direct_arguments["search_type"] = function_name
                    direct_action = SearchAction(**direct_arguments)
                    query_plan = QueryPlan(
                        actions=[direct_action],
                        needs_primary_secondary_linking=True,
                        is_followup=bool(state.get("conversation_history")),
                        reasoning=(
                            f"Normalized the planner's direct {function_name} action "
                            "into a complete query plan"
                        ),
                    )
                else:
                    logger.warning("Router returned unexpected tool function: %s", function_name)
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Router returned an invalid search plan; using fallback: %s", exc)

        if query_plan is None:
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

                try:
                    author_response = await bibliography_search_service.search_by_author(
                        author_name=canonical_name,
                        query=state["user_query"],
                        num_results=8,
                    )
                    author_results = [
                        bibliography_result_to_dict(result)
                        for result in author_response.results
                    ]
                    bibliography_results.extend(author_results)
                    for result in author_results:
                        if result.get("shelf_marks_mentioned"):
                            shelf_marks_in_bibliography.update(result["shelf_marks_mentioned"])
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
                bibliography_results = deduplicate_bibliography_results(fallback_results)
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

        if not bibliography_results and not graph_results:
            state["error_type"] = "NO_RELEVANT_SOURCES"
            state["error"] = "Neither bibliography nor graph retrieval returned evidence."
            state["processing_steps"].append(
                "No bibliography or graph evidence found; synthesis will return an explicit limitation."
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
   Format quotes as: Author, p. X: "quote text"
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
9. Do not discuss the retrieval system, missing source categories, prompts, or the absence of
   knowledge-graph evidence. Answer only with the scholarly evidence that is actually present."""

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

        user_message = f"""{system_prompt}{exclusion_note}

RETRIEVED SCHOLARLY SOURCES:

{chr(10).join(bib_context) if bib_context else "No scholarly sources retrieved."}
{graph_section}

USER QUERY:
{state['user_query']}

Provide your scholarly synthesis. Use only the retrieved source text and structured graph
evidence above, following their distinct provenance rules."""

        messages = [{"role": "user", "content": user_message}]

        # Allow a per-request synthesis model override (chosen in the UI for testing);
        # fall back to the configured default when none is supplied.
        synthesis_model = state.get("synthesis_model_override") or self.synthesis_model
        logger.info(f"Synthesizing with model: {synthesis_model}")
        draft_answer = await self._call_llm(
            messages=messages,
            model=synthesis_model,
            temperature=0.2
        )

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
            )
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            for revision in json.loads(clean).get("revisions", []):
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
        if state.get("error_type") == "NO_RELEVANT_SOURCES":
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
                "\nItems rejected from the previous draft are listed below. If an item or "
                "equivalent claim remains, mark the complete proposition NOT_SUPPORTED:\n"
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

For bibliography sources, generated catalog summaries are orientation aids and are not
sufficient evidence by themselves. Factual support must be traceable to text labeled
"Original page text (quoteable)" or to the source's citation line and title. Neo4j records
may support graph claims only.

Do not separately return direct quotes or shelf-mark identifiers; those are checked by
deterministic matchers. You must still verify the surrounding proposition containing them.
{exclusion_note}

SOURCE CHUNKS:
{sources_text}

DRAFT ANSWER:
{draft}

The type must be attribution, factual_claim, graph_claim, or citation_claim.
source_number must be an integer for supported claims and null otherwise.
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
                verdict = "unsupported"
                reasoning = "Verifier marked this supported but supplied no valid source number"

            if verdict == "supported":
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
            )
            try:
                clean = raw_response.strip()
                if clean.startswith("```"):
                    clean = re.sub(r"^```[a-z]*\n?", "", clean)
                    clean = re.sub(r"\n?```$", "", clean)
                verification_result = json.loads(clean)
                raw_claims = verification_result.get("verified_claims", [])
                if not isinstance(raw_claims, list) or not all(
                    isinstance(claim, dict) for claim in raw_claims
                ):
                    raise ValueError("verified_claims must be a list of objects")
                return raw_claims, str(verification_result.get("summary") or ""), None
            except (AttributeError, json.JSONDecodeError, TypeError, ValueError) as exc:
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

        if error_type == "NO_RELEVANT_SOURCES":
            state["final_answer"] = state["draft_answer"]
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

        final_answer = self._linkify_all_shelfmarks(state["draft_answer"], state)

        soft_flags = state.get("soft_flagged_claims", [])
        if soft_flags:
            final_answer, enriched_flags = annotate_answer_with_flags(final_answer, soft_flags)
            state["soft_flagged_claims"] = enriched_flags
            anchored = sum(1 for flag in enriched_flags if flag.get("answer_span"))
            state["processing_steps"].append(
                f"Flagged {len(enriched_flags)} unverified claims for user review "
                f"({anchored} highlighted inline)"
            )

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
                    continue

                best = response.results[0]
                doc_id = best.doc_id
                actual_sm = (best.metadata.shelf_mark if best.metadata else None) or doc_id

                # Verify it's a genuine match (normalise query so "Box" etc. don't break substring check)
                score = best.similarity_score or 0
                norm_q = ShelfmarkNormalizer.normalize_for_search(sm).lower()
                norm_d = (actual_sm or "").strip().lower()
                if score > 0.5 or norm_q in norm_d or norm_d in norm_q:
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
                            "similarity_score": score,
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
            "conversation_history": conversation_history,
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
            "supported_evidence_units": [],
            "verification_feedback_history": [],
        }

        final_state = await self.graph.ainvoke(initial_state)

        return AgenticRAGResponse(
            answer=final_state["final_answer"] or "Unable to generate answer",
            success=final_state["error_type"] is None,
            error_type=final_state.get("error_type"),
            query_plan=final_state.get("query_plan"),
            bibliography_results=final_state["bibliography_results"],
            primary_source_results=final_state["primary_source_results"],
            graph_results=final_state.get("graph_results", []),
            verified_claims=final_state["verified_claims"],
            verification_summary=final_state["verification_summary"],
            flagged_claims=self._flags_to_models(final_state.get("soft_flagged_claims", [])),
            processing_steps=final_state["processing_steps"]
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
            "conversation_history": conversation_history,
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
            "supported_evidence_units": [],
            "verification_feedback_history": [],
        }

        node_status_map = {
            "route_query": "Planning search strategy...",
            "execute_searches": "Searching scholarly sources...",
            "link_primary_secondary": "Fetching manuscripts mentioned by scholars...",
            "synthesize_answer": "Synthesizing scholarly analysis...",
            "verify_claims": "Verifying claims...",
            "finalize_response": "Finalizing response..."
        }

        # Accumulate state deltas so we can build the final response without
        # a second pipeline invocation.
        accumulated: dict = dict(initial_state)

        async for event in self.graph.astream(initial_state, stream_mode="updates"):
            for node_name, updates in event.items():
                accumulated.update(updates)
                status = node_status_map.get(node_name, f"Processing {node_name}...")

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
                    "query_plan": query_plan_data,
                    "bibliography_count": len(updates.get("bibliography_results", [])),
                    "primary_count": len(updates.get("primary_source_results", [])),
                    "graph_count": len(updates.get("graph_results", [])),
                    "verified_claims_count": len(updates.get("verified_claims", []))
                }

        final_result = AgenticRAGResponse(
            answer=accumulated.get("final_answer") or "Unable to generate answer",
            success=accumulated.get("error_type") is None,
            error_type=accumulated.get("error_type"),
            query_plan=accumulated.get("query_plan"),
            bibliography_results=accumulated.get("bibliography_results", []),
            primary_source_results=accumulated.get("primary_source_results", []),
            graph_results=accumulated.get("graph_results", []),
            verified_claims=accumulated.get("verified_claims", []),
            verification_summary=accumulated.get("verification_summary", {}),
            flagged_claims=self._flags_to_models(accumulated.get("soft_flagged_claims", [])),
            processing_steps=accumulated.get("processing_steps", [])
        )
        if hasattr(final_result, 'model_dump'):
            result_data = final_result.model_dump()
        else:
            result_data = final_result.dict()
        yield {"type": "final", "data": result_data}


# Global service instance
agentic_rag_service = AgenticRAGService()
