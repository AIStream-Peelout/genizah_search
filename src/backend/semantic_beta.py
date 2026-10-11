"""Opt-in semantic beta search: embedder routing and the beta contract check.

Since 2026-10-11 the beta is ``genizah_merged_v10`` (text representation
``semantic_v4``), embedded by the fine-tuned model revision that the
``embedding-v9`` compose service serves; the v9 index was abandoned. The
service and the ``BETA_*`` environment names keep their v9-era names.

The default index (``ELASTICSEARCH_INDEX``, genizah_merged_v8) keeps the stock
embedder and today's code path untouched: :meth:`SemanticRouter.resolve` hands
it the default client without reading Elasticsearch. Every *other* index a
semantic or hybrid request targets is routed by the embedding contract
recorded in its mapping ``_meta`` (``model_name``/``model_revision``, or the
older ``embedding_model``/``embedding_model_revision`` keys):

* same model@revision as the default index -> the default embedder;
* the beta index (``BETA_SEMANTIC_INDEX``) or its model@revision -> the beta
  embedder, and only while :class:`SemanticBeta` reports it verified;
* anything else, including no recorded model -> :class:`SemanticRouteError`
  (409/503), never a search over a mismatched vector space.

Indices whose mapping has a ``semantic_eligible`` field hold records without
a vector; every vector leg over them must be restricted to
``semantic_eligible: true`` (see :func:`semantic_eligible_clause`).

Shelf marks were masked out of the beta vectors, so a semantic or hybrid
query on the beta index that is just a shelf mark, a shelf-mark prefix or a
volume ("T-S 13J", "Mosseri VII") is answered by shelf-mark search, unless
shelf-mark search finds nothing.

The contract check is non-fatal and runs in the background: v8 serving never
waits for or depends on it. See ``docs/v9_semantic_beta.md``.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import httpx
import numpy as np
from elasticsearch import NotFoundError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.backend.embedding_client import EmbeddingClient
from src.backend.shelfmark_normalizer import SHELFMARK_PATTERN

logger = logging.getLogger(__name__)

SEMANTIC_ELIGIBLE_FIELD = "semantic_eligible"
DEFAULT_BETA_SEMANTIC_INDEX = "genizah_merged_v10"
DEFAULT_BETA_TEXT_REPRESENTATION_VERSION = "semantic_v4"
CANARY_MIN_COSINE = 0.99
# Index mappings change only on re-index; a short TTL keeps routing to one
# small cached read per index instead of one per request.
INDEX_CONTRACT_TTL_SECONDS = 300.0
# Background check cadence: back off while the beta embedder is missing or
# still loading, re-verify periodically once it serves.
RETRY_INITIAL_SECONDS = 15.0
RETRY_MAX_SECONDS = 300.0
RECHECK_INTERVAL_SECONDS = 1800.0
# Lower bound between an on-demand re-check (after a failed beta request) and
# the previous check, so a burst of failures cannot hammer the embedder.
MIN_CHECK_SPACING_SECONDS = 10.0
# (stored canary key, key of its text, encoding mode of its stored vector)
BETA_CANARIES: Tuple[Tuple[str, str, str], ...] = (
    ("canary", "string", "document"),
    ("query_canary", "string", "query"),
    ("document_canary", "text", "document"),
)
# Trailing punctuation dropped before deciding a query is a bare shelf mark.
_SHELFMARK_TRAILING = ".,;:"
# A Roman-numeral volume ("Mosseri VII", "Mosseri Ia", "mosseri iv") is the
# number of a shelf-mark prefix just as a digit is. Mosseri volumes are I-X,
# some with an "a" part (Ia, IIIa, Va, VIa, IXa, Xa); no other suffix is
# accepted, so words such as "Is", "It", "In" or "Id" never count.
_ROMAN_VOLUME = re.compile(r"\b(?:[IVX]+a?|[ivx]+a?)\b")
_MOSSERI_START = re.compile(r"(?i)(?:moss\.|mosseri)\s")

Identity = Tuple[str, str]


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    :param name: Variable name.
    :param default: Value when the variable is unset or blank.
    :return: ``True`` for 1/true/yes/on (any case), ``False`` otherwise.
    :rtype: bool
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def semantic_eligible_clause() -> Dict[str, Any]:
    """Filter clause keeping only records that carry a vector.

    :return: A ``term`` clause on ``semantic_eligible: true``.
    :rtype: Dict[str, Any]
    """
    return {"term": {SEMANTIC_ELIGIBLE_FIELD: True}}


def looks_like_shelfmark(query: str) -> bool:
    """Whether a query is a bare shelf mark, shelf-mark prefix or volume (and nothing else).

    Reuses the corpus-wide detector :data:`SHELFMARK_PATTERN`, which already
    accepts prefixes such as "T-S 13J", "T-S NS 324" or "ENA 2558"; a shelf
    mark mentioned inside a longer question does not count. Every real shelf
    mark or volume carries a number, so a digit or a Roman-numeral volume
    ("Mosseri VII") is also required: the pattern's Mosseri / ``Moss.`` /
    ``L-G`` branches accept any word after the prefix, and topical queries such
    as "Mosseri collection" or "L-G Arabic" must stay semantic.

    :param query: Raw search query.
    :return: ``True`` when the whole query is one shelf mark, prefix or volume.
    :rtype: bool
    """
    candidate = (query or "").strip().rstrip(_SHELFMARK_TRAILING).strip()
    if _MOSSERI_START.match(candidate):
        # The catalogue writes Mosseri marks with a comma ("Moss. VII,5"),
        # where the shared pattern expects a period.
        candidate = candidate.replace(",", ".")
    has_number = any(ch.isdigit() for ch in candidate) or _ROMAN_VOLUME.search(candidate) is not None
    return bool(candidate) and has_number and SHELFMARK_PATTERN.fullmatch(candidate) is not None


class SemanticRouteError(HTTPException):
    """A semantic request that must not run (unknown index, wrong embedder, beta down).

    Subclasses :class:`HTTPException` so it reaches the client with its own
    status; the search handlers re-raise it instead of wrapping it in a 500.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        """Build the error.

        :param status_code: HTTP status for the client (404/409/503).
        :param detail: Human-readable explanation.
        """
        super().__init__(status_code=status_code, detail=detail)


@dataclass(frozen=True)
class IndexContract:
    """Embedding contract an index records in its mapping ``_meta``."""

    index: str
    model: Optional[str] = None
    revision: Optional[str] = None
    text_representation_version: Optional[str] = None
    eligibility_mapped: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def identity(self) -> Optional[Identity]:
        """``(model, revision)`` when both are recorded.

        :return: The embedder identity, or ``None``.
        :rtype: Optional[Identity]
        """
        if self.model and self.revision:
            return (self.model, self.revision)
        return None

    def describe(self) -> str:
        """Short ``model@revision`` label for messages.

        :return: The label, or a note that no model is recorded.
        :rtype: str
        """
        if self.identity is None:
            return "no recorded embedding model"
        return f"{self.model}@{str(self.revision)[:12]}"


def _contract_from_body(index: str, mappings: Mapping[str, Any]) -> IndexContract:
    """Build one contract from the ``mappings`` body of a single index.

    :param index: Index name the request used.
    :param mappings: ``{"_meta": ..., "properties": ...}`` of one index.
    :return: The parsed contract.
    :rtype: IndexContract
    """
    meta = mappings.get("_meta") or {}
    properties = mappings.get("properties") or {}
    return IndexContract(
        index=index,
        model=meta.get("model_name") or meta.get("embedding_model"),
        revision=meta.get("model_revision") or meta.get("embedding_model_revision"),
        text_representation_version=meta.get("text_representation_version"),
        eligibility_mapped=SEMANTIC_ELIGIBLE_FIELD in properties,
        meta=meta,
    )


def contract_from_mapping(index: str, mapping: Mapping[str, Any]) -> IndexContract:
    """Parse a ``get_mapping`` response into one embedding contract.

    A name resolving to several concrete indices (alias, wildcard, comma
    list) is accepted only when they all share one contract.

    :param index: Index name the request used.
    :param mapping: ``es.indices.get_mapping(index=index)`` response.
    :return: The shared contract.
    :rtype: IndexContract
    :raises SemanticRouteError: 404 when nothing matched, 409 when the
        concrete indices disagree.
    """
    contracts = {_contract_from_body(index, body.get("mappings") or {}) for body in mapping.values()}
    if not contracts:
        raise SemanticRouteError(404, f"Index '{index}' not found")
    if len(contracts) > 1:
        raise SemanticRouteError(
            409,
            f"'{index}' resolves to indices with different embedding contracts; "
            "semantic search needs a single index.",
        )
    return contracts.pop()


def fetch_index_contract(es: Any, index: str) -> IndexContract:
    """Read an index's embedding contract from Elasticsearch (uncached).

    :param es: Synchronous Elasticsearch client.
    :param index: Index name.
    :return: The parsed contract.
    :rtype: IndexContract
    :raises SemanticRouteError: 404 when the index does not exist.
    """
    try:
        mapping = es.indices.get_mapping(index=index)
    except NotFoundError as exc:
        raise SemanticRouteError(404, f"Index '{index}' not found") from exc
    return contract_from_mapping(index, mapping)


def count_eligible_documents(es: Any, index: str) -> int:
    """Count the records of an index that carry a vector.

    :param es: Synchronous Elasticsearch client.
    :param index: Index with a ``semantic_eligible`` field.
    :return: Number of top-level records with ``semantic_eligible: true``.
    :rtype: int
    """
    return int(es.count(index=index, query=semantic_eligible_clause())["count"])


class IndexContractCache:
    """TTL cache of :class:`IndexContract` per index name."""

    def __init__(
        self,
        ttl_seconds: float = INDEX_CONTRACT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty cache.

        :param ttl_seconds: How long a read stays valid.
        :param clock: Monotonic clock (injectable for tests).
        """
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: Dict[str, Tuple[float, IndexContract]] = {}

    def get(self, es: Any, index: str) -> IndexContract:
        """Return the cached contract, reading the mapping when stale.

        :param es: Synchronous Elasticsearch client.
        :param index: Index name.
        :return: The contract.
        :rtype: IndexContract
        """
        cached = self._entries.get(index)
        if cached is not None and self._clock() - cached[0] < self._ttl:
            return cached[1]
        return self.put(index, fetch_index_contract(es, index))

    def put(self, index: str, contract: IndexContract) -> IndexContract:
        """Store a freshly read contract.

        :param index: Index name.
        :param contract: Contract to cache.
        :return: The same contract.
        :rtype: IndexContract
        """
        self._entries[index] = (self._clock(), contract)
        return contract


class SemanticBetaStatus(BaseModel):
    """Payload of ``GET /semantic-beta``."""

    available: bool
    index: str
    model: Optional[str] = None
    revision: Optional[str] = None
    text_representation_version: Optional[str] = None
    eligible_documents: Optional[int] = None
    reason: Optional[str] = None


class SemanticBeta:
    """State and contract check of the opt-in beta embedder.

    ``available`` turns true only after every check passed: the beta is
    enabled; the beta index ``_meta`` records the expected text representation
    version (``semantic_v4`` for genizah_merged_v10) and a model; the
    embedder's ``/health`` reports that exact model@revision; and it
    reproduces the ``canary`` (document mode), ``query_canary`` (query mode)
    and ``document_canary`` (document mode) vectors with cosine > 0.99.
    """

    def __init__(
        self,
        *,
        index: str,
        client: EmbeddingClient,
        enabled: bool,
        contracts: IndexContractCache,
        text_representation_version: str = DEFAULT_BETA_TEXT_REPRESENTATION_VERSION,
        retry_initial_seconds: float = RETRY_INITIAL_SECONDS,
        retry_max_seconds: float = RETRY_MAX_SECONDS,
        recheck_interval_seconds: float = RECHECK_INTERVAL_SECONDS,
        min_check_spacing_seconds: float = MIN_CHECK_SPACING_SECONDS,
    ) -> None:
        """Create the beta in the "not yet verified" state.

        :param index: Beta index name.
        :param client: Client of the beta embedding service.
        :param enabled: ``SEMANTIC_BETA_ENABLED``.
        :param contracts: Shared index-contract cache (refreshed by each check).
        :param text_representation_version: ``text_representation_version`` the
            beta index ``_meta`` must record.
        :param retry_initial_seconds: First retry delay while unavailable.
        :param retry_max_seconds: Cap of the doubling retry delay.
        :param recheck_interval_seconds: Re-verification interval while available.
        :param min_check_spacing_seconds: Delay before an on-demand re-check.
        """
        self.index = index
        self.client = client
        self.enabled = enabled
        self._contracts = contracts
        self.text_representation_version = text_representation_version
        self._retry_initial = retry_initial_seconds
        self._retry_max = retry_max_seconds
        self._recheck_interval = recheck_interval_seconds
        self._min_check_spacing = min_check_spacing_seconds
        self.available = False
        self.reason: Optional[str] = (
            "contract check has not completed yet" if enabled
            else "semantic beta is disabled (SEMANTIC_BETA_ENABLED=false)"
        )
        self.contract: Optional[IndexContract] = None
        self.eligible_documents: Optional[int] = None
        self._task: Optional[asyncio.Task] = None
        self._wake: Optional[asyncio.Event] = None

    @property
    def identity(self) -> Optional[Identity]:
        """Model@revision the beta index records (from the last check).

        :return: The identity, or ``None`` before the mapping was read.
        :rtype: Optional[Identity]
        """
        return self.contract.identity if self.contract else None

    def claims(self, index: str, contract: IndexContract) -> bool:
        """Whether a target index belongs to the beta embedder.

        :param index: Requested index name.
        :param contract: Its contract.
        :return: ``True`` for the beta index or an index with its model@revision.
        :rtype: bool
        """
        return index == self.index or (contract.identity is not None and contract.identity == self.identity)

    def serves(self, identity: Optional[Identity]) -> bool:
        """Whether the verified beta embedder can embed queries for an identity.

        :param identity: Target index model@revision.
        :return: ``True`` only while available and the identities match.
        :rtype: bool
        """
        return self.enabled and self.available and identity is not None and identity == self.identity

    def status(self) -> SemanticBetaStatus:
        """Current state for ``GET /semantic-beta``.

        :return: The status payload.
        :rtype: SemanticBetaStatus
        """
        contract = self.contract
        return SemanticBetaStatus(
            available=self.available,
            index=self.index,
            model=contract.model if contract else None,
            revision=contract.revision if contract else None,
            text_representation_version=contract.text_representation_version if contract else None,
            eligible_documents=self.eligible_documents,
            reason=None if self.available else self.reason,
        )

    async def check(self, es: Any) -> bool:
        """Run the contract check once and record the outcome.

        Never raises: Elasticsearch or embedder failures become the reason.

        :param es: Synchronous Elasticsearch client.
        :return: Whether the beta is available afterwards.
        :rtype: bool
        """
        if not self.enabled:
            self._record("semantic beta is disabled (SEMANTIC_BETA_ENABLED=false)")
            return False
        try:
            problem = await self._evaluate(es)
        except Exception as exc:  # I/O boundary: a failed check must never escape the background task
            problem = f"contract check could not complete: {type(exc).__name__}: {exc}"
        self._record(problem)
        return self.available

    async def _evaluate(self, es: Any) -> Optional[str]:
        """Run every check in order, stopping at the first problem.

        :param es: Synchronous Elasticsearch client.
        :return: The first problem found, or ``None`` when all checks pass.
        :rtype: Optional[str]
        """
        contract = await asyncio.to_thread(fetch_index_contract, es, self.index)
        self._contracts.put(self.index, contract)
        self.contract = contract
        if contract.eligibility_mapped:
            self.eligible_documents = await asyncio.to_thread(count_eligible_documents, es, self.index)
        if contract.identity is None:
            return f"{self.index}: mapping _meta records no model_name/model_revision"
        if contract.text_representation_version != self.text_representation_version:
            return (
                f"{self.index}: text_representation_version is "
                f"{contract.text_representation_version!r}, expected {self.text_representation_version!r}"
            )
        health = await self.client.get_health()
        served = (health.get("model"), health.get("revision"))
        if served != contract.identity:
            return (
                f"beta embedder serves {served[0]}@{served[1]}, but {self.index} was built with "
                f"{contract.model}@{contract.revision}"
            )
        for label, text_key, mode in BETA_CANARIES:
            problem = await self._verify_canary(contract.meta, label, text_key, mode)
            if problem:
                return problem
        return None

    async def _verify_canary(self, meta: Mapping[str, Any], label: str, text_key: str, mode: str) -> Optional[str]:
        """Check that the embedder reproduces one stored canary vector.

        Every measured cosine is logged at INFO, passing or not, so each check
        can be reported from the backend log.

        :param meta: Beta index ``_meta``.
        :param label: Canary key (``canary``, ``query_canary``, ``document_canary``).
        :param text_key: Key of the canary's text.
        :param mode: Encoding mode of the stored vector.
        :return: A problem description, or ``None`` when cosine > 0.99.
        :rtype: Optional[str]
        """
        canary = meta.get(label) or {}
        text, vector = canary.get(text_key), canary.get("vector")
        if not text or not vector:
            return f"{self.index}: mapping _meta has no usable {label}"
        cosine = await self.client.canary_cosine(text, vector, mode=mode)
        logger.info("Semantic beta %s %s cosine %.6f (%s mode)", self.index, label, cosine, mode)
        if not cosine > CANARY_MIN_COSINE:  # also rejects NaN
            return f"{self.index}: {label} cosine {cosine:.4f} <= {CANARY_MIN_COSINE} ({mode} mode)"
        return None

    def _record(self, problem: Optional[str]) -> None:
        """Store a check outcome, logging only transitions and new reasons.

        :param problem: ``None`` when available, else why not.
        """
        was_available, previous = self.available, self.reason
        self.available = problem is None
        self.reason = problem
        if self.available and not was_available:
            logger.info(
                "Semantic beta available: %s via %s (%s eligible documents)",
                self.index, self.contract.describe() if self.contract else "?", self.eligible_documents,
            )
        elif not self.available and (was_available or problem != previous):
            logger.warning("Semantic beta unavailable: %s", problem)

    def mark_unavailable(self, reason: str) -> None:
        """Take the beta offline after a failed request and schedule a re-check.

        :param reason: Why the beta embedder failed.
        """
        self._record(reason)
        if self._wake is not None:
            self._wake.set()

    async def run(self, es: Any) -> None:
        """Background loop: check, then re-check with backoff or periodically.

        :param es: Synchronous Elasticsearch client.
        """
        self._wake = asyncio.Event()
        retry_delay = self._retry_initial
        while True:
            if await self.check(es):
                delay, retry_delay = self._recheck_interval, self._retry_initial
            elif not self.enabled:
                return
            else:
                delay, retry_delay = retry_delay, min(retry_delay * 2, self._retry_max)
            await self._sleep_or_wake(delay)

    async def _sleep_or_wake(self, seconds: float) -> None:
        """Sleep until the next scheduled check or an on-demand wake-up.

        :param seconds: Scheduled delay.
        """
        wake = self._wake
        try:
            await asyncio.wait_for(wake.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return
        wake.clear()
        await asyncio.sleep(self._min_check_spacing)

    def start(self, es: Any) -> asyncio.Task:
        """Start the background loop (idempotent). Never blocks startup.

        :param es: Synchronous Elasticsearch client.
        :return: The running task.
        :rtype: asyncio.Task
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(es), name="semantic-beta-contract-check")
        return self._task

    def stop(self) -> None:
        """Cancel the background loop if it is running."""
        if self._task is not None and not self._task.done():
            self._task.cancel()


@dataclass(frozen=True)
class SemanticRoute:
    """Which embedder serves a semantic request and how its vector legs are filtered."""

    index: str
    client: EmbeddingClient
    beta: bool = False
    eligible_only: bool = False


@dataclass
class ShelfmarkRouteRequest:
    """Shelf-mark search request built from a routed beta semantic or hybrid query.

    Unlike the requests of the shelf-mark endpoint and of chat retrieval, it
    sets ``prefix`` and ``page``: shelf-mark search then also treats the query
    as a prefix or volume (child marks rank first, both Mosseri spellings are
    tried) and pages its results like semantic and hybrid search do.
    """

    shelf_mark: str
    filters: Optional[Dict[str, Any]] = None
    num_results: int = 10
    page: int = 1
    exact_match: bool = False
    include_embeddings: bool = False
    prefix: bool = True

    @classmethod
    def from_query(
        cls, query: str, filters: Optional[Dict[str, Any]], num_results: Optional[int], page: Optional[int] = None
    ) -> "ShelfmarkRouteRequest":
        """Build a partial-match shelf-mark request from a bare shelf-mark query.

        :param query: The semantic or hybrid query (already known to be a shelf mark).
        :param filters: The request's filters.
        :param num_results: The request's page size.
        :param page: The request's 1-based page number.
        :return: The shelf-mark request.
        :rtype: ShelfmarkRouteRequest
        """
        shelf_mark = query.strip().rstrip(_SHELFMARK_TRAILING).strip()
        return cls(shelf_mark=shelf_mark, filters=filters, num_results=num_results or 10, page=page or 1)


class SemanticRouter:
    """Pick the query embedder that matches the target index's vectors."""

    def __init__(self, *, contracts: IndexContractCache, beta: SemanticBeta) -> None:
        """Create the router.

        :param contracts: Shared index-contract cache.
        :param beta: Beta state.
        """
        self._contracts = contracts
        self.beta = beta

    async def resolve(
        self,
        es: Any,
        index: str,
        default_index: str,
        default_client: EmbeddingClient,
    ) -> SemanticRoute:
        """Route a semantic/hybrid request to an embedder, or refuse it.

        The default index short-circuits to the default client with no
        Elasticsearch read and no extra filter: the v8 path is unchanged and
        stays guarded by the startup canary gate.

        :param es: Synchronous Elasticsearch client.
        :param index: Target index of the request.
        :param default_index: Configured default index.
        :param default_client: The default embedding client.
        :return: The route.
        :rtype: SemanticRoute
        :raises SemanticRouteError: 404 unknown index; 409 no embedder serves
            its model; 503 the beta embedder is not (or no longer) verified.
        """
        if index == default_index:
            return SemanticRoute(index=index, client=default_client)
        contract = self._contracts.get(es, index)
        if contract.identity is None:
            raise SemanticRouteError(
                409,
                f"Semantic search is not available on '{index}': its mapping records no embedding "
                "model, so the query embedder cannot be matched to its vectors. Use keyword or "
                "shelf-mark search on this index.",
            )
        if contract.identity == self._contracts.get(es, default_index).identity:
            return SemanticRoute(index=index, client=default_client, eligible_only=contract.eligibility_mapped)
        if self.beta.claims(index, contract):
            if not self.beta.serves(contract.identity):
                raise SemanticRouteError(
                    503, f"The beta semantic search ({self.beta.index}) is unavailable: {self.beta.reason}"
                )
            return SemanticRoute(
                index=index, client=self.beta.client, beta=True, eligible_only=contract.eligibility_mapped
            )
        raise SemanticRouteError(
            409,
            f"Semantic search is not available on '{index}': it was embedded with "
            f"{contract.describe()}, which no running embedder serves.",
        )

    async def embed_query(self, route: SemanticRoute, text: str) -> np.ndarray:
        """Embed a query with the route's client.

        The default route makes exactly the call the search paths always made.
        A beta embedder failure takes the beta offline (with a re-check
        scheduled) and surfaces as a 503 rather than a generic 500.

        :param route: Route from :meth:`resolve`.
        :param text: Query text.
        :return: The query vector.
        :rtype: np.ndarray
        :raises SemanticRouteError: 503 when the beta embedder fails.
        """
        if not route.beta:
            return await route.client.get_embedding(text, image=None, use_cache=False)
        try:
            return await route.client.get_embedding(text, image=None, use_cache=False)
        except httpx.HTTPError as exc:
            self.beta.mark_unavailable(f"beta embedder request failed: {type(exc).__name__}: {exc}")
            raise SemanticRouteError(
                503,
                f"The beta semantic search ({self.beta.index}) is temporarily unavailable (its "
                "embedder did not answer); it will be re-checked shortly.",
            ) from exc


BETA_SEMANTIC_INDEX = os.getenv("BETA_SEMANTIC_INDEX", DEFAULT_BETA_SEMANTIC_INDEX)
BETA_TEXT_REPRESENTATION_VERSION = os.getenv(
    "BETA_TEXT_REPRESENTATION_VERSION", DEFAULT_BETA_TEXT_REPRESENTATION_VERSION
)
# The compose service keeps its v9 name; it serves the v10 model revision.
BETA_EMBEDDING_SERVICE_URL = os.getenv("BETA_EMBEDDING_SERVICE_URL", "http://embedding-v9:8001")
SEMANTIC_BETA_ENABLED = env_flag("SEMANTIC_BETA_ENABLED", True)

index_contracts = IndexContractCache()
semantic_beta = SemanticBeta(
    index=BETA_SEMANTIC_INDEX,
    client=EmbeddingClient(base_url=BETA_EMBEDDING_SERVICE_URL),
    enabled=SEMANTIC_BETA_ENABLED,
    contracts=index_contracts,
    text_representation_version=BETA_TEXT_REPRESENTATION_VERSION,
)
semantic_router = SemanticRouter(contracts=index_contracts, beta=semantic_beta)

router = APIRouter()


@router.get("/semantic-beta", response_model=SemanticBetaStatus)
async def get_semantic_beta_status() -> SemanticBetaStatus:
    """Whether the opt-in beta semantic search (genizah_merged_v10) can be offered right now.

    :return: Availability, the beta index contract, the number of records
        with a vector, and the reason when unavailable.
    :rtype: SemanticBetaStatus
    """
    return semantic_beta.status()
