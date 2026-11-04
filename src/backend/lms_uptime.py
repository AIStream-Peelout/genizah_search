"""LM Studio uptime probe for external monitors: ``GET /lms_uptime``.

Chat depends on LM Studio (on the Mac Studio) serving the router, synthesis
and verification models. LM Studio has died silently several times — process
gone, server hung, or the pinned models evicted — leaving chat broken while
``/health`` (which checks nothing) stayed green. This endpoint gives an uptime
monitor a status code to alert on: 200 when LM Studio answers and every chat
model is loaded, 503 otherwise.

When the router runs on its own LM Studio (``ROUTER_LLM_URL``), the router
role is checked there and the other roles on the Studio. With
``ROUTER_FALLBACK=spine`` chat keeps working without the router, so a router
host that is down (or lacks the router) reports ``degraded`` with a 200.

It only reads LM Studio's model list (``GET /api/v0/models``): no inference,
and nothing that makes LM Studio JIT-load a model. ``/v1/models`` is not a
substitute — with JIT loading on it lists every downloaded model, loaded or
not. A model counts as loaded under its exact id or as a numbered instance
(``id:2``), the rule :meth:`AgenticRAGService.resolve_model` serves with.

The endpoint is public, so results are cached for a few seconds and concurrent
requests share one probe; it cannot be used to hammer LM Studio. Error details
stay in the server log — the response never carries LM Studio's address. The
full list of resident models (other projects' checkpoints included) is only
returned to callers presenting the chat API key (``X-API-Key``).
"""

import asyncio
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

import httpx
from fastapi import APIRouter, Header, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODELS_PATH = "/api/v0/models"
# A healthy LM Studio answers the model list in milliseconds; a slow answer is
# the hung-server failure mode this probe exists to catch.
PROBE_TIMEOUT_SECONDS = 5.0
# Uptime monitors poll every 30-300s; a short TTL bounds LM Studio traffic
# from a public endpoint without hiding an outage for long.
CACHE_TTL_SECONDS = 15.0


class ModelState(BaseModel):
    """Whether one configured chat model is resident in LM Studio."""

    model: str
    loaded: bool
    # The resident id serving it: the configured id or a numbered instance.
    serving_id: Optional[str] = None


class LmsUptime(BaseModel):
    """Response body of ``/lms_uptime``."""

    # degraded: chat works, but on the spine fallback (router host trouble).
    status: Literal["ok", "degraded", "down"]
    # ok | lm_studio_unreachable | models_not_loaded | chat_service_not_initialized
    # | router_host_unreachable | router_not_loaded
    reason: str
    lm_studio_reachable: bool
    latency_ms: Optional[float] = None
    models: Dict[str, ModelState] = {}
    # Every resident model; null unless the caller sent the chat API key.
    loaded_models: Optional[List[str]] = None
    error: Optional[str] = None
    checked_at: str


def serving_instance(model_id: str, loaded_ids: Iterable[str]) -> Optional[str]:
    """Return the resident model id that serves ``model_id``, if any.

    :param model_id: Configured model id, e.g. ``qwen/qwen3.8-27b``.
    :param loaded_ids: Ids LM Studio reports with ``state == "loaded"``.
    :return: ``model_id`` itself when loaded, else a loaded numbered instance of
        the same model (``qwen/qwen3.8-27b:2``), else ``None``.
    :rtype: Optional[str]
    """
    loaded = list(loaded_ids)
    if model_id in loaded:
        return model_id
    base = model_id.split(":")[0]
    return next((entry for entry in loaded if entry.split(":")[0] == base), None)


def _utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _describe(exc: Exception) -> str:
    """Name a probe failure without echoing the URL it carries.

    :param exc: Exception raised while reading the model list.
    :return: e.g. ``ConnectTimeout`` or ``HTTPStatusError 500``.
    :rtype: str
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTPStatusError {exc.response.status_code}"
    return type(exc).__name__


class LmsUptimeProbe:
    """Cached, coalesced check that LM Studio serves the chat models."""

    def __init__(
        self,
        base_url: str,
        models: Dict[str, str],
        ttl_seconds: float = CACHE_TTL_SECONDS,
        timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        clock: Callable[[], float] = time.monotonic,
        router_base_url: Optional[str] = None,
        router_optional: bool = False,
    ) -> None:
        """
        :param base_url: LM Studio server root, e.g. ``http://192.168.8.122:1234``.
        :param models: Chat role -> configured model id.
        :param ttl_seconds: How long one probe result is reused.
        :param timeout_seconds: Timeout of each phase (connect, read) of the model-list request.
        :param transport: httpx transport override (tests).
        :param clock: Monotonic clock (tests).
        :param router_base_url: LM Studio serving the ``router`` role
            (``ROUTER_LLM_URL``); ``None`` or ``base_url`` checks one host.
        :param router_optional: Chat survives router-host trouble
            (``ROUTER_FALLBACK=spine``): report it as ``degraded`` (200), not ``down``.
        """
        self.base_url = base_url.rstrip("/")
        router_root = (router_base_url or "").rstrip("/")
        self.router_base_url = router_root if router_root and router_root != self.base_url else None
        self.router_optional = router_optional
        self.models = dict(models)
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cached: Optional[LmsUptime] = None
        self._cached_at = 0.0

    @classmethod
    def for_service(cls, service: Any) -> "LmsUptimeProbe":
        """Build a probe for the models an :class:`AgenticRAGService` serves with.

        :param service: The app's agentic RAG service.
        :return: A probe over its LM Studio URL(s) and router/synthesis/verification models.
        :rtype: LmsUptimeProbe
        """
        return cls(
            base_url=service.llm_studio_base_url,
            models={
                "router": service.router_model,
                "synthesis": service.synthesis_model,
                "verification": service.verification_model,
            },
            router_base_url=service.router_base_url,
            router_optional=service.router_fallback == "spine",
        )

    async def check(self) -> LmsUptime:
        """Return the latest result, probing LM Studio when the cache is stale.

        Concurrent callers wait on one in-flight probe instead of each
        issuing their own.

        :return: Current LM Studio uptime state.
        :rtype: LmsUptime
        """
        async with self._lock:
            if self._cached is None or self._clock() - self._cached_at >= self.ttl_seconds:
                result = await self._probe()
                self._log_transition(result)
                self._cached, self._cached_at = result, self._clock()
            return self._cached

    async def _read_loaded(self, client: httpx.AsyncClient, base_url: str) -> List[str] | Exception:
        """Read one host's model list, returning a failure instead of raising.

        :param client: Client to read with.
        :param base_url: LM Studio server root.
        :return: Ids reported with ``state == "loaded"``, or the exception
            that stopped the read.
        :rtype: List[str] | Exception
        """
        try:
            response = await client.get(f"{base_url}{MODELS_PATH}")
            response.raise_for_status()
            entries = response.json()["data"]
            return [entry["id"] for entry in entries if entry.get("id") and entry.get("state") == "loaded"]
        except Exception as exc:
            host = "Router host" if base_url == self.router_base_url else "LM Studio"
            logger.warning("%s uptime probe failed: %r", host, exc)
            return exc

    def _model_states(
        self, loaded: List[str] | Exception, router_loaded: List[str] | Exception,
    ) -> Dict[str, ModelState]:
        """Each role's state, read from the host that serves it.

        :param loaded: The Studio's loaded ids, or its read failure.
        :param router_loaded: The router host's, when it has its own host.
        :return: Role -> model state.
        :rtype: Dict[str, ModelState]
        """
        states = {}
        for role, model in self.models.items():
            host_loaded = router_loaded if self.router_base_url and role == "router" else loaded
            serving = None if isinstance(host_loaded, Exception) else serving_instance(model, host_loaded)
            states[role] = ModelState(model=model, loaded=serving is not None, serving_id=serving)
        return states

    async def _probe(self) -> LmsUptime:
        """Read each LM Studio's model list once and evaluate it.

        :return: Uptime state from this single read.
        :rtype: LmsUptime
        """
        started = self._clock()
        router_loaded: List[str] | Exception = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self._transport) as client:
            if self.router_base_url is None:
                loaded = await self._read_loaded(client, self.base_url)
            else:
                loaded, router_loaded = await asyncio.gather(
                    self._read_loaded(client, self.base_url),
                    self._read_loaded(client, self.router_base_url),
                )
        states = self._model_states(loaded, router_loaded)
        if isinstance(loaded, Exception):
            return LmsUptime(
                status="down",
                reason="lm_studio_unreachable",
                lm_studio_reachable=False,
                models=states,
                error=_describe(loaded),
                checked_at=_utc_now(),
            )
        latency_ms = round((self._clock() - started) * 1000, 1)
        separate_router = self.router_base_url is not None and "router" in states
        status, reason, error = "ok", "ok", None
        if not all(state.loaded for role, state in states.items() if role != "router" or not separate_router):
            status, reason = "down", "models_not_loaded"
        elif separate_router and not states["router"].loaded:
            status = "degraded" if self.router_optional else "down"
            if isinstance(router_loaded, Exception):
                reason, error = "router_host_unreachable", _describe(router_loaded)
            else:
                reason = "router_not_loaded"
        if separate_router and not isinstance(router_loaded, Exception):
            loaded = loaded + [model for model in router_loaded if model not in loaded]
        return LmsUptime(
            status=status,
            reason=reason,
            lm_studio_reachable=True,
            latency_ms=latency_ms,
            models=states,
            loaded_models=loaded,
            error=error,
            checked_at=_utc_now(),
        )

    def _log_transition(self, result: LmsUptime) -> None:
        """Log when the state changes, so the backend log dates every LM Studio outage.

        :param result: The probe result about to replace the cached one.
        """
        previous = self._cached.reason if self._cached else None
        if result.reason == previous:
            return
        if result.status == "ok":
            logger.info("LM Studio uptime: ok (was %s)", previous or "unknown")
        elif result.status == "degraded":
            logger.warning(
                "LM Studio uptime: degraded, %s (was %s) error=%s; chat runs on the spine fallback",
                result.reason, previous or "unknown", result.error,
            )
        else:
            missing = [s.model for s in result.models.values() if not s.loaded]
            logger.error(
                "LM Studio uptime: %s (was %s) error=%s missing=%s",
                result.reason, previous or "unknown", result.error, missing,
            )


def create_router(probe: Optional[LmsUptimeProbe], api_key: str = "") -> APIRouter:
    """Build the router serving ``/lms_uptime``.

    :param probe: The probe to serve, or ``None`` when the chat service failed
        to initialize (always reported down).
    :param api_key: The chat API key; callers presenting it also get every
        resident model. Empty (local/dev) shows the list to everyone, like
        ``require_chat_api_key``.
    :return: Router with ``GET``/``HEAD /lms_uptime``.
    :rtype: APIRouter
    """
    router = APIRouter()

    # HEAD too: several uptime services probe with HEAD by default.
    @router.api_route("/lms_uptime", methods=["GET", "HEAD"], response_model=LmsUptime)
    async def lms_uptime(response: Response, x_api_key: Optional[str] = Header(default=None)) -> LmsUptime:
        """Whether LM Studio is up and serving every chat model — 200 if so, else 503.

        ``degraded`` (router host trouble under ``ROUTER_FALLBACK=spine``) is a 200.

        Reads LM Studio's model list only; never runs or loads a model.

        :param response: Used to set the status code and disable caching.
        :param x_api_key: Optional chat API key unlocking ``loaded_models``.
        :return: Reachability and per-role model state, plus every resident
            model for key holders.
        :rtype: LmsUptime
        """
        if probe is None:
            result = LmsUptime(
                status="down",
                reason="chat_service_not_initialized",
                lm_studio_reachable=False,
                checked_at=_utc_now(),
            )
        else:
            result = await probe.check()
        trusted = not api_key or (x_api_key is not None and secrets.compare_digest(x_api_key, api_key))
        if not trusted:
            result = result.model_copy(update={"loaded_models": None})
        response.status_code = 503 if result.status == "down" else 200
        response.headers["Cache-Control"] = "no-store"
        return result

    return router
