"""Scripted LM Studio for the router-offload tests (mocked network only).

:class:`FakeLMStudio` answers every LM Studio request the chat pipeline makes
and records it. :func:`run_every_model_call` sends each router-role call
(route, route retry, resolver, rewrite, Hebrew variant, re-plan) and each
Studio-role call (synthesis, verification, quote smoothing) through the real
call helpers once.

This module uses only APIs that predate the router offload, so the same driver
produced ``fixtures/router_offload_golden.json`` on the untouched base
(f536c1c). Regenerate it only from a known-good base::

    PYTHONPATH=. .venv/bin/python -m src.backend.tests.router_offload_scenarios
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

STUDIO_URL = "http://studio.test:1234"
ROUTER_URL = "http://router.test:1234"
GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "router_offload_golden.json"

# Every variable that changes how the service talks to LM Studio. Each
# scenario starts with all of them unset, so a developer's src/backend/.env
# (loaded at import) cannot leak into the tests.
LM_ENV_KEYS = (
    "LLM_STUDIO_URL", "ROUTER_MODEL", "ROUTER_MODEL_OVERRIDE", "SYNTHESIS_MODEL",
    "VERIFICATION_MODEL", "SYNTHESIS_REASONING_EFFORT", "LM_STUDIO_MODEL_TTL",
    "LM_STUDIO_REQUEST_TIMEOUT", "LM_STUDIO_ALLOW_JIT_LOAD", "LM_STUDIO_MAX_CONCURRENCY",
    "QUOTE_FAILURE_MODE", "QUOTE_SMOOTHING_MODEL", "QUOTE_SMOOTHING_REASONING_EFFORT",
    "ROUTER_LLM_URL", "ROUTER_REASONING_EFFORT", "ROUTER_CONNECT_TIMEOUT",
    "ROUTER_REQUEST_TIMEOUT", "ROUTER_MODEL_TTL", "ROUTER_FALLBACK", "ROUTER_MAX_CONCURRENCY",
)

# Model settings of the golden scenarios. "prod_like" mirrors the production
# roles; "code_defaults" leaves the ids at their code defaults, where the
# router and verification models share one id.
CONFIGS: Dict[str, Dict[str, str]] = {
    "prod_like": {
        "LLM_STUDIO_URL": STUDIO_URL,
        "ROUTER_MODEL": "qwen/qwen3-4b-2507",
        "SYNTHESIS_MODEL": "qwen/qwen3.8-27b",
        "VERIFICATION_MODEL": "qwen/qwen3.6-35b-a3b",
        "SYNTHESIS_REASONING_EFFORT": "low",
    },
    "code_defaults": {
        "LLM_STUDIO_URL": STUDIO_URL,
        "SYNTHESIS_REASONING_EFFORT": "low",
    },
}

# Call kinds by role.
ROUTER_KINDS = ("route", "route_retry", "resolve", "rewrite", "hebrew_variant", "replan")
STUDIO_KINDS = ("synthesis", "verify", "smoothing")

ROUTE_QUERY = "Tell me about the Purim poems of Yannai"
FOLLOWUP_QUERY = "What do the verses say?"
HISTORY = [
    {"role": "user", "content": ROUTE_QUERY},
    {"role": "assistant", "content": "Yannai wrote piyyutim for Purim; scholars have edited several of them."},
]


def scenario_env(config: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """Environment for one scenario: every LM variable unset unless named.

    :param config: Key of :data:`CONFIGS`.
    :param extra: Further variables to set (e.g. ``ROUTER_LLM_URL``).
    :returns: Variable name -> value, ``None`` meaning unset.
    :rtype: Dict[str, Optional[str]]
    """
    env: Dict[str, Optional[str]] = {key: None for key in LM_ENV_KEYS}
    env.update(CONFIGS[config])
    env.update(extra or {})
    return env


def call_kind(request: httpx.Request, body: Optional[Dict[str, Any]]) -> str:
    """Name the pipeline call a request belongs to, from its path and prompt.

    :param request: Outgoing request.
    :param body: Its JSON body, if any.
    :returns: One of :data:`ROUTER_KINDS`, :data:`STUDIO_KINDS`,
        ``models_list``, ``jit_load`` or ``unknown``.
    :rtype: str
    """
    if request.url.path == "/api/v0/models":
        return "models_list"
    messages = (body or {}).get("messages") or []
    first = str(messages[0].get("content") or "") if messages else ""
    last = str(messages[-1].get("content") or "") if messages else ""
    if "tools" in (body or {}):
        return "route_retry" if "Respond ONLY by calling the create_search_plan tool" in last else "route"
    if first.startswith("You prepare the latest user message"):
        return "resolve"
    if first.startswith("You rewrite the latest user message"):
        return "rewrite"
    if last.startswith("Translate the topical content"):
        return "hebrew_variant"
    if last.startswith("Retrieval failed to find material"):
        return "replan"
    if last == "SYNTHESIS PROMPT":
        return "synthesis"
    if last == "VERIFIER PROMPT":
        return "verify"
    if "PARAGRAPH 0:" in last:
        return "smoothing"
    if last == "ping":
        return "jit_load"
    return "unknown"


_COMPLETIONS: Dict[str, Dict[str, Any]] = {
    # No tool call, so the route node retries.
    "route": {"content": "", "tool_calls": []},
    "route_retry": {"content": "", "tool_calls": [{
        "type": "function",
        "function": {"name": "create_search_plan", "arguments": json.dumps({
            "actions": [{"search_type": "bibliography_hybrid", "query": "Yannai Purim piyyutim",
                         "keyword_weight": 70, "semantic_weight": 30}],
            "needs_primary_secondary_linking": True,
            "reasoning": "Scholarship on Yannai's Purim poems",
        })},
    }]},
    # Repeats the message, so the resolver runs its dedicated rewrite pass.
    "resolve": {"content": f"FOLLOWUP: yes\nQUESTION: {FOLLOWUP_QUERY}"},
    "rewrite": {"content": "What do the verses of the Purim poems of Yannai say?"},
    "hebrew_variant": {"content": "פיוטי פורים של ינאי"},
    "replan": {"content": '{"searches": [{"query": "Yannai Purim piyyut", "keyword_weight": 70}]}'},
    "synthesis": {"content": "An answer."},
    "verify": {"content": '{"verified_claims": [], "summary": "ok"}'},
    "smoothing": {"content": '{"paragraphs": []}'},
    "jit_load": {"content": "pong"},
}


class FakeLMStudio:
    """Answers LM Studio's model list and chat completions, recording each request."""

    def __init__(
        self,
        loaded: Dict[str, List[str]],
        failures: Optional[Dict[str, Callable[[httpx.Request], httpx.Response]]] = None,
    ) -> None:
        """
        :param loaded: Base URL -> model ids its model list reports as loaded.
        :param failures: Base URL -> handler answering every request to that
            host instead (it may raise, e.g. ``httpx.ConnectError``).
        """
        self.loaded = loaded
        self.failures = failures or {}
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer it.

        :param request: Outgoing request.
        :returns: The scripted response.
        :rtype: httpx.Response
        """
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        body = json.loads(request.content) if request.content else None
        kind = call_kind(request, body)
        self.calls.append({
            "kind": kind,
            "method": request.method,
            "url": str(request.url),
            "json": body,
            # The exact bytes sent, key order included.
            "body_sha256": hashlib.sha256(request.content).hexdigest()[:16] if request.content else None,
            "timeout": dict(request.extensions.get("timeout") or {}),
        })
        if base in self.failures:
            return self.failures[base](request)
        if kind == "models_list":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": model, "state": "loaded"} for model in self.loaded.get(base, [])
            ]})
        return httpx.Response(200, json={
            "choices": [{"message": _COMPLETIONS.get(kind, {"content": ""}), "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    def of_kind(self, kind: str) -> List[Dict[str, Any]]:
        """Recorded calls of one kind.

        :param kind: A value :func:`call_kind` returns.
        :returns: The matching calls, in order.
        :rtype: List[Dict[str, Any]]
        """
        return [call for call in self.calls if call["kind"] == kind]


def client_factory(fake: FakeLMStudio) -> Callable[..., httpx.AsyncClient]:
    """An ``httpx.AsyncClient`` stand-in whose clients all use ``fake``.

    :param fake: Scripted LM Studio.
    :returns: Callable with ``AsyncClient``'s signature.
    :rtype: Callable[..., httpx.AsyncClient]
    """
    real_client = httpx.AsyncClient

    def make(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        """Build a real client over the mock transport.

        :param args: Positional ``AsyncClient`` arguments.
        :param kwargs: Keyword ``AsyncClient`` arguments.
        :returns: The client.
        :rtype: httpx.AsyncClient
        """
        kwargs["transport"] = httpx.MockTransport(fake)
        return real_client(*args, **kwargs)

    return make


async def run_every_model_call(service: Any) -> None:
    """Send each router-role and Studio-role call through the real helpers once.

    :param service: An ``AgenticRAGService`` whose ``httpx.AsyncClient`` is mocked.
    """
    from src.backend.lms_agentic_search import SYNTHESIS_MAX_TOKENS
    from src.backend.quote_check import GAP_MARKER

    await service._resolve_query_node({
        "user_query": FOLLOWUP_QUERY,
        "conversation_history": list(HISTORY),
        "resolved_query": None,
        "is_followup": False,
        "resolved_entities": [],
        "processing_steps": [],
    })
    route_state = {
        "user_query": ROUTE_QUERY,
        "conversation_history": [],
        "resolved_query": None,
        "is_followup": False,
        "resolved_entities": [],
        "processing_steps": [],
    }
    await service._route_query_node(route_state)
    await service._replan_after_failed_retrieval(route_state, ["Yannai"], "Yannai Purim")
    await service._call_llm(
        messages=[{"role": "user", "content": "SYNTHESIS PROMPT"}],
        model=service.synthesis_model,
        temperature=0.2,
        max_tokens=SYNTHESIS_MAX_TOKENS,
    )
    await service._call_verifier("VERIFIER PROMPT")
    if service.quote_smoothing_model:
        await service._request_smoothing({0: f"First sentence. {GAP_MARKER} Next sentence."})


def _digest(value: Any) -> str:
    """Short stable digest of a JSON value.

    :param value: Any JSON-serializable value.
    :returns: ``sha256:<first 16 hex chars>``.
    :rtype: str
    """
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]


def golden_view(call: Dict[str, Any]) -> Dict[str, Any]:
    """A recorded call with its long parts (prompts, tool schema) digested.

    :param call: One entry of :attr:`FakeLMStudio.calls`.
    :returns: The call, comparable byte for byte with the golden file.
    :rtype: Dict[str, Any]
    """
    body = call["json"]
    if body is not None:
        body = dict(body)
        if "messages" in body:
            body["messages"] = [
                {**message, "content": _digest(message.get("content"))} for message in body["messages"]
            ]
        for key in ("tools", "response_format"):
            if key in body:
                body[key] = _digest(body[key])
    return {**call, "json": body}


async def capture(config: str) -> List[Dict[str, Any]]:
    """Run :func:`run_every_model_call` under one config and return its calls.

    Sets ``os.environ`` directly: for the golden generator only, never in tests.

    :param config: Key of :data:`CONFIGS`.
    :returns: Golden views of every request, in order.
    :rtype: List[Dict[str, Any]]
    """
    from src.backend import lms_agentic_search as agent_module

    for key, value in scenario_env(config).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    service = agent_module.AgenticRAGService()
    fake = FakeLMStudio({STUDIO_URL: [service.router_model, service.synthesis_model, service.verification_model]})
    original = agent_module.httpx.AsyncClient
    agent_module.httpx.AsyncClient = client_factory(fake)
    try:
        await run_every_model_call(service)
    finally:
        agent_module.httpx.AsyncClient = original
    return [golden_view(call) for call in fake.calls]


def main() -> None:
    """Write the golden file from the current code."""
    golden = {config: asyncio.run(capture(config)) for config in CONFIGS}
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} ({', '.join(f'{k}: {len(v)} calls' for k, v in golden.items())})")


if __name__ == "__main__":
    main()
