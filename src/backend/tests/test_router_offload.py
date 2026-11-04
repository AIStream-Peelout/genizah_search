"""Router model on its own LM Studio (``ROUTER_LLM_URL`` and friends).

Mocked network only: every LM Studio request goes to
:class:`router_offload_scenarios.FakeLMStudio`, and an autouse guard makes any
other request or socket connection raise.
"""

import asyncio
import json
import socket
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend import lms_agentic_search as agent_module
from src.backend import lms_uptime as lu
from src.backend.lms_agentic_search import (
    LOCAL_MODEL_CAPACITY_MESSAGE,
    AgenticRAGService,
    ModelUnavailableError,
    RouterHostError,
)
from src.backend.tests.router_offload_scenarios import (
    CONFIGS,
    FOLLOWUP_QUERY,
    GOLDEN_PATH,
    HISTORY,
    ROUTE_QUERY,
    ROUTER_KINDS,
    ROUTER_URL,
    STUDIO_KINDS,
    STUDIO_URL,
    FakeLMStudio,
    client_factory,
    golden_view,
    run_every_model_call,
    scenario_env,
)

SYNTH = "qwen/qwen3.8-27b"
VERIFY = "qwen/qwen3.6-35b-a3b"
ROUTER = "genizah-router"
# The variables this change adds; compose passes each as ${VAR:-}.
NEW_VARIABLES = (
    "ROUTER_LLM_URL", "ROUTER_REASONING_EFFORT", "ROUTER_CONNECT_TIMEOUT", "ROUTER_REQUEST_TIMEOUT",
    "ROUTER_MODEL_TTL", "ROUTER_FALLBACK", "ROUTER_MAX_CONCURRENCY", "ROUTER_MODEL_OVERRIDE",
)
OFFLOAD = {"ROUTER_LLM_URL": ROUTER_URL, "ROUTER_MODEL": ROUTER}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any request that bypasses the mock transport raise.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    def refuse(*args: Any, **kwargs: Any) -> None:
        """Refuse a real network operation.

        :param args: Ignored.
        :param kwargs: Ignored.
        :raises RuntimeError: Always.
        """
        raise RuntimeError("unmocked network call in a router-offload test")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


def make_service(monkeypatch: pytest.MonkeyPatch, config: str = "prod_like", **extra: str) -> AgenticRAGService:
    """Build a service under one scenario's environment.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param config: Key of ``CONFIGS``.
    :param extra: Further variables, e.g. ``ROUTER_LLM_URL``.
    :returns: The service.
    :rtype: AgenticRAGService
    """
    for key, value in scenario_env(config, extra).items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return AgenticRAGService()


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeLMStudio) -> FakeLMStudio:
    """Route every ``httpx.AsyncClient`` the service builds to ``fake``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param fake: Scripted LM Studio.
    :returns: ``fake``.
    :rtype: FakeLMStudio
    """
    monkeypatch.setattr(agent_module.httpx, "AsyncClient", client_factory(fake))
    return fake


def offload_fake(router_models: Optional[List[str]] = None, **failures: Callable) -> FakeLMStudio:
    """A Studio serving synthesis/verification and a router host serving the router.

    :param router_models: Ids the router host reports loaded.
    :param failures: ``router=handler`` replaces every router-host answer.
    :returns: The fake.
    :rtype: FakeLMStudio
    """
    return FakeLMStudio(
        {STUDIO_URL: [SYNTH, VERIFY], ROUTER_URL: [ROUTER] if router_models is None else router_models},
        failures={ROUTER_URL: failures["router"]} if "router" in failures else None,
    )


def golden(config: str) -> List[Dict[str, Any]]:
    """Golden calls of one config, recorded on the untouched base.

    :param config: Key of ``CONFIGS``.
    :returns: Golden views, in order.
    :rtype: List[Dict[str, Any]]
    """
    return json.loads(GOLDEN_PATH.read_text())[config]


def golden_of_kind(config: str, kind: str) -> List[Dict[str, Any]]:
    """Golden calls of one kind.

    :param config: Key of ``CONFIGS``.
    :param kind: Call kind.
    :returns: The matching golden views.
    :rtype: List[Dict[str, Any]]
    """
    return [call for call in golden(config) if call["kind"] == kind]


# ---------------------------------------------------------------------------
# (a) nothing set: byte-identical to today
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank_passthroughs", [False, True], ids=["unset", "blank"])
@pytest.mark.parametrize("config", list(CONFIGS))
async def test_a_nothing_set_is_byte_identical_to_today(
    monkeypatch: pytest.MonkeyPatch, config: str, blank_passthroughs: bool,
) -> None:
    """Unset or blank (compose ``${VAR:-}``) new variables change nothing on the wire."""
    extra = {name: "" for name in NEW_VARIABLES} if blank_passthroughs else {}
    service = make_service(monkeypatch, config, **extra)
    fake = install(monkeypatch, FakeLMStudio(
        {STUDIO_URL: [service.router_model, service.synthesis_model, service.verification_model]}
    ))

    await run_every_model_call(service)

    assert [golden_view(call) for call in fake.calls] == golden(config)
    assert all(call["url"].startswith(STUDIO_URL + "/") for call in fake.calls)
    assert service._router_gateway is None


# ---------------------------------------------------------------------------
# (b) router-role calls go to ROUTER_LLM_URL, by role and not by model id
# ---------------------------------------------------------------------------

async def test_b_router_calls_go_to_the_router_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route, retry, resolver, rewrite, Hebrew variant and re-plan hit the
    router host; synthesis, verification and smoothing stay on the Studio,
    byte-identical to today."""
    service = make_service(monkeypatch, **OFFLOAD)
    fake = install(monkeypatch, offload_fake())

    await run_every_model_call(service)

    for kind in ROUTER_KINDS:
        calls = fake.of_kind(kind)
        assert calls, kind
        assert all(call["url"] == f"{ROUTER_URL}/v1/chat/completions" for call in calls), kind
        assert all(call["json"]["model"] == ROUTER for call in calls), kind
    for kind in STUDIO_KINDS:
        calls = fake.of_kind(kind)
        assert [call["url"] for call in calls] == [f"{STUDIO_URL}/v1/chat/completions"], kind
        assert [call["body_sha256"] for call in calls] == [
            call["body_sha256"] for call in golden_of_kind("prod_like", kind)
        ], kind
    assert sorted(call["url"] for call in fake.of_kind("models_list")) == [
        f"{ROUTER_URL}/api/v0/models", f"{STUDIO_URL}/api/v0/models",
    ]
    assert not fake.of_kind("jit_load") and not fake.of_kind("unknown")


async def test_b_shared_model_id_is_routed_by_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the code defaults the router and verifier share one id; the
    verifier must still go to the Studio."""
    service = make_service(monkeypatch, "code_defaults", ROUTER_LLM_URL=ROUTER_URL)
    assert service.router_model == service.verification_model
    fake = install(monkeypatch, FakeLMStudio({
        STUDIO_URL: [service.synthesis_model, service.verification_model],
        ROUTER_URL: [service.router_model],
    }))

    await run_every_model_call(service)

    assert {call["url"] for kind in ROUTER_KINDS for call in fake.of_kind(kind)} == {
        f"{ROUTER_URL}/v1/chat/completions"
    }
    assert [call["url"] for call in fake.of_kind("verify")] == [f"{STUDIO_URL}/v1/chat/completions"]
    assert [call["url"] for call in fake.of_kind("synthesis")] == [f"{STUDIO_URL}/v1/chat/completions"]


# ---------------------------------------------------------------------------
# (c) ROUTER_REASONING_EFFORT reaches router payloads only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offloaded", [True, False], ids=["router_host", "router_on_studio"])
@pytest.mark.parametrize("config", list(CONFIGS))
async def test_c_reasoning_effort_only_on_router_payloads(
    monkeypatch: pytest.MonkeyPatch, config: str, offloaded: bool,
) -> None:
    """Every router payload, resolver included, carries ``reasoning_effort``;
    every other payload is byte-identical to today."""
    extra = {"ROUTER_REASONING_EFFORT": "none"}
    if offloaded:
        extra["ROUTER_LLM_URL"] = ROUTER_URL
    service = make_service(monkeypatch, config, **extra)
    every_model = [service.router_model, service.synthesis_model, service.verification_model]
    fake = install(monkeypatch, FakeLMStudio({STUDIO_URL: every_model, ROUTER_URL: [service.router_model]}))

    await run_every_model_call(service)

    for kind in ROUTER_KINDS:
        assert [call["json"].get("reasoning_effort") for call in fake.of_kind(kind)] == ["none"], kind
    for kind in STUDIO_KINDS:
        assert [call["body_sha256"] for call in fake.of_kind(kind)] == [
            call["body_sha256"] for call in golden_of_kind(config, kind)
        ], kind


# ---------------------------------------------------------------------------
# (d) ROUTER_MODEL_TTL; timeouts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ttl, expected", [("0", None), ("120", 120)])
async def test_d_router_ttl(monkeypatch: pytest.MonkeyPatch, ttl: str, expected: Optional[int]) -> None:
    """``ROUTER_MODEL_TTL=0`` omits ``ttl`` from router payloads; Studio payloads keep theirs."""
    service = make_service(monkeypatch, ROUTER_MODEL_TTL=ttl, **OFFLOAD)
    fake = install(monkeypatch, offload_fake())

    await run_every_model_call(service)

    for kind in ROUTER_KINDS:
        assert [call["json"].get("ttl") for call in fake.of_kind(kind)] == [expected], kind
    for kind in STUDIO_KINDS:
        assert [call["json"]["ttl"] for call in fake.of_kind(kind)] == [3600], kind


async def test_router_timeouts_apply_to_router_calls_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ROUTER_CONNECT_TIMEOUT`` / ``ROUTER_REQUEST_TIMEOUT`` reach router calls only."""
    service = make_service(monkeypatch, ROUTER_CONNECT_TIMEOUT="2", ROUTER_REQUEST_TIMEOUT="30", **OFFLOAD)
    fake = install(monkeypatch, offload_fake())

    await run_every_model_call(service)

    for kind in ROUTER_KINDS:
        assert [call["timeout"] for call in fake.of_kind(kind)] == [
            {"connect": 2.0, "read": 30.0, "write": 30.0, "pool": 30.0}
        ], kind
    for kind in STUDIO_KINDS:
        assert [call["timeout"]["connect"] for call in fake.of_kind(kind)] == [300.0], kind
    router_list = [call for call in fake.of_kind("models_list") if call["url"].startswith(ROUTER_URL)]
    assert [call["timeout"]["connect"] for call in router_list] == [2.0]


# ---------------------------------------------------------------------------
# No JIT loading of the router; its own lane
# ---------------------------------------------------------------------------

async def test_router_is_never_jit_loaded_when_offloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """With JIT loading allowed, a router missing on its host fails fast and
    nothing asks either host to load it."""
    service = make_service(monkeypatch, LM_STUDIO_ALLOW_JIT_LOAD="1", **OFFLOAD)
    assert service.allow_jit_load is True
    fake = install(monkeypatch, offload_fake(router_models=[]))

    with pytest.raises(RouterHostError) as excinfo:
        await service._call_llm(messages=[{"role": "user", "content": "hi"}], model=ROUTER, router=True)
    with pytest.raises(ValueError, match="never JIT-loaded"):
        await service.load_model(ROUTER)

    assert excinfo.value.reason == "model_unavailable"
    assert str(excinfo.value) == LOCAL_MODEL_CAPACITY_MESSAGE
    assert [call["kind"] for call in fake.calls] == ["models_list"]
    assert fake.calls[0]["url"] == f"{ROUTER_URL}/api/v0/models"


@pytest.mark.parametrize("configured, expected", [(None, 2), ("3", 3)])
async def test_router_host_has_its_own_lane(
    monkeypatch: pytest.MonkeyPatch, configured: Optional[str], expected: int,
) -> None:
    """A router call is not queued behind a Studio call that holds the shared slot."""
    extra = dict(OFFLOAD)
    if configured:
        extra["ROUTER_MAX_CONCURRENCY"] = configured
    service = make_service(monkeypatch, **extra)
    install(monkeypatch, offload_fake())
    assert service._router_gateway is not None
    assert service._router_gateway._max_concurrency == expected

    held, release = asyncio.Event(), asyncio.Event()

    async def hold_studio_slot() -> None:
        """Occupy the Studio lane until released."""
        async with agent_module.lm_studio_gateway.slot("27b"):
            held.set()
            await release.wait()

    holder = asyncio.create_task(hold_studio_slot())
    await held.wait()
    try:
        assert agent_module.lm_studio_gateway._ensure_semaphore().locked()
        reply = await asyncio.wait_for(
            service._call_llm(
                messages=[{"role": "user", "content": "Translate the topical content of: Purim"}],
                model=ROUTER, router=True,
            ),
            timeout=2.0,
        )
    finally:
        release.set()
        await holder
    assert reply == "פיוטי פורים של ינאי"


# ---------------------------------------------------------------------------
# (e) ROUTER_FALLBACK
# ---------------------------------------------------------------------------

def _raise(error: type) -> Callable[[httpx.Request], httpx.Response]:
    """Handler raising an httpx transport error.

    :param error: httpx exception class.
    :returns: The handler.
    :rtype: Callable[[httpx.Request], httpx.Response]
    """
    def handler(request: httpx.Request) -> httpx.Response:
        """Raise ``error`` for ``request``.

        :param request: Incoming request.
        :raises httpx.TransportError: Always.
        """
        raise error("router host failure", request=request)

    return handler


def _list_then(answer: Callable[[httpx.Request], httpx.Response]) -> Callable[[httpx.Request], httpx.Response]:
    """Handler serving the model list normally and ``answer`` for completions.

    :param answer: Handler for chat completions.
    :returns: The handler.
    :rtype: Callable[[httpx.Request], httpx.Response]
    """
    def handler(request: httpx.Request) -> httpx.Response:
        """Answer the list, delegate the rest.

        :param request: Incoming request.
        :returns: The response.
        :rtype: httpx.Response
        """
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": [{"id": ROUTER, "state": "loaded"}]})
        return answer(request)

    return handler


# Router-host failure -> (fake, reason recorded under spine).
FAILURES: Dict[str, tuple[Callable[[], FakeLMStudio], str]] = {
    "connect_refused": (lambda: offload_fake(router=_raise(httpx.ConnectError)), "host_connect"),
    "connect_timeout": (lambda: offload_fake(router=_raise(httpx.ConnectTimeout)), "host_connect"),
    "read_timeout": (lambda: offload_fake(router=_list_then(_raise(httpx.ReadTimeout))), "host_timeout"),
    "server_error": (
        lambda: offload_fake(router=_list_then(lambda r: httpx.Response(500, text="boom"))), "host_http_error",
    ),
    "list_server_error": (
        lambda: offload_fake(router=lambda r: httpx.Response(503, text="busy")), "host_http_error",
    ),
    "model_not_loaded": (lambda: offload_fake(router_models=[]), "model_unavailable"),
}


def stub_downstream_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every node after routing, so a turn runs without ES/Neo4j/Studio.

    Must run before the service is built: the graph binds the node methods.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    async def passthrough(self: AgenticRAGService, state: Dict[str, Any]) -> Dict[str, Any]:
        """Leave the state unchanged."""
        return state

    async def synthesize(self: AgenticRAGService, state: Dict[str, Any]) -> Dict[str, Any]:
        """Write a fixed draft."""
        state["draft_answer"] = "A draft answer."
        return state

    async def finalize(self: AgenticRAGService, state: Dict[str, Any]) -> Dict[str, Any]:
        """Publish the draft."""
        state["final_answer"] = state["draft_answer"]
        return state

    for name in ("_execute_searches_node", "_link_primary_secondary_node", "_check_quotes_node",
                 "_verify_claims_node", "_repair_answer_node"):
        monkeypatch.setattr(AgenticRAGService, name, passthrough)
    monkeypatch.setattr(AgenticRAGService, "_synthesize_answer_node", synthesize)
    monkeypatch.setattr(AgenticRAGService, "_finalize_response_node", finalize)


@pytest.mark.parametrize("failure", list(FAILURES))
async def test_e_spine_completes_the_turn_on_the_bibliography_plan(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """A router-host failure under ``spine`` falls back to today's
    bibliography-only plan and records why."""
    make_fake, reason = FAILURES[failure]
    stub_downstream_nodes(monkeypatch)
    service = make_service(monkeypatch, ROUTER_FALLBACK="spine", **OFFLOAD)
    fake = install(monkeypatch, make_fake())

    response = await service.chat(ROUTE_QUERY)

    assert response.success is True
    assert response.answer == "A draft answer."
    actions = response.query_plan.actions
    assert [(a.search_type, a.query, a.keyword_weight, a.semantic_weight) for a in actions] == [
        ("bibliography_hybrid", ROUTE_QUERY, 70, 30)
    ]
    assert response.query_plan.reasoning == f"Fallback: bibliography-only search (router_fallback: {reason})"
    assert f"router_fallback: {reason} (bibliography-only plan)" in response.processing_steps
    assert response.metrics["router_fallback"] == reason
    # One route attempt, no retry and no Hebrew variant against a failed host.
    router_posts = [c for c in fake.calls if c["url"].startswith(ROUTER_URL) and c["method"] == "POST"]
    assert [c["kind"] for c in router_posts] in ([], ["route"])
    assert not fake.of_kind("hebrew_variant")


async def test_e_spine_resolver_uses_the_literal_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A follow-up whose resolver call fails keeps the user's own wording."""
    stub_downstream_nodes(monkeypatch)
    service = make_service(monkeypatch, ROUTER_FALLBACK="spine", **OFFLOAD)
    install(monkeypatch, offload_fake(router=_raise(httpx.ConnectError)))

    response = await service.chat(FOLLOWUP_QUERY, conversation_history=list(HISTORY))

    assert response.success is True
    assert response.resolved_query is None
    assert "router_fallback: host_connect (using the message as written)" in response.processing_steps
    assert "router_fallback: host_connect (bibliography-only plan)" in response.processing_steps
    assert response.query_plan.actions[0].query == FOLLOWUP_QUERY
    assert response.metrics["router_fallback"] == "host_connect"


@pytest.mark.parametrize("failure", ["connect_refused", "read_timeout", "model_not_loaded"])
@pytest.mark.parametrize("history", [False, True], ids=["route", "resolver"])
async def test_e_none_keeps_todays_error(monkeypatch: pytest.MonkeyPatch, failure: str, history: bool) -> None:
    """``ROUTER_FALLBACK`` unset: the turn fails with today's capacity error."""
    stub_downstream_nodes(monkeypatch)
    service = make_service(monkeypatch, **OFFLOAD)
    assert service.router_fallback == "none"
    install(monkeypatch, FAILURES[failure][0]())

    with pytest.raises(ModelUnavailableError) as excinfo:
        await service.chat(FOLLOWUP_QUERY if history else ROUTE_QUERY,
                           conversation_history=list(HISTORY) if history else None)

    assert str(excinfo.value) == LOCAL_MODEL_CAPACITY_MESSAGE


async def test_e_healthy_router_host_records_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deploy check: a normal turn plans on the router host with no ``router_fallback``."""
    stub_downstream_nodes(monkeypatch)
    service = make_service(monkeypatch, ROUTER_FALLBACK="spine", ROUTER_REASONING_EFFORT="none", **OFFLOAD)
    fake = install(monkeypatch, offload_fake())

    response = await service.chat(ROUTE_QUERY)

    assert "router_fallback" not in response.metrics
    assert not any(step.startswith("router_fallback") for step in response.processing_steps)
    assert response.query_plan.reasoning == "Scholarship on Yannai's Purim poems"
    assert [c["url"] for c in fake.of_kind("route_retry")] == [f"{ROUTER_URL}/v1/chat/completions"]
    route_stage = response.metrics["llm_by_stage"]["route_query"]
    assert route_stage["calls"] == 3  # route, retry, Hebrew variant


def test_unknown_fallback_mode_means_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in ROUTER_FALLBACK must not crash startup or enable the fallback."""
    assert make_service(monkeypatch, ROUTER_FALLBACK="spien").router_fallback == "none"


# ---------------------------------------------------------------------------
# (f) /lms_uptime per role
# ---------------------------------------------------------------------------

MODELS = {"router": ROUTER, "synthesis": SYNTH, "verification": VERIFY}


def _uptime_client(fake: FakeLMStudio, router_optional: bool) -> TestClient:
    """TestClient over the uptime router with a separate router host.

    :param fake: Scripted LM Studio for both hosts.
    :param router_optional: ``ROUTER_FALLBACK=spine``.
    :returns: The client.
    :rtype: TestClient
    """
    probe = lu.LmsUptimeProbe(
        STUDIO_URL, MODELS, transport=httpx.MockTransport(fake),
        router_base_url=ROUTER_URL, router_optional=router_optional,
    )
    app = FastAPI()
    app.include_router(lu.create_router(probe))
    return TestClient(app)


@pytest.mark.parametrize("router_optional", [False, True])
def test_f_uptime_reports_each_role_on_its_host(router_optional: bool) -> None:
    """Router checked on the router host, the rest on the Studio; reads only."""
    fake = offload_fake()

    response = _uptime_client(fake, router_optional).get("/lms_uptime")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["reason"] == "ok"
    assert {role: (s["loaded"], s["serving_id"]) for role, s in body["models"].items()} == {
        "router": (True, ROUTER), "synthesis": (True, SYNTH), "verification": (True, VERIFY),
    }
    assert body["loaded_models"] == [SYNTH, VERIFY, ROUTER]
    assert sorted((c["method"], c["url"]) for c in fake.calls) == [
        ("GET", f"{ROUTER_URL}/api/v0/models"), ("GET", f"{STUDIO_URL}/api/v0/models"),
    ]


@pytest.mark.parametrize("failure, reason, error", [
    ("connect_refused", "router_host_unreachable", "ConnectError"),
    ("list_server_error", "router_host_unreachable", "HTTPStatusError 503"),
    ("model_not_loaded", "router_not_loaded", None),
])
@pytest.mark.parametrize("router_optional, status, code", [(True, "degraded", 200), (False, "down", 503)])
def test_f_router_host_trouble_is_200_only_with_spine(
    failure: str, reason: str, error: Optional[str], router_optional: bool, status: str, code: int,
) -> None:
    """Under spine chat still works without the router: 200 ``degraded``."""
    response = _uptime_client(FAILURES[failure][0](), router_optional).get("/lms_uptime")

    assert response.status_code == code
    body = response.json()
    assert (body["status"], body["reason"], body["error"]) == (status, reason, error)
    assert body["lm_studio_reachable"] is True
    assert {role: s["loaded"] for role, s in body["models"].items()} == {
        "router": False, "synthesis": True, "verification": True,
    }
    assert "test:1234" not in json.dumps({k: v for k, v in body.items() if k != "loaded_models"})


def test_f_studio_down_is_503_even_with_spine() -> None:
    """Spine covers the router only; the router role is still reported from its host."""
    fake = FakeLMStudio({ROUTER_URL: [ROUTER]}, failures={STUDIO_URL: _raise(httpx.ConnectError)})

    response = _uptime_client(fake, router_optional=True).get("/lms_uptime")

    assert response.status_code == 503
    body = response.json()
    assert (body["status"], body["reason"], body["error"]) == ("down", "lm_studio_unreachable", "ConnectError")
    assert {role: s["loaded"] for role, s in body["models"].items()} == {
        "router": True, "synthesis": False, "verification": False,
    }


def test_f_for_service_reads_the_router_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe follows the service's router host and fallback mode."""
    probe = lu.LmsUptimeProbe.for_service(
        make_service(monkeypatch, ROUTER_FALLBACK="spine", **{**OFFLOAD, "ROUTER_LLM_URL": ROUTER_URL + "/"})
    )

    assert (probe.base_url, probe.router_base_url, probe.router_optional) == (STUDIO_URL, ROUTER_URL, True)
    assert probe.models["router"] == ROUTER


# ---------------------------------------------------------------------------
# Compose hands the root .env's ROUTER_MODEL over as ROUTER_MODEL_OVERRIDE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override, expected", [("", "qwen/qwen3-4b-2507"), (ROUTER, ROUTER)])
def test_router_model_override(monkeypatch: pytest.MonkeyPatch, override: str, expected: str) -> None:
    """A blank override keeps ROUTER_MODEL (from src/backend/.env in prod)."""
    service = make_service(monkeypatch, ROUTER_MODEL_OVERRIDE=override)

    assert service.router_model == expected
