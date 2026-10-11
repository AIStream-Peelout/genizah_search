"""Tests for ``GET /lms_uptime``, the LM Studio probe for the external uptime monitor."""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend import lms_uptime as lu

BASE_URL = "http://192.168.8.122:1234"
MODELS = {
    "router": "qwen/qwen3-4b-2507",
    "synthesis": "qwen/qwen3.8-27b",
    "verification": "qwen/qwen3.6-35b-a3b",
}


def _entry(model_id: str, state: str = "loaded") -> Dict[str, Any]:
    """One ``/api/v0/models`` entry."""
    return {"id": model_id, "object": "model", "type": "llm", "state": state}


ALL_LOADED = [_entry(m) for m in MODELS.values()] + [
    _entry("qwen3-vl-8b-heb-v21b-step1200"),
    _entry("google/gemma-4-31b-qat", "not-loaded"),
]


class FakeLmStudio:
    """Records requests and answers ``/api/v0/models`` with a scripted handler."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


def _models_response(entries: List[Dict[str, Any]]) -> Callable[[httpx.Request], httpx.Response]:
    """Handler answering with the given model entries."""
    return lambda request: httpx.Response(200, json={"object": "list", "data": entries})


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _probe(fake: FakeLmStudio, clock: Optional[FakeClock] = None, models: Optional[Dict[str, str]] = None) -> lu.LmsUptimeProbe:
    """A probe over the fake LM Studio."""
    return lu.LmsUptimeProbe(
        BASE_URL,
        models or MODELS,
        transport=httpx.MockTransport(fake),
        clock=clock or FakeClock(),
    )


def _client(probe: Optional[lu.LmsUptimeProbe], api_key: str = "") -> TestClient:
    """TestClient over an app serving only the uptime router."""
    app = FastAPI()
    app.include_router(lu.create_router(probe, api_key=api_key))
    return TestClient(app)


def test_all_models_loaded_is_200() -> None:
    fake = FakeLmStudio(_models_response(ALL_LOADED))
    response = _client(_probe(fake)).get("/lms_uptime")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["reason"] == "ok"
    assert body["lm_studio_reachable"] is True
    assert {role: s["loaded"] for role, s in body["models"].items()} == {
        "router": True, "synthesis": True, "verification": True,
    }
    assert body["loaded_models"] == [*MODELS.values(), "qwen3-vl-8b-heb-v21b-step1200"]
    assert response.headers["cache-control"] == "no-store"


def test_only_reads_the_model_list() -> None:
    """The probe must never run inference or trigger a JIT load."""
    fake = FakeLmStudio(_models_response(ALL_LOADED))
    _client(_probe(fake)).get("/lms_uptime")

    assert [(r.method, str(r.url)) for r in fake.requests] == [("GET", f"{BASE_URL}/api/v0/models")]


def test_numbered_instance_counts_as_loaded() -> None:
    entries = [_entry("qwen/qwen3-4b-2507"), _entry("qwen/qwen3.8-27b:2"), _entry("qwen/qwen3.6-35b-a3b")]
    response = _client(_probe(FakeLmStudio(_models_response(entries)))).get("/lms_uptime")

    assert response.status_code == 200
    assert response.json()["models"]["synthesis"]["serving_id"] == "qwen/qwen3.8-27b:2"


def test_evicted_model_is_503() -> None:
    entries = [_entry("qwen/qwen3-4b-2507"), _entry("qwen/qwen3.8-27b", "not-loaded"), _entry("qwen/qwen3.6-35b-a3b")]
    response = _client(_probe(FakeLmStudio(_models_response(entries)))).get("/lms_uptime")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "down" and body["reason"] == "models_not_loaded"
    assert body["lm_studio_reachable"] is True
    assert body["models"]["synthesis"] == {"model": "qwen/qwen3.8-27b", "loaded": False, "serving_id": None}


@pytest.mark.parametrize(
    "handler, error",
    [
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)), "ConnectError"),
        (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("hung", request=r)), "ReadTimeout"),
        (lambda r: httpx.Response(500, text="boom"), "HTTPStatusError 500"),
        (lambda r: httpx.Response(200, text="<html>not json</html>"), "JSONDecodeError"),
        (lambda r: httpx.Response(200, json={"unexpected": True}), "KeyError"),
    ],
)
def test_unreachable_or_broken_lm_studio_is_503(handler: Callable[[httpx.Request], httpx.Response], error: str) -> None:
    response = _client(_probe(FakeLmStudio(handler))).get("/lms_uptime")

    assert response.status_code == 503
    body = response.json()
    assert body["reason"] == "lm_studio_unreachable"
    assert body["lm_studio_reachable"] is False
    assert body["error"] == error
    assert {role: s["loaded"] for role, s in body["models"].items()} == {
        "router": False, "synthesis": False, "verification": False,
    }
    assert "192.168" not in json.dumps(body)


def test_service_not_initialized_is_503() -> None:
    response = _client(None).get("/lms_uptime")

    assert response.status_code == 503
    assert response.json()["reason"] == "chat_service_not_initialized"


def test_head_is_supported() -> None:
    response = _client(_probe(FakeLmStudio(_models_response(ALL_LOADED)))).head("/lms_uptime")

    assert response.status_code == 200


def test_result_is_cached_for_the_ttl() -> None:
    fake = FakeLmStudio(_models_response(ALL_LOADED))
    clock = FakeClock()
    client = _client(_probe(fake, clock))

    client.get("/lms_uptime")
    clock.now += lu.CACHE_TTL_SECONDS - 1
    client.get("/lms_uptime")
    assert len(fake.requests) == 1

    clock.now += 1
    client.get("/lms_uptime")
    assert len(fake.requests) == 2


def test_recovery_is_seen_after_the_ttl() -> None:
    entries = {"value": []}
    fake = FakeLmStudio(lambda r: httpx.Response(200, json={"data": entries["value"]}))
    clock = FakeClock()
    client = _client(_probe(fake, clock))

    assert client.get("/lms_uptime").status_code == 503
    entries["value"] = ALL_LOADED
    clock.now += lu.CACHE_TTL_SECONDS
    assert client.get("/lms_uptime").status_code == 200


async def test_concurrent_checks_share_one_probe() -> None:
    """Callers arriving while a probe is in flight wait for it instead of probing again."""
    release = asyncio.Event()
    requests: List[httpx.Request] = []

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await release.wait()
        return httpx.Response(200, json={"data": ALL_LOADED})

    probe = lu.LmsUptimeProbe(BASE_URL, MODELS, transport=httpx.MockTransport(slow_handler), clock=FakeClock())
    pending = [asyncio.create_task(probe.check()) for _ in range(20)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(requests) == 1

    release.set()
    results = await asyncio.gather(*pending)

    assert len(requests) == 1
    assert all(r.status == "ok" for r in results)


def test_failures_are_cached_too() -> None:
    """A dead LM Studio is not re-probed on every hit during the TTL."""
    fake = FakeLmStudio(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)))
    client = _client(_probe(fake))

    assert client.get("/lms_uptime").status_code == 503
    assert client.get("/lms_uptime").status_code == 503
    assert len(fake.requests) == 1


def test_loaded_models_hidden_without_the_key() -> None:
    """Other projects' checkpoints are not listed to anonymous callers; the uptime verdict is."""
    client = _client(_probe(FakeLmStudio(_models_response(ALL_LOADED))), api_key="secret")

    for headers in ({}, {"X-API-Key": "wrong"}):
        response = client.get("/lms_uptime", headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["loaded_models"] is None
        assert body["models"]["synthesis"]["loaded"] is True
        assert "qwen3-vl-8b-heb-v21b-step1200" not in response.text


def test_loaded_models_shown_with_the_key() -> None:
    client = _client(_probe(FakeLmStudio(_models_response(ALL_LOADED))), api_key="secret")

    body = client.get("/lms_uptime", headers={"X-API-Key": "secret"}).json()

    assert body["loaded_models"] == [*MODELS.values(), "qwen3-vl-8b-heb-v21b-step1200"]


def test_for_service_reads_the_effective_config() -> None:
    class Service:
        llm_studio_base_url = BASE_URL + "/"
        router_model = "r"
        synthesis_model = "s"
        verification_model = "v"

    probe = lu.LmsUptimeProbe.for_service(Service())

    assert probe.base_url == BASE_URL
    assert probe.models == {"router": "r", "synthesis": "s", "verification": "v"}


@pytest.mark.parametrize(
    "model_id, loaded, expected",
    [
        ("a/b", ["a/b"], "a/b"),
        ("a/b", ["a/b:2"], "a/b:2"),
        ("a/b:2", ["a/b"], "a/b"),
        ("a/b", ["a/bc", "x"], None),
        ("a/b", [], None),
    ],
)
def test_serving_instance(model_id: str, loaded: List[str], expected: Optional[str]) -> None:
    assert lu.serving_instance(model_id, loaded) == expected
