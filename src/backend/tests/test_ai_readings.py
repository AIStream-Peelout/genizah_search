"""Tests for the advanced Readers view: readings sidecars and line corrections."""

import copy
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from elasticsearch import NotFoundError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend import ai_readings as ar

TOKEN = "advanced-test-token"
EXPORT = "export-test-token"
MODEL = "qwen3-vl-8b-heb-v22i-a50"
OTHER_MODEL = "qwen3-vl-8b-heb-v21b-step1200"
KRAKEN = "kraken:MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv"


def _line(index: int, text: str, box: List[int], key: str = "box", **extra: Any) -> Dict[str, Any]:
    """One reader (``box``) or final (``bbox``) line."""
    return {"index": index, "text": text, key: box, **extra}


def _reader(reader_id: str, kind: str, scope: str, lines: Optional[List[Dict[str, Any]]], **extra: Any) -> Dict[str, Any]:
    """One ``readers[]`` entry."""
    return {
        "id": reader_id,
        "kind": kind,
        "scope": scope,
        "unit": extra.pop("unit", None),
        "role": extra.pop("role", None),
        "reply_index": extra.pop("reply_index", 0),
        "parsed": True,
        "health_label": extra.pop("health_label", "ok"),
        "crop_box_1000": extra.pop("crop_box_1000", None),
        "lines": lines,
        **extra,
    }


def legacy_sidecar(doc_id: str = "Cambridge_CUL_Or_1080_15_50", image_index: int = 1, model: str = MODEL) -> Dict[str, Any]:
    """A legacy page: whole-page VLM + Kraken, no policy."""
    return {
        "doc_id": doc_id,
        "image_index": image_index,
        "image_url": f"https://images.example/{doc_id}/{image_index}.jpg",
        "image_sha256": "a" * 64,
        "image_width": 2000,
        "image_height": 3000,
        "vlm_model": model,
        "vlm_revision": "rev1",
        "decoded_at": "2026-09-28T20:52:04Z",
        "rule_version": "lines-v2-20260909",
        "policy": None,
        "readers": [
            _reader(f"vlm:{model}:page", "vlm", "page", [_line(0, "שלום עליכם", [100, 100, 900, 150])]),
            _reader(KRAKEN, "htr", "page", [_line(0, "שלום עליכס", [110, 105, 880, 148], conf=0.91)], cache_key="k1"),
        ],
        "final": {
            "source": "legacy",
            "lines": [
                _line(0, "שלום עליכם", [100, 100, 900, 150], key="bbox", agreement=0.9, status="agreed",
                      htr_text="שלום עליכס", htr_fragments=[[110, 105, 880, 148]]),
                _line(1, "ברוך", [100, 200, 900, 250], key="bbox", agreement=None, status="unconfirmed",
                      htr_text=None, htr_fragments=[]),
            ],
        },
        "provenance": {"builder": "test"},
    }


def columns_sidecar() -> Dict[str, Any]:
    """A columns page that fell back to the whole-page read (eight readers)."""
    sidecar = legacy_sidecar("Manchester_JRL_C_131", 2)
    units = []
    for name, crop in (("col0", [500, 0, 1000, 1000]), ("col1", [0, 0, 500, 1000])):
        units.append({
            "name": name, "role": "column", "crop_box_1000": crop, "resolved_by": "share", "health_label": "loop",
            "attempts": [
                {"source": "crop", "label": "loop", "letters": 900, "n_lines": 60, "reply_index": 1},
                {"source": "halves", "label": "loop", "letters": 800, "n_lines": 50, "reply_index": 2},
            ],
        })
    sidecar["policy"] = {
        "action": "columns", "reason": "two_columns", "final_source": "whole_page", "calls": 7,
        "fallback": True, "mode": "auto", "units": units,
    }
    extra = [
        _reader(f"vlm:{MODEL}:col0:crop", "vlm", "unit", [], unit="col0", role="column", health_label="loop"),
        _reader(f"vlm:{MODEL}:col1:crop", "vlm", "unit", [], unit="col1", role="column", health_label="loop"),
        _reader(f"vlm:{MODEL}:col0.top:crop", "vlm", "unit", [], unit="col0.top", role="half"),
        _reader(f"vlm:{MODEL}:col0.bottom:crop", "vlm", "unit", [], unit="col0.bottom", role="half"),
        _reader(f"vlm:{MODEL}:col1.top:crop", "vlm", "unit", None, unit="col1.top", role="half"),
        _reader(f"vlm:{MODEL}:col1.bottom:crop", "vlm", "unit", [], unit="col1.bottom", role="half"),
    ]
    sidecar["readers"].extend(extra)
    sidecar["final"]["source"] = "whole_page"
    return sidecar


class FakeIndices:
    """``es.indices`` over the fake's dict of indices."""

    def __init__(self, store: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        self.store = store
        self.created: Dict[str, Dict[str, Any]] = {}

    def exists(self, index: str) -> bool:
        return index in self.store

    def create(self, index: str, **body: Any) -> None:
        self.store[index] = {}
        self.created[index] = body


def _not_found() -> NotFoundError:
    return NotFoundError("index_not_found_exception", MagicMock(status=404), {})


class FakeES:
    """Just enough of the Elasticsearch client for the stores: term/terms/range filters and sorting."""

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.indices = FakeIndices(self.store)
        self.searches = 0

    def put(self, index: str, doc_id: str, document: Dict[str, Any]) -> None:
        self.store.setdefault(index, {})[doc_id] = copy.deepcopy(document)

    def index(self, index: str, id: str, document: Dict[str, Any], refresh: Any = None) -> None:
        if index not in self.store:
            raise _not_found()
        self.put(index, id, document)

    @staticmethod
    def _matches(source: Dict[str, Any], clause: Dict[str, Any]) -> bool:
        if "match_all" in clause:
            return True
        if "bool" in clause:
            return all(FakeES._matches(source, sub) for sub in clause["bool"]["filter"])
        if "terms" in clause:
            (field, values), = clause["terms"].items()
            return source.get(field) in values
        if "term" in clause:
            (field, value), = clause["term"].items()
            return source.get(field) == value
        if "range" in clause:
            (field, bounds), = clause["range"].items()
            return (source.get(field) or "") >= bounds["gte"]
        raise AssertionError(f"unsupported clause {clause}")

    def hits(self, index: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        if index not in self.store:
            raise _not_found()
        return [
            {"_id": doc_id, "_source": copy.deepcopy(source)}
            for doc_id, source in self.store[index].items()
            if self._matches(source, query)
        ]

    def search(self, index: str, query: Dict[str, Any], sort: List[Dict[str, Any]] = (), size: int = 10,
               source_excludes: Optional[List[str]] = None) -> Dict[str, Any]:
        self.searches += 1
        hits = self.hits(index, query)
        for spec in reversed(list(sort)):
            (field, options), = spec.items()
            hits.sort(key=lambda h: (h["_source"].get(field) is None, h["_source"].get(field) or 0),
                      reverse=options["order"] == "desc")
        for hit in hits:
            for field in source_excludes or []:
                hit["_source"].pop(field, None)
        return {"hits": {"hits": hits[:size]}}


@pytest.fixture
def es(monkeypatch: pytest.MonkeyPatch) -> FakeES:
    fake = FakeES()
    monkeypatch.setattr(ar.helpers, "scan", lambda es, index, query: fake.hits(index, query["query"]))
    for sidecar in (legacy_sidecar(), columns_sidecar()):
        fake.put(ar.DEFAULT_READINGS_INDEX, ar.readings_document_id(sidecar), ar.readings_document(sidecar))
    return fake


def _client(es: FakeES, access: str = TOKEN, export: str = EXPORT, per_ip: int = 30, total: int = 300) -> TestClient:
    app = FastAPI()
    app.include_router(ar.create_router(
        ar.ReadingsStore(es, index=ar.DEFAULT_READINGS_INDEX),
        ar.CorrectionStore(es, index=ar.DEFAULT_CORRECTIONS_INDEX),
        access_token=access,
        export_token=export,
        per_ip_limiter=ar.RateLimiter(per_ip),
        total_limiter=ar.RateLimiter(total),
    ))
    return TestClient(app)


H = {"X-Advanced-Token": TOKEN}
CORRECTION = {
    "image_index": 1, "line_index": 0, "reader_id": "final", "corrected_text": "שלום עליכם ושלום",
    "contributor": "A. Scholar", "note": "  ", "bbox": [0, 0, 1, 1], "original_text": "forged",
}


# --- gating -----------------------------------------------------------------


@pytest.mark.parametrize("method,path", [
    ("get", "/readings/access"),
    ("get", "/documents/Cambridge_CUL_Or_1080_15_50/readings?image_index=1"),
    ("get", "/documents/Cambridge_CUL_Or_1080_15_50/readings/pages"),
    ("get", "/documents/Cambridge_CUL_Or_1080_15_50/readings/corrections?image_index=1"),
    ("post", "/documents/Cambridge_CUL_Or_1080_15_50/readings/corrections"),
])
def test_routes_are_404_when_feature_off_and_401_without_token(es: FakeES, method: str, path: str) -> None:
    kwargs = {"json": CORRECTION} if method == "post" else {}
    assert getattr(_client(es, access=""), method)(path, headers=H, **kwargs).status_code == 404
    client = _client(es)
    assert getattr(client, method)(path, **kwargs).status_code == 401
    assert getattr(client, method)(path, headers={"X-Advanced-Token": "wrong"}, **kwargs).status_code == 401
    assert es.store.get(ar.DEFAULT_CORRECTIONS_INDEX) is None  # nothing was written


def test_access_ok_with_token(es: FakeES) -> None:
    response = _client(es).get("/readings/access", headers=H)
    assert response.status_code == 200 and response.json() == {"ok": True}


def test_export_needs_its_own_token(es: FakeES) -> None:
    client = _client(es)
    assert client.get("/admin/corrections.jsonl", headers=H).status_code == 401
    assert client.get("/admin/corrections.jsonl", headers={"X-Export-Token": EXPORT}).status_code == 200
    assert _client(es, export="").get("/admin/corrections.jsonl", headers={"X-Export-Token": EXPORT}).status_code == 404


# --- readings ---------------------------------------------------------------


def test_get_readings_returns_the_sidecar_with_cache_header(es: FakeES) -> None:
    response = _client(es).get("/documents/Manchester_JRL_C_131/readings?image_index=2", headers=H)
    assert response.status_code == 200
    body = response.json()
    assert body == columns_sidecar()
    assert len(body["readers"]) == 8 and body["policy"]["action"] == "columns"
    assert response.headers["cache-control"] == "private, max-age=3600"


def test_missing_page_is_404(es: FakeES) -> None:
    client = _client(es)
    assert client.get("/documents/Manchester_JRL_C_131/readings?image_index=9", headers=H).status_code == 404
    assert client.get(f"/documents/Manchester_JRL_C_131/readings?image_index=2&vlm_model={OTHER_MODEL}",
                      headers=H).status_code == 404


def test_missing_index_is_404_not_500() -> None:
    response = _client(FakeES()).get("/documents/X/readings?image_index=0", headers=H)
    assert response.status_code == 404


def test_two_models_without_choice_is_409_with_both(es: FakeES) -> None:
    other = legacy_sidecar(model=OTHER_MODEL)
    es.put(ar.DEFAULT_READINGS_INDEX, ar.readings_document_id(other), ar.readings_document(other))
    client = _client(es)
    response = client.get("/documents/Cambridge_CUL_Or_1080_15_50/readings?image_index=1", headers=H)
    assert response.status_code == 409
    assert set(response.json()["readings"]) == {MODEL, OTHER_MODEL}
    chosen = client.get(f"/documents/Cambridge_CUL_Or_1080_15_50/readings?image_index=1&vlm_model={OTHER_MODEL}",
                        headers=H)
    assert chosen.status_code == 200 and chosen.json()["vlm_model"] == OTHER_MODEL


def test_old_oxford_id_finds_the_new_form() -> None:
    fake = FakeES()
    sidecar = legacy_sidecar("Oxford_Bodleian_Bodl_MS_heb_c_13_5", 0)
    fake.put(ar.DEFAULT_READINGS_INDEX, ar.readings_document_id(sidecar), ar.readings_document(sidecar))
    response = _client(fake).get("/documents/Oxford_Bodleian_MS_heb_c_13_5/readings?image_index=0", headers=H)
    assert response.status_code == 200


def test_pages_lists_lookup_fields_only(es: FakeES) -> None:
    body = _client(es).get("/documents/Manchester_JRL_C_131/readings/pages", headers=H).json()
    assert body["pages"] == [{
        "image_index": 2, "vlm_model": MODEL, "policy_action": "columns", "n_readers": 8,
        "decoded_at": "2026-09-28T20:52:04Z", "rule_version": "lines-v2-20260909", "final_source": "whole_page",
    }]


def test_hits_are_cached_and_misses_are_not(es: FakeES) -> None:
    clock = [0.0]
    store = ar.ReadingsStore(es, index=ar.DEFAULT_READINGS_INDEX, clock=lambda: clock[0])
    assert store.get("Manchester_JRL_C_131", 2)
    assert store.get("Manchester_JRL_C_131", 2)
    assert es.searches == 1
    assert store.get("Manchester_JRL_C_131", 3) == {}
    assert store.get("Manchester_JRL_C_131", 3) == {}
    assert es.searches == 3
    clock[0] = ar.DEFAULT_CACHE_TTL_S + 1
    store.get("Manchester_JRL_C_131", 2)
    assert es.searches == 4


# --- corrections ------------------------------------------------------------


def test_correction_is_stored_from_the_sidecar_listed_and_exported(es: FakeES) -> None:
    client = _client(es)
    response = client.post("/documents/Cambridge_CUL_Or_1080_15_50/readings/corrections", headers=H, json=CORRECTION)
    assert response.status_code == 201, response.text
    stored = response.json()["correction"]
    assert stored["original_text"] == "שלום עליכם"  # the sidecar's text, not the client's
    assert stored["bbox"] == [100, 100, 900, 150]
    assert stored["source_reader"] == "final:legacy"
    assert stored["page_hash"] == "a" * 64 and stored["vlm_model"] == MODEL
    assert stored["rule_version"] == "lines-v2-20260909" and stored["decoded_at"] == "2026-09-28T20:52:04Z"
    assert stored["note"] is None and stored["contributor"] == "A. Scholar"
    assert set(es.indices.created[ar.DEFAULT_CORRECTIONS_INDEX]["mappings"]["properties"]) >= set(stored) - {"id"}

    listed = client.get("/documents/Cambridge_CUL_Or_1080_15_50/readings/corrections?image_index=1", headers=H)
    assert listed.headers["cache-control"] == "no-store"
    assert [c["id"] for c in listed.json()["corrections"]] == [stored["id"]]

    exported = client.get("/admin/corrections.jsonl", headers={"X-Export-Token": EXPORT})
    assert exported.headers["content-type"].startswith("application/x-ndjson")
    rows = [json.loads(line) for line in exported.text.splitlines()]
    assert rows == [stored]
    later = client.get("/admin/corrections.jsonl?since=2999-01-01T00:00:00Z", headers={"X-Export-Token": EXPORT})
    assert later.text == ""


def test_correction_of_a_reader_line(es: FakeES) -> None:
    body = {**CORRECTION, "reader_id": KRAKEN}
    stored = _client(es).post("/documents/Cambridge_CUL_Or_1080_15_50/readings/corrections",
                              headers=H, json=body).json()["correction"]
    assert stored["source_reader"] == KRAKEN and stored["original_text"] == "שלום עליכס"


@pytest.mark.parametrize("change,status", [
    ({"line_index": 7}, 422),
    ({"reader_id": "vlm:nobody:page"}, 422),
    ({"corrected_text": "   "}, 422),
    ({"contributor": ""}, 422),
    ({"image_index": 5}, 404),
    ({"vlm_model": OTHER_MODEL}, 404),
])
def test_bad_corrections_are_rejected(es: FakeES, change: Dict[str, Any], status: int) -> None:
    response = _client(es).post("/documents/Cambridge_CUL_Or_1080_15_50/readings/corrections",
                                headers=H, json={**CORRECTION, **change})
    assert response.status_code == status
    assert not es.store.get(ar.DEFAULT_CORRECTIONS_INDEX)


def test_corrections_are_rate_limited(es: FakeES) -> None:
    client = _client(es, per_ip=2)
    path = "/documents/Cambridge_CUL_Or_1080_15_50/readings/corrections"
    assert [client.post(path, headers=H, json=CORRECTION).status_code for _ in range(2)] == [201, 201]
    limited = client.post(path, headers=H, json=CORRECTION)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) > 0
    other_ip = client.post(path, headers={**H, "CF-Connecting-IP": "203.0.113.9"}, json=CORRECTION)
    assert other_ip.status_code == 201


def test_rate_limiter_window_slides() -> None:
    clock = [0.0]
    limiter = ar.RateLimiter(1, window_s=10, clock=lambda: clock[0])
    assert limiter.allow("ip") == (True, 0)
    allowed, retry = limiter.allow("ip")
    assert not allowed and retry == 11
    clock[0] = 10.0
    assert limiter.allow("ip") == (True, 0)


# --- sidecar helpers --------------------------------------------------------


def test_fixtures_validate_and_broken_sidecars_do_not() -> None:
    assert ar.validate_sidecar(legacy_sidecar()) == []
    assert ar.validate_sidecar(columns_sidecar()) == []
    broken = legacy_sidecar()
    broken["readers"][0]["lines"][0]["box"] = [1, 2, 3]
    broken["final"]["lines"][0].pop("bbox")
    del broken["vlm_model"]
    problems = ar.validate_sidecar(broken)
    assert "missing vlm_model" in problems
    assert any("without a 4-number box" in p for p in problems)
    assert any("without a 4-number bbox" in p for p in problems)
    assert ar.validate_sidecar([]) == ["sidecar is not a JSON object"]


def test_readings_document_fits_the_strict_mapping() -> None:
    document = ar.readings_document(columns_sidecar())
    properties = ar.READINGS_INDEX_MAPPING["mappings"]["properties"]
    assert ar.READINGS_INDEX_MAPPING["mappings"]["dynamic"] == "strict"
    assert set(document) <= set(properties)
    assert properties["readings"] == {"type": "object", "enabled": False}
    assert document["policy_action"] == "columns" and document["n_readers"] == 8
    assert ar.readings_document(legacy_sidecar())["policy_action"] is None
    assert ar.readings_document_id(columns_sidecar()) == f"Manchester_JRL_C_131__2__{MODEL}"


def test_resolve_line_handles_final_readers_and_unattributed() -> None:
    sidecar = columns_sidecar()
    assert ar.resolve_line(sidecar, "final", 1)["source_reader"] == "final:whole_page"
    assert ar.resolve_line(sidecar, f"vlm:{MODEL}:col1.top:crop", 0) is None  # lines: null
    assert ar.resolve_line(sidecar, f"vlm:{MODEL}:page", 0)["box"] == [100, 100, 900, 150]
    assert ar.resolve_line(sidecar, "final", 99) is None
