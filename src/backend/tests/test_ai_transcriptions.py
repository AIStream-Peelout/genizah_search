"""Tests for the AI read schema, agreement rule, surfacing rule and service."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from elasticsearch import NotFoundError

from src.backend import ai_transcriptions as at

DOC = "Cambridge_CUL_T_S_8J22_22"
SURF = {"min_agreed_lines": 3, "min_agreed_share": 0.25}


def _sidecar(agreements=(0.86, 0.22, None, 0.95, 0.8), **overrides: Any) -> Dict[str, Any]:
    lines = []
    for i, a in enumerate(agreements):
        lines.append(
            {
                "text": f"שורה {i}",
                "bbox": [104, 100 + i * 100, 928, 180 + i * 100],
                "agreement": a,
                "htr_text": f"שורה {i}" if a is not None else None,
                "htr_fragments": [[104, 100 + i * 100, 500, 180 + i * 100]] if a is not None else [],
            }
        )
    base: Dict[str, Any] = {
        "vlm_model": "qwen3-vl-8b-heb-v20a-step1800",
        "vlm_revision": "af9df6a0",
        "htr_model": "MiDRASH_Gen_01",
        "rule_version": at.RULE_VERSION,
        "decoded_at": "2026-09-08T16:31:00",
        "parsed": True,
        "lines": at.build_lines(lines),
    }
    base.update(overrides)
    return base


def _raw_record(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "doc_id": DOC,
        "source_index": "genizah_merged_v4",
        "image_index": 0,
        "image_url": "https://example.org/img.jpg",
        "image_width": 2800,
        "image_height": 1785,
        "ai_read": _sidecar(),
    }
    base.update(overrides)
    return base


class TestAgreementRule:
    def test_threshold_is_0_8(self):
        assert at.line_status(0.8, "x") == "agreed"
        assert at.line_status(0.79, "x") == "unconfirmed"
        assert at.line_status(1.0, "x") == "agreed"

    def test_no_kraken_read_is_unconfirmed(self):
        assert at.line_status(0.95, None) == "unconfirmed"
        assert at.line_status(0.95, " ") == "unconfirmed"
        assert at.line_status(None, "x") == "unconfirmed"

    def test_build_lines_fills_index_and_status(self):
        lines = _sidecar()["lines"]
        assert [line.index for line in lines] == [0, 1, 2, 3, 4]
        assert [line.status for line in lines] == ["agreed", "unconfirmed", "unconfirmed", "agreed", "agreed"]

    def test_pipeline_status_must_match_rule(self):
        with pytest.raises(ValueError, match="rule"):
            at.AiLine(index=0, text="x", bbox=[0, 0, 10, 10], agreement=0.5, htr_text="y", status="agreed")

    def test_unknown_rule_version_rejected(self):
        with pytest.raises(ValueError, match="rule_version"):
            at.AiRead(**_sidecar(rule_version="lines-v0"))

    def test_both_known_rule_versions_accepted(self):
        for version in ("lines-v1-20260908", "lines-v2-20260909"):
            assert at.AiRead(**_sidecar(rule_version=version)).rule_version == version


class TestSurfacing:
    def test_rule(self):
        assert at.is_surfaced(True, 16, 3, SURF)
        assert not at.is_surfaced(True, 16, 2, SURF)
        assert at.is_surfaced(True, 8, 2, SURF)  # 25% share
        assert not at.is_surfaced(True, 9, 2, SURF)
        assert not at.is_surfaced(False, 16, 10, SURF)
        assert not at.is_surfaced(True, 0, 0, SURF)

    def test_record_derives_surfaced_text_and_counts(self, monkeypatch):
        monkeypatch.delenv("AI_READ_MIN_AGREED_LINES", raising=False)
        record = at.AiTranscriptionRecord(**_raw_record())
        assert record.ai_read.n_lines == 5 and record.ai_read.n_agreed == 3
        assert record.surfaced is True
        assert record.text_all.startswith("שורה 0\nשורה 1")
        assert record.text_agreed == "שורה 0\nשורה 3\nשורה 4"  # agreed lines only
        assert record.read_key == f"{DOC}__0"
        assert record.es_id() == f"{DOC}__0__qwen3-vl-8b-heb-v20a-step1800"

    def test_v1_record_with_text_field_still_parses(self):
        record = at.AiTranscriptionRecord(**_raw_record(text="stale joined text"))
        assert not hasattr(record, "text")
        assert record.text_all.startswith("שורה 0")

    def test_zero_agreed_lines_means_empty_text_agreed_and_not_surfaced(self, monkeypatch):
        monkeypatch.delenv("AI_READ_MIN_AGREED_LINES", raising=False)
        record = at.AiTranscriptionRecord(**_raw_record(ai_read=_sidecar(agreements=(0.1, None, 0.5))))
        assert record.text_agreed == ""
        assert record.text_all == "שורה 0\nשורה 1\nשורה 2"
        assert record.surfaced is False  # and search always filters surfaced=True

    def test_inconsistent_counts_rejected(self):
        with pytest.raises(ValueError, match="n_lines/n_agreed"):
            at.AiRead(**_sidecar(n_lines=5, n_agreed=1))


class TestSchema:
    def test_bbox_is_clamped_and_ordered(self):
        assert at.clamp_bbox([1200, -5, 100, 50.4]) == [100, 0, 1000, 50]
        with pytest.raises(ValueError):
            at.clamp_bbox([1, 2, 3])

    def test_mapping_covers_every_field(self):
        props = at.INDEX_MAPPING["mappings"]["properties"]
        assert set(at.AiTranscriptionRecord.model_fields) == set(props)
        assert set(at.AiRead.model_fields) == set(props["ai_read"]["properties"])
        lines = props["ai_read"]["properties"]["lines"]
        assert lines["type"] == "nested"
        assert set(at.AiLine.model_fields) == set(lines["properties"])

    def test_v2_mapping_search_fields(self):
        props = at.INDEX_MAPPING["mappings"]["properties"]
        line_text = props["ai_read"]["properties"]["lines"]["properties"]["text"]
        assert line_text["analyzer"] == "hebrew_fold"
        assert line_text["fields"]["confusable"]["analyzer"] == "hebrew_confusable"
        for field in ("text_agreed", "text_all"):
            assert props[field]["analyzer"] == "hebrew_fold"
        assert "text" not in props  # v1's joined field is gone in v2
        assert props["read_key"] == {"type": "keyword"}
        line_props = props["ai_read"]["properties"]["lines"]["properties"]
        assert line_props["htr_text"]["index"] is False and line_props["htr_fragments"]["index"] is False
        analyzers = at.INDEX_MAPPING["settings"]["analysis"]["analyzer"]
        assert analyzers["hebrew_fold"]["char_filter"] == ["hebrew_strip_points", "hebrew_final_forms"]
        assert analyzers["hebrew_confusable"]["char_filter"][-1] == "hebrew_confusables"


class TestHebrewFolding:
    """The Python mirror and the ES char_filter rules are built from one table."""

    def test_pointed_word_folds_to_bare_consonants(self):
        assert at.fold_hebrew("בְּרֵאשִׁית") == "בראשית"
        assert at.fold_hebrew("שָׁלוֹם") == "שלומ"  # points gone, final mem folded

    def test_final_forms_fold(self):
        assert at.fold_hebrew("ךםןףץ") == "כמנפצ"

    def test_confusable_fold_is_opt_in(self):
        assert at.fold_hebrew("דבר כסף", confusable=False) == "דבר כספ"
        assert at.fold_hebrew("דבר כסף", confusable=True) == "דבד במפ"
        # samekh and final mem land on the same letter, as the readers confuse them
        assert at.fold_hebrew("ס", confusable=True) == at.fold_hebrew("ם", confusable=True)

    def test_es_rules_mirror_python_tables(self):
        cf = at.ANALYSIS["char_filter"]
        assert cf["hebrew_final_forms"]["mappings"] == [f"{a}=>{b}" for a, b in at.FINAL_FORMS.items()]
        assert cf["hebrew_confusables"]["mappings"] == [f"{a}=>{b}" for a, b in at.CONFUSABLES.items()]
        assert cf["hebrew_strip_points"]["pattern"] == "[\\u0591-\\u05C7]"


class TestSearchQuery:
    @staticmethod
    def _nested(q):
        (nested,) = q["bool"]["must"]
        return nested["nested"]

    def test_default_searches_agreed_lines_only(self):
        q = at.build_search_query("שלום")
        assert {"term": {"surfaced": True}} in q["bool"]["filter"]
        assert {"term": {"published": True}} in q["bool"]["filter"]
        nested = self._nested(q)
        assert nested["path"] == "ai_read.lines"
        assert nested["query"]["bool"]["filter"] == [{"term": {"ai_read.lines.status": "agreed"}}]
        assert q["bool"]["should"] == [{"match_phrase": {"text_agreed": {"query": "שלום", "boost": 2.0}}}]

    def test_include_unconfirmed_widens(self):
        q = at.build_search_query("שלום", include_unconfirmed=True)
        assert "filter" not in self._nested(q)["query"]["bool"]
        assert q["bool"]["should"] == [{"match_phrase": {"text_all": {"query": "שלום", "boost": 2.0}}}]

    def test_clause_order_and_fuzziness(self):
        should = self._nested(at.build_search_query("שלום"))["query"]["bool"]["should"]
        phrase, terms, confusable = should
        assert phrase["match_phrase"]["ai_read.lines.text"]["boost"] == 3.0
        assert terms["match"]["ai_read.lines.text"] == {
            "query": "שלום", "operator": "and", "fuzziness": "AUTO:4,99", "prefix_length": 1,
        }  # at most one edit, none for tokens under four letters
        assert confusable["match"]["ai_read.lines.text.confusable"]["boost"] < 1
        assert "fuzziness" not in confusable["match"]["ai_read.lines.text.confusable"]

    def test_inner_hits_shape(self):
        inner = self._nested(at.build_search_query("x"))["inner_hits"]
        assert inner["name"] == "lines" and inner["size"] == 3
        assert set(inner["highlight"]["fields"]) == {"ai_read.lines.text", "ai_read.lines.text.confusable"}
        assert inner["highlight"]["number_of_fragments"] == 0

    def test_model_pin(self):
        q = at.build_search_query("x", vlm_model="v21b")
        assert {"term": {"ai_read.vlm_model": "v21b"}} in q["bool"]["filter"]


def _search_hit(**overrides: Any) -> Dict[str, Any]:
    """One ES hit as the v2 index returns it (collapsed, with nested inner hits)."""
    hit = {
        "_id": f"{DOC}__0__v21b",
        "_score": 4.2,
        "_source": {
            "doc_id": DOC, "source_index": "genizah_merged_v5", "image_index": 0,
            "image_url": "https://example.org/img.jpg",
            "ai_read": {"vlm_model": "v21b", "n_lines": 5, "n_agreed": 3},
        },
        "inner_hits": {"lines": {"hits": {"hits": [
            {
                "_source": {"index": 3, "text": "שורה 3", "agreement": 0.95, "status": "agreed", "bbox": [104, 400, 928, 480]},
                "highlight": {"ai_read.lines.text": ["<em>שורה</em> 3"]},
            },
            {
                "_source": {"index": 0, "text": "שורה 0", "agreement": 0.86, "status": "agreed", "bbox": [104, 100, 928, 180]},
                "highlight": {"ai_read.lines.text.confusable": ["<em>שורה</em> 0"]},
            },
        ]}}},
    }
    hit.update(overrides)
    return hit


def _search_response(hits: List[Dict[str, Any]], images: int, total_records: int) -> Dict[str, Any]:
    return {
        "hits": {"total": {"value": total_records}, "hits": hits},
        "aggregations": {
            "images": {"value": images},
            "models": {"buckets": [{"key": "v21b", "doc_count": total_records - 1}, {"key": "v20a", "doc_count": 1}]},
        },
    }


class FakeSearchES:
    """Records ``search``/``mget`` kwargs and returns canned v2-shaped responses."""

    def __init__(self, response: Any, labels: Dict[str, Dict[str, Any]]):
        self.response = response
        self.labels = labels
        self.search_calls: List[Dict[str, Any]] = []
        self.mget_calls: List[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        self.search_calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def mget(self, **kwargs: Any) -> Dict[str, Any]:
        self.mget_calls.append(kwargs)
        return {"docs": [
            {"_id": i, "found": i in self.labels, "_source": self.labels.get(i, {})} for i in kwargs["ids"]
        ]}


class TestParseLineHits:
    def test_highlight_prefers_folded_field_then_confusable(self):
        lines = at.parse_line_hits(_search_hit())
        assert [l.index for l in lines] == [3, 0]
        assert lines[0].highlight == "<em>שורה</em> 3"
        assert lines[1].highlight == "<em>שורה</em> 0"  # confusable fallback
        assert lines[0].bbox == [104, 400, 928, 480] and lines[0].status == "agreed"

    def test_no_inner_hits(self):
        assert at.parse_line_hits({"_source": {}}) == []


class TestSearchService:
    LABELS = {DOC: {"shelf_mark": "Cambridge CUL: T-S 8J22.22", "title": "Letter", "description": "A letter."}}

    def _service(self, response, monkeypatch, labels=None):
        monkeypatch.delenv("AI_TRANSCRIPTIONS_ENABLED", raising=False)
        fake = FakeSearchES(response, self.LABELS if labels is None else labels)
        return fake, at.AiTranscriptionService(fake, index="ai_v2", catalogue_index="merged")

    def test_shapes_hits_and_joins_catalogue(self, monkeypatch):
        fake, service = self._service(_search_response([_search_hit()], images=7, total_records=9), monkeypatch)
        out = service.search("שורה", limit=1, offset=0)
        assert out.enabled and out.available and out.index_name == "ai_v2"
        assert out.total == 7 and out.total_records == 9 and out.has_more is True
        assert [(m.vlm_model, m.count) for m in out.models] == [("v21b", 8), ("v20a", 1)]
        (hit,) = out.results
        assert hit.doc_id == DOC and hit.image_index == 0 and hit.vlm_model == "v21b"
        assert hit.n_agreed == 3 and hit.score == 4.2
        assert hit.shelf_mark == "Cambridge CUL: T-S 8J22.22" and hit.title == "Letter"
        assert [l.index for l in hit.lines] == [3, 0]
        # one mget per page, against the catalogue index
        assert len(fake.mget_calls) == 1 and fake.mget_calls[0]["index"] == "merged"
        call = fake.search_calls[0]
        assert call["collapse"] == {"field": "read_key"} and call["from_"] == 0 and call["size"] == 1
        assert call["query"] == at.build_search_query("שורה")

    def test_unlabelled_when_catalogue_misses(self, monkeypatch):
        fake, service = self._service(_search_response([_search_hit()], 1, 1), monkeypatch, labels={})
        (hit,) = service.search("x").results
        assert hit.shelf_mark is None and hit.title is None

    def test_no_catalogue_index_skips_join(self, monkeypatch):
        monkeypatch.delenv("AI_TRANSCRIPTIONS_ENABLED", raising=False)
        monkeypatch.delenv("ELASTICSEARCH_INDEX", raising=False)
        fake = FakeSearchES(_search_response([_search_hit()], 1, 1), {})
        out = at.AiTranscriptionService(fake, index="ai_v2", catalogue_index=None).search("x")
        assert len(out.results) == 1 and fake.mget_calls == []

    def test_missing_index_is_unavailable(self, monkeypatch):
        err = NotFoundError("index_not_found_exception", MagicMock(status=404), {})
        _, service = self._service(err, monkeypatch)
        out = service.search("x")
        assert out.enabled and not out.available and out.results == [] and "does not exist" in out.message

    def test_v1_mapping_is_unavailable_not_an_error(self, monkeypatch):
        from elasticsearch import BadRequestError
        err = BadRequestError("query_shard_exception", MagicMock(status=400), {})
        _, service = self._service(err, monkeypatch)
        out = service.search("x")
        assert out.enabled and not out.available and "not searchable" in out.message

    def test_feature_flag_off_skips_es(self, monkeypatch):
        fake, service = self._service(_search_response([_search_hit()], 1, 1), monkeypatch)
        monkeypatch.setenv("AI_TRANSCRIPTIONS_ENABLED", "false")
        out = service.search("x")
        assert not out.enabled and not out.available and out.results == []
        assert fake.search_calls == []


class TestSearchEndpoint:
    """``POST /search-ai-transcriptions`` with the ES client mocked."""

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient
        from src.backend import app as app_module

        monkeypatch.delenv("AI_TRANSCRIPTIONS_ENABLED", raising=False)
        fake = FakeSearchES(_search_response([_search_hit()], images=1, total_records=2), TestSearchService.LABELS)
        monkeypatch.setattr(
            app_module, "ai_transcription_service",
            at.AiTranscriptionService(fake, index="ai_v2", catalogue_index="merged"),
        )
        return TestClient(app_module.app), fake

    def test_search(self, client):
        tc, fake = client
        resp = tc.post("/search-ai-transcriptions", json={"query": "שורה", "limit": 5})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["enabled"] and body["available"] and body["total"] == 1 and body["has_more"] is False
        (hit,) = body["results"]
        assert hit["doc_id"] == DOC and hit["lines"][0]["highlight"] == "<em>שורה</em> 3"
        assert hit["lines"][0]["bbox"] == [104, 400, 928, 480]
        assert fake.search_calls[0]["query"] == at.build_search_query("שורה")

    def test_include_unconfirmed_and_model_are_forwarded(self, client):
        tc, fake = client
        resp = tc.post("/search-ai-transcriptions", json={"query": "x", "include_unconfirmed": True, "vlm_model": "v20a"})
        assert resp.status_code == 200
        assert fake.search_calls[0]["query"] == at.build_search_query("x", include_unconfirmed=True, vlm_model="v20a")

    def test_validation(self, client):
        tc, _ = client
        assert tc.post("/search-ai-transcriptions", json={"query": ""}).status_code == 422
        assert tc.post("/search-ai-transcriptions", json={"query": "x", "limit": 51}).status_code == 422

    def test_disabled_shape_matches_status_endpoint(self, client, monkeypatch):
        tc, _ = client
        monkeypatch.setenv("AI_TRANSCRIPTIONS_ENABLED", "false")
        search = tc.post("/search-ai-transcriptions", json={"query": "x"}).json()
        status = tc.get(f"/ai-transcriptions/{DOC}").json()
        assert search["enabled"] is False and search["available"] is False and search["results"] == []
        assert status["enabled"] is False and status["available"] is False


class FakeES:
    """Minimal stand-in for ``Elasticsearch.search`` over an in-memory list."""

    def __init__(self, docs: List[Dict[str, Any]], exists: bool = True):
        self.docs = docs
        self.exists = exists
        self.calls: List[Dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        if not self.exists:
            raise NotFoundError("index_not_found_exception", MagicMock(status=404), {})
        filters = kwargs["query"]["bool"]["filter"]
        hits = [d for d in self.docs if all(self._match(d, f["term"]) for f in filters)]
        hits.sort(key=lambda d: d["created_at"], reverse=True)
        excludes = kwargs.get("source_excludes", [])
        return {"hits": {"hits": [{"_source": self._strip(d, excludes)} for d in hits[: kwargs.get("size", 10)]]}}

    @staticmethod
    def _strip(doc: Dict[str, Any], excludes: List[str]) -> Dict[str, Any]:
        out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in doc.items()}
        for path in excludes:
            node = out
            *parents, leaf = path.split(".")
            for p in parents:
                node = node[p]
            node.pop(leaf, None)
        return out

    @staticmethod
    def _match(doc: Dict[str, Any], term: Dict[str, Any]) -> bool:
        (path, value), = term.items()
        node: Any = doc
        for part in path.split("."):
            node = node[part]
        return node == value


def _doc(**overrides: Any) -> Dict[str, Any]:
    record = at.AiTranscriptionRecord(**_raw_record(**overrides))
    return record.model_dump(mode="json")


class TestService:
    def test_missing_index_means_not_available(self, monkeypatch):
        monkeypatch.delenv("AI_TRANSCRIPTIONS_ENABLED", raising=False)
        service = at.AiTranscriptionService(FakeES([], exists=False), index="x")
        status = service.status("doc")
        assert status.enabled and not status.available and status.items == []
        assert service.get("doc", 0) is None

    def test_status_hides_unsurfaced_and_unpublished(self, monkeypatch):
        monkeypatch.delenv("AI_TRANSCRIPTIONS_ENABLED", raising=False)
        docs = [
            _doc(),
            _doc(image_index=1, ai_read=_sidecar(agreements=(0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))),
            _doc(image_index=2, ai_read=_sidecar(parsed=False)),
            _doc(image_index=3, published=False),
        ]
        service = at.AiTranscriptionService(FakeES(docs), index="x")
        status = service.status(DOC)
        assert status.available
        assert [item.image_index for item in status.items] == [0]
        assert status.items[0].ai_read.n_agreed == 3
        assert service.get(DOC, 1) is None

    def test_get_returns_newest_unless_model_pinned(self, monkeypatch):
        monkeypatch.delenv("AI_TRANSCRIPTIONS_ENABLED", raising=False)
        now = datetime.now(timezone.utc)
        old = _doc(ai_read=_sidecar(vlm_model="v19a"), created_at=now - timedelta(days=1))
        new = _doc(ai_read=_sidecar(vlm_model="v20a"), created_at=now)
        service = at.AiTranscriptionService(FakeES([old, new]), index="x")
        assert service.get(DOC, 0).ai_read.vlm_model == "v20a"
        assert service.get(DOC, 0, vlm_model="v19a").ai_read.vlm_model == "v19a"

    def test_feature_flag_off(self, monkeypatch):
        monkeypatch.setenv("AI_TRANSCRIPTIONS_ENABLED", "false")
        fake = FakeES([_doc()])
        status = at.AiTranscriptionService(fake, index="x").status(DOC)
        assert not status.enabled and not status.available
        assert fake.calls == []
