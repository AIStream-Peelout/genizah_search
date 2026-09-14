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
        assert record.text.startswith("שורה 0\nשורה 1")
        assert record.es_id() == f"{DOC}__0__qwen3-vl-8b-heb-v20a-step1800"

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
