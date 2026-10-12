"""Tests for collection-browser hierarchy helpers (no live Elasticsearch)."""

from typing import Any, Dict

from src.backend.search_service import ElasticsearchService


class FakeIndices:
    def __init__(self, props: Dict[str, Any]):
        self.props = props

    def get_mapping(self, index: str) -> Dict[str, Any]:
        return {index: {"mappings": {"properties": self.props}}}


class FakeES:
    def __init__(self, props: Dict[str, Any]):
        self.indices = FakeIndices(props)


def _service(props: Dict[str, Any]) -> ElasticsearchService:
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.es = FakeES(props)
    return svc


class TestAggregatableField:
    def test_prefers_keyword_subfield_then_keyword_type(self):
        svc = _service({
            "collection": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "shelf_mark": {"type": "keyword"},
            "description": {"type": "text"},
        })
        assert svc._aggregatable_field("idx", "collection") == "collection.keyword"
        assert svc._aggregatable_field("idx", "shelf_mark") == "shelf_mark"
        assert svc._aggregatable_field("idx", "description") is None
        assert svc._aggregatable_field("idx", "missing") is None

    def test_result_is_cached_per_index(self):
        svc = _service({"shelf_mark": {"type": "keyword"}})
        assert svc._aggregatable_field("idx", "shelf_mark") == "shelf_mark"
        svc.es.indices.props = {}
        assert svc._aggregatable_field("idx", "shelf_mark") == "shelf_mark"
        assert svc._aggregatable_field("other", "shelf_mark") is None


class TestPrune:
    def test_drops_dead_subcollections_and_collections(self):
        hierarchy = {
            "Live": {
                "name": "Live",
                "count": 3,
                "sub_collections": {
                    "A": {"name": "A", "count": 2, "shelfmarks": [{"name": "x", "count": 2}]},
                    "B": {"name": "B", "count": 1, "shelfmarks": []},
                },
            },
            "Ranges": {
                "name": "Ranges",
                "count": 2000,
                "sub_collections": {"_all": {"name": "All", "count": 2000, "shelfmarks": [], "sub_sub_collections": {"1-100": {}}}},
            },
            "Dead": {"name": "Dead", "count": 5, "sub_collections": {}},
            "DeadSubs": {"name": "DeadSubs", "count": 5, "sub_collections": {"Z": {"name": "Z", "count": 5, "shelfmarks": []}}},
        }
        pruned = ElasticsearchService.prune_unbrowsable(hierarchy)
        assert set(pruned) == {"Live", "Ranges"}
        assert set(pruned["Live"]["sub_collections"]) == {"A"}
        assert pruned["Live"]["count"] == 3
