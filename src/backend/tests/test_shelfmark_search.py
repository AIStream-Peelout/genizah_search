"""Tests for shelf-mark query building and relevance ranking (no Elasticsearch)."""

from src.backend.search_service import ElasticsearchService


def _service() -> ElasticsearchService:
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.index_name = "test"
    return svc


class TestHelpers:
    def test_key_collapses_punctuation_but_keeps_digit_groups(self):
        assert ElasticsearchService.shelfmark_key("T-S B11.93") == "T_S_B11_93"
        assert ElasticsearchService.shelfmark_key("L-G Ar. II.70") == ElasticsearchService.shelfmark_key("L-G Ar.II.70")
        assert ElasticsearchService.shelfmark_key("T-S B11.93") != ElasticsearchService.shelfmark_key("T-S B1.193")
        assert ElasticsearchService.shelfmark_key("") == ""

    def test_regex_is_spacing_tolerant_and_escapes(self):
        rx = ElasticsearchService.shelfmark_regex("L-G Ar. II.70")
        assert rx == r"L-G ?Ar\. ? ?II\. ?70"
        assert ElasticsearchService.shelfmark_regex("MS heb. e.39/163") == r"MS ?heb\. ? ?e\. ?39/163"
        assert ElasticsearchService.shelfmark_regex("a+b") == r"a\+b"


class TestQuery:
    def test_exact_mode_has_no_fuzzy_clauses(self):
        q = _service()._build_shelfmark_query("T-S B11.93", ["TS B11.93"], exact_match=True)
        clauses = q["bool"]["should"]
        assert not any("match" in c for c in clauses)
        terms = [c["term"]["shelf_mark"]["value"] for c in clauses if "term" in c and "shelf_mark" in c["term"]]
        assert terms == ["T-S B11.93", "TS B11.93"]
        regexes = [c["regexp"]["shelf_mark"]["value"] for c in clauses if "regexp" in c]
        assert regexes[0] == r"(.*[ :])?T-S ?B11\. ?93"
        wild = [c["wildcard"]["doc_id"]["value"] for c in clauses if "wildcard" in c]
        assert wild == ["*_T_S_B11_93", "*_TS_B11_93"]

    def test_partial_mode_adds_low_boost_neighbours(self):
        q = _service()._build_shelfmark_query("T-S B11.93", [], exact_match=False)
        clauses = q["bool"]["should"]
        fuzzy = [c for c in clauses if "match" in c]
        assert fuzzy and fuzzy[0]["match"]["shelf_mark"]["boost"] < 5
        strong = max(c["term"]["shelf_mark"]["boost"] for c in clauses if "term" in c and "shelf_mark" in c["term"])
        assert strong > fuzzy[0]["match"]["shelf_mark"]["boost"]


class TestRelevance:
    def test_exact_and_suffix_beat_neighbours(self):
        svc = _service()
        exact = svc._calculate_shelfmark_relevance({"shelf_mark": "Cambridge CUL: T-S B11.93"}, "T-S B11.93", False)
        by_id = svc._calculate_shelfmark_relevance({"doc_id": "Cambridge_CUL_T_S_B11_93"}, "T-S B11.93", False)
        neighbour = svc._calculate_shelfmark_relevance({"shelf_mark": "T-S H11.90"}, "T-S B11.93", False)
        assert exact == 1.0
        assert by_id >= 0.95
        assert neighbour < 0.5
        assert exact > by_id > neighbour

    def test_spacing_variants_match(self):
        svc = _service()
        # "Cambridge Lewis-Gibson:" is not a stripped prefix, so this is a suffix match; still outranks neighbours.
        assert svc._calculate_shelfmark_relevance({"shelf_mark": "Cambridge Lewis-Gibson: L-G Ar.II.70"}, "L-G Ar. II.70", False) >= 0.95
        assert svc._calculate_shelfmark_relevance({"shelf_mark": "L-G Ar.II.70"}, "L-G Ar. II.70", False) == 1.0

    def test_partial_containment(self):
        svc = _service()
        assert 0.5 < svc._calculate_shelfmark_relevance({"shelf_mark": "T-S B11.93.2"}, "B11.93", False) < 1.0
