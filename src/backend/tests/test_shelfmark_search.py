"""Tests for shelf-mark query building and relevance ranking (no Elasticsearch)."""

import asyncio
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.backend.search_service import SHELFMARK_CHILD_BOOST, ElasticsearchService
from src.backend.semantic_beta import ShelfmarkRouteRequest


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


# ---------------------------------------------------------------------------
# Routed beta queries (prefixes, volumes, paging) over real v10 record shapes
# ---------------------------------------------------------------------------

V8 = "genizah_merged_v8"
V10 = "genizah_merged_v10"
# A rare keyword value's BM25 weight. On v10, fuzzy neighbours ("T-S NS J2"
# for "T-S NS 32") outscored the boost-4 contains regexp the same way.
FUZZY_SCORE = 8.0
# (shelf_mark, doc_id) exactly as stored in genizah_merged_v10 (read 2026-10-11).
V10_RECORDS = [
    ("Paris Mosseri: Moss. VIII399.1", "Cambridge_Mosseri_VIII399_1"),
    ("Moss. VIII,111.2", "Cambridge_Mosseri_VIII_111_2"),
    ("Paris Mosseri: Moss. VI145.2", "Cambridge_Mosseri_VI145_2"),
    ("Moss. VII,157", "Cambridge_Mosseri_VII_157"),
    ("Moss. VII,5", "Cambridge_Mosseri_VII_5"),
    ("Moss. V,8", "Cambridge_Mosseri_V_8"),
    ("Moss. V,105.4", "Cambridge_Mosseri_V_105_4"),
    ("Moss. Ia,10.1", "Cambridge_Mosseri_Ia_10_1"),
    ("Moss. I,101", "Cambridge_Mosseri_I_101"),
    ("Mosseri, Jacques, Paris, France Ms. II 1", "Cambridge_Mosseri_II_1"),
    ("T-S NS J2", "Cambridge_CUL_T_S_NS_J2"),
    ("T-S NS J32", "Cambridge_CUL_T_S_NS_J32"),
    ("T-S NS 320.1", "Cambridge_CUL_T_S_NS_320_1"),
    ("T-S NS 32.100", "Cambridge_CUL_T_S_NS_32_100"),
    ("Cambridge University Library, Cambridge, England Ms. T-S NS 32.11", "Cambridge_CUL_T_S_NS_32_11"),
    ("T-S NS 32.7", "Cambridge_CUL_T_S_NS_32_7"),
    ("Cambridge CUL: T-S NS 32", "Cambridge_CUL_T_S_NS_32"),
]
NS32_CHILDREN = {"Cambridge_CUL_T_S_NS_32_7", "Cambridge_CUL_T_S_NS_32_11", "Cambridge_CUL_T_S_NS_32_100"}


def _levenshtein(a: str, b: str) -> int:
    """Edit distance.

    :param a: First string.
    :param b: Second string.
    :return: Number of single-character edits.
    """
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


class ScoringES:
    """Elasticsearch stand-in that scores shelf-mark ``bool`` queries over stored records.

    As in Elasticsearch, ``term`` / ``regexp`` / ``wildcard`` clauses are
    constant-score (their boost) and a document's score is the sum of its
    matching ``should`` clauses; a fuzzy ``match`` (AUTO fuzziness on the
    keyword value) adds ``FUZZY_SCORE`` x boost. Regexps are anchored (Python
    ``fullmatch`` reads the constructs the query uses the same way Lucene does).
    Supports ``from_`` / ``size``, the ``doc_id`` tie-break sort, and ``terms``
    lookups on ``doc_id``.
    """

    def __init__(self, records: Dict[str, List[Dict[str, str]]]) -> None:
        """Store the records.

        :param records: ``{index: [_source, ...]}``.
        """
        self.records = records
        self.calls: List[Dict[str, Any]] = []

    @staticmethod
    def _clause_score(clause: Dict[str, Any], source: Dict[str, str]) -> float:
        """Score of one leaf clause for one record.

        :param clause: ``{"term"|"regexp"|"wildcard"|"match": {field: spec}}``.
        :param source: The record.
        :return: The clause's contribution (0 when it does not match).
        """
        (kind, body), = clause.items()
        (field, spec), = body.items()
        value = source.get(field, "")
        flags = re.IGNORECASE if spec.get("case_insensitive") else 0
        if kind == "term":
            matched = value == spec["value"]
        elif kind == "regexp":
            matched = re.fullmatch(spec["value"], value, flags) is not None
        elif kind == "wildcard":
            pattern = re.escape(spec["value"]).replace(r"\*", ".*").replace(r"\?", ".")
            matched = re.fullmatch(pattern, value, flags) is not None
        else:
            query = spec["query"]
            allowed = 0 if len(query) <= 2 else 1 if len(query) <= 5 else 2
            return FUZZY_SCORE * spec["boost"] if _levenshtein(query, value) <= allowed else 0.0
        return float(spec["boost"]) if matched else 0.0

    def search(self, index: str, query: Dict[str, Any], size: int = 10, from_: int = 0,
               sort: Optional[List[Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Run one search.

        :param index: Index name.
        :param query: Shelf-mark ``bool`` query, or a ``terms`` lookup on ``doc_id``.
        :param size: Page size.
        :param from_: Offset.
        :param sort: When given, ties are broken by ``doc_id`` (as the routed request asks).
        :param kwargs: Ignored (``_source``, ``track_scores``).
        :return: An Elasticsearch-shaped response.
        """
        self.calls.append(dict(index=index, query=query, size=size, from_=from_, sort=sort))
        records = self.records.get(index, [])
        if "terms" in query:
            wanted = set(query["terms"]["doc_id"])
            scored = [(1.0, pos, src) for pos, src in enumerate(records) if src["doc_id"] in wanted]
        else:
            should = query.get("bool", {}).get("must", [query])[0]["bool"]["should"]
            scored = [(sum(self._clause_score(c, src) for c in should), pos, src) for pos, src in enumerate(records)]
            scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[2]["doc_id"] if sort else item[1]))
        page = scored[from_:from_ + size]
        return {"hits": {"total": {"value": len(scored), "relation": "eq"},
                         "hits": [{"_id": src["doc_id"], "_score": score, "_source": dict(src)}
                                  for score, _, src in page]}}


def _scoring_service(catalogue_ids: Optional[set] = None) -> ElasticsearchService:
    """Search service over :class:`ScoringES` holding :data:`V10_RECORDS`.

    :param catalogue_ids: Doc ids also present in the catalogue index (default: all).
    :return: The service; its catalogue index is genizah_merged_v8.
    """
    v10 = [{"shelf_mark": sm, "doc_id": doc_id} for sm, doc_id in V10_RECORDS]
    v8 = [src for src in v10 if catalogue_ids is None or src["doc_id"] in catalogue_ids]
    svc = ElasticsearchService.__new__(ElasticsearchService)
    svc.es = ScoringES({V10: v10, V8: v8})
    svc.index_name = V8
    svc.consensus_service = None
    return svc


def _routed(svc: ElasticsearchService, query: str, page: int = 1, size: int = 10):
    """Run a routed beta shelf-mark search.

    :param svc: Service.
    :param query: Routed query.
    :param page: 1-based page.
    :param size: Page size.
    :return: The response.
    """
    request = SimpleNamespace(query=query, filters=None, num_results=size, page=page)
    return asyncio.run(svc._beta_shelfmark_search(request, V10))


class TestRoutedVolumes:
    @pytest.mark.parametrize("query", ["Mosseri VII", "mosseri vii", "Moss. VII"])
    def test_mosseri_volume_finds_its_records_before_other_volumes(self, query: str) -> None:
        """Volume VII is stored as "Moss. VII,…": both spellings find it, and VIII never ranks among it."""
        out = _routed(_scoring_service(), query)
        ids = [r.doc_id for r in out.results]
        assert set(ids[:2]) == {"Cambridge_Mosseri_VII_157", "Cambridge_Mosseri_VII_5"}
        assert "Cambridge_Mosseri_VI145_2" not in ids

    def test_volume_v_excludes_vi_and_viii_from_its_records(self) -> None:
        """"Moss. V": every V record (also "V,105.4", no fuzzy match) ahead of the VI / VIII marks containing "Moss. V"."""
        ids = [r.doc_id for r in _routed(_scoring_service(), "Moss. V").results]
        assert set(ids[:2]) == {"Cambridge_Mosseri_V_8", "Cambridge_Mosseri_V_105_4"}

    def test_volume_i_excludes_ia(self) -> None:
        """"Mosseri I" ranks volume I before volume Ia (a letter after I is another volume)."""
        ids = [r.doc_id for r in _routed(_scoring_service(), "Mosseri I").results]
        assert ids[0] == "Cambridge_Mosseri_I_101"

    def test_long_form_found_through_doc_id(self) -> None:
        """"Mosseri II" finds "Mosseri, Jacques, Paris, France Ms. II 1" through its doc id."""
        ids = [r.doc_id for r in _routed(_scoring_service(), "Mosseri II").results]
        assert ids[0] == "Cambridge_Mosseri_II_1"


class TestRoutedPrefixPaging:
    def test_children_outrank_fuzzy_neighbours(self) -> None:
        """"T-S NS 32": the mark and its children (even "32.100", no fuzzy match) before "J2"/"J32"/"320.1"."""
        ids = [r.doc_id for r in _routed(_scoring_service(), "T-S NS 32").results]
        assert ids[0] == "Cambridge_CUL_T_S_NS_32"
        assert set(ids[1:4]) == NS32_CHILDREN

    def test_stored_prefix_record_leads_even_one_per_page(self) -> None:
        """"Cambridge CUL: T-S NS 32" outscores "T-S NS 32.7" (child + fuzzy), so it heads any page size."""
        out = _routed(_scoring_service(), "T-S NS 32", size=1)
        assert [r.doc_id for r in out.results] == ["Cambridge_CUL_T_S_NS_32"]

    def test_pages_are_disjoint_and_report_more(self) -> None:
        """Routed results page like semantic search: page 2 continues page 1, has_more is truthful."""
        svc = _scoring_service()
        first, second = _routed(svc, "T-S NS 32", page=1, size=2), _routed(svc, "T-S NS 32", page=2, size=2)
        assert (first.page, first.total_pages, first.has_more) == (1, 4, True)
        assert (second.page, second.has_more) == (2, True)
        first_ids, second_ids = [r.doc_id for r in first.results], [r.doc_id for r in second.results]
        assert not set(first_ids) & set(second_ids)
        assert set(first_ids[1:] + second_ids) == NS32_CHILDREN
        last = _routed(svc, "T-S NS 32", page=4, size=2)
        assert last.has_more is False and len(last.results) == 1
        routed_call = svc.es.calls[-2]
        assert routed_call["from_"] == 6 and routed_call["size"] == 2 and routed_call["sort"]


class TestRoutedCataloguePages:
    def test_flag_follows_catalogue_membership(self) -> None:
        """A routed beta result links to /fragment/ only when its doc id is in the catalogue index."""
        svc = _scoring_service(catalogue_ids={"Cambridge_CUL_T_S_NS_32", "Cambridge_CUL_T_S_NS_32_7"})
        out = _routed(svc, "T-S NS 32", size=3)
        flags = {r.doc_id: r.catalogue_page for r in out.results}
        assert flags["Cambridge_CUL_T_S_NS_32"] is True and flags["Cambridge_CUL_T_S_NS_32_7"] is True
        assert False in flags.values()
        lookup = svc.es.calls[-1]
        assert lookup["index"] == V8 and set(lookup["query"]["terms"]["doc_id"]) >= set(flags)


class TestNonRoutedUnchanged:
    def test_endpoint_and_chat_requests_keep_one_page_of_thirty(self) -> None:
        """Requests without prefix/page (shelf-mark endpoint, chat): no child clauses, top-30 window, one page."""
        svc = _scoring_service()
        request = SimpleNamespace(shelf_mark="T-S NS 32", exact_match=False, num_results=2, include_embeddings=False)
        out = asyncio.run(svc.search_by_shelfmark(request, V10))
        call = svc.es.calls[-1]
        assert call["size"] == 30 and call["from_"] == 0 and call["sort"] is None
        assert "doc_id" not in {f for c in call["query"]["bool"]["should"] if "regexp" in c for f in c["regexp"]}
        assert (out.page, out.total_pages, out.has_more) == (1, 1, False)
        assert all(r.catalogue_page is None for r in out.results)

    def test_route_request_opts_in(self) -> None:
        """Only the routed request carries prefix and page."""
        request = ShelfmarkRouteRequest.from_query(" Mosseri VII. ", None, 5, 3)
        assert (request.shelf_mark, request.prefix, request.page, request.num_results) == ("Mosseri VII", True, 3, 5)
        assert ShelfmarkRouteRequest.from_query("T-S 13J", None, None).page == 1


class TestChildClauses:
    @pytest.mark.parametrize("form,child,not_child", [
        ("Moss. VII", "Moss. VII,157", "Paris Mosseri: Moss. VIII399.1"),
        ("Moss. V", "Moss. V,8", "Paris Mosseri: Moss. VI145.2"),
        ("Moss. I", "Moss. I,101", "Moss. Ia,10.1"),
        ("T-S NS 32", "T-S NS 32.7", "T-S NS 320.1"),
        ("T-S 13J", "T-S 13J20.20", "T-S 13Ja1.1"),
    ])
    def test_child_regex_stops_at_the_volume(self, form: str, child: str, not_child: str) -> None:
        """The anchored child regexp matches marks under the prefix, never a longer sibling volume."""
        q = _service()._build_shelfmark_query(form, [], exact_match=False, prefix=True)
        rx = [c["regexp"]["shelf_mark"]["value"] for c in q["bool"]["should"]
              if c.get("regexp", {}).get("shelf_mark", {}).get("boost") == SHELFMARK_CHILD_BOOST]
        assert len(rx) == 1
        assert re.fullmatch(rx[0], child, re.IGNORECASE)
        assert not re.fullmatch(rx[0], not_child, re.IGNORECASE)

    def test_mosseri_spellings(self) -> None:
        """Both spellings for Mosseri marks, nothing for other collections."""
        assert ElasticsearchService.mosseri_spellings("Mosseri VII") == ["Moss. VII", "Mosseri VII"]
        assert ElasticsearchService.mosseri_spellings("Moss. Ia") == ["Moss. Ia", "Mosseri Ia"]
        assert ElasticsearchService.mosseri_spellings("T-S 13J") == []


class TestRoutedPagingOrder:
    @pytest.mark.parametrize("query", ["T-S NS 32", "Mosseri VII", "Moss. V"])
    def test_pages_concatenate_to_the_single_page_order(self, query: str) -> None:
        """Paged routed results keep Elasticsearch's order, so "Load more" pages join up in one order."""
        svc = _scoring_service()
        whole = [r.doc_id for r in _routed(svc, query, page=1, size=100).results]
        joined: List[str] = []
        for page in range(1, len(whole) + 1):
            joined += [r.doc_id for r in _routed(svc, query, page=page, size=2).results]
            if len(joined) >= len(whole):
                break
        assert joined == whole


class TestMosseriCommaRouting:
    @pytest.mark.parametrize("query,routed", [
        ("Moss. VII,5", True),
        ("Mosseri VII,5", True),
        ("Moss. Ia,1", True),
        ("Mosseri, Jacques", False),
        ("Mosseri collection", False),
    ])
    def test_catalogue_comma_form_routes(self, query: str, routed: bool) -> None:
        """The catalogue's "Moss. VII,5" form is a shelf mark; a name with a comma is not."""
        from src.backend.semantic_beta import looks_like_shelfmark

        assert looks_like_shelfmark(query) is routed


class _MappingES:
    """Records ``get_mapping`` calls and answers with fixed properties."""

    def __init__(self, properties: Dict[str, Any]) -> None:
        """Store the properties.

        :param properties: Mapping properties to return for any index.
        """
        self.properties = properties
        self.calls: List[str] = []

    class _Indices:
        """``es.indices`` stand-in."""

        def __init__(self, outer: "_MappingES") -> None:
            """Bind to the outer fake.

            :param outer: The fake client.
            """
            self.outer = outer

        def get_mapping(self, index: str) -> Dict[str, Any]:
            """Return the fixed mapping.

            :param index: Index name.
            :return: Elasticsearch-shaped mapping.
            """
            self.outer.calls.append(index)
            return {index: {"mappings": {"properties": self.outer.properties}}}

    @property
    def indices(self) -> "_MappingES._Indices":
        """The ``indices`` namespace.

        :return: Indices stand-in.
        """
        return _MappingES._Indices(self)


class TestFacetFiltersOnlyWhereMapped:
    @staticmethod
    def _service(properties: Dict[str, Any]) -> ElasticsearchService:
        """Service whose client maps the given properties.

        :param properties: Mapping properties.
        :return: The service.
        """
        svc = ElasticsearchService.__new__(ElasticsearchService)
        svc.es = _MappingES(properties)
        svc.index_name = V8
        return svc

    def test_v8_ignores_genres_instead_of_matching_nothing(self) -> None:
        """On an index without the facet fields a genres filter adds no clause (as before facets)."""
        svc = self._service({"shelf_mark": {"type": "keyword"}})
        assert svc._build_filters({"genres": "Piyyut", "subjects": "Liturgy"}, V8) == []

    def test_v10_filters_on_the_keyword_subfield(self) -> None:
        """Where genres is mapped with a .keyword subfield the filter applies."""
        svc = self._service({"genres": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}})
        clauses = svc._build_filters({"genres": "Piyyut"}, V10)
        assert clauses and "genres.keyword" in str(clauses)

    def test_no_facet_key_means_no_mapping_lookup(self) -> None:
        """Ordinary filters never trigger the mapping lookup."""
        svc = self._service({})
        svc._build_filters({"language": "Hebrew"}, V8)
        assert svc.es.calls == []
