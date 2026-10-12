"""Tests for the pure planning code of ``scripts/kg/consolidate_scholars.py`` (no database)."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from scripts.kg import consolidate_scholars as cs
from scripts.kg.consolidate_scholars import Edge, NameOp, Scholar
from scripts.kg.person_names import needs_review


def scholar(n: int, name: str, degree: int = 0, **props: Any) -> Scholar:
    """Build a Scholar fixture with elementId ``4:x:<n>``.

    :param n: Numeric part of the elementId.
    :param name: Scholar name.
    :param degree: Number of relationships.
    :param props: Extra node properties.
    :returns: The Scholar.
    :rtype: Scholar
    """
    return Scholar(f"4:x:{n}", name, degree, ["Scholar"], {"name": name, "data_sources": ["enriched"], **props})


def edge(eid: str, rel_type: str, start: str, end: str, **props: Any) -> Edge:
    """Build an Edge fixture.

    :param eid: Relationship id.
    :param rel_type: Relationship type.
    :param start: Start node id.
    :param end: End node id.
    :param props: Relationship properties.
    :returns: The Edge.
    :rtype: Edge
    """
    return Edge(eid, rel_type, start, end, props)


def merge_op(survivor: Scholar, *losers: Scholar, canonical: Optional[str] = None) -> NameOp:
    """Build a MERGE operation fixture.

    :param survivor: Surviving node.
    :param losers: Nodes merged away.
    :param canonical: Canonical name (default: the survivor's name).
    :returns: The operation.
    :rtype: NameOp
    """
    return NameOp("merge", "key", canonical or survivor.name, survivor, list(losers))


def by_kind(edge_ops: List[cs.EdgeOp], kind: str) -> List[cs.EdgeOp]:
    """Filter edge operations by kind.

    :param edge_ops: Planned edge operations.
    :param kind: ``move``, ``merge`` or ``drop_self_loop``.
    :returns: The matching operations.
    :rtype: List[cs.EdgeOp]
    """
    return [e for e in edge_ops if e.kind == kind]


# ---------------------------------------------------------------- grouping, canonical, survivor


def test_groups_become_merges_and_singletons_renames() -> None:
    """Same identity key -> MERGE; a singleton not in standard form -> RENAME; others untouched.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "Hathaway, Jane", 1), scholar(2, "Jane Hathaway", 25), scholar(3, "Zinger, Oded", 4),
                scholar(4, "Mark R. Cohen", 9), scholar(5, "Martin Cohen", 2), scholar(6, "Cohen M", 1)]
    plan = cs.plan_names(scholars)
    summary = [(op.kind, op.canonical, op.survivor.eid, [s.eid for s in op.losers]) for op in plan.ops]
    assert summary == [("merge", "Jane Hathaway", "4:x:2", ["4:x:1"]),
                       ("rename", "M. Cohen", "4:x:6", []),
                       ("rename", "Oded Zinger", "4:x:3", [])]
    assert plan.review == []
    assert plan.scholar_count == 6


def test_survivor_is_the_node_already_named_canonical() -> None:
    """The canonical-name holder survives even with fewer edges.

    :returns: None
    :rtype: None
    """
    plan = cs.plan_names([scholar(1, "Hathaway, Jane", 10), scholar(2, "Jane Hathaway", 1)])
    assert plan.ops[0].survivor.eid == "4:x:2"


def test_survivor_falls_back_to_highest_degree_then_lowest_id() -> None:
    """Without a canonical holder the highest degree survives; ties go to the lowest elementId.

    :returns: None
    :rtype: None
    """
    plan = cs.plan_names([scholar(1, "Goitein, S. D.", 3), scholar(2, "S.D. Goitein", 5)])
    assert (plan.ops[0].canonical, plan.ops[0].survivor.eid) == ("S. D. Goitein", "4:x:2")
    tie = [scholar(10, "Goitein, S. D.", 4), scholar(9, "S.D. Goitein", 4)]
    assert cs.choose_survivor(tie, "S. D. Goitein").eid == "4:x:9"


def test_canonical_comes_from_choose_display() -> None:
    """The cluster's display name keeps the diacritics; its holder survives.

    :returns: None
    :rtype: None
    """
    plan = cs.plan_names([scholar(1, "Tijana Krstic", 9), scholar(2, "Krstić, Tijana", 1),
                          scholar(3, "Tijana Krstić", 1)])
    op = plan.ops[0]
    assert (op.canonical, op.survivor.eid) == ("Tijana Krstić", "4:x:3")
    assert sorted(s.eid for s in op.losers) == ["4:x:1", "4:x:2"]


def test_review_skips_flagged_groups_and_renames() -> None:
    """Groups or renames with a needs_review member are skipped with a reason; flagged no-change names are listed.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "Peter Scha?fer"), scholar(2, "Peter Schäfer"), scholar(3, "Drawne!"),
                scholar(4, "Goitein"), scholar(5, "Smith, Dr. Jane"), scholar(6, "Dr. Jane Smith II")]
    plan = cs.plan_names(scholars)
    assert plan.ops == []
    rows = {(r.kind, r.proposed): {why for _, why in r.reasons} for r in plan.review}
    assert rows == {("merge", "Peter Schäfer"): {"contains '?' (lost accent)"},
                    ("rename", "Drawne"): {"single token"},
                    ("rename", "Dr. Jane Smith"): {"title prefix"},
                    ("review_only", "Goitein"): {"single token"},
                    ("review_only", "Dr. Jane Smith II"): {"title prefix"}}


def test_every_flagged_scholar_is_on_the_review_sheet() -> None:
    """Each Scholar needs_review flags gets a review row, also when the standard leaves it unchanged.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "the editor"), scholar(2, "Footnote 100"), scholar(3, "Wiet, M. G."),
                scholar(4, "M. G. Wiet"), scholar(5, "Josep Pellicer I Bru"), scholar(6, "Jane Hathaway"),
                scholar(7, "Udovitch, A. L.", 2), scholar(8, "Udovitch A L", 1), scholar(9, "Haelst J V"),
                scholar(10, "Yaqut, Ibn 'Abdallah al-Hamawi"), scholar(11, "D.S. and Roth Loewinger")]
    plan = cs.plan_names(scholars)
    assert plan.ops == []
    flagged = {s.eid for s in scholars if needs_review(s.name)}
    assert flagged == {s.eid for s in scholars} - {"4:x:4", "4:x:6", "4:x:7"}
    listed = {m.eid for r in plan.review for m in r.members}
    assert flagged <= listed and "4:x:6" not in listed
    rows = {(r.kind, tuple(m.eid for m in r.members)) for r in plan.review}
    assert rows == {("rename", ("4:x:1",)), ("review_only", ("4:x:2",)), ("merge", ("4:x:3", "4:x:4")),
                    ("rename", ("4:x:5",)), ("merge", ("4:x:7", "4:x:8")), ("rename", ("4:x:9",)),
                    ("rename", ("4:x:10",)), ("rename", ("4:x:11",))}


def test_lost_accent_members_drop_out_and_the_clean_rest_merges() -> None:
    """'?' members go to a lost_accent row; the clean members merge when at least two remain.

    :returns: None
    :rtype: None
    """
    plan = cs.plan_names([scholar(1, "Poznanski, Samuel", 2), scholar(2, "Samuel Poznan?ski", 9),
                          scholar(3, "Samuel Poznanski", 1)])
    assert [(op.kind, op.canonical, op.survivor.eid, [s.eid for s in op.losers]) for op in plan.ops] == \
        [("merge", "Samuel Poznanski", "4:x:3", ["4:x:1"])]
    assert [(r.kind, r.proposed, [m.eid for m in r.members]) for r in plan.review] == \
        [("lost_accent", "Samuel Poznanski", ["4:x:2"])]
    one_left = cs.plan_names([scholar(1, "Peter Scha?fer"), scholar(2, "Schäfer, Peter")])
    assert one_left.ops == [] and [r.kind for r in one_left.review] == ["merge"]


def test_canonical_form_is_weighted_by_degree() -> None:
    """Rule (e) counts edges, not members: one well-connected node outweighs two sparse ones.

    :returns: None
    :rtype: None
    """
    plan = cs.plan_names([scholar(1, "Ze’evi, Dror", 1), scholar(2, "Dror Ze’evi", 1), scholar(3, "Dror Ze'evi", 7)])
    assert (plan.ops[0].canonical, plan.ops[0].survivor.eid) == ("Dror Ze'evi", "4:x:3")


def test_scholar_person_clashes_are_reported() -> None:
    """Scholar names equal to Person names go to review, blocking any operation on them.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "Asher"), scholar(2, "Yalon, H."), scholar(3, "H. Yalon"), scholar(4, "Jane Hathaway")]
    plan = cs.plan_names(scholars, clashes={"Asher", "H. Yalon"})
    assert plan.ops == []
    kinds = sorted((r.kind, r.proposed) for r in plan.review)
    assert kinds == [("clash", "Asher"), ("merge", "H. Yalon")]
    merge_row = next(r for r in plan.review if r.kind == "merge")
    assert ("H. Yalon", "scholar/person name clash") in merge_row.reasons


def test_keys_differing_only_by_spaces_are_listed_for_review() -> None:
    """Groups whose identity keys differ only by spaces get a near_exact review row; clean groups still merge.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "M. Ben-Sasson", 4), scholar(2, "M. Ben Sasson", 1), scholar(3, "זאב בן-חיים"),
                scholar(4, "זאב בן חיים"), scholar(5, "Ben-Sasson, M.", 1), scholar(6, "Mark R. Cohen"),
                scholar(7, "M. R. Cohen")]
    plan = cs.plan_names(scholars)
    assert [(op.kind, op.canonical, op.survivor.eid, [s.eid for s in op.losers]) for op in plan.ops] == \
        [("merge", "M. Ben-Sasson", "4:x:1", ["4:x:5"])]
    rows = {r.key: (r.kind, [m.eid for m in r.members]) for r in plan.review}
    assert rows == {"mbensasson": ("near_exact", ["4:x:1", "4:x:2", "4:x:5"]),
                    "זאבבןחיים": ("near_exact", ["4:x:3", "4:x:4"])}
    assert all({why for _, why in r.reasons} == {cs.NEAR_EXACT_REASON} for r in plan.review)


def test_glued_initials_and_decomposed_names_join_their_clusters() -> None:
    """"A.M.Haberman" merges with "A. M. Haberman"; a decomposed (NFD) member never sets the canonical name.

    :returns: None
    :rtype: None
    """
    plan = cs.plan_names([scholar(1, "A.M.Haberman", 1), scholar(2, "A. M. Haberman", 3)])
    assert [(op.kind, op.canonical, op.survivor.eid) for op in plan.ops] == [("merge", "A. M. Haberman", "4:x:2")]
    composed = "Sim\u1e25a Assaf"
    plan = cs.plan_names([scholar(1, "Simh\u0323a Assaf", 1), scholar(2, "Simha Assaf", 2),
                          scholar(3, "Assaf, Sim\u1e25a", 2)])
    op = plan.ops[0]
    assert (op.canonical, op.survivor.eid) == (composed, "4:x:2")
    assert cs.survivor_update(op)["aliases"] == sorted(["Simh\u0323a Assaf", "Simha Assaf", "Assaf, Sim\u1e25a"])
    rename = cs.plan_names([scholar(1, "Muh\u0323ammad Assaf")]).ops
    assert [(op.kind, op.canonical) for op in rename] == [("rename", "Mu\u1e25ammad Assaf")]


def test_names_with_an_empty_key_are_never_grouped() -> None:
    """Names with no letters (lost script) each get their own review row, never one merge candidate.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "???????? ?????????? ???????"), scholar(2, "?. ?. ???????"), scholar(3, "?. ?. ?????")]
    groups = cs.group_by_key(scholars)
    assert sorted(groups) == ["empty:4:x:1", "empty:4:x:2", "empty:4:x:3"]
    plan = cs.plan_names(scholars)
    assert plan.ops == []
    assert [(r.kind, [m.eid for m in r.members]) for r in plan.review] == \
        [("empty_key", ["4:x:1"]), ("empty_key", ["4:x:2"]), ("empty_key", ["4:x:3"])]
    assert all("empty" in {why for _, why in r.reasons} for r in plan.review)


def test_check_targets_rejects_a_name_held_outside_the_group() -> None:
    """A target name owned by a Scholar outside the operation is a bug in the standard.

    :returns: None
    :rtype: None
    """
    a, b = scholar(1, "Zinger, Oded"), scholar(2, "Oded Zinger")
    with pytest.raises(AssertionError):
        cs.check_targets([NameOp("rename", "k", "Oded Zinger", a)], [a, b])
    cs.check_targets([NameOp("merge", "k", "Oded Zinger", b, [a])], [a, b])


def test_geresh_variants_are_listed_for_review() -> None:
    """"בנימין ריצ'לר" and "בנימין ריצלר" have different keys (צ' is not צ) but share a near_exact row.

    :returns: None
    :rtype: None
    """
    plan = cs.plan_names([scholar(1, "בנימין ריצ'לר", 3), scholar(2, "בנימין ריצלר", 1)])
    assert plan.ops == []
    assert [(r.kind, r.key, r.proposed, [m.eid for m in r.members]) for r in plan.review] == \
        [("near_exact", "בנימיןריצלר", "בנימין ריצ'לר", ["4:x:1", "4:x:2"])]


def test_a_given_name_that_is_a_place_goes_to_review() -> None:
    """"Gil, Palestine" (a book citation) is not flipped to "Palestine Gil"; an initial named like a place is.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "Gil, Palestine"), scholar(2, "Neubauer, A."), scholar(3, "Gil, Moshe")]
    plan = cs.plan_names(scholars, places={"Palestine", "A", "Moshe Gil"})
    assert [(op.kind, op.canonical) for op in plan.ops] == [("rename", "A. Neubauer"), ("rename", "Moshe Gil")]
    assert [(r.kind, r.proposed, r.reasons) for r in plan.review] == \
        [("rename", "", [("Gil, Palestine", cs.PLACE_REASON)])]
    assert "Palestine Gil" in [op.canonical for op in cs.plan_names(scholars).ops]


def test_review_rows_propose_nothing_when_the_standard_is_unreliable(tmp_path: Path) -> None:
    """Rows with a reason in UNRELIABLE_PROPOSAL have an empty proposal; '?', title and one-word rows keep theirs.

    :param tmp_path: Pytest fixture.
    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "Wiet, M. G."), scholar(2, "M. G. Wiet"), scholar(3, "Smith, Dr. Jane"),
                scholar(4, "Peter Scha?fer"), scholar(5, "Peter Schäfer"), scholar(6, "Haelst J V"),
                scholar(7, "the editor"), scholar(8, "Footnote 100"), scholar(9, "Drawne!")]
    plan = cs.attach_edges(cs.plan_names(scholars), [])
    proposals = {tuple(m.name for m in r.members): (r.kind, r.proposed) for r in plan.review}
    assert proposals == {("Wiet, M. G.", "M. G. Wiet"): ("merge", ""),
                         ("Smith, Dr. Jane",): ("rename", "Dr. Jane Smith"),
                         ("Peter Scha?fer", "Peter Schäfer"): ("merge", "Peter Schäfer"),
                         ("Haelst J V",): ("rename", ""), ("the editor",): ("rename", ""),
                         ("Footnote 100",): ("review_only", ""), ("Drawne!",): ("rename", "Drawne")}
    cs.write_reports(plan, tmp_path / "dryrun", {})
    rows = [line.split("\t") for line in (tmp_path / "dryrun" / "review.tsv").read_text().splitlines()[1:]]
    assert ["merge", "", "Wiet, M. G. | M. G. Wiet"] in [row[:1] + row[2:4] for row in rows]


class FakeReadTx:
    """Answers the planner's read queries from fixed rows."""

    def __init__(self, scholars: List[Scholar], places: List[str]) -> None:
        """Hold the rows.

        :param scholars: Scholar nodes to return.
        :param places: Place names to return.
        :returns: None
        :rtype: None
        """
        self.scholars, self.places = scholars, places

    def run(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        """Return the rows the query asks for (no clashes, no edges).

        :param query: Cypher.
        :param params: Parameters.
        :returns: Rows as dicts.
        :rtype: List[Dict[str, Any]]
        """
        if query == cs.SCHOLAR_QUERY:
            return [{"eid": s.eid, "labels": s.labels, "props": s.props, "degree": s.degree} for s in self.scholars]
        if query == cs.PLACE_QUERY:
            return [{"name": name} for name in self.places]
        return []


class FakeReadSession:
    """Runs a read unit of work against a FakeReadTx."""

    def __init__(self, tx: FakeReadTx) -> None:
        """Hold the transaction.

        :param tx: The fake transaction.
        :returns: None
        :rtype: None
        """
        self.tx = tx

    def execute_read(self, work: Any) -> Any:
        """Run the unit of work.

        :param work: Callable taking a transaction.
        :returns: Its result.
        :rtype: Any
        """
        return work(self.tx)


def test_read_plan_reads_place_names() -> None:
    """read_plan passes the graph's Place names to the planner, so "Gil, Palestine" reaches review.

    :returns: None
    :rtype: None
    """
    plan = cs.read_plan(FakeReadSession(FakeReadTx([scholar(1, "Gil, Palestine")], ["Palestine"])))
    assert plan.ops == [] and [r.reasons for r in plan.review] == [[("Gil, Palestine", cs.PLACE_REASON)]]


# ---------------------------------------------------------------- edge identity and planning


def test_edge_identity_ignores_direction_only_for_symmetric_types() -> None:
    """A COLLABORATED_WITH and its reverse share an identity; WROTE out/in and pgpid do not.

    :returns: None
    :rtype: None
    """
    remap = {"L": "S"}
    forward = edge("r1", "COLLABORATED_WITH", "S", "X")
    reverse = edge("r2", "COLLABORATED_WITH", "X", "L")
    assert cs.edge_identity(forward, "S", remap) == cs.edge_identity(reverse, "S", remap)
    assert cs.edge_identity(edge("r3", "WROTE", "S", "X"), "S", remap) != \
        cs.edge_identity(edge("r4", "WROTE", "X", "L"), "S", remap)
    assert cs.edge_identity(edge("r5", "SCRIBE_OF", "L", "F", pgpid=1), "S", remap) != \
        cs.edge_identity(edge("r6", "SCRIBE_OF", "S", "F", pgpid=2), "S", remap)
    assert cs.edge_identity(edge("r7", "REFERENCES", "L", "F", pages="12 -  13"), "S", remap) == \
        cs.edge_identity(edge("r8", "REFERENCES", "S", "F", pages="12 - 13"), "S", remap)


def test_plan_edges_moves_merges_and_drops_self_loops() -> None:
    """Loser edges move, collide into survivor edges (also reversed symmetric ones) or are dropped as loops.

    :returns: None
    :rtype: None
    """
    survivor, loser = scholar(1, "Jane Hathaway"), scholar(2, "Hathaway, Jane")
    s, lo = survivor.eid, loser.eid
    edges = [
        edge("5:x:10", "WROTE", s, "B1", data_sources=["enriched"], page_uuid="p1", confidence="high",
             book_uuid="bk1", evidence="ev1"),
        edge("5:x:11", "WROTE", lo, "B1", data_sources=["pgp"]),
        edge("5:x:12", "WROTE", lo, "B2", data_sources=["pgp"], source="pgp"),
        edge("5:x:13", "COLLABORATED_WITH", lo, s, data_sources=["enriched"]),
        edge("5:x:14", "COLLABORATED_WITH", "X", lo, data_sources=["enriched"], confidence="high"),
        edge("5:x:15", "COLLABORATED_WITH", s, "X", data_sources=["enriched"], confidence="medium"),
        edge("5:x:16", "MENTIONS_PERSON", "F", lo, data_sources=["enriched"], role="author"),
    ]
    edge_ops, problems = cs.plan_edges(merge_op(survivor, loser), edges)
    assert problems == []
    assert [e.sources[0].eid for e in by_kind(edge_ops, "drop_self_loop")] == ["5:x:13"]
    merges = {e.target.eid: e for e in by_kind(edge_ops, "merge")}
    assert set(merges) == {"5:x:10", "5:x:15"}
    assert merges["5:x:10"].props["data_sources"] == ["enriched", "pgp"]
    assert merges["5:x:10"].props["book_uuid"] == "bk1"
    assert (merges["5:x:15"].start, merges["5:x:15"].end) == (s, "X")
    assert merges["5:x:15"].props["confidence"] == "high"
    moves = {e.sources[0].eid: e for e in by_kind(edge_ops, "move")}
    assert (moves["5:x:12"].start, moves["5:x:12"].end, moves["5:x:12"].props) == \
        (s, "B2", {"data_sources": ["pgp"], "source": "pgp"})
    assert (moves["5:x:16"].start, moves["5:x:16"].end) == ("F", s)


def test_two_losers_with_one_identity_become_one_moved_edge() -> None:
    """Several loser edges with a new identity create one edge (the primary's direction, merged props).

    :returns: None
    :rtype: None
    """
    survivor, l1, l2 = scholar(1, "S. D. Goitein"), scholar(2, "Goitein, S. D."), scholar(3, "S.D. Goitein")
    edges = [edge("5:x:20", "WROTE", l1.eid, "B", data_sources=["pgp"]),
             edge("5:x:21", "WROTE", l2.eid, "B", data_sources=["enriched"], page_uuid="p9", evidence="e")]
    edge_ops, _ = cs.plan_edges(merge_op(survivor, l1, l2), edges)
    assert len(edge_ops) == 1
    move = edge_ops[0]
    assert (move.kind, move.start, move.end, len(move.sources)) == ("move", survivor.eid, "B", 2)
    assert move.props["data_sources"] == ["enriched", "pgp"]
    assert move.props["page_uuid"] == "p9"


def test_primary_rank_order() -> None:
    """page_uuid beats confidence high, which beats being on the survivor, which beats the lowest id.

    :returns: None
    :rtype: None
    """
    on_survivor = {"5:x:3"}
    candidates = [edge("5:x:1", "WROTE", "a", "b"), edge("5:x:2", "WROTE", "a", "b", confidence="high"),
                  edge("5:x:3", "WROTE", "a", "b"), edge("5:x:9", "WROTE", "a", "b", page_uuid="p")]
    ordered = sorted(candidates, key=lambda e: cs.primary_rank(e, on_survivor))
    assert [e.eid for e in ordered] == ["5:x:9", "5:x:2", "5:x:3", "5:x:1"]
    assert cs.eid_sort_key("4:x:9") < cs.eid_sort_key("4:x:10")


def test_merge_edge_props_follows_the_policy() -> None:
    """Unions, kept primary scalars with alternative lists, max/AND/OR rules and notes.

    :returns: None
    :rtype: None
    """
    primary = edge("5:x:2", "WROTE", "S", "B", data_sources=["enriched"], source_books=["bookA"],
                   book_uuid="u1", page_uuid="p1", evidence="first", evidence_page=7, source="bookA",
                   confidence="medium", pipeline_version="v3", uncertain=True, deceased=False,
                   has_transcription=False, transcription_extent="Partial", doc_relation="Edition",
                   sub_collection="T-S 10")
    other = edge("5:x:1", "WROTE", "S", "B", data_sources=["pgp", "enriched"], source_books=["bookB"],
                 book_uuid="u2", page_uuid="p2", evidence="second", evidence_page=9, source="bookB",
                 confidence="high", pipeline_version="v10", uncertain=False, deceased=True,
                 has_transcription=True, transcription_extent="Full", doc_relation="Translation ; Edition",
                 sub_collection="T-S 12")
    result = cs.merge_edge_props([other, primary], on_survivor={"5:x:2"})
    props = result.props
    assert result.primary == "5:x:1"  # both have page_uuid; confidence high wins before "on survivor"
    assert props["data_sources"] == ["pgp", "enriched"]
    assert props["source_books"] == ["bookB", "bookA"]
    assert (props["book_uuid"], props["book_uuids"]) == ("u2", ["u2", "u1"])
    assert (props["page_uuid"], props["page_uuids"]) == ("p2", ["p2", "p1"])
    assert (props["evidence"], props["evidence_alt"], props["evidence_page"]) == ("second", ["first"], 9)
    assert props["source"] == "bookB"
    assert props["confidence"] == "high"
    assert props["uncertain"] is False and props["deceased"] is True and props["has_transcription"] is True
    assert props["transcription_extent"] == "Full"
    assert props["doc_relation"] == "Translation ; Edition"
    assert (props["sub_collection"], props["sub_collections"]) == ("T-S 12", ["T-S 12", "T-S 10"])
    assert props["pipeline_version"] == "v10"
    assert result.problems == [] and result.notes == ["pipeline_version differs: ['v3', 'v10']"]


def test_merge_edge_props_flattens_existing_lists() -> None:
    """An edge that already has book_uuids/evidence_alt (merged by an earlier run) yields flat lists.

    :returns: None
    :rtype: None
    """
    first = edge("4:x:1", "WROTE", "S", "B", data_sources=["enriched"], book_uuid="u1", book_uuids=["u1", "u2"],
                 evidence="e1", evidence_alt=["e2"], page_uuid="p1", page_uuids=["p1", "p0"])
    second = edge("4:x:2", "WROTE", "L", "B", data_sources=["pgp"], book_uuid="u3", evidence="e3", page_uuid="p9")
    props = cs.merge_edge_props([first, second], {"4:x:1"}).props
    assert props["book_uuids"] == ["u1", "u2", "u3"]
    assert props["page_uuids"] == ["p1", "p0", "p9"]
    assert props["evidence_alt"] == ["e2", "e3"]
    assert not any(isinstance(item, list) for value in props.values() if isinstance(value, list) for item in value)


def test_merge_edge_props_adds_no_lists_when_values_agree() -> None:
    """Identical values produce no book_uuids/page_uuids/evidence_alt lists.

    :returns: None
    :rtype: None
    """
    a = edge("5:x:1", "WROTE", "S", "B", data_sources=["pgp"], book_uuid="u", evidence="same")
    b = edge("5:x:2", "WROTE", "S", "B", data_sources=["pgp"], book_uuid="u", evidence="same")
    assert cs.merge_edge_props([a, b]).props == {"data_sources": ["pgp"], "book_uuid": "u", "evidence": "same"}


def test_null_data_sources_against_a_list_blocks_the_merge() -> None:
    """NULL data_sources (the PGP map tier) is never merged with a list: the MERGE goes to review.

    :returns: None
    :rtype: None
    """
    survivor, loser = scholar(1, "Jane Hathaway"), scholar(2, "Hathaway, Jane")
    edges = [edge("5:x:1", "LIVED_IN", survivor.eid, "P"),
             edge("5:x:2", "LIVED_IN", loser.eid, "P", data_sources=["enriched"])]
    assert cs.merge_edge_props(edges).problems == ["data_sources NULL vs list"]
    plan = cs.attach_edges(cs.plan_names([survivor, loser]), edges)
    assert plan.ops == []
    assert plan.review[0].reasons == [("LIVED_IN out P", "data_sources NULL vs list")]


def test_attach_edges_chains_edges_between_two_groups() -> None:
    """An edge between losers of two groups is moved by the first MERGE and re-moved by the second.

    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "Jane Hathaway", 5), scholar(2, "Hathaway, Jane", 1),
                scholar(3, "Oded Zinger", 5), scholar(4, "Zinger, Oded", 1)]
    cross = edge("5:x:30", "COLLABORATED_WITH", "4:x:2", "4:x:4", data_sources=["enriched"])
    plan = cs.attach_edges(cs.plan_names(scholars), [cross])
    first, second = plan.ops
    assert (first.canonical, second.canonical) == ("Jane Hathaway", "Oded Zinger")
    assert [(e.kind, e.start, e.end, e.new_eid) for e in first.edges] == [("move", "4:x:1", "4:x:4", "new:0:0")]
    assert [(e.kind, e.sources[0].eid, e.start, e.end) for e in second.edges] == \
        [("move", "new:0:0", "4:x:1", "4:x:3")]
    assert cs.expected_counts(plan.ops, {"nodes": 10, "relationships": 20}) == {"nodes": 8, "relationships": 20}


def test_survivor_update_sets_name_aliases_and_unions() -> None:
    """Aliases = sorted old names and existing aliases minus the canonical; sources unioned (merge only).

    :returns: None
    :rtype: None
    """
    survivor = scholar(2, "S.D. Goitein", data_sources=["biblio"], aliases=["ש ד גויטיין"])
    loser = scholar(1, "Goitein, S. D.", data_sources=["pgp", "biblio"], source_books=["b1"])
    update = cs.survivor_update(NameOp("merge", "k", "S. D. Goitein", survivor, [loser]))
    assert update == {"name": "S. D. Goitein", "aliases": ["Goitein, S. D.", "S.D. Goitein", "ש ד גויטיין"],
                      "data_sources": ["biblio", "pgp"], "source_books": ["b1"]}
    rename = cs.survivor_update(NameOp("rename", "k", "Oded Zinger", scholar(3, "Zinger, Oded")))
    assert rename == {"name": "Oded Zinger", "aliases": ["Zinger, Oded"]}


def test_works_and_parallel_checks() -> None:
    """works_by_member counts distinct WROTE BookArticles; parallel_failures flags new duplicate identities.

    :returns: None
    :rtype: None
    """
    survivor, loser = scholar(1, "Jane Hathaway"), scholar(2, "Hathaway, Jane")
    op = merge_op(survivor, loser)
    op.member_edges = [edge("5:x:1", "WROTE", survivor.eid, "B1"), edge("5:x:2", "WROTE", loser.eid, "B1"),
                       edge("5:x:3", "WROTE", loser.eid, "B2"), edge("5:x:4", "WROTE", loser.eid, "J")]
    refs = {"B1": {"label": "BookArticle"}, "B2": {"label": "BookArticle"}, "J": {"label": "Journal"}}
    assert cs.works_by_member(op, refs) == {survivor.eid: 1, loser.eid: 2}
    clean = [edge("5:x:1", "WROTE", survivor.eid, "B1"), edge("5:x:9", "WROTE", survivor.eid, "B2")]
    assert cs.parallel_failures(op, clean) == []
    doubled = clean + [edge("5:x:8", "WROTE", survivor.eid, "B1")]
    assert len(cs.parallel_failures(op, doubled)) == 1


# ---------------------------------------------------------------- serialisation, reports, apply/revert helpers


def fixture_plan() -> cs.Plan:
    """A small plan with one MERGE (move + merge), one RENAME and one review row.

    :returns: The plan.
    :rtype: cs.Plan
    """
    scholars = [scholar(1, "Hathaway, Jane", 2), scholar(2, "Jane Hathaway", 3), scholar(3, "Zinger, Oded", 1),
                scholar(4, "Drawne!", 1)]
    edges = [edge("5:x:1", "WROTE", "4:x:2", "B1", data_sources=["enriched"]),
             edge("5:x:2", "WROTE", "4:x:1", "B1", data_sources=["pgp"]),
             edge("5:x:3", "STUDIED", "4:x:1", "F1", data_sources=["enriched"])]
    plan = cs.attach_edges(cs.plan_names(scholars), edges)
    plan.refs.update({"B1": {"label": "BookArticle", "key": "art-1"}, "F1": {"label": "Fragment", "key": "T-S 1"}})
    return plan


def test_plan_json_is_deterministic() -> None:
    """The same graph read in another order gives byte-identical plan JSON (what --apply compares).

    :returns: None
    :rtype: None
    """
    first = cs.canonical_dumps(cs.plan_json(fixture_plan()))
    scholars = [scholar(4, "Drawne!", 1), scholar(3, "Zinger, Oded", 1), scholar(2, "Jane Hathaway", 3),
                scholar(1, "Hathaway, Jane", 2)]
    edges = [edge("5:x:3", "STUDIED", "4:x:1", "F1", data_sources=["enriched"]),
             edge("5:x:2", "WROTE", "4:x:1", "B1", data_sources=["pgp"]),
             edge("5:x:1", "WROTE", "4:x:2", "B1", data_sources=["enriched"])]
    random.Random(7).shuffle(edges)
    plan = cs.attach_edges(cs.plan_names(scholars), edges)
    plan.refs.update({"B1": {"label": "BookArticle", "key": "art-1"}, "F1": {"label": "Fragment", "key": "T-S 1"}})
    assert cs.canonical_dumps(cs.plan_json(plan)) == first
    assert cs.canonical_dumps(json.loads(first)) == first


def test_write_reports(tmp_path: Path) -> None:
    """The dry run writes summary.json, plan.json and the four TSVs with matching counts.

    :param tmp_path: Pytest fixture.
    :returns: None
    :rtype: None
    """
    out = tmp_path / "dryrun"
    summary = cs.write_reports(fixture_plan(), out, {"uri": "bolt://scratch"})
    assert sorted(p.name for p in out.iterdir()) == ["edge_ops.tsv", "merges.tsv", "plan.json", "renames.tsv",
                                                     "review.tsv", "summary.json"]
    assert (summary["merges"], summary["renames"], summary["nodes_removed"]) == (1, 1, 1)
    assert summary["edges"] == {"moved": 1, "merged": 1, "collisions": 1, "self_loops_dropped": 0,
                                "relationships_removed": 1, "ops_by_type": {"merge:WROTE": 1, "move:STUDIED": 1}}
    assert summary["review"]["by_reason"] == {"single token": 1}
    assert json.loads((out / "summary.json").read_text())["plan_sha256"] == summary["plan_sha256"]
    renames = (out / "renames.tsv").read_text().splitlines()
    assert renames == ["eid\told_name\tnew_name", "4:x:3\tZinger, Oded\tOded Zinger"]
    edge_rows = (out / "edge_ops.tsv").read_text().splitlines()
    assert len(edge_rows) == 3
    assert "\tmove\tSTUDIED\tout\tF1\tFragment\tT-S 1\t" in edge_rows[1]
    assert "\tmerge\tWROTE\tout\tB1\tBookArticle\tart-1\t5:x:2\t5:x:1\t" in edge_rows[2]
    merges = (out / "merges.tsv").read_text().splitlines()
    assert merges[0].startswith("key\tcanonical\tcategory\t") and "\tJane Hathaway\tfull_name\t" in merges[1]


def test_initials_only_merges_are_reported(tmp_path: Path) -> None:
    """Merges of initials + surname ("F. Rosenthal" / "Rosenthal F") still merge but are marked initials_only.

    :param tmp_path: Pytest fixture.
    :returns: None
    :rtype: None
    """
    scholars = [scholar(1, "F. Rosenthal", 2), scholar(2, "Rosenthal F", 1), scholar(3, "Franz Rosenthal", 3),
                scholar(4, "Rosenthal, Franz", 1), scholar(5, "ש' אסף", 1), scholar(6, "ש. אסף", 1)]
    plan = cs.attach_edges(cs.plan_names(scholars), [])
    assert [op.canonical for op in plan.ops] == ["F. Rosenthal", "Franz Rosenthal", "ש' אסף"]
    summary = cs.write_reports(plan, tmp_path / "dryrun", {})
    assert (summary["merges"], summary["merges_initials_only"]) == (3, 2)
    rows = [line.split("\t")[1:3] for line in (tmp_path / "dryrun" / "merges.tsv").read_text().splitlines()[1:]]
    assert rows == [["F. Rosenthal", "initials_only"], ["Franz Rosenthal", "full_name"], ["ש' אסף", "initials_only"]]


class FakeResult:
    """Stands in for a neo4j Result with ``single()``."""

    def __init__(self, row: Dict[str, Any]) -> None:
        """Hold one row.

        :param row: The row.
        :returns: None
        :rtype: None
        """
        self.row = row

    def single(self) -> Dict[str, Any]:
        """Return the row.

        :returns: The row.
        :rtype: Dict[str, Any]
        """
        return self.row


class FakeTx:
    """Records queries; CREATEs return fresh ids and counted writes report full success."""

    def __init__(self) -> None:
        """Start with no calls.

        :returns: None
        :rtype: None
        """
        self.calls: List[tuple] = []

    def run(self, query: str, **params: Any) -> FakeResult:
        """Record one query and answer it.

        :param query: Cypher.
        :param params: Parameters.
        :returns: A fake result.
        :rtype: FakeResult
        """
        self.calls.append((query, params))
        if "RETURN elementId(" in query:
            return FakeResult({"eid": f"created:{len(self.calls)}"})
        return FakeResult({"n": len(set(params["eids"])) if "eids" in params else 1})


def test_apply_op_order_and_created_ids() -> None:
    """apply_op creates/merges edges, deletes sources, deletes losers, then renames the survivor.

    :returns: None
    :rtype: None
    """
    op = cs.plan_json(fixture_plan())["ops"][0]
    tx = FakeTx()
    created = cs.apply_op(tx, op)
    verbs = [" ".join(q.split()[:1] + [w for w in ("CREATE", "SET", "DELETE") if w in q]) for q, _ in tx.calls]
    assert verbs == ["MATCH CREATE SET", "MATCH DELETE", "MATCH SET", "MATCH DELETE", "MATCH DELETE", "MATCH SET"]
    assert created == {"new:0:0": "created:1"}
    assert tx.calls[1][1] == {"eids": ["5:x:3"]} and tx.calls[3][1] == {"eids": ["5:x:2"]}
    assert tx.calls[4][1] == {"eids": ["4:x:1"]}
    assert tx.calls[-1][1] == {"eid": "4:x:2", "old": "Jane Hathaway",
                               "props": {"name": "Jane Hathaway", "aliases": ["Hathaway, Jane"],
                                         "data_sources": ["enriched"]}}


def test_revert_op_recreates_losers_and_edges() -> None:
    """revert_op restores the survivor, deletes created edges and rewires recreated losers' edges.

    :returns: None
    :rtype: None
    """
    before = cs.plan_json(fixture_plan())["ops"][0]
    tx = FakeTx()
    maps = cs.revert_op(tx, before, {"new:0:0": "5:x:77"}, {}, {"5:x:77": "5:x:78"})
    assert tx.calls[0][1]["props"]["name"] == "Jane Hathaway" and tx.calls[0][1]["eid"] == "4:x:2"
    assert tx.calls[1][1]["eids"] == ["5:x:78"]
    assert maps["nodes"] == {"4:x:1": "created:3"}
    assert set(maps["edges"]) == {"5:x:2", "5:x:3"}
    recreated = [params for query, params in tx.calls if "CREATE (a)-[r:" in query]
    assert all(params["start"] == "created:3" for params in recreated)
    target_restore = tx.calls[-1][1]
    assert target_restore == {"eid": "5:x:1", "props": {"data_sources": ["enriched"]}}


def test_resolve_placeholders_and_id_maps() -> None:
    """Placeholders become real ids; recorded ids are looked up once (own map first), never chained.

    :returns: None
    :rtype: None
    """
    op = {"edges": [{"sources": [{"eid": "new:0:0"}, {"eid": "5:x:1"}], "target": {"eid": "new:0:1"}}]}
    resolved = cs.resolve_placeholders(op, {"new:0:0": "5:x:9", "new:0:1": "5:x:10"})
    assert resolved["edges"][0]["sources"] == [{"eid": "5:x:9"}, {"eid": "5:x:1"}]
    assert resolved["edges"][0]["target"] == {"eid": "5:x:10"}
    assert op["edges"][0]["sources"][0]["eid"] == "new:0:0"
    assert cs.current_id("a", {}, {"a": "b", "b": "c"}) == "b"
    assert cs.current_id("a", {"a": "x"}, {"a": "b"}) == "x"
    assert cs.current_id("z", {"a": "x"}, {"a": "b"}) == "z"


class ReusingTx(FakeTx):
    """FakeTx whose node CREATEs return scripted ids, as Neo4j does when it reuses freed ids."""

    def __init__(self, node_ids: List[str]) -> None:
        """Queue the ids the node CREATEs return.

        :param node_ids: elementIds handed out in order.
        :returns: None
        :rtype: None
        """
        super().__init__()
        self.node_ids = list(node_ids)

    def run(self, query: str, **params: Any) -> FakeResult:
        """Answer node CREATEs from the queue, everything else like FakeTx.

        :param query: Cypher.
        :param params: Parameters.
        :returns: A fake result.
        :rtype: FakeResult
        """
        if query.startswith("CREATE (n:"):
            self.calls.append((query, params))
            return FakeResult({"eid": self.node_ids.pop(0)})
        return super().run(query, **params)


def two_loser_before() -> Dict[str, Any]:
    """A before-record: survivor 4:db:1, losers L1 4:db:10 and L2 4:db:11, three moved edges and one merge.

    :returns: The record.
    :rtype: Dict[str, Any]
    """
    def ref(eid: str) -> Dict[str, Any]:
        """Endpoint reference.

        :param eid: Node id.
        :returns: The reference.
        :rtype: Dict[str, Any]
        """
        return {"eid": eid, "label": "Scholar", "key": eid}

    def source(eid: str, start: str, end: str, rel_type: str = "WROTE") -> Dict[str, Any]:
        """One consumed loser edge.

        :param eid: Edge id.
        :param start: Start node id.
        :param end: End node id.
        :param rel_type: Relationship type.
        :returns: The edge record.
        :rtype: Dict[str, Any]
        """
        return {"eid": eid, "type": rel_type, "start": ref(start), "end": ref(end), "props": {"n": eid}}

    def move(src: Dict[str, Any], new_eid: str) -> Dict[str, Any]:
        """A move edge operation.

        :param src: The consumed edge.
        :param new_eid: Placeholder of the created edge.
        :returns: The edge operation record.
        :rtype: Dict[str, Any]
        """
        return {"kind": "move", "type": src["type"], "start": ref("4:db:1"), "end": src["end"], "sources": [src],
                "target": None, "props": src["props"], "new_eid": new_eid, "notes": []}

    target = {"eid": "5:db:30", "type": "WROTE", "start": ref("4:db:1"), "end": ref("4:db:502"),
              "props": {"before": True}}
    return {"event": "before", "op": 0, "kind": "merge", "key": "k", "canonical": "S",
            "survivor": {"eid": "4:db:1", "name": "S", "labels": ["Scholar"], "props": {"name": "S"}},
            "losers": [{"eid": "4:db:10", "name": "L1", "labels": ["Scholar"], "props": {"name": "L1"}},
                       {"eid": "4:db:11", "name": "L2", "labels": ["Scholar"], "props": {"name": "L2"}}],
            "survivor_update": {"name": "S"},
            "edges": [move(source("5:db:1", "4:db:10", "4:db:500"), "new:0:0"),
                      move(source("5:db:2", "4:db:11", "4:db:501"), "new:0:1"),
                      move(source("5:db:3", "4:db:10", "4:db:20", "COLLABORATED_WITH"), "new:0:2"),
                      {"kind": "merge", "type": "WROTE", "start": ref("4:db:1"), "end": ref("4:db:502"),
                       "sources": [source("5:db:4", "4:db:11", "4:db:502")], "target": target,
                       "props": {"after": True}, "new_eid": None, "notes": []}]}


@pytest.mark.parametrize("node_ids, node_map, expected", [
    # L1 gets its own old id back.
    (["4:db:10", "4:db:12"], {}, {"5:db:1": "4:db:10", "5:db:2": "4:db:12", "5:db:3": ("4:db:10", "4:db:20")}),
    # L1 gets L2's freed id, L2 a fresh one.
    (["4:db:11", "4:db:12"], {}, {"5:db:1": "4:db:11", "5:db:2": "4:db:12", "5:db:3": ("4:db:11", "4:db:20")}),
    # An operation reverted earlier recreated its loser 4:db:20 on L1's freed id; L1 now gets 4:db:20.
    (["4:db:20", "4:db:12"], {"4:db:20": "4:db:10"},
     {"5:db:1": "4:db:20", "5:db:2": "4:db:12", "5:db:3": ("4:db:20", "4:db:10")}),
])
def test_revert_op_survives_neo4j_id_reuse(node_ids: List[str], node_map: Dict[str, str],
                                           expected: Dict[str, Any]) -> None:
    """Recreated losers that receive recorded (freed) ids still get exactly their own edges back.

    :param node_ids: Ids the recreated losers receive, in order.
    :param node_map: Old -> new node ids from operations reverted earlier.
    :param expected: Consumed edge id -> recreated start (or (start, end)).
    :returns: None
    :rtype: None
    """
    tx = ReusingTx(node_ids)
    edge_map = {"5:db:8": "5:db:9", "5:db:7": "5:db:8", "5:db:30": "5:db:31", "5:db:31": "5:db:32"}
    maps = cs.revert_op(tx, two_loser_before(), {"new:0:0": "5:db:7"}, node_map, edge_map)
    assert maps["nodes"] == {"4:db:10": node_ids[0], "4:db:11": node_ids[1]}
    recreated = {params["props"]["n"]: (params["start"], params["end"]) for query, params in tx.calls
                 if "CREATE (a)-[r:" in query}
    assert recreated == {"5:db:1": (expected["5:db:1"], "4:db:500"), "5:db:2": (expected["5:db:2"], "4:db:501"),
                         "5:db:3": expected["5:db:3"], "5:db:4": (expected["5:db:2"], "4:db:502")}
    deleted = [params["eids"] for query, params in tx.calls if "DELETE r" in query]
    assert deleted == [["5:db:8"]]
    restored = [params for query, params in tx.calls if "SET r = $props" in query and "CREATE" not in query]
    assert restored == [{"eid": "5:db:31", "props": {"before": True}}]


def test_pending_reverts() -> None:
    """Only done, not yet reverted operations are returned; an unfinished operation stops the revert.

    :returns: None
    :rtype: None
    """
    records = [{"event": "before", "op": 0}, {"event": "done", "op": 0, "created_edges": {"new:0:0": "e1"}},
               {"event": "before", "op": 1}, {"event": "done", "op": 1, "created_edges": {}},
               {"event": "before", "op": 2}, {"event": "failed", "op": 2, "error": "x"},
               {"event": "reverted", "op": 1, "node_map": {"n1": "n2"}, "edge_map": {"e5": "e6"}}]
    todo, node_map, edge_map = cs.pending_reverts(records)
    assert [(before["op"], created) for before, created in todo] == [(0, {"new:0:0": "e1"})]
    assert (node_map, edge_map) == ({"n1": "n2"}, {"e5": "e6"})
    with pytest.raises(SystemExit):
        cs.pending_reverts(records + [{"event": "before", "op": 3}])
    with pytest.raises(SystemExit):
        cs.pending_reverts(records + [{"event": "before", "op": 3}, {"event": "unknown", "op": 3, "error": "x"}])


class RaisingSession:
    """A session whose write transactions raise the given error."""

    def __init__(self, error: Exception) -> None:
        """Hold the error.

        :param error: Raised by ``execute_write``.
        :returns: None
        :rtype: None
        """
        self.error = error

    def __enter__(self) -> "RaisingSession":
        """Enter the context.

        :returns: The session.
        :rtype: RaisingSession
        """
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Leave the context without swallowing errors.

        :param exc: Exception info.
        :returns: False
        :rtype: bool
        """
        return False

    def execute_write(self, work: Any, *args: Any) -> Any:
        """Raise the error instead of writing.

        :param work: Transaction function.
        :param args: Its arguments.
        :returns: Never.
        :rtype: Any
        """
        raise self.error


class RaisingDriver:
    """A driver whose sessions raise on write."""

    def __init__(self, error: Exception) -> None:
        """Hold the error.

        :param error: Raised by every write.
        :returns: None
        :rtype: None
        """
        self.error = error

    def session(self, **kwargs: Any) -> RaisingSession:
        """Open a session.

        :param kwargs: Ignored session options.
        :returns: The session.
        :rtype: RaisingSession
        """
        return RaisingSession(self.error)


@pytest.mark.parametrize("unknown_outcome", [True, False])
def test_apply_logs_an_unknown_commit_outcome_and_revert_refuses_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unknown_outcome: bool) -> None:
    """IncompleteCommit (the commit may have happened) is logged ``unknown`` and blocks the revert; others ``failed``.

    :param tmp_path: Pytest fixture.
    :param monkeypatch: Pytest fixture.
    :param unknown_outcome: Raise IncompleteCommit (else a plain error).
    :returns: None
    :rtype: None
    """
    from neo4j.exceptions import IncompleteCommit

    (tmp_path / "plan.json").write_text(cs.canonical_dumps(cs.plan_json(fixture_plan())), encoding="utf-8")
    monkeypatch.setattr(cs, "read_plan", lambda session: fixture_plan())
    monkeypatch.setattr(cs, "graph_counts", lambda session: {"nodes": 10, "relationships": 20})
    error = IncompleteCommit("connection lost while waiting for COMMIT") if unknown_outcome else RuntimeError("x")
    with pytest.raises(type(error)):
        cs.run_apply(RaisingDriver(error), "neo4j", tmp_path, None)
    (log,) = tmp_path.glob("changelog_*.jsonl")
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [(r["event"], r["op"]) for r in records] == [("before", 0), ("unknown" if unknown_outcome else "failed", 0)]
    if unknown_outcome:
        with pytest.raises(SystemExit):
            cs.pending_reverts(records)
    else:
        assert cs.pending_reverts(records) == ([], {}, {})


def test_cli_requires_plan_for_apply() -> None:
    """--apply without --plan, or --plan without --apply, is refused.

    :returns: None
    :rtype: None
    """
    with pytest.raises(SystemExit):
        cs.parse_args(["--apply"])
    with pytest.raises(SystemExit):
        cs.parse_args(["--plan", "x"])
    assert cs.parse_args([]).uri == "bolt://localhost:7681"


@pytest.mark.parametrize("limit", ["-1", "0"])
def test_cli_rejects_a_limit_below_one(limit: str) -> None:
    """A negative or zero --limit is refused (a negative slice would apply all but the last N).

    :param limit: The --limit value.
    :returns: None
    :rtype: None
    """
    with pytest.raises(SystemExit):
        cs.parse_args(["--apply", "--plan", "x", "--limit", limit])
    assert cs.parse_args(["--apply", "--plan", "x", "--limit", "2"]).limit == 2
