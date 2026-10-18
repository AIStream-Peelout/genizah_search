"""Merge exact-duplicate Scholar nodes and apply the scholar name standard (dry run by default).

The name standard is ``scripts/kg/person_names.py``; the edge identity and property merge
policy is ``docs/kg_consolidation/census_2026-10-12/edge_merge_policy.md``.

* **Dry run** (default, read-only): reads every Scholar in one read transaction, groups them by
  ``identity_key`` (a group of two or more is a MERGE, a singleton whose standard name differs
  from its name is a RENAME; a name with an empty key is a group of its own; groups whose keys
  differ only by spaces or an inner Hebrew geresh are listed for review as ``near_exact``), plans
  every edge move/merge
  and writes to ``--out``::

      summary.json   counts by category and the plan's sha256
      plan.json      the full plan, which --apply re-checks and executes
      merges.tsv  renames.tsv  edge_ops.tsv  review.tsv

* **Apply** (``--apply --plan <dry-run dir> [--only NAMES] [--limit N]``): re-reads and re-plans
  (with the member degrees recorded in ``plan.json``, so the names and survivors are derived as in
  the dry run), then checks each selected operation of ``plan.json`` on its own: one the graph
  already shows applied (an earlier, partial run) is skipped and reported; one the re-plan no
  longer reproduces, or one a change log says was applied while the graph does not show it, makes
  the run refuse. Then runs one write transaction per operation, appending the full before-state
  to ``<dry-run dir>/changelog_<UTC>.jsonl`` before each one, and verifies. Each change log starts
  with its target (URI, database, elementId store id); logs of another target are ignored.
  ``--only`` takes comma-separated names (an operation matches when its canonical name or a
  member's name equals one), so the plan can be applied in stages. Operations that interact (one
  touches another's members or edges, ``op_groups``) are applied together: ``--only`` refuses a
  selection that splits such a group, and ``--limit`` extends its cut to whole groups.
* **Revert** (``--revert <changelog> [older changelogs ...]``): replays the logs backwards,
  newest operation first. A newer change log of the same plan directory that still has applied
  operations must be reverted first (earlier in the same list); the id maps that earlier reverts
  of newer logs wrote are loaded, so reverting in separate runs is as exact as in one.

Usage (from the repo root)::

    python scripts/kg/consolidate_scholars.py [--uri bolt://localhost:7681] [--out DIR]
    python scripts/kg/consolidate_scholars.py --apply --plan DIR [--only "Name A,Name B"] [--limit N]
    python scripts/kg/consolidate_scholars.py --revert DIR/changelog_<UTC>.jsonl

Keep the dry-run directory in the repo (``docs/kg_consolidation/dryrun_<UTC>/``): its change logs
are the only way to revert, so ``--apply`` refuses a plan directory under a temporary directory
unless ``--allow-temp-plan`` is given. Credentials come from ``NEO4J_AUTH`` in the environment or
the repo-root ``.env`` and are never printed. Never run ``--apply`` against production without
Isaac's approval of the dry run.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kg.person_names import (  # noqa: E402
    UNRELIABLE_PROPOSAL, choose_display, comma_given, identity_key, initials_only, is_initials_token, near_key,
    review_reasons)

DEFAULT_URI = "bolt://localhost:7681"
READ_ACCESS = "READ"  # neo4j.READ_ACCESS, spelled out so the pure planning code needs no driver

#: Edge types whose direction carries no meaning: (a)-[r]->(b) and (b)-[r]->(a) are the same fact.
SYMMETRIC_TYPES = frozenset({"COLLABORATED_WITH", "RELATED_TO", "MARRIED_TO", "JOINED_WITH"})
#: Properties that are part of an edge's identity (different values = different facts).
IDENTITY_PROPS = ("pgpid", "role", "pages")
#: Business key per label, written to the change log next to each elementId.
BUSINESS_KEYS = {"Scholar": "name", "Person": "name", "BookArticle": "article_id",
                 "Fragment": "canonical_shelfmark", "Place": "name", "Institution": "name",
                 "BiblicalPerson": "name"}

# Property merge policy (edge_merge_policy.md section 3). Everything not listed keeps the primary's value.
UNION_PROPS = ("data_sources", "source_books")
ALL_VALUES_LISTS = {"book_uuid": "book_uuids", "page_uuid": "page_uuids", "sub_collection": "sub_collections"}
ALT_LISTS = {"evidence": "evidence_alt", "raw_citation": "raw_citation_alt", "location": "location_alt",
             "notes": "notes_alt"}
RANKED_PROPS = {"confidence": ("low", "medium", "high"),
                "transcription_extent": ("Unknown", "Partial", "Full"),
                "translation_extent": ("Unknown", "Partial", "Full")}
OR_FLAGS = ("certain", "deceased", "has_transcription", "has_translation", "has_discussion", "primary")
AND_FLAGS = ("uncertain",)
TOKEN_UNION = {"doc_relation": " ; "}

SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: Group key prefix of a Scholar whose identity key is empty (no letters or digits): it groups with nothing.
EMPTY_KEY_PREFIX = "empty:"
#: Review reason for identity groups whose keys differ only by word spaces or an inner geresh.
NEAR_EXACT_REASON = "same name apart from spaces or a geresh"
#: Review reason for "Gil, Palestine": the part after the comma is one word that names a Place.
PLACE_REASON = "given name is a place"
#: Merge initials + surname clusters ("F. Rosenthal" / "Rosenthal F") automatically; off: they go to review.
INCLUDE_INITIALS_ONLY_MERGES = False
#: Review reason for an initials + surname merge held back by ``INCLUDE_INITIALS_ONLY_MERGES``.
INITIALS_ONLY_REASON = "initials only: same surname + initials is not proof of the same person"
#: A plan.json placeholder for an edge an earlier operation creates: ``new:<op index>:<n>``.
PLACEHOLDER = re.compile(r"^new:(\d+):(\d+)$")
#: plan.json format; --apply refuses any other (format 2 records member degrees and the format itself).
PLAN_FORMAT = 2
#: Event of the first change-log record: the target (uri, database, store) the log was written for.
TARGET_EVENT = "target"
#: Directories whose contents the OS may delete: --apply refuses a plan directory under one.
TEMP_ROOTS = ("/tmp", "/private/tmp", "/var/tmp", "/var/folders", "/private/var/folders")
#: Any node's elementId, whose middle part identifies the database store.
STORE_QUERY = "MATCH (n) RETURN elementId(n) AS eid LIMIT 1"


@dataclass
class Scholar:
    """One Scholar node as read from the graph.

    :param eid: elementId.
    :param name: Current ``name``.
    :param degree: Number of relationships (bookkeeping for the survivor choice).
    :param labels: Node labels.
    :param props: All node properties (``name`` included).
    """

    eid: str
    name: str
    degree: int = 0
    labels: List[str] = field(default_factory=lambda: ["Scholar"])
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """One relationship.

    :param eid: elementId (or a ``new:<op>:<n>`` placeholder for an edge an earlier operation creates).
    :param type: Relationship type.
    :param start: Start node elementId.
    :param end: End node elementId.
    :param props: All relationship properties.
    """

    eid: str
    type: str
    start: str
    end: str
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeOp:
    """What happens to one edge identity during a MERGE.

    :param kind: ``move`` (create one edge on the survivor), ``merge`` (into an existing survivor
        edge) or ``drop_self_loop`` (an edge between two members of the group).
    :param type: Relationship type.
    :param start: Start node of the resulting edge (None for a dropped self-loop).
    :param end: End node of the resulting edge (None for a dropped self-loop).
    :param sources: The loser edges this operation deletes.
    :param target: The existing survivor edge merged into (``merge`` only), with its before-props.
    :param props: Properties of the resulting edge.
    :param new_eid: Placeholder id of the edge a ``move`` creates.
    :param notes: Non-blocking remarks (e.g. differing ``pipeline_version``).
    """

    kind: str
    type: str
    start: Optional[str]
    end: Optional[str]
    sources: List[Edge]
    target: Optional[Edge] = None
    props: Optional[Dict[str, Any]] = None
    new_eid: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class NameOp:
    """One MERGE or RENAME.

    :param kind: ``merge`` or ``rename``.
    :param key: The group's identity key.
    :param canonical: The name the survivor ends up with.
    :param survivor: The node that keeps the edges.
    :param losers: Nodes merged into the survivor and deleted (empty for a rename).
    :param edges: Planned edge operations (merge only).
    :param member_edges: Edges touching the members before the operation (for verification).
    """

    kind: str
    key: str
    canonical: str
    survivor: Scholar
    losers: List[Scholar] = field(default_factory=list)
    edges: List[EdgeOp] = field(default_factory=list)
    member_edges: List[Edge] = field(default_factory=list)


@dataclass
class ReviewItem:
    """An operation skipped for a human decision.

    :param kind: ``merge``, ``rename``, ``clash`` (Scholar name that is also a Person name),
        ``near_exact`` (identity keys differ only by spaces or a geresh), ``empty_key`` (a name with no
        letters or digits, nothing planned), ``lost_accent`` (``?`` members left out of their
        group's automatic merge) or ``review_only`` (a flagged name the standard leaves unchanged).
    :param key: Identity key.
    :param proposed: The canonical name the standard would give; empty when a reason is in
        ``UNRELIABLE_PROPOSAL`` or ``PLACE_REASON`` (the standard form is known to be unreliable).
    :param members: The Scholars involved.
    :param reasons: ``(subject, reason)`` pairs.
    """

    kind: str
    key: str
    proposed: str
    members: List[Scholar]
    reasons: List[Tuple[str, str]]


@dataclass
class Plan:
    """The whole consolidation plan.

    :param ops: Operations to apply, in order.
    :param review: Skipped operations with reasons.
    :param refs: elementId -> ``{"label", "key"}`` business keys for the change log and reports.
    :param scholar_count: Scholars read.
    """

    ops: List[NameOp]
    review: List[ReviewItem]
    refs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scholar_count: int = 0


@dataclass
class PropMerge:
    """Result of merging the properties of edges with one identity.

    :param props: Merged properties.
    :param primary: elementId of the primary edge.
    :param problems: Blocking issues (the group goes to review).
    :param notes: Non-blocking remarks.
    """

    props: Dict[str, Any]
    primary: str
    problems: List[str]
    notes: List[str]


# ---------------------------------------------------------------- small helpers


def eid_sort_key(eid: str) -> Tuple[Tuple[int, Any], ...]:
    """Sort elementIds numerically where they are numeric ("4:uuid:9" before "4:uuid:10").

    :param eid: An elementId or placeholder.
    :returns: A key that compares any two ids.
    :rtype: Tuple[Tuple[int, Any], ...]
    """
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in eid.split(":"))


def as_list(value: Any) -> List[Any]:
    """Return a property value as a list (None -> [], scalar -> [scalar]).

    :param value: A property value.
    :returns: The value as a list.
    :rtype: List[Any]
    """
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def union_lists(values: Iterable[Any]) -> List[Any]:
    """Union of list (or scalar) values, keeping first-seen order and dropping None.

    :param values: Property values, primary first.
    :returns: The ordered union.
    :rtype: List[Any]
    """
    out: List[Any] = []
    for value in values:
        out.extend(item for item in as_list(value) if item is not None and item not in out)
    return out


def version_key(version: str) -> Tuple[Tuple[int, ...], str]:
    """Sort key for ``pipeline_version`` strings ("v10" after "v3").

    :param version: A version string.
    :returns: Its numbers, then the string.
    :rtype: Tuple[Tuple[int, ...], str]
    """
    return tuple(int(n) for n in re.findall(r"\d+", version)), version


def current_id(eid: str, own: Mapping[str, str], earlier: Mapping[str, str]) -> str:
    """The id a node or edge recorded in a change log has now: one lookup, never a chain.

    A revert recreates each deleted node or edge once and maps its recorded id straight to the
    new one. Neo4j hands freed ids to later CREATEs, so a new id can equal another recorded id;
    following the map further would then land on the wrong entity.

    :param eid: The id recorded in the operation's before-state.
    :param own: Old -> new ids recreated by the operation being reverted (checked first).
    :param earlier: Old -> new ids recreated by operations reverted before it.
    :returns: The id it has now (unchanged when nothing recreated it).
    :rtype: str
    """
    if eid in own:
        return own[eid]
    return earlier.get(eid, eid)


# ---------------------------------------------------------------- name planning


def group_by_key(scholars: Iterable[Scholar]) -> Dict[str, List[Scholar]]:
    """Group Scholars by ``identity_key`` (members sorted by elementId).

    An empty key identifies nothing ("?. ?. ?????" and "???????" are different people), so
    each such Scholar gets a group of its own, keyed ``empty:<elementId>``.

    :param scholars: All Scholar nodes.
    :returns: identity key -> members.
    :rtype: Dict[str, List[Scholar]]
    """
    groups: Dict[str, List[Scholar]] = defaultdict(list)
    for scholar in scholars:
        groups[identity_key(scholar.name) or EMPTY_KEY_PREFIX + scholar.eid].append(scholar)
    return {key: sorted(members, key=lambda s: eid_sort_key(s.eid)) for key, members in groups.items()}


def choose_survivor(members: Sequence[Scholar], canonical: str) -> Scholar:
    """Pick the node that keeps the edges: the one already named ``canonical``, else the highest degree.

    :param members: The group's Scholars.
    :param canonical: The group's canonical name.
    :returns: The survivor (ties on degree go to the lowest elementId).
    :rtype: Scholar
    """
    named = [m for m in members if m.name == canonical]
    if named:
        return named[0]
    return min(members, key=lambda m: (-m.degree, eid_sort_key(m.eid)))


def given_is_place(name: str, places: Collection[str]) -> bool:
    """Return True when a one-comma name's given part is one word naming a Place ("Gil, Palestine").

    :param name: A Scholar name.
    :param places: Place names in the graph.
    :returns: Whether the "given name" is a place (a book-title citation, not a person).
    :rtype: bool
    """
    given = comma_given(name)
    return bool(given) and " " not in given and not is_initials_token(given) and given in places


def group_review_reasons(members: Sequence[Scholar], clashes: Collection[str],
                         places: Collection[str] = ()) -> List[Tuple[str, str]]:
    """Collect why a group (or singleton) must not be changed automatically.

    :param members: The group's Scholars.
    :param clashes: Scholar names that are also Person names.
    :param places: Place names in the graph.
    :returns: ``(name, reason)`` pairs; empty when the group is clean.
    :rtype: List[Tuple[str, str]]
    """
    reasons = [(m.name, reason) for m in members for reason in review_reasons(m.name)]
    reasons += [(m.name, PLACE_REASON) for m in members if given_is_place(m.name, places)]
    reasons += [(m.name, "scholar/person name clash") for m in members if m.name in clashes]
    return reasons


def check_targets(ops: Sequence[NameOp], scholars: Sequence[Scholar]) -> None:
    """Assert that no operation gives its survivor a name another Scholar outside the group holds.

    Such a Scholar would have the same identity key and so must have been grouped; this
    catches a name standard that is not idempotent.

    :param ops: Planned operations.
    :param scholars: All Scholar nodes.
    :returns: None
    :rtype: None
    :raises AssertionError: When a target name belongs to a Scholar outside the operation.
    """
    owner = {s.name: s.eid for s in scholars}
    for op in ops:
        members = {op.survivor.eid, *(loser.eid for loser in op.losers)}
        holder = owner.get(op.canonical)
        if holder is not None and holder not in members:
            raise AssertionError(f"{op.kind} target {op.canonical!r} already belongs to Scholar {holder} "
                                 f"outside the group {sorted(members)}")


def near_exact_items(groups: Mapping[str, Sequence[Scholar]], clashes: Collection[str],
                     places: Collection[str] = ()) -> List[ReviewItem]:
    """Review rows for identity groups whose keys differ only by word spaces or an inner geresh.

    The standard keeps word spaces and a geresh between two Hebrew letters in the key, so
    "M. Ben-Sasson" / "M. Ben Sasson" and "בנימין ריצ'לר" / "בנימין ריצלר" are not merged; this
    lists them so they reach the review sheet. Each group is still merged or renamed on its own
    when it is clean.

    :param groups: identity key -> members, from ``group_by_key``.
    :param clashes: Scholar names that are also Person names.
    :param places: Place names in the graph.
    :returns: One ``near_exact`` row per ``near_key`` shared by two or more identity keys.
    :rtype: List[ReviewItem]
    """
    by_near: Dict[str, List[str]] = defaultdict(list)
    for key, members in groups.items():
        if not key.startswith(EMPTY_KEY_PREFIX):
            by_near[near_key(members[0].name)].append(key)
    items: List[ReviewItem] = []
    for near, keys in sorted(by_near.items()):
        if len(keys) < 2:
            continue
        members = sorted((m for key in keys for m in groups[key]), key=lambda s: eid_sort_key(s.eid))
        reasons = [(m.name, NEAR_EXACT_REASON) for m in members] + group_review_reasons(members, clashes, places)
        items.append(ReviewItem("near_exact", near, display_name(members), members, reasons))
    return items


def display_name(members: Sequence[Scholar]) -> str:
    """``choose_display`` over the members' names, weighted by their degrees.

    :param members: A group's Scholars.
    :returns: The canonical display name.
    :rtype: str
    """
    return choose_display([m.name for m in members], [m.degree for m in members])


def plan_names(scholars: Sequence[Scholar], clashes: Collection[str] = (), places: Collection[str] = ()) -> Plan:
    """Plan MERGE and RENAME operations from the Scholar names alone (no edges yet).

    A group whose only flagged members are ``?`` names (lost accents) leaves those to a
    ``lost_accent`` review row and merges the rest when at least two clean members remain. An
    initials + surname merge goes to review unless ``INCLUDE_INITIALS_ONLY_MERGES``.
    Every flagged Scholar the standard would leave unchanged gets a ``review_only`` row. A review
    row proposes no name when a reason says the standard form is unreliable ("M. G. Wiet").

    :param scholars: All Scholar nodes.
    :param clashes: Scholar names that are also Person names (reported for review).
    :param places: Place names in the graph ("Gil, Palestine" is reported for review).
    :returns: Merges (sorted by key), then renames (sorted by key), plus the review list
        (group rows sorted by key, then ``near_exact`` rows).
    :rtype: Plan
    """
    merges: List[NameOp] = []
    renames: List[NameOp] = []
    review: List[ReviewItem] = []
    groups = group_by_key(scholars)
    for key, members in sorted(groups.items()):
        lost = [m for m in members if "?" in m.name]
        rest = [m for m in members if "?" not in m.name]
        if lost and len(rest) > 1 and not group_review_reasons(rest, clashes, places):
            review.append(ReviewItem("lost_accent", key, display_name(rest), lost,
                                     group_review_reasons(lost, clashes, places)))
            members = rest
        canonical = display_name(members)
        reasons = group_review_reasons(members, clashes, places)
        kind = "merge" if len(members) > 1 else "rename"
        if kind == "rename" and canonical == members[0].name:
            name = members[0].name
            if key.startswith(EMPTY_KEY_PREFIX):
                review.append(ReviewItem("empty_key", key, canonical, list(members), reasons))
            elif name in clashes:
                review.append(ReviewItem("clash", key, canonical, list(members),
                                         [(name, "scholar/person name clash (no change planned)")]
                                         + [(name, reason) for reason in review_reasons(name)]))
            elif reasons:
                review.append(ReviewItem("review_only", key, canonical, list(members), reasons))
            continue
        if reasons:
            review.append(ReviewItem(kind, key, canonical, list(members), reasons))
            continue
        if kind == "merge" and not INCLUDE_INITIALS_ONLY_MERGES and initials_only(canonical):
            review.append(ReviewItem(kind, key, canonical, list(members), [(canonical, INITIALS_ONLY_REASON)]))
            continue
        survivor = choose_survivor(members, canonical)
        losers = [m for m in members if m.eid != survivor.eid]
        (merges if kind == "merge" else renames).append(NameOp(kind, key, canonical, survivor, losers))
    review += near_exact_items(groups, clashes, places)
    for item in review:
        if any(reason in UNRELIABLE_PROPOSAL or reason == PLACE_REASON for _, reason in item.reasons):
            item.proposed = ""
    ops = merges + renames
    check_targets(ops, scholars)
    refs = {s.eid: {"label": "Scholar", "key": s.name} for s in scholars}
    return Plan(ops, review, refs, len(scholars))


# ---------------------------------------------------------------- edge planning


def identity_value(value: Any) -> Any:
    """Normalise an identity property value (whitespace-collapsed strings, hashable lists).

    :param value: A property value.
    :returns: A hashable, comparable form.
    :rtype: Any
    """
    if isinstance(value, str):
        return " ".join(value.split())
    return tuple(value) if isinstance(value, list) else value


def edge_identity(edge: Edge, anchor: str, remap: Mapping[str, str]) -> Tuple[Any, ...]:
    """Identity of an edge as seen from ``anchor`` after the losers are replaced by the survivor.

    ``(type, direction, other endpoint, pgpid, role, pages)``; the direction is ``"*"`` for the
    symmetric types, so an edge and its reverse share one identity.

    :param edge: The relationship.
    :param anchor: The survivor's elementId.
    :param remap: Loser elementId -> survivor elementId.
    :returns: The identity tuple.
    :rtype: Tuple[Any, ...]
    """
    start, end = remap.get(edge.start, edge.start), remap.get(edge.end, edge.end)
    outgoing = start == anchor
    direction = "*" if edge.type in SYMMETRIC_TYPES else ("out" if outgoing else "in")
    other = end if outgoing else start
    return (edge.type, direction, other) + tuple(identity_value(edge.props.get(k)) for k in IDENTITY_PROPS)


def primary_rank(edge: Edge, on_survivor: Collection[str]) -> Tuple[Any, ...]:
    """Sort key choosing the primary edge: has page_uuid > confidence high > on the survivor > lowest id.

    :param edge: A candidate edge.
    :param on_survivor: elementIds of the edges already on the survivor.
    :returns: The sort key (smallest is primary).
    :rtype: Tuple[Any, ...]
    """
    return (edge.props.get("page_uuid") is None, edge.props.get("confidence") != "high",
            edge.eid not in on_survivor, eid_sort_key(edge.eid))


def merge_edge_props(edges: Sequence[Edge], on_survivor: Collection[str] = ()) -> PropMerge:
    """Merge the properties of edges sharing one identity (edge_merge_policy.md section 3).

    :param edges: Two or more edges with the same identity.
    :param on_survivor: elementIds of those already on the survivor.
    :returns: Merged properties, the primary's id, blocking problems and notes.
    :rtype: PropMerge
    """
    ordered = sorted(edges, key=lambda e: primary_rank(e, on_survivor))
    primary = ordered[0]
    merged = dict(primary.props)
    problems: List[str] = []
    notes: List[str] = []
    sources = [e.props.get("data_sources") for e in ordered]
    if None in sources and any(s is not None for s in sources):
        problems.append("data_sources NULL vs list")
    for key in UNION_PROPS:
        values = union_lists(e.props.get(key) for e in ordered)
        if values:
            merged[key] = values
    for key, list_key in ALL_VALUES_LISTS.items():
        values = union_lists(as_list(e.props.get(key)) + as_list(e.props.get(list_key)) for e in ordered)
        if values and values != [merged.get(key)]:
            merged[list_key] = values
    for key, alt_key in ALT_LISTS.items():
        alts = [v for v in union_lists(as_list(e.props.get(key)) + as_list(e.props.get(alt_key)) for e in ordered)
                if v != merged.get(key)]
        if alts:
            merged[alt_key] = alts
    for key, order in RANKED_PROPS.items():
        values = [e.props[key] for e in ordered if e.props.get(key) is not None]
        if values:
            merged[key] = max(values, key=lambda v: order.index(v) if v in order else -1)
    for keys, combine in ((OR_FLAGS, any), (AND_FLAGS, all)):
        for key in keys:
            values = [bool(e.props[key]) for e in ordered if e.props.get(key) is not None]
            if values:
                merged[key] = combine(values)
    for key, sep in TOKEN_UNION.items():
        tokens = union_lists([t.strip() for t in str(e.props[key]).split(";") if t.strip()]
                             for e in ordered if e.props.get(key) is not None)
        if tokens:
            merged[key] = sep.join(tokens)
    versions = {e.props["pipeline_version"] for e in ordered if e.props.get("pipeline_version") is not None}
    if versions:
        merged["pipeline_version"] = max(versions, key=version_key)
        if len(versions) > 1:
            notes.append(f"pipeline_version differs: {sorted(versions, key=version_key)}")
    return PropMerge(merged, primary.eid, problems, notes)


def plan_edges(op: NameOp, edges: Sequence[Edge]) -> Tuple[List[EdgeOp], List[Tuple[str, str]]]:
    """Plan what happens to every loser edge of one MERGE.

    A loser edge whose identity (seen from the survivor) matches an existing survivor edge is
    merged into it; loser edges with a new identity are moved (several with one identity become
    one moved edge); an edge between two members of the group is dropped as a would-be self-loop.

    :param op: The MERGE.
    :param edges: Every edge touching a member of the group.
    :returns: Edge operations, and blocking ``(subject, reason)`` problems.
    :rtype: Tuple[List[EdgeOp], List[Tuple[str, str]]]
    """
    survivor = op.survivor.eid
    remap = {loser.eid: survivor for loser in op.losers}
    existing: Dict[Tuple[Any, ...], List[Edge]] = defaultdict(list)
    incoming: Dict[Tuple[Any, ...], List[Edge]] = defaultdict(list)
    ops: List[EdgeOp] = []
    for edge in sorted(edges, key=lambda e: eid_sort_key(e.eid)):
        moved = edge.start in remap or edge.end in remap
        if moved and remap.get(edge.start, edge.start) == remap.get(edge.end, edge.end):
            ops.append(EdgeOp("drop_self_loop", edge.type, None, None, [edge]))
            continue
        (incoming if moved else existing)[edge_identity(edge, survivor, remap)].append(edge)
    problems: List[Tuple[str, str]] = []
    for identity in sorted(incoming, key=repr):
        sources, targets = incoming[identity], existing.get(identity, [])
        subject = f"{identity[0]} {identity[1]} {identity[2]}"
        if targets:
            target = min(targets, key=lambda e: primary_rank(e, {e.eid}))
            result = merge_edge_props([target] + sources, {t.eid for t in targets})
            notes = result.notes + ([f"survivor already has {len(targets)} edges with this identity"]
                                    if len(targets) > 1 else [])
            ops.append(EdgeOp("merge", target.type, target.start, target.end, sources, target, result.props,
                              notes=notes))
        else:
            result = (merge_edge_props(sources) if len(sources) > 1
                      else PropMerge(dict(sources[0].props), sources[0].eid, [], []))
            primary = next(e for e in sources if e.eid == result.primary)
            ops.append(EdgeOp("move", primary.type, remap.get(primary.start, primary.start),
                              remap.get(primary.end, primary.end), sources, props=result.props,
                              notes=result.notes))
        problems += [(subject, problem) for problem in result.problems]
    return ops, problems


class EdgeStore:
    """In-memory edges, updated as each planned MERGE is simulated, so later operations see
    the edges earlier ones create (an edge between members of two different groups)."""

    def __init__(self, edges: Iterable[Edge]) -> None:
        """Index the edges by node.

        :param edges: Edges touching any member of any MERGE.
        :returns: None
        :rtype: None
        """
        self.edges: Dict[str, Edge] = {}
        self.by_node: Dict[str, Set[str]] = defaultdict(set)
        for edge in edges:
            self.add(edge)

    def add(self, edge: Edge) -> None:
        """Add or replace one edge.

        :param edge: The edge.
        :returns: None
        :rtype: None
        """
        self.edges[edge.eid] = edge
        self.by_node[edge.start].add(edge.eid)
        self.by_node[edge.end].add(edge.eid)

    def remove(self, eid: str) -> None:
        """Remove one edge.

        :param eid: Its id.
        :returns: None
        :rtype: None
        """
        edge = self.edges.pop(eid)
        self.by_node[edge.start].discard(eid)
        self.by_node[edge.end].discard(eid)

    def incident(self, nodes: Iterable[str]) -> List[Edge]:
        """Return the edges touching any of the nodes.

        :param nodes: Node elementIds.
        :returns: Edges sorted by id.
        :rtype: List[Edge]
        """
        eids = set().union(*(self.by_node.get(n, set()) for n in nodes))
        return sorted((self.edges[e] for e in eids), key=lambda e: eid_sort_key(e.eid))

    def simulate(self, edge_ops: Sequence[EdgeOp]) -> None:
        """Apply planned edge operations to the in-memory edges.

        :param edge_ops: One MERGE's edge operations (moves carry their placeholder ids).
        :returns: None
        :rtype: None
        """
        for edge_op in edge_ops:
            for source in edge_op.sources:
                self.remove(source.eid)
            if edge_op.kind == "move":
                self.add(Edge(edge_op.new_eid, edge_op.type, edge_op.start, edge_op.end, dict(edge_op.props)))
            elif edge_op.kind == "merge":
                target = edge_op.target
                self.add(Edge(target.eid, target.type, target.start, target.end, dict(edge_op.props)))


def attach_edges(plan: Plan, edges: Sequence[Edge]) -> Plan:
    """Plan the edges of every MERGE in order; MERGEs with a blocking edge problem go to review.

    :param plan: The name plan from ``plan_names``.
    :param edges: Every edge touching a member of a MERGE.
    :returns: A new plan with ``edges`` and ``member_edges`` filled in.
    :rtype: Plan
    """
    store = EdgeStore(edges)
    ops: List[NameOp] = []
    review = list(plan.review)
    for op in plan.ops:
        if op.kind != "merge":
            ops.append(op)
            continue
        incident = store.incident([op.survivor.eid] + [loser.eid for loser in op.losers])
        edge_ops, problems = plan_edges(op, incident)
        if problems:
            review.append(ReviewItem("merge", op.key, op.canonical, [op.survivor] + op.losers, problems))
            continue
        for number, edge_op in enumerate(e for e in edge_ops if e.kind == "move"):
            edge_op.new_eid = f"new:{len(ops)}:{number}"
        store.simulate(edge_ops)
        ops.append(NameOp(op.kind, op.key, op.canonical, op.survivor, op.losers, edge_ops, incident))
    return Plan(ops, review, plan.refs, plan.scholar_count)


def survivor_update(op: NameOp) -> Dict[str, Any]:
    """Properties the survivor gets: canonical name, aliases and (for a merge) unioned sources.

    :param op: A MERGE or RENAME.
    :returns: Properties to SET on the survivor.
    :rtype: Dict[str, Any]
    """
    members = [op.survivor] + sorted(op.losers, key=lambda s: eid_sort_key(s.eid))
    names = {m.name for m in members} | {a for m in members for a in as_list(m.props.get("aliases"))}
    update: Dict[str, Any] = {"name": op.canonical, "aliases": sorted(names - {op.canonical})}
    if not op.losers:
        return update
    for key in UNION_PROPS:
        values = union_lists(m.props.get(key) for m in members)
        if values:
            update[key] = values
    return update


def works_by_member(op: NameOp, refs: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """Distinct BookArticles each member WROTE before the operation.

    :param op: A planned MERGE.
    :param refs: elementId -> business key info (``label`` used).
    :returns: member elementId -> works count.
    :rtype: Dict[str, int]
    """
    members = [op.survivor] + op.losers
    return {m.eid: len({e.end for e in op.member_edges if e.type == "WROTE" and e.start == m.eid
                        and refs.get(e.end, {}).get("label") == "BookArticle"}) for m in members}


# ---------------------------------------------------------------- serialisation and reports


def ref_json(eid: Optional[str], refs: Mapping[str, Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """An endpoint as ``{"eid", "label", "key"}``.

    :param eid: Node elementId.
    :param refs: elementId -> business key info.
    :returns: The endpoint description, or None.
    :rtype: Optional[Dict[str, Any]]
    """
    if eid is None:
        return None
    ref = refs.get(eid, {})
    return {"eid": eid, "label": ref.get("label"), "key": ref.get("key", eid)}


def edge_json(edge: Edge, refs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """An edge with its endpoints' business keys.

    :param edge: The edge.
    :param refs: elementId -> business key info.
    :returns: JSON-ready dict.
    :rtype: Dict[str, Any]
    """
    return {"eid": edge.eid, "type": edge.type, "start": ref_json(edge.start, refs),
            "end": ref_json(edge.end, refs), "props": edge.props}


def scholar_json(scholar: Scholar) -> Dict[str, Any]:
    """A Scholar node with its labels, properties and degree (--apply re-plans with the recorded degree).

    :param scholar: The node.
    :returns: JSON-ready dict.
    :rtype: Dict[str, Any]
    """
    return {"eid": scholar.eid, "name": scholar.name, "labels": scholar.labels, "props": scholar.props,
            "degree": scholar.degree}


def op_json(op: NameOp, refs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """One operation in the form apply executes and the change log records.

    :param op: The operation.
    :param refs: elementId -> business key info.
    :returns: JSON-ready dict.
    :rtype: Dict[str, Any]
    """
    edges = [{"kind": e.kind, "type": e.type, "start": ref_json(e.start, refs), "end": ref_json(e.end, refs),
              "sources": [edge_json(s, refs) for s in e.sources],
              "target": edge_json(e.target, refs) if e.target else None,
              "props": e.props, "new_eid": e.new_eid, "notes": e.notes} for e in op.edges]
    return {"kind": op.kind, "key": op.key, "canonical": op.canonical, "survivor": scholar_json(op.survivor),
            "losers": [scholar_json(s) for s in op.losers], "survivor_update": survivor_update(op), "edges": edges}


def review_json(item: ReviewItem) -> Dict[str, Any]:
    """One review row.

    :param item: The skipped operation.
    :returns: JSON-ready dict.
    :rtype: Dict[str, Any]
    """
    return {"kind": item.kind, "key": item.key, "proposed": item.proposed,
            "members": [{"eid": m.eid, "name": m.name} for m in item.members],
            "reasons": [f"{subject}: {reason}" for subject, reason in item.reasons]}


def plan_json(plan: Plan) -> Dict[str, Any]:
    """The whole plan as JSON (written to plan.json and compared by --apply).

    :param plan: The plan.
    :returns: JSON-ready dict.
    :rtype: Dict[str, Any]
    """
    return {"format": PLAN_FORMAT, "ops": [op_json(op, plan.refs) for op in plan.ops],
            "review": [review_json(r) for r in plan.review]}


def canonical_dumps(obj: Any) -> str:
    """Deterministic JSON text used to compare and hash plans.

    :param obj: JSON-ready data.
    :returns: Sorted-key JSON.
    :rtype: str
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def merge_category(op: NameOp) -> str:
    """merges.tsv category: ``initials_only`` when the canonical name is initials + surname.

    Such members ("F. Rosenthal" / "Rosenthal F") share one standard string, but nothing shows
    they are one person rather than two with the same initials: they reach the plan only with
    ``INCLUDE_INITIALS_ONLY_MERGES`` and are then marked for Isaac.

    :param op: A MERGE.
    :returns: ``initials_only`` or ``full_name``.
    :rtype: str
    """
    return "initials_only" if initials_only(op.canonical) else "full_name"


def summarize(plan: Plan) -> Dict[str, Any]:
    """Counts by category for summary.json.

    :param plan: The plan.
    :returns: The summary.
    :rtype: Dict[str, Any]
    """
    merges = [op for op in plan.ops if op.kind == "merge"]
    edge_ops = [e for op in merges for e in op.edges]
    return {
        "scholars": plan.scholar_count,
        "merges": len(merges),
        "merge_members": sum(1 + len(op.losers) for op in merges),
        "nodes_removed": sum(len(op.losers) for op in merges),
        "merges_initials_only": sum(1 for op in merges if merge_category(op) == "initials_only"),
        "renames": sum(1 for op in plan.ops if op.kind == "rename"),
        "edges": {
            "moved": sum(1 for e in edge_ops if e.kind == "move" and len(e.sources) == 1),
            "merged": sum(len(e.sources) - (e.kind == "move") for e in edge_ops if e.kind != "drop_self_loop"),
            "collisions": sum(1 for e in edge_ops if e.kind == "merge" or len(e.sources) > 1),
            "self_loops_dropped": sum(1 for e in edge_ops if e.kind == "drop_self_loop"),
            "relationships_removed": sum(len(e.sources) - (e.kind == "move") for e in edge_ops),
            "ops_by_type": dict(sorted(Counter(f"{e.kind}:{e.type}" for e in edge_ops).items())),
        },
        "review": {
            "total": len(plan.review),
            "by_kind": dict(sorted(Counter(r.kind for r in plan.review).items())),
            "by_reason": dict(sorted(Counter(reason for r in plan.review
                                             for reason in {why for _, why in r.reasons}).items())),
        },
    }


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """Write one tab-separated file.

    :param path: Output file.
    :param header: Column names.
    :param rows: Data rows.
    :returns: None
    :rtype: None
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def edge_op_row(op: NameOp, edge_op: EdgeOp, refs: Mapping[str, Mapping[str, Any]]) -> List[Any]:
    """One edge_ops.tsv row.

    :param op: The MERGE.
    :param edge_op: The edge operation.
    :param refs: elementId -> business key info.
    :returns: The row.
    :rtype: List[Any]
    """
    survivor = op.survivor.eid
    if edge_op.kind == "drop_self_loop":
        direction, other = "self", edge_op.sources[0].end
    else:
        direction = "out" if edge_op.start == survivor else "in"
        other = edge_op.end if edge_op.start == survivor else edge_op.start
    other_ref = ref_json(other, refs)
    return [op.key, op.canonical, edge_op.kind, edge_op.type, direction, other, other_ref["label"],
            other_ref["key"], " ".join(s.eid for s in edge_op.sources),
            edge_op.target.eid if edge_op.target else "", edge_op.new_eid or "",
            canonical_dumps(edge_op.props) if edge_op.props is not None else "", "; ".join(edge_op.notes)]


def write_reports(plan: Plan, out: Path, meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Write the dry-run directory: summary.json, plan.json and the four TSVs.

    :param plan: The plan.
    :param out: Output directory (created).
    :param meta: Extra summary fields (uri, database, timestamp).
    :returns: The summary.
    :rtype: Dict[str, Any]
    """
    out.mkdir(parents=True, exist_ok=False)
    data = plan_json(plan)
    plan_text = canonical_dumps(data)
    (out / "plan.json").write_text(plan_text, encoding="utf-8")
    summary = {**meta, **summarize(plan), "plan_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    merges = [op for op in plan.ops if op.kind == "merge"]
    write_tsv(out / "merges.tsv",
              ["key", "canonical", "category", "survivor_eid", "survivor_name", "members", "member_eids",
               "edges_moved", "edges_merged", "collisions", "self_loops_dropped"],
              ([op.key, op.canonical, merge_category(op), op.survivor.eid, op.survivor.name,
                " | ".join(m.name for m in [op.survivor] + op.losers),
                " ".join(m.eid for m in [op.survivor] + op.losers),
                sum(1 for e in op.edges if e.kind == "move" and len(e.sources) == 1),
                sum(len(e.sources) - (e.kind == "move") for e in op.edges if e.kind != "drop_self_loop"),
                sum(1 for e in op.edges if e.kind == "merge" or len(e.sources) > 1),
                sum(1 for e in op.edges if e.kind == "drop_self_loop")] for op in merges))
    write_tsv(out / "renames.tsv", ["eid", "old_name", "new_name"],
              ([op.survivor.eid, op.survivor.name, op.canonical] for op in plan.ops if op.kind == "rename"))
    write_tsv(out / "edge_ops.tsv",
              ["key", "canonical", "op", "type", "direction", "other_eid", "other_label", "other_key",
               "source_edges", "target_edge", "new_edge", "props", "notes"],
              (edge_op_row(op, e, plan.refs) for op in merges for e in op.edges))
    write_tsv(out / "review.tsv", ["kind", "key", "proposed", "names", "eids", "reasons"],
              ([r.kind, r.key, r.proposed, " | ".join(m.name for m in r.members), " ".join(m.eid for m in r.members),
                "; ".join(f"{s}: {why}" for s, why in r.reasons)] for r in plan.review))
    return summary


# ---------------------------------------------------------------- reading the graph


def business_key_expr(var: str) -> str:
    """Cypher expression for a node's business key (elementId when it has none).

    :param var: The node variable.
    :returns: A CASE expression.
    :rtype: str
    """
    cases = " ".join(f"WHEN {var}:{label} THEN {var}.{prop}" for label, prop in BUSINESS_KEYS.items())
    return f"CASE {cases} ELSE elementId({var}) END"


def business_label(labels: Sequence[str]) -> Optional[str]:
    """The label whose key identifies the node (first of BUSINESS_KEYS present, else the first label).

    :param labels: Node labels.
    :returns: The label used in the change log.
    :rtype: Optional[str]
    """
    return next((label for label in BUSINESS_KEYS if label in labels), labels[0] if labels else None)


SCHOLAR_QUERY = ("MATCH (s:Scholar) RETURN elementId(s) AS eid, labels(s) AS labels, properties(s) AS props, "
                 "COUNT { (s)--() } AS degree")
CLASH_QUERY = "MATCH (s:Scholar) MATCH (p:Person {name: s.name}) RETURN DISTINCT s.name AS name"
PLACE_QUERY = "MATCH (p:Place) WHERE p.name IS NOT NULL RETURN DISTINCT p.name AS name"
EDGE_QUERY = (
    "UNWIND $eids AS id MATCH (s)-[r]-() WHERE elementId(s) = id WITH DISTINCT r "
    "WITH r, startNode(r) AS a, endNode(r) AS b "
    "RETURN elementId(r) AS eid, type(r) AS type, properties(r) AS props, "
    f"elementId(a) AS start, labels(a) AS start_labels, {business_key_expr('a')} AS start_key, "
    f"elementId(b) AS end, labels(b) AS end_labels, {business_key_expr('b')} AS end_key"
)


def read_edges(tx: Any, node_eids: Sequence[str], refs: Dict[str, Dict[str, Any]]) -> List[Edge]:
    """Read every edge touching the nodes, recording endpoint business keys in ``refs``.

    :param tx: A transaction.
    :param node_eids: Node elementIds.
    :param refs: Filled with elementId -> ``{"label", "key"}``.
    :returns: The edges.
    :rtype: List[Edge]
    """
    edges = []
    for row in tx.run(EDGE_QUERY, eids=list(node_eids)):
        edges.append(Edge(row["eid"], row["type"], row["start"], row["end"], dict(row["props"])))
        refs[row["start"]] = {"label": business_label(row["start_labels"]), "key": row["start_key"]}
        refs[row["end"]] = {"label": business_label(row["end_labels"]), "key": row["end_key"]}
    return edges


def read_scholars(tx: Any) -> List[Scholar]:
    """Read every Scholar node with its degree.

    :param tx: A transaction.
    :returns: The Scholars.
    :rtype: List[Scholar]
    """
    return [Scholar(r["eid"], r["props"]["name"], r["degree"], list(r["labels"]), dict(r["props"]))
            for r in tx.run(SCHOLAR_QUERY)]


def read_plan(session: Any, degrees: Optional[Mapping[str, int]] = None) -> Plan:
    """Read the Scholars, the Person and Place names they are checked against and their edges in one
    read transaction, and plan everything.

    :param session: A read session.
    :param degrees: elementId -> degree to plan with instead of the current one (--apply passes the
        degrees recorded in plan.json, so an operation that changed a member's degree, such as a
        collision merge of an interacting operation, does not change the name chosen for it).
    :returns: The plan.
    :rtype: Plan
    """

    def work(tx: Any) -> Plan:
        """Read and plan inside the transaction.

        :param tx: Read transaction.
        :returns: The plan.
        :rtype: Plan
        """
        scholars = read_scholars(tx)
        for scholar in scholars:
            scholar.degree = (degrees or {}).get(scholar.eid, scholar.degree)
        plan = plan_names(scholars, {r["name"] for r in tx.run(CLASH_QUERY)},
                          {r["name"] for r in tx.run(PLACE_QUERY)})
        members = sorted({m.eid for op in plan.ops if op.kind == "merge" for m in [op.survivor] + op.losers})
        return attach_edges(plan, read_edges(tx, members, plan.refs))

    return session.execute_read(work)


def graph_counts(session: Any) -> Dict[str, int]:
    """Count nodes and relationships.

    :param session: A session.
    :returns: ``{"nodes": n, "relationships": m}``.
    :rtype: Dict[str, int]
    """
    def work(tx: Any) -> Dict[str, int]:
        """Count inside one transaction.

        :param tx: Read transaction.
        :returns: The counts.
        :rtype: Dict[str, int]
        """
        nodes = tx.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = tx.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": nodes, "relationships": rels}

    return session.execute_read(work)


# ---------------------------------------------------------------- writing (apply / revert)


def cypher_name(name: str) -> str:
    """Backtick a label or relationship type after checking it is a plain identifier.

    :param name: Label or type.
    :returns: The quoted name.
    :rtype: str
    :raises ValueError: For anything but letters, digits and underscores.
    """
    if not SAFE_NAME.match(name):
        raise ValueError(f"unsafe label or relationship type: {name!r}")
    return f"`{name}`"


def run_expect(tx: Any, query: str, expected: int, **params: Any) -> None:
    """Run a write query returning ``n`` and fail (rolling back) unless ``n == expected``.

    :param tx: Write transaction.
    :param query: Cypher returning one row with ``n``.
    :param expected: Rows the query must touch.
    :param params: Query parameters.
    :returns: None
    :rtype: None
    :raises RuntimeError: When the count differs (the graph changed under the plan).
    """
    count = tx.run(query, **params).single()["n"]
    if count != expected:
        raise RuntimeError(f"expected {expected}, touched {count}: {query[:80]}")


def create_edge(tx: Any, rel_type: str, start: str, end: str, props: Mapping[str, Any]) -> str:
    """Create one relationship with the given properties.

    :param tx: Write transaction.
    :param rel_type: Relationship type.
    :param start: Start node elementId.
    :param end: End node elementId.
    :param props: Properties.
    :returns: The new relationship's elementId.
    :rtype: str
    :raises RuntimeError: When an endpoint is missing.
    """
    row = tx.run(f"MATCH (a) WHERE elementId(a) = $start MATCH (b) WHERE elementId(b) = $end "
                 f"CREATE (a)-[r:{cypher_name(rel_type)}]->(b) SET r = $props RETURN elementId(r) AS eid",
                 start=start, end=end, props=dict(props)).single()
    if row is None:
        raise RuntimeError(f"cannot create {rel_type}: endpoint {start} or {end} is missing")
    return row["eid"]


def delete_edges(tx: Any, eids: Sequence[str]) -> None:
    """Delete relationships by elementId (all must exist).

    :param tx: Write transaction.
    :param eids: Relationship elementIds.
    :returns: None
    :rtype: None
    """
    if eids:
        run_expect(tx, "MATCH ()-[r]->() WHERE elementId(r) IN $eids DELETE r RETURN count(*) AS n",
                   len(set(eids)), eids=list(eids))


def set_edge_props(tx: Any, eid: str, props: Mapping[str, Any]) -> None:
    """Replace one relationship's properties.

    :param tx: Write transaction.
    :param eid: Relationship elementId.
    :param props: The new full property map.
    :returns: None
    :rtype: None
    """
    run_expect(tx, "MATCH ()-[r]->() WHERE elementId(r) = $eid SET r = $props RETURN count(*) AS n", 1,
               eid=eid, props=dict(props))


def apply_op(tx: Any, op: Mapping[str, Any]) -> Dict[str, str]:
    """Execute one operation: rewire/merge edges, delete the losers, update the survivor.

    :param tx: Write transaction.
    :param op: One operation from plan.json, placeholders already resolved.
    :returns: Placeholder id -> elementId of every edge created.
    :rtype: Dict[str, str]
    """
    created: Dict[str, str] = {}
    for edge_op in op["edges"]:
        if edge_op["kind"] == "move":
            created[edge_op["new_eid"]] = create_edge(tx, edge_op["type"], edge_op["start"]["eid"],
                                                      edge_op["end"]["eid"], edge_op["props"])
        elif edge_op["kind"] == "merge":
            set_edge_props(tx, edge_op["target"]["eid"], edge_op["props"])
        delete_edges(tx, [s["eid"] for s in edge_op["sources"]])
    losers = [loser["eid"] for loser in op["losers"]]
    if losers:
        run_expect(tx, "MATCH (n) WHERE elementId(n) IN $eids DELETE n RETURN count(*) AS n", len(losers),
                   eids=losers)
    run_expect(tx, "MATCH (s) WHERE elementId(s) = $eid AND s.name = $old SET s += $props RETURN count(*) AS n", 1,
               eid=op["survivor"]["eid"], old=op["survivor"]["name"], props=op["survivor_update"])
    return created


def resolve_placeholders(op: Dict[str, Any], created: Mapping[str, str]) -> Dict[str, Any]:
    """Replace ``new:<op>:<n>`` ids of edges created by earlier operations with their elementIds.

    :param op: One operation from plan.json.
    :param created: Placeholder -> elementId of the edges created so far.
    :returns: A copy of the operation with real ids.
    :rtype: Dict[str, Any]
    """
    op = json.loads(json.dumps(op))
    for edge_op in op["edges"]:
        for edge in edge_op["sources"] + ([edge_op["target"]] if edge_op["target"] else []):
            edge["eid"] = created.get(edge["eid"], edge["eid"])
    return op


def append_log(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record to the change log and flush it to disk.

    :param path: Change log file.
    :param record: The record.
    :returns: None
    :rtype: None
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc_stamp() -> str:
    """Current UTC time as ``YYYYMMDDTHHMMSSZ``.

    :returns: The stamp.
    :rtype: str
    """
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def failure_event(error: BaseException) -> str:
    """Change-log event for an operation whose write raised: ``unknown`` when it may have committed.

    The driver raises ``IncompleteCommit`` when the connection drops while the COMMIT reply is
    pending: the merge may have happened, so the log must not say ``failed`` (revert skips
    failed operations).

    :param error: The exception from ``execute_write``.
    :returns: ``"unknown"`` or ``"failed"``.
    :rtype: str
    """
    from neo4j.exceptions import IncompleteCommit  # only the write path needs the driver

    return "unknown" if isinstance(error, IncompleteCommit) else "failed"


def select_ops(ops: Sequence[Mapping[str, Any]], only: Optional[Collection[str]]) -> List[int]:
    """Indices of the plan.json operations to apply: all of them, or those ``--only`` names.

    An operation matches when its canonical name or a member's name (as in plan.json) equals one
    of the names, compared in NFC.

    :param ops: The operations of plan.json.
    :param only: Names from ``--only``, or None for every operation.
    :returns: Matching indices in plan order.
    :rtype: List[int]
    :raises SystemExit: When a name matches no operation (a typo must not silently apply nothing).
    """
    if only is None:
        return list(range(len(ops)))
    wanted = {unicodedata.normalize("NFC", name) for name in only}
    indices, matched = [], set()
    for index, op in enumerate(ops):
        names = {op["canonical"], op["survivor"]["name"], *(loser["name"] for loser in op["losers"])}
        hits = wanted & {unicodedata.normalize("NFC", name) for name in names}
        if hits:
            indices.append(index)
            matched |= hits
    if wanted - matched:
        raise SystemExit(f"--only names that match no operation in plan.json: {sorted(wanted - matched)}")
    return indices


def store_id(eids: Iterable[str]) -> str:
    """The database part of elementIds (``"4:<store>:12"`` -> ``"<store>"``).

    :param eids: Node elementIds of one database.
    :returns: The store id (empty when there are none).
    :rtype: str
    :raises RuntimeError: When the ids come from more than one store.
    """
    stores = {eid.split(":")[1] for eid in eids if eid.count(":") == 2}
    if len(stores) > 1:
        raise RuntimeError(f"elementIds from several stores: {sorted(stores)}")
    return next(iter(stores), "")


def target_of(uri: str, database: str, eids: Iterable[str]) -> Dict[str, str]:
    """The target a change log is written for and applies to.

    :param uri: Bolt URI.
    :param database: Database name.
    :param eids: Node elementIds read from that database.
    :returns: ``{"uri", "database", "store"}``.
    :rtype: Dict[str, str]
    """
    return {"uri": uri, "database": database, "store": store_id(eids)}


def read_log(path: Path) -> List[Dict[str, Any]]:
    """Read one change log.

    :param path: A ``changelog_*.jsonl`` file.
    :returns: Its records in file order.
    :rtype: List[Dict[str, Any]]
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log_target(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The target recorded by a change log's first record.

    :param records: The log's records.
    :returns: ``{"uri", "database", "store"}``, or None for a log without a target record.
    :rtype: Optional[Dict[str, Any]]
    """
    first = records[0] if records else {}
    return {k: first.get(k) for k in ("uri", "database", "store")} if first.get("event") == TARGET_EVENT else None


def target_logs(plan_dir: Path, target: Mapping[str, str]) -> Tuple[List[Path], List[Path]]:
    """Split a directory's change logs into those of ``target`` and the others.

    :param plan_dir: A dry-run directory.
    :param target: ``{"uri", "database", "store"}``.
    :returns: Logs of this target and logs of another (or no recorded) target, each oldest first.
    :rtype: Tuple[List[Path], List[Path]]
    """
    mine: List[Path] = []
    other: List[Path] = []
    for log in sorted(plan_dir.glob("changelog_*.jsonl")):
        (mine if log_target(read_log(log)) == dict(target) else other).append(log)
    return mine, other


def logged_state(plan_dir: Path, target: Mapping[str, str]) -> Tuple[Set[int], Dict[str, str], List[str]]:
    """Read the change logs of earlier applies of this plan to this target.

    :param plan_dir: The dry-run directory holding plan.json and its ``changelog_*.jsonl``.
    :param target: ``{"uri", "database", "store"}`` of the graph being applied to; logs written for
        another target (a scratch copy, say) are ignored.
    :returns: plan.json indices logged ``done`` and not ``reverted``, placeholder -> elementId of
        the edges those operations created, and the names of the ignored logs.
    :rtype: Tuple[Set[int], Dict[str, str], List[str]]
    """
    done: Dict[int, Dict[str, str]] = {}
    mine, other = target_logs(plan_dir, target)
    for log in mine:
        for record in read_log(log):
            if record["event"] == "done":
                done[record["op"]] = record["created_edges"]
            elif record["event"] == "reverted":
                done.pop(record["op"], None)
    created = {placeholder: eid for edges in done.values() for placeholder, eid in edges.items()}
    return set(done), created, [log.name for log in other]


def member_eids(op: Mapping[str, Any]) -> List[str]:
    """The elementIds of a plan.json operation's members, survivor first.

    :param op: One plan.json operation.
    :returns: Survivor and loser ids.
    :rtype: List[str]
    """
    return [op["survivor"]["eid"]] + [loser["eid"] for loser in op["losers"]]


def op_groups(ops: Sequence[Mapping[str, Any]]) -> List[int]:
    """Group the plan.json operations that interact; they must be applied together.

    Operation i touches operation j when an edge operation of i has an endpoint (of the resulting
    edge, a consumed edge or the merged-into edge) among j's members, or when both reference one
    edge (one creates, merges into or deletes an edge the other consumes or merges into). The dry
    run planned each of them on the graph as the other leaves it, with the degrees from before
    both: applying one alone can apply the other's edge data twice or make it stale for good.

    :param ops: The operations of plan.json.
    :returns: Per operation, the lowest index of its connected group.
    :rtype: List[int]
    """
    parent = list(range(len(ops)))

    def find(index: int) -> int:
        """Root of an operation's group.

        :param index: Operation index.
        :returns: The root index.
        :rtype: int
        """
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        """Join two operations' groups under the lower root.

        :param first: Operation index.
        :param second: Operation index.
        :returns: None
        :rtype: None
        """
        roots = sorted((find(first), find(second)))
        parent[roots[1]] = roots[0]

    owner = {eid: index for index, op in enumerate(ops) for eid in member_eids(op)}
    users: Dict[str, Set[int]] = defaultdict(set)
    for index, op in enumerate(ops):
        for edge_op in op["edges"]:
            edges = edge_op["sources"] + ([edge_op["target"]] if edge_op["target"] else [])
            for eid in [edge["eid"] for edge in edges] + ([edge_op["new_eid"]] if edge_op.get("new_eid") else []):
                users[eid].add(index)
            ends = [edge_op["start"], edge_op["end"]] + [edge[end] for edge in edges for end in ("start", "end")]
            for end in ends:
                if end and owner.get(end["eid"], index) != index:
                    union(index, owner[end["eid"]])
    for indices in users.values():
        first, *rest = sorted(indices)
        for other in rest:
            union(first, other)
    return [find(index) for index in range(len(ops))]


def group_gaps(groups: Sequence[int], selected: Collection[int], settled: Collection[int]) -> List[int]:
    """Operations that share a group with a selected one but are neither selected nor already applied.

    :param groups: ``op_groups`` of plan.json.
    :param selected: Selected indices.
    :param settled: Indices the graph shows applied.
    :returns: The missing indices, in plan order.
    :rtype: List[int]
    """
    wanted = {groups[index] for index in selected}
    return [index for index, group in enumerate(groups)
            if group in wanted and index not in selected and index not in settled]


def limit_to_groups(applicable: Sequence[int], limit: Optional[int], groups: Sequence[int]) -> List[int]:
    """The first ``limit`` applicable operations, extended to whole groups (``op_groups``).

    :param applicable: Applicable indices in plan order.
    :param limit: ``--limit``, or None for all.
    :param groups: ``op_groups`` of plan.json.
    :returns: Indices to apply, in plan order.
    :rtype: List[int]
    """
    if limit is None:
        return list(applicable)
    wanted = {groups[index] for index in applicable[:limit]}
    return [index for index in applicable if groups[index] in wanted]


def recorded_degrees(ops: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """The member degrees plan.json recorded at the dry run.

    :param ops: The operations of plan.json.
    :returns: elementId -> degree.
    :rtype: Dict[str, int]
    """
    return {member["eid"]: member["degree"] for op in ops for member in [op["survivor"], *op["losers"]]}


def under_temp_dir(path: Path) -> bool:
    """Return True when the path lies in a directory the OS may clear (``/tmp``, ``/var/folders``).

    :param path: A directory.
    :returns: Whether it is under a temporary directory.
    :rtype: bool
    """
    resolved = path.resolve()
    roots = {Path(tempfile.gettempdir()).resolve(), *(Path(root).resolve() for root in TEMP_ROOTS)}
    return any(resolved == root or root in resolved.parents for root in roots)


def comparable(value: Any, keys: Sequence[str]) -> Any:
    """An operation's JSON in the form two plans of one graph state share, however many ops were applied.

    Placeholders ``new:<op index>:<n>`` become ``new:<that op's key>:<n>`` (positions shift once
    earlier operations are applied) and endpoint references drop their business key (an endpoint
    renamed by an applied operation keeps its elementId).

    :param value: An operation (or part of one) from ``plan_json``.
    :param keys: The identity key of each operation of that plan, by position.
    :returns: The normalised copy.
    :rtype: Any
    """
    if isinstance(value, dict):
        endpoint = set(value) == {"eid", "label", "key"}
        return {k: comparable(v, keys) for k, v in value.items() if not (endpoint and k == "key")}
    if isinstance(value, list):
        return [comparable(v, keys) for v in value]
    match = PLACEHOLDER.match(value) if isinstance(value, str) else None
    return f"new:{keys[int(match[1])]}:{match[2]}" if match else value


def scholar_names(plan: Plan) -> Dict[str, str]:
    """Every Scholar's current name, from a plan's business keys.

    :param plan: A plan made by ``read_plan``.
    :returns: elementId -> name.
    :rtype: Dict[str, str]
    """
    return {eid: ref["key"] for eid, ref in plan.refs.items() if ref.get("label") == "Scholar"}


def applied_in_graph(op: Mapping[str, Any], names: Mapping[str, str]) -> bool:
    """Whether the graph shows the operation applied: its losers are gone and its survivor bears the canonical name.

    :param op: One plan.json operation.
    :param names: elementId -> name of every Scholar now.
    :returns: True when applied.
    :rtype: bool
    """
    return (names.get(op["survivor"]["eid"]) == op["canonical"]
            and not any(loser["eid"] in names for loser in op["losers"]))


def check_ops(expected: Sequence[Mapping[str, Any]], indices: Sequence[int], plan: Plan, done: Collection[int],
              created: Mapping[str, str]) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    """Check the selected plan.json operations one by one against a fresh plan of the graph.

    An operation the graph shows applied is skipped. One a change log of this target records as
    applied while the graph does not show it is a problem (reverted or restored outside this
    tool). Any other is applicable when the re-plan has an operation with its key that equals it
    (``comparable`` forms, after the elementIds of edges that applied operations created are put
    in for their placeholders) and every edge it consumes exists or is created by a selected
    operation before it; otherwise it is stale.

    :param expected: The operations of plan.json.
    :param indices: The selected indices, in plan order.
    :param plan: The plan made from the graph now (with the degrees plan.json recorded).
    :param done: Indices the change logs of this target record as applied.
    :param created: Placeholder -> elementId of the edges applied operations created.
    :returns: ``{"op", "canonical", "seen_in"}`` per applied operation, applicable indices, and one
        message per stale operation.
    :rtype: Tuple[List[Dict[str, Any]], List[int], List[str]]
    """
    names = scholar_names(plan)
    replanned = json.loads(canonical_dumps(plan_json(plan)))["ops"]
    by_key = {op["key"]: op for op in replanned}
    expected_keys, replanned_keys = [op["key"] for op in expected], [op["key"] for op in replanned]
    applied: List[Dict[str, Any]] = []
    applicable: List[int] = []
    problems: List[str] = []
    for index in indices:
        op, label = expected[index], f"op {index} ({expected[index]['canonical']})"
        if applied_in_graph(op, names):
            applied.append({"op": index, "canonical": op["canonical"],
                            "seen_in": "change log and graph" if index in done else "graph"})
            continue
        if index in done:
            problems.append(f"{label}: a change log of this target says it was applied, but the graph does not "
                            "show it (reverted or restored outside this tool?)")
            continue
        resolved = resolve_placeholders(op, created)
        fresh = by_key.get(op["key"])
        needs = sorted({int(m[1]) for e in resolved["edges"] for edge in e["sources"] + [e["target"] or {}]
                        if (m := PLACEHOLDER.match(edge.get("eid", "")))} - set(indices))
        if fresh is None or canonical_dumps(comparable(fresh, replanned_keys)) != \
                canonical_dumps(comparable(resolved, expected_keys)):
            problems.append(f"{label}: its members or their edges no longer match the plan")
        elif needs:
            problems.append(f"{label}: consumes edges that ops {needs} create; select those too (an applied "
                            "op's created edge ids come from a change log of this target)")
        else:
            applicable.append(index)
    return applied, applicable, problems


def load_plan_ops(plan_dir: Path) -> List[Dict[str, Any]]:
    """Read the operations of a dry run's plan.json, refusing a plan of another format.

    :param plan_dir: The dry-run directory.
    :returns: The operations.
    :rtype: List[Dict[str, Any]]
    :raises SystemExit: When plan.json was written by another version of this tool.
    """
    data = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    if data.get("format") != PLAN_FORMAT:
        raise SystemExit(f"{plan_dir / 'plan.json'} has format {data.get('format')!r}, this tool needs "
                         f"{PLAN_FORMAT}: it was written by another version of the tool, run the dry run again")
    return data["ops"]


def run_apply(driver: Any, database: str, plan_dir: Path, limit: Optional[int],
              only: Optional[Collection[str]] = None, uri: str = DEFAULT_URI) -> Dict[str, Any]:
    """Re-plan, check the selected operations one by one, apply the pending ones and verify.

    :param driver: Neo4j driver.
    :param database: Database name.
    :param plan_dir: The dry-run directory holding plan.json (and the change logs of earlier runs).
    :param limit: Apply only the first N pending operations, extended to whole groups of
        interacting operations (``op_groups``).
    :param only: Apply only the operations matching these names (``select_ops``); the selection
        must hold every pending operation of the groups it touches.
    :param uri: The Bolt URI ``driver`` is connected to (recorded in the change log; logs of other
        targets are ignored).
    :returns: The verification report, with the operations skipped as already applied.
    :rtype: Dict[str, Any]
    :raises SystemExit: When ``--only`` splits a group, or a selected operation no longer matches the graph.
    """
    expected = load_plan_ops(plan_dir)
    selected = select_ops(expected, only)
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        plan = read_plan(session, recorded_degrees(expected))
        before = graph_counts(session)
    target = target_of(uri, database, plan.refs)
    done, created, ignored = logged_state(plan_dir, target)
    names = scholar_names(plan)
    groups = op_groups(expected)
    gaps = group_gaps(groups, selected, {i for i, op in enumerate(expected) if applied_in_graph(op, names)})
    if gaps:
        raise SystemExit("--only must take whole groups of interacting operations (one touches the other's members "
                         "or edges); add these names: " + ", ".join(expected[i]["canonical"] for i in gaps))
    applied, applicable, problems = check_ops(expected, selected, plan, done, created)
    if problems:
        raise SystemExit("The graph changed since the dry run for these operations (run the dry run again, or "
                         "leave them out with --only):\n" + "\n".join(problems))
    todo = limit_to_groups(applicable, limit, groups)
    extra = todo[limit:] if limit is not None else []
    if extra:
        print(f"--limit {limit}: also applying ops {extra}, which interact with the first {limit}: "
              + ", ".join(expected[i]["canonical"] for i in extra), file=sys.stderr, flush=True)
    result: Dict[str, Any] = {"changelog": None, "applied": 0, "already_applied": applied,
                              "ignored_changelogs": ignored}
    if not todo:
        return {**result, "ok": True}
    log_path = plan_dir / f"changelog_{utc_stamp()}.jsonl"
    print(f"change log: {log_path}", file=sys.stderr, flush=True)
    try:
        append_log(log_path, {"event": TARGET_EVENT, **target})
        with driver.session(database=database) as session:
            for index in todo:
                op = resolve_placeholders(expected[index], created)
                append_log(log_path, {"event": "before", "op": index, **op})
                try:
                    new_edges = session.execute_write(apply_op, op)
                except Exception as error:  # record why this operation did not (or may not) happen, then stop
                    append_log(log_path, {"event": failure_event(error), "op": index, "error": str(error)})
                    raise
                created.update(new_edges)
                append_log(log_path, {"event": "done", "op": index, "created_edges": new_edges})
        replanned = {op.key: op for op in plan.ops}
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            report = verify(session, plan, [replanned[expected[index]["key"]] for index in todo], before)
    finally:
        print(f"change log: {log_path}", file=sys.stderr, flush=True)
    return {**result, "changelog": str(log_path), "applied": len(todo), **report}


def revert_state(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Tuple[Dict, Dict]], List[int], Dict, Dict]:
    """Find the applied, not yet reverted operations of one change log, and the id maps of its reverts.

    :param records: The log's records in file order.
    :returns: ``(before, created_edges)`` pairs in apply order; the operations that may have been
        applied (a before-state with neither a done nor a failed record, or an ``unknown`` commit
        outcome); and the node and edge id maps written by earlier reverts of this log, in file order.
    :rtype: Tuple[List[Tuple[Dict, Dict]], List[int], Dict, Dict]
    """
    events: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        if "op" in record:
            events[record["event"]][record["op"]] = record
    unsure = sorted((set(events["before"]) - set(events["done"]) - set(events["failed"])) | set(events["unknown"]))
    node_map: Dict[str, str] = {}
    edge_map: Dict[str, str] = {}
    for record in records:
        if record["event"] == "reverted":
            node_map.update(record["node_map"])
            edge_map.update(record["edge_map"])
    todo = [(dict(events["before"][i]), dict(events["done"][i]["created_edges"]))
            for i in sorted(events["done"]) if i not in events["reverted"]]
    return todo, unsure, node_map, edge_map


def pending_reverts(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Tuple[Dict, Dict]], Dict, Dict]:
    """Find the applied, not yet reverted operations of one change log.

    :param records: The log's records in file order.
    :returns: ``(before, created_edges)`` pairs in apply order, plus the node and edge id maps
        written by earlier (interrupted) reverts of this log.
    :rtype: Tuple[List[Tuple[Dict, Dict]], Dict, Dict]
    :raises SystemExit: When an operation has a before-state but neither a done nor a failed
        record, or its commit outcome is ``unknown``: it may have been applied.
    """
    todo, unsure, node_map, edge_map = revert_state(records)
    if unsure:
        raise SystemExit(f"operations {unsure} may have been applied (no done/failed record, or the commit outcome "
                         "is unknown): check them by hand before reverting")
    return todo, node_map, edge_map


def revert_op(tx: Any, before: Mapping[str, Any], created: Mapping[str, str], node_map: Mapping[str, str],
              edge_map: Mapping[str, str]) -> Dict[str, Dict[str, str]]:
    """Undo one operation from its before-state.

    Restores the survivor's properties (and name), deletes the edges the operation created,
    recreates the losers and every consumed edge, and restores merged-into edges' properties.

    :param tx: Write transaction.
    :param before: The operation's ``before`` record.
    :param created: Placeholder -> elementId of the edges it created.
    :param node_map: Old -> new node ids from operations reverted earlier in this run (one
        lookup per id, see ``current_id``).
    :param edge_map: Old -> new edge ids from operations reverted earlier in this run.
    :returns: ``{"nodes": old -> new, "edges": old -> new}`` for the ids this revert recreated.
    :rtype: Dict[str, Dict[str, str]]
    """
    survivor = before["survivor"]
    run_expect(tx, "MATCH (s) WHERE elementId(s) = $eid SET s = $props RETURN count(*) AS n", 1,
               eid=current_id(survivor["eid"], {}, node_map), props=survivor["props"])
    delete_edges(tx, [current_id(eid, {}, edge_map) for eid in created.values()])
    new_nodes: Dict[str, str] = {}
    for loser in before["losers"]:
        labels = ":".join(cypher_name(label) for label in loser["labels"])
        new_nodes[loser["eid"]] = tx.run(f"CREATE (n:{labels}) SET n = $props RETURN elementId(n) AS eid",
                                         props=loser["props"]).single()["eid"]
    new_edges: Dict[str, str] = {}
    for edge_op in before["edges"]:
        for source in edge_op["sources"]:
            new_edges[source["eid"]] = create_edge(tx, source["type"],
                                                   current_id(source["start"]["eid"], new_nodes, node_map),
                                                   current_id(source["end"]["eid"], new_nodes, node_map),
                                                   source["props"])
        if edge_op["kind"] == "merge":
            target = edge_op["target"]
            set_edge_props(tx, current_id(target["eid"], new_edges, edge_map), target["props"])
    return {"nodes": new_nodes, "edges": new_edges}


def current_target(driver: Any, database: str, uri: str) -> Dict[str, str]:
    """The target a session of ``driver`` reaches, for comparison with a change log's.

    :param driver: Neo4j driver.
    :param database: Database name.
    :param uri: The Bolt URI the driver is connected to.
    :returns: ``{"uri", "database", "store"}``.
    :rtype: Dict[str, str]
    """
    def work(tx: Any) -> List[str]:
        """Read one elementId.

        :param tx: Read transaction.
        :returns: Zero or one elementId.
        :rtype: List[str]
        """
        return [row["eid"] for row in tx.run(STORE_QUERY)]

    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        return target_of(uri, database, session.execute_read(work))


def revert_order_problems(logs: Sequence[Path], target: Mapping[str, str]) -> List[str]:
    """Check that no log is reverted while a newer log of its directory still has applied operations.

    A newer change log with applied, unreverted (or possibly applied) operations must be reverted
    first: in this run and earlier in ``logs``. Logs of another target are ignored.

    :param logs: Resolved change logs, in the order they will be reverted.
    :param target: ``{"uri", "database", "store"}`` of the graph.
    :returns: One message per violation.
    :rtype: List[str]
    """
    position = {log: index for index, log in enumerate(logs)}
    problems: List[str] = []
    for index, log in enumerate(logs):
        for sibling in target_logs(log.parent, target)[0]:
            todo, unsure, _, _ = revert_state(read_log(sibling))
            if sibling.name > log.name and (todo or unsure) and position.get(sibling, len(logs)) > index:
                problems.append(f"{sibling.name} is newer than {log.name} and still has applied operations: revert "
                                f"it first (list it before {log.name})")
    return problems


def later_maps(log: Path, target: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """The node and edge id maps written by reverts of newer logs of the directory, then of ``log`` itself.

    Every newer log was reverted after ``log`` was applied (``revert_order_problems``), so its maps
    hold the new ids of entities ``log`` recorded. Newest first, so the revert closest to ``log``
    wins a clash of two recorded ids (the same as reverting them in one run).

    :param log: The change log about to be reverted (resolved).
    :param target: ``{"uri", "database", "store"}`` of the graph.
    :returns: ``(node_map, edge_map)``.
    :rtype: Tuple[Dict[str, str], Dict[str, str]]
    """
    newer = [sibling for sibling in target_logs(log.parent, target)[0] if sibling.name > log.name]
    node_map: Dict[str, str] = {}
    edge_map: Dict[str, str] = {}
    for source in sorted(newer, key=lambda path: path.name, reverse=True) + [log]:
        _, _, nodes, edges = revert_state(read_log(source))
        node_map.update(nodes)
        edge_map.update(edges)
    return node_map, edge_map


def run_revert(driver: Any, database: str, logs: Sequence[Path], uri: str = DEFAULT_URI) -> Dict[str, Any]:
    """Revert change logs, newest first, one write transaction per operation.

    :param driver: Neo4j driver.
    :param database: Database name.
    :param logs: Change logs, newest first.
    :param uri: The Bolt URI ``driver`` is connected to; each log must have been written for this
        URI, database and store.
    :returns: Operations reverted per log.
    :rtype: Dict[str, Any]
    :raises SystemExit: When a log was written for another target, or a newer log of its directory
        still has applied operations and is not reverted before it.
    """
    target = current_target(driver, database, uri)
    logs = [Path(log).resolve() for log in logs]
    problems = [f"{log.name} was written for {log_target(read_log(log)) or 'an unrecorded target'}, not {target}"
                for log in logs if log_target(read_log(log)) != target]
    problems += [f"{log.name}: operations {unsure} may have been applied (no done/failed record, or the commit "
                 "outcome is unknown): check them by hand before reverting"
                 for log in logs for unsure in [revert_state(read_log(log))[1]] if unsure]
    problems += revert_order_problems(logs, target)
    if problems:
        raise SystemExit("Refusing to revert:\n" + "\n".join(problems))
    reverted: Dict[str, int] = {}
    with driver.session(database=database) as session:
        for log in logs:
            todo, _, _ = pending_reverts(read_log(log))
            node_map, edge_map = later_maps(log, target)
            for before, created in reversed(todo):
                maps = session.execute_write(revert_op, before, created, dict(node_map), dict(edge_map))
                node_map.update(maps["nodes"])
                edge_map.update(maps["edges"])
                append_log(log, {"event": "reverted", "op": before["op"], "node_map": maps["nodes"],
                                 "edge_map": maps["edges"]})
            reverted[str(log)] = len(todo)
    return {"reverted": reverted}


# ---------------------------------------------------------------- verification


def expected_counts(applied: Sequence[NameOp], before: Mapping[str, int]) -> Dict[str, int]:
    """Node and relationship counts the graph should have after the operations.

    :param applied: Operations applied.
    :param before: Counts before the apply.
    :returns: Expected ``{"nodes", "relationships"}``.
    :rtype: Dict[str, int]
    """
    removed = sum(len(e.sources) - (e.kind == "move") for op in applied for e in op.edges)
    return {"nodes": before["nodes"] - sum(len(op.losers) for op in applied),
            "relationships": before["relationships"] - removed}


def parallel_failures(op: NameOp, after_edges: Sequence[Edge]) -> List[str]:
    """Identities the survivor holds more often after the operation than before (parallel edges created).

    :param op: An applied operation.
    :param after_edges: Edges touching the survivor now.
    :returns: Descriptions of the offending identities.
    :rtype: List[str]
    """
    survivor = op.survivor.eid
    loser_ids = {loser.eid for loser in op.losers}
    before = Counter(edge_identity(e, survivor, {}) for e in op.member_edges
                     if survivor in (e.start, e.end) and not loser_ids & {e.start, e.end})
    after = Counter(edge_identity(e, survivor, {}) for e in after_edges if survivor in (e.start, e.end))
    return [f"{op.canonical}: {identity[:3]} x{n}" for identity, n in after.items() if n > max(1, before[identity])]


def verify(session: Any, plan: Plan, applied: Sequence[NameOp], before: Mapping[str, int]) -> Dict[str, Any]:
    """Check counts, canonical names, parallel edges and works counts after an apply.

    The canonical-name check expects exactly one Scholar per applied operation's identity key, the
    survivor with the canonical name, apart from the ``?`` members a ``lost_accent`` review row
    deliberately left out of the merge (they share the key and stay as they are).

    :param session: A read session.
    :param plan: The plan that was applied (before-state edges, refs and review rows).
    :param applied: The operations applied.
    :param before: Counts before the apply.
    :returns: The report, with ``ok``.
    :rtype: Dict[str, Any]
    """
    after = graph_counts(session)
    expected = expected_counts(applied, before)
    groups = group_by_key(session.execute_read(read_scholars))
    held = {m.eid for item in plan.review if item.kind == "lost_accent" for m in item.members}
    holders = {op.key: [(s.eid, s.name) for s in groups.get(op.key, []) if s.eid not in held] for op in applied}
    names = [f"{op.key}: {holders[op.key]}" for op in applied if holders[op.key] != [(op.survivor.eid, op.canonical)]]
    survivors = [op.survivor.eid for op in applied if op.kind == "merge"]
    after_edges = session.execute_read(lambda tx: read_edges(tx, survivors, {}))
    parallels = [f for op in applied if op.kind == "merge" for f in parallel_failures(op, after_edges)]
    works_after = dict(session.execute_read(lambda tx: [
        (r["eid"], r["works"]) for r in tx.run(
            "MATCH (s) WHERE elementId(s) IN $eids OPTIONAL MATCH (s)-[:WROTE]->(b:BookArticle) "
            "RETURN elementId(s) AS eid, count(DISTINCT b) AS works", eids=survivors)]))
    works_before = {op.survivor.eid: max(works_by_member(op, plan.refs).values()) for op in applied
                    if op.kind == "merge"}
    works = [f"{eid}: {works_after.get(eid, 0)} < {n}" for eid, n in works_before.items()
             if works_after.get(eid, 0) < n]
    report = {"counts": {"before": dict(before), "expected": expected, "after": after, "ok": after == expected},
              "canonical_names": {"failures": names, "ok": not names},
              "parallel_edges": {"failures": parallels, "ok": not parallels},
              "works_count": {"failures": works, "ok": not works}}
    report["ok"] = all(part["ok"] for part in report.values())
    return report


# ---------------------------------------------------------------- command line


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default: sys.argv).
    :returns: The parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--uri", default=DEFAULT_URI, help=f"Bolt URI (default {DEFAULT_URI})")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--out", type=Path, help="dry-run directory (default docs/kg_consolidation/dryrun_<UTC>)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write the plan in --plan to the graph")
    mode.add_argument("--revert", type=Path, nargs="+", metavar="CHANGELOG", help="change logs, newest first")
    parser.add_argument("--plan", type=Path, help="dry-run directory to apply")
    parser.add_argument("--limit", type=int,
                        help="apply only the first N pending operations (extended to whole groups of interacting ones)")
    parser.add_argument("--only", metavar="NAMES",
                        help="apply only the operations whose canonical or member name is one of these "
                             "comma-separated names (give the canonical name for a member name with a comma)")
    parser.add_argument("--allow-temp-plan", action="store_true",
                        help="allow --plan under a temporary directory (its change logs may be deleted by the OS)")
    args = parser.parse_args(argv)
    if args.apply and args.plan is None:
        parser.error("--apply needs --plan <dry-run dir>")
    if not args.apply and (args.plan is not None or args.limit is not None or args.only is not None
                           or args.allow_temp_plan):
        parser.error("--plan, --limit, --only and --allow-temp-plan only go with --apply")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.only is not None:
        args.only = [name.strip() for name in args.only.split(",") if name.strip()]
        if not args.only:
            parser.error("--only needs at least one name")
    return args


def check_plan_location(plan: Path, allow_temp: bool) -> None:
    """Refuse to apply from a plan directory the OS may clear: its change logs are the only way to revert.

    :param plan: The ``--plan`` directory.
    :param allow_temp: ``--allow-temp-plan`` was given.
    :returns: None
    :rtype: None
    :raises SystemExit: When the directory is under a temporary directory and that is not allowed.
    """
    if under_temp_dir(plan) and not allow_temp:
        raise SystemExit(f"--plan {plan} is under a temporary directory: the change logs written there are the only "
                         "way to revert. Run the dry run into docs/kg_consolidation/ (or pass --allow-temp-plan)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a dry run, an apply or a revert.

    :param argv: Arguments (default: sys.argv).
    :returns: Process exit code (1 when verification fails).
    :rtype: int
    """
    args = parse_args(argv)
    if args.apply:
        check_plan_location(args.plan, args.allow_temp_plan)
    from neo4j import GraphDatabase  # the planning code above needs no driver

    from scripts.kg.export_graph import neo4j_auth

    with GraphDatabase.driver(args.uri, auth=neo4j_auth()) as driver:
        if args.revert:
            result = run_revert(driver, args.database, args.revert, args.uri)
        elif args.apply:
            result = run_apply(driver, args.database, args.plan, args.limit, args.only, args.uri)
        else:
            stamp = utc_stamp()
            out = args.out or REPO_ROOT / "docs" / "kg_consolidation" / f"dryrun_{stamp}"
            with driver.session(database=args.database, default_access_mode=READ_ACCESS) as session:
                plan = read_plan(session)
            shown = out.resolve().relative_to(REPO_ROOT) if REPO_ROOT in out.resolve().parents else out
            result = write_reports(plan, out, {"generated_utc": stamp, "uri": args.uri, "database": args.database,
                                               "out": str(shown)})
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
