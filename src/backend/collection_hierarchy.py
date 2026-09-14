"""Browse-by-collection tree built from canonical document ids.

The merged index's ``collection`` / ``sub_collection`` fields come from three
sources that label the same holdings differently ("Taylor-Schechter",
"Cambridge CUL", "Lewis-Gibson", "Oriental Manuscripts" are all Cambridge
University Library).  Every merged document id, however, starts with one
descriptive institution token (``Cambridge_CUL_T_S_AS_1``,
``New_York_JTS_ENA_NS_10_1``, ``Manchester_JRL_B_1922`` …; see
``historical-document-analysis/src/datasets/merging/institution_tokens.py``),
followed by the shelf mark's own segments.  This module classifies each
document from that id into

    institution  →  series  →  (sub-series or numeric range)  →  shelf marks

and emits exactly the JSON shape ``CollectionBrowser.jsx`` renders.  Cambridge
is the one institution large enough to need the third level everywhere
(T-S Additional Series boxes, old-series classes, Mosseri Roman series, L-G
sub-series); smaller institutions fall back to numeric ranges or a flat list.

The tree needs every document id once, so it is built from a ``search_after``
scan (~72k rows, a few seconds) and cached in memory and on the data volume.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_TTL_S = 24 * 3600
RANGE_THRESHOLD = 300  # shelf marks above which a series is split into numeric ranges

# ---------------------------------------------------------------------------
# Institutions (longest token first so "Cambridge_Lewis_Gibson" wins over
# "Cambridge").  Cambridge's three tokens deliberately share one label.
# ---------------------------------------------------------------------------

_CAMBRIDGE = "Cambridge University Library"
INSTITUTIONS: List[Tuple[str, str]] = [
    ("Cambridge_Lewis_Gibson", _CAMBRIDGE),
    ("Cambridge_Mosseri", _CAMBRIDGE),
    ("Cambridge_CUL", _CAMBRIDGE),
    ("New_York_JewishMuseum", "New York, Jewish Museum"),
    ("New_York_Columbia", "New York, Columbia University"),
    ("New_York_JTS", "New York, Jewish Theological Seminary (JTS)"),
    ("Manchester_JRL", "Manchester, John Rylands Library"),
    ("Oxford_Bodleian", "Oxford, Bodleian Library"),
    ("Paris_AIU", "Paris, Alliance Israélite Universelle (AIU)"),
    ("Paris_BNF", "Paris, Bibliothèque nationale de France"),
    ("London_BL", "London, British Library"),
    ("StPetersburg_NLR", "St Petersburg, National Library of Russia"),
    ("StPetersburg_IOM", "St Petersburg, Institute of Oriental Manuscripts"),
    ("Budapest_MTA", "Budapest, Hungarian Academy of Sciences (Kaufmann)"),
    ("Cincinnati_HUC", "Cincinnati, Hebrew Union College"),
    ("Philadelphia_CAJS", "Philadelphia, Penn Katz Center (CAJS)"),
    ("Jerusalem_NLI", "Jerusalem, National Library of Israel"),
    ("Berlin_Community", "Berlin, Jewish Community"),
    ("Berlin_JCB", "Berlin, Jewish Community"),
    ("Berlin_SBB", "Berlin, Staatsbibliothek"),
    ("Vienna_ONB", "Vienna, Austrian National Library"),
    ("Cairo_ENL", "Cairo, Egyptian National Library / MIAC"),
    ("Cairo_JCC", "Cairo, Jewish Community"),
    ("Cairo_Karaite", "Cairo, Karaite Community"),
    ("TelAviv_TAU", "Tel Aviv University"),
    ("Princeton_PUL", "Princeton University Library"),
    ("Washington_Freer", "Washington, Freer Gallery"),
    ("Birmingham_Mingana", "Birmingham, Mingana Collection"),
    ("Geneva", "Geneva, Bibliothèque de Genève"),
    ("Strasbourg", "Strasbourg, BNU"),
    ("Frankfurt", "Frankfurt"),
    ("Heidelberg", "Heidelberg, Papyrology"),
    ("Utah", "Utah"),
    ("Reinach", "Reinach"),
    ("Unknown", "Unidentified holdings"),
]
OTHER_LABEL = "Other holdings"

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
_KTIV_HEAD = re.compile(r"^.*?\bMs\.?\s+", re.IGNORECASE)
_INST_PREFIX = re.compile(r"^[^:]{2,60}:\s*")


@dataclass
class Placement:
    """Where one document sits in the tree.

    :param institution: Top-level label.
    :param series: Sub-collection label.
    :param series_order: Sort key for the sub-collection within its institution.
    :param subseries: Optional third-level key (None → decided by count).
    :param subseries_label: Display name of the third level.
    :param subseries_order: Sort key for the third level.
    :param number: Leading shelf-mark number, used for numeric ranges.
    """

    institution: str
    series: str
    series_order: int = 500
    subseries: Optional[str] = None
    subseries_label: Optional[str] = None
    subseries_order: int = 0
    number: Optional[int] = None


def _num(seg: Optional[str]) -> Optional[int]:
    """Leading integer of a segment, or ``None``.

    :param seg: Id segment such as ``"08J"`` or ``"1922"``.
    :return: Its leading number.
    :rtype: Optional[int]
    """
    m = re.match(r"\d+", seg or "")
    return int(m.group()) if m else None


def _roman_order(token: str) -> int:
    """Sort key for a Roman series token with optional suffix (``"IXa"`` → 9.5).

    :param token: Roman numeral, optionally followed by a lowercase letter.
    :return: Integer sort key (value × 10, +5 for a suffix); 9999 if unparseable.
    :rtype: int
    """
    m = re.fullmatch(r"([IVX]+)([a-z]?)", token or "")
    if not m or m.group(1) not in _ROMAN:
        return 9999
    return _ROMAN[m.group(1)] * 10 + (5 if m.group(2) else 0)


def split_id(doc_id: str) -> Tuple[Optional[str], str, List[str]]:
    """Split a canonical id into its institution token, label and remaining segments.

    :param doc_id: Merged canonical id.
    :return: ``(token or None, institution label, segments after the token)``.
    :rtype: Tuple[Optional[str], str, List[str]]
    """
    for token, label in INSTITUTIONS:
        if doc_id == token or doc_id.startswith(token + "_"):
            rest = doc_id[len(token):].lstrip("_")
            return token, label, [s for s in rest.split("_") if s]
    return None, OTHER_LABEL, [s for s in doc_id.split("_") if s]


def shelfmark_label(shelf_mark: Optional[str], doc_id: str) -> str:
    """Short display form of a shelf mark (institution prefixes and KTIV heads removed).

    :param shelf_mark: Stored ``shelf_mark`` value.
    :param doc_id: Fallback source when the shelf mark is empty.
    :return: Display label.
    :rtype: str
    """
    s = (shelf_mark or "").strip()
    if not s:
        _, _, segs = split_id(doc_id)
        return " ".join(segs) or doc_id
    if re.search(r"\bMs\.?\s", s, re.IGNORECASE):
        s = _KTIV_HEAD.sub("", s)
    elif ":" in s:
        s = _INST_PREFIX.sub("", s)
    return s.strip() or shelf_mark or doc_id


# ---------------------------------------------------------------------------
# Per-institution classifiers
# ---------------------------------------------------------------------------


def _cambridge(token: str, s: List[str]) -> Placement:
    inst = _CAMBRIDGE
    if token == "Cambridge_Lewis_Gibson":
        sub = s[2] if len(s) > 2 and s[0] == "L" and s[1] == "G" else (s[0] if s else "")
        names = {"Ar": "L-G Ar.", "Bib": "L-G Bib.", "Lit": "L-G Lit.", "Misc": "L-G Misc.", "Talm": "L-G Talm.", "Glass": "L-G Glass"}
        label = names.get(sub, f"L-G {sub}".strip())
        return Placement(inst, "Lewis-Gibson collection (L-G)", 60, sub or "other", label, list(names).index(sub) if sub in names else 99)
    if token == "Cambridge_Mosseri":
        series = s[0] if s else "other"
        m = re.fullmatch(r"([IVX]+[a-z]?)(\d+)?", series)
        roman = m.group(1) if m else series
        return Placement(inst, "Mosseri collection (Moss.)", 61, roman, f"Moss. {roman}", _roman_order(roman), _num(s[1] if len(s) > 1 else None))
    # Cambridge_CUL
    if len(s) >= 2 and s[0] == "T" and s[1] == "S":
        t = s[2] if len(s) > 2 else ""
        n = _num(s[3] if len(s) > 3 else None)
        if t == "AS":
            return Placement(inst, "Taylor-Schechter Additional Series (T-S AS)", 10, number=n)
        if t == "NS":
            return Placement(inst, "Taylor-Schechter New Series (T-S NS)", 11, number=n)
        if t == "Ar":
            return Placement(inst, "Taylor-Schechter Arabic (T-S Ar.)", 12, number=n)
        if t == "Misc":
            return Placement(inst, "Taylor-Schechter Miscellaneous (T-S Misc.)", 13, number=n)
        m = re.fullmatch(r"([A-K])(\d+)", t)
        if m:
            letter = m.group(1)
            return Placement(inst, f"Taylor-Schechter old series {letter} (T-S {letter})", 20 + ord(letter) - 65, t, f"T-S {t}", int(m.group(2)))
        m = re.fullmatch(r"0*(\d+)([A-Z]?)(\d*)", t)
        if m:
            box = f"{int(m.group(1))}{m.group(2)}"
            return Placement(inst, "Taylor-Schechter old series: box classes (T-S 6–32)", 40, box, f"T-S {box}", int(m.group(1)) * 10 + (ord(m.group(2)) - 64 if m.group(2) else 0), _num(m.group(3) or (s[3] if len(s) > 3 else None)))
        return Placement(inst, "Taylor-Schechter: other", 45, number=_num(t))
    if s and s[0] == "Or":
        return Placement(inst, "CUL Or. (Oriental manuscripts)", 50, number=_num(s[1] if len(s) > 1 else None))
    if s and s[0] == "Add":
        return Placement(inst, "CUL Add. (Additional manuscripts)", 51, number=_num(s[1] if len(s) > 1 else None))
    return Placement(inst, "Cambridge: other shelf marks", 90, number=_num(s[0] if s else None))


def _jts(inst: str, s: List[str]) -> Placement:
    if s and s[0] == "ENA":
        if len(s) > 1 and s[1] == "NS":
            return Placement(inst, "ENA New Series (ENA NS)", 11, number=_num(s[2] if len(s) > 2 else None))
        return Placement(inst, "Elkan Nathan Adler collection (ENA)", 10, number=_num(s[1] if len(s) > 1 else None))
    if s and s[0] in ("MS", "Lutzki", "Krengel", "Schechter"):
        return Placement(inst, "JTS manuscripts (MS, Lutzki, Krengel, Schechter)", 20, s[0], s[0], ["MS", "Lutzki", "Krengel", "Schechter"].index(s[0]))
    return Placement(inst, "JTS: other shelf marks", 90, number=_num(s[0] if s else None))


def _manchester(inst: str, s: List[str]) -> Placement:
    head = s[0] if s else ""
    if re.fullmatch(r"[A-Z]{1,2}", head):
        return Placement(inst, f"Series {head}", 10 + ord(head[0]), number=_num(s[1] if len(s) > 1 else None))
    if head == "Genizah":
        kind = "Genizah Ar." if len(s) > 1 and s[1] == "Ar" else "Genizah fragments"
        return Placement(inst, kind, 60, number=_num(s[-1]))
    if head in ("Gaster", "Bible"):
        return Placement(inst, "Gaster series", 70, " ".join(s[:3]), " ".join(s[:3]), 0, _num(s[-1]))
    if head == "Glass":
        return Placement(inst, "Glass", 80, number=_num(s[-1]))
    return Placement(inst, "Manchester: other shelf marks", 90, number=_num(s[-1] if s else None))


def _aiu(inst: str, s: List[str]) -> Placement:
    roman = s[0] if s and s[0] in _ROMAN else None
    if roman:
        letter = s[1] if len(s) > 1 and re.fullmatch(r"[A-Z]", s[1]) else None
        return Placement(inst, f"Series {roman}", _roman_order(roman), letter or "_", f"{roman}.{letter}" if letter else f"Series {roman}", ord(letter) if letter else 0, _num(s[2] if letter and len(s) > 2 else (s[1] if len(s) > 1 else None)))
    return Placement(inst, "AIU: other shelf marks", 900, number=_num(s[0] if s else None))


def _bodleian(inst: str, s: List[str]) -> Placement:
    low = [x.lower() for x in s]
    for i, x in enumerate(low):
        if x in ("heb", "arab") and i + 1 < len(s) and re.fullmatch(r"[a-g]", low[i + 1]):
            fam = "MS heb." if x == "heb" else "MS Arab."
            letter = low[i + 1]
            return Placement(inst, f"{fam} {letter}", (0 if x == "heb" else 10) + ord(letter), number=_num(s[i + 2] if i + 2 < len(s) else None))
    return Placement(inst, "Bodleian: other shelf marks", 90, number=_num(s[-1] if s else None))


def _bl(inst: str, s: List[str]) -> Placement:
    if s and s[0].lower() == "or":
        return Placement(inst, "Or. (Oriental manuscripts)", 10, number=_num(s[1] if len(s) > 1 else None))
    if "Add" in s:
        return Placement(inst, "Add. (Additional manuscripts)", 20, number=_num(s[s.index("Add") + 1] if s.index("Add") + 1 < len(s) else None))
    return Placement(inst, "British Library: other shelf marks", 90)


def _nlr(inst: str, s: List[str]) -> Placement:
    up = [x.upper() for x in s]
    if up and up[0] == "RNL":
        up, s = up[1:], s[1:]
    if up and up[0] in ("YEVR", "EVR"):
        if len(up) > 1 and up[1] == "ARAB":
            roman = up[2] if len(up) > 2 else ""
            return Placement(inst, f"Yevr.-Arab. {roman}", 20 + _roman_order(roman), number=_num(s[3] if len(s) > 3 else None))
        if len(up) > 1 and up[1] == "ANTONIN":
            return Placement(inst, "Yevr. Antonin", 60, number=_num(s[-1]))
        roman = up[1] if len(up) > 1 else ""
        letter = up[2] if len(up) > 2 and re.fullmatch(r"[A-K]", up[2]) else ""
        return Placement(inst, f"Yevr. {roman} {letter}".strip(), _roman_order(roman), number=_num(s[3] if letter and len(s) > 3 else (s[2] if len(s) > 2 else None)))
    if "FIRK" in up:
        return Placement(inst, "Firkovich", 70, number=_num(s[-1]))
    return Placement(inst, "NLR: other shelf marks", 90)


def _budapest(inst: str, s: List[str]) -> Placement:
    head = s[0] if s else ""
    if head.isdigit():
        return Placement(inst, "MTA numbering (FJP)", 10, number=int(head))
    if head == "DK":
        return Placement(inst, "DK numbering (Kaufmann, PGP)", 20, number=_num(s[1] if len(s) > 1 else None))
    if head == "Kaufmann":
        return Placement(inst, "Kaufmann GEN / A (KTIV)", 30, number=_num(s[-1]))
    if head == "AS":
        return Placement(inst, "AS", 40, number=_num(s[1] if len(s) > 1 else None))
    return Placement(inst, "Budapest: other shelf marks", 90)


def classify(doc_id: str) -> Placement:
    """Place one document in the tree from its canonical id.

    :param doc_id: Merged canonical id.
    :return: Its :class:`Placement`.
    :rtype: Placement
    """
    token, inst, s = split_id(doc_id)
    if token and token.startswith("Cambridge"):
        return _cambridge(token, s)
    if token == "New_York_JTS":
        return _jts(inst, s)
    if token == "Manchester_JRL":
        return _manchester(inst, s)
    if token == "Paris_AIU":
        return _aiu(inst, s)
    if token == "Oxford_Bodleian":
        return _bodleian(inst, s)
    if token == "London_BL":
        return _bl(inst, s)
    if token == "StPetersburg_NLR":
        return _nlr(inst, s)
    if token == "Budapest_MTA":
        return _budapest(inst, s)
    if token is None:
        head = " ".join(s[:2]) if s else "?"
        return Placement(inst, head, 500, number=_num(s[-1] if s else None))
    return Placement(inst, "All shelf marks", 10, number=_num(next((x for x in s if x[:1].isdigit()), None)))


# ---------------------------------------------------------------------------
# Tree assembly
# ---------------------------------------------------------------------------


def _range_size(max_number: int) -> int:
    """Pick a numeric range width that yields a manageable number of groups.

    :param max_number: Largest leading number in the series.
    :return: 10, 100, 500 or 1000.
    :rtype: int
    """
    for size in (10, 100, 500, 1000):
        if max_number // size < 40:
            return size
    return 1000


def build_tree(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the browser tree from ``(doc_id, shelf_mark)`` rows.

    :param rows: Dicts with ``doc_id`` and ``shelf_mark``.
    :return: ``{institution: {name, count, sub_collections: {series: …}}}`` in
        the shape ``CollectionBrowser.jsx`` renders.
    :rtype: Dict[str, Any]
    """
    # institution -> series -> (subseries key) -> shelfmark name -> [count, label, number]
    tree: Dict[str, Dict[str, Dict[Optional[str], Dict[str, list]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    sub_meta: Dict[Tuple[str, str, str], Tuple[str, int]] = {}
    for row in rows:
        doc_id = row.get("doc_id") or row.get("_id") or ""
        if not doc_id:
            continue
        p = classify(doc_id)
        name = (row.get("shelf_mark") or "").strip() or doc_id
        bucket = tree[p.institution][p.series][p.subseries]
        if name in bucket:
            bucket[name][0] += 1
        else:
            bucket[name] = [1, shelfmark_label(row.get("shelf_mark"), doc_id), p.number]
        meta.setdefault((p.institution, p.series), {"order": p.series_order})
        if p.subseries is not None:
            sub_meta[(p.institution, p.series, p.subseries)] = (p.subseries_label or p.subseries, p.subseries_order)

    out: Dict[str, Any] = {}
    for inst, series_map in tree.items():
        subs: Dict[str, Any] = {}
        inst_count = 0
        for series, by_sub in series_map.items():
            entries: List[Tuple[str, int, str, Optional[int], Optional[str]]] = [
                (name, c, label, number, sub) for sub, marks in by_sub.items() for name, (c, label, number) in marks.items()
            ]
            count = sum(e[1] for e in entries)
            inst_count += count
            explicit = [e for e in entries if e[4] is not None]
            sub_sub: Dict[str, Any] = {}
            if explicit:
                for name, c, label, number, sub in entries:
                    key = sub if sub is not None else "other"
                    label_s, order_s = sub_meta.get((inst, series, key), (key.title(), 9998))
                    node = sub_sub.setdefault(key, {"name": label_s, "range_start": order_s, "range_end": order_s, "count": 0, "shelfmarks": []})
                    node["count"] += c
                    node["shelfmarks"].append({"name": name, "label": label, "count": c, "_n": number})
            elif len(entries) > RANGE_THRESHOLD:
                numbers = [e[3] for e in entries if e[3] is not None]
                size = _range_size(max(numbers) if numbers else 0)
                for name, c, label, number, _ in entries:
                    if number is None:
                        key, disp, start = "other", "Other", 10**9
                    else:
                        start = (number // size) * size
                        key, disp = str(start), f"{start}–{start + size - 1}"
                    node = sub_sub.setdefault(key, {"name": disp, "range_start": start, "range_end": start + size - 1, "count": 0, "shelfmarks": []})
                    node["count"] += c
                    node["shelfmarks"].append({"name": name, "label": label, "count": c, "_n": number})
            entry: Dict[str, Any] = {"name": series, "count": count, "order": meta[(inst, series)]["order"]}
            if sub_sub:
                for node in sub_sub.values():
                    node["shelfmarks"] = _sorted_marks(node["shelfmarks"])
                entry.update({"is_large": True, "shelfmarks": [], "sub_sub_collections": sub_sub})
            else:
                entry.update({"is_large": False, "shelfmarks": _sorted_marks([{"name": n, "label": l, "count": c, "_n": num} for n, c, l, num, _ in entries])})
            subs[series] = entry
        out[inst] = {"name": inst, "count": inst_count, "sub_collections": subs}
    return out


def _sorted_marks(marks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort shelf marks numerically then by label, dropping the helper key.

    :param marks: Shelf-mark dicts carrying a private ``_n`` number.
    :return: Clean, ordered dicts.
    :rtype: List[Dict[str, Any]]
    """
    marks.sort(key=lambda m: (m["_n"] if m["_n"] is not None else 10**9, _natural(m["label"])))
    return [{"name": m["name"], "label": m["label"], "count": m["count"]} for m in marks]


def _natural(text: str) -> List[Any]:
    """Natural-sort key (digits compared numerically).

    :param text: Any label.
    :return: Mixed list of ints and lowercase strings.
    :rtype: List[Any]
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------


def scan_rows(es: Any, index: str, page: int = 5000) -> Iterator[Dict[str, Any]]:
    """Yield ``{doc_id, shelf_mark}`` for every document via ``search_after``.

    :param es: Elasticsearch client.
    :param index: Index to scan.
    :param page: Page size.
    :return: Iterator of rows.
    :rtype: Iterator[Dict[str, Any]]
    """
    after = None
    while True:
        kwargs: Dict[str, Any] = dict(index=index, size=page, _source=["doc_id", "shelf_mark"], sort=[{"doc_id": "asc"}], query={"match_all": {}})
        if after:
            kwargs["search_after"] = after
        hits = es.search(**kwargs)["hits"]["hits"]
        if not hits:
            return
        for h in hits:
            src = h["_source"]
            yield {"doc_id": src.get("doc_id") or h["_id"], "shelf_mark": src.get("shelf_mark")}
        after = hits[-1]["sort"]


_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_lock = threading.Lock()


def _cache_path(index: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", index)
    return os.path.join(DATA_DIR, f"collection_hierarchy_{safe}.json")


def get_hierarchy(es: Any, index: str, refresh: bool = False) -> Dict[str, Any]:
    """Return the cached tree for an index, building it when stale or missing.

    :param es: Elasticsearch client.
    :param index: Index name.
    :param refresh: Ignore caches and rebuild.
    :return: The tree (see :func:`build_tree`).
    :rtype: Dict[str, Any]
    """
    now = time.time()
    if not refresh:
        hit = _cache.get(index)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
        path = _cache_path(index)
        if os.path.exists(path) and now - os.path.getmtime(path) < CACHE_TTL_S:
            with open(path, encoding="utf-8") as fh:
                tree = json.load(fh)
            _cache[index] = (os.path.getmtime(path), tree)
            return tree
    with _lock:
        hit = _cache.get(index)
        if hit and not refresh and now - hit[0] < CACHE_TTL_S:
            return hit[1]
        started = time.time()
        tree = build_tree(scan_rows(es, index))
        logger.info("Built collection hierarchy for %s in %.1fs (%d institutions)", index, time.time() - started, len(tree))
        _cache[index] = (time.time(), tree)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_cache_path(index), "w", encoding="utf-8") as fh:
            json.dump(tree, fh, ensure_ascii=False)
        return tree
