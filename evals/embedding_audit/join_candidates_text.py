"""Text-adjacency join candidates for identified Talmud Bavli fragments (no image model needed).

Two leaves torn from one codex carry *consecutive* text. KTIV frames give each identified
Bavli fragment an amud range (``[Talmud Bavli]: Yevamot 74 a – 75 b``). Pairs of fragments of
the same tractate whose ranges abut (gap <= ``--max-gap`` amudim, no overlap) are join
candidates; physical metadata (material, lines per page, script style/region, page size) then
filters them. Known joins (FJP clean lists + shared PGP documents) measure how much of the
known join set this cheap heuristic recovers.
"""

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple

from embed_utils import AUDIT_ROOT

MERGED = Path.home() / "Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
FRAME = re.compile(r"\[Talmud Bavli\]:\s*([A-Za-z' \-]+?)\s+(\d+)\s*([ab])\s*(?:[–-]\s*(?:(\d+)\s*)?([ab]))?")


def amud(folio: int, side: str) -> int:
    """Linear amud index (2a -> 4, 2b -> 5).

    :param folio: Folio number.
    :param side: ``a`` or ``b``.
    :returns: Index.
    :rtype: int
    """
    return folio * 2 + (side == "b")


def parse(frame: str) -> Optional[Tuple[str, int, int]]:
    """Parse a Bavli frame into (tractate, start amud, end amud).

    :param frame: Frame string.
    :returns: Tuple or None.
    """
    m = FRAME.match(frame)
    if not m:
        return None
    tractate, f0, s0, f1, s1 = m.groups()
    start = amud(int(f0), s0)
    end = amud(int(f1) if f1 else int(f0), s1) if s1 else start
    return tractate.strip(), start, max(start, end)


def first(v):
    """First element of a scalar-or-list value.

    :param v: Value.
    :returns: First item or the value.
    """
    return v[0] if isinstance(v, list) and v else v


def main() -> None:
    """Find and score candidates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-gap", type=int, default=1, help="amudim allowed between ranges")
    args = parser.parse_args()
    frags = {}
    with open(MERGED, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ktiv = (rec.get("sources") or {}).get("ktiv") or {}
            spans, meta = set(), {}
            for e in ktiv.get("scholarly_entries") or []:
                sub = e.get("subsections") or {}
                wc = sub.get("writing_characteristics") or {}
                for fr in (wc.get("frame") if isinstance(wc.get("frame"), list) else [wc.get("frame")]):
                    p = parse(fr) if isinstance(fr, str) else None
                    if p:
                        spans.add(p)
                w2 = sub.get("wrinting_characteristics") or {}
                pd = sub.get("physical_description") or {}
                for k, v in (("style", w2.get("script_style")), ("region", w2.get("region_of_script")),
                             ("material", pd.get("material_type")), ("rows", pd.get("no_of_rows")),
                             ("dims", pd.get("page_dimensions"))):
                    if first(v) and k not in meta:
                        meta[k] = str(first(v))
            if spans:
                frags[rec["canonical_id"]] = {"spans": sorted(spans), "meta": meta, "coll": rec.get("collection")}
    by_tractate = defaultdict(list)
    for cid, f in frags.items():
        for t, s, e in f["spans"]:
            by_tractate[t].append((s, e, cid))
    cands = {}
    for t, items in by_tractate.items():
        items.sort()
        for (s1, e1, a), (s2, e2, b) in combinations(items, 2):
            if a == b:
                continue
            gap = s2 - e1 - 1 if s2 > e1 else s1 - e2 - 1
            if 0 <= gap <= args.max_gap:
                ma, mb = frags[a]["meta"], frags[b]["meta"]
                conflict = [k for k in ("material", "style", "region") if ma.get(k) and mb.get(k) and ma[k] != mb[k]]
                ra, rb = ma.get("rows", ""), mb.get("rows", "")
                rows_ok = None
                try:
                    rows_ok = abs(int(re.findall(r"\d+", ra)[0]) - int(re.findall(r"\d+", rb)[0])) <= 2
                except (IndexError, ValueError):
                    pass
                key = tuple(sorted((a, b)))
                cands[key] = {"tractate": t, "gap": gap, "conflicts": conflict, "rows_match": rows_ok}
    manifest = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())
    known = {tuple(sorted(p)) for p in manifest["joins_fjp_clean"] + manifest["joins_pgp"]}
    known_bavli = {k for k in known if k[0] in frags and k[1] in frags}
    no_conflict = {k: v for k, v in cands.items() if not v["conflicts"]}
    strong = {k: v for k, v in no_conflict.items() if v["rows_match"]}
    out = {
        "bavli_fragments_with_parsed_ranges": len(frags),
        "candidate_pairs_adjacent": len(cands),
        "candidates_no_metadata_conflict": len(no_conflict),
        "candidates_no_conflict_and_rows_match": len(strong),
        "known_joins_among_bavli_fragments": len(known_bavli),
        "known_recovered_by_adjacency": len(known_bavli & set(cands)),
        "known_in_strong": len(known_bavli & set(strong)),
        "novel_strong_candidates": len(set(strong) - known),
    }
    (AUDIT_ROOT / "join_candidates_text_bavli.json").write_text(json.dumps(
        {"summary": out, "strong": [{"a": a, "b": b, **v, "known": (a, b) in known} for (a, b), v in sorted(strong.items())]},
        ensure_ascii=False, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
