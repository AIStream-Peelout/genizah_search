"""Build the image-similarity ground truth: joins, scribes, script labels -> one image per fragment.

Ground truth (all keyed by merged ``canonical_id``):

* ``joins_fjp`` — FJP/FGP ``joins_data.joinedManuscripts`` (literary joins), shelfmarks
  resolved through the FJP shelfmarks carried on merged records.
* ``joins_pgp`` — fragments that share a PGP document id (documentary joins: one
  document spread over several shelfmarks).
* ``scribe`` — PGP person->document ``Scribe`` relations (certain only).
* ``script_style`` / ``script_region`` / ``material`` — KTIV palaeographic fields.

Each fragment gets its first preferred image URL (GCS / KTIV / Bodleian) via the
indexer's own resolver. Writes ``image_manifest_v1.json`` to the audit root.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
sys.path.insert(0, str(HDA))
from src.datasets.document_models.genizah_document import _merged_image_urls  # noqa: E402

from embed_utils import AUDIT_ROOT  # noqa: E402

MERGED = HDA / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
FJP_ALL = HDA / "src/datasets/raw_data/cairo_genizah/fjp_all/merged_princeton_friedberger_all_documents_final.json"
PGP_RELATIONS = Path.home() / "Documents/GitHub/ktiv-scraper/princeton/pgp_person_document_relations.csv"


def shelf_key(shelfmark: str) -> str:
    """Normalise an FJP-style shelfmark for matching.

    :param shelfmark: e.g. ``"Cambridge, CUL:  T-S F1(2).11"``.
    :returns: e.g. ``"cambridge_cul_t_s_f1_2_11"``.
    :rtype: str
    """
    s = re.sub(r"\(alt[^)]*\)", "", shelfmark, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def first(value) -> Optional[str]:
    """Return the first string of a scalar-or-list field.

    :param value: Field value.
    :returns: First string or None.
    :rtype: Optional[str]
    """
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, str) and value.strip() else None


def main() -> None:
    """Assemble and write the manifest."""
    images: Dict[str, str] = {}
    n_images: Dict[str, int] = {}
    shelf_to_cid: Dict[str, str] = {}
    pgp_to_cids: Dict[str, Set[str]] = defaultdict(set)
    collection: Dict[str, str] = {}
    attrs: Dict[str, dict] = defaultdict(dict)
    with open(MERGED, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid = rec["canonical_id"]
            urls = _merged_image_urls(rec.get("images") or {})
            urls = [u for u in urls if u and not u.endswith("/")]
            if urls:
                images[cid] = urls[0]
                n_images[cid] = len(urls)
            collection[cid] = f"{rec.get('institution')}|{rec.get('collection')}"
            for fjp in (rec.get("sources") or {}).get("fjp") or []:
                if fjp.get("shelf_mark"):
                    shelf_to_cid.setdefault(shelf_key(fjp["shelf_mark"]), cid)
            for pid in rec.get("pgpids") or []:
                pgp_to_cids[str(pid)].add(cid)
            ktiv = (rec.get("sources") or {}).get("ktiv") or {}
            for entry in ktiv.get("scholarly_entries") or []:
                sub = entry.get("subsections") or {}
                wc = sub.get("wrinting_characteristics") or {}
                for field, key in (("script_style", "script_style"), ("region_of_script", "script_region"),
                                   ("script_type", "script_type")):
                    val = first(wc.get(field))
                    if val and key not in attrs[cid]:
                        attrs[cid][key] = val
                mat = first((sub.get("physical_description") or {}).get("material_type"))
                if mat and "material" not in attrs[cid]:
                    attrs[cid]["material"] = mat

    # FJP joins. The FJP scrape has a prefix bug: a shelfmark that is a string prefix of
    # another (e.g. "Or.1080 4.5" vs "Or.1080 4.55") inherits the longer one's join list.
    # Flag those items (identical list to a longer-prefixed sibling that does not list them).
    fjp = json.loads(FJP_ALL.read_text())
    items = list(fjp.values() if isinstance(fjp, dict) else fjp)
    lists = {}
    for item in items:
        joined = ((item.get("joins_data") or {}).get("joinedManuscripts")) or []
        if joined:
            lists[shelf_key(item.get("shelf_mark") or "")] = frozenset(shelf_key(j.get("shelfmark") or "") for j in joined)
    by_list = defaultdict(list)
    for key, js in lists.items():
        by_list[js].append(key)
    suspects = {x for group in by_list.values() if len(group) > 1 for x in group for y in group
                if x != y and y.startswith(x) and x not in lists.get(y, ())}
    fjp_pairs: Set[tuple] = set()
    fjp_clean: Set[tuple] = set()
    unresolved = Counter()
    for item in items:
        joined = ((item.get("joins_data") or {}).get("joinedManuscripts")) or []
        if not joined:
            continue
        suspect = shelf_key(item.get("shelf_mark") or "") in suspects
        group = [item.get("shelf_mark") or ""] + [j.get("shelfmark") or "" for j in joined]
        cids = []
        for sm in group:
            cid = shelf_to_cid.get(shelf_key(sm))
            if cid:
                cids.append(cid)
            else:
                unresolved["fjp"] += 1
        cids = sorted(set(cids))
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                fjp_pairs.add((cids[i], cids[j]))
                if not suspect:
                    fjp_clean.add((cids[i], cids[j]))
    # PGP joins: fragments sharing a PGP document id
    pgp_pairs: Set[tuple] = set()
    for cids in pgp_to_cids.values():
        c = sorted(cids)
        if 2 <= len(c) <= 6:  # very large multi-fragment groups are composite dossiers, not physical joins
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    pgp_pairs.add((c[i], c[j]))
    # scribes
    scribe: Dict[str, str] = {}
    with open(PGP_RELATIONS, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["relation"].strip() != "Scribe":
                continue
            for cid in pgp_to_cids.get(row["pgpid"].strip(), ()):
                scribe.setdefault(cid, row["person_slug"])

    def with_img(pairs):
        return sorted(p for p in pairs if p[0] in images and p[1] in images)

    manifest = {
        "joins_fjp": with_img(fjp_pairs),
        "joins_fjp_clean": with_img(fjp_clean),
        "joins_pgp": with_img(pgp_pairs),
        "scribe": {c: s for c, s in scribe.items() if c in images},
        "attrs": {c: a for c, a in attrs.items() if c in images and a},
        "images": images,
        "n_images": n_images,
        "collection": collection,
    }
    stats = {
        "fragments_with_image": len(images),
        "fjp_pairs_total": len(fjp_pairs), "fjp_pairs_with_images": len(manifest["joins_fjp"]),
        "fjp_unresolved_shelfmarks": unresolved["fjp"],
        "fjp_items_with_join_list": len(lists), "fjp_prefix_bug_suspects": len(suspects),
        "fjp_clean_pairs_with_images": len(manifest["joins_fjp_clean"]),
        "pgp_pairs_total": len(pgp_pairs), "pgp_pairs_with_images": len(manifest["joins_pgp"]),
        "scribe_fragments_with_image": len(manifest["scribe"]),
        "scribes_ge5": sum(1 for s, n in Counter(manifest["scribe"].values()).items() if n >= 5),
        "attr_counts": {k: Counter(a.get(k) for a in manifest["attrs"].values() if a.get(k)).most_common(8)
                        for k in ("script_style", "script_region", "script_type", "material")},
        "image_hosts": Counter(re.sub(r"^https?://([^/]+)/.*$", r"\1", u) for u in images.values()).most_common(6),
    }
    manifest["stats"] = stats
    (AUDIT_ROOT / "image_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
