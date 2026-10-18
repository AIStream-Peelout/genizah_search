"""Primary↔secondary training pairs: scholarship pages ↔ the fragments they discuss.

Reads (read-only) every page of the bibliography index that lists ``shelf_marks_mentioned`` and resolves each
mention to a merged-corpus record with the sibling repo's ``ShelfmarkNormalizer.loose_key``. Writes
``scholarship_pairs.jsonl``: {page_id, book_title, page_text, doc_id, mention}. Each pair can train both
directions (page passage → record; record → page), which is the product's primary/secondary bridge.
"""

import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
sys.path.insert(0, str(HDA))
from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402

from embed_utils import AUDIT_ROOT  # noqa: E402
from es_read import scan  # noqa: E402

BIB_INDEX = "bibliography_text_only_0.7"
MERGED = HDA / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"


def record_keys() -> dict:
    """loose_key -> canonical ids, from each merged record's display shelfmark and known variants.

    :returns: Mapping.
    :rtype: dict
    """
    keys = defaultdict(set)
    for line in open(MERGED, encoding="utf-8"):
        rec = json.loads(line)
        cid = rec["canonical_id"]
        for sm in {rec.get("shelfmark_display") or "", *(f.get("shelf_mark") or "" for f in (rec.get("sources") or {}).get("fjp") or [])}:
            if sm:
                keys[ShelfmarkNormalizer.loose_key(sm)].add(cid)
    return keys


def main() -> None:
    """Resolve mentions and write pairs."""
    keys = record_keys()
    stats = Counter()
    out = open(AUDIT_ROOT / "scholarship_pairs.jsonl", "w", encoding="utf-8")
    for hit in scan(BIB_INDEX, {"exists": {"field": "shelf_marks_mentioned"}},
                    ["doc_id", "title", "full_text_content", "shelf_marks_mentioned", "page_number"]):
        s = hit["_source"]
        raw = s.get("shelf_marks_mentioned") or []
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                raw = [raw]
        stats["pages"] += 1
        for m in raw:
            stats["mentions"] += 1
            cids = keys.get(ShelfmarkNormalizer.loose_key(str(m)), set())
            if len(cids) == 1:
                stats["resolved"] += 1
                out.write(json.dumps({"page_id": s.get("doc_id") or hit["_id"], "book_title": s.get("title"),
                                      "page_text": (s.get("full_text_content") or "")[:4000], "doc_id": next(iter(cids)),
                                      "mention": m}, ensure_ascii=False) + "\n")
            elif cids:
                stats["ambiguous"] += 1
    out.close()
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
