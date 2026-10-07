"""Assemble the review UI's ``items.json`` from synthetic-query outputs.

Joins each sub-agent output line (``out/*.jsonl``) with its input record (``batches/*.jsonl``), splits the
record's embedded text into labelled fields for display, adds a link to the source catalogue page (PGP or
KTIV) from the merged record, and gives every record a db-safe key ``k`` and every query a stable ``qid``.
By default every eval-pool record plus ``--train-sample`` random training records are included.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from embed_utils import AUDIT_ROOT

MERGED = Path.home() / "Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
HIDE = {"Document ID"}


def fields_of(text: str) -> dict:
    """Split ``Label: value`` lines of the embedded text into an ordered dict.

    :param text: Record text representation.
    :returns: Label -> value (continuation lines appended).
    :rtype: dict
    """
    out, last = {}, None
    for line in text.split("\n"):
        m = re.match(r"^([A-Z][A-Za-z ]{2,30}):\s?(.*)$", line)
        if m:
            last = m.group(1)
            out[last] = (out.get(last, "") + " " + m.group(2)).strip()
        elif last:
            out[last] += "\n" + line
    return {k: v for k, v in out.items() if k not in HIDE and v}


def source_urls(doc_ids: set) -> dict:
    """Source catalogue page per record (PGP document page, else KTIV item page).

    :param doc_ids: Records needed.
    :returns: doc_id -> URL.
    :rtype: dict
    """
    urls = {}
    with open(MERGED, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid = rec["canonical_id"]
            if cid not in doc_ids:
                continue
            src = rec.get("sources") or {}
            pgp_docs = (src.get("pgp") or {}).get("documents") or []
            ktiv = src.get("ktiv") or {}
            url = (pgp_docs[0].get("url") if pgp_docs else None) or ktiv.get("page_url")
            if url:
                urls[cid] = url
    return urls


def main() -> None:
    """Write items.json next to the review UI page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", default="r1")
    parser.add_argument("--train-sample", type=int, default=300)
    parser.add_argument("--out", default=str(Path(__file__).parent / "review_ui" / "items.json"))
    args = parser.parse_args()
    base = AUDIT_ROOT / "synthetic_queries" / args.round
    inputs = {}
    for f in sorted((base / "batches").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            inputs[r["doc_id"]] = r
    outputs, he_out = {}, {}
    for f in sorted((base / "out").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                if o.get("queries"):
                    outputs[o["doc_id"]] = o
    for f in sorted((base / "out_he").glob("*.jsonl")):  # Hebrew-implicit / Hebrew-topical / transliterated round
        for line in open(f, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                if o.get("queries"):
                    he_out[o["doc_id"]] = o["queries"]
    rows = [inputs[d] | {"queries": o["queries"], "queries_he": he_out.get(d, [])} for d, o in outputs.items() if d in inputs]
    rng = random.Random(7)
    evals = [r for r in rows if r["pool"] == "eval"]
    train = [r for r in rows if r["pool"] == "train"]
    rng.shuffle(train)
    chosen = evals + train[: args.train_sample]
    urls = source_urls({r["doc_id"] for r in chosen})
    items = []
    for r in chosen:
        k = hashlib.sha1(r["doc_id"].encode()).hexdigest()[:20]
        f = fields_of(r["text"])
        items.append({
            "k": k, "doc_id": r["doc_id"], "pool": r["pool"], "stratum": r["stratum"], "topics": r.get("topics", []),
            "round": args.round, "shelfmark": f.pop("Shelf Mark", ""), "fields": f, "source_url": urls.get(r["doc_id"]),
            "queries": [{"qid": f"{k}:{i}", **q} for i, q in enumerate(r["queries"])]
                       + [{"qid": f"{k}:he{i}", **q} for i, q in enumerate(r.get("queries_he", []))],
        })
    # eval candidates first (Sukkot seed interleaved), then the training sample
    items.sort(key=lambda it: (it["pool"] != "eval", rng.random()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False))
    print(json.dumps({"items": len(items), "eval": len(evals), "train_in_ui": min(len(train), args.train_sample),
                      "queries": sum(len(i["queries"]) for i in items), "with_source_url": len(urls)}))


if __name__ == "__main__":
    main()
