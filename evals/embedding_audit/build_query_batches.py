"""Select documents for LLM-written synthetic queries and write them as batch files for sub-agents.

Round design (``--round r1``): a stratified sample so the generated queries cover the corpus's genres,
languages and the 11 audit topics, split by the same stable doc-id hash as the pilot (20 % test side):

* train pool — framed literary records (stratified by work), PGP documentary records (by type),
  unframed Hebrew-catalogued records, and records with transcriptions;
* eval pool — the 125 human-verified Sukkot seed records + a stratified sample of test-side records.
  Queries for eval-pool records are candidates for the scholar-validated evaluation set and are never
  used for training.

Each batch file is JSONL of ``{doc_id, pool, stratum, shelfmark, text}`` where ``text`` is the record's
production embedding text (truncated) — exactly what the retriever sees.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from embed_utils import AUDIT_ROOT

HEB = re.compile(r"[֐-׿]")


def is_test(doc_id: str) -> bool:
    """Stable 20 % test split (same rule as the LoRA pilot).

    :param doc_id: Record id.
    :returns: True for the test side.
    :rtype: bool
    """
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 5 == 0


def work_of(row: dict) -> str:
    """Coarse work/genre of a framed record from its first frame.

    :param row: Corpus row.
    :returns: Work label.
    :rtype: str
    """
    for f in row.get("frames") or []:
        m = re.match(r"\[([^\]]+)\]", f)
        w = m.group(1) if m else ("Piyyut" if re.search(r"\(|''", f) else "other")
        if "Mishneh Torah" in w:
            return "Mishneh Torah"
        if "Hilkhot ha-Rif" in w:
            return "Rif"
        if w in ("Talmud Bavli", "Mishnah", "Bible"):
            return w
        if "Piyyut" in w or "פיוט" in w or "Hoshanot" in f:
            return "Piyyut"
        return "other literary"
    return "unframed"


def stratum(row: dict) -> str:
    """Assign a sampling stratum.

    :param row: Corpus row.
    :returns: Stratum name.
    :rtype: str
    """
    if row["framed"]:
        return "framed:" + work_of(row)
    if row.get("pgp_type"):
        return "pgp:" + row["pgp_type"]
    if "Transcription:" in row["text"]:
        return "transcribed"
    if HEB.search(row.get("description") or ""):
        return "hebrew-catalogue"
    return "other"


QUOTA_TRAIN = {
    "framed:Talmud Bavli": 100, "framed:Mishneh Torah": 80, "framed:Bible": 60, "framed:Piyyut": 80,
    "framed:Mishnah": 40, "framed:Rif": 20, "framed:other literary": 30,
    "pgp:Letter": 150, "pgp:Legal document": 120, "pgp:List or table": 60, "pgp:Literary text": 40,
    "pgp:State document": 30, "pgp:Paraliterary text": 20, "pgp:Legal query or responsum": 20,
    "pgp:Credit instrument or private receipt": 20,
    "transcribed": 150, "hebrew-catalogue": 150, "other": 70,
}
QUOTA_EVAL = {k: max(5, v // 7) for k, v in QUOTA_TRAIN.items()}


def main() -> None:
    """Sample, then write batch files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", default="r1")
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--train-only", action="store_true", help="later rounds: no eval pool, skip records used before")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the training quotas")
    args = parser.parse_args()
    used = set()
    if args.train_only:
        for prev in (AUDIT_ROOT / "synthetic_queries").glob("r*/batches/*.jsonl"):
            used.update(json.loads(l)["doc_id"] for l in open(prev, encoding="utf-8"))
    rng = random.Random(2026)
    rows = [json.loads(l) for l in open(AUDIT_ROOT / "corpus_v1.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    seed = []
    for r in rows:
        if r.get("sukkot_gold"):
            if not args.train_only:
                seed.append(r)
            continue
        if len(r["text"]) < 120 or r["doc_id"] in used:  # nothing to write about / already queried
            continue
        by[("eval" if is_test(r["doc_id"]) else "train", stratum(r))].append(r)
    picked = [("eval", "sukkot-seed", r) for r in seed]
    plan = [("train", {k: int(v * args.scale) for k, v in QUOTA_TRAIN.items()})]
    if not args.train_only:
        plan.append(("eval", QUOTA_EVAL))
    for pool, quota in plan:
        for st, n in quota.items():
            cands = by.get((pool, st), [])
            # prefer topic-labelled framed records so all audit topics are covered
            cands.sort(key=lambda r: (-len(r["topics"]), rng.random()))
            picked += [(pool, st, r) for r in cands[:n]]
    rng.shuffle(picked)
    out_dir = AUDIT_ROOT / "synthetic_queries" / args.round / "batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(picked), args.batch_size):
        with open(out_dir / f"batch_{b // args.batch_size:03d}.jsonl", "w", encoding="utf-8") as fh:
            for pool, st, r in picked[b:b + args.batch_size]:
                fh.write(json.dumps({"doc_id": r["doc_id"], "pool": pool, "stratum": st,
                                     "topics": r["topics"], "text": r["text"][: args.max_chars]},
                                    ensure_ascii=False) + "\n")
    print(json.dumps({"docs": len(picked), "batches": (len(picked) + args.batch_size - 1) // args.batch_size,
                      "by_pool": Counter(p for p, _, _ in picked),
                      "by_stratum": Counter(s for _, s, _ in picked)}, indent=1))


if __name__ == "__main__":
    main()
