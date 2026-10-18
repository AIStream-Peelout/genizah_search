"""Probe T3: known-item retrieval for LLM-written queries on held-out (eval-pool) records.

For every eval-pool record with synthetic queries, each query should retrieve its own record from the full
73.5k corpus. Reports, per query type (implicit/topical/hebrew/specific): MRR, R@1/10/100, median rank. This is
the pre-validation baseline; once Isaac's verdicts exist (``--reviews``), only queries he kept are scored.
BM25 on the same queries is included as the keyword reference.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from embed_utils import AUDIT_ROOT, DEFAULT_TASK, CachedTextEncoder
from eval_bm25 import bm25_matrix
from eval_corpus import load_vectors

HERE = Path(__file__).parent


def stats(ranks: list) -> dict:
    """Rank summary.

    :param ranks: 1-based ranks.
    :returns: Metrics dict.
    :rtype: dict
    """
    r = np.array(ranks)
    return {"n": int(len(r)), "MRR": round(float((1 / r).mean()), 4), "R@1": round(float((r <= 1).mean()), 4),
            "R@10": round(float((r <= 10).mean()), 4), "R@100": round(float((r <= 100).mean()), 4),
            "median_rank": int(np.median(r))}


def main() -> None:
    """Score dense + BM25 on eval-pool synthetic queries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--vec-dir", default=str(AUDIT_ROOT / "corpus_vectors" / "qwen3-0.6b__all__8192"))
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--round", default="r1")
    parser.add_argument("--reviews", default=None, help="JSON list of review docs exported from the review UI")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    base = AUDIT_ROOT / "synthetic_queries" / args.round
    pool = {}
    for f in (base / "batches").glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            b = json.loads(line)
            pool[b["doc_id"]] = b["pool"]
    kept = None
    if args.reviews:
        kept = {}
        for rv in json.loads(Path(args.reviews).read_text()):
            for q in rv.get("queries", []):
                if q.get("verdict") in ("good", "fixed"):
                    kept[(rv["doc_id"], q["qid"].split(":")[-1])] = q["text"]
    queries = []  # (doc_id, type, text)
    for f in list((base / "out").glob("*.jsonl")) + list((base / "out_he").glob("*.jsonl")):
        he_round = f.parent.name == "out_he"
        for line in open(f, encoding="utf-8"):
            o = json.loads(line)
            if pool.get(o["doc_id"]) != "eval":
                continue
            for i, q in enumerate(o.get("queries") or []):
                i = f"he{i}" if he_round else i
                text = q["text"]
                if kept is not None:
                    if (o["doc_id"], str(i)) not in kept:
                        continue
                    text = kept[(o["doc_id"], str(i))]
                queries.append((o["doc_id"], q["type"], text))
    ids, mat = load_vectors(Path(args.vec_dir))
    pos = {d: i for i, d in enumerate(ids)}
    queries = [q for q in queries if q[0] in pos]
    enc = CachedTextEncoder(args.model, max_seq_length=args.max_seq, local_path=args.local_path)
    qv = enc.encode([q[2] for q in queries], mode="query", task=DEFAULT_TASK)
    dense = defaultdict(list)
    for (doc, typ, _), v in zip(queries, qv):
        s = mat @ v
        dense[typ].append(int((s > s[pos[doc]]).sum()) + 1)
    texts = {}
    for line in open(AUDIT_ROOT / "corpus_v1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        texts[r["doc_id"]] = r["text"]
    order = [d for d in ids]
    vec, W = bm25_matrix([texts[d] for d in order])
    Q = vec.transform([q[2] for q in queries])
    Q.data[:] = 1.0
    S = (Q @ W.T).toarray()
    lex = defaultdict(list)
    for qi, (doc, typ, _) in enumerate(queries):
        s = S[qi]
        lex[typ].append(int((s > s[pos[doc]]).sum()) + 1)
    out = {"model": args.model, "tag": args.tag, "n_queries": len(queries), "reviewed_only": kept is not None,
           "dense": {t: stats(r) for t, r in sorted(dense.items())}, "bm25": {t: stats(r) for t, r in sorted(lex.items())},
           "ran_at": datetime.now().isoformat(timespec="seconds")}
    name = f"t3_synthetic_{args.model}{('_' + args.tag) if args.tag else ''}.json"
    (HERE / "results" / name).write_text(json.dumps(out, indent=1))
    for t in sorted(dense):
        d, b = out["dense"][t], out["bm25"][t]
        print(f"{t:9s} n={d['n']:4d} dense MRR={d['MRR']:.3f} R@10={d['R@10']:.3f} R@100={d['R@100']:.3f} med={d['median_rank']:6d} | "
              f"BM25 MRR={b['MRR']:.3f} R@10={b['R@10']:.3f} R@100={b['R@100']:.3f} med={b['median_rank']}", flush=True)


if __name__ == "__main__":
    main()
