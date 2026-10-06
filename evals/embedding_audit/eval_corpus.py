"""Probe T2: topic retrieval over the real corpus with catalogue-derived labels.

Uses document vectors from ``embed_corpus.py`` and the nuanced queries from
``concept_probe_v1.json`` (which never name the festival/topic) to measure:

A. full-corpus precision of the top-k against silver labels (strict = catalogue
   identification; lenient = identified OR the record's text names the topic);
B. retrieval inside the 5.6k KTIV-framed pool (labels ~complete there): AP, nDCG@10, R@100;
C. Sukkot gold recall (human-verified seed), incl. the records that never name Sukkot;
D. neighbourhood structure: do a labelled record's 10 nearest neighbours share its
   topic, its genre/domain, or merely its language?  (what the space organises by).
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from embed_utils import AUDIT_ROOT, DEFAULT_TASK, DOMAIN_TASK, CachedTextEncoder

HERE = Path(__file__).parent
EVAL_TOPICS = ["sukkot", "pesach", "yom_kippur", "rosh_hashanah", "shabbat", "purim",
               "marriage", "divorce", "kashrut", "niddah", "tefillin_mezuzah"]


def load_vectors(vec_dir: Path) -> tuple:
    """Load ids + stacked shard vectors.

    :param vec_dir: Directory written by embed_corpus.py.
    :returns: (ids, matrix).
    :rtype: tuple
    """
    ids = json.loads((vec_dir / "ids.json").read_text())
    shards = sorted(p for p in vec_dir.glob("shard_*.npy") if ".tmp" not in p.name)
    mat = np.concatenate([np.load(s) for s in shards])
    assert len(mat) == len(ids), f"incomplete vectors {len(mat)} != {len(ids)}"
    return ids, mat


def ndcg_at_k(rel: np.ndarray, k: int, n_pos: int) -> float:
    """Binary nDCG@k.

    :param rel: Binary relevance of the ranked list.
    :param k: Cutoff.
    :param n_pos: Number of positives in the pool.
    :returns: nDCG@k.
    :rtype: float
    """
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((rel[:k] * disc[: len(rel[:k])]).sum())
    idcg = float(disc[: min(k, n_pos)].sum())
    return dcg / idcg if idcg else 0.0


def average_precision(rel: np.ndarray, n_pos: int) -> float:
    """Average precision of a full ranking.

    :param rel: Binary relevance over the full ranked pool.
    :param n_pos: Number of positives.
    :returns: AP.
    :rtype: float
    """
    hits = np.cumsum(rel)
    prec = hits / np.arange(1, len(rel) + 1)
    return float((prec * rel).sum() / n_pos) if n_pos else 0.0


def main() -> None:
    """Run all T2 measurements for one model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--vec-dir", required=True)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--corpus", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--probe", default=str(HERE / "concept_probe_v1.json"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--exclude-ids", default=None, help="JSON list of doc ids to drop (pilot train docs)")
    args = parser.parse_args()

    meta: Dict[str, dict] = {}
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            row.pop("text")
            meta[row["doc_id"]] = row
    ids, mat = load_vectors(Path(args.vec_dir))
    if args.exclude_ids:
        drop = set(json.loads(Path(args.exclude_ids).read_text()))
        keep = [i for i, d in enumerate(ids) if d not in drop]
        ids, mat = [ids[i] for i in keep], mat[keep]
    rows = [meta[i] for i in ids]
    topics = [set(r["topics"]) for r in rows]
    lexical = [set(r["lexical"]) for r in rows]
    framed = np.array([r["framed"] for r in rows])
    probe = json.loads(Path(args.probe).read_text())
    enc = CachedTextEncoder(args.model, max_seq_length=args.max_seq, local_path=args.local_path)

    report: dict = {"model": args.model, "tag": args.tag, "n_docs": len(ids), "n_framed": int(framed.sum()),
                    "ran_at": datetime.now().isoformat(timespec="seconds"), "variants": {}}
    framed_idx = np.where(framed)[0]
    # oracle_expansion = default instruction + the gold topic name appended to the query: an upper
    # bound for a perfect query-side concept expander (terminology map / LLM rewrite) with no training.
    variants = (("default_instruction", DEFAULT_TASK, False), ("domain_instruction", DOMAIN_TASK, False),
                ("oracle_expansion", DEFAULT_TASK, True))
    for vname, task, expand in variants:
        per_topic = {}
        for topic in EVAL_TOPICS:
            queries = probe["concepts"][topic]["queries"]
            if expand:
                queries = [f"{q} ({probe['concepts'][topic]['variants'][0]})" for q in queries]
            qv = enc.encode(queries, mode="query", task=task)
            sims = qv @ mat.T  # (q, N)
            strict = np.array([topic in t for t in topics])
            lenient = strict | np.array([topic in l for l in lexical])
            res = defaultdict(list)
            for qi in range(len(queries)):
                order = np.argsort(-sims[qi])
                res["p10_strict"].append(strict[order[:10]].mean())
                res["p10_lenient"].append(lenient[order[:10]].mean())
                res["p50_lenient"].append(lenient[order[:50]].mean())
                # framed pool (labels ~complete)
                fo = framed_idx[np.argsort(-sims[qi, framed_idx])]
                rel = strict[fo].astype(float)
                npos = int(rel.sum())
                res["framed_ap"].append(average_precision(rel, npos))
                res["framed_ndcg10"].append(ndcg_at_k(rel, 10, npos))
                res["framed_r100"].append(rel[:100].sum() / max(npos, 1))
            per_topic[topic] = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
            per_topic[topic]["n_strict"] = int(strict.sum())
            per_topic[topic]["n_framed_pos"] = int(strict[framed_idx].sum())
            per_topic[topic]["random_p10_framed"] = round(float(strict[framed_idx].mean()), 4)
        agg = {k: round(float(np.mean([pt[k] for pt in per_topic.values()])), 4)
               for k in ("p10_strict", "p10_lenient", "p50_lenient", "framed_ap", "framed_ndcg10", "framed_r100")}
        agg["random_framed_ap"] = round(float(np.mean([pt["random_p10_framed"] for pt in per_topic.values()])), 4)
        report["variants"][vname] = {"macro": agg, "per_topic": per_topic}

        # C. Sukkot gold recall over the full corpus (union of all sukkot queries, max-sim)
        sq = probe["concepts"]["sukkot"]["queries"]
        if expand:
            sq = [f"{q} (Sukkot)" for q in sq]
        qv = enc.encode(sq, mode="query", task=task)
        best = (qv @ mat.T).max(0)
        rank = np.argsort(-best)
        pos = {i: r for r, i in enumerate(rank)}
        gold = [i for i, r in enumerate(rows) if r.get("sukkot_gold")]
        gold_nm = [i for i in gold if rows[i].get("sukkot_gold_no_mention")]
        def rec(idx, k): return round(float(np.mean([pos[i] < k for i in idx])), 4) if idx else None
        report["variants"][vname]["sukkot_gold"] = {
            "n_gold": len(gold), "n_gold_no_mention": len(gold_nm),
            "recall@100": rec(gold, 100), "recall@1000": rec(gold, 1000),
            "no_mention_recall@100": rec(gold_nm, 100), "no_mention_recall@1000": rec(gold_nm, 1000),
            "median_rank_gold": int(np.median([pos[i] for i in gold])) if gold else None,
            "median_rank_no_mention": int(np.median([pos[i] for i in gold_nm])) if gold_nm else None,
        }

    # D. neighbourhood structure for topic-labelled docs (k=10 NN over the full corpus)
    lab = [i for i, t in enumerate(topics) if t & set(EVAL_TOPICS)]
    rng = np.random.default_rng(0)
    sample = rng.choice(lab, size=min(1500, len(lab)), replace=False)
    same = Counter()
    for i in sample:
        s = mat @ mat[i]
        s[i] = -9
        nn = np.argpartition(-s, 10)[:10]
        same["topic"] += np.mean([bool(topics[j] & topics[i]) for j in nn])
        same["domain"] += np.mean([rows[j]["domain"] == rows[i]["domain"] and rows[i]["domain"] is not None for j in nn])
        same["lang"] += np.mean([rows[j]["lang"] == rows[i]["lang"] and rows[i]["lang"] is not None for j in nn])
        same["framed"] += np.mean([rows[j]["framed"] for j in nn])
    report["neighbourhood_k10"] = {k: round(v / len(sample), 4) for k, v in same.items()}
    report["neighbourhood_k10"]["n_sampled"] = len(sample)
    tag = f"_{args.tag}" if args.tag else ""
    out = HERE / "results" / f"t2_corpus_{args.model}{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for vname, v in report["variants"].items():
        print(args.model, vname, json.dumps(v["macro"]), json.dumps(v["sukkot_gold"]), flush=True)
    print("neighbourhood", report["neighbourhood_k10"], flush=True)


if __name__ == "__main__":
    main()
