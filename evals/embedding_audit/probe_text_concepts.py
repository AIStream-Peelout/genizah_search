"""Probe T1: do nuanced Jewish queries land on the right topic without naming it?

For each model x instruction variant, encodes the concept anchors (document
mode) and the queries/variants (query mode) from ``concept_probe_v1.json`` and
scores nearest-centroid topic assignment: top-1 accuracy, MRR, and the margin
between the correct topic and the best wrong one. Results are written as JSON
next to this script under ``results/``.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from embed_utils import DEFAULT_TASK, DOMAIN_TASK, CachedTextEncoder

HERE = Path(__file__).parent
HEBREW = re.compile(r"[֐-׿]")


def score(qvecs: np.ndarray, gold: List[int], centroids: np.ndarray) -> Dict[str, float]:
    """Score nearest-centroid assignment.

    :param qvecs: Query vectors (n, d), normalized.
    :param gold: Gold concept index per query.
    :param centroids: Concept centroids (k, d), normalized.
    :returns: Dict with top1, top3, mrr, mean_margin.
    :rtype: Dict[str, float]
    """
    sims = qvecs @ centroids.T
    order = np.argsort(-sims, axis=1)
    ranks = np.array([int(np.where(order[i] == g)[0][0]) + 1 for i, g in enumerate(gold)])
    margins = []
    for i, g in enumerate(gold):
        wrong = np.delete(sims[i], g)
        margins.append(float(sims[i, g] - wrong.max()))
    return {
        "n": len(gold),
        "top1": float(np.mean(ranks == 1)),
        "top3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_margin": float(np.mean(margins)),
    }


def run(model_key: str, tasks: Dict[str, Optional[str]], probe: dict, local_path: Optional[str] = None) -> dict:
    """Run the concept probe for one model across instruction variants.

    :param model_key: Registered model key (or a new key when ``local_path`` is given).
    :param local_path: Optional local checkpoint directory (fine-tuned model).
    :param tasks: Mapping of variant name -> instruction (None = no instruction).
    :param probe: Parsed probe JSON.
    :returns: Results dict.
    :rtype: dict
    """
    enc = CachedTextEncoder(model_key, local_path=local_path)
    names = list(probe["concepts"])
    anchors, anchor_gold = [], []
    for ci, name in enumerate(names):
        for a in probe["concepts"][name]["anchors"]:
            anchors.append(a)
            anchor_gold.append(ci)
    avecs = enc.encode(anchors, mode="document")
    centroids = np.stack([avecs[np.array(anchor_gold) == ci].mean(0) for ci in range(len(names))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    out = {"model": model_key, "concepts": names, "variants": {}}
    for tname, task in tasks.items():
        res = {}
        for field in ("queries", "variants"):
            texts, gold = [], []
            for ci, name in enumerate(names):
                for q in probe["concepts"][name][field]:
                    texts.append(q)
                    gold.append(ci)
            qvecs = enc.encode(texts, mode="query", task=task)
            heb = [bool(HEBREW.search(t)) for t in texts]
            res[field] = score(qvecs, gold, centroids)
            res[field + "_hebrew"] = score(qvecs[np.array(heb)], [g for g, h in zip(gold, heb) if h], centroids)
            res[field + "_latin"] = score(qvecs[~np.array(heb)], [g for g, h in zip(gold, heb) if not h], centroids)
            # per-query detail for the report (rank of gold + top wrong concept)
            sims = qvecs @ centroids.T
            detail = []
            for i, (t, g) in enumerate(zip(texts, gold)):
                order = list(np.argsort(-sims[i]))
                detail.append({"q": t, "gold": names[g], "rank": order.index(g) + 1,
                               "top": names[order[0]], "sim_gold": round(float(sims[i, g]), 4),
                               "sim_top": round(float(sims[i, order[0]]), 4)})
            res[field + "_detail"] = detail
        out["variants"][tname] = res
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="qwen3-0.6b")
    parser.add_argument("--probe", default=str(HERE / "concept_probe_v1.json"))
    parser.add_argument("--local-path", default=None, help="fine-tuned checkpoint dir (use with a single --models key)")
    args = parser.parse_args()
    probe = json.loads(Path(args.probe).read_text())
    tasks = {"default_instruction": DEFAULT_TASK, "domain_instruction": DOMAIN_TASK, "no_instruction": None}
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    for key in args.models.split(","):
        result = run(key, tasks, probe, args.local_path)
        result["ran_at"] = datetime.now().isoformat(timespec="seconds")
        (results_dir / f"t1_concepts_{key}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        for tname, res in result["variants"].items():
            q, v = res["queries"], res["variants"]
            print(f"{key:18s} {tname:20s} queries top1={q['top1']:.2f} mrr={q['mrr']:.2f} margin={q['mean_margin']:+.3f} "
                  f"(he {res['queries_hebrew']['top1']:.2f} / lat {res['queries_latin']['top1']:.2f}) | "
                  f"variants top1={v['top1']:.2f}", flush=True)


if __name__ == "__main__":
    main()
