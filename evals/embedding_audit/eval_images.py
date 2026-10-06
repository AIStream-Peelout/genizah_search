"""Score image features on joins, scribes, script attributes, and the imaging confound.

For each ``<encoder>__<view>`` feature set:

* **joins** — every fragment with a known join partner queries the whole probe gallery
  (~5.3k images incl. collection-stratified distractors). R@1/10/100 and MRR of the
  first partner, split by same- vs cross-collection partners, plus AUC of join-pair
  similarity vs. random *same-collection* non-join pairs (so imaging conditions alone
  cannot score well).
* **scribes** — PGP scribe labels, leave-one-out over scribe-labelled fragments; pairs
  from the same PGP document (physical joins) are excluded so "same hand" is not just
  "same sheet". P@1, mAP, AUC same- vs different-scribe.
* **script** — kNN (k=10) balanced accuracy for KTIV script style and region.
* **confound** — kNN balanced accuracy for predicting the holding *collection*: high
  values mean the vector encodes photography setup rather than the manuscript.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from embed_utils import AUDIT_ROOT

HERE = Path(__file__).parent
# FJP join lists with the scrape's prefix-bug suspects removed (see build_image_manifest.py)
JOIN_KEY = "joins_fjp_clean"


def load_feats(name: str) -> tuple:
    """Load ids and features for one encoder/view, or an equal-weight fusion ``a+b+...``.

    :param name: ``<encoder>__<view>`` or several joined by ``+`` (late fusion: each block
        L2-normalised, concatenated, renormalised; equivalent to averaging cosine similarities).
    :returns: (ids, feats).
    :rtype: tuple
    """
    if "+" in name:
        loaded = [load_feats(n) for n in name.split("+")]
        common = sorted(set.intersection(*[set(i) for i, _ in loaded]))
        blocks = []
        for ids, X in loaded:
            pos = {c: k for k, c in enumerate(ids)}
            blocks.append(X[[pos[c] for c in common]])
        X = np.concatenate(blocks, axis=1)
        return common, X / np.linalg.norm(X, axis=1, keepdims=True)
    parts = sorted((AUDIT_ROOT / "cache" / "image_feats" / name).glob("part_*.npz"))
    ids, feats = [], []
    for p in parts:
        z = np.load(p)
        ids += list(z["ids"])
        feats.append(z["feats"])
    return ids, np.concatenate(feats)


def balanced_knn(sims: np.ndarray, labels: List[str], k: int = 10) -> float:
    """Leave-one-out kNN balanced accuracy.

    :param sims: Square similarity matrix over the labelled items.
    :param labels: Label per item.
    :param k: Neighbours.
    :returns: Balanced accuracy.
    :rtype: float
    """
    s = sims.copy()
    np.fill_diagonal(s, -9)
    nn = np.argpartition(-s, k, axis=1)[:, :k]
    pred = [Counter(labels[j] for j in row).most_common(1)[0][0] for row in nn]
    per = defaultdict(list)
    for p, y in zip(pred, labels):
        per[y].append(p == y)
    return float(np.mean([np.mean(v) for v in per.values()]))


def pair_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(pos > neg) via ranks.

    :param pos: Positive-pair similarities.
    :param neg: Negative-pair similarities.
    :returns: AUC.
    :rtype: float
    """
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(name: str, manifest: dict, sel: dict) -> dict:
    """Run every image measurement for one feature set.

    :param name: ``<encoder>__<view>``.
    :param manifest: Image manifest.
    :param sel: Probe selection.
    :returns: Results dict.
    :rtype: dict
    """
    ids, X = load_feats(name)
    pos = {c: i for i, c in enumerate(ids)}
    coll = [manifest["collection"][c] for c in ids]
    S = X @ X.T
    rng = np.random.default_rng(0)
    res: dict = {"features": name, "n_images": len(ids)}

    # joins
    partners: Dict[int, set] = defaultdict(set)
    source: Dict[int, set] = defaultdict(set)
    for key in (JOIN_KEY, "joins_pgp"):
        for a, b in manifest[key]:
            if a in pos and b in pos:
                partners[pos[a]].add(pos[b])
                partners[pos[b]].add(pos[a])
                source[pos[a]].add(key)
                source[pos[b]].add(key)
    ranks, same_c, cross_c, by_src = [], [], [], defaultdict(list)
    for q, ps in partners.items():
        s = S[q].copy()
        s[q] = -9
        order = np.argsort(-s)
        r = int(min(np.where(np.isin(order, list(ps)))[0])) + 1
        ranks.append(r)
        (same_c if any(coll[p] == coll[q] for p in ps) else cross_c).append(r)
        for src in source[q]:
            by_src[src].append(r)
    ranks = np.array(ranks)

    def rstats(rs):
        rs = np.array(rs)
        return {"n": int(len(rs)), "R@1": round(float((rs <= 1).mean()), 4), "R@10": round(float((rs <= 10).mean()), 4),
                "R@100": round(float((rs <= 100).mean()), 4), "MRR": round(float((1 / rs).mean()), 4),
                "median_rank": int(np.median(rs))} if len(rs) else {"n": 0}
    pos_pairs = [(q, p) for q, ps in partners.items() for p in ps if q < p]
    join_sims = np.array([S[a, b] for a, b in pos_pairs])
    by_coll = defaultdict(list)
    for i, c in enumerate(coll):
        by_coll[c].append(i)
    neg = []
    for a, b in pos_pairs:
        cands = by_coll[coll[a]]
        for _ in range(3):
            j = int(rng.choice(cands))
            if j != a and j not in partners[a]:
                neg.append(S[a, j])
    # analytic "collection prior" baseline: same-collection items first, random order
    prior_r10 = float(np.mean([min(1.0, 10 * len(ps) / max(1, len(by_coll[coll[q]]) - 1)) for q, ps in partners.items()]))
    res["joins"] = {"all": rstats(ranks), "same_collection": rstats(same_c), "cross_collection": rstats(cross_c),
                    "fjp_literary": rstats(by_src[JOIN_KEY]), "pgp_documentary": rstats(by_src["joins_pgp"]),
                    "auc_vs_same_collection_nonjoin": round(pair_auc(join_sims, np.array(neg)), 4),
                    "mean_sim_join": round(float(join_sims.mean()), 4), "mean_sim_samecoll_nonjoin": round(float(np.mean(neg)), 4),
                    "collection_prior_R@10_upper": round(prior_r10, 4)}

    # scribes (exclude same-PGP-document pairs)
    same_doc = set()
    for a, b in manifest["joins_pgp"]:
        same_doc.add((a, b))
        same_doc.add((b, a))
    sc = [(pos[c], s) for c, s in manifest["scribe"].items() if c in pos and c in set(sel["scribes"])]
    if sc:
        idx = [i for i, _ in sc]
        lab = [s for _, s in sc]
        sub = S[np.ix_(idx, idx)]
        p1, aps, sp, dp = [], [], [], []
        for qi in range(len(idx)):
            valid = [j for j in range(len(idx)) if j != qi and (ids[idx[qi]], ids[idx[j]]) not in same_doc]
            rel = np.array([lab[j] == lab[qi] for j in valid])
            if rel.sum() == 0:
                continue
            order = np.argsort(-sub[qi, valid])
            relo = rel[order]
            p1.append(relo[0])
            hits = np.cumsum(relo)
            aps.append(float(((hits / np.arange(1, len(relo) + 1)) * relo).sum() / relo.sum()))
            for j, r in zip(valid, rel):
                (sp if r else dp).append(sub[qi, j])
        chance_p1 = float(np.mean([(Counter(lab)[l] - 1) / (len(lab) - 1) for l in lab]))
        res["scribes"] = {"n_fragments": len(idx), "n_scribes": len(set(lab)), "P@1": round(float(np.mean(p1)), 4),
                          "mAP": round(float(np.mean(aps)), 4), "chance_P@1": round(chance_p1, 4),
                          "auc_same_vs_diff_scribe": round(pair_auc(np.array(sp), np.array(dp)), 4)}

    # script attributes
    res["script"] = {}
    for key, classes in (("script_style", {"Square", "Semi-Cursive", "Cursive", "Naskhi", "Rabbinical"}),
                         ("script_region", {"Oriental", "Spanish", "Yemenite", "Italian", "Ashkenazi", "North African", "Syrian"})):
        items = [(pos[c], a[key]) for c, a in manifest["attrs"].items() if c in pos and a.get(key) in classes]
        cnt = Counter(l for _, l in items)
        items = [(i, l) for i, l in items if cnt[l] >= 8]
        if len(items) > 20:
            idx = [i for i, _ in items]
            res["script"][key] = {"n": len(items), "classes": dict(Counter(l for _, l in items)),
                                  "knn10_balanced_acc": round(balanced_knn(S[np.ix_(idx, idx)], [l for _, l in items]), 4),
                                  "chance": round(1 / len({l for _, l in items}), 4)}
    # confound: predict collection
    top = [c for c, n in Counter(coll).most_common(12) if n >= 30]
    idx = [i for i, c in enumerate(coll) if c in top]
    res["confound_collection"] = {"n": len(idx), "n_collections": len(top),
                                  "knn10_balanced_acc": round(balanced_knn(S[np.ix_(idx, idx)], [coll[i] for i in idx]), 4),
                                  "chance": round(1 / len(top), 4)}
    return res


def main() -> None:
    """Evaluate all available (or the given) feature sets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default="", help="comma list; default = all present")
    args = parser.parse_args()
    manifest = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())
    sel = json.loads((AUDIT_ROOT / "image_selection_v1.json").read_text())
    names = args.features.split(",") if args.features else sorted(
        p.name for p in (AUDIT_ROOT / "cache" / "image_feats").iterdir()
        if any(p.glob("part_*.npz")) and not (p / "SKIP").exists() and not (p / "LOCK").exists())
    names += [n for n in ("dinov2-base__masked+dinov2-base__mpatches",
                          "dinov2-base__masked+dinov2-base__mpatches+qwen3vl-vit512__masked")
              if all((AUDIT_ROOT / "cache" / "image_feats" / part).exists() for part in n.split("+"))]
    out = []
    for name in names:
        r = evaluate(name, manifest, sel)
        r["ran_at"] = datetime.now().isoformat(timespec="seconds")
        out.append(r)
        j, s = r["joins"], r.get("scribes", {})
        print(f"{name:28s} joins R@10={j['all']['R@10']:.3f} (cross {j['cross_collection'].get('R@10', 0):.3f}, "
              f"fjp {j['fjp_literary'].get('R@10', 0):.3f}, pgp {j['pgp_documentary'].get('R@10', 0):.3f}) "
              f"AUC={j['auc_vs_same_collection_nonjoin']:.3f} | scribe P@1={s.get('P@1', 0):.3f} (chance {s.get('chance_P@1', 0):.3f}) "
              f"AUC={s.get('auc_same_vs_diff_scribe', 0):.3f} | style={r['script'].get('script_style', {}).get('knn10_balanced_acc', 0):.3f} "
              f"region={r['script'].get('script_region', {}).get('knn10_balanced_acc', 0):.3f} | "
              f"collection={r['confound_collection']['knn10_balanced_acc']:.3f} (chance {r['confound_collection']['chance']:.3f})", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "i1_images.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
