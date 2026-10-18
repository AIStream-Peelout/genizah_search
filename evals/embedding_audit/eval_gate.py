"""Pre-index acceptance gate gate_v1 for the text embedder (plan section E): pure numpy, standalone, ships to Colab.

No repo imports: the file is copied next to the packaged dataset and imported on Colab like ``eval_v3.py``. The gate
dir is written on the Mac by ``build_gate_v1.py`` (json only)::

    gate_dir/queries.jsonl     {qid, text, family, concept, form, blocked, lang, source, report_only}
    gate_dir/concepts.json     {cid: {family, relevant, canon, unjudged, alt_only, held_out_relevant, ...}}  eligible ids
    gate_dir/records.jsonl     {doc_id, source_group, desc_lang, len_bucket, held_out, new_since_v9, has_alt_titles,
                                text_hash, canon_kind, v9_trained, ...}  one row per pool (eligible) record
    gate_dir/probes.json       {rank_regressions, title_position, rank_sweep, ...}
    gate_dir/thresholds.json   frozen criteria (DEFAULT_THRESHOLDS merged with the spec; Isaac freezes, sha printed)

Usage (Colab, after ``eval_v3.evaluate`` for the same model)::

    from eval_gate import evaluate_gate, verdict, format_verdict, load_thresholds, thresholds_sha256
    print("thresholds sha256", thresholds_sha256(GATE))            # printed before training (pre-registration)
    rep = evaluate_gate(ids, mat, qfn, GATE, doc_vectors_fn=dfn, probe_text="semantic_v4")
    reports = {"P": {"gate": rep_p, "eval_v3": ev_p, "extra": {}}, ..., "ckpt1": {"gate": rep, "eval_v3": ev,
               "extra": {"bib": bib_report, "lco": {"cand": ev_lco, "base": ev_lco_base},
                         "leak": blocked_query_leaks(GATE, mix_anchors)}}}
    v = verdict(reports, "ckpt1", {"P": "P", "B": "B", "B3": "B3", "V": "V"}, load_thresholds(GATE))
    print(format_verdict(v))

Per query (concept queries; relevance from source fields, never from the embedded text): the pool minus the concept's
``unjudged`` records is ranked; ``canon`` records are misses. Ties are pessimistic (a tied irrelevant record ranks
first). Metrics: P@10, P@10_dd (rank, then collapse identical ``text_hash`` into one slot), P@20, AP, RR, MRR@100,
R@10, R@100, canon_share@20, law_share@20 (canon records that are tractate / law-code records), the alt-title-only
share of the relevant top-10 hits with its base rate. Views (G1/G2): ``desc_only`` (alt-only relevant records become
unjudged) and ``broad`` (unjudged records count as relevant; canon stays a miss). Population cells: AP and R@100 of the
relevant records inside a cell (other relevant records out of the ranking), for source_group, desc_lang, len_bucket,
held_out, new_since_v9, source_group*desc_lang, plus the query-language cut. Probes: H5 title position (needs
``doc_vectors_fn``), H6 seed rank regressions and held-out pair ranks. Structure: distinct@10, top hub, score spread,
random doc-doc cosine, participation ratio, out-of-domain AUC (G12).

:func:`verdict` implements H0-H12 with the thresholds file; NOT MEASURED counts as FAIL. Paired deltas use a cluster
bootstrap (clusters = concepts; known-item: the target record); SE = (ci_high - ci_low) / 3.92.

Self-test (numpy only)::

    python3 eval_gate.py --self-test
    python3 eval_gate.py --random-smoke --gate-dir GATE_DIR    # real gate files, random 64-d vectors
"""

import argparse
import hashlib
import json
import re
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GATE_FILES = {"queries": "queries.jsonl", "concepts": "concepts.json", "records": "records.jsonl",
              "probes": "probes.json", "thresholds": "thresholds.json"}
RECORD_CUTS = ("source_group", "desc_lang", "len_bucket", "held_out", "new_since_v9")
CROSS_CUTS = (("source_group", "desc_lang"),)
VIEW_FAMILIES = ("G1", "G2")
METRICS = ("P@10", "P@10_dd", "P@20", "AP", "RR", "MRR@100", "R@10", "R@100", "canon_share@20", "law_share@20")
VIEW_METRICS = ("P@10", "P@10_dd", "AP", "MRR@100")
LAW_KINDS = ("tractate", "law_code")
TOP_K = 200        # candidates kept per query for the top lists (dd collapse needs more than 10)
STORE_TOP = 20     # doc ids stored per query (review sheet, G11 per-query report)
NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")

# Pure data (read by build_gate_v1.py with ast.literal_eval, so: literals only). thresholds.json = this dict
# deep-merged with the spec's "thresholds"; verdict() reads ONLY the file it is given (no fallback to these values).
DEFAULT_THRESHOLDS = {
    "version": 1,
    "status": "proposed - Isaac freezes",
    "bootstrap": {"n_resamples": 2000, "seed": 0, "alpha": 0.05},
    "primary_metric": {"G1": "P@10_dd", "G2": "AP", "G3": "AP", "G4": "AP", "G5": "AP", "G6": "P@10_dd",
                       "G7": "P@10", "G10": "RR", "G11": "AP"},
    "H0": {"families": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G10"],
           "delta": {"P@10_dd": 0.03, "P@10": 0.03, "R@10": 0.03, "R@100": 0.03, "AP": 0.02, "RR": 0.02,
                     "MRR@100": 0.02},
           "frames_canon_margin": 0.03, "frames_canon_families": ["G1"], "frames_nonneg_families": ["G3", "G10"]},
    "H1": {"families": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G10"], "refs": ["B", "P"],
           "metrics": {"G1": ["P@10_dd"], "G2": ["AP", "P@10_dd"], "G3": ["AP", "P@10"], "G4": ["AP", "P@10_dd"],
                       "G5": ["AP", "P@10_dd"], "G6": ["P@10_dd", "R@100"], "G7": ["P@10"], "G10": ["RR", "R@10"]},
           "delta": {"P@10_dd": 0.03, "P@10": 0.03, "R@10": 0.03, "R@100": 0.03, "AP": 0.02, "RR": 0.02,
                     "MRR@100": 0.02},
           "se_mult": 2.0},
    "H2": {"refs": ["B", "P"], "families": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G10"], "metric": "AP",
           "cuts": ["source_group", "desc_lang", "len_bucket", "held_out", "new_since_v9", "source_group*desc_lang",
                    "q_lang"],
           "min_queries": 10, "min_pairs": 50, "floor": 0.10, "se_mult": 2.0, "max_cells_below_se_per_cut": 1,
           "hard_se_mult": 3.0, "hard_floor": 0.20},
    "H3": {"ref": "P", "families": ["G1"], "metric": "P@10_dd", "strict_se_mult": 1.0, "desc_only_ratio": 0.82,
           "broad_min_delta": 0.0, "worst_festival_max_drop": 0.10, "bare_name_max_drop": 0.30,
           "bare_name_min_ref": 0.3,
           "non_bare_forms": ["descriptive", "nuanced", "probe", "law", "router", "user_turn", "seed_descriptive"],
           "alt_lift_macro_max": 1.25, "alt_lift_festival_max": 1.5, "alt_lift_min_relevant": 30},
    "H4": {"ref": "P", "families": ["G1"], "canon_share_margin": 0.05, "law_share_margin": 0.03,
           "guard_family": "G3"},
    "H5": {"direction": "tail_to_head", "text_version": "semantic_v4", "max_abs_mean_dz": 0.25,
           "max_excess_over_B": 0.10,
           "probe": {"n_per_direction": 150, "families": ["G1", "G2", "G4", "G5"], "max_qids": 6,
                     "seed": "gate_v1"}},
    "H6": {"seed_max_rank": 50, "sweep_ref": "P", "sweep_families": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
           "sweep_max_ref_rank": 20, "sweep_min_share": 0.95, "sweep_rank_floor": 50, "sweep_rank_factor": 3,
           "sweep_deep_rank": 500, "sweep_max_deep_share": 0.02, "sweep_min_pairs": 50,
           "family_max_below_floor_share": None},
    "H7": {"refs": ["P", "B"], "families": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G10"],
           "min_r100_ratio": 0.90, "min_group_queries": 10, "lift_group": "KTIV",
           "lift_families": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"], "lift_margin": 0.25,
           "new_cell": "new_since_v9=True"},
    "H8": {"ki_types_vs_v": ["specific", "topical", "implicit"], "ki_max_drop_vs_v": 0.01,
           "ki_he_types": ["hebrew", "translit"], "ki_he_max_drop_vs_v": 0.05, "seen_ap_max_drop_vs_v": 0.02,
           "seen_worse_margin": 0.05, "seen_worse_max_share": 0.02},
    "H9": {"collection_lift_margin": 0.10, "top_hub_floor": 0.02, "distinct10_margin": 0.3,
           "memorisation_cap": 0.04, "v9_net_gap": 0.038},
    "H10": {"sections": ["dense_strict", "prod_hybrid_rrf_60_40_strict", "prod_hybrid_rrf_30_70_strict",
                         "chat_default_hybrid_rrf_50_50_cand40_strict"]},
    "H11": {"min_queries": {"G1": 80, "G2": 60, "G3": 30, "G4": 100, "G5": 50, "G6": 50, "G7": 15, "G10": 1900,
                            "G11": 30},
            "min_relevant": 5, "min_relevant_exempt": ["G10"], "max_text_hash_mismatch": 0},
    "H12": {"ap_floor": 0.182, "se_mult": 1.0},
    "soft": {"score_spread_ratio": [0.5, 2.0], "typo_margin": 0.05, "ood_auc_margin": 0.05},
}


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
def _read_jsonl(path: Path) -> List[dict]:
    """Rows of a JSONL file.

    :param path: File path.
    :returns: Row dicts.
    :rtype: List[dict]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256_file(path: Path) -> str:
    """sha256 hex digest of a file.

    :param path: File path.
    :returns: Hex digest.
    :rtype: str
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_gate_dir(gate_dir) -> dict:
    """Load the gate files written by ``build_gate_v1.py``.

    :param gate_dir: Directory holding :data:`GATE_FILES`.
    :returns: ``{queries, concepts, records, probes, thresholds, files_sha256}``.
    :rtype: dict
    """
    gate_dir = Path(gate_dir)
    out = {}
    for key, name in GATE_FILES.items():
        path = gate_dir / name
        out[key] = _read_jsonl(path) if name.endswith(".jsonl") else json.loads(path.read_text(encoding="utf-8"))
    out["files_sha256"] = {name: _sha256_file(gate_dir / name) for name in GATE_FILES.values()}
    return out


def load_thresholds(gate_dir) -> dict:
    """The frozen thresholds of a gate dir.

    :param gate_dir: Gate dir.
    :returns: Thresholds dict.
    :rtype: dict
    """
    return json.loads((Path(gate_dir) / GATE_FILES["thresholds"]).read_text(encoding="utf-8"))


def thresholds_sha256(gate_dir) -> str:
    """sha256 of ``thresholds.json`` (print it in the notebook before training: pre-registration).

    :param gate_dir: Gate dir.
    :returns: Hex digest.
    :rtype: str
    """
    return _sha256_file(Path(gate_dir) / GATE_FILES["thresholds"])


def v9_trained_ids(gate_dir) -> List[str]:
    """Pool records the v9 model trained on (pass them as ``ignore_ids`` when scoring V).

    :param gate_dir: Gate dir.
    :returns: Doc ids with ``v9_trained`` true in ``records.jsonl``.
    :rtype: List[str]
    """
    return [r["doc_id"] for r in _read_jsonl(Path(gate_dir) / GATE_FILES["records"]) if r.get("v9_trained")]


def normalise_rows(mat: np.ndarray) -> np.ndarray:
    """L2-row-normalised float32 copy.

    :param mat: (n, d) array.
    :returns: Normalised array.
    :rtype: np.ndarray
    """
    mat = np.asarray(mat, dtype=np.float32)
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)


def query_lang(text: str) -> str:
    """``he`` when the text holds Hebrew letters, else ``latin``.

    :param text: Query text.
    :returns: Language tag.
    :rtype: str
    """
    return "he" if re.search(r"[\u05D0-\u05EA]", text) else "latin"


# ---------------------------------------------------------------------------------------------------------------
# ranking primitives
# ---------------------------------------------------------------------------------------------------------------
def rank_positives(pos: np.ndarray, kept_asc: np.ndarray, rel_asc: np.ndarray) -> np.ndarray:
    """Pessimistic 1-based ranks of a set of positives, best first.

    The ranking holds the kept records (``kept_asc``, sorted ascending; it contains every relevant record). The
    positives are a subset of the relevant records; the other relevant records are out of the ranking (neither hit
    nor miss), so a positive's rank is its position among the positives plus the irrelevant kept records scoring at
    least as high (ties count against).

    :param pos: Scores of the positives.
    :param kept_asc: Sorted scores of every kept record (relevant included).
    :param rel_asc: Sorted scores of every relevant record.
    :returns: Ascending ranks.
    :rtype: np.ndarray
    """
    if not len(pos):
        return np.zeros(0, dtype=np.int64)
    p = np.sort(np.asarray(pos))[::-1]
    n_kept_ge = len(kept_asc) - np.searchsorted(kept_asc, p, side="left")
    n_rel_ge = len(rel_asc) - np.searchsorted(rel_asc, p, side="left")
    return np.arange(1, len(p) + 1) + (n_kept_ge - n_rel_ge)


def rank_metrics(ranks: np.ndarray) -> Dict[str, float]:
    """Rank-based metrics of a positive set.

    :param ranks: Ascending ranks of every positive (:func:`rank_positives`).
    :returns: ``P@10, P@20, AP, RR, MRR@100, R@10, R@100``.
    :rtype: Dict[str, float]
    """
    m = len(ranks)
    rr = 1.0 / float(ranks[0])
    return {"P@10": float((ranks <= 10).sum()) / 10, "P@20": float((ranks <= 20).sum()) / 20,
            "AP": float((np.arange(1, m + 1) / ranks).mean()), "RR": rr, "MRR@100": rr if ranks[0] <= 100 else 0.0,
            "R@10": float((ranks <= 10).sum()) / m, "R@100": float((ranks <= 100).sum()) / m}


def pessimistic_top(s: np.ndarray, exclude: Optional[np.ndarray], is_rel: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-``k`` records by score, ties broken pessimistically (irrelevant first).

    :param s: Scores over the pool.
    :param exclude: Boolean mask of records out of the ranking (None = none).
    :param is_rel: Boolean relevance mask.
    :param k: List length.
    :returns: Ordered pool indices.
    :rtype: np.ndarray
    """
    sv = s if exclude is None else np.where(exclude, -np.inf, s)
    n_ok = len(sv) if exclude is None else int((~exclude).sum())
    k = min(k, n_ok)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.argpartition(-sv, k - 1)[:k]
    return idx[np.lexsort((is_rel[idx], -sv[idx]))]


def dedup_slots(top: np.ndarray, hash_ids: np.ndarray, k: int = 10) -> np.ndarray:
    """First record of each distinct text in a ranked list, up to ``k`` slots.

    :param top: Ranked pool indices.
    :param hash_ids: Text-group id per pool record.
    :param k: Number of slots.
    :returns: Pool indices of the slot representatives.
    :rtype: np.ndarray
    """
    seen, reps = set(), []
    for i in top:
        h = int(hash_ids[i])
        if h in seen:
            continue
        seen.add(h)
        reps.append(int(i))
        if len(reps) == k:
            break
    return np.asarray(reps, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------------------------
def cluster_bootstrap(diffs: Dict[str, Sequence[float]], n_resamples: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> Dict[str, float]:
    """Cluster bootstrap of the mean paired difference over queries (clusters resampled with replacement).

    With fewer than two clusters every query is its own cluster.

    :param diffs: Cluster -> per-query differences (candidate minus reference).
    :param n_resamples: Bootstrap resamples.
    :param seed: RNG seed.
    :param alpha: Two-sided level.
    :returns: ``{delta, ci_low, ci_high, se, n_clusters, n_queries}`` (se = CI width / 3.92).
    :rtype: Dict[str, float]
    """
    groups = [np.asarray(v, dtype=np.float64) for v in diffs.values() if len(v)]
    if not groups:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "n_clusters": 0, "n_queries": 0}
    if len(groups) < 2:
        groups = [np.asarray([x]) for x in groups[0]]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups], dtype=np.float64)
    delta = float(sums.sum() / counts.sum())
    if len(groups) < 2:
        return {"delta": round(delta, 5), "ci_low": round(delta, 5), "ci_high": round(delta, 5), "se": 0.0,
                "n_clusters": len(groups), "n_queries": int(counts.sum())}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), size=(n_resamples, len(groups)))
    stats = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"delta": round(delta, 5), "ci_low": round(float(lo), 5), "ci_high": round(float(hi), 5),
            "se": round(float(hi - lo) / 3.92, 5), "n_clusters": len(groups), "n_queries": int(counts.sum())}


def two_stage_bootstrap(a: Dict[str, Sequence[float]], b: Dict[str, Sequence[float]], n_resamples: int = 1000,
                        seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Two-stage clustered paired bootstrap of ``mean_c mean_q (b - a)`` (same algorithm as
    ``eval_v3.paired_bootstrap``, re-implemented here so this file has no repo imports).

    :param a: Cluster -> per-query values of the reference.
    :param b: Cluster -> per-query values of the candidate, aligned with ``a``.
    :param n_resamples: Bootstrap resamples.
    :param seed: RNG seed.
    :param alpha: Two-sided level.
    :returns: ``{delta, ci_low, ci_high, se, n_clusters, n_queries}``.
    :rtype: Dict[str, float]
    :raises ValueError: When a cluster's query lists differ in length.
    """
    diffs = []
    for c in sorted(set(a) & set(b)):
        da, db = np.asarray(a[c], dtype=np.float64), np.asarray(b[c], dtype=np.float64)
        if len(da) != len(db):
            raise ValueError(f"cluster {c!r}: {len(da)} vs {len(db)} queries; results are not aligned")
        if len(da):
            diffs.append(db - da)
    if not diffs:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "n_clusters": 0, "n_queries": 0}
    rng = np.random.default_rng(seed)
    k = len(diffs)
    within = np.empty((k, n_resamples))
    for i, d in enumerate(diffs):
        within[i] = d[rng.integers(0, len(d), size=(n_resamples, len(d)))].mean(axis=1)
    pick = rng.integers(0, k, size=(n_resamples, k))
    stats = within[pick, np.arange(n_resamples)[:, None]].mean(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"delta": round(float(np.mean([d.mean() for d in diffs])), 5), "ci_low": round(float(lo), 5),
            "ci_high": round(float(hi), 5), "se": round(float(hi - lo) / 3.92, 5), "n_clusters": k,
            "n_queries": int(sum(len(d) for d in diffs))}


# ---------------------------------------------------------------------------------------------------------------
# evaluate_gate
# ---------------------------------------------------------------------------------------------------------------
def _pool_context(gd: dict, row_of: Dict[str, int], ignore: set, hash_field: str) -> dict:
    """Pool arrays shared by every query: attributes, text groups, cell masks, concept index arrays.

    :param gd: Loaded gate dir.
    :param row_of: ``doc_id -> row of the model's doc matrix``.
    :param ignore: Ids removed from the pool.
    :param hash_field: Record field used for duplicate-text collapse.
    :returns: Context dict.
    :rtype: dict
    """
    recs = [r for r in gd["records"] if r["doc_id"] in row_of and r["doc_id"] not in ignore]
    pos = {r["doc_id"]: i for i, r in enumerate(recs)}
    n = len(recs)
    hashes: Dict[str, int] = {}
    hash_ids = np.array([hashes.setdefault(str(r.get(hash_field) or f"__doc__{r['doc_id']}"), len(hashes))
                         for r in recs], dtype=np.int64)
    attrs = {cut: [None if r.get(cut) is None else str(r.get(cut)) for r in recs] for cut in RECORD_CUTS}
    for a, b in CROSS_CUTS:
        attrs[f"{a}*{b}"] = [None if x is None or y is None else f"{x}|{y}" for x, y in zip(attrs[a], attrs[b])]
    cells = {}
    for cut, vals in attrs.items():
        arr = np.array(["" if v is None else v for v in vals], dtype=object)
        cells[cut] = {v: np.flatnonzero(arr == v) for v in sorted(set(vals) - {None})}
    groups = np.array([str(r.get("source_group") or "other") for r in recs], dtype=object)
    law = np.array([r.get("canon_kind") in LAW_KINDS for r in recs], dtype=bool)

    def idx(ids: Iterable[str]) -> np.ndarray:
        return np.array(sorted(pos[d] for d in ids if d in pos), dtype=np.int64)

    concepts = {}
    for cid, c in gd["concepts"].items():
        concepts[cid] = {"family": c.get("family"), "rel": idx(c.get("relevant", [])),
                         "canon": idx(c.get("canon", [])), "unj": idx(c.get("unjudged", [])),
                         "alt": idx(c.get("alt_only", [])),
                         "held": np.array([pos.get(d, -1) for d in c.get("held_out_relevant", [])], dtype=np.int64)}
    return {"recs": recs, "pos": pos, "n": n, "hash_ids": hash_ids, "cells": cells, "groups": groups, "law": law,
            "concepts": concepts, "doc_ids": [r["doc_id"] for r in recs]}


def _score_query(s: np.ndarray, q: dict, ctx: dict, views: bool, sweep: bool, store_top: int,
                 top_k: int) -> dict:
    """All per-query numbers for one concept query (see the module docstring).

    :param s: Scores of the query over the pool.
    :param q: Query row.
    :param ctx: :func:`_pool_context` output.
    :param views: Also compute the desc_only / broad views.
    :param sweep: Also store the held-out pair ranks (H6 sweep).
    :param store_top: Doc ids to store from the top of the main ranking.
    :param top_k: Candidates for the top lists.
    :returns: Per-query dict.
    :rtype: dict
    """
    n = ctx["n"]
    c = ctx["concepts"][q["concept"]]
    rel, canon, unj, alt = c["rel"], c["canon"], c["unj"], c["alt"]
    is_rel = np.zeros(n, dtype=bool)
    is_rel[rel] = True
    excl = np.zeros(n, dtype=bool)
    excl[unj] = True
    excl[rel] = False                           # defensive: relevance wins over unjudged
    kept_asc = np.sort(s[~excl])
    rel_asc = np.sort(s[rel])
    ranks = rank_positives(s[rel], kept_asc, rel_asc)
    met = rank_metrics(ranks)
    top = pessimistic_top(s, excl, is_rel, top_k)
    slots = dedup_slots(top, ctx["hash_ids"], 10)
    met["P@10_dd"] = float(is_rel[slots].sum()) / 10
    is_canon = np.zeros(n, dtype=bool)
    is_canon[canon] = True
    is_canon &= ~is_rel
    top20 = top[:20]
    met["canon_share@20"] = float(is_canon[top20].sum()) / 20
    met["law_share@20"] = float((is_canon & ctx["law"])[top20].sum()) / 20
    hits10 = top[:10][is_rel[top[:10]]]
    is_alt = np.zeros(n, dtype=bool)
    is_alt[alt] = True
    out = {"family": q["family"], "concept": q["concept"], "form": q.get("form"), "lang": q["lang"],
           "n_rel": int(len(rel)), "n_canon": int(len(canon)), "n_unjudged": int(len(unj)),
           "metrics": {k: round(v, 5) for k, v in met.items()},
           "alt": [int(is_alt[hits10].sum()), int(len(hits10)), round(float(is_alt[rel].mean()), 5)],
           "distinct@10": int(len({int(ctx["hash_ids"][i]) for i in top[:10]})),
           "top": [ctx["doc_ids"][i] for i in top[:store_top]]}
    out["metrics"]["alt_only_share"] = round(out["alt"][0] / out["alt"][1], 5) if out["alt"][1] else None
    g_hits = Counter(ctx["groups"][hits10].tolist())
    out["g_hits10"] = dict(g_hits)
    out["g_rel"] = dict(Counter(ctx["groups"][rel].tolist()))
    cells = {f"q_lang={q['lang']}": [out["metrics"]["AP"], out["metrics"]["R@100"], int(len(rel))]}
    for cut, values in ctx["cells"].items():
        for v, members in values.items():
            sub = np.intersect1d(rel, members, assume_unique=True)
            if len(sub):
                r = rank_positives(s[sub], kept_asc, rel_asc)
                cells[f"{cut}={v}"] = [round(float((np.arange(1, len(r) + 1) / r).mean()), 5),
                                       round(float((r <= 100).mean()), 5), int(len(sub))]
    out["cells"] = cells
    if views:
        out["views"] = {}
        keep_desc = np.setdiff1d(rel, alt, assume_unique=True)
        if len(keep_desc):
            r = rank_positives(s[keep_desc], kept_asc, rel_asc)
            m = rank_metrics(r)
            is_rel_d = is_rel & ~is_alt
            top_d = [i for i in top if not is_alt[i]]
            m["P@10_dd"] = float(is_rel_d[dedup_slots(np.asarray(top_d, dtype=np.int64), ctx["hash_ids"], 10)].sum()) / 10
            out["views"]["desc_only"] = {k: round(m[k], 5) for k in VIEW_METRICS}
        else:
            out["views"]["desc_only"] = None
        broad = np.union1d(rel, unj)
        is_rel_b = np.zeros(n, dtype=bool)
        is_rel_b[broad] = True
        r = rank_positives(s[broad], np.sort(s), np.sort(s[broad]))
        m = rank_metrics(r)
        top_b = pessimistic_top(s, None, is_rel_b, top_k)
        m["P@10_dd"] = float(is_rel_b[dedup_slots(top_b, ctx["hash_ids"], 10)].sum()) / 10
        out["views"]["broad"] = {k: round(m[k], 5) for k in VIEW_METRICS}
    if sweep and len(c["held"]):
        held = c["held"]
        ok = held >= 0
        ranks_h = np.full(len(held), -1, dtype=np.int64)
        ranks_h[ok] = len(kept_asc) - np.searchsorted(kept_asc, s[held[ok]], side="left")
        out["pair_ranks"] = [int(x) if x > 0 else None for x in ranks_h]
    return out


def _aggregate(per_query: Dict[str, dict], keys: Sequence[str], path: str = "metrics") -> Dict[str, Optional[float]]:
    """Mean of each metric over queries (None values skipped).

    :param per_query: ``qid -> per-query dict`` (subset).
    :param keys: Metric names.
    :param path: ``metrics`` or a view name under ``views``.
    :returns: ``{metric: mean or None, n_queries}``.
    :rtype: Dict[str, Optional[float]]
    """
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        if path == "metrics":
            vals = [p["metrics"].get(k) for p in per_query.values()]
        else:
            vals = [(p.get("views") or {}).get(path, None) for p in per_query.values()]
            vals = [v.get(k) if v else None for v in vals]
        vals = [v for v in vals if v is not None]
        out[k] = round(float(np.mean(vals)), 5) if vals else None
    out["n_queries"] = len(per_query)
    return out


def group_lift(per_query: Dict[str, dict], group: str) -> Optional[float]:
    """Hits-weighted lift of one source group among relevant top-10 hits (share of hits / share of relevant).

    :param per_query: Per-query dicts.
    :param group: Source group, e.g. ``KTIV``.
    :returns: Lift, or None when there is no relevant hit.
    :rtype: Optional[float]
    """
    hits = g_hits = base_w = 0.0
    for p in per_query.values():
        h = sum(p.get("g_hits10", {}).values())
        if not h:
            continue
        hits += h
        g_hits += p["g_hits10"].get(group, 0)
        base_w += h * p.get("g_rel", {}).get(group, 0) / max(p["n_rel"], 1)
    if not hits or not base_w:
        return None
    return round((g_hits / hits) / (base_w / hits), 4)


def alt_lift(per_query: Dict[str, dict]) -> Optional[float]:
    """Hits-weighted alt-title-only lift: share of alt-only records among relevant top-10 hits / their base rate.

    :param per_query: Per-query dicts (with ``alt`` = [alt hits, hits, base rate]).
    :returns: Lift, or None without relevant hits or alt-only records.
    :rtype: Optional[float]
    """
    a = sum(p["alt"][0] for p in per_query.values() if "alt" in p)
    h = sum(p["alt"][1] for p in per_query.values() if "alt" in p)
    w = sum(p["alt"][1] * p["alt"][2] for p in per_query.values() if "alt" in p)
    if not h or not w:
        return None
    return round((a / h) / (w / h), 4)


def _structure(P: np.ndarray, per_query: Dict[str, dict], top1: Dict[str, float], sigma: Dict[str, float],
               queries: Dict[str, dict], seed: int) -> dict:
    """Structure / drift numbers of one model.

    :param P: Normalised pool matrix.
    :param per_query: Scored per-query dicts.
    :param top1: ``qid -> best pool score`` for every query.
    :param sigma: ``qid -> std of the query's pool scores``.
    :param queries: ``qid -> query row``.
    :param seed: RNG seed for the samples.
    :returns: Structure dict.
    :rtype: dict
    """
    rng = np.random.default_rng(seed)
    n = len(P)
    out: dict = {}
    scored = list(per_query.values())
    out["distinct@10"] = round(float(np.mean([p["distinct@10"] for p in scored])), 4) if scored else None
    slots = Counter(d for p in scored for d in p["top"][:10])
    out["top_hubs"] = [{"doc_id": d, "slots": c, "share_of_queries": round(c / len(scored), 4)}
                       for d, c in slots.most_common(5)] if scored else []
    out["top_hub_share"] = out["top_hubs"][0]["share_of_queries"] if out["top_hubs"] else None
    out["score_spread"] = round(float(np.mean(list(sigma.values()))), 5) if sigma else None
    if n >= 2:
        i = rng.integers(0, n, size=5000)
        j = rng.integers(0, n, size=5000)
        keep = i != j
        out["random_doc_cosine"] = round(float(np.einsum("ij,ij->i", P[i[keep]], P[j[keep]]).mean()), 4)
        sample = P[rng.choice(n, size=min(n, 4000), replace=False)].astype(np.float64)
        sv = np.linalg.svd(sample - sample.mean(axis=0), compute_uv=False)
        lam = sv ** 2
        out["participation_ratio"] = round(float(lam.sum() ** 2 / (lam ** 2).sum()), 3)
        out["top10_eig_share"] = round(float(lam[:10].sum() / lam.sum()), 4)
    ood = [top1[q] for q, r in queries.items() if r["family"] == "G12" and q in top1]
    ind = [top1[q] for q in per_query if q in top1]
    if ood and ind:
        a, b = np.asarray(ind)[:, None], np.asarray(ood)[None, :]
        out["ood_auc"] = round(float((a > b).mean() + 0.5 * (a == b).mean()), 4)
        out["n_ood"] = len(ood)
    else:
        out["ood_auc"] = None
    return out


def _typo_robustness(per_query: Dict[str, dict], primary: Dict[str, str]) -> Optional[dict]:
    """Mean (typo - clean) primary metric per concept, averaged over concepts that have both kinds of query.

    :param per_query: Scored per-query dicts.
    :param primary: Family -> primary metric.
    :returns: ``{mean_delta, n_concepts}`` or None.
    :rtype: Optional[dict]
    """
    by: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"typo": [], "clean": []})
    for p in per_query.values():
        m = p["metrics"].get(primary.get(p["family"], "AP"))
        if m is None:
            continue
        form = str(p.get("form") or "")
        if "typo" in form:
            by[p["concept"]]["typo"].append(m)
        elif form not in ("descriptive", "nuanced", "probe", "law"):
            by[p["concept"]]["clean"].append(m)
    deltas = [np.mean(v["typo"]) - np.mean(v["clean"]) for v in by.values() if v["typo"] and v["clean"]]
    return {"mean_delta": round(float(np.mean(deltas)), 4), "n_concepts": len(deltas)} if deltas else None


def _title_position(probes: List[dict], text_version: Optional[str], doc_vectors_fn: Optional[Callable],
                    qvec: Dict[str, np.ndarray], qtext: Dict[str, str], sigma: Dict[str, float]) -> Optional[dict]:
    """H5 title-position probe: z-scored score change when the concept text moves position in the record text.

    :param probes: ``probes.json["title_position"]``.
    :param text_version: Text version of this model's documents (``semantic_v4`` / ``semantic_v3``); None = skip.
    :param doc_vectors_fn: Document encoder (list of texts -> (n, d)); None = skip.
    :param qvec: Query text -> vector.
    :param qtext: qid -> query text.
    :param sigma: qid -> std of the query's pool scores.
    :returns: ``{text_version, directions: {dir: {mean_dz, se, n_records, n_pairs, mean_abs_dz}}}`` or None.
    :rtype: Optional[dict]
    """
    if doc_vectors_fn is None or text_version is None:
        return None
    use = [p for p in probes if p.get("text_version") == text_version and any(q in sigma for q in p["qids"])]
    if not use:
        return {"text_version": text_version, "directions": {}}
    texts = list(dict.fromkeys(t for p in use for t in (p["orig_text"], p["variant_text"])))
    vecs = normalise_rows(doc_vectors_fn(texts))
    if len(vecs) != len(texts):
        raise ValueError(f"doc_vectors_fn returned {len(vecs)} vectors for {len(texts)} texts")
    tv = dict(zip(texts, vecs))
    per_dir: Dict[str, List[float]] = defaultdict(list)
    n_pairs: Counter = Counter()
    for p in use:
        dz = [float((tv[p["variant_text"]] - tv[p["orig_text"]]) @ qvec[qtext[q]]) / max(sigma[q], 1e-9)
              for q in p["qids"] if q in sigma]
        if dz:
            per_dir[p["direction"]].append(float(np.mean(dz)))
            n_pairs[p["direction"]] += len(dz)
    out = {}
    for d, vals in per_dir.items():
        v = np.asarray(vals)
        out[d] = {"mean_dz": round(float(v.mean()), 4),
                  "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 4) if len(v) > 1 else 0.0,
                  "mean_abs_dz": round(float(np.abs(v).mean()), 4), "n_records": int(len(v)),
                  "n_pairs": int(n_pairs[d])}
    return {"text_version": text_version, "directions": out}


def evaluate_gate(doc_ids: Sequence[str], doc_matrix: np.ndarray, query_vectors_fn: Callable[[List[str]], np.ndarray],
                  gate_dir, ignore_ids: Optional[Iterable[str]] = None, *,
                  doc_vectors_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
                  probe_text: Optional[str] = None, text_hashes: Optional[Dict[str, str]] = None,
                  hash_field: str = "text_hash", view_families: Sequence[str] = VIEW_FAMILIES,
                  chunk: int = 256, top_k: int = TOP_K, store_top: int = STORE_TOP, seed: int = 0) -> dict:
    """Score one model on the gate (per query, per family, per population cell, probes, structure).

    :param doc_ids: Record ids, one per row of ``doc_matrix`` (extra rows, e.g. ineligible records, are ignored).
    :param doc_matrix: (n, d) document vectors of THIS model's text (cosine; normalised here).
    :param query_vectors_fn: List of query strings -> (n, d) array; applies the model's query prefix itself.
    :param gate_dir: Directory written by ``build_gate_v1.py``.
    :param ignore_ids: Records removed from the pool entirely (``v9_trained_ids(gate_dir)`` when scoring V).
    :param doc_vectors_fn: Optional document encoder for the H5 title-position probe (None: H5 not measured).
    :param probe_text: Text version of this model's documents for H5 (``semantic_v4`` for B / candidates,
        ``semantic_v3`` for B3 / V: the label-swap diagnostic); None skips the probe.
    :param text_hashes: Optional ``doc_id -> hash`` of the texts actually embedded; compared with ``hash_field`` of
        ``records.jsonl`` (``text_hash`` = md5 of the gate corpus ``text``; ``text_hash_v3`` = md5 of ``text_v3``).
    :param hash_field: Record field for the duplicate-text collapse and the hash check.
    :param view_families: Families whose queries also get the desc_only / broad views.
    :param chunk: Queries per score block.
    :param top_k: Candidates per query for the top lists.
    :param store_top: Doc ids stored per query.
    :param seed: RNG seed for the structure samples.
    :returns: Report dict (``meta``, ``per_query``, ``families``, ``concepts``, ``cells``, ``groups``, ``probes``,
        ``structure``, ``soft``, ``report_only``).
    :rtype: dict
    """
    t0 = time.time()
    gd = load_gate_dir(gate_dir)
    th = gd["thresholds"]
    ignore = set(ignore_ids or [])
    row_of = {d: i for i, d in enumerate(doc_ids)}
    ctx = _pool_context(gd, row_of, ignore, hash_field)
    P = normalise_rows(np.asarray(doc_matrix)[[row_of[d] for d in ctx["doc_ids"]]])
    sweep_fams = set(th.get("H6", {}).get("sweep_families", []))
    queries = {q["qid"]: q for q in gd["queries"]}
    for q in queries.values():
        q.setdefault("lang", query_lang(q["text"]))
    scorable, unscorable, report_only = [], [], []
    for qid, q in queries.items():
        c = ctx["concepts"].get(q.get("concept")) if q.get("concept") else None
        if q.get("report_only") or c is None:
            report_only.append(qid)
        elif len(c["rel"]) == 0:
            unscorable.append(qid)
        else:
            scorable.append(qid)
    seeds = [c for c in gd["probes"].get("rank_regressions", []) if c.get("doc_id")]
    texts = list(dict.fromkeys([q["text"] for q in queries.values()] + [c["text"] for c in seeds]))
    qmat = normalise_rows(query_vectors_fn(texts)) if texts else np.zeros((0, P.shape[1]), dtype=np.float32)
    if len(qmat) != len(texts):
        raise ValueError(f"query_vectors_fn returned {len(qmat)} vectors for {len(texts)} texts")
    qvec = dict(zip(texts, qmat))
    per_query: Dict[str, dict] = {}
    top1: Dict[str, float] = {}
    sigma: Dict[str, float] = {}
    ro_out: Dict[str, dict] = {}
    order = scorable + report_only
    Q = np.stack([qvec[queries[q]["text"]] for q in order]) if order else np.zeros((0, P.shape[1]), np.float32)
    for start in range(0, len(order), chunk):
        S = Q[start:start + chunk] @ P.T
        for i, s in enumerate(S):
            qid = order[start + i]
            q = queries[qid]
            top1[qid] = float(s.max()) if len(s) else 0.0
            sigma[qid] = float(s.std()) if len(s) else 0.0
            if qid in report_only or not len(s):
                top = np.argsort(-s)[:store_top]
                ro_out[qid] = {"family": q["family"], "concept": q.get("concept"), "form": q.get("form"),
                               "top1": round(top1[qid], 5), "top": [ctx["doc_ids"][j] for j in top]}
                continue
            res = _score_query(s, q, ctx, q["family"] in view_families, q["family"] in sweep_fams, store_top, top_k)
            res["sigma"], res["top1"] = round(sigma[qid], 6), round(top1[qid], 5)
            per_query[qid] = res
    primary = th.get("primary_metric", DEFAULT_THRESHOLDS["primary_metric"])
    fam_q: Dict[str, Dict[str, dict]] = defaultdict(dict)
    con_q: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for qid, p in per_query.items():
        fam_q[p["family"]][qid] = p
        con_q[p["concept"]][qid] = p
    keys = METRICS + ("alt_only_share",)
    families = {}
    for f, pq in sorted(fam_q.items()):
        families[f] = _aggregate(pq, keys)
        if f in view_families:
            families[f]["views"] = {v: _aggregate(pq, VIEW_METRICS, v) for v in ("desc_only", "broad")}
        families[f]["alt_lift"] = alt_lift(pq)
    concepts = {cid: {"family": next(iter(pq.values()))["family"], "n_relevant": next(iter(pq.values()))["n_rel"],
                      **_aggregate(pq, ("P@10_dd", "AP", "RR", "canon_share@20")), "alt_lift": alt_lift(pq)}
                for cid, pq in sorted(con_q.items())}
    cells: Dict[str, Dict[str, dict]] = defaultdict(dict)
    acc: Dict[str, List[list]] = defaultdict(list)
    for p in per_query.values():
        for key, v in p["cells"].items():
            acc[key].append(v)
    for key, vals in sorted(acc.items()):
        cut, val = key.split("=", 1)
        cells[cut][val] = {"n_queries": len(vals), "n_pairs": int(sum(v[2] for v in vals)),
                           "AP": round(float(np.mean([v[0] for v in vals])), 5),
                           "R@100": round(float(np.mean([v[1] for v in vals])), 5)}
    group_names = sorted({str(g) for g in ctx["groups"]})
    groups = {g: {"lift@10": group_lift(per_query, g), **cells.get("source_group", {}).get(g, {})}
              for g in group_names}
    rank_reg = {}
    for c in seeds:
        cid = c.get("case_id") or f"{c['text']}|{c['doc_id']}"
        if c["doc_id"] not in ctx["pos"]:
            rank_reg[cid] = {"rank": None, "max_rank": c.get("max_rank"), "in_pool": False}
            continue
        s = P @ qvec[c["text"]]
        target = ctx["pos"][c["doc_id"]]
        hits = np.flatnonzero(ctx["hash_ids"] == ctx["hash_ids"][target])
        best = s[hits].max()
        rank_reg[cid] = {"rank": int((s >= best).sum() - (s[hits] >= best).sum()) + 1,
                         "max_rank": c.get("max_rank"), "in_pool": True}
    qtext = {qid: q["text"] for qid, q in queries.items()}
    mismatch = None
    if text_hashes is not None:
        mismatch = sum(text_hashes.get(r["doc_id"]) != r.get(hash_field) for r in ctx["recs"])
    return {"meta": {"gate_dir": str(gate_dir), "files_sha256": gd["files_sha256"],
                     "n_pool_frozen": len(gd["records"]), "n_pool_scored": ctx["n"],
                     "n_missing_vectors": sum(r["doc_id"] not in row_of for r in gd["records"]),
                     "n_ignored": sum(r["doc_id"] in ignore and r["doc_id"] in row_of for r in gd["records"]),
                     "n_queries": len(queries), "n_scored": len(per_query), "n_unscorable": len(unscorable),
                     "unscorable": unscorable[:50], "n_report_only": len(report_only), "hash_field": hash_field,
                     "text_hash_mismatch": mismatch, "probe_text": probe_text,
                     "seconds": round(time.time() - t0, 1)},
            "per_query": per_query, "report_only": ro_out, "families": families, "concepts": concepts,
            "cells": dict(cells), "groups": groups,
            "probes": {"rank_regressions": rank_reg,
                       "title_position": _title_position(gd["probes"].get("title_position", []), probe_text,
                                                         doc_vectors_fn, qvec, qtext, sigma)},
            "structure": _structure(P, per_query, top1, sigma, queries, seed),
            "soft": {"typo": _typo_robustness(per_query, primary)}}


def blocked_query_leaks(gate_dir, anchors: Iterable[str], min_contain: int = 3) -> dict:
    """H11 leak check: blocked gate queries that are a training anchor (exact) or sit inside one (contiguous tokens).

    Tokens: niqqud stripped, lower-cased, punctuation as spaces (``build_training_mix_v3.norm_tokens``). A query of
    at least ``min_contain`` tokens counts as contained when its token run occurs inside an anchor.

    :param gate_dir: Gate dir (``queries.jsonl``).
    :param anchors: Training anchor texts of the mix (every row's ``anchor``).
    :param min_contain: Minimum query length (tokens) for the containment test.
    :returns: ``{n_blocked, exact, contained, examples}``.
    :rtype: dict
    """
    def toks(text: str) -> Tuple[str, ...]:
        return tuple(re.sub(r"[^\w\s]|_", " ", NIQQUD_RE.sub("", unicodedata.normalize("NFC", text)).lower()).split())

    blocked = {toks(q["text"]): q["qid"] for q in _read_jsonl(Path(gate_dir) / GATE_FILES["queries"])
               if q.get("blocked")}
    blocked.pop((), None)
    exact, contained = [], []
    by_len: Dict[int, Dict[Tuple[str, ...], str]] = defaultdict(dict)
    for t, qid in blocked.items():
        if len(t) >= min_contain:
            by_len[len(t)][t] = qid
    seen_exact, seen_cont = set(), set()
    for a in anchors:
        at = toks(a)
        if at in blocked and blocked[at] not in seen_exact:
            seen_exact.add(blocked[at])
            exact.append([blocked[at], a])
        for n, table in by_len.items():
            for i in range(len(at) - n + 1):
                hit = table.get(at[i:i + n])
                if hit and hit not in seen_cont and at != at[i:i + n]:
                    seen_cont.add(hit)
                    contained.append([hit, a])
    return {"n_blocked": len(blocked), "exact": len(exact), "contained": len(contained),
            "examples": (exact + contained)[:20]}


def unmatched_top(report: dict, gate_dir, k: int = 20) -> Dict[str, List[str]]:
    """Rule audit helper: per concept, the top-ranked records that are neither relevant, canon nor unjudged.

    Run on B's report to fill the "20 unmatched from B's top 20" half of the review sheet.

    :param report: :func:`evaluate_gate` output.
    :param gate_dir: Gate dir.
    :param k: Records per concept.
    :returns: ``concept -> doc ids`` (best first, deduplicated over the concept's queries).
    :rtype: Dict[str, List[str]]
    """
    concepts = json.loads((Path(gate_dir) / GATE_FILES["concepts"]).read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = defaultdict(list)
    for p in report["per_query"].values():
        c = concepts[p["concept"]]
        judged = set(c.get("relevant", [])) | set(c.get("canon", [])) | set(c.get("unjudged", []))
        for d in p["top"]:
            if d not in judged and d not in out[p["concept"]] and len(out[p["concept"]]) < k:
                out[p["concept"]].append(d)
    return dict(out)


# ---------------------------------------------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------------------------------------------
def _get(pq: Optional[dict], path: Tuple) -> Optional[float]:
    """A number inside a per-query dict (``("metrics", "AP")``, ``("views", "broad", "P@10_dd")``,
    ``("cells", "source_group=KTIV", 0)``), or None.

    :param pq: Per-query dict.
    :param path: Key path.
    :returns: Value or None.
    :rtype: Optional[float]
    """
    cur = pq
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and key < len(cur):
            cur = cur[key]
        else:
            return None
    return None if cur is None else float(cur)


def paired_delta(rep_ref: dict, rep_cand: dict, qids: Iterable[str], path: Tuple, boot: dict) -> Optional[dict]:
    """Paired delta (candidate minus reference) over the queries both reports scored, cluster bootstrap by concept.

    :param rep_ref: Reference gate report.
    :param rep_cand: Candidate gate report.
    :param qids: Query ids to pair.
    :param path: Metric path (see :func:`_get`).
    :param boot: ``{n_resamples, seed, alpha}``.
    :returns: Bootstrap dict plus ``ref`` / ``cand`` means, or None when no query pairs.
    :rtype: Optional[dict]
    """
    diffs: Dict[str, List[float]] = defaultdict(list)
    va, vb = [], []
    for q in qids:
        pa, pb = rep_ref["per_query"].get(q), rep_cand["per_query"].get(q)
        a, b = _get(pa, path), _get(pb, path)
        if a is None or b is None:
            continue
        diffs[pa["concept"]].append(b - a)
        va.append(a)
        vb.append(b)
    if not va:
        return None
    out = cluster_bootstrap(diffs, boot.get("n_resamples", 2000), boot.get("seed", 0), boot.get("alpha", 0.05))
    out["ref"], out["cand"] = round(float(np.mean(va)), 5), round(float(np.mean(vb)), 5)
    return out


def _qids(rep: dict, families: Iterable[str]) -> List[str]:
    """Scored query ids of the given families.

    :param rep: Gate report.
    :param families: Family ids.
    :returns: Query ids.
    :rtype: List[str]
    """
    fams = set(families)
    return [q for q, p in rep["per_query"].items() if p["family"] in fams]


def _check(name: str, passed: Optional[bool], measured: bool, value=None, threshold=None, detail=None) -> dict:
    """One sub-check of a criterion.

    :param name: Check name.
    :param passed: Outcome (None when not measured).
    :param measured: Whether the inputs existed.
    :param value: Measured value.
    :param threshold: Threshold / rule text.
    :param detail: Extra numbers.
    :returns: Check dict.
    :rtype: dict
    """
    return {"name": name, "pass": bool(passed) if measured else None, "measured": bool(measured),
            "value": value, "threshold": threshold, "detail": detail}


def _not_measured(name: str, why: str) -> dict:
    """A check whose inputs are missing (counts as FAIL).

    :param name: Check name.
    :param why: Missing input.
    :returns: Check dict.
    :rtype: dict
    """
    return _check(name, None, False, None, None, why)


def _criterion(rule: str, checks: List[dict]) -> dict:
    """Combine sub-checks: PASS only when every check is measured and passes; any measured failure gives FAIL
    (``pass`` False); otherwise a missing input gives NOT MEASURED (``pass`` None), which also counts as FAIL.

    :param rule: Criterion text.
    :param checks: Sub-checks.
    :returns: ``{pass, measured, value, threshold, detail}``.
    :rtype: dict
    """
    measured = bool(checks) and all(c["measured"] for c in checks)
    failed = any(c["measured"] and not c["pass"] for c in checks)
    n_ok = sum(bool(c["pass"]) for c in checks)
    return {"pass": False if failed else (True if measured else None), "measured": measured,
            "value": f"{n_ok}/{len(checks)} checks pass" if checks else "no checks",
            "threshold": rule, "detail": {"checks": checks or [_not_measured("inputs", "no checks could be built")]}}


def _gate(reports: dict, label: Optional[str]) -> Optional[dict]:
    """The gate report of a label, or None.

    :param reports: ``{label: {"gate", "eval_v3", "extra"}}``.
    :param label: Label.
    :returns: Gate report or None.
    :rtype: Optional[dict]
    """
    return (reports.get(label) or {}).get("gate") if label else None


def _family_delta_checks(cand: dict, ref: dict, ref_name: str, families: Sequence[str], metrics: Dict[str, List[str]],
                         delta: Dict[str, float], se_mult: Optional[float], boot: dict, prefix: str) -> List[dict]:
    """H1-style non-inferiority checks per family x metric: delta >= -d and (optionally) delta >= -se_mult * SE.

    :param cand: Candidate gate report.
    :param ref: Reference gate report.
    :param ref_name: Reference role (for names).
    :param families: Families.
    :param metrics: Family -> metric names.
    :param delta: Metric -> tolerance.
    :param se_mult: SE multiple (None: tolerance only).
    :param boot: Bootstrap settings.
    :param prefix: Check-name prefix.
    :returns: Checks.
    :rtype: List[dict]
    """
    out = []
    for fam in families:
        qids = _qids(cand, [fam])
        for m in metrics.get(fam, []):
            name = f"{prefix} {fam} {m} vs {ref_name}"
            d = paired_delta(ref, cand, qids, ("metrics", m), boot)
            if d is None:
                out.append(_not_measured(name, "no paired queries"))
                continue
            tol = delta.get(m, 0.02)
            ok = d["delta"] >= -tol and (se_mult is None or d["delta"] >= -se_mult * d["se"])
            rule = f">= -{tol}" + (f" and >= -{se_mult}*SE" if se_mult is not None else "")
            out.append(_check(name, ok, True, d["delta"], rule, d))
    return out


def _h0(reports: dict, refs: dict, th: dict) -> dict:
    """H0: B (chosen text) vs B3 (v9 text), stock model: every family's primary metric delta >= -delta.

    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H0"]
    b, b3 = _gate(reports, refs.get("B")), _gate(reports, refs.get("B3"))
    rule = "B(chosen text) vs B3 per family primary-metric delta >= -delta"
    if b is None or b3 is None:
        return _criterion(rule, [_not_measured("B vs B3", "B or B3 gate report missing")])
    metrics = {f: [th["primary_metric"][f]] for f in t["families"]}
    return _criterion(rule, _family_delta_checks(b, b3, "B3", t["families"], metrics, t["delta"], None,
                                                 th["bootstrap"], "H0"))


def text_choice(reports: dict, th: dict, b: str = "B", b3: str = "B3", bf: str = "Bf", bnf: str = "Bnf") -> dict:
    """Session-1 text decision: H0 (v4 vs v3 text) and the frames rule (Bf vs Bnf, stock model only).

    Frames are kept when, against Bnf, every family's primary delta >= -delta, canon_share@20 <= Bnf + margin and
    the G3 / G10 deltas are >= 0.

    :param reports: Reports with the stock-model labels.
    :param th: Thresholds.
    :param b: Label of B on the chosen v4 text.
    :param b3: Label of B on the v9 (semantic_v3) text.
    :param bf: Label of B on v4 text with KTIV scholarly frames.
    :param bnf: Label of B on v4 text without frames.
    :returns: ``{H0, frames: "include"|"exclude"|None, frames_checks}``.
    :rtype: dict
    """
    t = th["H0"]
    out = {"H0": _h0(reports, {"B": b, "B3": b3}, th)}
    gf, gnf = _gate(reports, bf), _gate(reports, bnf)
    if gf is None or gnf is None:
        out["frames"], out["frames_checks"] = None, [_not_measured("Bf vs Bnf", "Bf or Bnf gate report missing")]
        return out
    metrics = {f: [th["primary_metric"][f]] for f in t["families"]}
    checks = _family_delta_checks(gf, gnf, "Bnf", t["families"], metrics, t["delta"], None, th["bootstrap"], "frames")
    for fam in t["frames_nonneg_families"]:
        d = paired_delta(gnf, gf, _qids(gf, [fam]), ("metrics", th["primary_metric"][fam]), th["bootstrap"])
        checks.append(_check(f"frames {fam} delta >= 0", d is not None and d["delta"] >= 0, d is not None,
                             d and d["delta"], ">= 0", d))
    qids = _qids(gf, t["frames_canon_families"])
    d = paired_delta(gnf, gf, qids, ("metrics", "canon_share@20"), th["bootstrap"])
    checks.append(_check("frames canon_share@20 <= Bnf + margin", d is not None and d["delta"] <= t["frames_canon_margin"],
                         d is not None, d and d["delta"], f"<= +{t['frames_canon_margin']}", d))
    ok = all(c["measured"] and c["pass"] for c in checks)
    out["frames"], out["frames_checks"] = ("include" if ok else "exclude"), checks
    return out


def _h1(cand: dict, reports: dict, refs: dict, th: dict) -> Tuple[dict, Optional[float]]:
    """H1 family non-inferiority vs B and vs P; also the mean primary delta vs B (checkpoint selection).

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: (criterion, mean primary-metric delta vs B or None).
    :rtype: Tuple[dict, Optional[float]]
    """
    t = th["H1"]
    checks: List[dict] = []
    for role in t["refs"]:
        ref = _gate(reports, refs.get(role))
        if ref is None:
            checks.append(_not_measured(f"H1 vs {role}", f"{role} gate report missing"))
            continue
        checks += _family_delta_checks(cand, ref, role, t["families"], t["metrics"], t["delta"], t["se_mult"],
                                       th["bootstrap"], "H1")
    mean_b = None
    ref_b = _gate(reports, refs.get("B"))
    if ref_b is not None:
        ds = [paired_delta(ref_b, cand, _qids(cand, [f]), ("metrics", th["primary_metric"][f]), th["bootstrap"])
              for f in t["families"]]
        ds = [d["delta"] for d in ds if d is not None]
        mean_b = round(float(np.mean(ds)), 5) if ds else None
    return _criterion("each family x metric, vs B and P: delta >= -delta and >= -2*SE", checks), mean_b


def _cell_qids(rep: dict, families: Sequence[str], key: str) -> List[str]:
    """Scored queries of the families that have a value for population cell ``key``.

    :param rep: Gate report.
    :param families: Families.
    :param key: ``cut=value``.
    :returns: Query ids.
    :rtype: List[str]
    """
    return [q for q in _qids(rep, families) if key in rep["per_query"][q]["cells"]]


def _h2_cells(cand: dict, ref: dict, ref_name: str, t: dict, boot: dict, only: Optional[List[str]] = None) -> List[dict]:
    """H2 population-cell checks against one reference.

    :param cand: Candidate gate report.
    :param ref: Reference gate report.
    :param ref_name: Reference role.
    :param t: H2 thresholds.
    :param boot: Bootstrap settings.
    :param only: Restrict to these ``cut=value`` keys (H7's new-since-v9 cell).
    :returns: Checks.
    :rtype: List[dict]
    """
    keys = sorted({k for p in cand["per_query"].values() if p["family"] in t["families"] for k in p["cells"]})
    if only is not None:
        keys = [k for k in keys if k in only]
    metric_idx = {"AP": 0, "R@100": 1}[t.get("metric", "AP")]
    checks, below_by_cut = [], Counter()
    for key in keys:
        cut = key.split("=", 1)[0]
        if only is None and cut not in t["cuts"]:
            continue
        qids = _cell_qids(cand, t["families"], key)
        n_pairs = sum(cand["per_query"][q]["cells"][key][2] for q in qids)
        if len(qids) < t["min_queries"] and n_pairs < t["min_pairs"]:
            continue
        d = paired_delta(ref, cand, qids, ("cells", key, metric_idx), boot)
        name = f"H2 {key} vs {ref_name}"
        if d is None:
            checks.append(_not_measured(name, "no paired queries"))
            continue
        lim = -max(t["floor"], t["se_mult"] * d["se"])
        hard = d["delta"] < -t["hard_se_mult"] * d["se"] or d["delta"] < -t["hard_floor"]
        if d["delta"] < -t["se_mult"] * d["se"]:
            below_by_cut[cut] += 1
        checks.append(_check(name, d["delta"] >= lim and not hard, True, d["delta"],
                             f">= {lim:.4f}; not < -{t['hard_se_mult']}*SE or < -{t['hard_floor']}",
                             {**d, "n_pairs": n_pairs}))
    for cut, nb in sorted(below_by_cut.items()):
        checks.append(_check(f"H2 cut {cut} vs {ref_name}: cells below -{t['se_mult']}*SE",
                             nb <= t["max_cells_below_se_per_cut"], True, nb,
                             f"<= {t['max_cells_below_se_per_cut']}"))
    return checks


def _h2(cand: dict, reports: dict, refs: dict, th: dict) -> dict:
    """H2: population cells with enough queries / pairs, vs B and vs P.

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H2"]
    checks: List[dict] = []
    for role in t["refs"]:
        ref = _gate(reports, refs.get(role))
        if ref is None:
            checks.append(_not_measured(f"H2 vs {role}", f"{role} gate report missing"))
            continue
        checks += _h2_cells(cand, ref, role, t, th["bootstrap"])
    return _criterion("cells: delta >= -max(0.10, 2SE); <= 1 cell per cut below -2SE; none below -3SE or -0.20",
                      checks)


def _mean_common(ref: dict, cand: dict, qids: Sequence[str], path: Tuple) -> Optional[Tuple[float, float, int]]:
    """Means of a metric over the queries both reports have a value for.

    :param ref: Reference report.
    :param cand: Candidate report.
    :param qids: Query ids.
    :param path: Metric path.
    :returns: (reference mean, candidate mean, n) or None.
    :rtype: Optional[Tuple[float, float, int]]
    """
    pairs = [(_get(ref["per_query"].get(q), path), _get(cand["per_query"].get(q), path)) for q in qids]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    return float(np.mean([a for a, _ in pairs])), float(np.mean([b for _, b in pairs])), len(pairs)


def _h3(cand: dict, reports: dict, refs: dict, th: dict) -> dict:
    """H3 holiday analogues of the handoff's B1-B7, vs P.

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H3"]
    rule = "holiday: strict >= P-1SE; desc-only >= 0.82P; broad >= P; worst festival / bare-name drops; alt lift"
    ref = _gate(reports, refs.get(t["ref"]))
    if ref is None:
        return _criterion(rule, [_not_measured("H3", f"{t['ref']} gate report missing")])
    qids = _qids(cand, t["families"])
    m = t["metric"]
    checks = []
    d = paired_delta(ref, cand, qids, ("metrics", m), th["bootstrap"])
    checks.append(_check("H3 strict " + m, d is not None and d["delta"] >= -t["strict_se_mult"] * d["se"], d is not None,
                         d and d["delta"], f">= -{t['strict_se_mult']}*SE", d))
    mc = _mean_common(ref, cand, qids, ("views", "desc_only", m))
    checks.append(_check("H3 desc-only", mc is not None and mc[1] >= t["desc_only_ratio"] * mc[0], mc is not None,
                         mc and round(mc[1], 4), f">= {t['desc_only_ratio']} x P ({mc and round(mc[0], 4)})"))
    mc = _mean_common(ref, cand, qids, ("views", "broad", m))
    checks.append(_check("H3 broad", mc is not None and mc[1] - mc[0] >= t["broad_min_delta"], mc is not None,
                         mc and round(mc[1] - mc[0], 4), f"delta >= {t['broad_min_delta']}"))
    by_con: Dict[str, List[str]] = defaultdict(list)
    for q in qids:
        by_con[cand["per_query"][q]["concept"]].append(q)
    drops = {}
    for cid, qs in by_con.items():
        mc = _mean_common(ref, cand, qs, ("metrics", m))
        if mc is not None:
            drops[cid] = round(mc[0] - mc[1], 4)
    worst = max(drops.items(), key=lambda kv: kv[1]) if drops else None
    checks.append(_check("H3 worst festival drop", worst is not None and worst[1] <= t["worst_festival_max_drop"],
                         worst is not None, worst and worst[1], f"<= {t['worst_festival_max_drop']}",
                         {"worst": worst and worst[0], "drops": drops}))
    bare = [q for q in qids if str(cand["per_query"][q].get("form")) not in set(t["non_bare_forms"])]
    bare_drops = {}
    for q in bare:
        a, b = _get(ref["per_query"].get(q), ("metrics", m)), _get(cand["per_query"].get(q), ("metrics", m))
        if a is not None and b is not None and a >= t["bare_name_min_ref"]:
            bare_drops[q] = round(a - b, 4)
    wb = max(bare_drops.values()) if bare_drops else 0.0
    checks.append(_check("H3 worst bare-name drop", wb <= t["bare_name_max_drop"], bool(qids), wb,
                         f"<= {t['bare_name_max_drop']} (queries with P >= {t['bare_name_min_ref']})",
                         {"n_qualifying": len(bare_drops)}))
    sel = {q: cand["per_query"][q] for q in qids}
    lift = alt_lift(sel)       # None: no relevant top-10 hit or no alt-only record (vacuous; strict P@10 covers it)
    checks.append(_check("H3 alt-only lift (macro, hits-weighted)",
                         lift is None or lift <= t["alt_lift_macro_max"], bool(qids), lift,
                         f"<= {t['alt_lift_macro_max']} (None = no hit / no alt-only record)"))
    for cid, qs in sorted(by_con.items()):
        if cand["per_query"][qs[0]]["n_rel"] < t["alt_lift_min_relevant"]:
            continue
        lf = alt_lift({q: cand["per_query"][q] for q in qs})
        if lf is not None:
            checks.append(_check(f"H3 alt-only lift {cid}", lf <= t["alt_lift_festival_max"], True, lf,
                                 f"<= {t['alt_lift_festival_max']}"))
    return _criterion(rule, checks)


def _h4(cand: dict, reports: dict, refs: dict, th: dict, h1: dict) -> dict:
    """H4 canon pull vs P, plus the G3 over-correction guard (G3 must pass H1).

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :param h1: H1 criterion (its G3 checks are reused).
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H4"]
    rule = "canon_share@20 <= P+0.05; law_share@20 <= P+0.03; G3 passes H1"
    ref = _gate(reports, refs.get(t["ref"]))
    if ref is None:
        return _criterion(rule, [_not_measured("H4", f"{t['ref']} gate report missing")])
    qids = _qids(cand, t["families"])
    checks = []
    for metric, margin in (("canon_share@20", t["canon_share_margin"]), ("law_share@20", t["law_share_margin"])):
        mc = _mean_common(ref, cand, qids, ("metrics", metric))
        checks.append(_check(f"H4 {metric}", mc is not None and mc[1] <= mc[0] + margin, mc is not None,
                             mc and round(mc[1], 4), f"<= P ({mc and round(mc[0], 4)}) + {margin}"))
    g3 = [c for c in h1["detail"]["checks"] if f" {t['guard_family']} " in c["name"]]
    if g3:
        checks.append(_check(f"H4 {t['guard_family']} passes H1", all(c["pass"] for c in g3),
                             all(c["measured"] for c in g3), sum(bool(c["pass"]) for c in g3), f"{len(g3)} H1 checks"))
    else:
        checks.append(_not_measured(f"H4 {t['guard_family']} passes H1", "no H1 checks for the guard family"))
    return _criterion(rule, checks)


def _h5(cand: dict, reports: dict, refs: dict, th: dict) -> dict:
    """H5 title-position probe: |mean dz| <= 0.25 and <= |B| + 0.10.

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H5"]
    rule = f"{t['direction']} probe: |mean dz| <= {t['max_abs_mean_dz']} and <= |B| + {t['max_excess_over_B']}"

    def get(rep: Optional[dict]) -> Optional[dict]:
        tp = (rep or {}).get("probes", {}).get("title_position") if rep else None
        if not tp or tp.get("text_version") != t["text_version"]:
            return None
        return tp["directions"].get(t["direction"])

    c, b = get(cand), get(_gate(reports, refs.get("B")))
    if c is None or b is None:
        return _criterion(rule, [_not_measured("H5", "title-position probe missing for the candidate or B "
                                                     "(pass doc_vectors_fn and probe_text)")])
    lim = min(t["max_abs_mean_dz"], abs(b["mean_dz"]) + t["max_excess_over_B"])
    return _criterion(rule, [_check("H5 |mean dz|", abs(c["mean_dz"]) <= lim, True, c["mean_dz"], f"|.| <= {lim:.4f}",
                                    {"cand": c, "B": b})])


def _h6(cand: dict, reports: dict, refs: dict, th: dict) -> dict:
    """H6 rank regressions: seed cases at rank <= 50; held-out sweep over pairs where P ranks <= 20.

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H6"]
    rule = (f"seed cases rank <= {t['seed_max_rank']}; sweep: >= {t['sweep_min_share']} within max("
            f"{t['sweep_rank_floor']}, {t['sweep_rank_factor']}x P rank), <= {t['sweep_max_deep_share']} below "
            f"{t['sweep_deep_rank']}")
    checks = []
    cases = {k: v for k, v in cand["probes"]["rank_regressions"].items() if v.get("in_pool")}
    if cases:
        bad = {k: v["rank"] for k, v in cases.items() if v["rank"] > (v.get("max_rank") or t["seed_max_rank"])}
        checks.append(_check("H6 seed cases", not bad, True, f"{len(cases) - len(bad)}/{len(cases)}",
                             f"rank <= {t['seed_max_rank']}", {"failing": bad}))
    else:
        checks.append(_not_measured("H6 seed cases", "no rank-regression case in the pool (seed absent?)"))
    ref = _gate(reports, refs.get(t["sweep_ref"]))
    if ref is None:
        checks.append(_not_measured("H6 sweep", f"{t['sweep_ref']} gate report missing"))
        return _criterion(rule, checks)
    n = ok = deep = 0
    fam_n, fam_below = Counter(), Counter()
    for q, pc in cand["per_query"].items():
        pr = ref["per_query"].get(q)
        if not pr or "pair_ranks" not in pc or "pair_ranks" not in pr:
            continue
        for a, b in zip(pr["pair_ranks"], pc["pair_ranks"]):
            if a is None or b is None or a > t["sweep_max_ref_rank"]:
                continue
            n += 1
            ok += b <= max(t["sweep_rank_floor"], t["sweep_rank_factor"] * a)
            deep += b > t["sweep_deep_rank"]
            fam_n[pc["family"]] += 1
            fam_below[pc["family"]] += b > t["sweep_rank_floor"]
    if n < t["sweep_min_pairs"]:
        checks.append(_not_measured("H6 sweep", f"only {n} held-out pairs with P rank <= {t['sweep_max_ref_rank']}"))
        return _criterion(rule, checks)
    checks.append(_check("H6 sweep within bound", ok / n >= t["sweep_min_share"], True, round(ok / n, 4),
                         f">= {t['sweep_min_share']}", {"n_pairs": n}))
    checks.append(_check("H6 sweep deep drops", deep / n <= t["sweep_max_deep_share"], True, round(deep / n, 4),
                         f"<= {t['sweep_max_deep_share']}", {"n_pairs": n}))
    if t.get("family_max_below_floor_share") is not None:
        for f in sorted(fam_n):
            share = fam_below[f] / fam_n[f]
            checks.append(_check(f"H6 sweep {f} below rank {t['sweep_rank_floor']}",
                                 share <= t["family_max_below_floor_share"], True, round(share, 4),
                                 f"<= {t['family_max_below_floor_share']}"))
    return _criterion(rule, checks)


def _h7(cand: dict, reports: dict, refs: dict, th: dict) -> dict:
    """H7 population: per source group R@100 ratio vs P and B; KTIV lift@10; new-since-v9 cell passes H2.

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H7"]
    rule = (f"per source group R@100 ratio vs P and B >= {t['min_r100_ratio']}; {t['lift_group']} lift@10 <= P + "
            f"{t['lift_margin']}; {t['new_cell']} passes H2")
    checks = []
    groups = sorted({k.split("=", 1)[1] for p in cand["per_query"].values() for k in p["cells"]
                     if k.startswith("source_group=")})
    for role in t["refs"]:
        ref = _gate(reports, refs.get(role))
        if ref is None:
            checks.append(_not_measured(f"H7 vs {role}", f"{role} gate report missing"))
            continue
        for g in groups:
            key = f"source_group={g}"
            qids = _cell_qids(cand, t["families"], key)
            if len(qids) < t["min_group_queries"]:
                continue
            mc = _mean_common(ref, cand, qids, ("cells", key, 1))
            if mc is None:
                checks.append(_not_measured(f"H7 R@100 {g} vs {role}", "no paired queries"))
                continue
            ratio = mc[1] / mc[0] if mc[0] > 0 else (1.0 if mc[1] >= mc[0] else 0.0)
            checks.append(_check(f"H7 R@100 {g} vs {role}", ratio >= t["min_r100_ratio"], True, round(ratio, 4),
                                 f">= {t['min_r100_ratio']}", {"ref": round(mc[0], 4), "cand": round(mc[1], 4),
                                                               "n_queries": mc[2]}))
    ref_p = _gate(reports, refs.get("P"))
    if ref_p is None:
        checks.append(_not_measured("H7 lift", "P gate report missing"))
        checks.append(_not_measured("H7 new cell", "P gate report missing"))
        return _criterion(rule, checks)
    qids = [q for q in _qids(cand, t["lift_families"]) if q in ref_p["per_query"]]
    lc = group_lift({q: cand["per_query"][q] for q in qids}, t["lift_group"])
    lp = group_lift({q: ref_p["per_query"][q] for q in qids}, t["lift_group"])
    checks.append(_check(f"H7 {t['lift_group']} lift@10", lc is not None and lp is not None and lc <= lp + t["lift_margin"],
                         lc is not None and lp is not None, lc, f"<= P ({lp}) + {t['lift_margin']}"))
    h2t = dict(th["H2"])
    new_checks = []
    for role in h2t["refs"]:
        ref = _gate(reports, refs.get(role))
        if ref is not None:
            cells = _h2_cells(cand, ref, role, h2t, th["bootstrap"], only=[t["new_cell"]])
            new_checks += [c for c in cells if not c["name"].startswith("H2 cut")]
    if new_checks:
        checks.append(_check(f"H7 {t['new_cell']} passes H2", all(c["pass"] for c in new_checks),
                             all(c["measured"] for c in new_checks), [c["value"] for c in new_checks],
                             "H2 cell rule", new_checks))
    else:
        checks.append(_not_measured(f"H7 {t['new_cell']} passes H2", "cell absent or below the H2 minimum size"))
    return _criterion(rule, checks)


def _ev3(reports: dict, label: Optional[str]) -> Optional[dict]:
    """The eval_v3 result of a label, or None.

    :param reports: Reports.
    :param label: Label.
    :returns: eval_v3 result or None.
    :rtype: Optional[dict]
    """
    return (reports.get(label) or {}).get("eval_v3") if label else None


def _ki_by_type(ev: dict, types: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    """Known-item reciprocal ranks per target record for the given query types (eval_v3 per_query lists).

    :param ev: eval_v3 result.
    :param types: Query types.
    :returns: ``target doc -> [RR, ...]`` in query order.
    :rtype: Dict[str, Dict[str, List[float]]]
    """
    pq = ev["known_item"]["per_query"]
    out: Dict[str, List[float]] = defaultdict(list)
    for d, ty, r in zip(pq["doc_id"], pq["type"], pq["rank"]):
        if ty in types:
            out[d].append(1.0 / r)
    return out


def _h8(reports: dict, cand_label: str, refs: dict, th: dict) -> dict:
    """H8 keep v9's gains (eval_v3 results): focus AP, known-item MRR by type, seen-subject AP.

    :param reports: Reports.
    :param cand_label: Candidate label.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H8"]
    boot = th["bootstrap"]
    rule = "focus AP CI vs V not all < 0; KI MRR vs V; he/translit vs V and CI-low > 0 vs B; seen AP; seen worse share"
    c, v, b = _ev3(reports, cand_label), _ev3(reports, refs.get("V")), _ev3(reports, refs.get("B"))
    if c is None or v is None or b is None:
        return _criterion(rule, [_not_measured("H8", "eval_v3 result missing for the candidate, V or B")])
    checks = []
    fa = (v.get("subjects", {}).get("focus") or {}).get("per_query")
    fc = (c.get("subjects", {}).get("focus") or {}).get("per_query")
    if fa and fc:
        d = two_stage_bootstrap({s: x["AP"] for s, x in fa.items()}, {s: x["AP"] for s, x in fc.items()},
                                boot["n_resamples"], boot["seed"], boot["alpha"])
        checks.append(_check("H8 focus AP vs V", d["ci_high"] >= 0, d["n_queries"] > 0, d["delta"],
                             "CI not entirely below 0", d))
    else:
        checks.append(_not_measured("H8 focus AP vs V", "focus group missing"))
    def mrr_drop(ty: str, max_drop: float) -> dict:
        mv = v["known_item"]["by_type"].get(ty, {}).get("MRR")
        mc = c["known_item"]["by_type"].get(ty, {}).get("MRR")
        ok = mv is not None and mc is not None
        return _check(f"H8 KI {ty} MRR vs V", ok and mc - mv >= -max_drop, ok, round(mc - mv, 4) if ok else None,
                      f">= -{max_drop}")

    for ty in t["ki_types_vs_v"]:
        checks.append(mrr_drop(ty, t["ki_max_drop_vs_v"]))
    for ty in t["ki_he_types"]:
        checks.append(mrr_drop(ty, t["ki_he_max_drop_vs_v"]))
        rb, rc = _ki_by_type(b, [ty]), _ki_by_type(c, [ty])
        d = two_stage_bootstrap(rb, rc, boot["n_resamples"], boot["seed"], boot["alpha"]) if rb and rc else None
        checks.append(_check(f"H8 KI {ty} RR vs B CI-low > 0", d is not None and d["ci_low"] > 0, d is not None,
                             d and d["delta"], "ci_low > 0", d))
    sv = v.get("subjects", {}).get("seen", {}).get("macro", {}).get("AP")
    sc = c.get("subjects", {}).get("seen", {}).get("macro", {}).get("AP")
    checks.append(_check("H8 seen AP vs V", sv is not None and sc is not None and sc - sv >= -t["seen_ap_max_drop_vs_v"],
                         sv is not None and sc is not None, sv is not None and sc is not None and round(sc - sv, 4),
                         f">= -{t['seen_ap_max_drop_vs_v']}"))
    pb = b.get("subjects", {}).get("seen", {}).get("per_subject", {})
    pc = c.get("subjects", {}).get("seen", {}).get("per_subject", {})
    common = sorted(set(pb) & set(pc))
    if common:
        worse = [s for s in common if pc[s]["AP"] - pb[s]["AP"] < -t["seen_worse_margin"]]
        share = len(worse) / len(common)
        checks.append(_check("H8 seen subjects worse vs B", share <= t["seen_worse_max_share"], True, round(share, 4),
                             f"<= {t['seen_worse_max_share']}", {"n_worse": len(worse), "n": len(common),
                                                                 "worst": sorted(worse)[:20]}))
    else:
        checks.append(_not_measured("H8 seen subjects worse vs B", "seen per-subject results missing"))
    return _criterion(rule, checks)


def _h9(cand: dict, reports: dict, cand_label: str, refs: dict, th: dict) -> dict:
    """H9 structure: collection lift, duplicate/short share, top hubs, distinct@10, memorisation net gap.

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param cand_label: Candidate label.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H9"]
    rule = "collection lift <= B+0.10; dup/short <= B; top hub <= max(B, 0.02); distinct@10 >= B-0.3; memo <= min(V, 0.04)"
    checks = []
    c, b = _ev3(reports, cand_label), _ev3(reports, refs.get("B"))
    gb = _gate(reports, refs.get("B"))
    if c is None or b is None:
        checks.append(_not_measured("H9 eval_v3", "eval_v3 result missing for the candidate or B"))
    else:
        lc = (c.get("collection") or {}).get("all", {}).get("lift")
        lb = (b.get("collection") or {}).get("all", {}).get("lift")
        ok = lc is not None and lb is not None
        checks.append(_check("H9 collection lift", ok and lc <= lb + t["collection_lift_margin"], ok, lc,
                             f"<= B ({lb}) + {t['collection_lift_margin']}"))
        hc, hb = (c.get("hubness") or {}).get("share"), (b.get("hubness") or {}).get("share")
        checks.append(_check("H9 duplicate/short share", hc is not None and hb is not None and hc <= hb,
                             hc is not None and hb is not None, hc, f"<= B ({hb})"))
        tc = ((c.get("hubness") or {}).get("top_hubs") or [{}])[0].get("share_of_queries")
        tb = ((b.get("hubness") or {}).get("top_hubs") or [{}])[0].get("share_of_queries")
        checks.append(_check("H9 top hub (eval_v3 subject queries)",
                             tc is not None and tb is not None and tc <= max(tb, t["top_hub_floor"]),
                             tc is not None and tb is not None, tc, f"<= max(B ({tb}), {t['top_hub_floor']})"))
        mt_c, mt_b = c.get("memorisation_trained"), b.get("memorisation_trained")
        if mt_c and mt_b:
            d = two_stage_bootstrap(mt_b["per_query_gap"], mt_c["per_query_gap"], th["bootstrap"]["n_resamples"],
                                    th["bootstrap"]["seed"], th["bootstrap"]["alpha"])
            v_gap = ((reports.get(refs.get("V")) or {}).get("extra") or {}).get("memorisation_net_gap", t["v9_net_gap"])
            lim = min(v_gap, t["memorisation_cap"])
            checks.append(_check("H9 memorisation net gap", d["delta"] <= lim, d["n_queries"] > 0, d["delta"],
                                 f"<= min(V {v_gap}, {t['memorisation_cap']})", d))
        else:
            checks.append(_not_measured("H9 memorisation net gap", "memorisation_trained missing (pass "
                                                                   "trained_subject_positives to eval_v3)"))
    if gb is None:
        checks.append(_not_measured("H9 gate structure", "B gate report missing"))
    else:
        tc, tb = cand["structure"].get("top_hub_share"), gb["structure"].get("top_hub_share")
        ok = tc is not None and tb is not None
        checks.append(_check("H9 top hub (gate queries)", ok and tc <= max(tb, t["top_hub_floor"]), ok, tc,
                             f"<= max(B ({tb}), {t['top_hub_floor']})", cand["structure"].get("top_hubs")))
        dc, db = cand["structure"].get("distinct@10"), gb["structure"].get("distinct@10")
        checks.append(_check("H9 distinct@10", dc is not None and db is not None and dc >= db - t["distinct10_margin"],
                             dc is not None and db is not None, dc, f">= B ({db}) - {t['distinct10_margin']}"))
    return _criterion(rule, checks)


def _paired_blocks(node, path: str = "") -> List[Tuple[str, dict]]:
    """Every ``paired_*`` bootstrap dict inside a bib_v9_check report section (recursive).

    :param node: Section (dict / list).
    :param path: Path so far.
    :returns: (path, bootstrap dict) pairs.
    :rtype: List[Tuple[str, dict]]
    """
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k.startswith("paired") and isinstance(v, dict) and "ci_high" in v:
                out.append((f"{path}/{k}", v))
            else:
                out += _paired_blocks(v, f"{path}/{k}")
    return out


def _h10(reports: dict, cand_label: str, th: dict) -> dict:
    """H10 bibliography (``extra["bib"]`` = bib_v9_check score report): no paired CI entirely below 0.

    :param reports: Reports.
    :param cand_label: Candidate label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H10"]
    rule = "bib_v9_check: no paired CI vs base entirely below 0 (dense, 60/40, 30/70, chat 50/50)"
    bib = ((reports.get(cand_label) or {}).get("extra") or {}).get("bib")
    if not bib:
        return _criterion(rule, [_not_measured("H10", "extra['bib'] missing (run bib_v9_check score)")])
    checks = []
    for sec in t["sections"]:
        node = (bib.get("known_item") or {}).get(sec, bib.get(sec))
        blocks = _paired_blocks(node, sec) if node else []
        if not blocks:
            checks.append(_not_measured(f"H10 {sec}", "section missing"))
            continue
        harm = {p: b["ci_high"] for p, b in blocks if b["ci_high"] < 0}
        checks.append(_check(f"H10 {sec}", not harm, True, f"{len(blocks) - len(harm)}/{len(blocks)}",
                             "ci_high >= 0", {"harm": harm}))
    return _criterion(rule, checks)


def _h11(cand: dict, reports: dict, cand_label: str, refs: dict, th: dict) -> dict:
    """H11 integrity: text hashes, every query scored, minimum sizes, same gate files, no blocked-query leak.

    :param cand: Candidate gate report.
    :param reports: Reports.
    :param cand_label: Candidate label.
    :param refs: Role -> label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H11"]
    rule = "text-hash mismatch 0; every query scored; minimum sizes; same gate files; no blocked query in the mix"
    checks = []
    mism = [x for x in (cand["meta"].get("text_hash_mismatch"),
                        ((_ev3(reports, cand_label) or {}).get("pool") or {}).get("text_hash_mismatch")) if x is not None]
    checks.append(_check("H11 text-hash mismatch", bool(mism) and max(mism) <= t["max_text_hash_mismatch"], bool(mism),
                         max(mism) if mism else None, f"<= {t['max_text_hash_mismatch']}"))
    checks.append(_check("H11 every query scored", cand["meta"]["n_unscorable"] == 0, True, cand["meta"]["n_unscorable"],
                         "0 unscorable", cand["meta"].get("unscorable")))
    for fam, mn in sorted(t["min_queries"].items()):
        n = cand["families"].get(fam, {}).get("n_queries", 0)
        checks.append(_check(f"H11 size {fam}", n >= mn, True, n, f">= {mn}"))
    shas = {lb: (r.get("gate") or {}).get("meta", {}).get("files_sha256") for lb, r in reports.items()
            if lb == cand_label or lb in set(refs.values())}
    want = cand["meta"]["files_sha256"]
    diff = sorted(lb for lb, s in shas.items() if s is not None and s != want)
    checks.append(_check("H11 same gate files", not diff, True, diff or "all equal", "identical files_sha256"))
    leak = ((reports.get(cand_label) or {}).get("extra") or {}).get("leak")
    if leak:
        checks.append(_check("H11 blocked queries not in the mix", leak["exact"] == 0 and leak["contained"] == 0, True,
                             {"exact": leak["exact"], "contained": leak["contained"]}, "0 exact, 0 contained",
                             leak.get("examples")))
    else:
        checks.append(_not_measured("H11 blocked queries not in the mix", "extra['leak'] missing "
                                                                          "(blocked_query_leaks(gate_dir, anchors))"))
    return _criterion(rule, checks)


def _h12(reports: dict, cand_label: str, th: dict) -> dict:
    """H12 leave-concept-out (``extra["lco"] = {"cand": eval_v3 LCO result, "base": B's eval_v3 on the LCO dir}``).

    :param reports: Reports.
    :param cand_label: Candidate label.
    :param th: Thresholds.
    :returns: Criterion.
    :rtype: dict
    """
    t = th["H12"]
    rule = f"held-out subject AP: CI-low > 0 vs B and >= {t['ap_floor']} - {t['se_mult']} SE"
    lco = ((reports.get(cand_label) or {}).get("extra") or {}).get("lco")
    if not lco or not lco.get("cand") or not lco.get("base"):
        return _criterion(rule, [_not_measured("H12", "extra['lco'] = {'cand', 'base'} missing")])
    pc = lco["cand"]["subjects"]["held_out"]["per_query"]
    pb = lco["base"]["subjects"]["held_out"]["per_query"]
    boot = th["bootstrap"]
    d = two_stage_bootstrap({s: x["AP"] for s, x in pb.items()}, {s: x["AP"] for s, x in pc.items()},
                            boot["n_resamples"], boot["seed"], boot["alpha"])
    gate_ap = (lco["cand"]["subjects"].get("gate") or {}).get("AP")
    checks = [_check("H12 CI-low vs B", d["ci_low"] > 0, d["n_queries"] > 0, d["delta"], "ci_low > 0", d),
              _check("H12 held-out AP floor", gate_ap is not None and gate_ap >= t["ap_floor"] - t["se_mult"] * d["se"],
                     gate_ap is not None, gate_ap, f">= {t['ap_floor']} - {t['se_mult']}*SE ({d['se']})")]
    return _criterion(rule, checks)


def verdict(reports: Dict[str, dict], cand: str, refs: Dict[str, str], thresholds: dict) -> dict:
    """Apply the hard criteria H0-H12 (plan section E) to one candidate.

    :param reports: ``{label: {"gate": evaluate_gate report, "eval_v3": eval_v3.evaluate result or None,
        "extra": {"bib": bib_v9_check report, "lco": {"cand", "base"}, "leak": blocked_query_leaks(...),
        "memorisation_net_gap": float (V only, optional)}}}``.
    :param cand: Candidate label.
    :param refs: ``{"P": label, "B": label, "B3": label, "V": label}``.
    :param thresholds: The frozen ``thresholds.json`` (no fallback to defaults).
    :returns: ``{"pass": bool, "criteria": {id: {pass, measured, value, threshold, detail}}, "summary": {...}}``;
        NOT MEASURED counts as FAIL.
    :rtype: dict
    """
    th = thresholds
    c = _gate(reports, cand)
    crit: Dict[str, dict] = {"H0": _h0(reports, refs, th)}
    if c is None:
        for k in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H9", "H11"):
            crit[k] = _criterion(k, [_not_measured(k, "candidate gate report missing")])
        mean_b = None
    else:
        crit["H1"], mean_b = _h1(c, reports, refs, th)
        crit["H2"] = _h2(c, reports, refs, th)
        crit["H3"] = _h3(c, reports, refs, th)
        crit["H4"] = _h4(c, reports, refs, th, crit["H1"])
        crit["H5"] = _h5(c, reports, refs, th)
        crit["H6"] = _h6(c, reports, refs, th)
        crit["H7"] = _h7(c, reports, refs, th)
        crit["H9"] = _h9(c, reports, cand, refs, th)
        crit["H11"] = _h11(c, reports, cand, refs, th)
    crit["H8"] = _h8(reports, cand, refs, th)
    crit["H10"] = _h10(reports, cand, th)
    crit["H12"] = _h12(reports, cand, th)
    order = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12"]
    crit = {k: crit[k] for k in order}
    soft = {}
    gb = _gate(reports, refs.get("B"))
    if c is not None and gb is not None:
        sc, sb = c["structure"].get("score_spread"), gb["structure"].get("score_spread")
        soft["score_spread_ratio"] = round(sc / sb, 4) if sc and sb else None
        soft["ood_auc"] = {"cand": c["structure"].get("ood_auc"), "B": gb["structure"].get("ood_auc")}
        soft["typo"] = {"cand": c["soft"].get("typo"), "B": gb["soft"].get("typo")}
        soft["drift"] = {k: {"cand": c["structure"].get(k), "B": gb["structure"].get(k)}
                         for k in ("random_doc_cosine", "participation_ratio", "top10_eig_share")}
    return {"pass": all(v["pass"] is True for v in crit.values()), "candidate": cand, "refs": dict(refs),
            "criteria": crit, "summary": {"h1_mean_delta_vs_B": mean_b,
                                          "failing": [k for k, v in crit.items() if v["pass"] is not True]},
            "soft": soft}


def select_checkpoint(verdicts: Dict[str, dict], order: Sequence[str], tie: float = 0.005) -> Optional[str]:
    """Pre-registered selection: among passing checkpoints, the highest mean H1 delta vs B; within ``tie`` of the best,
    the earliest in ``order`` (list checkpoints in step order, Arm A before Arm B at equal steps).

    :param verdicts: ``label -> verdict()``.
    :param order: Preference order of the labels.
    :param tie: Tie band.
    :returns: Selected label, or None when nothing passes (do not index).
    :rtype: Optional[str]
    """
    ok = [lb for lb in order if verdicts.get(lb, {}).get("pass") and
          verdicts[lb]["summary"]["h1_mean_delta_vs_B"] is not None]
    if not ok:
        return None
    best = max(verdicts[lb]["summary"]["h1_mean_delta_vs_B"] for lb in ok)
    return next(lb for lb in ok if verdicts[lb]["summary"]["h1_mean_delta_vs_B"] >= best - tie)


def format_verdict(v: dict, show_checks: str = "failing") -> str:
    """Readable table of a verdict.

    :param v: :func:`verdict` output.
    :param show_checks: ``failing`` (checks that fail or are not measured), ``all`` or ``none``.
    :returns: Multi-line text.
    :rtype: str
    """
    def status(x: Optional[bool], measured: bool) -> str:
        return "PASS" if x is True else ("FAIL" if x is False else "NOT MEASURED")

    lines = [f"gate_v1 verdict for {v.get('candidate')}: {'PASS' if v['pass'] else 'FAIL'}  refs={v.get('refs')}",
             f"{'id':<5} {'status':<13} {'value':<22} threshold"]
    for k, c in v["criteria"].items():
        lines.append(f"{k:<5} {status(c['pass'], c['measured']):<13} {str(c['value']):<22} {c['threshold']}")
        if show_checks == "none":
            continue
        for ch in c["detail"]["checks"]:
            if show_checks == "all" or ch["pass"] is not True:
                det = ch["detail"] if isinstance(ch["detail"], str) else ""
                lines.append(f"      - {status(ch['pass'], ch['measured']):<13} {ch['name']}: {ch['value']} "
                             f"({ch['threshold']}) {det}".rstrip())
    s = v.get("summary", {})
    lines.append(f"mean H1 delta vs B: {s.get('h1_mean_delta_vs_B')}; failing: {s.get('failing')}")
    if v.get("soft"):
        lines.append(f"soft: {json.dumps(v['soft'], ensure_ascii=False)}")
    return "\n".join(lines)


def format_report(rep: dict) -> str:
    """Short text summary of one gate report (families, groups, structure).

    :param rep: :func:`evaluate_gate` output.
    :returns: Multi-line text.
    :rtype: str
    """
    lines = [f"pool {rep['meta']['n_pool_scored']}/{rep['meta']['n_pool_frozen']}, queries scored "
             f"{rep['meta']['n_scored']} (unscorable {rep['meta']['n_unscorable']}, report-only "
             f"{rep['meta']['n_report_only']}), {rep['meta']['seconds']} s"]
    for f, m in rep["families"].items():
        lines.append(f"{f:<4} n={m['n_queries']:<5} P@10_dd={m['P@10_dd']} AP={m['AP']} RR={m['RR']} "
                     f"canon@20={m['canon_share@20']} alt_lift={m.get('alt_lift')}")
    for g, m in rep["groups"].items():
        lines.append(f"group {g:<6} R@100={m.get('R@100')} lift@10={m.get('lift@10')} n={m.get('n_queries')}")
    st = rep["structure"]
    lines.append(f"structure: distinct@10={st.get('distinct@10')} top_hub={st.get('top_hub_share')} "
                 f"spread={st.get('score_spread')} doc-doc={st.get('random_doc_cosine')} "
                 f"PR={st.get('participation_ratio')} ood_auc={st.get('ood_auc')}")
    tp = rep["probes"]["title_position"]
    lines.append(f"title position: {json.dumps(tp) if tp else 'not run'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------------------------
def _write_gate_dir(path: Path, gd: dict) -> None:
    """Write an in-memory gate dir (self-test helper).

    :param path: Target directory.
    :param gd: Dict keyed like :data:`GATE_FILES`.
    """
    path.mkdir(parents=True, exist_ok=True)
    for key, name in GATE_FILES.items():
        if name.endswith(".jsonl"):
            with open(path / name, "w", encoding="utf-8") as fh:
                for row in gd[key]:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            (path / name).write_text(json.dumps(gd[key], ensure_ascii=False), encoding="utf-8")


def _brute_ranks(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Reference pessimistic ranks via a full sort.

    :param pos: Positive scores.
    :param neg: Negative scores.
    :returns: Ascending ranks.
    :rtype: np.ndarray
    """
    scores = np.concatenate([pos, neg])
    rel = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.lexsort((rel, -scores))
    return np.sort(np.where(rel[order] == 1)[0] + 1)


def _synthetic_gate(rng: np.random.Generator, n: int = 900, d: int = 48) -> Tuple[dict, dict]:
    """A synthetic gate dir plus the latent geometry used to plant models.

    :param rng: RNG.
    :param n: Pool size.
    :param d: Vector dimension.
    :returns: (gate dict keyed like :data:`GATE_FILES`, latent dict).
    :rtype: Tuple[dict, dict]
    """
    groups = np.array(["KTIV", "FJP", "PGP", "other"])[rng.choice(4, size=n, p=[0.35, 0.3, 0.3, 0.05])]
    langs = np.array(["he", "en", "ar", "mixed"])[rng.choice(4, size=n, p=[0.4, 0.45, 0.05, 0.1])]
    buckets = np.array(["lt100", "100-300", "300-1k", "ge1k"])[rng.integers(0, 4, size=n)]
    ids = [f"doc{i:04d}" for i in range(n)]
    hashes = [f"h{i}" for i in range(n)]
    for i in range(0, 40, 2):
        hashes[i + 1] = hashes[i]
    records = [{"doc_id": ids[i], "source_group": str(groups[i]), "desc_lang": str(langs[i]),
                "len_bucket": str(buckets[i]), "held_out": i % 4 == 0, "new_since_v9": i % 6 == 0,
                "has_alt_titles": groups[i] == "KTIV", "text_hash": hashes[i],
                "canon_kind": "tractate" if i % 3 == 0 else "bible", "v9_trained": i % 9 == 1} for i in range(n)]
    free = list(rng.permutation(np.arange(40, n)))
    concepts, queries, owner = {}, [], {}
    forms = ["english", "translit", "hebrew", "typo", "descriptive"]
    fam_sizes = {"G1": 4, "G2": 3, "G3": 3, "G4": 3, "G5": 3, "G6": 3, "G7": 2}
    k = 0
    for fam, nc in fam_sizes.items():
        for j in range(nc):
            cid = f"{fam.lower()}_c{j}"
            rel = [free.pop() for _ in range(32)]
            canon = [free.pop() for _ in range(14)] if fam == "G1" else []
            unj = [free.pop() for _ in range(5)]
            alt = [x for x in rel if groups[x] == "KTIV"][:8]
            concepts[cid] = {"family": fam, "relevant": sorted(ids[x] for x in rel),
                             "canon": sorted(ids[x] for x in canon), "unjudged": sorted(ids[x] for x in unj),
                             "alt_only": sorted(ids[x] for x in alt),
                             "held_out_relevant": sorted(ids[x] for x in rel if x % 4 == 0)}
            owner[cid] = {"k": k, "rel": rel, "canon": canon, "unj": unj}
            for qi in range(5):
                text = f"{cid} query {qi}" + (" שבת" if qi == 2 else "")
                queries.append({"qid": f"{fam}:{cid}:{qi}", "text": text, "family": fam, "concept": cid,
                                "form": forms[qi], "blocked": qi != 0, "lang": query_lang(text), "source": "spec",
                                "report_only": False})
            k += 1
    for t in range(30):                                   # known-item: target + its twin for the first ones
        x = free.pop() if t >= 5 else 2 * t
        rel = [ids[x]] + ([ids[x + 1]] if t < 5 else [])
        cid = f"ki:{ids[x]}"
        concepts[cid] = {"family": "G10", "relevant": rel, "canon": [], "unjudged": [], "alt_only": [],
                         "held_out_relevant": []}
        owner[cid] = {"k": None, "doc": x}
        queries.append({"qid": f"ki:{t}", "text": f"known item {t}", "family": "G10", "concept": cid,
                        "form": "specific", "blocked": True, "lang": "latin", "source": "known_item",
                        "report_only": False})
    for t in range(6):
        queries.append({"qid": f"ood:{t}", "text": f"ood {t}", "family": "G12", "concept": None, "form": "ood",
                        "blocked": False, "lang": "latin", "source": "spec", "report_only": True})
    g1 = [c for c in concepts if concepts[c]["family"] == "G1"]
    seeds = [{"case_id": f"seed{t}", "text": f"{g1[t % len(g1)]} query 0",
              "doc_id": concepts[g1[t % len(g1)]]["relevant"][t], "max_rank": 50} for t in range(4)]
    probes = []
    for cid in g1:
        for x in owner[cid]["rel"][:6]:
            probes.append({"pid": f"tp:{ids[x]}", "doc_id": ids[x], "concept": cid, "direction": "tail_to_head",
                           "text_version": "semantic_v4", "kind": "position",
                           "qids": [q["qid"] for q in queries if q["concept"] == cid][:3],
                           "orig_text": f"orig {ids[x]}", "variant_text": f"variant {ids[x]}"})
    th = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    th["bootstrap"]["n_resamples"] = 400
    th["H11"]["min_queries"] = {"G1": 15, "G2": 10, "G3": 10, "G10": 25}
    th["H6"]["sweep_min_pairs"] = 20
    th["H3"]["alt_lift_min_relevant"] = 30
    gd = {"queries": queries, "concepts": concepts, "records": records,
          "probes": {"rank_regressions": seeds, "title_position": probes}, "thresholds": th}
    latent = {"ids": ids, "owner": owner, "centres": normalise_rows(rng.normal(size=(k, d))), "d": d,
              "base": normalise_rows(rng.normal(size=(n, d))), "q_noise": normalise_rows(rng.normal(size=(len(queries), d)))}
    return gd, latent


def _plant(latent: dict, gd: dict, rel_w: float, canon_w: float, noise: float, seed: int) -> Tuple[np.ndarray, Callable]:
    """Document matrix + query function of a planted model.

    :param latent: Latent geometry.
    :param gd: Gate dict.
    :param rel_w: Weight of the concept centre on relevant records.
    :param canon_w: Weight of the concept centre on canon records (G1).
    :param noise: Weight of record-specific noise.
    :param seed: RNG seed of the model-specific noise.
    :returns: (doc matrix, query function).
    :rtype: Tuple[np.ndarray, Callable]
    """
    rng = np.random.default_rng(seed)
    n, d = len(latent["ids"]), latent["d"]
    M = noise * latent["base"] + 0.15 * normalise_rows(rng.normal(size=(n, d)))
    C = latent["centres"]
    for cid, o in latent["owner"].items():
        if o["k"] is None:
            continue
        for x in o["rel"]:
            M[x] += rel_w * C[o["k"]]
        for x in o["canon"]:
            M[x] += canon_w * C[o["k"]]
        for x in o["unj"]:
            M[x] += 2.0 * C[o["k"]]                   # unjudged records sit on the query: must not matter
    for i in range(0, 40, 2):
        M[i + 1] = M[i]
    M = normalise_rows(M)
    qv = {}
    for i, q in enumerate(gd["queries"]):
        o = latent["owner"].get(q["concept"]) if q["concept"] else None
        if o is None:
            qv[q["text"]] = latent["q_noise"][i]
        elif o["k"] is None:
            qv[q["text"]] = M[o["doc"]] + 0.25 * latent["q_noise"][i]
        else:
            qv[q["text"]] = C[o["k"]] + 0.35 * latent["q_noise"][i]

    def qfn(texts: List[str]) -> np.ndarray:
        return np.stack([qv[t] for t in texts])

    return M, qfn


def _fake_eval_v3(rng: np.random.Generator, q: float) -> dict:
    """Minimal eval_v3-shaped result of quality ``q`` (self-test of H8 / H9 / H12 plumbing).

    :param rng: RNG.
    :param q: Quality knob (higher = better).
    :returns: eval_v3-like dict.
    :rtype: dict
    """
    subs = [f"topic:s{i}" for i in range(6)]
    seen = {f"work:w{i}": {"AP": round(0.3 + 0.1 * q + 0.01 * i, 4)} for i in range(50)}
    types = ["specific", "topical", "implicit", "hebrew", "translit"]
    pq = {"qid": [], "doc_id": [], "type": [], "rank": []}
    for t in types:
        for j in range(40):
            pq["qid"].append(f"{t}{j}")
            pq["doc_id"].append(f"doc{j}")
            pq["type"].append(t)
            pq["rank"].append(max(1, int(round(8 / q)) + (j % 3)))
    by_type = {t: {"MRR": round(float(np.mean([1 / r for r, ty in zip(pq["rank"], pq["type"]) if ty == t])), 4)}
               for t in types}
    return {"pool": {"text_hash_mismatch": 0},
            "subjects": {"focus": {"per_query": {s: {"AP": [round(0.1 * q + 0.01 * j, 4) for j in range(5)]}
                                                 for s in subs}},
                         "seen": {"macro": {"AP": round(float(np.mean([v["AP"] for v in seen.values()])), 4)},
                                  "per_subject": seen},
                         "held_out": {"per_query": {s: {"AP": [round(0.1 * q + 0.02 * j, 4) for j in range(5)]}
                                                    for s in subs}},
                         "gate": {"AP": round(0.1 * q + 0.04, 4)}},
            "known_item": {"by_type": by_type, "per_query": pq},
            "memorisation_trained": {"per_query_gap": {s: [round(0.002 * q + 0.001 * j, 4) for j in range(4)]
                                                       for s in subs}},
            "collection": {"all": {"lift": 2.8}},
            "hubness": {"share": round(0.2 - 0.02 * q, 4), "top_hubs": [{"share_of_queries": 0.025}]}}


def self_test(seed: int = 0) -> None:
    """Synthetic checks: rank primitives and dd collapse, bootstrap, a good model passes every criterion, a canon-
    pulling model fails H4 (and H1 / H3), a position-sensitive model fails H5, missing extras are NOT MEASURED (FAIL),
    the leak check, the selection rule.

    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    for _ in range(200):                                      # 1. rank primitive vs brute force with ties
        pos = rng.integers(0, 6, size=rng.integers(1, 8)).astype(float)
        neg = rng.integers(0, 6, size=rng.integers(0, 30)).astype(float)
        other = rng.integers(0, 6, size=rng.integers(0, 5)).astype(float)   # other relevant: out of the ranking
        kept = np.sort(np.concatenate([pos, neg, other]))
        assert np.array_equal(rank_positives(pos, kept, np.sort(np.concatenate([pos, other]))),
                              _brute_ranks(pos, neg)), (pos, neg, other)
    s = np.array([0.9, 0.9, 0.8, 0.8, 0.7, 0.1])
    is_rel = np.array([True, False, True, False, True, False])
    top = pessimistic_top(s, None, is_rel, 6)
    assert top[0] == 1 and top[2] == 3, top                    # tied irrelevant record first
    hid = np.array([0, 0, 1, 2, 3, 4])
    assert list(dedup_slots(top, hid, 3)) == [1, 3, 2], dedup_slots(top, hid, 3)
    same = cluster_bootstrap({"a": [0.0, 0.0], "b": [0.0]}, 100)
    assert same["delta"] == 0 and same["se"] == 0
    up = cluster_bootstrap({f"c{i}": [0.1 + 0.01 * i] for i in range(20)}, 300)
    assert up["ci_low"] > 0, up

    gd, lat = _synthetic_gate(rng)
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / "gate"
        _write_gate_dir(g, gd)
        th = load_thresholds(g)
        ids = lat["ids"]

        def dfn_for(M: np.ndarray, shift: float, qtexts: Dict[str, str]) -> Callable:
            """Document encoder over probe texts: variant = orig (+ shift towards the concept queries)."""
            vec = {}
            for p in gd["probes"]["title_position"]:
                x = ids.index(p["doc_id"])
                o = lat["owner"][p["concept"]]
                vec[p["orig_text"]] = M[x]
                vec[p["variant_text"]] = M[x] + shift * lat["centres"][o["k"]]
            return lambda texts: np.stack([vec[t] for t in texts])

        models = {"P": (0.9, 0.9, 1.0, 11, 0.04), "B": (0.9, 0.85, 1.0, 12, 0.03), "B3": (0.9, 0.85, 1.0, 13, 0.03),
                  "V": (1.1, 1.6, 1.0, 14, 0.05), "good": (1.3, 0.3, 1.0, 15, 0.0), "bad": (0.8, 2.2, 1.0, 16, 0.0),
                  "pos": (1.3, 0.3, 1.0, 17, 0.6)}
        reports = {}
        for lb, (rw, cw, nz, sd, shift) in models.items():
            M, qfn = _plant(lat, gd, rw, cw, nz, sd)
            hashes = {r["doc_id"]: r["text_hash"] for r in gd["records"]}
            rep = evaluate_gate(ids, M, qfn, g, ignore_ids=v9_trained_ids(g) if lb == "V" else None,
                                doc_vectors_fn=dfn_for(M, shift, {}), probe_text="semantic_v4", text_hashes=hashes)
            q_good = {"good": 1.6, "pos": 1.6, "bad": 1.6, "V": 1.4}.get(lb, 1.0)
            reports[lb] = {"gate": rep, "eval_v3": _fake_eval_v3(rng, q_good), "extra": {}}
        bib = {"known_item": {sec: {"all": {"paired_RR_tuned_minus_base": {"delta": 0.02, "ci_low": -0.01,
                                                                           "ci_high": 0.05}}}
                              for sec in th["H10"]["sections"]}}
        extra = {"bib": bib, "lco": {"cand": _fake_eval_v3(rng, 2.0), "base": _fake_eval_v3(rng, 1.0)},
                 "leak": {"n_blocked": 10, "exact": 0, "contained": 0, "examples": []}}
        for lb in ("good", "bad", "pos"):
            reports[lb]["extra"] = dict(extra)
        refs = {"P": "P", "B": "B", "B3": "B3", "V": "V"}
        vg = verdict(reports, "good", refs, th)
        vb = verdict(reports, "bad", refs, th)
        vp = verdict(reports, "pos", refs, th)
        assert vg["pass"], format_verdict(vg)
        assert not vb["pass"] and vb["criteria"]["H4"]["pass"] is False, format_verdict(vb)
        assert vb["criteria"]["H1"]["pass"] is False and vb["criteria"]["H3"]["pass"] is False
        assert vp["criteria"]["H5"]["pass"] is False and not vp["pass"], format_verdict(vp)
        rg, rb = reports["good"]["gate"], reports["bad"]["gate"]
        assert rg["families"]["G1"]["canon_share@20"] < 0.05 < rb["families"]["G1"]["canon_share@20"]
        assert rg["meta"]["n_unscorable"] == 0 and rg["meta"]["text_hash_mismatch"] == 0
        assert reports["V"]["gate"]["meta"]["n_ignored"] == sum(r["v9_trained"] for r in gd["records"])
        assert rg["structure"]["ood_auc"] is not None and rg["probes"]["title_position"]["directions"]
        assert abs(rg["probes"]["title_position"]["directions"]["tail_to_head"]["mean_dz"]) < 1e-3
        assert set(rg["cells"]) >= {"source_group", "desc_lang", "held_out", "new_since_v9", "q_lang"}
        # NOT MEASURED counts as FAIL: drop the candidate's extras and B3
        bare = {**reports, "good": {**reports["good"], "extra": {}}}
        vn = verdict(bare, "good", refs, th)
        assert not vn["pass"] and vn["criteria"]["H10"]["pass"] is None and not vn["criteria"]["H10"]["measured"]
        assert vn["criteria"]["H12"]["pass"] is None and vn["criteria"]["H11"]["pass"] is None
        vn3 = verdict(reports, "good", {**refs, "B3": "missing"}, th)
        assert vn3["criteria"]["H0"]["pass"] is None and not vn3["pass"]
        # the unjudged records sit exactly on the query in every model; they never count against
        assert rg["families"]["G1"]["P@10"] > 0.8, rg["families"]["G1"]
        tc = text_choice(reports, th, b="B", b3="B3", bf="good", bnf="B")
        assert tc["H0"]["pass"] and tc["frames"] == "include", tc
        assert select_checkpoint({"good": vg, "bad": vb}, ["bad", "good"]) == "good"
        assert select_checkpoint({"bad": vb}, ["bad"]) is None
        leak = blocked_query_leaks(g, [gd["queries"][1]["text"], "prefix " + gd["queries"][6]["text"] + " suffix",
                                       gd["queries"][0]["text"]])
        assert leak["exact"] == 1 and leak["contained"] == 1, leak     # query 0 is not blocked
        um = unmatched_top(rg, g, 5)
        assert all(len(v) <= 5 for v in um.values())
        print(format_report(rg))
        print(format_verdict(vg, show_checks="none"))
        print(format_verdict(vb))
        print(json.dumps({"self_test": "ok", "good_pass": vg["pass"], "bad_pass": vb["pass"],
                          "bad_failing": vb["summary"]["failing"], "pos_failing": vp["summary"]["failing"],
                          "canon_share@20": {"good": rg["families"]["G1"]["canon_share@20"],
                                             "bad": rb["families"]["G1"]["canon_share@20"],
                                             "P": reports["P"]["gate"]["families"]["G1"]["canon_share@20"]},
                          "seconds": rg["meta"]["seconds"]}, indent=1))


def random_smoke(gate_dir: str, dim: int = 64, seed: int = 0) -> None:
    """Run :func:`evaluate_gate` on a real gate dir with random vectors (loading / timing check).

    :param gate_dir: Real gate dir.
    :param dim: Random vector dimension.
    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    ids = [r["doc_id"] for r in _read_jsonl(Path(gate_dir) / GATE_FILES["records"])]
    mat = rng.normal(size=(len(ids), dim)).astype(np.float32)
    rep = evaluate_gate(ids, mat, lambda t: rng.normal(size=(len(t), dim)).astype(np.float32), gate_dir)
    print(format_report(rep))


def main() -> None:
    """CLI: ``--self-test`` or ``--random-smoke --gate-dir DIR``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--random-smoke", action="store_true")
    parser.add_argument("--gate-dir", default=None)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.random_smoke:
        random_smoke(args.gate_dir, args.dim)


if __name__ == "__main__":
    main()
