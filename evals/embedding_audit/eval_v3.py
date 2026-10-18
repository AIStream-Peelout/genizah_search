"""Frozen evaluation v3 for the semantic text embedder (text track step 2 of 4): pure numpy, importable on Colab.

Scores one model's vectors against the frozen eval set written by ``build_eval_v3.py`` (``AUDIT_ROOT/v3/eval``)::

    from eval_v3 import evaluate, compare, summary
    res = evaluate(doc_ids, doc_matrix, lambda texts: encode_queries(texts), "/path/to/v3/eval")
    print(summary(res))
    compare(res_base, res_tuned, group="held_out", metric="AP")   # paired, subject-clustered bootstrap CI
    compare(res_base, res_tuned, group="holiday", metric="P@10")  # paired, festival-clustered bootstrap CI

Sections (all over the frozen *eval pool* = every eligible record of ``corpus_semantic.jsonl``):

* ``subjects``     — subject retrieval. Each query of a subject ranks the whole pool; relevant = records carrying the
  subject (strong label); records carrying it only as a *weak* label (KTIV general title, incidental mention) and
  *unjudged* unlabelled records whose catalogue lines name the subject are dropped from that subject's ranking.
  Each query has a ``rel_mode``: ``bare`` (a bare name; ``canon`` carriers -- the subject named only on a scripture /
  tractate / law-code frame -- are misses) or ``law`` (a law-name form, probe and llm_t2 queries; canon carriers are
  relevant); see :func:`subject_sets`. AP, P@10, R@100 per query -> mean per subject -> macro over subjects (also
  ``macro_by_mode``), separately for
  ``held_out`` subjects (the generalisation **gate**), ``linked`` subjects (held out with them) and ``seen`` subjects.
  Final-run eval dirs (``build_eval_v3.py --final``) hold no held-out subject (``gate`` is None) and add ``focus``
  subjects: trained on, so they are scored over HELD-OUT records only (relevant = strong carriers that are held out;
  their train-side carriers are left out of the ranking). ``subjects.primary_group`` names the group that decides a
  comparison: ``held_out`` when the eval dir has held-out subjects, else ``focus``.
* ``memorisation`` — for seen subjects, AP of the subject's name queries when the positives are a size-matched sample
  of TRAIN-side records vs of held-out records (same negatives): ``gap = AP_train - AP_held``.
* ``known_item``   — eval-pool synthetic queries -> their own record. Tie-aware rank: identical-text twins count as
  hits, ties count against. MRR / R@k by query type, plus the target's rank among records sharing its subject.
* ``collection``   — share of a record's top-10 neighbours from its own collection series, divided by the share
  expected by chance (sum of squared series shares in the sample). 1.0 = no collection clustering.
* ``hubness``      — share of all subject queries' top-10 slots taken by duplicate-text or < 60-char records.
* ``holiday``      — (eval dirs with ``holiday_queries.json``) every name form of every festival over the whole pool:
  P@10, P@20, MRR@100, ``canon_share@20`` (canon carriers in the top 20) and the alt-title-only share with its base
  rate and lift; unjudged records removed, canon records misses; ``held_out`` = the same over held-out relevant
  records only. ``compare(..., group="holiday")`` clusters by festival.

Ties are always resolved pessimistically (a tied irrelevant record ranks above the relevant one), so a model that
collapses many records onto one vector cannot gain from it.

Self-test (numpy only, tiny)::

    python3 eval_v3.py --self-test
    python3 eval_v3.py --random-smoke --eval-dir AUDIT_ROOT/v3/eval   # real eval files, random 64-d vectors
"""

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

FILES = {
    "held_out_subjects": "held_out_subjects.json",
    "held_out_records": "held_out_records.json",
    "subject_queries": "subject_queries.json",
    "subject_relevance": "subject_relevance.json",
    "known_item": "known_item.jsonl",
    "memorization_pairs": "memorization_pairs.json",
    "collection_sample": "collection_sample.json",
    "eval_pool": "eval_pool.jsonl",
}
HOLIDAY_FILE = "holiday_queries.json"  # optional (eval dirs built after 2026-10-08): the "holiday" section
SECTIONS = ("subjects", "memorisation", "known_item", "collection", "hubness", "holiday")
STATUSES = ("held_out", "linked", "focus", "seen")
ALWAYS_REPORTED = ("held_out", "linked", "seen")  # present in every result (possibly empty); "focus" only when used
REL_MODES = ("bare", "law")  # subject query relevance modes (subject_queries.json "rel_mode"; default "bare")
HOLIDAY_METRICS = ("P@10", "P@20", "MRR@100", "canon_share@20")  # per-query holiday metrics usable in compare()
MIN_RELEVANT = 5          # a subject needs this many strong carriers in the pool to be scored
WITHIN_SUBJECT_MIN = 10   # within-subject rank only for subjects with at least this many carriers
SHORT_CHARS = 60          # hubness: records shorter than this count as "short"
LONG_CHARS = 200          # collection lift: the "long" subset
TOP_K = 10


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
def load_eval_dir(eval_dir) -> dict:
    """Load the frozen eval files written by ``build_eval_v3.py``.

    :param eval_dir: Directory holding the files named in :data:`FILES`.
    :returns: Dict keyed like :data:`FILES` (``known_item`` and ``eval_pool`` as lists of row dicts), plus
        ``holiday`` (None when absent) and ``eval_dir`` (the directory, as a string).
    :rtype: dict
    """
    eval_dir = Path(eval_dir)
    out: dict = {"eval_dir": str(eval_dir)}
    for key, name in FILES.items():
        path = eval_dir / name
        if name.endswith(".jsonl"):
            with open(path, encoding="utf-8") as fh:
                out[key] = [json.loads(line) for line in fh if line.strip()]
        else:
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    holiday_path = eval_dir / HOLIDAY_FILE
    out["holiday"] = json.loads(holiday_path.read_text(encoding="utf-8")) if holiday_path.exists() else None
    return out


def normalise_rows(mat: np.ndarray) -> np.ndarray:
    """Return an L2-row-normalised float32 copy (or the input itself when it already is one).

    :param mat: (n, d) array.
    :returns: Normalised (n, d) float32 array.
    :rtype: np.ndarray
    """
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if mat.size and np.allclose(norms, 1.0, atol=1e-3):
        return mat
    return mat / np.maximum(norms, 1e-12)


def _rows_by_id(corpus_rows) -> Dict[str, dict]:
    """Index corpus rows by ``doc_id``.

    :param corpus_rows: Iterable of row dicts, or a dict ``doc_id -> row``.
    :returns: Dict ``doc_id -> row``.
    :rtype: Dict[str, dict]
    """
    if isinstance(corpus_rows, dict):
        return corpus_rows
    return {r["doc_id"]: r for r in corpus_rows}


# ---------------------------------------------------------------------------------------------------------------
# ranking primitives
# ---------------------------------------------------------------------------------------------------------------
def positive_ranks(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """1-based ranks of the positives in a pessimistic ranking of ``pos`` + ``neg``.

    A negative whose score equals a positive's ranks above it. Runs in O(N log m) for N negatives and m positives
    (no full sort of the pool).

    :param pos: Scores of the m relevant candidates.
    :param neg: Scores of the irrelevant candidates.
    :returns: Ranks of the positives, best positive first (ascending).
    :rtype: np.ndarray
    """
    m = len(pos)
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    p_asc = np.sort(pos)
    idx = np.searchsorted(p_asc, neg, side="right")  # number of positives scoring <= each negative
    counts = np.bincount(idx, minlength=m + 1)
    beaten_by = counts[::-1].cumsum()[::-1][1:]       # [k] = negatives scoring >= p_asc[k]
    return np.arange(1, m + 1) + beaten_by[::-1]


def ranking_metrics(ranks: np.ndarray, n_pos: int) -> Dict[str, float]:
    """AP, P@10 and R@100 from the positives' ranks.

    :param ranks: Ascending 1-based ranks of every positive (output of :func:`positive_ranks`).
    :param n_pos: Number of positives.
    :returns: Dict with ``AP``, ``P@10``, ``R@100``.
    :rtype: Dict[str, float]
    """
    if n_pos == 0:
        return {"AP": 0.0, "P@10": 0.0, "R@100": 0.0}
    ap = float((np.arange(1, n_pos + 1) / ranks).mean())
    return {"AP": ap, "P@10": float((ranks <= 10).sum() / 10), "R@100": float((ranks <= 100).sum() / n_pos)}


def hit_rank(scores: np.ndarray, hits: np.ndarray, candidates: Optional[np.ndarray] = None) -> int:
    """Tie-aware 1-based rank of the best-scoring hit among candidates (ties count against).

    :param scores: Scores over the whole pool.
    :param hits: Pool indices that count as hits (target + its identical-text twins).
    :param candidates: Pool indices to rank among (None = the whole pool). Must contain the hits.
    :returns: 1 + number of non-hit candidates scoring >= the best hit.
    :rtype: int
    """
    best = scores[hits].max()
    if candidates is None:
        return int((scores >= best).sum() - (scores[hits] >= best).sum()) + 1
    cand = scores[candidates]
    return int((cand >= best).sum() - (scores[hits] >= best).sum()) + 1


def rank_stats(ranks: Sequence[int]) -> Dict[str, float]:
    """MRR / R@1 / R@10 / R@100 / median rank of 1-based ranks.

    :param ranks: Ranks.
    :returns: Metrics dict (``n`` = number of ranks).
    :rtype: Dict[str, float]
    """
    r = np.asarray(ranks, dtype=np.float64)
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "MRR": round(float((1 / r).mean()), 4), "R@1": round(float((r <= 1).mean()), 4),
            "R@10": round(float((r <= 10).mean()), 4), "R@100": round(float((r <= 100).mean()), 4),
            "median_rank": float(np.median(r))}


def _mean_dict(dicts: List[Dict[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    """Mean of each key over a list of metric dicts.

    :param dicts: Metric dicts.
    :param keys: Keys to average.
    :returns: Dict of rounded means (empty when ``dicts`` is empty).
    :rtype: Dict[str, float]
    """
    if not dicts:
        return {}
    return {k: round(float(np.mean([d[k] for d in dicts])), 4) for k in keys}


# ---------------------------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------------------------
def paired_bootstrap(a: Dict[str, Sequence[float]], b: Dict[str, Sequence[float]], n_resamples: int = 1000,
                     seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Two-stage clustered paired bootstrap of the macro difference ``mean_c mean_q (b - a)``.

    Clusters (subjects / topics, or target records for known-item) are resampled with replacement, then the paired
    per-query differences inside each drawn cluster. With only a handful of clusters (the 6 held-out subjects) the
    interval is honestly wide.

    :param a: Cluster -> per-query values of model A (baseline).
    :param b: Cluster -> per-query values of model B, aligned query by query with ``a``.
    :param n_resamples: Bootstrap resamples.
    :param seed: RNG seed.
    :param alpha: Two-sided level (0.05 -> 95 % interval).
    :returns: ``{delta, ci_low, ci_high, p_b_better, n_clusters, n_queries}``.
    :rtype: Dict[str, float]
    """
    clusters = sorted(set(a) & set(b))
    diffs = []
    for c in clusters:
        da, db = np.asarray(a[c], dtype=np.float64), np.asarray(b[c], dtype=np.float64)
        if len(da) != len(db):
            raise ValueError(f"cluster {c!r}: {len(da)} vs {len(db)} queries; results are not aligned")
        if len(da):
            diffs.append(db - da)
    if not diffs:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_b_better": 0.0, "n_clusters": 0, "n_queries": 0}
    rng = np.random.default_rng(seed)
    k = len(diffs)
    within = np.empty((k, n_resamples))
    for i, d in enumerate(diffs):
        within[i] = d[rng.integers(0, len(d), size=(n_resamples, len(d)))].mean(axis=1)
    pick = rng.integers(0, k, size=(n_resamples, k))
    stats = within[pick, np.arange(n_resamples)[:, None]].mean(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"delta": round(float(np.mean([d.mean() for d in diffs])), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "p_b_better": round(float((stats > 0).mean()), 4),
            "n_clusters": k, "n_queries": int(sum(len(d) for d in diffs))}


def _holiday_clusters(res: dict, group: str, metric: str) -> Dict[str, List[tuple]]:
    """Per-festival ``(query key, value)`` lists of a holiday section (sorted by key, for paired comparisons).

    :param res: :func:`evaluate` result.
    :param group: ``holiday`` (whole pool) or ``holiday_held_out``.
    :param metric: One of :data:`HOLIDAY_METRICS`.
    :returns: ``festival -> [(key, value), ...]`` (empty when the section is absent).
    :rtype: Dict[str, List[tuple]]
    """
    if metric not in HOLIDAY_METRICS:
        raise ValueError(f"holiday metric must be one of {HOLIDAY_METRICS}, not {metric!r}")
    sec = res.get("holiday") or {}
    if group == "holiday_held_out":
        sec = sec.get("held_out") or {}
    out: Dict[str, List[tuple]] = defaultdict(list)
    for key in sorted(sec.get("per_query", {})):
        row = sec["per_query"][key]
        out[row["festival"]].append((key, row[metric]))
    return out


def compare(res_a: dict, res_b: dict, group: str = "held_out", metric: str = "AP", n_resamples: int = 1000,
            seed: int = 0, rel_mode: Optional[str] = None) -> Dict[str, float]:
    """Paired bootstrap of model B minus model A on one metric of two :func:`evaluate` results.

    :param res_a: Baseline result.
    :param res_b: Candidate result (same eval dir, so the same queries in the same order).
    :param group: ``held_out`` / ``linked`` / ``focus`` / ``seen`` (subject-clustered; an empty or absent group gives
        ``n_clusters == 0``), ``known_item`` (clustered by target
        record; metric ``RR``, ``R@10`` or ``R@100``), ``memorisation`` / ``memorisation_trained`` (per-query gap
        ``AP_train - AP_held``, subject-clustered; metric ignored), or ``holiday`` / ``holiday_held_out``
        (clustered by festival; metric one of :data:`HOLIDAY_METRICS`).
    :param metric: Per-query metric (``AP``, ``P@10``, ``R@100`` for subject groups).
    :param n_resamples: Bootstrap resamples.
    :param seed: RNG seed.
    :param rel_mode: Subject groups only: restrict to queries of this ``rel_mode`` (``bare`` / ``law``).
    :returns: Output of :func:`paired_bootstrap`.
    :rtype: Dict[str, float]
    :raises ValueError: When the two results do not hold the same queries.
    """
    if group in ("holiday", "holiday_held_out"):
        ca, cb = _holiday_clusters(res_a, group, metric), _holiday_clusters(res_b, group, metric)
        for f in set(ca) & set(cb):
            if [k for k, _ in ca[f]] != [k for k, _ in cb[f]]:
                raise ValueError(f"holiday queries of {f} differ between the two results")
        return paired_bootstrap({f: [v for _, v in x] for f, x in ca.items()},
                                {f: [v for _, v in x] for f, x in cb.items()}, n_resamples, seed)
    if group in ("memorisation", "memorisation_trained"):
        # metric is ignored: per-query memorisation gaps, clustered by subject. With the base model as A this is the
        # tuned model's gap net of the base model's (selection-bias) calibration gap.
        return paired_bootstrap(res_a[group]["per_query_gap"], res_b[group]["per_query_gap"], n_resamples, seed)
    if group == "known_item":
        def clusters(res: dict) -> Dict[str, List[float]]:
            pq = res["known_item"]["per_query"]
            out: Dict[str, List[float]] = defaultdict(list)
            for doc, rank in zip(pq["doc_id"], pq["rank"]):
                val = {"RR": 1.0 / rank, "R@10": float(rank <= 10), "R@100": float(rank <= 100)}[metric]
                out[doc].append(val)
            return out
        if res_a["known_item"]["per_query"]["qid"] != res_b["known_item"]["per_query"]["qid"]:
            raise ValueError("known-item query lists differ between the two results")
        return paired_bootstrap(clusters(res_a), clusters(res_b), n_resamples, seed)
    pa = res_a["subjects"].get(group, {}).get("per_query", {})
    pb = res_b["subjects"].get(group, {}).get("per_query", {})

    def values(pq: dict) -> Dict[str, List[float]]:
        if rel_mode is None:
            return {s: v[metric] for s, v in pq.items()}
        out = {s: [x for x, m in zip(v[metric], v.get("rel_mode") or ["bare"] * len(v[metric])) if m == rel_mode]
               for s, v in pq.items()}
        return {s: x for s, x in out.items() if x}

    return paired_bootstrap(values(pa), values(pb), n_resamples, seed)


# ---------------------------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------------------------
def _score_chunks(Q: np.ndarray, P: np.ndarray, chunk: int) -> Iterable[tuple]:
    """Yield ``(start, scores)`` blocks of ``Q @ P.T``.

    :param Q: (nq, d) query matrix.
    :param P: (N, d) pool matrix.
    :param chunk: Queries per block.
    :yields: (first query index, (b, N) score block).
    """
    for start in range(0, len(Q), chunk):
        yield start, Q[start:start + chunk] @ P.T


def subject_sets(r: dict, rel_mode: str, held: Optional[set] = None) -> tuple:
    """Relevant and left-out doc ids of one subject for one query mode.

    * ``bare`` (a bare subject / festival name): relevant = strong carriers minus ``canon`` carriers; canon carriers
      stay in the ranking as misses ("Purim" -> the Book of Esther is wrong).
    * ``law`` (a law-name form such as "laws of Sukkah" / "הלכות פסח", and every probe / llm_t2 query):
      relevant = strong carriers plus canon carriers (the tractate and the law code answer it).

    Left out of the ranking (neither relevant nor irrelevant): ``unjudged`` records (always; they lose only to the
    relevant set) and weak carriers that are not relevant (in ``bare`` mode a weak canon carrier that is not unjudged
    stays in the ranking as a miss). ``held`` (focus subjects: trained on) restricts the relevant set to held-out
    records and leaves the train-side relevant records out of the ranking. Eval dirs built before rel_mode / canon
    existed have no ``canon`` / ``rel_mode``: both modes then reduce to the old rule (relevant = strong).

    :param r: ``subject_relevance.json`` entry (``strong``, ``weak``, optional ``canon`` / ``unjudged``).
    :param rel_mode: ``bare`` or ``law``.
    :param held: Held-out doc ids for a focus subject, else None.
    :returns: (relevant ids, left-out ids, train-side relevant ids moved to the left-out set).
    :rtype: tuple
    """
    if rel_mode not in REL_MODES:
        raise ValueError(f"unknown rel_mode {rel_mode!r} (expected one of {REL_MODES})")
    strong, canon = set(r.get("strong", [])), set(r.get("canon", []))
    unjudged = set(r.get("unjudged", []))
    relevant = (strong | canon) if rel_mode == "law" else (strong - canon)
    excluded = (set(r.get("weak", [])) | unjudged) - relevant
    if rel_mode == "bare":
        excluded -= canon - unjudged  # canon carriers are misses for a bare name
    train_side: set = set()
    if held is not None:
        train_side = {d for d in relevant if d not in held}
        relevant -= train_side
        excluded |= train_side
    return relevant, excluded, train_side


def subject_retrieval(P: np.ndarray, pool_pos: Dict[str, int], qvec: Dict[str, np.ndarray], ev: dict,
                      chunk: int = 256) -> tuple:
    """Subject retrieval for every subject in ``subject_queries.json`` (+ top-10 lists for hubness).

    Each query carries a ``rel_mode`` (``bare`` / ``law``; default ``bare``) that picks its relevant set
    (:func:`subject_sets`); a (subject, mode) with fewer than :data:`MIN_RELEVANT` relevant pool records is skipped.
    ``focus`` subjects (final-run eval dirs) were trained on: their relevant set is restricted to held-out records
    (``held_out_records.json``) and their train-side relevant carriers are left out of the ranking, like weak labels.

    :param P: Normalised pool matrix.
    :param pool_pos: ``doc_id -> row of P``.
    :param qvec: Query text -> normalised vector.
    :param ev: Loaded eval dir.
    :param chunk: Queries per score block.
    :returns: (section dict, list of top-10 pool-index arrays for every scored query).
    :rtype: tuple
    """
    rel = ev["subject_relevance"]
    held = set(ev["held_out_records"]["held_out"])
    present = {e["status"] for e in ev["subject_queries"]["subjects"].values()}

    def idx(ids: Iterable[str]) -> np.ndarray:
        return np.array(sorted(pool_pos[d] for d in ids if d in pool_pos), dtype=np.int64)

    jobs = []  # (subject, status, source, rel_mode, text, relevant idx, left-out idx)
    skipped = Counter()
    skipped_queries = Counter()
    n_rel: Dict[str, Dict[str, int]] = defaultdict(dict)
    n_train_side: Dict[str, Dict[str, int]] = defaultdict(dict)
    for sid, entry in ev["subject_queries"]["subjects"].items():
        r = rel.get(sid, {})
        focus = entry["status"] == "focus"  # trained subject: only its held-out carriers are judged
        sets: Dict[str, tuple] = {}
        n_jobs = 0
        for q in entry["queries"]:
            mode = q.get("rel_mode", "bare")
            if mode not in sets:
                relevant, excluded, train_side = subject_sets(r, mode, held if focus else None)
                sets[mode] = (idx(relevant), idx(excluded))
                if len(sets[mode][0]) >= MIN_RELEVANT:
                    n_rel[sid][mode] = len(sets[mode][0])
                    if focus:
                        n_train_side[sid][mode] = int(sum(d in pool_pos for d in train_side))
            good, out_idx = sets[mode]
            if len(good) < MIN_RELEVANT:
                skipped_queries[mode] += 1
                continue
            jobs.append((sid, entry["status"], q["source"], mode, q["text"], good, out_idx))
            n_jobs += 1
        if not n_jobs:
            skipped[entry["status"]] += 1
    out: Dict[str, dict] = {s: {"per_query": {}, "per_subject": {}} for s in STATUSES
                            if s in ALWAYS_REPORTED or s in present}
    tops: List[np.ndarray] = []
    N = P.shape[0]
    Q = np.stack([qvec[j[4]] for j in jobs]) if jobs else np.zeros((0, P.shape[1]), dtype=np.float32)
    for start, S in _score_chunks(Q, P, chunk):
        for i, s in enumerate(S):
            sid, status, source, mode, _, good, out_idx = jobs[start + i]
            mask = np.ones(N, dtype=bool)
            mask[good] = False
            mask[out_idx] = False
            m = ranking_metrics(positive_ranks(s[good], s[mask]), len(good))
            pq = out[status]["per_query"].setdefault(sid, {"AP": [], "P@10": [], "R@100": [], "source": [],
                                                           "rel_mode": []})
            for k, v in m.items():
                pq[k].append(round(v, 5))
            pq["source"].append(source)
            pq["rel_mode"].append(mode)
            tops.append(np.argpartition(-s, TOP_K)[:TOP_K] if N > TOP_K else np.arange(N))
    keys = ("AP", "P@10", "R@100")

    def by_label(pq: dict, field: str) -> Dict[str, Dict[str, float]]:
        return {lab: {k: round(float(np.mean([pq[k][i] for i, x in enumerate(pq[field]) if x == lab])), 4)
                      for k in keys} for lab in sorted(set(pq[field]))}

    for status in [s for s in STATUSES if s in out]:
        per_subject = {}
        by_source: Dict[str, List[dict]] = defaultdict(list)
        by_mode: Dict[str, List[dict]] = defaultdict(list)
        for sid, pq in out[status]["per_query"].items():
            per_subject[sid] = {k: round(float(np.mean(pq[k])), 4) for k in keys}
            per_subject[sid]["n_queries"] = len(pq["AP"])
            main_mode = "bare" if "bare" in pq["rel_mode"] else pq["rel_mode"][0]
            per_subject[sid]["n_relevant"] = n_rel[sid][main_mode]
            per_subject[sid]["n_relevant_by_mode"] = {m: n_rel[sid][m] for m in sorted(set(pq["rel_mode"]))}
            if sid in n_train_side:
                per_subject[sid]["n_excluded_train_side"] = n_train_side[sid][main_mode]
            per_subject[sid]["by_source"] = by_label(pq, "source")
            per_subject[sid]["by_mode"] = by_label(pq, "rel_mode")
            for src, v in per_subject[sid]["by_source"].items():
                by_source[src].append(v)
            for mode, v in per_subject[sid]["by_mode"].items():
                by_mode[mode].append(v)
        out[status]["per_subject"] = per_subject
        out[status]["macro"] = _mean_dict(list(per_subject.values()), keys)
        out[status]["macro"]["n_subjects"] = len(per_subject)
        out[status]["macro_by_source"] = {src: {**_mean_dict(v, keys), "n_subjects": len(v)} for src, v in by_source.items()}
        out[status]["macro_by_mode"] = {m: {**_mean_dict(v, keys), "n_subjects": len(v)}
                                        for m, v in sorted(by_mode.items())}
        kinds: Dict[str, List[dict]] = defaultdict(list)
        for sid, ps in per_subject.items():
            kinds[sid.split(":", 1)[0]].append(ps)
        out[status]["macro_by_kind"] = {k: {**_mean_dict(v, keys), "n_subjects": len(v)} for k, v in sorted(kinds.items())}
    # no held-out subject (final run): no gate; the focus group is the primary comparison instead
    out["gate"] = dict(out["held_out"]["macro"]) if "held_out" in present else None
    out["primary_group"] = "held_out" if "held_out" in present else "focus" if "focus" in present else None
    out["skipped_subjects_lt_min_relevant"] = dict(skipped)
    out["skipped_queries_lt_min_relevant_by_mode"] = dict(skipped_queries)
    return out, tops


def _seeded_sample(ids: Sequence[str], n: int, key: str) -> List[str]:
    """Deterministic size-``n`` sample of ids (independent of the model being scored).

    :param ids: Candidate ids.
    :param n: Sample size (``<= len(ids)``).
    :param key: Seed key (e.g. ``"memo-trained:<subject>"``).
    :returns: Sorted sample.
    :rtype: List[str]
    """
    ordered = sorted(ids)
    if n <= 0:
        return []
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    pick = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    return sorted(ordered[i] for i in pick)


def memorisation(P: np.ndarray, pool_pos: Dict[str, int], qvec: Dict[str, np.ndarray], ev: dict,
                 n_resamples: int = 1000, train_override: Optional[Dict[str, Sequence[str]]] = None) -> dict:
    """Train-side vs held-out subject AP for seen subjects (size-matched positives, identical negatives).

    By default the train side is the frozen ``memorization_pairs`` sample of TRAIN-SIDE carriers, many of which a
    given mixture never uses as a positive (the gap is then diluted). ``train_override`` replaces it with the records
    the model was actually trained on for that subject (``train_embedder_v3.trained_subject_positives``): a seeded
    sample of them, size-matched with a seeded sample of the frozen held-out side.

    :param P: Normalised pool matrix.
    :param pool_pos: ``doc_id -> row of P``.
    :param qvec: Query text -> normalised vector.
    :param ev: Loaded eval dir.
    :param n_resamples: Bootstrap resamples for the gap's CI.
    :param train_override: ``subject -> doc ids`` used as the train side instead of the frozen sample.
    :returns: Section dict (``macro`` with ``AP_train``, ``AP_held``, ``gap`` and its CI).
    :rtype: dict
    """
    rel = ev["subject_relevance"]
    queries = ev["subject_queries"]["subjects"]
    N = P.shape[0]
    per_subject, ap_train, ap_held = {}, {}, {}
    for sid, pair in ev["memorization_pairs"]["subjects"].items():
        if train_override is None:
            tr_ids = [d for d in pair["train"] if d in pool_pos]
            he_ids = [d for d in pair["held_out"] if d in pool_pos]
            n = min(len(tr_ids), len(he_ids))
            tr_ids, he_ids = tr_ids[:n], he_ids[:n]
        else:
            tr_all = sorted({d for d in train_override.get(sid, ()) if d in pool_pos})
            he_all = [d for d in pair["held_out"] if d in pool_pos]
            n = min(len(tr_all), len(he_all))
            tr_ids = _seeded_sample(tr_all, n, f"memo-trained:{sid}:train")
            he_ids = _seeded_sample(he_all, n, f"memo-trained:{sid}:held")
        if n < MIN_RELEVANT or sid not in queries:
            continue
        tr = np.array([pool_pos[d] for d in tr_ids], dtype=np.int64)
        he = np.array([pool_pos[d] for d in he_ids], dtype=np.int64)
        carriers = [pool_pos[d] for d in rel[sid]["strong"] + rel[sid]["weak"] if d in pool_pos]
        neg_mask = np.ones(N, dtype=bool)
        neg_mask[carriers] = False
        texts = [q["text"] for q in queries[sid]["queries"]]
        S = np.stack([qvec[t] for t in texts]) @ P.T
        a_tr = [ranking_metrics(positive_ranks(s[tr], s[neg_mask]), n)["AP"] for s in S]
        a_he = [ranking_metrics(positive_ranks(s[he], s[neg_mask]), n)["AP"] for s in S]
        ap_train[sid], ap_held[sid] = a_tr, a_he
        per_subject[sid] = {"n": int(n), "AP_train": round(float(np.mean(a_tr)), 4),
                            "AP_held": round(float(np.mean(a_he)), 4),
                            "gap": round(float(np.mean(a_tr) - np.mean(a_he)), 4)}
    keys = ("AP_train", "AP_held", "gap")
    kinds: Dict[str, List[dict]] = defaultdict(list)
    for sid, ps in per_subject.items():
        kinds[sid.split(":", 1)[0]].append(ps)
    boot = paired_bootstrap(ap_held, ap_train, n_resamples=n_resamples)
    return {"macro": {**_mean_dict(list(per_subject.values()), keys), "n_subjects": len(per_subject),
                      "gap_ci": [boot["ci_low"], boot["ci_high"]]},
            "macro_by_kind": {k: {**_mean_dict(v, keys), "n_subjects": len(v)} for k, v in sorted(kinds.items())},
            "per_subject": per_subject,
            # per name query: AP_train - AP_held (for compare(): the gap of a tuned model minus the base model's gap)
            "per_query_gap": {sid: [round(a - b, 5) for a, b in zip(ap_train[sid], ap_held[sid])] for sid in ap_train}}


def known_item(P: np.ndarray, pool_pos: Dict[str, int], qvec: Dict[str, np.ndarray], ev: dict,
               twins: Dict[str, np.ndarray], chunk: int = 256, include_shelfmark: bool = False) -> dict:
    """Known-item retrieval of eval-pool synthetic queries, plus within-subject rank.

    :param P: Normalised pool matrix.
    :param pool_pos: ``doc_id -> row of P``.
    :param qvec: Query text -> normalised vector.
    :param ev: Loaded eval dir.
    :param twins: ``text_hash -> pool indices`` sharing that text.
    :param chunk: Queries per score block.
    :param include_shelfmark: Also score queries flagged as naming a shelf mark (off: the semantic text has none).
    :returns: Section dict.
    :rtype: dict
    """
    hash_of = {r["doc_id"]: r["text_hash"] for r in ev["eval_pool"]}
    carriers: Dict[str, np.ndarray] = {}
    doc_subjects: Dict[str, List[str]] = defaultdict(list)
    for sid, r in ev["subject_relevance"].items():
        idx = np.array(sorted(pool_pos[d] for d in r["strong"] if d in pool_pos), dtype=np.int64)
        if len(idx) >= WITHIN_SUBJECT_MIN:
            carriers[sid] = idx
            for d in r["strong"]:
                doc_subjects[d].append(sid)
    skipped = Counter()
    rows = []
    for q in ev["known_item"]:
        if q.get("shelfmark_query") and not include_shelfmark:
            skipped["shelfmark_query"] += 1
        elif q["doc_id"] not in pool_pos:
            skipped["target_not_in_pool"] += 1
        else:
            rows.append(q)
    by_type: Dict[str, List[int]] = defaultdict(list)
    within: List[tuple] = []  # (rank, n_carriers)
    pq = {"qid": [], "doc_id": [], "type": [], "rank": []}
    Q = np.stack([qvec[q["text"]] for q in rows]) if rows else np.zeros((0, P.shape[1]), dtype=np.float32)
    for start, S in _score_chunks(Q, P, chunk):
        for i, s in enumerate(S):
            q = rows[start + i]
            hits = twins[hash_of[q["doc_id"]]]
            rank = hit_rank(s, hits)
            by_type[q["type"]].append(rank)
            for key, val in (("qid", q["qid"]), ("doc_id", q["doc_id"]), ("type", q["type"]), ("rank", rank)):
                pq[key].append(val)
            for sid in doc_subjects.get(q["doc_id"], []):
                cand = carriers[sid]
                sub_hits = np.intersect1d(hits, cand)
                within.append((hit_rank(s, sub_hits, cand), len(cand)))
    w_r = np.array([w[0] for w in within], dtype=np.float64)
    w_n = np.array([w[1] for w in within], dtype=np.float64)
    within_stats = {**rank_stats(w_r), "median_percentile": round(float(np.median((w_r - 1) / (w_n - 1))), 4)} \
        if len(within) else {"n": 0}
    return {"by_type": {t: rank_stats(r) for t, r in sorted(by_type.items())},
            "all": rank_stats([r for v in by_type.values() for r in v]),
            "macro_by_type": {k: round(float(np.mean([rank_stats(r)[k] for r in by_type.values()])), 4)
                              for k in ("MRR", "R@10", "R@100")} if by_type else {},
            "within_subject": within_stats, "skipped": dict(skipped), "per_query": pq}


def collection_lift(P: np.ndarray, pool_pos: Dict[str, int], ev: dict, held: set, chunk: int = 256) -> dict:
    """Same-series share of top-10 neighbours over the frozen sample, relative to chance.

    :param P: Normalised pool matrix.
    :param pool_pos: ``doc_id -> row of P``.
    :param ev: Loaded eval dir.
    :param held: Held-out doc ids (for the train/held-out breakdown).
    :param chunk: Sample records per score block.
    :returns: Section dict with ``all``, ``long`` (>= 200 chars) and ``by_side``.
    :rtype: dict
    """
    meta = {r["doc_id"]: r for r in ev["eval_pool"]}
    pool_ids = [None] * len(pool_pos)
    for d, i in pool_pos.items():
        pool_ids[i] = d
    series = np.array([meta[d]["series"] for d in pool_ids], dtype=object)
    sample = [d for d in ev["collection_sample"]["doc_ids"] if d in pool_pos]
    idx = np.array([pool_pos[d] for d in sample], dtype=np.int64)
    share = np.zeros(len(sample))
    for start in range(0, len(idx), chunk):
        S = P[idx[start:start + chunk]] @ P.T
        for i, s in enumerate(S):
            j = idx[start + i]
            s[j] = -np.inf
            nn = np.argpartition(-s, TOP_K)[:TOP_K]
            share[start + i] = float(np.mean([series[k] == series[j] and series[j] != "" for k in nn]))
    pool_counts = Counter(series.tolist())

    def block(sel: List[int]) -> dict:
        if not sel:
            return {"n": 0}
        ser = Counter(series[idx[i]] for i in sel)
        n = len(sel)
        expected = sum((c / n) ** 2 for c in ser.values())
        expected_pool = sum((c / n) * (pool_counts[s] / len(pool_ids)) for s, c in ser.items())
        observed = float(share[sel].mean())
        return {"n": n, "observed": round(observed, 4), "expected": round(expected, 4),
                "lift": round(observed / expected, 3) if expected else None,
                "expected_pool": round(expected_pool, 4),
                "lift_vs_pool": round(observed / expected_pool, 3) if expected_pool else None}

    allsel = list(range(len(sample)))
    return {"all": block(allsel),
            "long": block([i for i in allsel if meta[sample[i]]["n_chars"] >= LONG_CHARS]),
            "by_side": {"train": block([i for i in allsel if sample[i] not in held]),
                        "held_out": block([i for i in allsel if sample[i] in held])}}


def hubness(tops: List[np.ndarray], ev: dict, pool_pos: Dict[str, int]) -> dict:
    """Share of subject-query top-10 slots taken by duplicate-text or very short records.

    :param tops: Top-10 pool indices per subject query (from :func:`subject_retrieval`).
    :param ev: Loaded eval dir.
    :param pool_pos: ``doc_id -> row of P``.
    :returns: Section dict (``share``, chance ``baseline`` = that share of the pool, ``ratio``, top hubs).
    :rtype: dict
    """
    meta = {r["doc_id"]: r for r in ev["eval_pool"]}
    pool_ids = [None] * len(pool_pos)
    for d, i in pool_pos.items():
        pool_ids[i] = d
    flag = np.array([meta[d]["dup_n"] > 1 or meta[d]["n_chars"] < SHORT_CHARS for d in pool_ids])
    if not tops:
        return {"n_queries": 0}
    allt = np.concatenate(tops)
    share = float(flag[allt].mean())
    base = float(flag.mean())
    freq = Counter(allt.tolist())
    return {"n_queries": len(tops), "share": round(share, 4), "baseline": round(base, 4),
            "ratio": round(share / base, 3) if base else None,
            "dup_share": round(float(np.mean([meta[pool_ids[k]]["dup_n"] > 1 for k in allt])), 4),
            "short_share": round(float(np.mean([meta[pool_ids[k]]["n_chars"] < SHORT_CHARS for k in allt])), 4),
            "top_hubs": [{"doc_id": pool_ids[k], "slots": c, "share_of_queries": round(c / len(tops), 4),
                          "n_chars": meta[pool_ids[k]]["n_chars"], "dup_n": meta[pool_ids[k]]["dup_n"]}
                         for k, c in freq.most_common(10)]}


# ---------------------------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------------------------
def pessimistic_top(s: np.ndarray, cand: np.ndarray, is_rel: np.ndarray, is_canon: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` candidates by score with pessimistic tie-breaking.

    Order: score descending; among tied scores irrelevant records first, and among those canon records first. So a
    model that collapses many records onto one vector cannot gain from the pool order.

    :param s: Scores over the whole pool.
    :param cand: Pool indices that take part in the ranking.
    :param is_rel: Boolean mask over the pool: relevant.
    :param is_canon: Boolean mask over the pool: canon (a miss that is reported separately).
    :param k: Number of records to return.
    :returns: Up to ``k`` pool indices, best first.
    :rtype: np.ndarray
    """
    k = min(k, len(cand))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    sc = s[cand]
    kth = np.partition(sc, len(sc) - k)[len(sc) - k]  # k-th largest candidate score
    sel = cand[sc >= kth]
    order = np.lexsort(((~is_canon[sel]).astype(np.int8), is_rel[sel].astype(np.int8), -s[sel]))
    return sel[order][:k]


def _holiday_agg(rows: List[dict]) -> Dict[str, Optional[float]]:
    """Aggregate holiday per-query rows.

    ``P@10`` / ``P@20`` / ``MRR@100`` / ``canon_share@20`` are means over queries. The alt-title-only figures are
    weighted by relevant top-10 hits: ``alt_only_share`` = alt-only hits / hits, ``alt_only_base_rate`` = the
    expected share if hits were drawn at random from each query's relevant set (sum of hits x festival base rate /
    hits), ``alt_only_lift`` = share / base rate (1.0 = no preference for records that name the festival only under
    Alternate titles).

    :param rows: Per-query rows.
    :returns: Aggregated metrics with ``n_queries`` and ``n_hits@10``.
    :rtype: Dict[str, Optional[float]]
    """
    out: Dict[str, Optional[float]] = {}
    for key in HOLIDAY_METRICS:
        out[key] = round(float(np.mean([r[key] for r in rows])), 4) if rows else None
    hits = sum(r["hits@10"] for r in rows)
    alt = sum(r["alt_only_hits@10"] for r in rows)
    bases = [r["alt_only_base_rate"] for r in rows if r["alt_only_base_rate"] is not None]
    share = alt / hits if hits else None
    if hits:
        base = sum(r["hits@10"] * (r["alt_only_base_rate"] or 0.0) for r in rows) / hits
    else:
        base = float(np.mean(bases)) if bases else None
    out["alt_only_share"] = round(share, 4) if share is not None else None
    out["alt_only_base_rate"] = round(base, 4) if base is not None else None
    out["alt_only_lift"] = round(share / base, 3) if share is not None and base else None
    out["n_queries"] = len(rows)
    out["n_hits@10"] = int(hits)
    return out


def _holiday_section(per_query: Dict[str, dict], fest_info: Dict[str, dict], skipped: List[str]) -> dict:
    """Group holiday per-query rows into the reported section.

    :param per_query: ``"<festival>|<text>" -> row``.
    :param fest_info: Festival -> pool counts (``n_relevant``, ``n_canon``, ``n_unjudged``, ...).
    :param skipped: Festivals with fewer than :data:`MIN_RELEVANT` relevant pool records.
    :returns: ``{macro, by_form, per_festival, worst_festival, per_query, skipped_festivals_lt_min_relevant}``.
    :rtype: dict
    """
    by_form: Dict[str, List[dict]] = defaultdict(list)
    by_fest: Dict[str, List[dict]] = defaultdict(list)
    for r in per_query.values():
        by_form[r["form"]].append(r)
        by_fest[r["festival"]].append(r)
    per_festival = {f: {**_holiday_agg(v), **fest_info[f]} for f, v in sorted(by_fest.items())}
    worst = min(per_festival.items(), key=lambda kv: (kv[1]["P@10"], kv[0])) if per_festival else None
    return {"macro": _holiday_agg(list(per_query.values())),
            "by_form": {f: _holiday_agg(v) for f, v in sorted(by_form.items())},
            "per_festival": per_festival, "worst_festival": {"id": worst[0], **worst[1]} if worst else None,
            "per_query": per_query, "skipped_festivals_lt_min_relevant": sorted(skipped)}


def holiday(P: np.ndarray, pool_pos: Dict[str, int], qvec: Dict[str, np.ndarray], ev: dict) -> dict:
    """Holiday regression section: every name form of every festival over the whole pool (``holiday_queries.json``).

    Per festival the file lists ``relevant`` (strong carriers), ``canon`` (the festival named only on a scripture /
    tractate / law-code frame: "Purim" -> Esther, "פסח" -> Pesahim), ``unjudged`` (title-only, incidental and
    mention-only records, and Sukkot-gold records that are canon or incidental), ``alt_only`` (relevant records whose
    catalogue names the festival only under Alternate titles) and ``held_out_relevant``. Unjudged records are removed
    from the ranking; canon records are misses. Ties are broken pessimistically (:func:`pessimistic_top`;
    ``P@k`` / ``MRR@100`` come from :func:`positive_ranks`, which is consistent with it).

    Per query (keyed ``"<festival>|<text>"``): ``P@10``, ``P@20``, ``MRR@100``, ``canon_share@20`` (canon records in
    the top 20), ``alt_only_share`` (alt-only records among the relevant top-10 hits; None without a hit) with the
    festival's ``alt_only_base_rate`` (alt-only share of its relevant pool records) and ``alt_only_lift`` (share /
    base rate). Reported per form, per festival (with the worst festival by P@10) and as a macro over queries (the
    alt-only figures weighted by hits, :func:`_holiday_agg`). ``held_out`` repeats everything with relevant =
    ``held_out_relevant`` only, the train-side relevant records removed from the ranking (records the model never
    trained on; None when no festival has :data:`MIN_RELEVANT` held-out relevant records in the pool).

    Holiday files written before 2026-10-09 (``strong`` / ``weak`` / ``alt_only``) are read as relevant = strong,
    canon = weak, no unjudged; their held-out set is taken from ``held_out_records.json`` when the eval dir has it.

    :param P: Normalised pool matrix.
    :param pool_pos: ``doc_id -> row of P``.
    :param qvec: Query text -> normalised vector.
    :param ev: Loaded eval dir (``ev["holiday"]`` not None).
    :returns: Section dict (``held_out`` holds the held-out-only variant).
    :rtype: dict
    """
    N = P.shape[0]
    held = set(ev["held_out_records"]["held_out"]) if ev.get("held_out_records") else None

    def mask(ids: Iterable[str]) -> np.ndarray:
        m = np.zeros(N, dtype=bool)
        m[[pool_pos[d] for d in ids if d in pool_pos]] = True
        return m

    variants = {"all": ({}, {}, []), "held_out": ({}, {}, [])}  # (per_query, festival info, skipped)
    for sid, fest in ev["holiday"]["festivals"].items():
        relevant = fest.get("relevant", fest.get("strong", []))
        held_rel = fest.get("held_out_relevant")
        if held_rel is None and held is not None:
            held_rel = [d for d in relevant if d in held]
        rel_all = mask(relevant)
        unjudged = mask(fest.get("unjudged", [])) & ~rel_all
        canon = mask(fest.get("canon", fest.get("weak", []))) & ~rel_all & ~unjudged
        alt = mask(fest.get("alt_only", [])) & rel_all
        setups = {"all": (rel_all, ~unjudged)}
        if held_rel is not None:
            rel_held = mask(held_rel) & rel_all
            setups["held_out"] = (rel_held, ~unjudged & ~(rel_all & ~rel_held))  # train-side relevant left out
        for variant, (rel, in_rank) in setups.items():
            per_query, info, skipped = variants[variant]
            n_rel = int(rel.sum())
            if n_rel < MIN_RELEVANT:
                skipped.append(sid)
                continue
            base = float((alt & rel).sum() / n_rel)
            info[sid] = {"n_relevant": n_rel, "n_canon": int((canon & in_rank).sum()),
                         "n_unjudged": int(unjudged.sum()), "n_alt_only": int((alt & rel).sum()),
                         "alt_only_base_rate": round(base, 4)}
            if variant == "held_out":
                info[sid]["n_excluded_train_side"] = int((rel_all & ~rel).sum())
            cand = np.nonzero(in_rank)[0]
            rel_idx = np.nonzero(rel & in_rank)[0]
            neg_idx = np.nonzero(in_rank & ~rel)[0]
            for q in fest["queries"]:
                s = P @ qvec[q["text"]]
                ranks = positive_ranks(s[rel_idx], s[neg_idx])
                top = pessimistic_top(s, cand, rel, canon, 20)
                hits10 = int(rel[top[:10]].sum())
                alt10 = int(alt[top[:10]].sum())
                share = alt10 / hits10 if hits10 else None
                per_query[f"{sid}|{q['text']}"] = {
                    "festival": sid, "form": q["form"], "text": q["text"],
                    "P@10": float((ranks <= 10).sum() / 10), "P@20": float((ranks <= 20).sum() / 20),
                    "MRR@100": float(1.0 / ranks[0]) if len(ranks) and ranks[0] <= 100 else 0.0,
                    "canon_share@20": float(canon[top].sum() / 20), "hits@10": hits10, "alt_only_hits@10": alt10,
                    "alt_only_share": round(share, 4) if share is not None else None,
                    "alt_only_base_rate": round(base, 4),
                    "alt_only_lift": round(share / base, 3) if share is not None and base else None}
    out = _holiday_section(*variants["all"])
    held_section = _holiday_section(*variants["held_out"])
    out["held_out"] = held_section if held_section["per_query"] else None
    return out


def evaluate(doc_ids: Sequence[str], doc_matrix: np.ndarray, query_vectors_fn: Callable[[List[str]], np.ndarray],
             eval_dir, corpus_rows=None, ignore_ids: Optional[Iterable[str]] = None,
             sections: Sequence[str] = SECTIONS, chunk: int = 256, n_resamples: int = 1000,
             include_shelfmark_queries: bool = False,
             trained_subject_positives: Optional[Dict[str, Sequence[str]]] = None,
             text_hash_key: str = "text_hash") -> dict:
    """Score one model on the frozen v3 eval set.

    :param doc_ids: Record ids, one per row of ``doc_matrix`` (any superset/subset of the pool; extra rows such as
        ineligible records are ignored).
    :param doc_matrix: (n, d) document vectors of the v3 *semantic* text (cosine; normalised here if needed).
    :param query_vectors_fn: Callable mapping a list of query strings to an (n, d) array. It must apply the model's
        query instruction/prefix itself. It is called once with every unique query text.
    :param eval_dir: Directory written by ``build_eval_v3.py``.
    :param corpus_rows: Optional rows (or ``doc_id -> row``) of the corpus that was embedded. When given, the pool's
        ``text_hash`` is checked against ``row["text_hash"]`` (or the md5 of ``row["text"]``) and mismatches are
        reported: vectors of a different text version (e.g. production text with shelf marks) are flagged.
    :param ignore_ids: Records to remove from the pool entirely (e.g. ``old_split_trained`` when scoring a model
        trained on the v2 split).
    :param sections: Subset of :data:`SECTIONS` to run (hubness needs subjects).
    :param chunk: Queries per score block (memory ~ chunk x pool x 4 bytes).
    :param n_resamples: Bootstrap resamples for the memorisation-gap CI.
    :param include_shelfmark_queries: Score known-item queries that name a shelf mark too.
    :param trained_subject_positives: ``subject -> doc ids`` the training mixture used as positives for that subject's
        name anchors. When given, a second memorisation section ``memorisation_trained`` uses them as the train side
        (the frozen sample is only train-SIDE, so its gap is diluted by records the mixture never trained on). Pass
        the same mapping for every model of a comparison (base models included: their gap is the calibration).
    :param text_hash_key: Pool field the ``corpus_rows`` hashes are checked against: ``text_hash`` (md5 of the
        embedded text of the eval dir's corpus) or ``text_hash_v3`` (md5 of its ``semantic_v3`` text, for models
        scored on the v9 text when the eval dir was built from a ``semantic_v4`` corpus). Identical-text twins and
        the pool are always those of ``text_hash``, so every model of a comparison is scored on the same sets.
    :returns: Result dict (see the module docstring); ``summary(result)`` gives the headline numbers.
    :rtype: dict
    """
    t0 = time.time()
    ev = load_eval_dir(eval_dir)
    ignore = set(ignore_ids or [])
    row_of = {d: i for i, d in enumerate(doc_ids)}
    pool_rows = [r for r in ev["eval_pool"] if r["doc_id"] in row_of and r["doc_id"] not in ignore]
    pool_pos = {r["doc_id"]: i for i, r in enumerate(pool_rows)}
    P = normalise_rows(np.asarray(doc_matrix)[[row_of[r["doc_id"]] for r in pool_rows]])
    twins: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(pool_rows):
        twins[r["text_hash"]].append(i)
    twins_arr = {h: np.array(v, dtype=np.int64) for h, v in twins.items()}
    held = set(ev["held_out_records"]["held_out"])
    mismatch = None
    if corpus_rows is not None:
        rows = _rows_by_id(corpus_rows)
        mismatch = 0
        for r in pool_rows:
            c = rows.get(r["doc_id"])
            h = c.get("text_hash") if c else None
            if c is not None and h is None:
                h = hashlib.md5(c.get("text", "").encode("utf-8")).hexdigest()
            mismatch += h != r.get(text_hash_key)
    texts: List[str] = []
    if "subjects" in sections or "memorisation" in sections:
        texts += [q["text"] for e in ev["subject_queries"]["subjects"].values() for q in e["queries"]]
    if "known_item" in sections:
        texts += [q["text"] for q in ev["known_item"]]
    if "holiday" in sections and ev["holiday"]:
        texts += [q["text"] for f in ev["holiday"]["festivals"].values() for q in f["queries"]]
    texts = list(dict.fromkeys(texts))
    qmat = normalise_rows(query_vectors_fn(texts)) if texts else np.zeros((0, P.shape[1]), dtype=np.float32)
    if len(qmat) != len(texts):
        raise ValueError(f"query_vectors_fn returned {len(qmat)} vectors for {len(texts)} texts")
    qvec = dict(zip(texts, qmat))
    res = {"pool": {"n_pool_frozen": len(ev["eval_pool"]), "n_pool_scored": len(pool_rows),
                    "n_missing_vectors": sum(r["doc_id"] not in row_of for r in ev["eval_pool"]),
                    "n_ignored": sum(r["doc_id"] in ignore and r["doc_id"] in row_of for r in ev["eval_pool"]),
                    "text_hash_mismatch": mismatch, "text_hash_key": text_hash_key, "n_query_texts": len(texts)}}
    tops: List[np.ndarray] = []
    if "subjects" in sections:
        res["subjects"], tops = subject_retrieval(P, pool_pos, qvec, ev, chunk)
    if "memorisation" in sections:
        res["memorisation"] = memorisation(P, pool_pos, qvec, ev, n_resamples)
        if trained_subject_positives is not None:
            res["memorisation_trained"] = memorisation(P, pool_pos, qvec, ev, n_resamples,
                                                       train_override=trained_subject_positives)
    if "known_item" in sections:
        res["known_item"] = known_item(P, pool_pos, qvec, ev, twins_arr, chunk, include_shelfmark_queries)
    if "collection" in sections:
        res["collection"] = collection_lift(P, pool_pos, ev, held, chunk)
    if "hubness" in sections and tops:
        res["hubness"] = hubness(tops, ev, pool_pos)
    if "holiday" in sections and ev["holiday"]:
        res["holiday"] = holiday(P, pool_pos, qvec, ev)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summary(res: dict) -> dict:
    """Headline numbers of an :func:`evaluate` result (no per-query lists).

    :param res: Result dict.
    :returns: Compact dict.
    :rtype: dict
    """
    out = {"pool": res.get("pool")}
    if "subjects" in res:
        sub = res["subjects"]
        out["primary_group"] = sub.get("primary_group", "held_out")
        out["gate_held_out"] = sub["gate"]  # None when the eval dir holds no held-out subject (final run)
        out["held_out_by_source"] = sub["held_out"]["macro_by_source"]
        out["held_out_per_subject"] = {s: {k: v[k] for k in ("AP", "P@10", "R@100")}
                                       for s, v in sub["held_out"]["per_subject"].items()}
        if "focus" in sub:
            out["focus"] = sub["focus"]["macro"]
            out["focus_by_source"] = sub["focus"]["macro_by_source"]
            out["focus_per_subject"] = {s: {k: v.get(k) for k in ("AP", "P@10", "R@100", "n_relevant",
                                                                  "n_relevant_by_mode", "n_excluded_train_side",
                                                                  "by_source", "by_mode")}
                                        for s, v in sub["focus"]["per_subject"].items()}
            out["focus_by_mode"] = sub["focus"].get("macro_by_mode")
        out["linked"] = sub["linked"]["macro"]
        out["seen"] = sub["seen"]["macro"]
        out["seen_by_kind"] = sub["seen"]["macro_by_kind"]
        out["seen_by_mode"] = sub["seen"].get("macro_by_mode")
    if "memorisation" in res:
        out["memorisation"] = res["memorisation"]["macro"]
    if "memorisation_trained" in res:
        out["memorisation_trained"] = res["memorisation_trained"]["macro"]
    if "known_item" in res:
        ki = res["known_item"]
        out["known_item"] = {"all": ki["all"], "by_type": {t: {k: v.get(k) for k in ("n", "MRR", "R@10", "R@100")}
                                                            for t, v in ki["by_type"].items()},
                             "within_subject": ki["within_subject"]}
    for key in ("collection", "hubness"):
        if key in res:
            out[key] = {k: v for k, v in res[key].items() if k != "top_hubs"}
    if "holiday" in res:
        hol = res["holiday"]
        out["holiday"] = {k: v for k, v in hol.items() if k not in ("per_query", "held_out")}
        held = hol.get("held_out")
        out["holiday"]["held_out"] = {k: v for k, v in held.items() if k != "per_query"} if held else None
    return out


# ---------------------------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------------------------
def _write_eval_dir(path: Path, ev: dict) -> None:
    """Write an in-memory eval set in the :data:`FILES` layout (self-test helper).

    :param path: Target directory.
    :param ev: Dict keyed like :data:`FILES`.
    """
    path.mkdir(parents=True, exist_ok=True)
    for key, name in FILES.items():
        if name.endswith(".jsonl"):
            with open(path / name, "w", encoding="utf-8") as fh:
                for row in ev[key]:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            (path / name).write_text(json.dumps(ev[key], ensure_ascii=False), encoding="utf-8")


def _brute_ranks(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Reference implementation of :func:`positive_ranks` via a full pessimistic sort.

    :param pos: Positive scores.
    :param neg: Negative scores.
    :returns: Ascending ranks of the positives.
    :rtype: np.ndarray
    """
    scores = np.concatenate([pos, neg])
    rel = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.lexsort((rel, -scores))  # by score desc, ties: irrelevant first
    return np.sort(np.where(rel[order] == 1)[0] + 1)


def self_test(seed: int = 0) -> None:
    """Tiny synthetic checks: rank primitives, ties, twins, planted-signal model beats random, bootstrap.

    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    # 1. rank primitive vs brute force, including heavy ties
    for _ in range(200):
        pos = rng.integers(0, 6, size=rng.integers(1, 8)).astype(float)
        neg = rng.integers(0, 6, size=rng.integers(0, 30)).astype(float)
        assert np.array_equal(positive_ranks(pos, neg), _brute_ranks(pos, neg)), (pos, neg)
    assert ranking_metrics(np.array([1, 2]), 2)["AP"] == 1.0
    assert hit_rank(np.array([0.5, 0.5, 0.9, 0.1]), np.array([0])) == 3          # tie with idx1 counts against
    assert hit_rank(np.array([0.5, 0.5, 0.9, 0.1]), np.array([0, 1])) == 2       # twin counts as hit
    assert hit_rank(np.array([0.5, 0.5, 0.9, 0.1]), np.array([0]), np.array([0, 3])) == 1

    # 2. synthetic eval dir: 4 subjects, 400 docs (some twins / short), 2 statuses + seen
    d, n = 32, 400
    centres = normalise_rows(rng.normal(size=(4, d)))
    sids = ["topic:a", "pgp:b", "work:c", "domain:d"]
    status = {"topic:a": "held_out", "pgp:b": "held_out", "work:c": "linked", "domain:d": "seen"}
    doc_ids = [f"Series{i % 7}_{i}" for i in range(n)]
    label = rng.integers(-1, 4, size=n)                       # -1 = no subject
    hashes = [f"h{i}" for i in range(n)]
    for i in range(0, 20, 2):                                  # 10 twin pairs
        hashes[i + 1] = hashes[i]
        label[i + 1] = label[i]
    strong = {s: [doc_ids[i] for i in range(n) if label[i] == k] for k, s in enumerate(sids)}
    weak = {s: [] for s in sids}
    weak["topic:a"] = [doc_ids[i] for i in range(n) if label[i] == 1][:5]   # weak-only labels on another subject
    decoys = [i for i in range(20, n) if label[i] == -1][:10]                # unlabelled records "about" topic:a
    held = [doc_ids[i] for i in range(n) if i % 5 == 0]
    pool = [{"doc_id": doc_ids[i], "text_hash": hashes[i], "series": f"Series{i % 7}",
             "n_chars": 40 if i % 50 == 0 else 300, "dup_n": 2 if i < 20 else 1, "held_out": doc_ids[i] in held}
            for i in range(n)]
    queries = {s: {"status": status[s], "kind": s.split(":")[0],
                   "queries": [{"text": f"{s} q{j}", "source": "probe" if j < 3 else "name", "lang": "en"}
                               for j in range(5)]} for s in sids}
    mem_train = [x for x in strong["domain:d"] if x not in held][:10]
    mem_held = [x for x in strong["domain:d"] if x in held][:10]
    ki = [{"qid": f"r1:{doc_ids[i]}:0", "doc_id": doc_ids[i], "type": "implicit" if i % 2 else "hebrew",
           "text": f"known {i}", "shelfmark_query": i == 3} for i in range(0, 60, 3)]
    ev = {"held_out_subjects": {"subjects": []}, "held_out_records": {"held_out": held},
          "subject_queries": {"subjects": queries},
          "subject_relevance": {s: {"strong": strong[s], "weak": weak[s]} for s in sids},
          "known_item": ki, "memorization_pairs": {"subjects": {"domain:d": {"train": mem_train, "held_out": mem_held}}},
          "collection_sample": {"doc_ids": [x for i, x in enumerate(doc_ids) if i >= 20][:150]}, "eval_pool": pool}
    base = normalise_rows(rng.normal(size=(n, d)))
    good = base.copy()
    for i in range(n):
        if label[i] >= 0:
            good[i] = centres[label[i]] + 0.6 * base[i]
    for i in range(0, 20, 2):
        good[i + 1] = good[i]
    for i in decoys:                                         # exactly on the topic:a centre: they outrank its carriers
        good[i] = centres[0]
    good = normalise_rows(good)
    memo = good.copy()                    # a model that memorised its train-side records and never learnt the subject
    for x in mem_held:
        memo[doc_ids.index(x)] = base[doc_ids.index(x)]
    qtext = {f"{s} q{j}": centres[k] + 0.3 * normalise_rows(rng.normal(size=(1, d)))[0]
             for k, s in enumerate(sids) for j in range(5)}
    for q in ki:
        qtext[q["text"]] = good[doc_ids.index(q["doc_id"])] + 0.2 * normalise_rows(rng.normal(size=(1, d)))[0]

    def qfn_good(texts: List[str]) -> np.ndarray:
        return np.stack([qtext[t] for t in texts])

    def qfn_rand(texts: List[str]) -> np.ndarray:
        return rng.normal(size=(len(texts), d))

    with tempfile.TemporaryDirectory() as tmp:
        _write_eval_dir(Path(tmp), ev)
        r_judged = evaluate(doc_ids, good, qfn_good, tmp, sections=("subjects",))
        ev["subject_relevance"]["topic:a"]["unjudged"] = [doc_ids[i] for i in decoys]
        _write_eval_dir(Path(tmp), ev)
        r_good = evaluate(doc_ids, good, qfn_good, tmp, corpus_rows=pool, n_resamples=200)
        r_rand = evaluate(doc_ids, base, qfn_rand, tmp, n_resamples=200)
        r_ign = evaluate(doc_ids, good, qfn_good, tmp, ignore_ids=held[:10], sections=("subjects",))
        r_memo = evaluate(doc_ids, memo, qfn_good, tmp, sections=("memorisation",), n_resamples=200)
        r_memo_tr = evaluate(doc_ids, memo, qfn_good, tmp, sections=("memorisation",), n_resamples=200,
                             trained_subject_positives={"domain:d": mem_train})
        r_good_tr = evaluate(doc_ids, good, qfn_good, tmp, sections=("memorisation",), n_resamples=200,
                             trained_subject_positives={"domain:d": mem_train})
    g, rr = r_good["subjects"]["gate"]["AP"], r_rand["subjects"]["gate"]["AP"]
    assert g > 0.8 and rr < 0.5, (g, rr)
    ap_a = r_good["subjects"]["held_out"]["per_subject"]["topic:a"]["AP"]
    ap_a_judged = r_judged["subjects"]["held_out"]["per_subject"]["topic:a"]["AP"]
    assert ap_a > ap_a_judged + 0.1, (ap_a, ap_a_judged)                      # unjudged decoys left out
    assert abs(r_good["memorisation"]["macro"]["gap"]) < 0.2, r_good["memorisation"]["macro"]
    assert r_memo["memorisation"]["macro"]["gap"] > 0.2, r_memo["memorisation"]["macro"]
    assert r_memo_tr["memorisation_trained"]["macro"]["gap"] > 0.2, r_memo_tr["memorisation_trained"]["macro"]
    assert abs(r_good_tr["memorisation_trained"]["macro"]["gap"]) < 0.2, r_good_tr["memorisation_trained"]["macro"]
    assert "memorisation_trained" not in r_memo
    net = compare(r_good_tr, r_memo_tr, "memorisation_trained", "gap", n_resamples=200)
    assert net["delta"] > 0.2, net
    assert r_good["pool"]["text_hash_mismatch"] == 0
    assert r_ign["pool"]["n_pool_scored"] == n - 10
    assert set(r_good["subjects"]["held_out"]["per_subject"]) == {"topic:a", "pgp:b"}
    assert "work:c" in r_good["subjects"]["linked"]["per_subject"]
    assert r_good["known_item"]["skipped"].get("shelfmark_query") == 1
    assert r_good["known_item"]["all"]["MRR"] > r_rand["known_item"]["all"]["MRR"]
    assert r_good["known_item"]["within_subject"]["n"] > 0
    assert 0 <= r_good["hubness"]["share"] <= 1 and r_good["collection"]["all"]["n"] == 150
    assert abs(r_rand["collection"]["all"]["lift"] - 1) < 0.5, r_rand["collection"]["all"]
    cmp = compare(r_rand, r_good, "held_out", "AP", n_resamples=300)
    assert cmp["delta"] > 0 and cmp["ci_low"] > 0, cmp
    cmp_ki = compare(r_rand, r_good, "known_item", "RR", n_resamples=300)
    assert cmp_ki["delta"] > 0, cmp_ki
    same = compare(r_good, r_good, "seen", "AP", n_resamples=100)
    assert same["delta"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    assert r_good["subjects"]["primary_group"] == "held_out" and "focus" not in r_good["subjects"]
    assert summary(r_good)["primary_group"] == "held_out" and "focus" not in summary(r_good)
    print(json.dumps({"self_test": "ok", "gate_good": g, "gate_random": rr, "bootstrap": cmp,
                      "topic_a_AP_unjudged_vs_judged": [ap_a, ap_a_judged],
                      "memorisation_good": r_good["memorisation"]["macro"],
                      "memorisation_memorising": r_memo["memorisation"]["macro"],
                      "collection_random": r_rand["collection"]["all"],
                      "known_item_good": r_good["known_item"]["all"]}, indent=1))


def self_test_final(seed: int = 1) -> None:
    """Final-run mode: no held-out subject, two ``focus`` subjects scored over held-out records only.

    Checks: no gate and ``primary_group == "focus"``; the focus relevant set is the held-out carriers; moving the
    train-side carriers anywhere (even exactly onto the query) leaves focus AP unchanged (they are out of the
    ranking); a model that only memorised its train-side carriers scores near chance; ``compare`` / ``summary`` work
    on the focus group and degrade gracefully on the empty held-out group.

    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    d, n = 32, 500
    sids = ["topic:a", "pgp:b", "work:c", "domain:d"]
    status = {"topic:a": "focus", "pgp:b": "focus", "work:c": "seen", "domain:d": "seen"}
    centres = normalise_rows(rng.normal(size=(4, d)))
    doc_ids = [f"Series{i % 7}_{i}" for i in range(n)]
    label = rng.integers(-1, 4, size=n)
    hashes = [f"h{i}" for i in range(n)]
    held = {doc_ids[i] for i in range(n) if i % 5 == 0}
    strong = {s: [doc_ids[i] for i in range(n) if label[i] == k] for k, s in enumerate(sids)}
    pool = [{"doc_id": doc_ids[i], "text_hash": hashes[i], "series": f"Series{i % 7}", "n_chars": 300, "dup_n": 1,
             "held_out": doc_ids[i] in held} for i in range(n)]
    queries = {s: {"status": status[s], "kind": s.split(":")[0],
                   "queries": [{"text": f"{s} q{j}", "source": ("probe", "llm_t2", "name")[j % 3], "lang": "en"}
                               for j in range(6)]} for s in sids}
    mem_train = [x for x in strong["domain:d"] if x not in held][:10]
    mem_held = [x for x in strong["domain:d"] if x in held][:10]
    ki = [{"qid": f"r1:{doc_ids[i]}:0", "doc_id": doc_ids[i], "type": "implicit", "text": f"known {i}",
           "shelfmark_query": False} for i in range(0, 100, 5)]
    ev = {"held_out_subjects": {"mode": "final", "subjects": [], "linked": {}, "concept_patterns": {}},
          "held_out_records": {"held_out": sorted(held)}, "subject_queries": {"subjects": queries},
          "subject_relevance": {s: {"strong": strong[s], "weak": []} for s in sids}, "known_item": ki,
          "memorization_pairs": {"subjects": {"domain:d": {"train": mem_train, "held_out": mem_held}}},
          "collection_sample": {"doc_ids": doc_ids[:150]}, "eval_pool": pool}
    base = normalise_rows(rng.normal(size=(n, d)))
    good = base.copy()
    for i in range(n):
        if label[i] >= 0:
            good[i] = centres[label[i]] + 0.6 * base[i]
    good = normalise_rows(good)
    focus_idx = [i for i in range(n) if label[i] in (0, 1)]
    train_focus = [i for i in focus_idx if doc_ids[i] not in held]
    on_query = good.copy()                       # topic:a's train-side carriers exactly on its centre
    for i in train_focus:
        if label[i] == 0:
            on_query[i] = centres[0]
    memo = base.copy()                            # only the train-side focus carriers were learnt
    for i in train_focus:
        memo[i] = centres[label[i]]
    qtext = {f"{s} q{j}": centres[k] + 0.3 * normalise_rows(rng.normal(size=(1, d)))[0]
             for k, s in enumerate(sids) for j in range(6)}
    for q in ki:
        qtext[q["text"]] = good[doc_ids.index(q["doc_id"])]

    def qfn(texts: List[str]) -> np.ndarray:
        return np.stack([qtext[t] for t in texts])

    def qfn_rand(texts: List[str]) -> np.ndarray:
        return rng.normal(size=(len(texts), d))

    hol = {"festivals": {"topic:a": {"queries": [{"text": "topic:a q0", "form": "translit"}],
                                     "relevant": strong["topic:a"], "canon": strong["pgp:b"][:5], "unjudged": [],
                                     "alt_only": strong["topic:a"][:3]}}}  # no held_out_relevant: from held_out_records
    with tempfile.TemporaryDirectory() as tmp:
        _write_eval_dir(Path(tmp), ev)
        (Path(tmp) / HOLIDAY_FILE).write_text(json.dumps(hol), encoding="utf-8")
        r_good = evaluate(doc_ids, good, qfn, tmp, n_resamples=200)
        r_onq = evaluate(doc_ids, on_query, qfn, tmp, sections=("subjects",))
        r_memo = evaluate(doc_ids, memo, qfn, tmp, sections=("subjects",))
        r_rand = evaluate(doc_ids, base, qfn_rand, tmp, n_resamples=200)
    sub = r_good["subjects"]
    assert sub["gate"] is None and sub["primary_group"] == "focus", (sub["gate"], sub["primary_group"])
    assert sub["held_out"]["macro"] == {"n_subjects": 0} and not sub["held_out"]["per_query"]
    assert set(sub["focus"]["per_subject"]) == {"topic:a", "pgp:b"} and set(sub["seen"]["per_subject"]) == {
        "work:c", "domain:d"}
    for k, s in enumerate(sids[:2]):
        ps = sub["focus"]["per_subject"][s]
        assert ps["n_relevant"] == sum(x in held for x in strong[s]), ps
        assert ps["n_excluded_train_side"] == sum(x not in held for x in strong[s]), ps
        assert set(ps["by_source"]) == {"probe", "llm_t2", "name"}, ps
    assert set(sub["focus"]["macro_by_source"]) == {"probe", "llm_t2", "name"}
    f_good, f_rand = sub["focus"]["macro"]["AP"], r_rand["subjects"]["focus"]["macro"]["AP"]
    f_memo = r_memo["subjects"]["focus"]["macro"]["AP"]
    a_good = sub["focus"]["per_query"]["topic:a"]["AP"]
    a_onq = r_onq["subjects"]["focus"]["per_query"]["topic:a"]["AP"]
    f_onq = r_onq["subjects"]["focus"]["per_subject"]["topic:a"]["AP"]
    assert f_good > 0.8 and a_onq == a_good, (a_good, a_onq)               # train-side carriers out of the ranking
    assert f_memo < 0.3 and f_rand < 0.3, (f_memo, f_rand)               # memorised train side earns nothing
    cmp_f = compare(r_rand, r_good, "focus", "AP", n_resamples=300)
    assert cmp_f["delta"] > 0 and cmp_f["ci_low"] > 0 and cmp_f["n_clusters"] == 2, cmp_f
    empty = compare(r_rand, r_good, "held_out", "AP", n_resamples=100)
    assert empty["n_clusters"] == 0 and empty["delta"] == 0, empty
    assert compare(r_rand, r_good, "linked", "AP", n_resamples=100)["n_clusters"] == 0
    s = summary(r_good)
    assert s["primary_group"] == "focus" and s["gate_held_out"] is None and s["focus"]["n_subjects"] == 2
    assert s["focus_per_subject"]["topic:a"]["n_relevant"] > 0 and "memorisation" in s and "known_item" in s
    assert r_good["known_item"]["all"]["MRR"] > r_rand["known_item"]["all"]["MRR"]
    hg = r_good["holiday"]
    assert hg["per_festival"]["topic:a"]["n_relevant"] == len(strong["topic:a"]) and hg["macro"]["P@10"] > 0.5, hg
    assert hg["held_out"]["per_festival"]["topic:a"]["n_relevant"] == sum(x in held for x in strong["topic:a"])
    assert s["holiday"]["macro"] == hg["macro"] and s["holiday"]["held_out"]["macro"] == hg["held_out"]["macro"]
    assert compare(r_rand, r_good, "holiday_held_out", "MRR@100", n_resamples=100)["n_clusters"] == 1
    print(json.dumps({"self_test_final": "ok", "focus_good": f_good, "focus_train_side_on_query": f_onq,
                      "focus_memorised_train_side": f_memo, "focus_random": f_rand, "bootstrap_focus": cmp_f,
                      "empty_held_out_compare": empty}, indent=1))


def self_test_rel_mode(seed: int = 3) -> None:
    """Subject ``rel_mode`` and ``unjudged`` on a toy pool (no files).

    One festival-like subject: 15 strong carriers, 10 canon carriers that are weak (festival canon) and 3 that are
    strong but listed as canon (non-festival canon), 5 unjudged records. A ``bare`` query sitting on the canon
    records is penalised (canon = misses) while a ``law`` query at the same place is not (canon = relevant); leaving
    the canon list out reproduces the old rule (weak canon carriers dropped from the ranking); unjudged records that
    sit exactly on the query are removed from the ranking; focus subjects restrict either mode to held-out records.

    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    n, d = 200, 16
    P = normalise_rows(rng.normal(size=(n, d)))
    ids = [f"d{i}" for i in range(n)]
    pool_pos = {x: i for i, x in enumerate(ids)}
    S, C = normalise_rows(rng.normal(size=(2, d)))
    strong_only, weak_canon, strong_canon, unj = ids[:15], ids[15:25], ids[25:28], ids[28:33]
    for x in strong_only:
        P[pool_pos[x]] = S + 0.5 * P[pool_pos[x]]
    for x in weak_canon + strong_canon:
        P[pool_pos[x]] = C + 0.3 * P[pool_pos[x]]
    for x in unj:
        P[pool_pos[x]] = S
    P = normalise_rows(P)
    rel = {"topic:f": {"strong": strong_only + strong_canon, "weak": weak_canon,
                       "canon": weak_canon + strong_canon, "unjudged": unj}}
    queries = [{"text": "f bare on canon", "source": "name", "rel_mode": "bare"},
               {"text": "laws of f on canon", "source": "name", "rel_mode": "law"},
               {"text": "f bare on strong", "source": "name", "rel_mode": "bare"},
               {"text": "f probe on canon", "source": "probe", "rel_mode": "law"}]
    qvec = {"f bare on canon": C, "laws of f on canon": C, "f bare on strong": S, "f probe on canon": C}
    ev = {"subject_relevance": rel, "held_out_records": {"held_out": ids[::2]},
          "subject_queries": {"subjects": {"topic:f": {"status": "seen", "kind": "topic", "queries": queries}}}}
    out, _ = subject_retrieval(P, pool_pos, qvec, ev)
    ps = out["seen"]["per_subject"]["topic:f"]
    ap = dict(zip([q["text"] for q in queries], out["seen"]["per_query"]["topic:f"]["AP"]))
    assert ps["n_relevant_by_mode"] == {"bare": 15, "law": 28} and ps["n_relevant"] == 15, ps
    assert ap["f bare on canon"] < 0.3 < ap["laws of f on canon"], ap                 # canon: miss vs relevant
    assert ap["f bare on strong"] > 0.9, ap                                            # unjudged on the query removed
    assert set(ps["by_mode"]) == {"bare", "law"} and out["seen"]["macro_by_mode"]["law"]["n_subjects"] == 1
    assert out["seen"]["per_query"]["topic:f"]["rel_mode"] == ["bare", "law", "bare", "law"]
    # old rule (no canon list): the weak canon carriers are left out of the bare ranking instead of counted as misses
    old = {"topic:f": {k: v for k, v in rel["topic:f"].items() if k != "canon"}}
    out_old, _ = subject_retrieval(P, pool_pos, qvec, {**ev, "subject_relevance": old})
    ap_old = out_old["seen"]["per_query"]["topic:f"]["AP"][0]
    assert ap_old > ap["f bare on canon"] + 0.1, (ap_old, ap["f bare on canon"])
    # focus: only held-out relevant records are judged, in both modes
    held = set(ids[::2])
    r_bare, x_bare, t_bare = subject_sets(rel["topic:f"], "bare", held)
    r_law, x_law, t_law = subject_sets(rel["topic:f"], "law", held)
    assert r_bare == {x for x in strong_only if x in held} and t_bare == {x for x in strong_only if x not in held}
    assert r_law == {x for x in strong_only + weak_canon + strong_canon if x in held} and t_law <= x_law
    assert not (set(weak_canon) - set(unj)) & x_bare  # weak canon carriers stay in the bare ranking (misses)
    # rel_mode-restricted paired comparison
    res_a = {"subjects": out}
    res_b = {"subjects": out_old}
    cmp_law = compare(res_a, res_b, "seen", "AP", n_resamples=100, rel_mode="law")
    assert cmp_law["n_queries"] == 2 and compare(res_a, res_b, "seen", "AP", n_resamples=100)["n_queries"] == 4
    print(json.dumps({"self_test_rel_mode": "ok", "AP": ap, "bare_on_canon_old_rule": ap_old}))


def self_test_holiday(seed: int = 2) -> None:
    """Holiday section on a toy pool.

    Checks: a query sitting on the canon carriers scores P@10 0 and a high ``canon_share@20``; a query sitting on the
    festival records scores P@10 1 and reports the alt-title-only share, base rate and lift; unjudged records on top
    of the query are removed from the ranking; a collapsed model (every vector equal) gets the pessimistic tie
    outcome exactly (P@10 0, canon first among the ties); the held-out variant leaves the train-side relevant records
    out and does not reward a model that only learnt them; per-query keys are ``<festival>|<text>``; pre-2026-10-09
    files (``strong`` / ``weak``) read the same; the festival-clustered paired comparison works.

    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    n, d = 80, 16
    P = normalise_rows(rng.normal(size=(n, d)).astype(np.float32))
    ids = [f"r{i}" for i in range(n)]
    pool_pos = {x: i for i, x in enumerate(ids)}
    relevant, canon, unjudged = ids[:12], ids[12:24], ids[24:30]
    held_rel, alt = ids[:6], ids[:4]
    centre = lambda xs: normalise_rows(P[[pool_pos[x] for x in xs]].mean(0, keepdims=True))[0]
    sharpen = lambda xs: normalise_rows((P[[pool_pos[x] for x in xs]] * 5).sum(0, keepdims=True))[0]
    P[:12] = normalise_rows(P[:12] + 3 * centre(relevant))
    P[12:24] = normalise_rows(P[12:24] + 3 * centre(canon))
    q_unj = normalise_rows((sharpen(relevant) + centre(unjudged))[None])[0]
    P[24:30] = q_unj                                       # unjudged records exactly on the "unj" query
    queries = [{"text": "Purim", "form": "translit"}, {"text": "פורים", "form": "hebrew"},
               {"text": "unj", "form": "typo"}]
    fest = {"queries": queries, "relevant": relevant, "canon": canon, "unjudged": unjudged, "alt_only": alt,
            "held_out_relevant": held_rel}
    ev = {"holiday": {"festivals": {"topic:purim": fest}}}
    qvec = {"Purim": sharpen(canon), "פורים": sharpen(relevant), "unj": q_unj}
    res = holiday(P, pool_pos, qvec, ev)
    bad, good, unj = (res["per_query"][f"topic:purim|{t}"] for t in ("Purim", "פורים", "unj"))
    assert bad["P@10"] == 0.0 and bad["canon_share@20"] >= 0.5 and bad["alt_only_share"] is None, bad
    assert good["P@10"] == 1.0 and good["canon_share@20"] == 0.0 and good["hits@10"] == 10, good
    assert good["alt_only_base_rate"] == round(4 / 12, 4) and good["alt_only_share"] == good["alt_only_hits@10"] / 10
    assert good["alt_only_lift"] == round(good["alt_only_share"] / (4 / 12), 3), good
    assert unj["P@10"] == 1.0, unj                                            # unjudged removed from the ranking
    no_unj = holiday(P, pool_pos, qvec, {"holiday": {"festivals": {"topic:purim": {**fest, "unjudged": []}}}})
    assert no_unj["per_query"]["topic:purim|unj"]["P@10"] <= 0.4, no_unj["per_query"]["topic:purim|unj"]
    assert res["worst_festival"]["id"] == "topic:purim" and res["by_form"]["hebrew"]["P@10"] == 1.0
    pf = res["per_festival"]["topic:purim"]
    assert pf["n_relevant"] == 12 and pf["n_canon"] == 12 and pf["n_unjudged"] == 6 and pf["n_alt_only"] == 4, pf
    m = res["macro"]
    assert m["n_hits@10"] == bad["hits@10"] + good["hits@10"] + unj["hits@10"]
    alt_hits = good["alt_only_hits@10"] + unj["alt_only_hits@10"]
    assert m["alt_only_share"] == round(alt_hits / m["n_hits@10"], 4) and m["alt_only_base_rate"] == round(4 / 12, 4)
    # collapsed model: every score tied -> irrelevant (canon first) above every relevant record
    flat = np.tile(normalise_rows(rng.normal(size=(1, d)))[0], (n, 1))
    res_flat = holiday(flat, pool_pos, qvec, ev)
    f = res_flat["per_query"]["topic:purim|פורים"]
    n_irrelevant = n - len(relevant) - len(unjudged)
    assert f["P@10"] == 0.0 and f["P@20"] == 0.0 and f["canon_share@20"] == 12 / 20, f
    assert abs(f["MRR@100"] - 1 / (n_irrelevant + 1)) < 1e-9, f
    top = pessimistic_top(np.zeros(n), np.arange(n), np.isin(np.arange(n), range(12)),
                          np.isin(np.arange(n), range(12, 24)), 20)
    assert set(top[:12].tolist()) == set(range(12, 24)) and not set(top.tolist()) & set(range(12)), top
    # held-out variant: a model that learnt only the train-side relevant records scores well on the whole pool only
    memo = P.copy()
    memo[6:12] = sharpen(relevant)                          # train-side relevant exactly on the query
    memo[:6] = normalise_rows(rng.normal(size=(6, d)))      # held-out relevant records: random
    res_memo = holiday(memo, pool_pos, qvec, ev)
    whole = res_memo["per_query"]["topic:purim|פורים"]["P@10"]
    held_only = res_memo["held_out"]["per_query"]["topic:purim|פורים"]["P@10"]
    assert whole >= 0.6 and held_only <= 0.2, (whole, held_only)
    assert res_memo["held_out"]["per_festival"]["topic:purim"]["n_relevant"] == 6
    assert res_memo["held_out"]["per_festival"]["topic:purim"]["n_excluded_train_side"] == 6
    # pre-2026-10-09 file shape reads as relevant = strong, canon = weak, no unjudged
    old_fest = {"queries": queries, "strong": relevant, "weak": canon, "alt_only": alt}
    res_old = holiday(P, pool_pos, qvec, {"holiday": {"festivals": {"topic:purim": old_fest}}})
    assert res_old["per_query"]["topic:purim|Purim"] == no_unj["per_query"]["topic:purim|Purim"]
    assert res_old["held_out"] is None
    cmp_h = compare({"holiday": res_flat}, {"holiday": res}, "holiday", "P@10", n_resamples=100)
    assert cmp_h["delta"] > 0 and cmp_h["n_clusters"] == 1 and cmp_h["n_queries"] == 3, cmp_h
    s = summary({"holiday": res})
    assert "per_query" not in s["holiday"] and "per_query" not in s["holiday"]["held_out"]
    print(json.dumps({"self_test_holiday": "ok", "macro": res["macro"],
                      "held_out_memorised": [whole, held_only], "collapsed": f}, ensure_ascii=False))


def random_smoke(eval_dir: str, dim: int = 64, seed: int = 0) -> None:
    """Run :func:`evaluate` on the real eval files with random vectors (loading/timing check; numpy only).

    Random vectors should give gate AP close to the subjects' base rate, collection lift close to 1 and known-item
    MRR close to 0.

    :param eval_dir: Real eval dir.
    :param dim: Random vector dimension (small to keep memory tiny).
    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    ev_pool = [json.loads(line) for line in open(Path(eval_dir) / FILES["eval_pool"], encoding="utf-8")]
    ids = [r["doc_id"] for r in ev_pool]
    mat = rng.normal(size=(len(ids), dim)).astype(np.float32)
    res = evaluate(ids, mat, lambda t: rng.normal(size=(len(t), dim)).astype(np.float32), eval_dir,
                   n_resamples=200)
    print(json.dumps(summary(res), indent=1, ensure_ascii=False))
    print("seconds", res["seconds"])


def main() -> None:
    """CLI: ``--self-test`` or ``--random-smoke``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--random-smoke", action="store_true")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        self_test_final()
        self_test_rel_mode()
        self_test_holiday()
    if args.random_smoke:
        random_smoke(args.eval_dir, args.dim)


if __name__ == "__main__":
    main()
