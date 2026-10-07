"""Join and scribe-attribution discovery candidates from cached fragment image features.

No model is run: everything comes from ``cache/image_feats`` (the 5,176 probe-gallery
images scored by ``eval_images.py``) plus the labels in ``image_manifest_v1.json`` and the
merged catalogue. Outputs go to ``AUDIT_ROOT/results/discovery_v1/`` (CSV + contact sheets
+ ``summary.json``) for scholarly review; the script never judges the images itself.

Joins
-----
1. *Known structure.* Fragments are grouped by every known link: FJP join lists (all,
   including the prefix-bug suspects), PGP shared-document pairs (any group size) and FJP
   join lists that the manifest could not resolve but that loosely match a gallery
   shelfmark. Pairs inside one group are never proposed; duplicate records of one shelfmark
   and near-identical photographs are listed separately.
2. *Similarity.* The best join fusion (DINOv2 masked + masked ink patches + v22b tower),
   ranked by CSLS when that beats raw cosine on the known joins. "Cross-collection" means a
   different holding (:func:`holding_key`), not a different manifest collection string.
3. *Calibration.* Positives are the clean FJP + PGP join pairs (plus the loose FJP
   rediscoveries); every other non-grouped gallery pair is a negative. Reported: (a)
   labelled precision vs. fused-similarity threshold per same/cross stratum over all ~13M
   gallery pairs; (b) precision vs. neighbour rank and mutual rank; (c) a logistic score
   ``f(sim, ranks, cross)`` on the top-k neighbour pool mapped through an isotonic fit, giving
   each candidate a *labelled-gallery* precision ``p_lab``. ``p_lab``, its reliability table and
   its AUC are OUT-OF-FOLD: the pool is split into folds by known-join component, and every pair
   is scored by a model fitted without its component (an in-sample isotonic fit is calibrated on
   its own data by construction, so in-sample reliability proves nothing).
4. *Base-rate correction.* ``p_lab`` is measured where the gallery was built around known
   joins (every known partner is present). If the gallery holds ``rho`` undiscovered joins
   per known join and they look like known ones, a non-known pair is a join with
   probability ``rho * p_lab / (1 - p_lab)``. ``rho`` is not observable; it is bracketed from
   catalogue-level join density (:func:`rho_estimate`), and the odds transfer itself is
   validated by hiding one label source (PGP or FJP) and comparing recovered vs. predicted.
5. *Compatibility filter.* KTIV material / script type / style / region drop a pair only when
   both values are present and differ, and only for attributes whose conflict rate on known
   joins is low.

Scribes
-------
Leave-one-out nearest-centroid attribution over the PGP scribes with >= 5 labelled
fragments (same-document fragments never inform the query). A margin threshold with LOO
precision >= 0.8 is chosen. Because that threshold is picked on the same LOO predictions, its
precision is re-estimated by nested cross-validation (threshold chosen on K-1 folds of labelled
fragments grouped by PGP document, applied to the held-out fold); the method-of-moments estimate
uses the cross-validated precision and coverage. Unlabelled documentary fragments above it are
candidates, except those already tied to a scribe-labelled fragment by a known join or a shared
PGP document (their attribution is known, not discovered). How many are right is estimated two
ways: (a) method of moments with the leave-one-scribe-out open-set false-positive rate; (b) a
date check, since PGP dates (and century tags) that fall outside the proposed scribe's active
years expose wrong attributions (corrected by how often LOO errors are date-incompatible).
"""

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw

from embed_utils import AUDIT_ROOT
from eval_images import JOIN_KEY, load_feats, pair_auc
from image_features import IMG_DIR, fragment_mask

HERE = Path(__file__).parent
HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
MERGED = HDA / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
FJP_ALL = HDA / "src/datasets/raw_data/cairo_genizah/fjp_all/merged_princeton_friedberger_all_documents_final.json"
OUT = AUDIT_ROOT / "results" / "discovery_v1"
JOIN_FEATURES = "dinov2-base__masked+dinov2-base__mpatches+qwen3vl-vit512-heb-v22b-step1200__masked"
SCRIBE_FEATURES = "qwen3vl-vit512-heb-v22b-step1200__masked"
DOCUMENTARY_TYPES = {"Letter", "Legal document", "List or table", "State document", "Legal query or responsum",
                     "Credit instrument or private receipt"}
ATTR_KEYS = ("material", "script_type", "script_style", "script_region")
GENIZAH_SIZE = 400_000  # order-of-magnitude count of Genizah fragments (base-rate prior only)


# ----------------------------------------------------------------------------- metadata

def shelf_key(shelfmark: str) -> str:
    """Normalise an FJP-style shelfmark (same rule as ``build_image_manifest.shelf_key``).

    Re-implemented here because importing the manifest builder pulls in the sibling
    project's document models.

    :param shelfmark: e.g. ``"Cambridge, CUL:  T-S F1(2).11"``.
    :returns: e.g. ``"cambridge_cul_t_s_f1_2_11"``.
    :rtype: str
    """
    s = re.sub(r"\(alt[^)]*\)", "", shelfmark or "", flags=re.I)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def pgp_years(docs: List[dict]) -> List[int]:
    """Years (CE) named in the PGP standardised or inferred dates of a fragment's documents.

    :param docs: PGP document dicts.
    :returns: Sorted unique years in 600-1900.
    :rtype: List[int]
    """
    years = set()
    for d in docs:
        for key in ("doc_date_standard", "inferred_date_standard"):
            years |= {int(y) for y in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", d.get(key) or "") if 600 <= int(y) <= 1900}
    return sorted(years)


def gallery_meta(ids: Sequence[str], cache: Path, extra_ids: Iterable[str] = ()) -> Dict[str, dict]:
    """Collect shelfmark, PGP (ids, types, tags, years) and FJP fields for the given fragments (cached as JSON).

    :param ids: Gallery fragment ids.
    :param cache: JSON cache path (the merged catalogue is ~700 MB, so it is streamed once).
    :param extra_ids: Further ids to include (scribe-labelled fragments outside the gallery, for date windows).
    :returns: cid -> {shelfmark, institution, pgpids, pgp_types, pgp_tags, years, fjp_shelfmarks}.
    :rtype: Dict[str, dict]
    """
    wanted = set(ids) | set(extra_ids)
    if cache.exists():
        meta = json.loads(cache.read_text())
        if wanted <= set(meta) and all("years" in m for m in meta.values()):
            return meta
    meta = {}
    with open(MERGED, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid = rec["canonical_id"]
            if cid not in wanted:
                continue
            src = rec.get("sources") or {}
            docs = (src.get("pgp") or {}).get("documents") or []
            meta[cid] = {
                "shelfmark": rec.get("shelfmark_display") or cid,
                "institution": rec.get("institution"),
                "pgpids": sorted({str(p) for p in rec.get("pgpids") or []} | {str(d.get("pgpid")) for d in docs if d.get("pgpid")}),
                "pgp_types": sorted({d.get("type") or "" for d in docs}),
                "pgp_tags": sorted({t.strip() for d in docs for t in (d.get("tags") or "").split(",") if t.strip()}),
                "years": pgp_years(docs),
                "fjp_shelfmarks": sorted({f.get("shelf_mark") for f in src.get("fjp") or [] if f.get("shelf_mark")}),
            }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def fjp_join_lists(ids: Sequence[str], meta: Dict[str, dict]) -> Dict[str, Set[str]]:
    """FJP ``joinedManuscripts`` shelf keys for each gallery fragment (raw, unresolved).

    :param ids: Gallery ids.
    :param meta: Metadata (gives each fragment's FJP shelfmarks).
    :returns: cid -> set of joined shelf keys.
    :rtype: Dict[str, Set[str]]
    """
    fjp = json.loads(FJP_ALL.read_text())
    lists: Dict[str, Set[str]] = {}
    for item in (fjp.values() if isinstance(fjp, dict) else fjp):
        joined = ((item.get("joins_data") or {}).get("joinedManuscripts")) or []
        if joined:
            lists[shelf_key(item.get("shelf_mark") or "")] = {shelf_key(j.get("shelfmark") or "") for j in joined}
    del fjp
    out = {}
    for cid in ids:
        m = meta.get(cid, {"fjp_shelfmarks": []})
        keys = set().union(*[lists.get(shelf_key(s), set()) for s in m["fjp_shelfmarks"]]) if m["fjp_shelfmarks"] else set()
        if keys:
            out[cid] = keys
    return out


def loose_keys(cid: str, m: dict) -> Set[str]:
    """Shelf keys under which a fragment may be named in someone else's FJP join list.

    :param cid: Fragment id.
    :param m: Its gallery metadata.
    :returns: Set of normalised keys (FJP shelfmarks, display shelfmark, canonical id).
    :rtype: Set[str]
    """
    keys = {shelf_key(s) for s in m.get("fjp_shelfmarks", [])} | {shelf_key(m.get("shelfmark") or ""), shelf_key(cid)}
    return {k for k in keys if len(k) >= 4}


def loose_fjp_pairs(ids: Sequence[str], meta: Dict[str, dict], lists: Dict[str, Set[str]]) -> Set[Tuple[int, int]]:
    """Gallery pairs that an FJP join list names under a shelfmark the manifest did not resolve.

    A joined key matches a fragment when it equals one of the fragment's keys or ends with
    ``"_" + key`` (the join list usually prefixes the library, the display shelfmark does not).
    Keys shorter than 6 characters are only matched exactly.

    :param ids: Gallery ids (index order).
    :param meta: Gallery metadata.
    :param lists: Output of :func:`fjp_join_lists`.
    :returns: Set of index pairs (i < j).
    :rtype: Set[Tuple[int, int]]
    """
    exact: Dict[str, Set[int]] = defaultdict(set)
    for i, cid in enumerate(ids):
        for k in loose_keys(cid, meta.get(cid, {})):
            exact[k].add(i)
    pos = {c: i for i, c in enumerate(ids)}
    out = set()
    for cid, joined in lists.items():
        a = pos[cid]
        for jk in joined:
            parts = jk.split("_")
            for start in range(len(parts)):  # longest underscore-aligned suffix first
                suffix = "_".join(parts[start:])
                if suffix in exact and (start == 0 or len(suffix) >= 6):
                    out |= {(min(a, b), max(a, b)) for b in exact[suffix] if b != a}
                    break
    return out


def is_documentary(m: dict) -> bool:
    """Whether a fragment carries a PGP documentary record (letter, legal, list, state, responsum, receipt).

    :param m: Gallery metadata for the fragment.
    :returns: True for documentary fragments.
    :rtype: bool
    """
    return bool(set(m.get("pgp_types", [])) & DOCUMENTARY_TYPES)


def holding_key(cid: str, collection: str) -> str:
    """Holding institution/collection of a fragment from its canonical id.

    The manifest's ``institution|collection`` strings differ by catalogue source (Manchester
    alone has four spellings), so "cross-collection" is decided on the id's leading
    alphabetic tokens instead: ``Cambridge_CUL`` (T-S, Or., Add.), ``Cambridge_Lewis``,
    ``Cambridge_Mosseri``, ``New_York_JTS``, ``Manchester_JRL``, ``Vienna_ONB``, ...

    :param cid: Canonical id.
    :param collection: Manifest collection string (fallback for numeric ids).
    :returns: Holding key.
    :rtype: str
    """
    toks = []
    for t in cid.split("_"):
        if any(ch.isdigit() for ch in t):
            break
        toks.append(t)
    toks = toks[: 3 if toks[:1] == ["New"] else 2]
    return "_".join(toks) if toks else collection


def same_record(a: str, b: str, ha: str, hb: str) -> bool:
    """Whether two ids are two catalogue records of one shelfmark in one holding.

    e.g. ``Vienna_ONB_H_14`` and ``Vienna_ONB_Austrian_National_Library_..._Rainer_Collection_H_14``.

    :param a: Canonical id.
    :param b: Canonical id.
    :param ha: Holding key of a.
    :param hb: Holding key of b.
    :returns: True when the shorter id tail (>= 2 tokens, containing a digit) ends the longer one.
    :rtype: bool
    """
    if ha != hb:
        return False
    ta, tb = sorted((a[len(ha):].strip("_"), b[len(hb):].strip("_")), key=len)
    if ta.count("_") < 1 or not any(ch.isdigit() for ch in ta):
        return False
    return tb == ta or tb.endswith("_" + ta)


# ----------------------------------------------------------------------------- similarity

def rank_matrix(S: np.ndarray, block: int = 512) -> np.ndarray:
    """Rank of every gallery item in every other item's neighbour list (1 = nearest; self = 0).

    :param S: Square similarity matrix (diagonal ignored).
    :param block: Rows per argsort block (bounds the int64 temporary).
    :returns: int16 matrix ``R[i, j]`` = rank of j among i's neighbours.
    :rtype: np.ndarray
    """
    n = len(S)
    R = np.zeros((n, n), dtype=np.int16)
    for s in range(0, n, block):
        rows = S[s:s + block].copy()
        rows[np.arange(len(rows)), np.arange(s, s + len(rows))] = np.inf  # self first -> rank 0
        order = np.argsort(-rows, axis=1)
        np.put_along_axis(R[s:s + block], order, np.arange(n, dtype=np.int16)[None, :].repeat(len(rows), 0), axis=1)
    return R


def csls(S: np.ndarray, k: int) -> np.ndarray:
    """Cross-domain similarity local scaling: ``2 s_ij - r_i - r_j`` (penalises hub images).

    :param S: Square cosine similarity matrix.
    :param k: Neighbourhood size for the local density ``r_i`` (mean of the top-k sims).
    :returns: CSLS matrix (float32, diagonal = -inf).
    :rtype: np.ndarray
    """
    T = S.copy()
    np.fill_diagonal(T, -np.inf)
    r = np.sort(np.partition(T, -k, axis=1)[:, -k:], axis=1).mean(axis=1).astype(np.float32)
    T *= 2
    T -= r[:, None]
    T -= r[None, :]
    return T


def join_retrieval(S: np.ndarray, partners: Dict[int, Set[int]], coll: Sequence[str]) -> dict:
    """R@1/10 of the first known partner (the ``eval_images`` measurement) for one similarity.

    :param S: Square similarity matrix.
    :param partners: index -> known partner indices.
    :param coll: Collection per index.
    :returns: {"all": {...}, "cross_collection": {...}}.
    :rtype: dict
    """
    ranks, cross = [], []
    for q, ps in partners.items():
        s = S[q].copy()
        s[q] = -np.inf
        r = int((s > max(s[p] for p in ps)).sum()) + 1
        ranks.append(r)
        if not any(coll[p] == coll[q] for p in ps):
            cross.append(r)

    def st(rs: List[int]) -> dict:
        """Recall/MRR summary of first-partner ranks.

        :param rs: Ranks.
        :returns: Stats dict.
        :rtype: dict
        """
        rs = np.array(rs)
        return {"n": int(len(rs)), "R@1": round(float((rs <= 1).mean()), 4), "R@10": round(float((rs <= 10).mean()), 4),
                "MRR": round(float((1 / rs).mean()), 4)}
    return {"all": st(ranks), "cross_collection": st(cross)}


# ----------------------------------------------------------------------------- known structure

class UnionFind:
    """Minimal union-find over gallery indices (join groups)."""

    def __init__(self, n: int) -> None:
        """Create singleton groups.

        :param n: Number of items.
        """
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """Root of x's group (path halving).

        :param x: Item index.
        :returns: Root index.
        :rtype: int
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Merge the groups of a and b.

        :param a: Item index.
        :param b: Item index.
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def index_pairs(pairs: Iterable[Sequence[str]], pos: Dict[str, int]) -> Set[Tuple[int, int]]:
    """Map id pairs to sorted gallery index pairs, dropping pairs not in the gallery.

    :param pairs: (cid, cid) pairs.
    :param pos: cid -> gallery index.
    :returns: Set of (i, j) with i < j.
    :rtype: Set[Tuple[int, int]]
    """
    out = set()
    for a, b in pairs:
        if a in pos and b in pos and a != b:
            i, j = pos[a], pos[b]
            out.add((min(i, j), max(i, j)))
    return out


def pgp_pairs(ids: Sequence[str], meta: Dict[str, dict]) -> Set[Tuple[int, int]]:
    """All gallery pairs that share any PGP document id (no group-size cap).

    :param ids: Gallery ids.
    :param meta: Gallery metadata.
    :returns: Set of index pairs.
    :rtype: Set[Tuple[int, int]]
    """
    by_doc: Dict[str, List[int]] = defaultdict(list)
    for i, cid in enumerate(ids):
        for p in meta.get(cid, {}).get("pgpids", []):
            by_doc[p].append(i)
    return {(a, b) for members in by_doc.values() for a in members for b in members if a < b}


def attr_conflicts(a: dict, b: dict, keys: Sequence[str]) -> List[str]:
    """KTIV attributes present on both fragments with different values.

    :param a: Attribute dict of fragment a.
    :param b: Attribute dict of fragment b.
    :param keys: Attribute keys to compare.
    :returns: Conflicting keys.
    :rtype: List[str]
    """
    return [k for k in keys if a.get(k) and b.get(k) and a[k] != b[k]]


# ----------------------------------------------------------------------------- calibration

def fit_logistic(F: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton/IRLS (intercept is the last weight).

    :param F: Feature matrix (n x d), already standardised.
    :param y: Binary labels.
    :param l2: Ridge strength (intercept unpenalised).
    :param iters: Newton steps.
    :returns: Weights of length d + 1.
    :rtype: np.ndarray
    """
    A = np.hstack([F, np.ones((len(F), 1))])
    w = np.zeros(A.shape[1])
    reg = np.full(A.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + reg * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def pair_features(sim: np.ndarray, r_ab: np.ndarray, r_ba: np.ndarray, cross: np.ndarray) -> np.ndarray:
    """Raw calibration features for candidate pairs.

    :param sim: Pair similarity.
    :param r_ab: Rank of b in a's list.
    :param r_ba: Rank of a in b's list.
    :param cross: 1 when the two fragments are held in different collections.
    :returns: (n x 5) matrix: sim, log best rank, log worst rank, cross, sim*cross.
    :rtype: np.ndarray
    """
    lo = np.log(np.minimum(r_ab, r_ba).astype(float))
    hi = np.log(np.maximum(r_ab, r_ba).astype(float))
    return np.stack([sim, lo, hi, cross, sim * cross], axis=1)


def threshold_table(S: np.ndarray, cross_mask_fn: Callable[[np.ndarray], np.ndarray], pos_pairs: Set[Tuple[int, int]],
                    ambiguous: Set[Tuple[int, int]], thresholds: np.ndarray) -> dict:
    """Labelled precision and recall of ``sim >= t`` over ALL gallery pairs, per stratum.

    :param S: Square similarity matrix.
    :param cross_mask_fn: f(rows) -> bool matrix (len(rows) x n), True where cross-collection.
    :param pos_pairs: Known join pairs (positives).
    :param ambiguous: Grouped-but-not-positive pairs (excluded from both sides).
    :param thresholds: Similarity thresholds.
    :returns: {stratum: [{t, pos, neg, precision, recall}, ...]}.
    :rtype: dict
    """
    n = len(S)
    cnt = {s: np.zeros(len(thresholds), dtype=np.int64) for s in ("same", "cross")}
    for s0 in range(0, n, 512):
        rows = np.arange(s0, min(n, s0 + 512))
        block = S[rows]
        upper = np.arange(n)[None, :] > rows[:, None]
        cm = cross_mask_fn(rows)
        for name, m in (("same", upper & ~cm), ("cross", upper & cm)):
            v = np.sort(block[m])
            cnt[name] += len(v) - np.searchsorted(v, thresholds, side="left")
    out = {}
    for name in ("same", "cross"):
        want_cross = name == "cross"
        sel = lambda pairs: np.array([S[a, b] for a, b in pairs if cross_mask_fn(np.array([a]))[0, b] == want_cross])
        ps, am = sel(pos_pairs), sel(ambiguous)
        npos = np.array([(ps >= t).sum() for t in thresholds]) if len(ps) else np.zeros(len(thresholds), int)
        namb = np.array([(am >= t).sum() for t in thresholds]) if len(am) else np.zeros(len(thresholds), int)
        nneg = cnt[name] - npos - namb
        out[name] = [{"t": round(float(t), 3), "pos": int(p), "neg": int(q),
                      "precision": round(float(p / max(1, p + q)), 4), "recall": round(float(p / max(1, len(ps))), 4)}
                     for t, p, q in zip(thresholds, npos, nneg)]
        out[name + "_n_pos"] = int(len(ps))
    return out


def rho_estimate(n_gallery: int, n_known: int, partners_per_fragment: float) -> float:
    """Undiscovered-to-known join ratio assumed for the gallery.

    Each gallery fragment is assumed to have ``partners_per_fragment`` undiscovered join
    partners somewhere in the Genizah, each landing in the gallery with probability
    ``n_gallery / GENIZAH_SIZE``; the expected number of undiscovered in-gallery pairs is
    divided by the known in-gallery join pairs.

    :param n_gallery: Gallery size.
    :param n_known: Known join pairs inside the gallery.
    :param partners_per_fragment: Assumed undiscovered partners per fragment.
    :returns: rho.
    :rtype: float
    """
    expected = n_gallery * partners_per_fragment * (n_gallery / GENIZAH_SIZE) / 2
    return expected / max(1, n_known)


def disc_precision(p_lab: np.ndarray, rho: float) -> np.ndarray:
    """Base-rate-corrected precision of a non-known pair: ``rho * p / (1 - p)``, capped at 1.

    :param p_lab: Labelled-gallery precision.
    :param rho: Undiscovered-to-known ratio.
    :returns: Discovery precision estimate.
    :rtype: np.ndarray
    """
    p = np.clip(p_lab, 1e-6, 1 - 1e-6)
    return np.minimum(1.0, rho * p / (1 - p))


def isotonic_blocks(score: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit of P(y | score), Laplace-smoothed per block.

    :param score: Scores.
    :param y: Binary labels.
    :returns: (upper score edge per block, smoothed precision per block), both ascending.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    order = np.argsort(score, kind="stable")
    sums: List[float] = []
    cnts: List[float] = []
    edges: List[float] = []
    for xi, yi in zip(score[order], y[order]):
        sums.append(float(yi))
        cnts.append(1.0)
        edges.append(float(xi))
        while len(sums) > 1 and sums[-2] / cnts[-2] >= sums[-1] / cnts[-1]:
            s_, c_, e_ = sums.pop(), cnts.pop(), edges.pop()
            sums[-1] += s_
            cnts[-1] += c_
            edges[-1] = e_
    prec = (np.array(sums) + 1) / (np.array(cnts) + 2)
    return np.array(edges), np.maximum.accumulate(prec)


class JoinCalibrator:
    """``P(known join | sim, ranks, cross)``: logistic score mapped through an isotonic fit."""

    def __init__(self, l2: float = 1.0) -> None:
        """Create an unfitted calibrator.

        :param l2: Ridge strength.
        """
        self.l2 = l2
        self.mu = None
        self.sd = None
        self.w = None
        self.edges = None
        self.prec = None

    def _logit(self, F: np.ndarray) -> np.ndarray:
        """Logistic linear score.

        :param F: Raw features.
        :returns: Linear predictor.
        :rtype: np.ndarray
        """
        return np.hstack([(F - self.mu) / self.sd, np.ones((len(F), 1))]) @ self.w

    def fit(self, F: np.ndarray, y: np.ndarray) -> "JoinCalibrator":
        """Standardise, fit the logistic model, then the isotonic map on its scores.

        :param F: Raw features from :func:`pair_features`.
        :param y: Labels.
        :returns: self.
        :rtype: JoinCalibrator
        """
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-9
        self.w = fit_logistic((F - self.mu) / self.sd, y, self.l2)
        self.edges, self.prec = isotonic_blocks(self._logit(F), y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Calibrated labelled precision.

        :param F: Raw features.
        :returns: Probabilities.
        :rtype: np.ndarray
        """
        k = np.searchsorted(self.edges, self._logit(F), side="left")
        return self.prec[np.minimum(k, len(self.prec) - 1)]


def unit_folds(unit: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    """Fold index per row such that rows sharing a ``unit`` id land in the same fold.

    :param unit: Group id per row (non-negative ints).
    :param n_folds: Number of folds.
    :param seed: Assignment seed.
    :returns: Fold index per row.
    :rtype: np.ndarray
    """
    units = np.unique(unit)
    lookup = np.zeros(int(units.max()) + 1, dtype=int)
    lookup[units] = np.random.default_rng(seed).permutation(len(units)) % n_folds
    return lookup[unit]


def crossfit_calibration(F: np.ndarray, y: np.ndarray, fit_rows: np.ndarray, unit: np.ndarray, n_folds: int,
                         seed: int = 0) -> np.ndarray:
    """Out-of-fold calibrated precision for every pool pair (grouped K-fold over known-join components).

    :param F: Raw pair features (:func:`pair_features`).
    :param y: Known-join label per pair.
    :param fit_rows: Rows a calibrator may be fitted on (not-grouped pairs and positives).
    :param unit: Component id per pair (the lower-index fragment's known-join group), so every pair of one join
        group is scored by a model fitted without any of them.
    :param n_folds: Number of folds.
    :param seed: Fold-assignment seed.
    :returns: Out-of-fold precision per pair.
    :rtype: np.ndarray
    """
    fold = unit_folds(unit, n_folds, seed)
    out = np.zeros(len(F))
    for k in range(n_folds):
        tr, te = fit_rows & (fold != k), fold == k
        out[te] = JoinCalibrator().fit(F[tr], y[tr].astype(float)).predict(F[te])
    return out


def neighbour_pool(R: np.ndarray, k: int) -> np.ndarray:
    """Unordered pairs where either fragment is in the other's top-k.

    :param R: Rank matrix.
    :param k: Neighbourhood size.
    :returns: (m x 2) int array of index pairs with i < j.
    :rtype: np.ndarray
    """
    a, b = np.nonzero((R >= 1) & (R <= k))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.unique(np.stack([lo, hi], axis=1), axis=0)


def select_candidates(pool: np.ndarray, score: np.ndarray, n_pick: int, max_per_fragment: int,
                      used: Optional[Counter] = None) -> List[int]:
    """Pick the ``n_pick`` best-scoring pairs, capping how often one fragment appears.

    :param pool: (m x 2) index pairs.
    :param score: Ranking score per pair (higher first).
    :param n_pick: Number of pairs wanted.
    :param max_per_fragment: Cap per fragment (stops one hub image flooding the list).
    :param used: Shared per-fragment usage counter (updated in place) when selecting in several passes.
    :returns: Row indices into pool, in selection order.
    :rtype: List[int]
    """
    used = Counter() if used is None else used
    picked: List[int] = []
    for r in np.argsort(-score, kind="stable"):
        if len(picked) >= n_pick:
            break
        a, b = pool[r]
        if used[a] >= max_per_fragment or used[b] >= max_per_fragment:
            continue
        used[a] += 1
        used[b] += 1
        picked.append(int(r))
    return picked


# ----------------------------------------------------------------------------- rendering

_THUMBS: Dict[Tuple[str, int], Image.Image] = {}


def masked_thumb(cid: str, size: int) -> Image.Image:
    """Masked fragment (backdrop blanked to grey, cropped) fitted in a ``size`` square.

    :param cid: Fragment id.
    :param size: Square side in pixels.
    :returns: Thumbnail on a grey canvas.
    :rtype: Image.Image
    """
    key = (cid, size)
    if key not in _THUMBS:
        img = Image.open(IMG_DIR / f"{cid.replace('/', '_')}.jpg").convert("RGB")
        masked, _ = fragment_mask(img)
        masked.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        canvas.paste(masked, ((size - masked.width) // 2, (size - masked.height) // 2))
        _THUMBS[key] = canvas
    return _THUMBS[key]


def render_sheet(rows: List[List[Tuple[str, str]]], headers: List[str], out: Path, size: int, cols: int) -> None:
    """Grid contact sheet: each cell is a group of thumbnails with one caption line each.

    :param rows: One entry per cell; each is a list of (cid, caption) thumbnails shown side by side.
    :param headers: One header line per cell.
    :param out: Output JPEG path.
    :param size: Thumbnail side.
    :param cols: Cells per sheet row.
    """
    per = max(len(r) for r in rows)
    cell_w, cell_h = per * (size + 4) + 16, size + 52
    nrow = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, nrow * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (cell, head) in enumerate(zip(rows, headers)):
        x0, y0 = (n % cols) * cell_w + 8, (n // cols) * cell_h + 4
        draw.text((x0, y0), head[:160], fill="black")
        for k, (cid, cap) in enumerate(cell):
            x = x0 + k * (size + 4)
            sheet.paste(masked_thumb(cid, size), (x, y0 + 16))
            draw.text((x, y0 + 18 + size), cap[: max(8, size // 6)], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


# ----------------------------------------------------------------------------- joins

def pick_with_quota(pool: np.ndarray, score: np.ndarray, cross: np.ndarray, eligible: np.ndarray, n_cross: int,
                    n_same: int, max_per_fragment: int) -> np.ndarray:
    """Best ``n_cross`` cross-collection pairs, then best ``n_same`` same-collection pairs (unused quota rolls over).

    :param pool: (m x 2) index pairs.
    :param score: Calibrated precision per pair.
    :param cross: Cross-collection flag per pair.
    :param eligible: Pairs that may be proposed.
    :param n_cross: Cross-collection quota.
    :param n_same: Same-collection quota.
    :param max_per_fragment: Cap per fragment across both passes.
    :returns: Row indices into pool (cross block first).
    :rtype: np.ndarray
    """
    used: Counter = Counter()
    rows = np.nonzero(eligible & (cross == 1))[0]
    out = list(rows[select_candidates(pool[rows], score[rows], n_cross, max_per_fragment, used)])
    rows = np.nonzero(eligible & (cross == 0))[0]
    out += list(rows[select_candidates(pool[rows], score[rows], n_cross + n_same - len(out), max_per_fragment, used)])
    return np.array(out, dtype=int)


def shelf_neighbours(a: str, b: str) -> bool:
    """Whether two ids differ only in their last token (e.g. consecutive items of one box or volume).

    :param a: Canonical id.
    :param b: Canonical id.
    :returns: True for shelfmark neighbours.
    :rtype: bool
    """
    ta, tb = a.split("_"), b.split("_")
    return len(ta) == len(tb) and len(ta) > 2 and ta[:-1] == tb[:-1]


def join_discovery(args: argparse.Namespace, manifest: dict, meta: Dict[str, dict]) -> dict:
    """Calibrate the join fusion, score the neighbour pool, write candidates and sheets.

    :param args: CLI arguments.
    :param manifest: Image manifest.
    :param meta: Gallery metadata.
    :returns: Join section of the summary.
    :rtype: dict
    """
    ids, X = load_feats(args.join_features)
    n = len(ids)
    pos = {c: i for i, c in enumerate(ids)}
    coll = np.array([manifest["collection"][c] for c in ids])
    holding = np.array([holding_key(c, manifest["collection"][c]) for c in ids])
    hold_code = np.unique(holding, return_inverse=True)[1]
    attrs = [manifest["attrs"].get(c, {}) for c in ids]
    cosine = (X @ X.T).astype(np.float32)
    del X
    assert np.isfinite(cosine).all(), "non-finite fused similarity"

    # known structure
    fjp_clean = index_pairs(manifest[JOIN_KEY], pos)
    fjp_all = index_pairs(manifest["joins_fjp"], pos)
    pgp_join = index_pairs(manifest["joins_pgp"], pos)
    pgp_any = pgp_pairs(ids, meta)
    loose = loose_fjp_pairs(ids, meta, fjp_join_lists(ids, meta)) - fjp_all - pgp_any
    positives = fjp_clean | pgp_join | loose
    uf = UnionFind(n)
    for a, b in fjp_all | pgp_any | loose:
        uf.union(a, b)
    group = np.array([uf.find(i) for i in range(n)])
    members: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(group):
        members[int(g)].append(i)
    ambiguous = {(a, b) for m in members.values() for a in m for b in m if a < b} - positives
    partners: Dict[int, Set[int]] = defaultdict(set)
    for a, b in positives:
        partners[a].add(b)
        partners[b].add(a)
    partners = dict(partners)

    # similarity choice: raw cosine vs CSLS on the known joins (cross = different holding institution/collection)
    retrieval = {"cosine": join_retrieval(cosine, partners, holding),
                 "cosine_manifest_collection_strings": join_retrieval(cosine, partners, coll)}
    S = cosine
    if args.csls_k:
        C = csls(cosine, args.csls_k)
        retrieval[f"csls_k{args.csls_k}"] = join_retrieval(C, partners, holding)
        use_csls = retrieval[f"csls_k{args.csls_k}"]["all"]["R@10"] > retrieval["cosine"]["all"]["R@10"]
        S = C if use_csls else cosine
        del C
    else:
        use_csls = False
    if S is cosine:
        S = cosine.copy()
    np.fill_diagonal(S, -np.inf)
    R = rank_matrix(S)
    print("retrieval", json.dumps(retrieval), "using", "csls" if use_csls else "cosine", flush=True)

    def cross_mask(rows: np.ndarray) -> np.ndarray:
        """Cross-holding mask for a block of rows.

        :param rows: Row indices.
        :returns: Bool (len(rows) x n).
        :rtype: np.ndarray
        """
        return hold_code[rows][:, None] != hold_code[None, :]

    # (a) labelled precision vs threshold over all pairs, for raw fused cosine and for the ranking similarity
    thr = {}
    for name, M in (("cosine", cosine), ("ranking_similarity", S)):
        pos_sims = np.array([M[a, b] for a, b in positives])
        thresholds = np.unique(np.round(np.quantile(pos_sims, np.linspace(0.05, 0.95, 19)), 3))
        thr[name] = threshold_table(M, cross_mask, positives, ambiguous, thresholds)

    # (b) precision vs rank
    rank_tab = {}
    for name, queries in (("all_gallery_queries", list(range(n))), ("queries_with_known_partner", sorted(partners))):
        row = {}
        for r in (1, 2, 3, 5, 10, 20, 50):
            hits = [int(np.nonzero(R[q] == r)[0][0]) in partners.get(q, ()) for q in queries]
            row[f"rank{r}"] = round(float(np.mean(hits)), 4)
        rank_tab[name] = row

    # (c) neighbour pool + calibration
    pool = neighbour_pool(R, args.pool_k)
    pa, pb = pool[:, 0], pool[:, 1]
    known = group[pa] == group[pb]
    is_pos = np.array([(a, b) in positives for a, b in pool])
    is_loose = np.array([(a, b) in loose for a, b in pool])
    sim = S[pa, pb]
    cos = cosine[pa, pb]
    r_ab, r_ba = R[pa, pb].astype(int), R[pb, pa].astype(int)
    cross = (hold_code[pa] != hold_code[pb]).astype(float)
    F = pair_features(sim, r_ab, r_ba, cross)
    fit_rows = ~known | is_pos
    cal = JoinCalibrator().fit(F[fit_rows], is_pos[fit_rows].astype(float))  # full fit: reported weights only
    p_in_sample = cal.predict(F)
    p_cal = crossfit_calibration(F, is_pos, fit_rows, group[pa], args.cv_folds)  # out-of-fold: used everywhere below
    mutual = np.maximum(r_ab, r_ba)
    mutual_tab = []
    for lo, hi in ((1, 1), (2, 3), (4, 10), (11, 30), (31, n)):
        m = fit_rows & (mutual >= lo) & (mutual <= hi)
        for cname, cm in (("same", cross == 0), ("cross", cross == 1)):
            mm = m & cm
            mutual_tab.append({"mutual_rank": f"{lo}-{hi}" if hi < n else f">={lo}", "stratum": cname, "pairs": int(mm.sum()),
                               "known_joins": int(is_pos[mm].sum()),
                               "precision": round(float(is_pos[mm].mean()), 4) if mm.any() else None})
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.01]
    reliability = {}
    for name, pred in (("out_of_fold", p_cal), ("in_sample", p_in_sample)):
        reliability[name] = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = fit_rows & (pred >= lo) & (pred < hi)
            reliability[name].append({"p_cal": f"{lo}-{min(hi, 1)}", "pairs": int(m.sum()),
                                      "mean_pred": round(float(pred[m].mean()), 4) if m.any() else None,
                                      "observed": round(float(is_pos[m].mean()), 4) if m.any() else None})

    # compatibility filter: attributes whose conflict rate on known joins is low
    conf_rate = {}
    neg_rows = np.nonzero(fit_rows & ~is_pos)[0]
    for key in ATTR_KEYS:
        both_pos = [(a, b) for a, b in positives if attrs[a].get(key) and attrs[b].get(key)]
        both_neg = [(pa[r], pb[r]) for r in neg_rows if attrs[pa[r]].get(key) and attrs[pb[r]].get(key)]
        conf_rate[key] = {"known_join_pairs_with_both": len(both_pos),
                          "known_join_conflict_rate": round(float(np.mean([attrs[a][key] != attrs[b][key] for a, b in both_pos])), 4) if both_pos else None,
                          "pool_nonjoin_pairs_with_both": len(both_neg),
                          "pool_nonjoin_conflict_rate": round(float(np.mean([attrs[a][key] != attrs[b][key] for a, b in both_neg])), 4) if both_neg else None}
    filter_keys = [k for k, v in conf_rate.items() if v["known_join_conflict_rate"] is not None
                   and v["known_join_pairs_with_both"] >= 20 and v["known_join_conflict_rate"] <= args.max_attr_conflict]
    compatible = np.array([not attr_conflicts(attrs[a], attrs[b], filter_keys) for a, b in pool])

    # duplicate records of one fragment (same holding, same shelfmark tail) and near-identical photographs are not joins
    record_dup = np.array([same_record(ids[a], ids[b], holding[a], holding[b]) for a, b in pool])
    duplicate = record_dup | (cos >= args.dup_cos)
    pos_cos = np.array([cosine[a, b] for a, b in positives])

    # base-rate correction + selection (expected-value order: cross-collection pairs weighted up)
    rhos = {f"partners_per_fragment={u}": rho_estimate(n, len(positives), u) for u in args.partners_per_fragment}
    rho_mid = rhos[f"partners_per_fragment={args.partners_per_fragment[1]}"]
    eligible = ~known & compatible & ~duplicate
    rows_idx = pick_with_quota(pool, p_cal, cross, eligible, args.n_cross, args.n_same, args.max_per_fragment)

    # validation of the odds transfer: hide one label source, re-fit on the rest, re-run the selection
    validation = {}
    for hidden_name, visible, visible_links, hidden in (("hide_pgp", fjp_clean | loose, fjp_all | loose, pgp_join),
                                                         ("hide_fjp", pgp_join, pgp_any, fjp_clean | loose)):
        vf = UnionFind(n)
        for a, b in visible_links:
            vf.union(a, b)
        vg = np.array([vf.find(i) for i in range(n)])
        v_known = vg[pa] == vg[pb]
        hidden_new = {(a, b) for a, b in hidden if vg[a] != vg[b]}  # hidden joins the visible labels do not already group
        v_pos = np.array([(a, b) in visible for a, b in pool])
        h_pos = np.array([(a, b) in hidden_new for a, b in pool])
        v_fit = ~v_known | v_pos
        v_p = crossfit_calibration(F, v_pos, v_fit, vg[pa], args.cv_folds)  # same out-of-fold procedure as the main fit
        rho_h = len(hidden_new) / max(1, len(visible))
        v_pick = pick_with_quota(pool, v_p, cross, ~v_known & compatible & ~duplicate, args.n_cross, args.n_same, args.max_per_fragment)
        validation[hidden_name] = {"visible_pairs": len(visible), "hidden_pairs": len(hidden_new), "rho_hidden": round(rho_h, 4),
                                   "hidden_pairs_in_pool": int(h_pos.sum()), "n_selected": int(len(v_pick)),
                                   "hidden_joins_found": int(h_pos[v_pick].sum()),
                                   "predicted_hidden_joins": round(float(disc_precision(v_p[v_pick], rho_h).sum()), 2),
                                   "selected_cross_collection": int((cross[v_pick] == 1).sum()),
                                   "found_cross_collection": int((h_pos[v_pick] & (cross[v_pick] == 1)).sum()),
                                   "predicted_cross_collection": round(float(disc_precision(v_p[v_pick][cross[v_pick] == 1], rho_h).sum()), 2)}

    # outputs
    def band(p: float) -> str:
        """Labelled-precision band label.

        :param p: Calibrated labelled precision.
        :returns: Band.
        :rtype: str
        """
        return "A (>=0.4)" if p >= 0.4 else "B (0.2-0.4)" if p >= 0.2 else "C (0.1-0.2)" if p >= 0.1 else "D (<0.1)"

    def row_for(r: int) -> dict:
        """CSV row for pool pair r.

        :param r: Pool row.
        :returns: Row dict.
        :rtype: dict
        """
        a, b = pool[r]
        ca, cb = ids[a], ids[b]
        d = {"pair": "", "id_a": ca, "id_b": cb, "shelfmark_a": meta.get(ca, {}).get("shelfmark", ca),
             "shelfmark_b": meta.get(cb, {}).get("shelfmark", cb), "holding_a": holding[a], "holding_b": holding[b],
             "collection_a": coll[a], "collection_b": coll[b], "cross_collection": int(cross[r]),
             "similarity": round(float(sim[r]), 4), "cosine": round(float(cos[r]), 4),
             "rank_a_to_b": int(r_ab[r]), "rank_b_to_a": int(r_ba[r]),
             "p_labelled": round(float(p_cal[r]), 4), "band_labelled": band(float(p_cal[r])),
             "p_discovery_est": round(float(disc_precision(p_cal[r:r + 1], rho_mid)[0]), 5),
             "near_identical_image": int(cos[r] >= args.near_cos), "same_box_or_volume": int(shelf_neighbours(ca, cb))}
        for key in ATTR_KEYS:
            d[f"{key}_a"], d[f"{key}_b"] = attrs[a].get(key, ""), attrs[b].get(key, "")
        d["pgp_types_a"] = ";".join(meta.get(ca, {}).get("pgp_types", []))
        d["pgp_types_b"] = ";".join(meta.get(cb, {}).get("pgp_types", []))
        return d

    cands = []
    for k, r in enumerate(rows_idx, 1):
        d = row_for(int(r))
        d["pair"] = f"J{k:03d}"
        cands.append(d)
    fields = list(row_for(0).keys())
    write_csv(OUT / "join_candidates.csv", cands, fields)
    write_csv(OUT / "possible_duplicate_records.csv",
              [dict(row_for(int(r)), pair="same_record" if record_dup[r] else "near_identical_photo")
               for r in np.nonzero(duplicate & ~known)[0]], fields)
    write_csv(OUT / "fjp_unresolved_joins_in_pool.csv", [dict(row_for(int(r)), pair="fjp_loose") for r in np.nonzero(is_loose)[0]], fields)
    if not args.no_render:
        for s0 in range(0, len(cands), 20):
            chunk = cands[s0:s0 + 20]
            render_sheet([[(c["id_a"], c["shelfmark_a"]), (c["id_b"], c["shelfmark_b"])] for c in chunk],
                         [f"{c['pair']} cos {c['cosine']:.3f} r{c['rank_a_to_b']}/{c['rank_b_to_a']} "
                          f"{'CROSS' if c['cross_collection'] else 'same'} band {c['band_labelled'][0]}" for c in chunk],
                         OUT / "join_sheets" / f"join_sheet_{s0 // 20 + 1:02d}.jpg", args.thumb, 4)

    sel_p = p_cal[rows_idx]
    expected = {name: round(float(disc_precision(sel_p, rho).sum()), 2) for name, rho in rhos.items()}
    vt = [validation[k] for k in validation]
    transfer = sum(v["hidden_joins_found"] for v in vt) / max(1e-9, sum(v["predicted_hidden_joins"] for v in vt))
    return {
        "features": args.join_features, "ranking_similarity": f"csls_k{args.csls_k}" if use_csls else "cosine", "n_gallery": n,
        "cross_collection_definition": "different holding (institution token(s) of the canonical id; Cambridge CUL, Lewis-Gibson and Mosseri kept apart)",
        "retrieval_known_joins": retrieval,
        "known": {"fjp_clean_pairs": len(fjp_clean), "fjp_all_pairs": len(fjp_all), "pgp_join_pairs": len(pgp_join),
                  "pgp_shared_doc_pairs_any_size": len(pgp_any), "fjp_unresolved_loose_pairs": len(loose),
                  "calibration_positives": len(positives), "calibration_positives_cross_collection": int(sum(hold_code[a] != hold_code[b] for a, b in positives)),
                  "ambiguous_grouped_pairs": len(ambiguous), "fragments_with_known_partner": len(partners)},
        "calibration": {"threshold_all_pairs": thr, "precision_vs_rank": rank_tab, "precision_vs_mutual_rank": mutual_tab,
                        "pool_k": args.pool_k, "pool_pairs": int(len(pool)), "pool_known_joins": int(is_pos.sum()),
                        "logistic_weights_[sim,logminrank,logmaxrank,cross,sim*cross,intercept]": [round(float(w), 4) for w in cal.w],
                        "cv_folds": args.cv_folds,
                        "auc_on_pool_out_of_fold": round(pair_auc(p_cal[fit_rows & is_pos], p_cal[fit_rows & ~is_pos]), 4),
                        "auc_on_pool_in_sample": round(pair_auc(p_in_sample[fit_rows & is_pos], p_in_sample[fit_rows & ~is_pos]), 4),
                        "reliability_isotonic": reliability, "loose_fjp_pairs_in_pool": int(is_loose.sum())},
        "attribute_filter": {"conflict_rates": conf_rate, "applied_keys": filter_keys, "max_known_join_conflict": args.max_attr_conflict,
                             "pool_pairs_removed": int((~compatible & ~known).sum())},
        "duplicates": {"same_record_pairs": int((record_dup & ~known).sum()), "near_identical_cosine": args.dup_cos,
                       "near_identical_pairs": int(((cos >= args.dup_cos) & ~record_dup & ~known).sum()),
                       "known_join_pairs_at_or_above_cosine": int((pos_cos >= args.dup_cos).sum()),
                       "known_join_cosine_quantiles_50_90_99_max": [round(float(q), 4) for q in np.quantile(pos_cos, [0.5, 0.9, 0.99, 1.0])]},
        "base_rate": {"genizah_size_assumed": GENIZAH_SIZE, "rho_by_assumption": {k: round(v, 5) for k, v in rhos.items()},
                      "rho_used_for_p_discovery_est": round(rho_mid, 5)},
        "validation_hide_one_source": validation,
        "validation_found_over_predicted": round(transfer, 2),
        "candidates": {"n": len(cands), "cross_collection": int(sum(c["cross_collection"] for c in cands)),
                       "quota_cross_same": [args.n_cross, args.n_same],
                       "near_identical_image_flagged": int(sum(c["near_identical_image"] for c in cands)),
                       "same_box_or_volume_flagged": int(sum(c["same_box_or_volume"] for c in cands)),
                       "bands_labelled": dict(sorted(Counter(c["band_labelled"] for c in cands).items())),
                       "bands_labelled_cross": dict(sorted(Counter(c["band_labelled"] for c in cands if c["cross_collection"]).items())),
                       "mean_p_labelled": round(float(sel_p.mean()), 4) if len(cands) else None,
                       "expected_real_joins_by_assumption": expected,
                       "expected_real_central_cross_vs_same": [round(float(disc_precision(sel_p[cross[rows_idx] == 1], rho_mid).sum()), 2),
                                                               round(float(disc_precision(sel_p[cross[rows_idx] == 0], rho_mid).sum()), 2)],
                       "expected_real_joins_central_x_validation_ratio": round(expected[f"partners_per_fragment={args.partners_per_fragment[1]}"] * transfer, 2)},
    }


# ----------------------------------------------------------------------------- scribes

def centroid_scores(Xq: np.ndarray, Xl: np.ndarray, lab: np.ndarray, names: List[str], exclude: Optional[np.ndarray] = None
                    ) -> np.ndarray:
    """Cosine of each query to each scribe centroid (centroids of L2-normalised features, renormalised).

    :param Xq: Query features (q x d).
    :param Xl: Labelled features (l x d).
    :param lab: Scribe index per labelled row.
    :param names: Scribe names (len = number of centroids).
    :param exclude: Optional bool (q x l) mask of labelled rows to leave out per query (LOO / same document).
    :returns: (q x n_scribes) score matrix; -inf where a scribe has no remaining exemplar.
    :rtype: np.ndarray
    """
    k = len(names)
    onehot = np.zeros((len(Xl), k), dtype=np.float32)
    onehot[np.arange(len(Xl)), lab] = 1
    sums = onehot.T @ Xl  # k x d
    if exclude is None:
        C = sums / np.linalg.norm(sums, axis=1, keepdims=True)
        return Xq @ C.T
    out = np.full((len(Xq), k), -np.inf, dtype=np.float32)
    for i in range(len(Xq)):
        drop = exclude[i]
        s = sums - onehot[drop].T @ Xl[drop]
        cnt = onehot.sum(0) - onehot[drop].sum(0)
        ok = cnt > 0
        C = s[ok] / np.linalg.norm(s[ok], axis=1, keepdims=True)
        out[i, ok] = C @ Xq[i]
    return out


def precision_threshold(score: np.ndarray, correct: np.ndarray, target: float, min_n: int) -> Optional[float]:
    """Lowest score threshold whose kept set (``score >= t``, at least ``min_n`` items) has precision >= target.

    :param score: Confidence per LOO query.
    :param correct: Whether the LOO prediction was right.
    :param target: Precision target.
    :param min_n: Smallest kept set considered.
    :returns: Threshold or None if never reached.
    :rtype: Optional[float]
    """
    order = np.argsort(-score)
    prec = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.nonzero(prec[min_n - 1:] >= target)[0]
    return float(score[order[min_n - 1 + ok[-1]]]) if len(ok) else None


def cv_rule_precision(score: np.ndarray, correct: np.ndarray, unit: np.ndarray, target: float, min_n: int,
                      n_folds: int, seed: int = 0) -> dict:
    """Honest precision/coverage of the threshold rule by nested cross-validation.

    For each fold the threshold is chosen by :func:`precision_threshold` on the other folds' LOO scores (``min_n``
    scaled to their size) and applied to the held-out fold, so no item's own outcome picks the threshold it is
    judged by. Folds keep PGP-document groups together.

    :param score: LOO confidence per labelled fragment.
    :param correct: Whether its LOO prediction was right.
    :param unit: Document-group id per labelled fragment.
    :param target: Precision target of the rule.
    :param min_n: Smallest kept set on the full data.
    :param n_folds: Number of folds.
    :param seed: Fold-assignment seed.
    :returns: {precision, coverage, kept, folds_with_threshold}.
    :rtype: dict
    """
    fold = unit_folds(unit, n_folds, seed)
    kept = np.zeros(len(score), dtype=bool)
    n_ok = 0
    for k in range(n_folds):
        tr = fold != k
        t = precision_threshold(score[tr], correct[tr], target, max(5, int(round(min_n * tr.mean()))))
        if t is None:
            continue
        n_ok += 1
        kept |= (fold == k) & (score >= t)
    return {"precision": round(float(correct[kept].mean()), 4) if kept.any() else None,
            "coverage": round(float(kept.mean()), 4), "kept": int(kept.sum()), "folds_with_threshold": n_ok}


def scribe_windows(manifest: dict, meta: Dict[str, dict], slack: int, min_dated: int) -> Dict[str, Tuple[int, int]]:
    """Active-date window per scribe from the PGP dates of all their labelled fragments.

    :param manifest: Image manifest (``scribe`` labels).
    :param meta: Metadata including ``years`` for the scribe-labelled fragments.
    :param slack: Years added on each side of the observed range.
    :param min_dated: Minimum dated fragments for a window.
    :returns: scribe -> (first, last) year.
    :rtype: Dict[str, Tuple[int, int]]
    """
    years: Dict[str, List[int]] = defaultdict(list)
    dated: Counter = Counter()
    for cid, scribe in manifest["scribe"].items():
        ys = meta.get(cid, {}).get("years", [])
        if ys:
            years[scribe] += ys
            dated[scribe] += 1
    return {s: (min(ys) - slack, max(ys) + slack) for s, ys in years.items() if dated[s] >= min_dated}


def date_intervals(m: dict) -> List[Tuple[int, int]]:
    """Date evidence for a fragment: exact PGP years plus PGP century tags ("12th c", "18th or 19th c").

    :param m: Fragment metadata (``years``, ``pgp_tags``).
    :returns: List of (first, last) year intervals; empty when undated.
    :rtype: List[Tuple[int, int]]
    """
    out = [(y, y) for y in m.get("years", [])]
    for tag in m.get("pgp_tags", []):
        hit = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)(?: or (\d{1,2})(?:st|nd|rd|th))? c\.?", tag.strip())
        if hit:
            first, last = int(hit.group(1)), int(hit.group(2) or hit.group(1))
            out.append(((first - 1) * 100, last * 100 - 1))
    return out


def date_status(dates: List[Tuple[int, int]], window: Optional[Tuple[int, int]]) -> str:
    """Compare a fragment's date intervals with a scribe window.

    :param dates: Output of :func:`date_intervals`.
    :param window: Scribe window or None.
    :returns: ``compatible``, ``incompatible`` or ``unknown``.
    :rtype: str
    """
    if not dates or window is None:
        return "unknown"
    return "compatible" if any(lo <= window[1] and hi >= window[0] for lo, hi in dates) else "incompatible"


def chance_compatible(dates: List[Tuple[int, int]], windows: Dict[str, Tuple[int, int]], exclude: Set[str]) -> Optional[float]:
    """Share of scribe windows (other than ``exclude``) compatible with a dated fragment.

    :param dates: Fragment date intervals.
    :param windows: Scribe windows.
    :param exclude: Scribes left out (the proposed one, and the true one when known).
    :returns: Fraction, or None when undated.
    :rtype: Optional[float]
    """
    if not dates:
        return None
    vals = [date_status(dates, w) == "compatible" for s, w in windows.items() if s not in exclude]
    return float(np.mean(vals)) if vals else None


def date_expected_correct(p_inc: float, q: float, kappa: float, n_compat: int, n_kept: int) -> Tuple[float, float]:
    """Error rate and expected correct attributions implied by the date check.

    A wrong attribution falls outside the proposed scribe's window with probability
    ``kappa * (1 - q)`` (``q``: share of other scribes' windows compatible with the fragment's
    date; ``kappa``: how much less often than that the leave-one-out errors are date-incompatible,
    because errors favour period-similar hands). So ``error = p_inc / (kappa * (1 - q))``.
    After dropping incompatible ones, dated survivors keep the wrong-but-compatible errors and
    undated ones keep the full error rate.

    :param p_inc: Share of dated candidates outside the proposed scribe's window.
    :param q: Chance compatibility of a wrong scribe.
    :param kappa: LOO correction factor (<= 1).
    :param n_compat: Dated, compatible candidates kept.
    :param n_kept: All candidates kept (compatible + undated).
    :returns: (error rate, expected correct attributions among the kept candidates).
    :rtype: Tuple[float, float]
    """
    err = min(1.0, p_inc / max(1e-9, kappa * (1 - q)))
    prec_compat = (1 - err) / max(1e-9, 1 - p_inc)
    return err, min(n_compat, prec_compat * n_compat) + (1 - err) * (n_kept - n_compat)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    :param k: Successes.
    :param n: Trials.
    :param z: Normal quantile (1.96 = 95%).
    :returns: (low, high).
    :rtype: Tuple[float, float]
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def scribe_discovery(args: argparse.Namespace, manifest: dict, sel: dict, meta: Dict[str, dict], features: str,
                     write: bool) -> dict:
    """LOO nearest-centroid calibration and attribution candidates for one feature set.

    :param args: CLI arguments.
    :param manifest: Image manifest.
    :param sel: Probe selection (``scribes`` = the 501-fragment labelled set).
    :param meta: Gallery metadata (with ``years`` for every scribe-labelled fragment).
    :param features: Feature-set name.
    :param write: Write CSV + sheets for this feature set.
    :returns: Scribe section of the summary.
    :rtype: dict
    """
    ids, X = load_feats(features)
    pos = {c: i for i, c in enumerate(ids)}
    counts = Counter(manifest["scribe"].values())
    sel_set = set(sel["scribes"])
    L = [pos[c] for c in sorted(sel_set) if c in pos and counts[manifest["scribe"][c]] >= 5]
    names = sorted({manifest["scribe"][ids[i]] for i in L})
    sidx = {s: k for k, s in enumerate(names)}
    lab = np.array([sidx[manifest["scribe"][ids[i]]] for i in L])
    Xl = X[L]
    docs = [set(meta.get(ids[i], {}).get("pgpids", [])) for i in L]
    same_doc = np.array([[bool(docs[a] & docs[b]) or a == b for b in range(len(L))] for a in range(len(L))])
    doc_uf = UnionFind(len(L))
    for a, b in zip(*np.nonzero(same_doc)):
        doc_uf.union(int(a), int(b))
    doc_unit = np.array([doc_uf.find(i) for i in range(len(L))])  # CV folds keep one document's fragments together
    windows = scribe_windows(manifest, meta, args.date_slack, 2)
    # scribes whose dates overlap no other scribe's: any document of their era is pulled to them, so date agreement says little
    isolated = {s for s, w in windows.items() if s in sidx and
                not any(o != s and o in sidx and w[0] <= v[1] and w[1] >= v[0] for o, v in windows.items())}
    holding = [holding_key(c, manifest["collection"][c]) for c in ids]

    def summarise(scores: np.ndarray) -> dict:
        """Prediction, top-1 score and top1-top2 margin per query.

        :param scores: (q x scribes) centroid scores.
        :returns: Dict of arrays.
        :rtype: dict
        """
        srt = np.sort(scores, axis=1)
        return {"pred": scores.argmax(1), "top1": srt[:, -1], "margin": srt[:, -1] - srt[:, -2]}

    loo = summarise(centroid_scores(Xl, Xl, lab, names, same_doc))
    correct = loo["pred"] == lab
    per_scribe = {names[k]: {"n": int((lab == k).sum()), "acc": round(float(correct[lab == k].mean()), 4),
                             "window": windows.get(names[k]),
                             "top_holding_share": round(Counter(holding[L[r]] for r in np.nonzero(lab == k)[0]).most_common(1)[0][1] / max(1, (lab == k).sum()), 3)}
                  for k in range(len(names))}
    sub = Xl @ Xl.T
    sub[same_doc] = -np.inf
    nn1 = float((lab[sub.argmax(1)] == lab).mean())  # 1-NN reference (the eval_images P@1 measurement)

    # leave-one-scribe-out: queries whose true scribe has no centroid (open-set negatives)
    loso_top1, loso_margin = np.zeros(len(L)), np.zeros(len(L))
    for k in range(len(names)):
        q = lab == k
        excl = np.repeat((lab == k)[None, :], q.sum(), axis=0) | same_doc[q]
        srt = np.sort(centroid_scores(Xl[q], Xl, lab, names, excl), axis=1)
        loso_top1[q], loso_margin[q] = srt[:, -1], srt[:, -1] - srt[:, -2]

    # do wrong LOO predictions land outside the predicted scribe's dates? (checks the date-based error estimate)
    loo_dates = [date_intervals(meta.get(ids[i], {})) for i in L]
    wrong_rows = [r for r in np.nonzero(~correct)[0] if loo_dates[r] and names[loo["pred"][r]] in windows]
    wrong_dated = [date_status(loo_dates[r], windows[names[loo["pred"][r]]]) for r in wrong_rows]
    wrong_q = [chance_compatible(loo_dates[r], windows, {names[loo["pred"][r]], names[lab[r]]}) for r in wrong_rows]
    loo_inc = float(np.mean([w == "incompatible" for w in wrong_dated])) if wrong_dated else None
    loo_inc_random = float(1 - np.mean([q for q in wrong_q if q is not None])) if wrong_rows else None
    kappa = min(1.0, loo_inc / loo_inc_random) if loo_inc and loo_inc_random else 1.0  # <1: errors favour period-compatible scribes

    # held-out labelled fragments (in the gallery via joins/distractors, not in the 501) and unlabelled documentary ones
    H = [i for i, c in enumerate(ids) if c in manifest["scribe"] and c not in sel_set and manifest["scribe"][c] in sidx]
    hlab = np.array([sidx[manifest["scribe"][ids[i]]] for i in H])
    # unlabelled documentary fragments, minus those tied to a scribe-labelled fragment by a known join or a shared
    # PGP document: their hand is known through that link (and a different proposed scribe would contradict it)
    labelled = set(manifest["scribe"])
    lab_docs = {d for c in labelled for d in meta.get(c, {}).get("pgpids", [])}
    linked = {b for key in ("joins_fjp", "joins_pgp") for a, b in manifest[key] if a in labelled} | \
             {a for key in ("joins_fjp", "joins_pgp") for a, b in manifest[key] if b in labelled}
    U_all = [i for i, c in enumerate(ids) if c not in labelled and is_documentary(meta.get(c, {}))]
    U = [i for i in U_all if ids[i] not in linked and not (set(meta.get(ids[i], {}).get("pgpids", [])) & lab_docs)]
    linked_out = sorted(ids[i] for i in set(U_all) - set(U))
    full_H = summarise(centroid_scores(X[H], Xl, lab, names)) if H else None
    full_U = summarise(centroid_scores(X[U], Xl, lab, names))
    u_dates = [date_intervals(meta.get(ids[i], {})) for i in U]

    rules = {}
    for score_name, loo_s, loso_s in (("top1", loo["top1"], loso_top1), ("margin", loo["margin"], loso_margin)):
        order = np.argsort(-loo_s)
        best_at_min = float(np.mean(correct[order[: args.scribe_min_n]]))
        t = precision_threshold(loo_s, correct, args.scribe_precision, args.scribe_min_n)
        if t is None:
            rules[score_name] = {"threshold": None, f"loo_precision_of_top_{args.scribe_min_n}": round(best_at_min, 4)}
            continue
        keep = loo_s >= t
        cov, prec = float(keep.mean()), float(correct[keep].mean())  # in-sample: t was picked to reach the target here
        cv = cv_rule_precision(loo_s, correct, doc_unit, args.scribe_precision, args.scribe_min_n, args.cv_folds)
        cov_h = cv["coverage"] if cv["precision"] is not None else cov
        prec_h = cv["precision"] if cv["precision"] is not None else prec
        fpr_open = float((loso_s >= t).mean())
        above = full_U[score_name] >= t
        r_obs = float(above.mean())
        # (a) method of moments with the cross-validated rule: r_obs = pi * cov + (1 - pi) * fpr_open
        pi = float(np.clip((r_obs - fpr_open) / max(1e-9, cov_h - fpr_open), 0, 1))
        p_mom = (pi * cov_h * prec_h / r_obs) if r_obs > 0 else 0.0
        # (b) date check: wrong attributions fall outside the proposed scribe's window with prob (1 - q)
        st = [date_status(u_dates[u], windows.get(names[full_U["pred"][u]])) for u in np.nonzero(above)[0]]
        dated = [x for x in st if x != "unknown"]
        p_inc = float(np.mean([x == "incompatible" for x in dated])) if dated else None
        q_chance = [chance_compatible(u_dates[u], windows, {names[full_U["pred"][u]]}) for u in np.nonzero(above)[0]
                    if date_status(u_dates[u], windows.get(names[full_U["pred"][u]])) != "unknown"]
        q = float(np.mean([x for x in q_chance if x is not None])) if any(x is not None for x in q_chance) else None
        n_kept = sum(1 for x in st if x != "incompatible")
        n_compat = sum(1 for x in st if x == "compatible")
        err, exp_date, exp_ci = None, None, None
        if p_inc is not None and q is not None:
            err, exp_date = date_expected_correct(p_inc, q, kappa, n_compat, n_kept)
            lo_p, hi_p = wilson(sum(x == "incompatible" for x in dated), len(dated))
            exp_ci = (round(date_expected_correct(hi_p, q, kappa, n_compat, n_kept)[1], 1),
                      round(date_expected_correct(lo_p, q, kappa, n_compat, n_kept)[1], 1))
        # the same date estimate restricted to candidates of scribes that share their era with another scribe
        sub = [u for u in np.nonzero(above)[0] if names[full_U["pred"][u]] not in isolated]
        st_s = [date_status(u_dates[u], windows.get(names[full_U["pred"][u]])) for u in sub]
        dated_s = [x for x in st_s if x != "unknown"]
        q_s = [chance_compatible(u_dates[u], windows, {names[full_U["pred"][u]]}) for u, x in zip(sub, st_s) if x != "unknown"]
        restricted = {"candidates": len(sub), "dated": len(dated_s), "date_incompatible": int(sum(x == "incompatible" for x in dated_s))}
        if dated_s:
            p_s, qq = float(np.mean([x == "incompatible" for x in dated_s])), float(np.mean(q_s))
            nk, nc = sum(1 for x in st_s if x != "incompatible"), sum(1 for x in st_s if x == "compatible")
            e_s, x_s = date_expected_correct(p_s, qq, kappa, nc, nk)
            lo_p, hi_p = wilson(restricted["date_incompatible"], len(dated_s))
            restricted.update({"kept": nk, "error_rate": round(e_s, 4), "expected_correct": round(x_s, 1),
                               "expected_correct_95ci": (round(date_expected_correct(hi_p, qq, kappa, nc, nk)[1], 1),
                                                         round(date_expected_correct(lo_p, qq, kappa, nc, nk)[1], 1))})
        rule = {"threshold": round(t, 4), "loo_coverage": round(cov, 4), "loo_precision": round(prec, 4), "loo_kept": int(keep.sum()),
                "loo_precision_note": "in-sample (threshold chosen on these LOO predictions); see nested_cv",
                "nested_cv": cv,
                "open_set_fpr_leave_one_scribe_out": round(fpr_open, 4),
                "unlabelled_documentary": len(U), "unlabelled_above_threshold": int(above.sum()), "unlabelled_rate_above": round(r_obs, 4),
                "estimate_moments": {"prior_by_these_scribes": round(pi, 4), "precision": round(p_mom, 4),
                                     "expected_correct": round(p_mom * int(above.sum()), 1)},
                "estimate_dates": {"above_dated": len(dated), "date_incompatible": int(sum(x == "incompatible" for x in dated)),
                                   "chance_wrong_scribe_compatible_q": round(q, 4) if q is not None else None, "kappa_from_loo": round(kappa, 3),
                                   "error_rate": round(err, 4) if err is not None else None,
                                   "kept_after_date_filter": n_kept,
                                   "expected_correct_after_date_filter": round(exp_date, 1) if exp_date is not None else None,
                                   "expected_correct_95ci_from_dated_sample": exp_ci,
                                   "excluding_era_isolated_scribes": restricted},
                "above_by_scribe": dict(Counter(names[k] for k in full_U["pred"][above]).most_common())}
        if H:
            hk = full_H[score_name] >= t
            rule["held_out_labelled"] = {"n": len(H), "above": int(hk.sum()), "coverage": round(float(hk.mean()), 4),
                                         "precision": round(float((full_H["pred"][hk] == hlab[hk]).mean()), 4) if hk.any() else None}
        rules[score_name] = rule
    usable = {k: v for k, v in rules.items() if v.get("threshold") is not None}
    chosen = max(usable, key=lambda k: usable[k]["loo_coverage"]) if usable else None

    res = {"features": features, "n_labelled": len(L), "n_scribes": len(names),
           "loo_nearest_centroid_acc": round(float(correct.mean()), 4),
           "loo_balanced_acc": round(float(np.mean([v["acc"] for v in per_scribe.values()])), 4),
           "loo_1nn_acc_reference": round(nn1, 4),
           "chance_majority": round(float(Counter(lab).most_common(1)[0][1] / len(lab)), 4),
           "unlabelled_documentary_all": len(U_all), "unlabelled_excluded_linked_to_labelled": len(linked_out),
           "unlabelled_linked_ids": linked_out[:50],
           "held_out_labelled_n": len(H), "held_out_acc": round(float((full_H["pred"] == hlab).mean()), 4) if H else None,
           "held_out_by_scribe": dict(Counter(names[k] for k in hlab).most_common()) if H else {},
           "loo_wrong_predictions_dated": len(wrong_dated),
           "loo_wrong_predictions_date_incompatible_rate": round(loo_inc, 4) if loo_inc is not None else None,
           "loo_wrong_date_incompatible_rate_if_random_scribe": round(loo_inc_random, 4) if loo_inc_random is not None else None,
           "kappa": round(kappa, 3),
           "era_isolated_scribes": sorted(isolated), "rules": rules, "chosen_score": chosen, "per_scribe": per_scribe}
    if not write or chosen is None:
        return res

    t = rules[chosen]["threshold"]
    keep = np.nonzero(full_U[chosen] >= t)[0]
    keep = keep[np.argsort(-full_U[chosen][keep])]
    lab_rows: Dict[int, List[int]] = defaultdict(list)
    for row, k in enumerate(lab):
        lab_rows[int(k)].append(row)
    cands, rejected, sheet_rows, headers = [], [], [], []
    for u in keep:
        cid = ids[U[u]]
        s_k = int(full_U["pred"][u])
        rows = lab_rows[s_k]
        sims = Xl[rows] @ X[U[u]]
        exemplars = [ids[L[rows[j]]] for j in np.argsort(-sims)[:3]]
        m = meta.get(cid, {})
        status = date_status(date_intervals(m), windows.get(names[s_k]))
        share = sum(holding[L[r]] == holding[U[u]] for r in rows) / len(rows)
        row = {"cand": "", "id": cid, "shelfmark": m.get("shelfmark", cid), "holding": holding[U[u]],
               "collection": manifest["collection"][cid], "proposed_scribe": names[s_k],
               "score_margin": round(float(full_U["margin"][u]), 4), "score_top1": round(float(full_U["top1"][u]), 4),
               "rule": chosen, "threshold": t, "pgp_dates": ";".join(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in date_intervals(m)),
               "scribe_window": "-".join(map(str, windows[names[s_k]])) if names[s_k] in windows else "",
               "date_check": status, "era_isolated_scribe": int(names[s_k] in isolated),
               "scribe_exemplars_same_holding_share": round(share, 3),
               "pgp_ids": ";".join(m.get("pgpids", [])), "pgp_types": ";".join(m.get("pgp_types", [])),
               "pgp_tags": ";".join(m.get("pgp_tags", []))[:200],
               "exemplar_1": exemplars[0] if exemplars else "", "exemplar_2": exemplars[1] if len(exemplars) > 1 else "",
               "exemplar_3": exemplars[2] if len(exemplars) > 2 else ""}
        if status == "incompatible":
            rejected.append(row)
            continue
        row["cand"] = f"S{len(cands) + 1:03d}"
        cands.append(row)
        sheet_rows.append([(cid, "CANDIDATE " + m.get("shelfmark", cid))] + [(e, meta.get(e, {}).get("shelfmark", e)) for e in exemplars])
        headers.append(f"{row['cand']} -> {names[s_k]}  {chosen}={full_U[chosen][u]:.3f} (thr {t:.3f}) date {status}")
    fields = list((cands or rejected or [{"cand": ""}])[0].keys())
    write_csv(OUT / "scribe_candidates.csv", cands, fields)
    write_csv(OUT / "scribe_candidates_rejected_by_date.csv", rejected, fields)
    if not args.no_render:
        for s0 in range(0, min(len(sheet_rows), args.max_scribe_sheet_rows), 10):
            render_sheet(sheet_rows[s0:s0 + 10], headers[s0:s0 + 10],
                         OUT / "scribe_sheets" / f"scribe_sheet_{s0 // 10 + 1:02d}.jpg", args.thumb, 2)
    res["candidates"] = {"n": len(cands), "rejected_by_date": len(rejected),
                         "date_check": dict(Counter(c["date_check"] for c in cands)),
                         "to_era_isolated_scribes": int(sum(c["era_isolated_scribe"] for c in cands)),
                         "by_scribe": dict(Counter(c["proposed_scribe"] for c in cands).most_common()),
                         "mean_exemplar_same_holding_share": round(float(np.mean([c["scribe_exemplars_same_holding_share"] for c in cands])), 3) if cands else None,
                         "rendered": min(len(cands), args.max_scribe_sheet_rows)}
    return res


# ----------------------------------------------------------------------------- main

def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    """Write dict rows as CSV (UTF-8 with BOM so spreadsheet apps show Hebrew).

    :param path: Output path.
    :param rows: Rows.
    :param fields: Column order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """Run join and scribe discovery and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--join-features", default=JOIN_FEATURES)
    parser.add_argument("--scribe-features", default=f"{SCRIBE_FEATURES},{JOIN_FEATURES}",
                        help="comma list; candidates are written for the one with the best LOO accuracy")
    parser.add_argument("--csls-k", type=int, default=10, help="0 disables the CSLS comparison")
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--n-cross", type=int, default=100, help="cross-collection quota (listed first)")
    parser.add_argument("--n-same", type=int, default=100, help="same-collection quota")
    parser.add_argument("--max-per-fragment", type=int, default=2)
    parser.add_argument("--max-attr-conflict", type=float, default=0.10)
    parser.add_argument("--dup-cos", type=float, default=0.995, help="fused cosine at/above which two records show one photograph")
    parser.add_argument("--near-cos", type=float, default=0.985, help="flag (not drop) candidates at/above this fused cosine")
    parser.add_argument("--partners-per-fragment", type=float, nargs=3, default=[0.1, 0.3, 1.0],
                        help="low / central / high assumption for undiscovered partners per fragment")
    parser.add_argument("--scribe-precision", type=float, default=0.8)
    parser.add_argument("--scribe-min-n", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5, help="folds for out-of-fold join calibration and the scribe rule")
    parser.add_argument("--date-slack", type=int, default=15, help="years added to each scribe's observed PGP date range")
    parser.add_argument("--max-scribe-sheet-rows", type=int, default=200)
    parser.add_argument("--thumb", type=int, default=300)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--skip-joins", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())
    sel = json.loads((AUDIT_ROOT / "image_selection_v1.json").read_text())
    ids, _ = load_feats(SCRIBE_FEATURES)
    meta = gallery_meta(ids, OUT / "gallery_meta.json", manifest["scribe"])
    print("meta", len(meta), "gallery documentary", sum(is_documentary(meta.get(c, {})) for c in ids), flush=True)

    summary: dict = {"gallery_images": len(ids)}
    if not args.skip_joins:
        summary["joins"] = join_discovery(args, manifest, meta)
        print("joins", json.dumps(summary["joins"]["candidates"]), json.dumps(summary["joins"]["validation_hide_one_source"]), flush=True)
    scribe_runs = [scribe_discovery(args, manifest, sel, meta, f, write=False) for f in args.scribe_features.split(",")]
    best = max(scribe_runs, key=lambda r: r["loo_nearest_centroid_acc"])
    summary["scribes"] = {"compared": [{k: r[k] for k in ("features", "loo_nearest_centroid_acc", "loo_balanced_acc", "loo_1nn_acc_reference",
                                                          "held_out_acc", "rules")} for r in scribe_runs],
                          "used": scribe_discovery(args, manifest, sel, meta, best["features"], write=True)}
    print("scribes", json.dumps(summary["scribes"]["compared"]), flush=True)
    summary["plain_statement"] = plain_statement(summary)
    summary["notes"] = [
        "p_labelled / band_labelled = share of gallery pairs with the same similarity/rank/cross profile that are KNOWN joins "
        "(isotonic-calibrated, out-of-fold by known-join component); p_discovery_est = rho * p/(1-p), the base-rate-corrected "
        "chance that a NOT-known pair is a join.",
        "rho (undiscovered-to-known join ratio in the gallery) is an assumption: gallery 5,176 of ~400k Genizah fragments; the "
        "hide-one-source validation checks the odds transfer itself, not rho.",
        "cross_collection uses holding_key() on the canonical id; the manifest collection strings split one holding into several "
        "spellings, which inflates eval_images' cross-collection counts"
        + (" ({} vs {} queries).".format(summary["joins"]["retrieval_known_joins"]["cosine_manifest_collection_strings"]["cross_collection"]["n"],
                                         summary["joins"]["retrieval_known_joins"]["cosine"]["cross_collection"]["n"]) if "joins" in summary else "."),
        "Discovery yield is limited by gallery coverage: an undiscovered partner is in this 5k gallery ~1.3% of the time. Extracting the "
        "same fused features for all 54k imaged fragments is the main lever for real join discovery.",
        "Scribe candidates are fragments with PGP documentary records and no PGP scribe relation; exemplars are the nearest labelled "
        "fragments of the proposed scribe (same-document pairs never used).",
    ]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (HERE / "results").mkdir(exist_ok=True)
    shutil.copy(OUT / "summary.json", HERE / "results" / "discovery_v1_summary.json")
    print(summary["plain_statement"])


def plain_statement(summary: dict) -> str:
    """Plain-language reading of the expected yield.

    :param summary: The assembled summary.
    :returns: Statement text.
    :rtype: str
    """
    parts = []
    j = summary.get("joins")
    if j:
        c = j["candidates"]
        lo, mid, hi = list(c["expected_real_joins_by_assumption"].values())
        v = j["validation_hide_one_source"]
        parts.append(
            f"JOINS: {c['n']} candidates ({c['cross_collection']} cross-collection, listed first; central expected real "
            f"{c['expected_real_central_cross_vs_same'][0]:g} cross vs {c['expected_real_central_cross_vs_same'][1]:g} same). "
            f"Among gallery pairs that look like these, "
            f"{c['mean_p_labelled']:.0%} are known joins on average (out-of-fold calibration, AUC "
            f"{j['calibration']['auc_on_pool_out_of_fold']}), but the gallery was built around known joins, so a pair that is "
            f"NOT already known is far less likely to be one: we expect about {mid:g} real new joins in the 200 "
            f"(range {lo:g}-{hi:g} for 0.1-1.0 undiscovered partners per fragment; x{j['validation_found_over_predicted']:g} "
            f"= {c['expected_real_joins_central_x_validation_ratio']:g} after the hide-one-source check, in which the procedure found "
            f"{v['hide_pgp']['hidden_joins_found']} hidden PGP joins vs {v['hide_pgp']['predicted_hidden_joins']:g} predicted and "
            f"{v['hide_fjp']['hidden_joins_found']} hidden FJP joins vs {v['hide_fjp']['predicted_hidden_joins']:g} predicted). "
            f"In other words: expect zero to a few real joins; the list is a calibrated review queue, not a set of finds.")
    s = summary["scribes"]["used"]
    if s.get("chosen_score"):
        r = s["rules"][s["chosen_score"]]
        m, d = r["estimate_moments"], r["estimate_dates"]
        x = d["excluding_era_isolated_scribes"]
        comp = "; ".join(f"{c['features'].split('__')[0] if '+' not in c['features'] else 'fusion'}: centroid {c['loo_nearest_centroid_acc']:.3f}, "
                         f"1-NN {c['loo_1nn_acc_reference']:.3f}, held-out {c['held_out_acc']}" for c in summary["scribes"]["compared"])
        parts.append(
            f"SCRIBES: leave-one-out accuracy over {s['n_scribes']} scribes ({comp}); candidates use {s['features']}. "
            f"At {s['chosen_score']} >= {r['threshold']} LOO precision is {r['loo_precision']:.2f} at {r['loo_coverage']:.0%} coverage "
            f"in-sample (the threshold was picked there); nested cross-validation gives "
            f"{r['nested_cv']['precision']} at {r['nested_cv']['coverage']:.0%} coverage. "
            f"{r['unlabelled_above_threshold']} of {r['unlabelled_documentary']} unlabelled documentary fragments clear it "
            f"({s['unlabelled_excluded_linked_to_labelled']} others were left out because a known join or shared PGP document "
            f"already ties them to a labelled fragment), "
            f"{s['candidates']['rejected_by_date']} fall outside the proposed scribe's PGP dates, {s['candidates']['n']} are listed. "
            f"{s['candidates']['to_era_isolated_scribes']} of those go to scribes who are alone in their period "
            f"({', '.join(s['era_isolated_scribes'])}); for them a matching date shows the period, not the hand. For the other "
            f"{x['candidates']} the date check gives error {x.get('error_rate')} and about {x.get('expected_correct')} correct "
            f"(95% range {x.get('expected_correct_95ci')}, from only {x['dated']} dated). Over all {s['candidates']['n']}: about "
            f"{d['expected_correct_after_date_filter']} correct by dates, {m['expected_correct']} by the open-set moments estimate; both "
            f"are optimistic for period- or script-isolated scribes. The 0.8 LOO precision does NOT carry over to unlabelled documents.")
    v = next((c for c in summary["scribes"]["compared"] if "+" not in c["features"]), None)
    if v and v["rules"].get("margin", {}).get("threshold") is not None:
        vr = v["rules"]["margin"]
        top = next(iter(vr["above_by_scribe"].items()), ("", 0))
        parts.append(
            f"The v22b tower alone is not usable for attribution: {top[1]} of its {vr['unlabelled_above_threshold']} confident picks go to "
            f"{top[0]}, and {vr['estimate_dates']['date_incompatible']} of {vr['estimate_dates']['above_dated']} dated picks are outside "
            f"the proposed scribe's dates.")
    return " ".join(parts)


if __name__ == "__main__":
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # Accelerate BLAS raises spurious matmul FP flags
        main()
