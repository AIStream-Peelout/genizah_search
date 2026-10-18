"""Join and scribe discovery over the FULL imaged gallery (v2), from the Colab gallery shards. NumPy only, no model.

v1 (``discover_joins_scribes.py``) searched the 5,176 benchmark images, which were chosen around known joins, so an
undiscovered partner was in its gallery ~1 % of the time. v2 searches every imaged fragment (55,406; features from
``colab/embed_genizah_gallery.ipynb``, private HF dataset ``isaacmg/genizah-image-gallery``).

Inputs: ``<AUDIT_ROOT>/gallery_v1/manifest.jsonl`` + ``known_joins.json`` (``build_gallery_manifest.py``) and the
shards (``--fetch`` downloads them into ``<AUDIT_ROOT>/gallery_v1/hf_cache`` with every HF cache on the NAS).
Outputs: ``<AUDIT_ROOT>/results/discovery_v2/``: ``join_candidates.csv``, ``join_candidates_ranked.csv``,
``possible_duplicate_records.csv``, ``calibration_heldout.csv``, ``scribe_candidates*.csv``, ``summary.json``,
``neighbours.npz`` (top-50 neighbours of every fragment) and ``pool.npz`` (calibration pool, for re-fitting).

Joins
-----
1. *Similarity.* Per row block, the cosine of each stored encoder (tuned ``merger`` and ``ds8``, ``head``, DINOv2
   ``masked`` and ``mpatches``) against the whole gallery; variants are means of those cosines: ``fused`` (merger +
   masked + mpatches, ranks the candidates), ``merger``, ``pooled`` (ds8 + merger = the tuned pooled readout),
   ``dinov2`` and ``head``. CSLS (k = 10) needs every item's neighbourhood density, so the gallery is scanned twice.
   Known-join retrieval (R@1/10/100, MRR, median rank of the first partner among all 55k) is reported per variant
   for cosine and CSLS, same- vs cross-holding; the ranking variant uses CSLS when it retrieves known joins better.
2. *Known structure.* Positives = FJP clean join lists + PGP shared documents (2-6 fragments) + FJP lists matched
   loosely or through the manifest's resolver fallbacks (not from prefix-bug items). A pair inside one ``link_group``
   (every known link, incl. FJP suspects, FJP items shared by several records and large PGP documents), or linked
   through a duplicate record of one side (``dup_group``), is never a candidate and is excluded from calibration
   unless it is a positive.
3. *Calibration with a grouped split.* Pool = pairs where either fragment is in the other's top ``pool_k``. Features
   (v1): ranking similarity, log best/worst mutual rank, cross-holding, similarity x cross. Link groups (and single
   fragments) are split in two halves; a pair belongs to the half of its first fragment's group, so a whole known
   join group is on one side and positives and negatives are sampled alike. Pairs that can never be proposed
   (duplicates, shared PGP document, container records) are not calibration negatives. A logistic + isotonic calibrator
   (:class:`CalibratorV2`: v1's logistic score; isotonic smoothing fixed, see :func:`isotonic_smoothed`; the isotonic
   map fitted per same-/cross-holding stratum) fitted on half A
   scores half B and vice versa; the precision tables are those held-out predictions, same- and
   cross-holding separately: ``p_labelled`` is the chance that a pair with this profile is a KNOWN join.
4. *Base rate.* For a pair that is not known, P(join) = rho p / (1 - p) with rho = undiscovered / known joins in the
   gallery (assumption bracket from ``partners_per_fragment``; ``--rho`` overrides). The odds transfer is checked by
   hiding one label source (PGP or FJP) and counting hidden joins recovered vs predicted.
5. *Exclusions.* Known/linked pairs, shared PGP document, the same shelfmark under two records (same holding and
   shelfmark tokens, or v1 ``same_record``), the same photograph (same URL or download SHA-1, the same file in both
   records' image lists, difference hash within ``dhash_max`` bits, or fused cosine >= ``dup_cos``), container
   records (box-level records whose first image is one of their items'), and material / script-type (etc.)
   conflicts for attributes that known joins rarely contradict.
6. *Output.* ``join_candidates.csv``: cross-holding quota first, then same-holding, at most ``max_per_fragment`` per
   fragment; ``join_candidates_ranked.csv``: the deeper ranked list. Both carry ``p_labelled``, ``p_discovery`` (central
   rho, low/high bracket) and the expected number of real joins (sum of ``p_discovery``) by list depth.

Scribes
-------
Leave-one-out nearest-centroid accuracy over the PGP scribes with >= 5 labelled fragments in the gallery (same-document
fragments never inform a query) measures each encoder; the best one scores every unlabelled PGP documentary fragment
not tied to a labelled one. The threshold differs from v1: v1 took the score where LOO precision reaches 0.8, which
ignores that most unlabelled documents are in OTHER hands and stops being selective once LOO accuracy exceeds 0.8 (on
synthetic data it accepted every document). v2 uses target-decoy competition: decoy "scribes" are sets of unlabelled
documentary fragments with each real scribe's size and holding mix (:func:`build_decoys`); every unlabelled document
competes for real and decoy centroids, and a decoy win marks what a false attribution looks like, so the precision of
the attributions above a score is ``1 - (decoy wins + 1) / (decoy sets x real wins)`` (:func:`decoy_rule`). The
threshold maximises the expected correct attributions at estimated precision >= 0.8; candidates carry the rule's and
their own score's (q-value-like) precision. (A mixture model with a leave-one-scribe-out open-set reference was tried
first and failed the synthetic test: removing a query's own, often large and generic, centroid makes that reference
score lower than real open-set documents.) The PGP-date check is an independent estimate; the v1 rule, its nested
cross-validation and the rare-scribe acceptances are reported for reference.

Fragments whose KTIV manuscript supplied the tuned tower's training lines (``tower_training_ms``, or a duplicate
record of one: ``tower_training_ms_twin``) are flagged, and the split of either side (``train`` / ``val`` = used for
checkpoint selection / ``test``) is in the CSVs; their known-join retrieval is reported separately.

Run (after the shards exist): ``VECLIB_MAXIMUM_THREADS=8 nice -n 10 <HDA .venv>/bin/python discover_joins_v2.py --fetch``.
``test_discover_joins_v2.py`` runs this end to end on synthetic features with planted joins/scribes.
"""

import argparse
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from discover_joins_scribes import (ATTR_KEYS, GENIZAH_SIZE, JoinCalibrator, UnionFind, attr_conflicts, centroid_scores,
                                    chance_compatible, cv_rule_precision, date_expected_correct, date_intervals,
                                    date_status, disc_precision, fit_logistic, pair_features, pick_with_quota,
                                    precision_threshold, rho_estimate, same_record, select_candidates, shelf_neighbours,
                                    wilson, write_csv)
from embed_utils import AUDIT_ROOT
from eval_images import pair_auc

HERE = Path(__file__).resolve().parent
HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
GALLERY = AUDIT_ROOT / "gallery_v1"
HF_CACHE = GALLERY / "hf_cache"
OUT = AUDIT_ROOT / "results" / "discovery_v2"
REPO = "isaacmg/genizah-image-gallery"
BLOCKS = ("merger", "ds8", "head", "dino_masked", "dino_mpatches")
VARIANTS: Dict[str, Dict[str, float]] = {
    "fused": {"merger": 1 / 3, "dino_masked": 1 / 3, "dino_mpatches": 1 / 3},
    "merger": {"merger": 1.0},
    "pooled": {"ds8": 0.5, "merger": 0.5},
    "dinov2": {"dino_masked": 0.5, "dino_mpatches": 0.5},
    "head": {"head": 1.0},
}
COS_RANGE, CSLS_RANGE, HIST_WIDTH = (-1.0, 1.0), (-4.0, 2.0), 0.001
CALIBRATION_LOG: List[dict] = []  # per-half fits of the last calibrate_halves call (reported in the summary)


# ------------------------------------------------------------------------------------------------- loading
def fetch_gallery(repo: str, dest: Path) -> Path:
    """Download the shards (and failure logs, benchmark outputs, index) into the NAS cache.

    All Hugging Face caches (``HF_HOME``, the xet chunk cache) are pointed at the NAS first: the Studio's internal disk
    is nearly full. The token is ``HF1_TOKEN`` from the sibling repo's ``.env`` (never printed).

    :param repo: Dataset repo id.
    :param dest: Local folder.
    :returns: The folder.
    :rtype: Path
    """
    os.environ["HF_HOME"] = str(AUDIT_ROOT / "hf")
    os.environ["HF_XET_CACHE"] = str(AUDIT_ROOT / "hf" / "xet")
    from dotenv import dotenv_values
    from huggingface_hub import snapshot_download

    token = dotenv_values(HDA / ".env").get("HF1_TOKEN")
    path = snapshot_download(repo, repo_type="dataset", token=token, local_dir=str(dest),
                             allow_patterns=["shards/*", "failures/*", "benchmark/*", "gallery_index.json"])
    return Path(path)


def read_manifest(path: Path) -> List[dict]:
    """Gallery manifest rows in file order.

    :param path: ``manifest.jsonl``.
    :returns: Rows.
    :rtype: List[dict]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


class Gallery:
    """Fragments with features, aligned with their manifest rows."""

    def __init__(self, shards_dir: Path, manifest: List[dict]) -> None:
        """Load every shard (``part_*.npz``; retry shards ``part_r*`` included; first occurrence of an id wins).

        :param shards_dir: Folder with the shards.
        :param manifest: Manifest rows.
        """
        order = {r["canonical_id"]: i for i, r in enumerate(manifest)}
        parts: Dict[str, list] = defaultdict(list)
        ids: List[str] = []
        seen: Set[str] = set()
        self.n_unknown = self.n_duplicate = 0
        files = sorted(shards_dir.glob("part_*.npz"))
        assert files, f"no shards in {shards_dir}"
        for f in files:
            z = np.load(f)
            fid = [str(c) for c in z["ids"]]
            if not fid:  # a shard whose every fragment failed is written with (0, 1) placeholder arrays
                continue
            keep = []
            for k, c in enumerate(fid):
                if c not in order:
                    self.n_unknown += 1
                elif c in seen:
                    self.n_duplicate += 1
                else:
                    seen.add(c)
                    keep.append(k)
            ids += [fid[k] for k in keep]
            for key in BLOCKS + ("sha1", "dhash", "mask_ok", "urls"):
                parts[key].append(z[key][keep])
        idx = np.argsort([order[c] for c in ids], kind="stable")
        self.ids = [ids[i] for i in idx]
        self.n = len(self.ids)
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.rows = [manifest[order[c]] for c in self.ids]
        self.X: Dict[str, np.ndarray] = {}
        for b in BLOCKS:
            X = np.concatenate(parts[b])[idx].astype(np.float32)
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            self.X[b] = X
        self.sha1 = np.concatenate(parts["sha1"])[idx]
        self.dhash = np.concatenate(parts["dhash"])[idx].astype(np.uint64)
        self.mask_ok = np.concatenate(parts["mask_ok"])[idx]
        self.urls = np.concatenate(parts["urls"])[idx]
        self.hold = np.array([r["holding"] for r in self.rows])
        self.hold_code = np.unique(self.hold, return_inverse=True)[1]
        self.container = np.array([bool(r.get("container_record")) for r in self.rows])
        self.tower_train = np.array([bool(r.get("tower_training_ms") or r.get("tower_training_ms_twin")) for r in self.rows])
        self.n_manifest = len(manifest)
        self.n_manifest_live = sum(1 for r in manifest if r.get("image_live") is not False)

    def pair_sims(self, a: np.ndarray, b: np.ndarray, weights: Dict[str, float], chunk: int = 16_384) -> np.ndarray:
        """Variant similarity of index pairs (a[i], b[i]).

        :param a: Indices.
        :param b: Indices.
        :param weights: Block weights of the variant.
        :param chunk: Pairs per step.
        :returns: float32 similarities.
        :rtype: np.ndarray
        """
        out = np.zeros(len(a), np.float32)
        for s in range(0, len(a), chunk):
            aa, bb = a[s:s + chunk], b[s:s + chunk]
            for blk, w in weights.items():
                out[s:s + chunk] += w * np.einsum("ij,ij->i", self.X[blk][aa], self.X[blk][bb])
        return out

    def row_sims(self, rows: np.ndarray, variants: Sequence[str]) -> Dict[str, np.ndarray]:
        """Similarities of a row block to the whole gallery for several variants (one matmul per encoder).

        :param rows: Row indices.
        :param variants: Variant names.
        :returns: {variant: (len(rows), n) float32}.
        :rtype: Dict[str, np.ndarray]
        """
        need = sorted({b for v in variants for b in VARIANTS[v]})
        with np.errstate(all="ignore"):  # spurious macOS Accelerate FP flags in float32 matmul
            base = {b: self.X[b][rows] @ self.X[b].T for b in need}
        out = {}
        for v in variants:
            S = None
            for b, w in VARIANTS[v].items():
                S = base[b] * np.float32(w) if S is None else S + base[b] * np.float32(w)
            out[v] = S
        return out

    def variant_matrix(self, variant: str, rows: Optional[Sequence[int]] = None) -> np.ndarray:
        """Unit vectors whose dot products are the variant similarity (blocks scaled by sqrt(weight)).

        :param variant: Variant name.
        :param rows: Optional row subset.
        :returns: (m, D) float32.
        :rtype: np.ndarray
        """
        sel = slice(None) if rows is None else np.asarray(rows)
        return np.concatenate([self.X[b][sel] * np.float32(math.sqrt(w)) for b, w in VARIANTS[variant].items()], axis=1)


# ------------------------------------------------------------------------------------------------- known structure
def known_structure(G: Gallery, known: dict, loose_positive: bool) -> dict:
    """Positive pairs by source, link groups and partners, restricted to the loaded gallery.

    :param G: Gallery.
    :param known: ``known_joins.json``.
    :param loose_positive: Count clean loosely matched FJP pairs as positives.
    :returns: {pos_by_source, positives, link (group code per index), partners, ambiguous}.
    :rtype: dict
    """
    def idx_pairs(pairs: Iterable[Sequence]) -> Set[Tuple[int, int]]:
        """Id pairs -> sorted gallery index pairs (pairs with a fragment outside the gallery dropped).

        :param pairs: Rows whose first two items are canonical ids.
        :returns: Index pairs (i < j).
        :rtype: Set[Tuple[int, int]]
        """
        out = set()
        for p in pairs:
            a, b = p[0], p[1]
            if a in G.pos and b in G.pos and a != b:
                i, j = G.pos[a], G.pos[b]
                out.add((min(i, j), max(i, j)))
        return out

    P = known["pairs"]
    by_src = {"fjp_clean": idx_pairs(P["fjp_clean"]), "pgp": idx_pairs(P["pgp"]),
              "fjp_loose": idx_pairs(p for p in P["fjp_loose"] if not p[3]) if loose_positive else set()}
    links = {"fjp_suspect": idx_pairs(P["fjp_suspect"]), "pgp_large_or_document_only": idx_pairs(P["pgp_large_or_document_only"]),
             "fjp_loose_any": idx_pairs(P["fjp_loose"]), "fjp_shared_item": idx_pairs(P.get("fjp_shared_item", []))}
    positives = set().union(*by_src.values())
    link = np.full(G.n, -1, dtype=np.int64)
    codes = {}
    for i, r in enumerate(G.rows):
        if r.get("link_group"):
            link[i] = codes.setdefault(r["link_group"], len(codes))
    members: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(link):
        if g >= 0:
            members[int(g)].append(i)
    grouped = {(a, b) for m in members.values() for a in m for b in m if a < b}
    missing = {p for p in positives if link[p[0]] < 0 or link[p[0]] != link[p[1]]}
    assert not missing, f"{len(missing)} positives outside a common link group (manifest/known_joins out of sync)"
    partners: Dict[int, Set[int]] = defaultdict(set)
    for a, b in positives:
        partners[a].add(b)
        partners[b].add(a)
    # exclusion closure: a known link seen through a duplicate record of one side (A' = A catalogued twice, A-B known)
    # is still known; containers (box records) are not in any dup_group, so they cannot merge a box's items
    uf = UnionFind(G.n)
    dup_members: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(G.rows):
        if r.get("dup_group"):
            dup_members[r["dup_group"]].append(i)
    for m in list(members.values()) + list(dup_members.values()):
        for x in m[1:]:
            uf.union(m[0], x)
    closure = np.array([uf.find(i) for i in range(G.n)], dtype=np.int64)
    return {"pos_by_source": by_src, "links": links, "positives": positives, "link": link, "grouped": grouped,
            "ambiguous": grouped - positives, "partners": dict(partners), "closure": closure,
            "dup_members": dict(dup_members)}


# ------------------------------------------------------------------------------------------------- scanning
def rstats(ranks: Sequence[int]) -> dict:
    """Recall / MRR / median of first-partner ranks.

    :param ranks: Ranks (1 = nearest).
    :returns: Stats.
    :rtype: dict
    """
    rs = np.asarray(ranks)
    if not len(rs):
        return {"n": 0}
    return {"n": int(len(rs)), "R@1": round(float((rs <= 1).mean()), 4), "R@10": round(float((rs <= 10).mean()), 4),
            "R@100": round(float((rs <= 100).mean()), 4), "MRR": round(float((1 / rs).mean()), 4),
            "median_rank": int(np.median(rs))}


class Histogram:
    """Fixed-width histogram of all gallery pair similarities (upper triangle), per same/cross-holding stratum."""

    def __init__(self, lo: float, hi: float, width: float = HIST_WIDTH) -> None:
        """Allocate bins.

        :param lo: Lower edge.
        :param hi: Upper edge.
        :param width: Bin width.
        """
        self.lo, self.width = lo, width
        self.nb = int(round((hi - lo) / width))
        self.counts = {"same": np.zeros(self.nb, np.int64), "cross": np.zeros(self.nb, np.int64)}

    def add(self, vals: np.ndarray, stratum: str) -> None:
        """Count values (clipped into the end bins).

        :param vals: 1-D values.
        :param stratum: ``same`` or ``cross``.
        """
        idx = np.clip(((vals - self.lo) / self.width).astype(np.int64), 0, self.nb - 1)
        self.counts[stratum] += np.bincount(idx, minlength=self.nb)

    def at_least(self, t: float, stratum: str) -> int:
        """Pairs with value >= t (bin resolution).

        :param t: Threshold.
        :param stratum: Stratum.
        :returns: Count.
        :rtype: int
        """
        k = int(np.clip(math.ceil((t - self.lo) / self.width - 1e-9), 0, self.nb))
        return int(self.counts[stratum][k:].sum())


def scan(G: Gallery, variants: Sequence[str], rank_variant: str, partners: Dict[int, Set[int]], csls_k: int,
         store_k: int, block: int) -> dict:
    """Two passes over the gallery: CSLS densities, then neighbour lists, retrieval ranks and pair histograms.

    :param G: Gallery.
    :param variants: Variants to evaluate.
    :param rank_variant: Variant whose neighbour lists and histograms are kept.
    :param partners: Known-join partners per index.
    :param csls_k: CSLS neighbourhood size.
    :param store_k: Neighbours stored per fragment (ranks beyond are censored).
    :param block: Rows per block.
    :returns: {r, ranks (variant -> {cosine, csls} -> {query: rank}), nn (rank variant: cosine/csls idx + val),
        hist (cosine/csls Histogram)}.
    :rtype: dict
    """
    n = G.n
    qs = np.array(sorted(partners), dtype=np.int64)
    is_q = np.zeros(n, bool)
    is_q[qs] = True
    pa = np.array([q for q in qs for p in partners[q]], dtype=np.int64)
    pb = np.array([p for q in qs for p in partners[q]], dtype=np.int64)
    best = {}
    for v in variants:
        s = G.pair_sims(pa, pb, VARIANTS[v])
        m = np.full(n, -np.inf, np.float32)
        np.maximum.at(m, pa, s)
        best[v] = (m, s)
    r = {v: np.zeros(n, np.float32) for v in variants}
    ranks = {v: {"cosine": np.zeros(n, np.int64), "csls": np.zeros(n, np.int64)} for v in variants}
    t0 = time.time()
    for s0 in range(0, n, block):  # pass A: densities + cosine retrieval ranks
        rows = np.arange(s0, min(n, s0 + block))
        sims = G.row_sims(rows, variants)
        for v, S in sims.items():
            S[np.arange(len(rows)), rows] = -np.inf
            r[v][rows] = np.partition(S, -csls_k, axis=1)[:, -csls_k:].mean(1)
            q = is_q[rows]
            if q.any():
                ranks[v]["cosine"][rows[q]] = 1 + (S[q] > best[v][0][rows[q]][:, None]).sum(1)
        if s0 // block % 20 == 0:
            print(f"  pass A {s0 + len(rows):,}/{n:,} ({time.time() - t0:.0f} s)", flush=True)
    best_c = {}
    for v in variants:
        c = 2 * best[v][1] - r[v][pa] - r[v][pb]
        m = np.full(n, -np.inf, np.float32)
        np.maximum.at(m, pa, c)
        best_c[v] = m
    nn = {k: (np.zeros((n, store_k), np.int32), np.zeros((n, store_k), np.float32)) for k in ("cosine", "csls")}
    hist = {"cosine": Histogram(*COS_RANGE), "csls": Histogram(*CSLS_RANGE)}
    cols = np.arange(n)
    t0 = time.time()
    for s0 in range(0, n, block):  # pass B: CSLS ranks, neighbour lists, all-pair histograms
        rows = np.arange(s0, min(n, s0 + block))
        sims = G.row_sims(rows, variants)
        for v, S in sims.items():
            S[np.arange(len(rows)), rows] = -np.inf
            C = 2 * S - r[v][rows][:, None] - r[v][None, :]
            q = is_q[rows]
            if q.any():
                ranks[v]["csls"][rows[q]] = 1 + (C[q] > best_c[v][rows[q]][:, None]).sum(1)
            if v != rank_variant:
                continue
            for key, M in (("cosine", S), ("csls", C)):
                top = np.argpartition(-M, store_k, axis=1)[:, :store_k]
                val = np.take_along_axis(M, top, axis=1)
                order = np.argsort(-val, axis=1, kind="stable")
                nn[key][0][rows] = np.take_along_axis(top, order, axis=1)
                nn[key][1][rows] = np.take_along_axis(val, order, axis=1)
                upper = cols[None, :] > rows[:, None]
                cross = G.hold_code[rows][:, None] != G.hold_code[None, :]
                hist[key].add(M[upper & ~cross], "same")
                hist[key].add(M[upper & cross], "cross")
        if s0 // block % 20 == 0:
            print(f"  pass B {s0 + len(rows):,}/{n:,} ({time.time() - t0:.0f} s)", flush=True)
    return {"r": r, "ranks": {v: {k: {int(q): int(x[q]) for q in qs} for k, x in d.items()} for v, d in ranks.items()},
            "nn": nn, "hist": hist}


def retrieval_report(G: Gallery, ranks: Dict[int, int], partners: Dict[int, Set[int]]) -> dict:
    """Known-join retrieval breakdown for one similarity.

    :param G: Gallery.
    :param ranks: query -> first-partner rank.
    :param partners: Partners per query.
    :returns: {all, cross_holding, same_holding, in_benchmark, outside_benchmark, touching_tower_training_ms}.
    :rtype: dict
    """
    groups = defaultdict(list)
    for q, rk in ranks.items():
        groups["all"].append(rk)
        cross = not any(G.hold[p] == G.hold[q] for p in partners[q])
        groups["cross_holding" if cross else "same_holding"].append(rk)
        groups["in_benchmark" if G.rows[q].get("in_benchmark") else "outside_benchmark"].append(rk)
        if G.rows[q].get("tower_training_ms") or any(G.rows[p].get("tower_training_ms") for p in partners[q]):
            groups["touching_tower_training_ms"].append(rk)
    return {k: rstats(groups.get(k, [])) for k in ("all", "same_holding", "cross_holding", "in_benchmark",
                                                    "outside_benchmark", "touching_tower_training_ms")}


# ------------------------------------------------------------------------------------------------- pool + calibration
def build_pool(nn_idx: np.ndarray, pool_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unordered pairs where either fragment is in the other's top ``pool_k``, with both directed ranks.

    :param nn_idx: (n, K) neighbour indices, best first.
    :param pool_k: Pool depth.
    :returns: (pairs (m, 2) with i < j, rank of j in i's list, rank of i in j's list); ranks beyond K are K + 1.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    n, K = nn_idx.shape
    src = np.repeat(np.arange(n, dtype=np.int64), K)
    dst = nn_idx.reshape(-1).astype(np.int64)
    rank = np.tile(np.arange(1, K + 1), n)
    keys = src * n + dst
    order = np.argsort(keys)
    keys, rank_sorted = keys[order], rank[order]
    sel = rank <= pool_k
    lo, hi = np.minimum(src[sel], dst[sel]), np.maximum(src[sel], dst[sel])
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)

    def lookup(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Rank of b in a's list (K + 1 when absent).

        :param a: Indices.
        :param b: Indices.
        :returns: Ranks.
        :rtype: np.ndarray
        """
        k = a * n + b
        at = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        return np.where(keys[at] == k, rank_sorted[at], K + 1)

    return pairs, lookup(pairs[:, 0], pairs[:, 1]), lookup(pairs[:, 1], pairs[:, 0])


def isotonic_smoothed(score: np.ndarray, y: np.ndarray, prior_n: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Isotonic P(y | score) with smoothing that keeps it monotone (replaces v1's ``isotonic_blocks`` here).

    v1 smoothed each pool-adjacent-violators block with Laplace (k + 1) / (n + 2) and then took a running maximum.
    A small all-negative block at the bottom (a known join with a very low score closes the zero run before it) then
    gets 1 / (n + 2) — e.g. 0.004 for 228 pairs — and the running maximum lifts every larger, lower block to that
    value, inflating ``p_labelled`` at the low end (on synthetic data the held-out total was 2.3x the observed).
    Here each block is shrunk towards the overall positive rate with ``prior_n`` pseudo-pairs, and adjacent blocks
    whose smoothed precision would decrease are pooled again.

    :param score: Scores.
    :param y: Binary labels.
    :param prior_n: Pseudo-pairs of the prior.
    :returns: (upper score edge per block, precision per block), both ascending.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    order = np.argsort(score, kind="stable")
    sums: List[float] = []
    cnts: List[float] = []
    edges: List[float] = []
    for xi, yi in zip(score[order].tolist(), y[order].astype(float).tolist()):  # pool adjacent violators (raw means)
        sums.append(yi)
        cnts.append(1.0)
        edges.append(xi)
        while len(sums) > 1 and sums[-2] * cnts[-1] >= sums[-1] * cnts[-2]:
            s_, c_, e_ = sums.pop(), cnts.pop(), edges.pop()
            sums[-1] += s_
            cnts[-1] += c_
            edges[-1] = e_
    base = float(np.mean(y))
    sm = lambda s_, c_: (s_ + prior_n * base) / (c_ + prior_n)  # noqa: E731
    stack: List[List[float]] = []
    for e_, s_, c_ in zip(edges, sums, cnts):
        stack.append([e_, s_, c_])
        while len(stack) > 1 and sm(stack[-2][1], stack[-2][2]) >= sm(stack[-1][1], stack[-1][2]):
            e2, s2, c2 = stack.pop()
            stack[-1][0] = e2
            stack[-1][1] += s2
            stack[-1][2] += c2
    return np.array([b[0] for b in stack]), np.array([sm(b[1], b[2]) for b in stack])


class CalibratorV2(JoinCalibrator):
    """v1's logistic score, mapped to a precision by :func:`isotonic_smoothed` fitted SEPARATELY per stratum.

    One pooled isotonic map (v1) assumes the logistic's cross-holding terms already put same- and cross-holding pairs
    on one scale. They need not: the fit is dominated by the ~10^5 low-score pairs, and on synthetic data the cross
    terms came out ~0 while held-out cross-holding pairs were over-predicted by ~35 % in every band above 0.05 (it only
    looked fine while duplicate photographs, mostly cross-holding, sat in the fit as top-scoring negatives). So the
    map is fitted per stratum (``cross`` column of :func:`pair_features`), falling back to the pooled map for a
    stratum with fewer than ``min_pos`` positives.
    """

    CROSS_COL = 3  # pair_features: sim, log best rank, log worst rank, cross, sim * cross

    def __init__(self, l2: float = 1.0, min_pos: int = 20) -> None:
        """Create an unfitted calibrator.

        :param l2: Ridge strength of the logistic model.
        :param min_pos: Fewest positives for a stratum's own isotonic map.
        """
        super().__init__(l2)
        self.min_pos = min_pos
        self.maps: Dict[bool, Tuple[np.ndarray, np.ndarray]] = {}
        self.own_map: Dict[bool, bool] = {}

    def fit(self, F: np.ndarray, y: np.ndarray) -> "CalibratorV2":
        """Standardise, fit the logistic model, then a monotone smoothed isotonic map per stratum on its scores.

        :param F: Raw features from :func:`pair_features`.
        :param y: Labels.
        :returns: self.
        :rtype: CalibratorV2
        """
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-9
        self.w = fit_logistic((F - self.mu) / self.sd, y, self.l2)
        z = self._logit(F)
        self.edges, self.prec = isotonic_smoothed(z, y)  # pooled map (fallback, reported)
        stratum = F[:, self.CROSS_COL] > 0.5
        for st in (False, True):
            m = stratum == st
            self.own_map[st] = bool(y[m].sum() >= self.min_pos and (y[m] == 0).any())
            self.maps[st] = isotonic_smoothed(z[m], y[m]) if self.own_map[st] else (self.edges, self.prec)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Calibrated labelled precision (each pair through its stratum's map).

        :param F: Raw features.
        :returns: Probabilities.
        :rtype: np.ndarray
        """
        z = self._logit(F)
        stratum = F[:, self.CROSS_COL] > 0.5
        out = np.empty(len(F))
        for st, (edges, prec) in self.maps.items():
            m = stratum == st
            k = np.searchsorted(edges, z[m], side="left")
            out[m] = prec[np.minimum(k, len(prec) - 1)]
        return out


def calibrate_halves(F: np.ndarray, y: np.ndarray, fit_rows: np.ndarray, unit_a: np.ndarray, seed: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Grouped two-half cross-fitting of :class:`CalibratorV2`.

    Units (link groups / single fragments) are split at random into two halves and every pair belongs to the half
    of its FIRST fragment's unit, so all pairs of a known join group fall on one side, and positives and negatives
    are sampled at the same rate (requiring both fragments in one half would keep every positive but only half of the
    negatives, which doubles the odds of the fitted precision). The calibrator fitted on half 0 scores half 1 and vice
    versa: every pool pair gets an out-of-half ``p_labelled``.

    :param F: Pair features (:func:`pair_features`).
    :param y: Known-join label per pair.
    :param fit_rows: Rows usable for fitting / evaluation (not ambiguous).
    :param unit_a: Unit of the first (lower-index) fragment of each pair.
    :param seed: Split seed.
    :returns: (out-of-half precision per pair, half per pair).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    uniq, inv = np.unique(unit_a, return_inverse=True)
    half = (np.random.default_rng(seed).permutation(len(uniq)) % 2)[inv]
    p = np.empty(len(F))
    CALIBRATION_LOG.clear()
    for h in (0, 1):
        tr = fit_rows & (half == h)
        assert y[tr].sum() >= 5, f"half {h} has only {int(y[tr].sum())} positives"
        cal = CalibratorV2().fit(F[tr], y[tr].astype(float))
        p[half == 1 - h] = cal.predict(F[half == 1 - h])
        CALIBRATION_LOG.append({"fitted_on_half": h, "pairs": int(tr.sum()), "positives": int(y[tr].sum()),
                                "positives_cross_holding": int(y[tr & (F[:, CalibratorV2.CROSS_COL] > 0.5)].sum()),
                                "weights": [round(float(v), 4) for v in cal.w],
                                "isotonic_blocks_same_cross": [int(len(cal.maps[st][1])) for st in (False, True)],
                                "own_isotonic_map_same_cross": [cal.own_map[st] for st in (False, True)],
                                "top_block_precision_same_cross": [round(float(cal.maps[st][1][-1]), 4) for st in (False, True)]})
    return p, half


def heldout_report(p: np.ndarray, y: np.ndarray, rows: np.ndarray, cross: np.ndarray, half: np.ndarray) -> dict:
    """Held-out calibration: reliability bins, precision above thresholds, AUC; all / same / cross, per half.

    :param p: Out-of-half precision.
    :param y: Labels.
    :param rows: Evaluation rows (fit rows inside a half).
    :param cross: Cross-holding flag per pair.
    :param half: Half per pair.
    :returns: Report dict.
    :rtype: dict
    """
    bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01]
    out = {}
    for name, m0 in (("all", rows), ("same_holding", rows & (cross == 0)), ("cross_holding", rows & (cross == 1))):
        rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = m0 & (p >= lo) & (p < hi)
            k, nn = int(y[m].sum()), int(m.sum())
            ci = wilson(k, nn) if nn else (None, None)
            rel.append({"p_band": f"{lo}-{min(hi, 1)}", "pairs": nn, "known_joins": k,
                        "mean_predicted": round(float(p[m].mean()), 4) if nn else None,
                        "observed": round(k / nn, 4) if nn else None,
                        "observed_95ci": [round(ci[0], 4), round(ci[1], 4)] if nn else None})
        prec = []
        for t in (0.6, 0.4, 0.2, 0.1, 0.05, 0.02):
            m = m0 & (p >= t)
            prec.append({"p_at_least": t, "pairs": int(m.sum()), "known_joins": int(y[m].sum()),
                         "precision": round(float(y[m].mean()), 4) if m.any() else None,
                         "recall_of_pool_positives": round(float(y[m].sum() / max(1, y[m0].sum())), 4)})
        per_half = {f"half{h}": {"pairs": int((m0 & (half == h)).sum()), "known_joins": int(y[m0 & (half == h)].sum()),
                                 "auc": round(pair_auc(p[m0 & (half == h) & y], p[m0 & (half == h) & ~y]), 4)
                                 if (m0 & (half == h) & y).any() and (m0 & (half == h) & ~y).any() else None}
                    for h in (0, 1)}
        out[name] = {"pairs": int(m0.sum()), "known_joins": int(y[m0].sum()),
                     "auc": round(pair_auc(p[m0 & y], p[m0 & ~y]), 4) if (m0 & y).any() and (m0 & ~y).any() else None,
                     "reliability": rel, "precision_above": prec, "per_half": per_half}
    return out


def hamming64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bit differences between uint64 hashes.

    :param a: uint64 array.
    :param b: uint64 array.
    :returns: int array.
    :rtype: np.ndarray
    """
    x = np.bitwise_xor(a.astype(np.uint64), b.astype(np.uint64))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def band(p: float) -> str:
    """Labelled-precision band.

    :param p: Out-of-half labelled precision.
    :returns: Band label.
    :rtype: str
    """
    return "A (>=0.4)" if p >= 0.4 else "B (0.2-0.4)" if p >= 0.2 else "C (0.1-0.2)" if p >= 0.1 else "D (<0.1)"


# ------------------------------------------------------------------------------------------------- joins
def join_discovery(args: argparse.Namespace, G: Gallery, K: dict, out: Path) -> dict:
    """Scan, calibrate, filter, select and write the join candidates.

    :param args: CLI arguments.
    :param G: Gallery.
    :param K: Output of :func:`known_structure`.
    :param out: Output folder.
    :returns: Join section of the summary.
    :rtype: dict
    """
    variants = [v for v in args.variants.split(",") if v]
    assert args.rank_variant in variants
    partners, positives = K["partners"], K["positives"]
    t0 = time.time()
    sc = scan(G, variants, args.rank_variant, partners, args.csls_k, args.store_k, args.block)
    print(f"scan done in {time.time() - t0:.0f} s", flush=True)
    retrieval = {v: {k: retrieval_report(G, sc["ranks"][v][k], partners) for k in ("cosine", "csls")} for v in variants}
    rv = retrieval[args.rank_variant]
    use_csls = args.csls_k > 0 and rv["csls"]["all"]["R@10"] > rv["cosine"]["all"]["R@10"]
    key = "csls" if use_csls else "cosine"
    w = VARIANTS[args.rank_variant]
    r = sc["r"][args.rank_variant]
    nn_idx, nn_val = sc["nn"][key]
    np.savez(out / "neighbours.npz", ids=np.array(G.ids), idx=nn_idx, val=nn_val.astype(np.float16),
             similarity=np.array(f"{args.rank_variant}/{key}"))

    # (a) labelled precision vs threshold over ALL gallery pairs (histogram counts)
    pos_a = np.array([a for a, b in sorted(positives)], dtype=np.int64)
    pos_b = np.array([b for a, b in sorted(positives)], dtype=np.int64)
    amb = sorted(K["ambiguous"])
    amb_a = np.array([a for a, b in amb], dtype=np.int64)
    amb_b = np.array([b for a, b in amb], dtype=np.int64)
    thr = {}
    for name in ("cosine", key) if key != "cosine" else ("cosine",):
        def sim_of(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Pair similarity on the scale of the current table (cosine or CSLS).

            :param a: Indices.
            :param b: Indices.
            :returns: Similarities.
            :rtype: np.ndarray
            """
            s = G.pair_sims(a, b, w)
            return s if name == "cosine" else 2 * s - r[a] - r[b]
        ps, am = sim_of(pos_a, pos_b), sim_of(amb_a, amb_b)
        pc = G.hold_code[pos_a] != G.hold_code[pos_b]
        ac = G.hold_code[amb_a] != G.hold_code[amb_b]
        thresholds = np.unique(np.round(np.quantile(ps, np.linspace(0.05, 0.95, 19)), 3))
        tab = {}
        for stratum, pm, amask in (("same", ~pc, ~ac), ("cross", pc, ac)):
            rows = []
            for t in thresholds:
                npos, namb = int((ps[pm] >= t).sum()), int((am[amask] >= t).sum())
                nall = sc["hist"][name].at_least(float(t), stratum)
                nneg = max(0, nall - npos - namb)
                rows.append({"t": round(float(t), 3), "pos": npos, "neg": nneg,
                             "precision": round(npos / max(1, npos + nneg), 5),
                             "recall": round(npos / max(1, int(pm.sum())), 4)})
            tab[stratum] = rows
            tab[f"{stratum}_n_pos"] = int(pm.sum())
        thr[name] = tab

    # (b) pool + features
    pool, r_ab, r_ba = build_pool(nn_idx, args.pool_k)
    pa, pb = pool[:, 0], pool[:, 1]
    cos = G.pair_sims(pa, pb, w)
    sim = 2 * cos - r[pa] - r[pb] if use_csls else cos
    cross = (G.hold_code[pa] != G.hold_code[pb]).astype(float)
    F = pair_features(sim, r_ab, r_ba, cross)
    link = K["link"]
    known_link = (link[pa] >= 0) & (link[pa] == link[pb])  # same link group (the catalogue's own links)
    known = K["closure"][pa] == K["closure"][pb]  # ... or linked through a duplicate record of one side
    pos_set = positives
    y = np.array([(a, b) in pos_set for a, b in pool])

    # (b2) duplicates: same shelfmark under two records, same photograph, shared PGP document, container records.
    # Computed BEFORE calibration: these pairs are never proposed, so as calibration negatives (same photograph =
    # top similarity, mutual rank 1) they would only drag the fitted precision of the top band down.
    same_shelf = np.array([G.hold[a] == G.hold[b] and (
        (G.rows[a]["shelf_norm"] and G.rows[a]["shelf_norm"] == G.rows[b]["shelf_norm"])
        or same_record(G.ids[a], G.ids[b], G.hold[a], G.hold[b])) for a, b in pool])
    overlap = {i: set(G.rows[i].get("image_overlap_ids") or ()) for i in range(G.n) if G.rows[i].get("image_overlap_ids")}
    same_file = np.array([G.ids[b] in overlap.get(a, ()) or G.ids[a] in overlap.get(b, ()) for a, b in pool], dtype=bool)
    same_photo_exact = (G.urls[pa] == G.urls[pb]) | (G.sha1[pa] == G.sha1[pb]) | same_file
    dh = hamming64(G.dhash[pa], G.dhash[pb])
    near_photo = (dh <= args.dhash_max) | (cos >= args.dup_cos)
    shared_doc = np.array([bool(set(G.rows[a]["pgpids"]) & set(G.rows[b]["pgpids"])) for a, b in pool])
    duplicate = same_shelf | same_photo_exact | near_photo
    container = G.container[pa] | G.container[pb]  # a box record's first image shows one of its items
    # a known join shown on ONE composite photograph (same file / near-identical image) is not a retrievable pair
    # either, so duplicates and containers leave the fit even when positive; shared-document pairs only when not
    # positive (every PGP positive shares a document by definition)
    not_target = duplicate | container | (shared_doc & ~y)
    fit_rows = (~known | y) & ~not_target
    U = K["closure"]  # split units: link groups joined through duplicate records, else single fragments
    assert np.isfinite(F).all(), "non-finite pair features"
    p_lab, half = calibrate_halves(F, y, fit_rows, U[pa], args.seed)
    assert np.isfinite(p_lab).all(), "non-finite calibrated precision"
    half_fits = list(CALIBRATION_LOG)
    np.savez(out / "pool.npz", pairs=pool.astype(np.int32), F=F.astype(np.float32), y=y, fit_rows=fit_rows, half=half,
             p_labelled=p_lab.astype(np.float32), known=known, duplicate=duplicate, container=container,
             shared_doc=shared_doc)  # for re-calibration / diagnostics without a re-scan
    full_cal = CalibratorV2().fit(F[fit_rows], y[fit_rows].astype(float))  # weights reported only
    held = heldout_report(p_lab, y, fit_rows, cross, half)
    rank_tab = []
    mutual = np.maximum(r_ab, r_ba)
    for lo, hi in ((1, 1), (2, 3), (4, 10), (11, 20), (21, args.store_k + 1)):
        for cname, cm in (("same", cross == 0), ("cross", cross == 1)):
            m = fit_rows & (mutual >= lo) & (mutual <= hi) & cm
            rank_tab.append({"mutual_rank": f"{lo}-{hi}" if hi <= args.store_k else f">={lo} (censored)", "stratum": cname,
                             "pairs": int(m.sum()), "known_joins": int(y[m].sum()),
                             "precision": round(float(y[m].mean()), 4) if m.any() else None})

    # (c) compatibility filter
    attrs = [{k: G.rows[i].get(k) for k in ATTR_KEYS if G.rows[i].get(k)} for i in range(G.n)]
    conf_rate = {}
    neg_rows = np.nonzero(fit_rows & ~y)[0]
    for k in ATTR_KEYS:
        both_pos = [(a, b) for a, b in positives if attrs[a].get(k) and attrs[b].get(k)]
        both_neg = [(pa[x], pb[x]) for x in neg_rows if attrs[pa[x]].get(k) and attrs[pb[x]].get(k)]
        conf_rate[k] = {"known_join_pairs_with_both": len(both_pos),
                        "known_join_conflict_rate": round(float(np.mean([attrs[a][k] != attrs[b][k] for a, b in both_pos])), 4) if both_pos else None,
                        "pool_nonjoin_pairs_with_both": len(both_neg),
                        "pool_nonjoin_conflict_rate": round(float(np.mean([attrs[a][k] != attrs[b][k] for a, b in both_neg])), 4) if both_neg else None}
    filter_keys = [k for k, v in conf_rate.items() if v["known_join_conflict_rate"] is not None
                   and v["known_join_pairs_with_both"] >= 20 and v["known_join_conflict_rate"] <= args.max_attr_conflict]
    compatible = np.array([not attr_conflicts(attrs[a], attrs[b], filter_keys) for a, b in pool])

    pos_cos = G.pair_sims(pos_a, pos_b, w)

    # (e) base rate, selection, expected yield
    n_pos = len(positives)
    rhos = {f"partners_per_fragment={u}": rho_estimate(G.n, n_pos, u) for u in args.partners_per_fragment}
    rho_mid = args.rho if args.rho is not None else rhos[f"partners_per_fragment={args.partners_per_fragment[1]}"]
    rho_lo, rho_hi = rhos[f"partners_per_fragment={args.partners_per_fragment[0]}"], rhos[f"partners_per_fragment={args.partners_per_fragment[2]}"]
    eligible = ~known & compatible & ~duplicate & ~shared_doc & ~container
    picked = pick_with_quota(pool, p_lab, cross, eligible, args.n_cross, args.n_same, args.max_per_fragment)
    p_disc = disc_precision(p_lab, rho_mid)
    elig_rows = np.nonzero(eligible)[0]
    deep = elig_rows[select_candidates(pool[elig_rows], p_lab[elig_rows], args.deep_list, args.deep_max_per_fragment)]

    # (f) hide-one-source validation of the odds transfer
    validation = {}
    srcs = K["pos_by_source"]
    lk = K["links"]
    fjp_links = srcs["fjp_clean"] | lk["fjp_suspect"] | lk["fjp_loose_any"] | lk["fjp_shared_item"]
    for hidden_name, visible, visible_links, hidden in (
            ("hide_pgp", srcs["fjp_clean"] | srcs["fjp_loose"], fjp_links, srcs["pgp"]),
            ("hide_fjp", srcs["pgp"], srcs["pgp"] | lk["pgp_large_or_document_only"], srcs["fjp_clean"] | srcs["fjp_loose"])):
        vf = UnionFind(G.n)
        for a, b in visible_links:
            vf.union(a, b)
        for m in K["dup_members"].values():  # duplicate records are catalogue structure, not a label source
            for x in m[1:]:
                vf.union(m[0], x)
        vg = np.array([vf.find(i) for i in range(G.n)])
        v_known = vg[pa] == vg[pb]
        hidden_new = {(a, b) for a, b in hidden if vg[a] != vg[b]}
        v_pos = np.array([(a, b) in visible for a, b in pool])
        h_pos = np.array([(a, b) in hidden_new for a, b in pool])
        # hidden links that are not hidden positives (large PGP documents, FJP suspects / shared items ...): their
        # status cannot be scored either way, so they are not selectable (they would count as misses otherwise)
        hidden_amb = known & ~v_known & ~h_pos
        if v_pos.sum() < 20 or not hidden_new:
            validation[hidden_name] = {"skipped": "too few visible or hidden pairs"}
            continue
        # duplicates / containers are not calibration negatives here either; the shared-PGP-document pairs stay
        # (with PGP hidden, sharing a PGP document IS the hidden label)
        v_fit = (~v_known | v_pos) & ~(duplicate | container)
        v_p, _ = calibrate_halves(F, v_pos, v_fit, vg[pa], args.seed)
        rho_h = len(hidden_new) / max(1, len(visible))
        v_pick = pick_with_quota(pool, v_p, cross, ~v_known & compatible & ~duplicate & ~container & ~hidden_amb,
                                 args.n_cross, args.n_same, args.max_per_fragment)
        validation[hidden_name] = {"visible_pairs": len(visible), "hidden_pairs": len(hidden_new), "rho_hidden": round(rho_h, 4),
                                   "hidden_pairs_in_pool": int(h_pos.sum()), "n_selected": int(len(v_pick)),
                                   "hidden_link_only_pairs_unselectable": int(hidden_amb.sum()),
                                   "hidden_joins_found": int(h_pos[v_pick].sum()),
                                   "predicted_hidden_joins": round(float(disc_precision(v_p[v_pick], rho_h).sum()), 2),
                                   "selected_cross_holding": int((cross[v_pick] == 1).sum()),
                                   "found_cross_holding": int((h_pos[v_pick] & (cross[v_pick] == 1)).sum()),
                                   "predicted_cross_holding": round(float(disc_precision(v_p[v_pick][cross[v_pick] == 1], rho_h).sum()), 2)}
    vt = [v for v in validation.values() if "predicted_hidden_joins" in v]
    transfer = sum(v["hidden_joins_found"] for v in vt) / max(1e-9, sum(v["predicted_hidden_joins"] for v in vt)) if vt else None

    # (g) outputs
    enc_cos = {b: G.pair_sims(pa, pb, {b: 1.0}) for b in ("merger", "dino_masked", "dino_mpatches")}

    def row_for(x: int) -> dict:
        """CSV row for pool pair x.

        :param x: Pool row.
        :returns: Row dict.
        :rtype: dict
        """
        a, b = pool[x]
        ra, rb = G.rows[a], G.rows[b]
        d = {"pair": "", "id_a": G.ids[a], "id_b": G.ids[b], "shelfmark_a": ra["shelfmark"], "shelfmark_b": rb["shelfmark"],
             "holding_a": G.hold[a], "holding_b": G.hold[b], "cross_holding": int(cross[x]),
             "similarity": round(float(sim[x]), 4), "similarity_kind": f"{args.rank_variant}/{key}",
             "cosine_fused": round(float(cos[x]), 4),
             **{f"cos_{bk}": round(float(v[x]), 4) for bk, v in enc_cos.items()},
             "rank_a_to_b": int(r_ab[x]), "rank_b_to_a": int(r_ba[x]),
             "p_labelled": round(float(p_lab[x]), 4), "band_labelled": band(float(p_lab[x])),
             "p_discovery": round(float(disc_precision(p_lab[x:x + 1], rho_mid)[0]), 5),
             "p_discovery_low": round(float(disc_precision(p_lab[x:x + 1], rho_lo)[0]), 5),
             "p_discovery_high": round(float(disc_precision(p_lab[x:x + 1], rho_hi)[0]), 5),
             "same_box_or_volume": int(shelf_neighbours(G.ids[a], G.ids[b])), "dhash_bits": int(dh[x]),
             "tower_training_ms": int(bool(G.tower_train[a]) or bool(G.tower_train[b])),
             "tower_ms_split_a": ra.get("tower_ms_split") or "", "tower_ms_split_b": rb.get("tower_ms_split") or "",
             "in_benchmark_a": int(bool(ra.get("in_benchmark"))), "in_benchmark_b": int(bool(rb.get("in_benchmark"))),
             "mask_ok_a": int(G.mask_ok[a]), "mask_ok_b": int(G.mask_ok[b]),
             "image_a": ra["image_url"], "image_b": rb["image_url"], "image2_a": ra.get("image_url_2") or "",
             "image2_b": rb.get("image_url_2") or ""}
        for k in ATTR_KEYS:
            d[f"{k}_a"], d[f"{k}_b"] = ra.get(k) or "", rb.get(k) or ""
        d["pgp_types_a"], d["pgp_types_b"] = ";".join(ra.get("pgp_types", [])), ";".join(rb.get("pgp_types", []))
        return d

    fields = list(row_for(0).keys())
    cands = []
    for k, x in enumerate(picked, 1):
        d = row_for(int(x))
        d["pair"] = f"J{k:04d}"
        cands.append(d)
    write_csv(out / "join_candidates.csv", cands, fields)
    deep_rows = []
    for k, x in enumerate(deep, 1):
        d = row_for(int(x))
        d["pair"] = f"R{k:05d}"
        deep_rows.append(d)
    write_csv(out / "join_candidates_ranked.csv", deep_rows, fields)
    dup_rows = []
    for x in np.nonzero(duplicate & ~known_link)[0]:
        reason = "same_shelfmark_record" if same_shelf[x] else "same_image_file" if same_photo_exact[x] else \
            "near_identical_photo"
        dup_rows.append(dict(row_for(int(x)), pair=reason))
    write_csv(out / "possible_duplicate_records.csv", dup_rows, fields)
    write_csv(out / "calibration_heldout.csv",
              [dict(stratum=s, **row) for s in ("all", "same_holding", "cross_holding") for row in held[s]["reliability"]],
              ["stratum", "p_band", "pairs", "known_joins", "mean_predicted", "observed", "observed_95ci"])

    sel_p = p_lab[picked]
    depth = {}
    for d in (50, 100, 200, 500, 1000, 2000, len(deep)):
        if d <= len(deep):
            pp = p_lab[deep[:d]]
            depth[str(d)] = {"expected_real_joins_central": round(float(disc_precision(pp, rho_mid).sum()), 2),
                             "low": round(float(disc_precision(pp, rho_lo).sum()), 2),
                             "high": round(float(disc_precision(pp, rho_hi).sum()), 2),
                             "cross_holding_pairs": int((cross[deep[:d]] == 1).sum()),
                             "mean_p_labelled": round(float(pp.mean()), 4)}
    return {
        "ranking_variant": args.rank_variant, "ranking_similarity": key, "n_gallery": G.n,
        "retrieval_known_joins": retrieval,
        "known": {"positives_by_source": {k: len(v) for k, v in srcs.items()}, "positives": n_pos,
                  "positives_cross_holding": int(sum(G.hold_code[a] != G.hold_code[b] for a, b in positives)),
                  "ambiguous_grouped_pairs": len(K["ambiguous"]), "fragments_with_known_partner": len(partners)},
        "calibration": {"threshold_all_pairs": thr, "precision_vs_mutual_rank": rank_tab, "pool_k": args.pool_k,
                        "store_k": args.store_k, "pool_pairs": int(len(pool)), "pool_known_joins": int(y.sum()),
                        "pool_known_joins_cross_holding": int((y & (cross == 1)).sum()),
                        "fit_rows": "pool pairs not known (closure) or a positive, minus duplicates (same record / same "
                                    "or near-identical photograph), container pairs and non-positive shared-PGP-document "
                                    "pairs: never proposed, so not the population the precision is predicted for",
                        "positives_reachable_in_pool": round(float(y.sum() / max(1, n_pos)), 4),
                        "split": "two halves of link groups (closed over duplicate records) / single fragments (seeded); "
                                 "a pair belongs to its first fragment's half; each half scored by the model fitted on "
                                 "the other",
                        "heldout": held, "half_fits": half_fits,
                        "logistic_weights_full_fit_[sim,logminrank,logmaxrank,cross,sim*cross,intercept]": [round(float(v), 4) for v in full_cal.w]},
        "attribute_filter": {"conflict_rates": conf_rate, "applied_keys": filter_keys, "max_known_join_conflict": args.max_attr_conflict,
                             "pool_pairs_removed": int((~compatible & ~known).sum())},
        "duplicates": {"same_shelfmark_record_pairs": int((same_shelf & ~known_link).sum()),
                       "same_photo_exact_pairs": int((same_photo_exact & ~known_link).sum()),
                       "same_image_file_via_record_lists_pairs": int((same_file & ~known_link).sum()),
                       "container_pairs_in_pool": int(container.sum()),
                       "pool_pairs_linked_only_through_a_duplicate_record": int((known & ~known_link).sum()),
                       "non_target_pairs_left_out_of_calibration": int((not_target & (~known | y)).sum()),
                       "positives_left_out_of_calibration_same_photo_or_container": int((not_target & y).sum()),
                       "near_identical_photo_pairs": int((near_photo & ~same_photo_exact & ~known_link).sum()),
                       "dhash_max_bits": args.dhash_max, "dup_cosine": args.dup_cos,
                       "shared_pgp_document_unlinked_pairs": int((shared_doc & ~known).sum()),
                       "known_join_pairs_at_or_above_dup_cosine": int((pos_cos >= args.dup_cos).sum()),
                       "known_join_cosine_quantiles_50_90_99_max": [round(float(q), 4) for q in np.quantile(pos_cos, [0.5, 0.9, 0.99, 1.0])]},
        "base_rate": {"genizah_size_assumed": GENIZAH_SIZE, "rho_by_assumption": {k: round(v, 4) for k, v in rhos.items()},
                      "rho_used_central": round(rho_mid, 4), "rho_overridden": args.rho is not None},
        "validation_hide_one_source": validation,
        "validation_found_over_predicted": round(transfer, 3) if transfer is not None else None,
        "candidates": {"n": len(cands), "cross_holding": int(sum(c["cross_holding"] for c in cands)),
                       "quota_cross_same": [args.n_cross, args.n_same], "max_per_fragment": args.max_per_fragment,
                       "eligible_pool_pairs": int(eligible.sum()),
                       "bands_labelled": dict(sorted(Counter(c["band_labelled"] for c in cands).items())),
                       "bands_labelled_cross": dict(sorted(Counter(c["band_labelled"] for c in cands if c["cross_holding"]).items())),
                       "mean_p_labelled": round(float(sel_p.mean()), 4) if len(cands) else None,
                       "expected_real_joins": {"central": round(float(disc_precision(sel_p, rho_mid).sum()), 2),
                                               "low": round(float(disc_precision(sel_p, rho_lo).sum()), 2),
                                               "high": round(float(disc_precision(sel_p, rho_hi).sum()), 2)},
                       "expected_real_cross_vs_same_central": [round(float(disc_precision(sel_p[cross[picked] == 1], rho_mid).sum()), 2),
                                                               round(float(disc_precision(sel_p[cross[picked] == 0], rho_mid).sum()), 2)],
                       "expected_x_validation_ratio": round(float(disc_precision(sel_p, rho_mid).sum()) * transfer, 2) if transfer is not None else None,
                       "touching_tower_training_ms": int(sum(c["tower_training_ms"] for c in cands)),
                       "touching_tower_val_ms": int(sum("val" in (c["tower_ms_split_a"], c["tower_ms_split_b"]) for c in cands)),
                       "same_box_or_volume": int(sum(c["same_box_or_volume"] for c in cands))},
        "ranked_list": {"n": len(deep_rows), "by_depth": depth},
    }


# ------------------------------------------------------------------------------------------------- scribes
def build_decoys(L: List[int], lab: np.ndarray, U: List[int], hold: np.ndarray, k: int, seed: int
                 ) -> Tuple[List[List[int]], List[int]]:
    """Decoy scribes: for every real scribe, ``k`` sets of unlabelled documentary fragments of its size and holding mix.

    A decoy centroid has the photography/holding component and the size of the real one but no common hand, so an
    unlabelled document by an unknown hand is as likely to land on it as on a wrong real scribe.

    :param L: Labelled gallery indices.
    :param lab: Real scribe index per labelled fragment.
    :param U: Unlabelled documentary gallery indices.
    :param hold: Holding per gallery index.
    :param k: Decoy sets per real scribe.
    :param seed: Sampling seed.
    :returns: (decoy member lists, mirrored real scribe per decoy).
    :rtype: Tuple[List[List[int]], List[int]]
    """
    rng = np.random.default_rng(seed)
    by_h: Dict[str, List[int]] = defaultdict(list)
    for i in rng.permutation(U):
        by_h[hold[i]].append(int(i))
    spare = [int(i) for i in rng.permutation(U)]
    used: Set[int] = set()
    decoys, mirror = [], []
    for _ in range(k):
        for sc in range(int(lab.max()) + 1):
            members: List[int] = []
            need = Counter(hold[L[j]] for j in np.nonzero(lab == sc)[0])
            for h, c in sorted(need.items()):
                for pool in (by_h.get(h, []), spare):  # same holding first, any unlabelled fragment when exhausted
                    while c and pool:
                        i = pool.pop()
                        if i not in used:
                            used.add(i)
                            members.append(i)
                            c -= 1
            decoys.append(members)
            mirror.append(sc)
    return decoys, mirror


def compete(Xu: np.ndarray, u_member: np.ndarray, Xl: np.ndarray, lab: np.ndarray, n_real: int, Sd: np.ndarray
            ) -> Dict[str, np.ndarray]:
    """Target-decoy competition: every unlabelled document against the real and the decoy centroids.

    :param Xu: Unlabelled vectors.
    :param u_member: Decoy index each unlabelled document belongs to (-1 = none); its own decoy centroid is
        recomputed without it.
    :param Xl: Labelled vectors.
    :param lab: Real scribe per labelled vector.
    :param n_real: Number of real scribes.
    :param Sd: (n_decoys, D) sums of the decoy members' vectors.
    :returns: {best (centroid index; < n_real = a real scribe), top1, margin (best - second best), is_target}.
    :rtype: Dict[str, np.ndarray]
    """
    onehot = np.zeros((len(Xl), n_real), dtype=np.float32)
    onehot[np.arange(len(Xl)), lab] = 1
    Cr = onehot.T @ Xl
    Cr /= np.linalg.norm(Cr, axis=1, keepdims=True)
    Cd = Sd / np.linalg.norm(Sd, axis=1, keepdims=True)
    S = np.concatenate([Xu @ Cr.T, Xu @ Cd.T], axis=1)
    for q in np.nonzero(u_member >= 0)[0]:
        d = int(u_member[q])
        rest = Sd[d] - Xu[q]
        S[q, n_real + d] = float(Xu[q] @ rest / max(np.linalg.norm(rest), 1e-12))
    srt = np.sort(S, axis=1)
    best = S.argmax(1)
    return {"best": best, "top1": srt[:, -1], "margin": srt[:, -1] - srt[:, -2], "is_target": best < n_real}


def decoy_rule(score: np.ndarray, is_target: np.ndarray, ratio: float, target: float, min_n: int) -> dict:
    """Threshold from target-decoy competition: estimated precision ``1 - (decoys + 1) * ratio / targets`` above it.

    The precision of the set of target hits scoring >= t is made monotone like a q-value (the best precision of any
    larger set), so a candidate's own estimate never exceeds that of a better-scoring one.

    :param score: Score per unlabelled document (``top1`` or ``margin``).
    :param is_target: Whether its best centroid is a real scribe.
    :param ratio: Real / decoy centroid count (1 / decoy sets).
    :param target: Precision target.
    :param min_n: Minimum accepted target hits.
    :returns: {threshold (None if unreachable), targets_above, decoys_above, precision_estimate, expected_correct,
        local_precision_bands, _prec (monotone precision per document, aligned with ``score``)}.
    :rtype: dict
    """
    order = np.argsort(-score, kind="stable")
    t_cum = np.cumsum(is_target[order])
    d_cum = np.cumsum(~is_target[order])
    prec = np.clip(1 - (d_cum + 1) * ratio / np.maximum(t_cum, 1), 0, 1)
    prec_mono = np.maximum.accumulate(prec[::-1])[::-1]
    per_doc = np.empty(len(score))
    per_doc[order] = prec_mono
    bands = []
    for b0 in range(0, min(len(order), 4000), 250):  # local estimate in bands of 250 documents by score
        sl = order[b0:b0 + 250]
        nt, nd = int(is_target[sl].sum()), int((~is_target[sl]).sum())
        bands.append({"rank_from": b0 + 1, "score_low": round(float(score[sl].min()), 4), "targets": nt, "decoys": nd,
                      "local_precision": round(max(0.0, 1 - nd * ratio / max(nt, 1)), 4)})
    ok = (prec >= target) & (t_cum >= min_n)
    out = {"local_precision_bands": bands, "_prec": per_doc}
    if not ok.any():
        j = int(np.argmax(np.where(t_cum >= min_n, prec, -1)))
        out.update(threshold=None, best_reachable_precision=round(float(prec[j]), 4), expected_correct=0.0)
        return out
    j = int(np.argmax(np.where(ok, t_cum * prec, -1)))
    out.update(threshold=round(float(score[order[j]]), 5), targets_above=int(t_cum[j]), decoys_above=int(d_cum[j]),
               precision_estimate=round(float(prec[j]), 4), expected_correct=round(float(t_cum[j] * prec[j]), 1))
    return out


def scribe_discovery(args: argparse.Namespace, G: Gallery, K: dict, variant: str, write: bool, out: Path) -> dict:
    """Scribe attribution of unlabelled PGP documentary fragments, with target-decoy precision estimates.

    LOO nearest-centroid accuracy over the PGP scribes with >= ``min_per_scribe`` labelled fragments (same-document
    fragments never inform a query) measures the encoder; unlabelled documents then compete for the real centroids
    and ``decoys`` sets of decoy centroids (:func:`build_decoys`), and the attribution threshold is the one with the
    most expected correct attributions at estimated precision >= ``scribe_precision`` (:func:`decoy_rule`). The PGP
    date check is reported as an independent estimate; v1's closed-set rule is reported for reference.

    :param args: CLI arguments.
    :param G: Gallery.
    :param K: Known structure.
    :param variant: Feature variant.
    :param write: Write the candidate CSVs.
    :param out: Output folder.
    :returns: Scribe section.
    :rtype: dict
    """
    rows = G.rows
    lab_all = {i: r["pgp_scribe"] for i, r in enumerate(rows) if r.get("pgp_scribe")}
    counts = Counter(lab_all.values())
    L = [i for i in sorted(lab_all) if counts[lab_all[i]] >= args.min_per_scribe]
    names = sorted({lab_all[i] for i in L})
    sidx = {s: k for k, s in enumerate(names)}
    lab = np.array([sidx[lab_all[i]] for i in L])
    labelled = set(lab_all)
    lab_units = {int(K["closure"][i]) for i in labelled}  # link groups + duplicate records of labelled fragments
    all_lab_docs = {d for i in labelled for d in rows[i]["pgpids"]}
    U_all = [i for i, r in enumerate(rows) if i not in labelled and r.get("documentary") and not G.container[i]]
    U = [i for i in U_all if int(K["closure"][i]) not in lab_units and not (set(rows[i]["pgpids"]) & all_lab_docs)]
    lab_docs = {d for i in L for d in rows[i]["pgpids"]}
    lab_l_units = {int(K["closure"][i]) for i in L}
    R = [i for i, s in lab_all.items() if counts[s] < args.min_per_scribe and not (set(rows[i]["pgpids"]) & lab_docs)
         and int(K["closure"][i]) not in lab_l_units and not G.container[i]]
    needed = sorted(set(L) | set(U) | set(R))
    loc = {i: k for k, i in enumerate(needed)}
    Xs = G.variant_matrix(variant, needed)  # only the rows used: the full fused/pooled matrix would be ~2 GB

    def X(idx: Sequence[int]) -> np.ndarray:
        """Variant vectors of gallery indices.

        :param idx: Gallery indices.
        :returns: (len(idx), D) float32.
        :rtype: np.ndarray
        """
        return Xs[[loc[i] for i in idx]]

    Xl = X(L)
    by_doc: Dict[str, List[int]] = defaultdict(list)
    for li, i in enumerate(L):
        for d in rows[i]["pgpids"]:
            by_doc[d].append(li)
    same_doc = np.eye(len(L), dtype=bool)
    for m in by_doc.values():
        same_doc[np.ix_(m, m)] = True
    doc_uf = UnionFind(len(L))
    for a, b in zip(*np.nonzero(same_doc)):
        doc_uf.union(int(a), int(b))
    doc_unit = np.array([doc_uf.find(i) for i in range(len(L))])  # CV folds keep one document's fragments together
    years: Dict[str, List[int]] = defaultdict(list)
    dated: Counter = Counter()
    for i, s in lab_all.items():
        if rows[i]["pgp_years"]:
            years[s] += rows[i]["pgp_years"]
            dated[s] += 1
    windows = {s: (min(ys) - args.date_slack, max(ys) + args.date_slack) for s, ys in years.items() if dated[s] >= 2}
    isolated = {s for s, w in windows.items() if s in sidx and
                not any(o != s and o in sidx and w[0] <= v[1] and w[1] >= v[0] for o, v in windows.items())}
    meta = lambda i: {"years": rows[i]["pgp_years"], "pgp_tags": rows[i]["pgp_tags"]}  # noqa: E731

    # closed set: leave-one-out nearest centroid among the labelled fragments
    loo_scores = centroid_scores(Xl, Xl, lab, names, same_doc)
    srt = np.sort(loo_scores, axis=1)
    loo = {"pred": loo_scores.argmax(1), "top1": srt[:, -1], "margin": srt[:, -1] - srt[:, -2]}
    correct = loo["pred"] == lab
    per_scribe = {names[k]: {"n": int((lab == k).sum()), "acc": round(float(correct[lab == k].mean()), 4),
                             "window": windows.get(names[k]),
                             "top_holding_share": round(Counter(G.hold[L[x]] for x in np.nonzero(lab == k)[0]).most_common(1)[0][1] / max(1, (lab == k).sum()), 3)}
                  for k in range(len(names))}
    sub = Xl @ Xl.T
    sub[same_doc] = -np.inf
    nn1 = float((lab[sub.argmax(1)] == lab).mean())
    loo_dates = [date_intervals(meta(i)) for i in L]
    wrong_rows = [x for x in np.nonzero(~correct)[0] if loo_dates[x] and names[loo["pred"][x]] in windows]
    wrong_dated = [date_status(loo_dates[x], windows[names[loo["pred"][x]]]) for x in wrong_rows]
    wrong_q = [chance_compatible(loo_dates[x], windows, {names[loo["pred"][x]], names[lab[x]]}) for x in wrong_rows]
    loo_inc = float(np.mean([s == "incompatible" for s in wrong_dated])) if wrong_dated else None
    loo_inc_random = float(1 - np.mean([q for q in wrong_q if q is not None])) if any(q is not None for q in wrong_q) else None
    kappa = min(1.0, loo_inc / loo_inc_random) if loo_inc and loo_inc_random else 1.0

    # open set: unlabelled documents compete for real and decoy centroids
    decoys, mirror = build_decoys(L, lab, U, G.hold, args.decoys, args.seed)
    Sd = np.stack([X(m).sum(0) for m in decoys])
    member = np.full(len(U), -1)
    upos = {i: k for k, i in enumerate(U)}
    for d, m in enumerate(decoys):
        for i in m:
            member[upos[i]] = d
    comp = compete(X(U), member, Xl, lab, len(names), Sd)
    ratio = len(names) / len(decoys)
    u_dates = [date_intervals(meta(i)) for i in U]
    rare = compete(X(R), np.full(len(R), -1), Xl, lab, len(names), Sd) if R else None

    rules = {}
    for score_name in ("top1", "margin"):
        rule = decoy_rule(comp[score_name], comp["is_target"], ratio, args.scribe_precision, args.scribe_min_n)
        t_v1 = precision_threshold(loo[score_name], correct, args.scribe_precision, args.scribe_min_n)
        rule["closed_set_v1_rule"] = {
            "threshold": round(t_v1, 4) if t_v1 is not None else None,
            "unlabelled_target_hits_above": int((comp["is_target"] & (comp[score_name] >= t_v1)).sum()) if t_v1 is not None else None,
            "note": "v1 picked the LOO-precision threshold; it ignores that most unlabelled documents are in other hands"}
        t = rule["threshold"]
        if t is None:
            rules[score_name] = rule
            continue
        above = comp["is_target"] & (comp[score_name] >= t)
        rule["loo_coverage_at_threshold"] = round(float((loo[score_name] >= t).mean()), 4)
        rule["loo_precision_at_threshold"] = round(float(correct[loo[score_name] >= t].mean()), 4) if (loo[score_name] >= t).any() else None
        rule["nested_cv_closed_set"] = cv_rule_precision(loo[score_name], correct, doc_unit, args.scribe_precision,
                                                         args.scribe_min_n, args.cv_folds)
        if rare is not None:
            rule["rare_scribe_fragments_accepted"] = f"{int((rare['is_target'] & (rare[score_name] >= t)).sum())}/{len(R)} " \
                                                     "(their scribe has no centroid: every acceptance is wrong)"
        st = [date_status(u_dates[u], windows.get(names[comp["best"][u]])) for u in np.nonzero(above)[0]]
        dated_st = [x for x in st if x != "unknown"]
        p_inc = float(np.mean([x == "incompatible" for x in dated_st])) if dated_st else None
        q_chance = [chance_compatible(u_dates[u], windows, {names[comp["best"][u]]}) for u in np.nonzero(above)[0]
                    if date_status(u_dates[u], windows.get(names[comp["best"][u]])) != "unknown"]
        q = float(np.mean([x for x in q_chance if x is not None])) if any(x is not None for x in q_chance) else None
        n_kept = sum(1 for x in st if x != "incompatible")
        n_compat = sum(1 for x in st if x == "compatible")
        err = exp_date = exp_ci = None
        if p_inc is not None and q is not None:
            err, exp_date = date_expected_correct(p_inc, q, kappa, n_compat, n_kept)
            lo_p, hi_p = wilson(sum(x == "incompatible" for x in dated_st), len(dated_st))
            exp_ci = (round(date_expected_correct(hi_p, q, kappa, n_compat, n_kept)[1], 1),
                      round(date_expected_correct(lo_p, q, kappa, n_compat, n_kept)[1], 1))
        rule.update({
            "unlabelled_documentary": len(U), "unlabelled_above_threshold": int(above.sum()),
            "estimate_dates": {"above_dated": len(dated_st), "date_incompatible": int(sum(x == "incompatible" for x in dated_st)),
                               "chance_wrong_scribe_compatible_q": round(q, 4) if q is not None else None,
                               "kappa_from_loo": round(kappa, 3), "error_rate": round(err, 4) if err is not None else None,
                               "kept_after_date_filter": n_kept,
                               "expected_correct_after_date_filter": round(exp_date, 1) if exp_date is not None else None,
                               "expected_correct_95ci_from_dated_sample": exp_ci},
            "above_by_scribe": dict(Counter(names[k] for k in comp["best"][above]).most_common())})
        rules[score_name] = rule
    usable = {k: v for k, v in rules.items() if v.get("threshold") is not None}
    chosen = max(usable, key=lambda k: usable[k]["expected_correct"]) if usable else None
    res = {"features": variant, "n_labelled": len(L), "n_scribes": len(names),
           "loo_nearest_centroid_acc": round(float(correct.mean()), 4),
           "loo_balanced_acc": round(float(np.mean([v["acc"] for v in per_scribe.values()])), 4),
           "loo_1nn_acc_reference": round(nn1, 4), "chance_majority": round(float(Counter(lab).most_common(1)[0][1] / len(lab)), 4),
           "unlabelled_documentary_all": len(U_all), "unlabelled_excluded_linked_to_labelled": len(U_all) - len(U),
           "decoys": {"sets_per_scribe": args.decoys, "decoy_centroids": len(decoys),
                      "fragments_used": int(sum(len(m) for m in decoys)),
                      "unlabelled_best_match_decoy": int((~comp["is_target"]).sum()),
                      "unlabelled_best_match_real": int(comp["is_target"].sum())},
           "loo_wrong_predictions_dated": len(wrong_dated),
           "loo_wrong_predictions_date_incompatible_rate": round(loo_inc, 4) if loo_inc is not None else None,
           "kappa": round(kappa, 3), "era_isolated_scribes": sorted(isolated), "rules": rules, "chosen_score": chosen,
           "per_scribe": per_scribe}
    for rule in rules.values():
        rule["_prec"] = None  # per-document estimates stay local (not JSON)
    if not write or chosen is None:
        return res
    t = rules[chosen]["threshold"]
    prec_doc = decoy_rule(comp[chosen], comp["is_target"], ratio, args.scribe_precision, args.scribe_min_n)["_prec"]
    keep = np.nonzero(comp["is_target"] & (comp[chosen] >= t))[0]
    keep = keep[np.argsort(-comp[chosen][keep])]
    lab_rows: Dict[int, List[int]] = defaultdict(list)
    for li, k in enumerate(lab):
        lab_rows[int(k)].append(li)
    cands, rejected = [], []
    for u in keep:
        i = U[u]
        s_k = int(comp["best"][u])
        lr = lab_rows[s_k]
        sims = Xl[lr] @ X([i])[0]
        ex = [L[lr[j]] for j in np.argsort(-sims)[:3]]
        r = rows[i]
        status = date_status(date_intervals(meta(i)), windows.get(names[s_k]))
        row = {"cand": "", "id": G.ids[i], "shelfmark": r["shelfmark"], "holding": G.hold[i], "proposed_scribe": names[s_k],
               "score_top1": round(float(comp["top1"][u]), 4), "score_margin": round(float(comp["margin"][u]), 4),
               "rule": chosen, "threshold": t, "precision_estimate_rule": rules[chosen]["precision_estimate"],
               "precision_of_set_down_to_this_score": round(float(prec_doc[u]), 4),
               "pgp_dates": ";".join(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in date_intervals(meta(i))),
               "scribe_window": "-".join(map(str, windows[names[s_k]])) if names[s_k] in windows else "",
               "date_check": status, "era_isolated_scribe": int(names[s_k] in isolated),
               "scribe_exemplars_same_holding_share": round(sum(G.hold[L[x]] == G.hold[i] for x in lr) / len(lr), 3),
               "tower_training_ms": int(bool(G.tower_train[i])), "tower_ms_split": r.get("tower_ms_split") or "",
               "pgp_ids": ";".join(r["pgpids"]), "pgp_types": ";".join(r["pgp_types"]), "pgp_tags": ";".join(r["pgp_tags"])[:200],
               "image": r["image_url"], "exemplar_1": G.ids[ex[0]] if ex else "", "exemplar_2": G.ids[ex[1]] if len(ex) > 1 else "",
               "exemplar_3": G.ids[ex[2]] if len(ex) > 2 else ""}
        if status == "incompatible":
            rejected.append(row)
            continue
        row["cand"] = f"S{len(cands) + 1:04d}"
        cands.append(row)
    fields = list((cands or rejected or [{"cand": ""}])[0].keys())
    write_csv(out / "scribe_candidates.csv", cands, fields)
    write_csv(out / "scribe_candidates_rejected_by_date.csv", rejected, fields)
    res["candidates"] = {"n": len(cands), "rejected_by_date": len(rejected),
                         "date_check": dict(Counter(c["date_check"] for c in cands)),
                         "to_era_isolated_scribes": int(sum(c["era_isolated_scribe"] for c in cands)),
                         "by_scribe": dict(Counter(c["proposed_scribe"] for c in cands).most_common()),
                         "expected_correct_decoy_estimate": round(rules[chosen]["precision_estimate"] * len(cands), 1),
                         "note": "precision_estimate x listed; the date filter removed likely-wrong ones, so this is "
                                 "conservative. precision_of_set_down_to_this_score is q-value-like (the best precision "
                                 "of any set extending at least to that score), not a per-candidate probability"}
    return res


# ------------------------------------------------------------------------------------------------- main
def plain_statement(s: dict) -> str:
    """Short plain-language reading of the yields.

    :param s: Summary.
    :returns: Text.
    :rtype: str
    """
    parts = []
    j = s.get("joins")
    if j:
        c, rv = j["candidates"], j["retrieval_known_joins"][j["ranking_variant"]][j["ranking_similarity"]]["all"]
        h = j["calibration"]["heldout"]["all"]
        parts.append(
            f"JOINS over {j['n_gallery']:,} fragments: known joins retrieved R@10 {rv.get('R@10')} (median rank "
            f"{rv.get('median_rank')}) with {j['ranking_variant']}/{j['ranking_similarity']}. Held-out calibration (other half "
            f"of the join groups) AUC {h['auc']}. {c['n']} candidates ({c['cross_holding']} cross-holding); expected real new "
            f"joins {c['expected_real_joins']['central']} (bracket {c['expected_real_joins']['low']}-"
            f"{c['expected_real_joins']['high']} for {', '.join(j['base_rate']['rho_by_assumption'])}); hide-one-source "
            f"found/predicted = {j['validation_found_over_predicted']}.")
    sc = s.get("scribes", {}).get("used")
    if sc and sc.get("chosen_score"):
        r = sc["rules"][sc["chosen_score"]]
        parts.append(
            f"SCRIBES ({sc['features']}): LOO centroid accuracy {sc['loo_nearest_centroid_acc']} over {sc['n_scribes']} scribes; "
            f"target-decoy rule {sc['chosen_score']} >= {r['threshold']}: {r['unlabelled_above_threshold']} of "
            f"{r['unlabelled_documentary']} unlabelled documentary fragments attributed, estimated precision "
            f"{r['precision_estimate']} ({r['decoys_above']} decoy hits above the threshold), about {r['expected_correct']} "
            f"correct; the PGP-date check gives {r['estimate_dates']['expected_correct_after_date_filter']}.")
    return " ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    """Load, discover joins and scribe attributions, write the summary.

    :param argv: CLI arguments (None = sys.argv).
    :returns: Summary dict.
    :rtype: dict
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="download the shards from the HF dataset first")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--shards-dir", type=Path, default=HF_CACHE / "shards")
    parser.add_argument("--manifest", type=Path, default=GALLERY / "manifest.jsonl")
    parser.add_argument("--known-joins", type=Path, default=GALLERY / "known_joins.json")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--variants", default="fused,merger,pooled,dinov2,head")
    parser.add_argument("--rank-variant", default="fused")
    parser.add_argument("--csls-k", type=int, default=10, help="0 = plain cosine")
    parser.add_argument("--store-k", type=int, default=50)
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--n-cross", type=int, default=200)
    parser.add_argument("--n-same", type=int, default=200)
    parser.add_argument("--max-per-fragment", type=int, default=2)
    parser.add_argument("--deep-list", type=int, default=5000, help="rows of join_candidates_ranked.csv")
    parser.add_argument("--deep-max-per-fragment", type=int, default=5, help="cap per fragment in the ranked list")
    parser.add_argument("--max-attr-conflict", type=float, default=0.10)
    parser.add_argument("--dup-cos", type=float, default=0.995)
    parser.add_argument("--dhash-max", type=int, default=4)
    parser.add_argument("--partners-per-fragment", type=float, nargs=3, default=[0.1, 0.3, 1.0])
    parser.add_argument("--rho", type=float, default=None, help="override the central undiscovered/known ratio")
    parser.add_argument("--no-loose-positive", action="store_true")
    parser.add_argument("--scribe-variants", default="merger,pooled,fused,head")
    parser.add_argument("--min-per-scribe", type=int, default=5)
    parser.add_argument("--decoys", type=int, default=2, help="decoy centroid sets per real scribe")
    parser.add_argument("--scribe-precision", type=float, default=0.8)
    parser.add_argument("--scribe-min-n", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--date-slack", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-joins", action="store_true")
    parser.add_argument("--skip-scribes", action="store_true")
    args = parser.parse_args(argv)
    np.seterr(divide="ignore", over="ignore", invalid="ignore")  # macOS Accelerate raises spurious float32 matmul
    # flags (values are finite); the arrays that matter are checked with np.isfinite below
    t0 = time.time()
    if args.fetch:
        args.shards_dir = fetch_gallery(args.repo, HF_CACHE) / "shards"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    known = json.loads(args.known_joins.read_text())
    G = Gallery(args.shards_dir, manifest)
    K = known_structure(G, known, not args.no_loose_positive)
    print(f"gallery {G.n:,} of {G.n_manifest:,} manifest fragments ({G.n_unknown} unknown ids, {G.n_duplicate} duplicates "
          f"in shards); {len(K['positives'])} known join pairs inside", flush=True)
    summary: dict = {"gallery": {"fragments_with_features": G.n, "manifest_fragments": G.n_manifest,
                                 "coverage": round(G.n / max(1, G.n_manifest), 5),
                                 "manifest_fragments_with_live_image": G.n_manifest_live,
                                 "coverage_of_live": round(G.n / max(1, G.n_manifest_live), 5),
                                 "container_records": int(G.container.sum()), "mask_fallback": int((~G.mask_ok).sum()),
                                 "shards_dir": str(args.shards_dir), "unknown_ids_in_shards": G.n_unknown,
                                 "duplicate_ids_in_shards": G.n_duplicate},
                     "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    if not args.skip_joins:
        summary["joins"] = join_discovery(args, G, K, args.out)
        print("joins", json.dumps(summary["joins"]["candidates"]), flush=True)
    if not args.skip_scribes:
        runs = [scribe_discovery(args, G, K, v, False, args.out) for v in args.scribe_variants.split(",") if v]
        # chosen by LOO accuracy, NOT by its own estimated yield: picking the variant whose estimate looks best selects
        # the most over-estimated one (on synthetic data a weak vector won with prior 0.29 vs a true 0.02)
        best = max(runs, key=lambda r: r["loo_nearest_centroid_acc"])
        summary["scribes"] = {"compared": [{k: r[k] for k in ("features", "loo_nearest_centroid_acc", "loo_balanced_acc",
                                                               "loo_1nn_acc_reference", "rules")} for r in runs],
                              "used": scribe_discovery(args, G, K, best["features"], True, args.out)}
        print("scribes", json.dumps([{k: r[k] for k in ("features", "loo_nearest_centroid_acc")} for r in runs]), flush=True)
    for run in summary.get("scribes", {}).get("compared", []) + ([summary["scribes"]["used"]] if "scribes" in summary else []):
        for rule in run["rules"].values():
            rule.pop("_prec", None)
    summary["plain_statement"] = plain_statement(summary)
    summary["seconds"] = round(time.time() - t0, 1)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    if args.out == OUT:
        (HERE / "results").mkdir(exist_ok=True)
        shutil.copy(args.out / "summary.json", HERE / "results" / "discovery_v2_summary.json")
    print(summary["plain_statement"], flush=True)
    return summary


if __name__ == "__main__":
    main()
