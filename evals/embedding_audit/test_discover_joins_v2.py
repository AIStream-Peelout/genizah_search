"""End-to-end test of ``discover_joins_v2.py`` on SYNTHETIC features over the real gallery manifest (no model).

Writes synthetic shards in the exact format of the Colab gallery run (``ids``, ``urls``, fp16 ``merger`` / ``ds8`` /
``head`` / ``dino_masked`` / ``dino_mpatches``, ``sha1``, ``dhash``, ``mask_ok`` ...) for every manifest fragment
(minus a random 0.3 % "failed downloads"), with planted structure whose truth only this test knows:

* a holding effect (photography confound) shared by all fragments of a holding;
* every known join group (FJP clean + PGP 2-6 + clean loose FJP: the discovery positives) gets a shared vector with a
  per-group strength drawn from U(0, ``smax``) — some joins are invisible, as in reality;
* ``n_hidden`` UNKNOWN joins between fragments outside every known link, drawn with the strength distribution and
  the same-/cross-holding mix of the known ones and without KTIV attribute conflicts — the "undiscovered" joins;
* a vector per PGP scribe on its labelled fragments, and on ``n_hidden_scribe`` unlabelled documentary fragments
  (date-compatible with the scribe when both are dated) — the attributions to discover; every other unlabelled
  documentary fragment gets an unknown hand (groups of ~3);
* ``n_dup`` pairs of unrelated fragments showing the same photograph (near-identical vectors, equal dHash; half
  also share the image URL).

Rows the real run skips (``image_live`` false) get no shard row; planted structure avoids linked, duplicate-record and
container records.

Then ``discover_joins_v2.main`` runs on them with ``--rho`` = the true hidden/known ratio, and the checks are:
held-out calibration (predicted vs observed known joins per band, other half of the join groups), the odds transfer
(planted hidden joins found among the candidates vs the sum of ``p_discovery``), exclusions (no known, linked,
linked-through-a-duplicate, container, same-shelfmark, same-photo / same-file or attribute-conflicting pair
proposed; planted duplicates listed), the hide-one-source
validation, and the target-decoy scribe precision estimate against the planted attributions (within 0.1).

Run (NumPy only): ``nice -n 10 <HDA .venv>/bin/python test_discover_joins_v2.py``; synthetic shards go to
``<AUDIT_ROOT>/gallery_v1/synthetic/``, outputs to ``<AUDIT_ROOT>/results/discovery_v2_synthetic/``, the verdict
to ``results/discovery_v2_synthetic_test.json``.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

import discover_joins_v2 as dj
from discover_joins_scribes import ATTR_KEYS, UnionFind, attr_conflicts, date_intervals, date_status
from embed_utils import AUDIT_ROOT

HERE = Path(__file__).resolve().parent
SYN = AUDIT_ROOT / "gallery_v1" / "synthetic"
OUT = AUDIT_ROOT / "results" / "discovery_v2_synthetic"
DIMS = {"merger": 192, "ds8": 192, "head": 48, "dino_masked": 96, "dino_mpatches": 96}


def positive_pairs(rows: List[dict], known: dict) -> Set[Tuple[str, str]]:
    """The discovery positives (same definition as ``discover_joins_v2.known_structure``).

    :param rows: Manifest rows.
    :param known: ``known_joins.json``.
    :returns: Id pairs.
    :rtype: Set[Tuple[str, str]]
    """
    P = known["pairs"]
    return {tuple(p[:2]) for p in P["fjp_clean"] + P["pgp"]} | {tuple(p[:2]) for p in P["fjp_loose"] if not p[3]}


def make_synthetic(rows: List[dict], known: dict, args: argparse.Namespace) -> dict:
    """Build and write the synthetic shards; return the planted truth.

    :param rows: Manifest rows.
    :param known: ``known_joins.json``.
    :param args: CLI arguments.
    :returns: Truth dict.
    :rtype: dict
    """
    rng = np.random.default_rng(args.seed)
    n = len(rows)
    ids = [r["canonical_id"] for r in rows]
    pos = {c: i for i, c in enumerate(ids)}
    hold = np.unique([r["holding"] for r in rows], return_inverse=True)[1]
    X = {b: rng.standard_normal((n, d)).astype(np.float32) / math.sqrt(d) for b, d in DIMS.items()}
    for b, d in DIMS.items():
        H = rng.standard_normal((hold.max() + 1, d)).astype(np.float32) / math.sqrt(d)
        X[b] += args.w_hold * H[hold]

    def plant(members: List[int], strength: float, blocks: Dict[str, int] = DIMS) -> None:
        """Add one shared random direction to a set of fragments.

        :param members: Fragment indices.
        :param strength: Weight of the shared vector.
        :param blocks: Blocks to add it to.
        """
        for b in blocks:
            v = rng.standard_normal(DIMS[b]).astype(np.float32) / math.sqrt(DIMS[b])
            X[b][members] += strength * v

    # known join groups (components of the positives)
    positives = sorted(positive_pairs(rows, known))  # sorted: set order of strings varies between processes
    uf = UnionFind(n)
    for a, b in positives:
        uf.union(pos[a], pos[b])
    comp = defaultdict(list)
    for i in sorted({pos[c] for p in positives for c in p}):
        comp[uf.find(i)].append(i)
    for root in sorted(comp):
        plant(comp[root], float(rng.uniform(0, args.smax)))
    cross_share = float(np.mean([rows[pos[a]]["holding"] != rows[pos[b]]["holding"] for a, b in positives]))

    # hidden (undiscovered) joins: groups with the size distribution of the known join groups, between fragments
    # outside every known link, same/cross-holding like the known pairs, without KTIV attribute conflicts
    sizes = np.array([len(set(m)) for m in comp.values()])
    free = [i for i, r in enumerate(rows) if not r.get("link_group") and not r.get("pgp_scribe") and not r.get("dup_group")
            and not r.get("container_record") and r.get("image_live") is not False]
    by_hold = defaultdict(list)
    for i in free:
        by_hold[rows[i]["holding"]].append(i)
    used: Set[int] = set()
    hidden_groups: List[List[int]] = []
    n_pairs = 0
    while n_pairs < args.n_hidden:
        size = int(rng.choice(sizes))
        a = int(rng.choice(free))
        if a in used:
            continue
        group = [a]
        for _ in range(200):
            if len(group) == size:
                break
            want_cross = rng.random() < cross_share
            cand = free if want_cross else by_hold[rows[a]["holding"]]
            b = int(cand[rng.integers(len(cand))])
            rb = rows[b]
            if b in used or b in group or (rows[a]["holding"] != rb["holding"]) != want_cross \
                    or any(rows[g]["shelf_norm"] == rb["shelf_norm"] or attr_conflicts(rows[g], rb, ATTR_KEYS)
                           or set(rows[g]["pgpids"]) & set(rb["pgpids"]) for g in group):
                continue
            group.append(b)
        if len(group) < 2:
            continue
        used |= set(group)
        hidden_groups.append(group)
        n_pairs += len(group) * (len(group) - 1) // 2
        plant(group, float(rng.uniform(0, args.smax)))
    hidden = [(g[i], g[j]) for g in hidden_groups for i in range(len(g)) for j in range(i + 1, len(g))]

    # scribes: labelled fragments + hidden members among unlabelled documentary fragments
    lab = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("pgp_scribe"):
            lab[r["pgp_scribe"]].append(i)
    years = defaultdict(list)
    for s, members in lab.items():
        for i in members:
            years[s] += rows[i]["pgp_years"]
    windows = {s: (min(y) - 15, max(y) + 15) for s, y in years.items() if len([i for i in lab[s] if rows[i]["pgp_years"]]) >= 2}
    labelled_links = {rows[i]["link_group"] for m in lab.values() for i in m if rows[i].get("link_group")}
    lab_docs = {d for m in lab.values() for i in m for d in rows[i]["pgpids"]}
    cand_u = [i for i, r in enumerate(rows) if r.get("documentary") and not r.get("pgp_scribe") and i not in used
              and r.get("link_group") not in labelled_links and not (set(r["pgpids"]) & lab_docs)]
    big = [s for s, m in lab.items() if len(m) >= 5]
    strength = {s: float(rng.uniform(args.scribe_min, args.scribe_max)) for s in lab}
    sv = {s: {b: rng.standard_normal(DIMS[b]).astype(np.float32) / math.sqrt(DIMS[b]) for b in DIMS} for s in lab}
    hidden_scribe: Dict[int, str] = {}
    weights = np.array([len(lab[s]) for s in big], float)
    order = rng.permutation(cand_u)
    for i in order:
        if len(hidden_scribe) >= args.n_hidden_scribe:
            break
        s = big[int(rng.choice(len(big), p=weights / weights.sum()))]
        st = date_status(date_intervals({"years": rows[i]["pgp_years"], "pgp_tags": rows[i]["pgp_tags"]}), windows.get(s))
        if st == "incompatible":
            continue
        hidden_scribe[int(i)] = s
    for s, members in lab.items():
        for b in DIMS:
            X[b][members] += strength[s] * sv[s][b]
    for i, s in hidden_scribe.items():
        for b in DIMS:
            X[b][i] += strength[s] * sv[s][b]
        used.add(i)
    # every other unlabelled documentary fragment is in some UNKNOWN scribe's hand (groups of ~3), as real documents
    # are: the leave-one-scribe-out open-set rate the estimate uses assumes exactly that
    others = [i for i in cand_u if int(i) not in hidden_scribe]
    hand = rng.integers(0, max(1, len(others) // 3), len(others))
    for h in np.unique(hand):
        plant([others[k] for k in np.nonzero(hand == h)[0]], float(rng.uniform(args.scribe_min, args.scribe_max)))

    # duplicate photographs between unrelated fragments
    urls = np.array([r["image_url"] for r in rows], dtype=object)
    sha1 = np.array([f"{x:040x}" for x in rng.integers(0, 2 ** 62, n)], dtype=object)
    dhash = rng.integers(0, 2 ** 64 - 1, n, dtype=np.uint64)
    dups = []
    while len(dups) < args.n_dup:
        a, b = (int(x) for x in rng.choice(free, 2, replace=False))
        if a in used or b in used or rows[a]["shelf_norm"] == rows[b]["shelf_norm"]:
            continue
        used |= {a, b}
        for blk in DIMS:
            X[blk][b] = X[blk][a] + 0.01 * rng.standard_normal(DIMS[blk]).astype(np.float32) / math.sqrt(DIMS[blk])
        dhash[b] = dhash[a]
        if len(dups) % 2 == 0:
            urls[b] = urls[a]
        dups.append((a, b))

    for b in DIMS:
        X[b] /= np.linalg.norm(X[b], axis=1, keepdims=True)
    drop = set(rng.choice(n, int(args.drop * n), replace=False).tolist()) - {i for p in hidden + dups for i in p}
    drop |= {i for i, r in enumerate(rows) if r.get("image_live") is False}  # the real run skips dead-image rows
    keep = [i for i in range(n) if i not in drop]
    shard_dir = SYN / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("part_*.npz"):
        old.unlink()
    for k in range(0, len(keep), 2000):
        sel = keep[k:k + 2000]
        np.savez(shard_dir / f"part_{k // 2000:04d}.npz", ids=np.array([ids[i] for i in sel], dtype=str),
                 urls=np.array([urls[i] for i in sel], dtype=str), sha1=np.array([sha1[i] for i in sel], dtype=str),
                 dhash=dhash[sel], mask_ok=np.ones(len(sel), bool), mask_frac=np.full(len(sel), 0.8, np.float32),
                 crop_wh=np.full((len(sel), 2), 1000, np.int32), max1600_wh=np.full((len(sel), 2), 1600, np.int32),
                 n_patches=np.full(len(sel), 16, np.int16), run_info=np.array(json.dumps({"synthetic": True})),
                 **{b: X[b][sel].astype(np.float16) for b in DIMS})
    in_gallery_pos = sum(1 for a, b in positives if pos[a] not in drop and pos[b] not in drop)
    return {"hidden": [(ids[a], ids[b]) for a, b in hidden], "hidden_scribe": {ids[i]: s for i, s in hidden_scribe.items()},
            "dups": [(ids[a], ids[b]) for a, b in dups], "dropped": len(drop), "n_kept": len(keep),
            "positives_in_gallery": in_gallery_pos, "rho_true": len(hidden) / max(1, in_gallery_pos),
            "known_cross_share": round(cross_share, 4), "hidden_groups": len(hidden_groups),
            "hidden_cross_share": round(float(np.mean([rows[a]["holding"] != rows[b]["holding"] for a, b in hidden])), 4)}


def read_csv(path: Path) -> List[dict]:
    """Read a discovery CSV.

    :param path: CSV path.
    :returns: Rows.
    :rtype: List[dict]
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def poisson_ok(found: float, expected: float, z: float = 3.0) -> bool:
    """Whether an observed count is within z Poisson SDs (+2 slack) of its expectation.

    :param found: Observed.
    :param expected: Expected.
    :param z: Tolerance in SDs.
    :returns: True when consistent.
    :rtype: bool
    """
    return abs(found - expected) <= z * math.sqrt(max(expected, 1.0)) + 2


def main() -> None:
    """Generate, run discovery, check, write the verdict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--w-hold", type=float, default=0.25)
    parser.add_argument("--smax", type=float, default=1.5)
    parser.add_argument("--n-hidden", type=int, default=800, help="hidden join PAIRS (planted as groups)")
    parser.add_argument("--n-hidden-scribe", type=int, default=300)
    parser.add_argument("--scribe-min", type=float, default=0.2)
    parser.add_argument("--scribe-max", type=float, default=0.7)
    parser.add_argument("--n-dup", type=int, default=40)
    parser.add_argument("--drop", type=float, default=0.003)
    parser.add_argument("--block", type=int, default=1024)
    args = parser.parse_args()
    t0 = time.time()
    rows = dj.read_manifest(dj.GALLERY / "manifest.jsonl")
    known = json.loads((dj.GALLERY / "known_joins.json").read_text())
    truth = make_synthetic(rows, known, args)
    print(json.dumps({k: v for k, v in truth.items() if k not in ("hidden", "hidden_scribe", "dups")}), flush=True)
    summary = dj.main(["--shards-dir", str(SYN / "shards"), "--out", str(OUT), "--rho", str(truth["rho_true"]),
                       "--block", str(args.block)])
    J, S = summary["joins"], summary["scribes"]["used"]
    checks: Dict[str, dict] = {}

    # gallery loading
    checks["coverage"] = {"ok": summary["gallery"]["fragments_with_features"] == truth["n_kept"],
                          "loaded": summary["gallery"]["fragments_with_features"], "written": truth["n_kept"]}

    # held-out calibration: predicted vs observed known joins, per band and in total (other half of the groups)
    held = J["calibration"]["heldout"]
    band_z = []
    for stratum in ("all", "same_holding", "cross_holding"):
        for r in held[stratum]["reliability"]:
            if r["pairs"] >= 200 and r["mean_predicted"] is not None:
                p = r["mean_predicted"]
                sd = math.sqrt(max(p * (1 - p), 1e-6) / r["pairs"])
                band_z.append({"stratum": stratum, "band": r["p_band"], "pairs": r["pairs"], "pred": p, "obs": r["observed"],
                               "z": round((r["observed"] - p) / sd, 2)})
    tot_pred = sum(r["mean_predicted"] * r["pairs"] for r in held["all"]["reliability"] if r["pairs"])
    tot_obs = sum(r["known_joins"] for r in held["all"]["reliability"])
    worst = max(abs(b["z"]) for b in band_z)
    checks["heldout_calibration"] = {"ok": abs(tot_obs - tot_pred) <= 0.15 * tot_obs and sum(abs(b["z"]) > 4 for b in band_z) <= 1,
                                     "predicted_known_joins": round(tot_pred, 1), "observed_known_joins": tot_obs,
                                     "auc_all": held["all"]["auc"], "worst_band_z": worst, "bands": band_z}

    # odds transfer: planted hidden joins among the candidates vs sum of p_discovery (rho = truth)
    hidden = {tuple(sorted(p)) for p in truth["hidden"]}
    res = {}
    for name in ("join_candidates.csv", "join_candidates_ranked.csv"):
        cands = read_csv(OUT / name)
        found = sum(tuple(sorted((c["id_a"], c["id_b"]))) in hidden for c in cands)
        expected = sum(float(c["p_discovery"]) for c in cands)
        found_x = sum(tuple(sorted((c["id_a"], c["id_b"]))) in hidden for c in cands if c["cross_holding"] == "1")
        expected_x = sum(float(c["p_discovery"]) for c in cands if c["cross_holding"] == "1")
        # known joins come in groups (a third have > 2 members) while the planted hidden joins are pairs, so their rank
        # profiles differ a little: accept Poisson agreement or a found/expected ratio within 0.7-1.4
        consistent = lambda f, e: poisson_ok(f, e) or 0.7 <= f / max(e, 1e-9) <= 1.4  # noqa: E731
        res[name] = {"n": len(cands), "hidden_found": found, "expected_sum_p_discovery": round(expected, 2),
                     "ratio": round(found / max(expected, 1e-9), 3), "cross_found": found_x, "cross_expected": round(expected_x, 2),
                     "ok": consistent(found, expected) and consistent(found_x, expected_x)}
    checks["odds_transfer"] = {"ok": all(v["ok"] for v in res.values()), **res, "rho_true": round(truth["rho_true"], 4)}

    # exclusions
    cands = read_csv(OUT / "join_candidates.csv") + read_csv(OUT / "join_candidates_ranked.csv")
    byid = {r["canonical_id"]: r for r in rows}
    dups = {tuple(sorted(p)) for p in truth["dups"]}
    applied = J["attribute_filter"]["applied_keys"]
    bad = Counter()
    # the exclusion closure: link groups joined through duplicate records (dup_group)
    uf = UnionFind(len(rows))
    pos = {r["canonical_id"]: i for i, r in enumerate(rows)}
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        for key in ("link_group", "dup_group"):
            if r.get(key):
                groups[f"{key}:{r[key]}"].append(i)
    for m in groups.values():
        for x in m[1:]:
            uf.union(m[0], x)
    for c in cands:
        a, b = byid[c["id_a"]], byid[c["id_b"]]
        if a.get("link_group") and a["link_group"] == b.get("link_group"):
            bad["known_or_linked"] += 1
        elif uf.find(pos[c["id_a"]]) == uf.find(pos[c["id_b"]]):
            bad["linked_through_duplicate_record"] += 1
        if a.get("container_record") or b.get("container_record"):
            bad["container_record"] += 1
        if c["id_b"] in (a.get("image_overlap_ids") or []) or c["id_a"] in (b.get("image_overlap_ids") or []):
            bad["same_image_file"] += 1
        if tuple(sorted((c["id_a"], c["id_b"]))) in dups:
            bad["planted_duplicate"] += 1
        if a["holding"] == b["holding"] and a["shelf_norm"] and a["shelf_norm"] == b["shelf_norm"]:
            bad["same_shelfmark"] += 1
        if attr_conflicts(a, b, applied):
            bad["attribute_conflict"] += 1
        if set(a["pgpids"]) & set(b["pgpids"]):
            bad["shared_pgp_document"] += 1
    listed = {tuple(sorted((r["id_a"], r["id_b"]))) for r in read_csv(OUT / "possible_duplicate_records.csv")}
    checks["exclusions"] = {"ok": not bad and len(dups & listed) == len(dups), "violations": dict(bad),
                            "planted_duplicates_listed": len(dups & listed), "planted_duplicates": len(dups)}

    # hide-one-source validation ran and is in a sane range (both sources are planted alike here)
    v = J["validation_hide_one_source"]
    ratio = J["validation_found_over_predicted"]
    checks["hide_one_source"] = {"ok": ratio is not None and 0.5 <= ratio <= 2.0, "found_over_predicted": ratio,
                                 **{k: {kk: vv for kk, vv in d.items() if kk in ("hidden_joins_found", "predicted_hidden_joins")}
                                    for k, d in v.items()}}

    # scribes: realised precision of the written candidates vs the estimates
    sc = read_csv(OUT / "scribe_candidates.csv")
    hs = truth["hidden_scribe"]
    right = sum(hs.get(c["id"]) == c["proposed_scribe"] for c in sc) if sc else 0
    if S.get("chosen_score") is None:
        checks["scribes"] = {"ok": False, "reason": "no attribution rule reached the precision target",
                             "compared": [{k: r[k] for k in ("features", "loo_nearest_centroid_acc")}
                                          for r in summary["scribes"]["compared"]]}
        return finish(checks, truth, J, args, t0)
    rule = S["rules"][S["chosen_score"]]
    est = rule["precision_estimate"]
    realised = right / max(1, len(sc))
    exp_correct = est * len(sc)
    # the date filter drops a few candidates after the estimate was made, so compare with a 0.1 tolerance both ways
    checks["scribes"] = {"ok": len(sc) > 0 and abs(realised - est) <= 0.1, "gap_realised_minus_estimate":
                         round(realised - est, 4), "candidates": len(sc), "correct": right,
                         "realised_precision": round(realised, 4), "decoy_precision_estimate": est,
                         "expected_correct_estimate": round(exp_correct, 1),
                         "decoy_hits_above": rule.get("decoys_above"), "features": S["features"],
                         "closed_set_v1_rule": rule["closed_set_v1_rule"], "loo_acc": S["loo_nearest_centroid_acc"],
                         "planted_hidden_members": len(hs), "recall_of_planted": round(right / max(1, len(hs)), 4)}

    finish(checks, truth, J, args, t0)


def finish(checks: Dict[str, dict], truth: dict, J: dict, args: argparse.Namespace, t0: float) -> None:
    """Write the verdict and exit with its status.

    :param checks: Check results.
    :param truth: Planted truth (summary fields only are written).
    :param J: Join section of the discovery summary.
    :param args: CLI arguments.
    :param t0: Start time.
    """
    verdict = {"status": "PASS" if all(c["ok"] for c in checks.values()) else "FAIL", "checks": checks,
               "truth": {k: v for k, v in truth.items() if k not in ("hidden", "hidden_scribe", "dups")},
               "retrieval_fused": J["retrieval_known_joins"]["fused"], "seconds": round(time.time() - t0, 1),
               "settings": vars(args)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "discovery_v2_synthetic_test.json").write_text(json.dumps(verdict, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in c.items() if kk != "bands"} for k, c in checks.items()}, indent=1, default=float))
    print(verdict["status"], f"{verdict['seconds']} s", flush=True)
    sys.exit(0 if verdict["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
