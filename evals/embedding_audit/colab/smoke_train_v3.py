"""CPU smoke test of ``train_embedder_v3`` end to end on a TINY random-init Qwen3 model (no real weights).

Builds a miniature dataset folder in the packaged layout from ``AUDIT_ROOT/hf_dataset/v3`` (~300 eval-pool records
chosen by ``step0_local.smoke_selection`` with a reduced eval dir, 64 mixture pairs over all three kinds, a few hundred
train-pool records incl. same-subject neighbours), then runs the notebook's stages with the packaged code copies:
mask self-test -> load_data -> evaluate base (original + semantic text) -> evaluate a PREVIOUS model (a different
tiny random model saved to disk and reloaded through ``load_previous``, standing in for ``v3-run1``) ->
mine_negatives -> train (masked loss, every kind) -> evaluate tuned -> compare_reports (``also_vs`` the previous
model) -> save_reports (no upload). Asserts that every kind trained for at least two batches, masking fired, losses
are finite, weights moved, the four reports line up, and the primary comparison uses the package's primary group
(the held-out gate for ``v3``; the focus group for ``v3-final``, which has no held-out subject).

Run through the guard (CPU, 2 threads, 3 GB cap, no mutex)::

    HF_HUB_OFFLINE=1 python3 guard.py --name t4_smoke --small -- $PY colab/smoke_train_v3.py
    HF_HUB_OFFLINE=1 python3 guard.py --name t4_smoke_final --small -- $PY colab/smoke_train_v3.py \
        --pkg AUDIT_ROOT/hf_dataset/v3-final --out AUDIT_ROOT/v3/t4_smoke_final
"""

import argparse
import importlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))

import eval_v3  # noqa: E402
import step0_local  # noqa: E402
from embed_utils import AUDIT_ROOT  # noqa: E402

PKG = AUDIT_ROOT / "hf_dataset" / "v3"
OUT = AUDIT_ROOT / "v3" / "t4_smoke"


def iter_jsonl(path: Path):
    """Yield JSONL rows.

    :param path: File.
    :yields: Row dicts.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    """Write JSONL rows.

    :param path: File.
    :param rows: Row dicts.
    :returns: Rows written.
    :rtype: int
    """
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def pick_pairs(mix: List[dict], n_q2d: int, n_d2d: int, n_t2t: int, rng: random.Random) -> tuple:
    """Mixture rows that exercise every masking rule: q2d rows sharing one subject across families and mask levels,
    d2d rows, t2t rows sharing concepts.

    :param mix: Full mixture.
    :param n_q2d: q2d rows.
    :param n_d2d: d2d rows.
    :param n_t2t: t2t rows.
    :param rng: RNG.
    :returns: (rows, the shared q2d subject).
    :rtype: tuple
    """
    q2d = [r for r in mix if r["kind"] == "q2d"]
    by_subject: Dict[str, List[dict]] = defaultdict(list)
    for r in q2d:
        for s in r["mask_ids"]:
            if s.startswith("pgp:"):
                by_subject[s].append(r)

    def mix_score(s: str) -> tuple:
        rows = by_subject[s]
        fams = {r["family"] for r in rows}
        return (("subject" in fams) + any(r["mask_level"] == "doc" for r in rows) + ("content" in fams), len(rows))

    subject = max(sorted(by_subject), key=mix_score)
    rows = by_subject[subject]
    picked = [r for r in rows if r["family"] == "subject" and r.get("anchor_subject") == subject][:6]
    picked += [r for r in rows if r["family"].startswith("synthetic_") and r["mask_level"] == "subject"][:6]
    picked += [r for r in rows if r["mask_level"] == "doc" and r["family"] != "content"][:5]
    picked += [r for r in rows if r["family"] == "content"][:3]
    seen = {id(r) for r in picked}
    rest = [r for r in q2d if id(r) not in seen]
    picked += rng.sample(rest, max(0, n_q2d - len(picked)))
    d2d = [r for r in mix if r["kind"] == "d2d"]
    picked += rng.sample(d2d, n_d2d)
    t2t = [r for r in mix if r["kind"] == "t2t"]
    by_concept: Dict[str, List[dict]] = defaultdict(list)
    for r in t2t:
        for c in r["concept_ids"]:
            by_concept[c].append(r)
    shared, used = [], set()
    for c in sorted(by_concept, key=lambda x: -len(by_concept[x])):
        group = [r for r in by_concept[c] if id(r) not in used][:2]
        texts = [t for r in group for t in (r["anchor"], r["positive"])]
        if len(group) == 2 and len(set(texts)) == len(texts):  # NO_DUPLICATES would keep exact repeats apart
            shared += group
            used.update(id(r) for r in group)
        if len(shared) >= n_t2t // 2:
            break
    picked += shared + rng.sample([r for r in t2t if id(r) not in used], n_t2t - len(shared))
    return picked, subject


def build_smoke_package(pkg: Path, out: Path, n_records: int, n_pool: int, seed: int) -> dict:
    """Write the miniature dataset folder ``out/pkg`` in the packaged layout.

    :param pkg: Full packaged folder.
    :param out: Smoke output root.
    :param n_records: Eval-pool records.
    :param n_pool: Random extra train-pool records (plus positives, anchors and same-subject neighbours).
    :param seed: RNG seed.
    :returns: Selection summary.
    :rtype: dict
    """
    rng = random.Random(seed)
    dst = out / "pkg"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "eval").mkdir(parents=True)
    ev = eval_v3.load_eval_dir(pkg / "eval")
    sel = step0_local.smoke_selection(ev, n_records, seed)
    step0_local.write_smoke_eval_dir(ev, sel, dst / "eval")
    for name in ("README.md", "build_stats.json", "focus_subjects.json"):
        if (pkg / "eval" / name).exists():
            shutil.copyfile(pkg / "eval" / name, dst / "eval" / name)
    eval_ids = set(sel["doc_ids"])
    mix = list(iter_jsonl(pkg / "train_mix_v3.jsonl"))
    pairs, subject = pick_pairs(mix, n_q2d=24, n_d2d=20, n_t2t=20, rng=rng)
    del mix
    need = {r["doc_id"] for r in pairs if r["doc_id"]} | {r["anchor_doc_id"] for r in pairs if r.get("anchor_doc_id")}
    pool_all = list(iter_jsonl(pkg / "train_pool.jsonl"))
    neighbours = [r["doc_id"] for r in pool_all if subject in r["subjects"] and r["doc_id"] not in need][:60]
    others = [r["doc_id"] for r in pool_all if r["doc_id"] not in need]
    keep = need | set(neighbours) | set(rng.sample(others, n_pool))
    n_pool_rows = write_jsonl(dst / "train_pool.jsonl", (r for r in pool_all if r["doc_id"] in keep))
    del pool_all
    write_jsonl(dst / "train_mix_v3.jsonl", pairs)
    write_jsonl(dst / "corpus_semantic.jsonl",
                (r for r in iter_jsonl(pkg / "corpus_semantic.jsonl") if r["doc_id"] in eval_ids))
    write_jsonl(dst / "corpus_original.jsonl",
                (r for r in iter_jsonl(pkg / "corpus_original.jsonl") if r["doc_id"] in eval_ids))
    for name in ("eval_v3.py", "train_embedder_v3.py", "manifest.json"):
        shutil.copyfile(pkg / name, dst / name)
    kinds = defaultdict(int)
    for r in pairs:
        kinds[f"{r['kind']}:{r['mask_level']}"] += 1
    return {"eval_records": len(eval_ids), "pairs": len(pairs), "by_kind_level": dict(kinds),
            "shared_subject": subject, "pool_records": n_pool_rows, "same_subject_neighbours": len(neighbours),
            "eval_subjects": sel["subjects"]}


def main() -> None:
    """Run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pkg", default=str(PKG))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--n-records", type=int, default=300)
    parser.add_argument("--n-pool", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-prev", action="store_true", help="skip the previous-model scoring path")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selection = build_smoke_package(Path(args.pkg), out, args.n_records, args.n_pool, args.seed)
    print(json.dumps({"selection": selection}, ensure_ascii=False), flush=True)

    sys.path.insert(0, str(out / "pkg"))  # the packaged copies of train_embedder_v3 / eval_v3
    te = importlib.import_module("train_embedder_v3")
    assert Path(te.__file__).parent == out / "pkg", te.__file__
    import torch

    te.self_test()
    data = te.load_data(str(out / "pkg"))
    model = te.tiny_model(hidden=64, layers=2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    reports = {}
    for label, text in (("base_original", "original"), ("base_semantic", "semantic")):
        reports[label] = te.evaluate(model, data, label, text=text, max_seq=256, batch_size=8, n_resamples=100,
                                     out_path=str(out / "reports" / f"{label}.json"))
        print(label, json.dumps(reports[label]["pool"]), reports[label]["meta"]["total_seconds"], "s", flush=True)

    # previous-model path (the notebook's v3-run1 cell): a different tiny model, saved and reloaded via load_previous
    also_vs = []
    if not args.no_prev:
        prev_dir = out / "prev_standin"
        if prev_dir.exists():
            shutil.rmtree(prev_dir)
        te.tiny_model(hidden=64, layers=2, seed=1).save(str(prev_dir))
        prev = te.load_previous(str(prev_dir), device="cpu")
        reports["prev_standin"] = te.evaluate(prev, data, "prev_standin", text="semantic", max_seq=256, batch_size=8,
                                              n_resamples=100, out_path=str(out / "reports" / "prev_standin.json"))
        print("prev_standin", json.dumps(reports["prev_standin"]["pool"]), flush=True)
        del prev
        also_vs = ["prev_standin"]

    pairs = te.mine_negatives(model, data, top_k=40, skip_top=1, pick_from=5, max_seq=128, batch_size=16)
    mining = te.mining_stats(pairs)
    print(json.dumps({"mining": mining}), flush=True)

    model_dir = out / "model"
    tuned = te.train(data, pairs, str(model_dir), epochs=1.0, lr=1e-3, batch=8, mini_batch=4, max_seq=128,
                     bf16=False, device="cpu", model=model, logging_steps=1)
    info = json.loads((model_dir / "training_info.json").read_text())
    reports["tuned_smoke"] = te.evaluate(tuned, data, "tuned_smoke", text="semantic", max_seq=256, batch_size=8,
                                         n_resamples=100, out_path=str(out / "reports" / "tuned_smoke.json"))
    # also_vs only when used: the v3 package ships the older trainer copy, whose compare_reports lacks it (--no-prev)
    comparison = te.compare_reports(reports, baseline="base_semantic", candidate="tuned_smoke", n_resamples=200,
                                    **({"also_vs": also_vs} if also_vs else {}))
    print(te.format_table(comparison), flush=True)
    rep_dir = te.save_reports(str(model_dir), reports, comparison, {"run_name": "smoke", "data_revision": "local"})

    # ---- assertions
    batches = defaultdict(int)
    for name, st in info["mask_stats"].items():
        batches[te.kind_of(name)] += st.get("batches", 0)
    assert all(batches[k] >= 2 for k in ("q2d", "d2d", "t2t")), dict(batches)
    assert sum(st.get("qd_masked", 0) for n, st in info["mask_stats"].items() if n.startswith("q2d")) > 0, \
        "no q2d candidate was masked although the q2d rows share a subject"
    losses = [h["loss"] for h in info["loss_history"]]
    assert losses and all(torch.isfinite(torch.tensor(x)) for x in losses), losses
    moved = sum(int(not torch.equal(before[n], p.detach())) for n, p in tuned.named_parameters())
    assert moved > 0, "no parameter changed"
    assert mining["total"].get("same_subject", 0) + mining["total"].get("any", 0) > 0, mining
    pools = {lb: (r["pool"]["n_pool_scored"], r["pool"]["n_query_texts"]) for lb, r in reports.items()}
    assert len(set(pools.values())) == 1, pools
    assert reports["base_semantic"]["pool"]["text_hash_mismatch"] == 0
    assert reports["tuned_smoke"]["pool"]["text_hash_mismatch"] == 0
    assert comparison["primary"] is not None and comparison["primary"]["n_clusters"] >= 1, comparison["primary"]
    final = (Path(args.pkg) / "eval" / "focus_subjects.json").exists()
    pg = "focus" if final else "held_out"
    assert comparison.get("primary_group", "held_out") == pg, comparison.get("primary_group")
    assert comparison.get("primary_name", f"tuned_smoke - base_semantic | {pg} AP") == \
        f"tuned_smoke - base_semantic | {pg} AP", comparison.get("primary_name")
    if final:
        sub = reports["tuned_smoke"]["subjects"]
        assert sub["gate"] is None and not sub["held_out"]["per_query"] and sub["focus"]["per_query"], sub["focus"]
        assert all(v["n_excluded_train_side"] > 0 for v in sub["focus"]["per_subject"].values()), sub["focus"]
        assert not any("| held_out" in k for k in comparison["paired"]), list(comparison["paired"])
    if also_vs:
        assert f"tuned_smoke - prev_standin | {pg} AP" in comparison["paired"], list(comparison["paired"])
        assert f"prev_standin - base_semantic | {pg} AP" in comparison["paired"], list(comparison["paired"])
        assert reports["prev_standin"]["pool"]["text_hash_mismatch"] == 0
    # the saved model reloads on the production input path (plain-text prompt, no chat template) at the production window
    from sentence_transformers import SentenceTransformer

    reloaded = SentenceTransformer(str(model_dir), device="cpu")
    te.check_text_path(reloaded)
    assert reloaded.max_seq_length == te.EVAL_MAX_SEQ, reloaded.max_seq_length
    assert all("memorisation_trained" in r for r in reports.values()), "trained-positive memorisation missing"
    result = {"smoke": "ok", "batches_per_kind": dict(batches), "global_steps": info["global_steps"],
              "datasets": info["datasets"]["rows"], "label_width": info["datasets"]["label_width"],
              "masked_share": info["masked_share"], "mask_stats": info["mask_stats"], "losses": losses,
              "params_moved": moved, "mining": mining, "primary_group": comparison.get("primary_group", "held_out"),
              "primary": comparison["primary"], "reports": str(rep_dir),
              "paired_vs_prev": {k: v for k, v in comparison["paired"].items() if "prev_standin" in k},
              "train_seconds": info["train_seconds"],
              "eval_seconds": {lb: r["meta"]["total_seconds"] for lb, r in reports.items()}}
    (out / "smoke_result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
