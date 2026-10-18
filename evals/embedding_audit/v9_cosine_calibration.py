"""v9 cosine calibration: how raw cosine scores move from today's embedder to the fine-tuned v9 embedder.

Today's production = base ``Qwen3-Embedding-0.6B`` (pinned revision) on the ORIGINAL production text
(``corpus_vectors/qwen3-0.6b__all__8192``). Index v9 = the fine-tuned ``isaacmg/genizah-embed-qwen3-0.6b`` (pinned
revision) on the v3 SEMANTIC text (``v3/corpus_semantic.jsonl``). Both use the production contract: documents raw,
queries prefixed with exactly :data:`QUERY_PREFIX` (NOT the tuned model's stock ``prompt_name="query"``), fp32,
L2-normalised, max_seq 8192.

Phases::

    # model job (GPU), through the guard; resumable; exits 99 above --soft-max-gb so a loop relaunches it
    guard.py --name gpu_v9_calib --max-gb 8 -- $PY v9_cosine_calibration.py embed --soft-max-gb 6.5
    # numpy only: local eval_v3 on the new vectors, compared with the Colab numbers of the same model (contract)
    $PY v9_cosine_calibration.py contract
    # numpy only: cosine distributions, gate curves, threshold mapping, hybrid-weight guidance
    $PY v9_cosine_calibration.py analyze

(``run_v9_calibration.sh`` chains a ``--small`` smoke test, the guarded embed loop, contract and analyze.)

``embed`` writes (all on the NAS):

* ``AUDIT_ROOT/v9_calibration/queries.json`` + ``queries_base.npy`` / ``queries_tuned.npy``: every eval_final query
  (subject + known-item) and the hand-written unanswerable queries of ``v9_ood_queries.json``, for both models;
* ``AUDIT_ROOT/corpus_vectors/genizah-v3final__semantic__8192/``: the eligible pool (60,493 records) embedded with the
  tuned model via ``step0_local``'s resumable shard machinery (``unique/`` shards, then ``ids.json`` + shards +
  ``meta.json`` + ``checks.json``), plus ``queries.{json,npy}`` (tuned eval queries) for later eval runs.

``analyze`` writes ``results/v9_cosine_calibration.json``. It runs with whatever is available: before the embed phase
has run it scores today's model only (base vectors + the stored step-0 base query vectors of the eval queries; no
unanswerable OOD queries, which need the model) and marks the v9 side as pending. ``--identity-test`` scores the base
model against itself as if it were the tuned one (every mapping must come out as the identity: a check of the code).

Smoke test (CPU, 3 GB cap; truncated models, so not the contract; writes ``AUDIT_ROOT/v9_calibration_smoke``)::

    guard.py --name v9_calib_smoke --small -- $PY v9_cosine_calibration.py embed --smoke 300 --smoke-layers 2
    $PY v9_cosine_calibration.py analyze --smoke
"""

import argparse
import gc
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np

import eval_v3
import step0_local as s0
from embed_utils import AUDIT_ROOT, DEFAULT_TASK, TEXT_MODELS, device
from eval_corpus import load_vectors
from guard import child_rss_gb

HERE = Path(__file__).parent
QUERY_PREFIX = "Instruct: Given a search query, retrieve relevant passages\nQuery: "
assert QUERY_PREFIX == f"Instruct: {DEFAULT_TASK}\nQuery: ", "prefix drifted from the production contract"
BASE = {**TEXT_MODELS["qwen3-0.6b"], "label": "base Qwen3-Embedding-0.6B"}
TUNED = {"id": "isaacmg/genizah-embed-qwen3-0.6b", "revision": "dbfab73f1e91a6b95eba728541cc5a391a88d933",
         "label": "tuned genizah-embed-qwen3-0.6b (v3-final-run1)"}
MAX_SEQ = 8192
EVAL_DIR = AUDIT_ROOT / "v3" / "eval_final"
SEMANTIC_CORPUS = AUDIT_ROOT / "v3" / "corpus_semantic.jsonl"
BASE_ORIG_VECS = AUDIT_ROOT / "corpus_vectors" / "qwen3-0.6b__all__8192"
BASE_SEM_QUERIES = AUDIT_ROOT / "corpus_vectors" / "qwen3-0.6b__semantic__8192"   # stored base query vectors (v3 set)
TUNED_VECS = AUDIT_ROOT / "corpus_vectors" / "genizah-v3final__semantic__8192"
CALIB_DIR = AUDIT_ROOT / "v9_calibration"
SMOKE_DIR = AUDIT_ROOT / "v9_calibration_smoke"
COLAB_EVAL = AUDIT_ROOT / "colab_runs" / "v3-final-run1" / "genizah_eval"
OOD_FILE = HERE / "v9_ood_queries.json"
RESULTS = HERE / "results" / "v9_cosine_calibration.json"
PROD_THRESHOLD = 0.4            # lms_agentic_search.SIMILARITY_THRESHOLD
HYBRID_WEIGHTS = [(90, 10), (80, 20), (70, 30), (60, 40), (50, 50), (40, 60), (30, 70), (10, 90)]  # (semantic, keyword)
CHUNK = 256
# Bibliography index (the index the chat's gate and author-constrained retrieval act on), embedded with BOTH models by
# bib_v9_check.py (same contract); its read-only production lexical ranks are in lexical.json.
BIB_DIR = AUDIT_ROOT / "v9_biblio"
BIB_KI = HERE / "results" / "v9_bib_known_item_queries.jsonl"
RRF_K = 60.0                    # search_bibliography.search_hybrid
ROUTER_SW, ROUTER_KW = 0.3, 0.7  # the router's usual bibliography_hybrid plan in 2026-09-15..10-07 traffic
EVIDENCE_LIMIT = 8              # build_bibliography_source_context(limit=8)

# What production compares SIMILARITY_THRESHOLD against (read from origin/prod-mbp; see the module report).
PRODUCTION_USAGE = {
    "similarity_threshold": {
        "where": "src/backend/lms_agentic_search.py: SIMILARITY_THRESHOLD = 0.4 (l.231); "
                 "all_results_below_threshold (l.512-516); _execute_searches_node fallback (l.2738-2810)",
        "compared_to": "similarity_score of the deduplicated BIBLIOGRAPHY results of all planned bibliography actions; "
                       "the gate fires when EVERY result is < 0.4 (i.e. the best one is < 0.4), only when the graph "
                       "returned nothing",
        "what_that_score_is": "search_bibliography.search_hybrid (l.335-511): weighted reciprocal-rank fusion, "
                              "score = 61 * sum_legs(weight_leg / (60 + rank_leg)), so a document ranked 1st by both "
                              "legs scores 1.0 and the top document of one leg alone scores that leg's weight. "
                              "'bibliography_semantic' actions also go through search_hybrid (weights 100/0), so their "
                              "top result always scores 1.0. Raw cosine is only recorded in retrieval_details.",
        "consequence": "the gate is rank-based and independent of the cosine scale: with both legs returning hits the "
                       "best score is >= max(semantic, keyword weight) >= 0.5, so it can only fire when the keyword "
                       "leg returns no hit at all and the semantic weight is < 40 % (or the semantic leg fails and "
                       "the keyword weight is < 40 %). Switching the embedder does not move it. The fallback branch "
                       "tests the RAW planner keyword_weight ((a.keyword_weight or 50) > 60) while the legs use "
                       "normalize_weights(), so a plan such as semantic_weight=30 with keyword_weight unset (effective "
                       "30/70) takes the 10/90 KEYWORD fallback; if that also finds no lexical hit it scores 0.1 and "
                       "NO_RELEVANT_SOURCES is set with a working embedder. The router in the 2026-09-15..10-07 "
                       "traffic always set keyword_weight=70 explicitly (semantic fallback, score 1.0), so this is "
                       "latent; it is embedder-independent either way.",
        "author_results": "search_by_author results score (_score/2 = (cos+1)/2), i.e. cosine-scaled, and are only "
                          "added when graph evidence exists, which disables the gate. But "
                          "deduplicate_bibliography_results() sorts them TOGETHER with the RRF-scored hybrid results "
                          "and build_bibliography_source_context() keeps the first 8, so the cosine scale does decide "
                          "the author-page / hybrid-page mix whenever a plan combines graph_scholar with bibliography "
                          "actions (not seen in the 2026-09-15..10-07 traffic: graph_scholar plans were alone)",
    },
    "primary_hybrid": {
        "where": "src/backend/search_service.py search_hybrid (l.1703-1890)",
        "formula": "function_score(match_all or filters; score_mode sum, boost_mode multiply): "
                   "semantic_weight * (cosine + 1) + keyword_weight * 1[fuzzy multi_match hit]; the keyword function "
                   "has no script, so it adds a constant per matching document (not BM25)",
        "callers": "the /search/hybrid endpoint (Advanced Search slider, default 50/50). The chat agent's "
                   "primary_hybrid action builds SearchRequest(semantic_weight=..., keyword_weight=...), whose model "
                   "has no weight fields, so search_hybrid's request.semanticWeight raises and the action fails "
                   "(logged as 'Search failed')",
        "consequence": "a document without a lexical hit outranks one with a hit only if its cosine is higher by more "
                       "than keyword_weight / semantic_weight; at 50/50 that needs a cosine gap > 1",
    },
    "primary_semantic": "search_service.search returns similarity_score = raw cosine (script_score cos + 1, minus 1); "
                        "the frontend shows it as '% match' (SearchResults.jsx, DocumentModel.jsx); no threshold. "
                        "search_bibliography.search (l.224-280) likewise returns raw cosine; ChatUI.jsx only prints "
                        "the bibliography similarity_score",
    "shared_embedder": "one embedding service (src/embedding_service, Qwen3-Embedding-0.6B @ 97b0c614) serves query "
                       "vectors for BOTH the fragment index (ELASTICSEARCH_INDEX) and the bibliography index "
                       "(ELASTICSEARCH_BIBLIOGRAPHY_INDEX); app.py verify_embedding_compatibility checks the canary "
                       "in BOTH indexes' mapping _meta at startup and disables all semantic search if either fails",
}


def log(msg: str) -> None:
    """Print a timestamped progress line.

    :param msg: Message text.
    """
    print(f"[v9cal {datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# query sets
# ---------------------------------------------------------------------------------------------------------------
def load_ood() -> List[dict]:
    """Hand-written unanswerable queries.

    :returns: Rows ``{text, lang, group}``.
    :rtype: List[dict]
    """
    return json.loads(OOD_FILE.read_text(encoding="utf-8"))["queries"]


def all_query_texts(ev: dict) -> List[str]:
    """Every query text both models must encode: eval_final subject + known-item queries, then the OOD set.

    :param ev: Loaded eval dir.
    :returns: Unique texts in a fixed order.
    :rtype: List[str]
    """
    return list(dict.fromkeys(s0.query_texts(ev) + [q["text"] for q in load_ood()]))


# ---------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------
def load_model(spec: dict, dtype_name: str, pad_multiple: int, n_layers: int = 0):
    """Load a pinned SentenceTransformer exactly like ``step0_local.load_encoder`` (dtype, pad multiple, 8192).

    :param spec: ``{id, revision}``.
    :param dtype_name: torch dtype name (``float32`` = production contract).
    :param pad_multiple: Pad every batch to a multiple of this many tokens.
    :param n_layers: Keep only the first N transformer layers (smoke only; 0 = all).
    :returns: The loaded model.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    dev = device()
    t0 = time.time()
    model = SentenceTransformer(spec["id"], revision=spec["revision"], device=dev,
                                model_kwargs={"dtype": getattr(torch, dtype_name)},
                                config_kwargs={"num_hidden_layers": n_layers} if n_layers else None)
    model.max_seq_length = MAX_SEQ
    if pad_multiple > 1:
        module = model[0]
        kwargs = dict(module.processing_kwargs)
        kwargs["text"] = {**kwargs.get("text", {}), "pad_to_multiple_of": pad_multiple}
        module.processing_kwargs = kwargs
        width = module.preprocess(["probe"])["input_ids"].shape[1]
        assert width % pad_multiple == 0, f"pad_to_multiple_of not applied (width {width})"
    log(f"loaded {spec['id']}@{spec['revision'][:8]} on {dev} ({next(model.parameters()).dtype}, layers "
        f"{n_layers or 'all'}) in {time.time() - t0:.0f} s; footprint {child_rss_gb(os.getpid()):.2f} GB")
    return model


def free_model(model) -> None:
    """Drop a model and release the MPS cache.

    :param model: Model to release.
    """
    import torch

    mps = model.device.type == "mps"
    del model
    gc.collect()
    if mps:
        torch.mps.empty_cache()


def model_checks(model, spec: dict, args: argparse.Namespace, texts: Sequence[str]) -> dict:
    """Contract checks on a loaded model before it embeds anything.

    * module stack: Transformer -> last-token Pooling -> Normalize, 1024-d, no default prompt (so ``encode`` without a
      prompt embeds documents raw);
    * batching invariance: ``texts`` encoded in the padded, length-bounded batches of the real run vs one at a time
      (min cosine must be >= ``--canary-min-cos``);
    * the stock ``prompt_name="query"`` text differs from :data:`QUERY_PREFIX` (recorded, so nobody uses it by accident).

    :param model: Loaded model.
    :param spec: ``{id, revision}``.
    :param args: CLI args (batch bounds).
    :param texts: Probe documents.
    :returns: Check results (``ok`` = all passed).
    :rtype: dict
    """
    mods = [type(m).__name__ for m in model]
    pooling = model[1].get_config_dict() if len(model) > 1 and hasattr(model[1], "get_config_dict") else {}
    lengths = s0.token_lengths(model.tokenizer, list(texts), MAX_SEQ)
    order = np.argsort(lengths, kind="stable")
    batched = s0.encode_texts(model, [texts[i] for i in order], lengths[order], args)
    single = np.stack([model.encode([texts[i]], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)[0]
                       for i in order])
    cos = (batched * single).sum(1)
    stock = (model.prompts or {}).get("query")
    probe = "laws of lulav"
    v_ours = model.encode([QUERY_PREFIX + probe], normalize_embeddings=True, convert_to_numpy=True)[0]
    v_stock = model.encode([probe], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)[0] \
        if stock else v_ours
    out = {"model": spec["id"], "revision": spec["revision"], "modules": mods,
           "pooling_mode": pooling.get("pooling_mode"), "dim": int(batched.shape[1]),
           "default_prompt_name": getattr(model, "default_prompt_name", None), "stock_query_prompt": stock,
           "cos_ours_vs_stock_query_prompt": round(float(v_ours @ v_stock), 4),
           "batching_invariance": {"n": len(texts), "min_cos": round(float(cos.min()), 6),
                                   "max_tokens": int(lengths.max())},
           "norms_ok": bool(np.allclose(np.linalg.norm(batched, axis=1), 1, atol=1e-3))}
    out["ok"] = bool(out["dim"] == 1024 and out["default_prompt_name"] is None and mods[-1] == "Normalize"
                     and (out["pooling_mode"] in (None, "lasttoken")) and cos.min() >= args.canary_min_cos
                     and out["norms_ok"])
    log(f"model checks: {out}")
    return out


def encode_queries(model, texts: List[str], args: argparse.Namespace) -> np.ndarray:
    """Encode queries with the explicit production prefix, length-sorted in bounded batches.

    :param model: Loaded model.
    :param texts: Raw query texts.
    :param args: CLI args (batch bounds).
    :returns: (n, d) normalised vectors in the order of ``texts``.
    :rtype: np.ndarray
    """
    full = [QUERY_PREFIX + t for t in texts]
    lengths = s0.token_lengths(model.tokenizer, full, MAX_SEQ)
    order = np.argsort(lengths, kind="stable")
    enc = s0.encode_texts(model, [full[j] for j in order], lengths[order], args)
    mat = np.empty_like(enc)
    mat[order] = enc
    return mat


# ---------------------------------------------------------------------------------------------------------------
# embed phase
# ---------------------------------------------------------------------------------------------------------------
def stored_base_queries() -> Dict[str, np.ndarray]:
    """The base query vectors stored by the step-0 run (production prefix; all eval_final query texts).

    :returns: ``text -> vector``.
    :rtype: Dict[str, np.ndarray]
    """
    stored = json.loads((BASE_SEM_QUERIES / "queries.json").read_text(encoding="utf-8"))
    if stored["prefix"] != QUERY_PREFIX:
        raise ValueError(f"stored base queries use a different prefix: {stored['prefix']!r}")
    return dict(zip(stored["texts"], np.load(BASE_SEM_QUERIES / "queries.npy")))


def stored_base_query_check(texts: List[str], mat: np.ndarray) -> dict:
    """Compare fresh base query vectors with the stored ones of the step-0 run (same model and prefix).

    :param texts: Query texts encoded now.
    :param mat: Their base vectors.
    :returns: ``{n_overlap, min_cos, ok}``.
    :rtype: dict
    """
    stored = stored_base_queries()
    pairs = [(i, stored[t]) for i, t in enumerate(texts) if t in stored]
    if not pairs:
        return {"ok": False, "n_overlap": 0}
    cos = np.array([float(mat[i] @ v) for i, v in pairs])
    return {"n_overlap": len(pairs), "min_cos": round(float(cos.min()), 6), "mean_cos": round(float(cos.mean()), 6),
            "ok": bool(cos.min() >= 0.999)}


def smoke_rows(ev: dict, n: int, seed: int = 0) -> set:
    """Doc ids for a smoke run: known-item targets and focus carriers first, then random fill.

    :param ev: Loaded eval dir.
    :param n: Target size.
    :param seed: RNG seed.
    :returns: Doc-id set.
    :rtype: set
    """
    pool = [r["doc_id"] for r in ev["eval_pool"]]
    in_pool = set(pool)
    keep = {q["doc_id"] for q in ev["known_item"][:40]} & in_pool
    for sid, e in ev["subject_queries"]["subjects"].items():
        if e["status"] == "focus":
            keep |= set(ev["subject_relevance"][sid]["strong"][:15]) & in_pool
    rng = random.Random(seed)
    rest = [d for d in pool if d not in keep]
    rng.shuffle(rest)
    return keep | set(rest[:max(0, n - len(keep))])


def run_embed(args: argparse.Namespace) -> None:
    """Embed queries (both models) and the eligible pool (tuned model), resumably.

    :param args: Parsed CLI args.
    """
    os.environ.setdefault("HF_HOME", str(AUDIT_ROOT / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    if not os.environ.get("AUDIT_THREADS"):
        torch.set_num_threads(args.cpu_threads)
    ev = eval_v3.load_eval_dir(EVAL_DIR)
    calib = SMOKE_DIR / "queries" if args.smoke else CALIB_DIR
    vec_out = SMOKE_DIR / "vectors" if args.smoke else TUNED_VECS
    calib.mkdir(parents=True, exist_ok=True)
    vec_out.mkdir(parents=True, exist_ok=True)
    texts = all_query_texts(ev)
    if args.smoke:
        texts = list(dict.fromkeys(texts[:200] + [q["text"] for q in load_ood()]))
    qjson = calib / "queries.json"
    if qjson.exists() and json.loads(qjson.read_text(encoding="utf-8"))["texts"] != texts:
        raise SystemExit(f"{qjson} holds a different query list; remove the {calib} query files to start over")
    s0.write_json(qjson, {"prefix": QUERY_PREFIX, "texts": texts, "n_ood": len(load_ood()), "eval_dir": str(EVAL_DIR)})
    rows, doc_texts = s0.load_semantic(SEMANTIC_CORPUS, smoke_rows(ev, args.smoke) if args.smoke else None)
    del ev
    checks_path = vec_out / "checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    probe_docs = [doc_texts[h] for h in sorted(doc_texts)[:: max(1, len(doc_texts) // 24)][:24]]

    lock = s0.acquire_lock(calib)
    try:
        if not (calib / "queries_base.npy").exists():
            model = load_model(BASE, args.dtype, args.pad_multiple, args.smoke_layers)
            t0 = time.time()
            mat = encode_queries(model, texts, args)
            checks["base_queries_vs_stored"] = stored_base_query_check(texts, mat) if not args.smoke_layers \
                else {"skipped": "truncated smoke model"}
            log(f"base queries: {len(texts)} in {time.time() - t0:.0f} s; check {checks['base_queries_vs_stored']}")
            if not args.smoke_layers and not checks["base_queries_vs_stored"]["ok"]:
                raise SystemExit("base query vectors do not reproduce the stored production-contract vectors")
            s0.save_npy(calib / "queries_base.npy", mat)
            s0.write_json(checks_path, checks)
            free_model(model)
        meta_done = (vec_out / "meta.json").exists() and json.loads((vec_out / "meta.json").read_text()).get("complete")
        if (calib / "queries_tuned.npy").exists() and meta_done:
            log("everything already embedded")
            return
    finally:
        lock.unlink(missing_ok=True)

    model = load_model(TUNED, args.dtype, args.pad_multiple, args.smoke_layers)
    if "tuned_model" not in checks:
        checks["tuned_model"] = model_checks(model, TUNED, args, probe_docs)
        s0.write_json(checks_path, checks)
    if not checks["tuned_model"]["ok"]:
        raise SystemExit(f"tuned model checks failed: {checks['tuned_model']}")
    lock = s0.acquire_lock(calib)
    try:
        if not (calib / "queries_tuned.npy").exists():
            t0 = time.time()
            s0.save_npy(calib / "queries_tuned.npy", encode_queries(model, texts, args))
            log(f"tuned queries: {len(texts)} in {time.time() - t0:.0f} s")
    finally:
        lock.unlink(missing_ok=True)
    meta_path = vec_out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not (meta.get("complete") and meta.get("n_records") == len(rows)):
        lock = s0.acquire_lock(vec_out)
        t0 = time.time()
        plan = s0.embed_unique(SimpleNamespace(model=model), doc_texts, vec_out, args)  # exits 99 above soft limit
        s0.expand(vec_out, rows, plan)
        meta = {"complete": True, "written": datetime.now().isoformat(timespec="seconds"),
                "model": {k: TUNED[k] for k in ("id", "revision")}, "dtype": args.dtype, "max_seq": MAX_SEQ,
                "document_prefix": "", "query_prefix": QUERY_PREFIX, "normalised": True,
                "text": str(SEMANTIC_CORPUS), "n_records": len(rows), "n_unique_texts": len(plan["hashes"]),
                "n_tokens": int(sum(plan["lengths"])), "n_shards_unique": len(plan["bounds"]),
                "device_last_session": str(model.device), "seconds_last_session": round(time.time() - t0, 1),
                "batching": {k: getattr(args, k) for k in ("pad_multiple", "max_batch", "max_batch_tokens",
                                                           "max_attn", "shard_size", "shard_tokens")},
                "smoke": bool(args.smoke), "smoke_layers": args.smoke_layers}
        s0.write_json(meta_path, meta)
        lock.unlink(missing_ok=True)
        log(f"expanded {len(plan['hashes'])} unique vectors to {len(rows)} records in {vec_out}")
    # tuned eval-query vectors next to the doc vectors (the layout step0 / eval runs read)
    ood = {o["text"] for o in load_ood()}
    keep = [i for i, t in enumerate(texts) if t not in ood]
    qmat = np.load(calib / "queries_tuned.npy")
    s0.save_npy(vec_out / "queries.npy", qmat[keep])
    s0.write_json(vec_out / "queries.json", {"task": DEFAULT_TASK, "prefix": QUERY_PREFIX,
                                             "texts": [texts[i] for i in keep]})
    free_model(model)
    log("embed phase done")


# ---------------------------------------------------------------------------------------------------------------
# contract phase: the local tuned vectors must reproduce the Colab eval of the same model
# ---------------------------------------------------------------------------------------------------------------
def headline(res: dict) -> Dict[str, float]:
    """The numbers compared against Colab.

    :param res: ``eval_v3.evaluate`` result (or the Colab JSON of one).
    :returns: Flat metric dict.
    :rtype: Dict[str, float]
    """
    ki, sub = res["known_item"]["all"], res["subjects"]
    return {"known_item_MRR": ki["MRR"], "known_item_R@10": ki["R@10"], "known_item_R@100": ki["R@100"],
            "focus_AP": sub["focus"]["macro"]["AP"], "focus_P@10": sub["focus"]["macro"]["P@10"],
            "seen_AP": sub["seen"]["macro"]["AP"]}


def run_contract(args: argparse.Namespace) -> dict:
    """Score the local tuned vectors (and today's base vectors) with eval_v3 and compare with Colab.

    :param args: Parsed CLI args (``tolerance``).
    :returns: Comparison dict (also written to ``TUNED_VECS/contract_vs_colab.json``).
    :rtype: dict
    """
    q = json.loads((CALIB_DIR / "queries.json").read_text(encoding="utf-8"))
    out = {}
    for name, vec_dir, qfile, colab in (
            ("tuned_semantic", TUNED_VECS, "queries_tuned.npy", "tuned_v3-final-run1.json"),
            ("base_original", BASE_ORIG_VECS, "queries_base.npy", "base_original.json")):
        qvec = dict(zip(q["texts"], np.load(CALIB_DIR / qfile)))
        ids, mat = load_vectors(vec_dir)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            res = eval_v3.evaluate(ids, mat, lambda ts: np.stack([qvec[t] for t in ts]), EVAL_DIR,
                                   sections=("subjects", "known_item"))
        del ids, mat
        ref = json.loads((COLAB_EVAL / colab).read_text())
        mine, theirs = headline(res), headline(ref)
        diff = {k: round(mine[k] - theirs[k], 4) for k in mine}
        out[name] = {"local": mine, "colab": theirs, "diff": diff, "pool": res["pool"],
                     "ok": bool(max(abs(v) for v in diff.values()) <= args.tolerance
                                and not res["pool"]["n_missing_vectors"])}
        log(f"contract {name}: {out[name]}")
    out["tolerance"] = args.tolerance
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    s0.write_json(TUNED_VECS / "contract_vs_colab.json", out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------
def pct(x: Sequence[float], qs: Sequence[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> Dict[str, float]:
    """Distribution summary.

    :param x: Values.
    :param qs: Percentiles.
    :returns: ``{n, mean, std, min, max, pXX...}``.
    :rtype: Dict[str, float]
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    out = {"n": int(len(a)), "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
           "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}
    out.update({f"p{q:g}": round(float(np.percentile(a, q)), 4) for q in qs})
    return out


def build_jobs(ev: dict, pool_pos: Dict[str, int]) -> List[dict]:
    """One job per scored query: group, text, relevant pool indices, indices to remove for the unanswerable variant.

    Groups: ``known_item`` (clean: never a training anchor, targets held out), ``focus_<source>`` (the six focus
    subjects: trained on, queries kept out of anchors), ``seen_name`` (in-distribution training labels), ``ood_<group>``.
    Relevant = the target's identical-text twins (known items) / strong carriers in the pool (subjects; for focus
    subjects only the held-out carriers, as eval_v3 scores them). Removed = twins (known items) / every strong, weak
    and unjudged carrier of the subject (subjects): the same query against a pool without its relevant records.

    :param ev: Loaded eval dir.
    :param pool_pos: ``doc_id -> pool row``.
    :returns: Job dicts.
    :rtype: List[dict]
    """
    hash_of = {r["doc_id"]: r["text_hash"] for r in ev["eval_pool"]}
    twins: Dict[str, List[int]] = defaultdict(list)
    for r in ev["eval_pool"]:
        if r["doc_id"] in pool_pos:
            twins[r["text_hash"]].append(pool_pos[r["doc_id"]])
    held = set(ev["held_out_records"]["held_out"])
    jobs = []
    for q in ev["known_item"]:
        if q.get("shelfmark_query") or q["doc_id"] not in pool_pos:
            continue
        hits = np.array(twins[hash_of[q["doc_id"]]], dtype=np.int64)
        jobs.append({"group": "known_item", "sub": q["type"], "text": q["text"], "rel": hits, "remove": hits})
    for sid, e in ev["subject_queries"]["subjects"].items():
        r = ev["subject_relevance"].get(sid, {})
        strong = [d for d in r.get("strong", []) if d in pool_pos]
        rel = [pool_pos[d] for d in strong if (e["status"] != "focus" or d in held)]
        if len(rel) < eval_v3.MIN_RELEVANT:
            continue
        remove = np.array(sorted({pool_pos[d] for d in r.get("strong", []) + r.get("weak", []) + r.get("unjudged", [])
                                  if d in pool_pos}), dtype=np.int64)
        for qq in e["queries"]:
            jobs.append({"group": f"{e['status']}_{qq['source']}", "sub": sid, "text": qq["text"],
                         "rel": np.array(sorted(rel), dtype=np.int64), "remove": remove})
    for q in load_ood():
        jobs.append({"group": f"ood_{q['group']}", "sub": q["lang"], "text": q["text"], "rel": None, "remove": None})
    return jobs


def per_query_stats(P: np.ndarray, Q: np.ndarray, jobs: List[dict], seed: int = 0, n_random: int = 200) -> dict:
    """Score every job's query against the pool and collect the per-query statistics.

    :param P: (N, d) normalised pool.
    :param Q: (n_jobs, d) normalised query vectors, aligned with ``jobs``.
    :param jobs: From :func:`build_jobs`.
    :param seed: RNG seed for random query-doc pairs.
    :param n_random: Random pool docs per query for the random-pair distribution.
    :returns: Arrays keyed by statistic (aligned with ``jobs``) plus ``random_pairs``.
    :rtype: dict
    """
    n = len(jobs)
    N = P.shape[0]
    keys = ("top1", "top10", "top100", "mean_top10", "median", "std", "best_rel", "rank_rel", "top1_removed",
            "top10_removed")
    out = {k: np.full(n, np.nan) for k in keys}
    rng = np.random.default_rng(seed)
    rand = []
    for start in range(0, n, CHUNK):
        S = Q[start:start + CHUNK] @ P.T
        b = len(S)
        part = -np.partition(-S, (0, 9, 99), axis=1)
        out["top1"][start:start + b] = part[:, 0]
        out["top10"][start:start + b] = part[:, 9]
        out["top100"][start:start + b] = part[:, 99]
        out["mean_top10"][start:start + b] = part[:, :10].mean(1)
        out["median"][start:start + b] = np.median(S, axis=1)
        out["std"][start:start + b] = S.std(axis=1)
        rand.append(S[np.arange(b)[:, None], rng.integers(0, N, size=(b, n_random))].ravel())
        for i, s in enumerate(S):
            j = jobs[start + i]
            if j["rel"] is not None:
                best = s[j["rel"]].max()
                out["best_rel"][start + i] = best
                out["rank_rel"][start + i] = int((s > best).sum()) + 1
            if j["remove"] is not None:
                s2 = s.copy()
                s2[j["remove"]] = -np.inf
                p2 = -np.partition(-s2, (0, 9))
                out["top1_removed"][start + i] = p2[0]
                out["top10_removed"][start + i] = p2[9]
    out["random_pairs"] = np.concatenate(rand) if rand else np.zeros(0)
    return out


def doc_doc_random(P: np.ndarray, n_pairs: int = 200_000, seed: int = 1) -> np.ndarray:
    """Cosine of random distinct document pairs (anisotropy of the document space).

    :param P: (N, d) normalised pool.
    :param n_pairs: Number of pairs.
    :param seed: RNG seed.
    :returns: Cosines.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(P), n_pairs), rng.integers(0, len(P), n_pairs)
    keep = a != b
    return np.einsum("ij,ij->i", P[a[keep]], P[b[keep]])


def pass_rate(x: np.ndarray, t: float) -> float:
    """Share of queries whose top-1 cosine clears ``t`` (a gate passes when any result is >= t).

    :param x: Top-1 cosines.
    :param t: Threshold.
    :returns: Share in [0, 1].
    :rtype: float
    """
    return float((x >= t).mean()) if len(x) else float("nan")


def match_threshold(base_vals: np.ndarray, tuned_vals: np.ndarray, t_base: float) -> dict:
    """The tuned threshold that keeps the same pass rate as ``t_base`` gives the base model on the same queries.

    :param base_vals: Base top-1 cosines.
    :param tuned_vals: Tuned top-1 cosines of the same queries.
    :param t_base: Base threshold.
    :returns: ``{base_pass_rate, tuned_threshold, tuned_interval}`` (the interval of tuned thresholds that reject
        exactly as many queries; open-ended when the base pass rate is 0 or 1).
    :rtype: dict
    """
    rate = pass_rate(base_vals, t_base)
    srt = np.sort(tuned_vals)
    n = len(srt)
    k = int(round((1 - rate) * n))  # number of queries to reject
    lo = float(srt[k - 1]) if k > 0 else float("-inf")
    hi = float(srt[k]) if k < n else float("inf")
    mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (hi if np.isfinite(hi) else lo)
    return {"base_pass_rate": round(rate, 4), "n": n, "tuned_threshold": round(mid, 4),
            "tuned_interval": [round(lo, 4) if np.isfinite(lo) else None, round(hi, 4) if np.isfinite(hi) else None]}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random answerable query's top-1 exceeds a random unanswerable one's (ties count half).

    :param pos: Answerable values.
    :param neg: Unanswerable values.
    :returns: ROC AUC.
    :rtype: float
    """
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_balanced(pos: np.ndarray, neg: np.ndarray, grid: np.ndarray) -> dict:
    """Threshold maximising balanced accuracy (answerable pass rate + unanswerable reject rate) / 2.

    :param pos: Answerable top-1 values.
    :param neg: Unanswerable top-1 values.
    :param grid: Candidate thresholds.
    :returns: ``{threshold, answerable_pass, unanswerable_reject, balanced_accuracy}``.
    :rtype: dict
    """
    ba = [(pass_rate(pos, t) + 1 - pass_rate(neg, t)) / 2 for t in grid]
    i = int(np.argmax(ba))
    return {"threshold": round(float(grid[i]), 3), "answerable_pass": round(pass_rate(pos, grid[i]), 4),
            "unanswerable_reject": round(1 - pass_rate(neg, grid[i]), 4), "balanced_accuracy": round(ba[i], 4)}


def query_vectors(model: str, calib: Path) -> Dict[str, np.ndarray]:
    """``text -> vector`` for one model: the embed phase's file when present, else (base) the stored step-0 vectors.

    :param model: ``base`` or ``tuned``.
    :param calib: Query dir of the embed phase.
    :returns: Mapping (empty when the model's queries are not embedded yet).
    :rtype: Dict[str, np.ndarray]
    """
    path = calib / f"queries_{model}.npy"
    if path.exists():
        texts = json.loads((calib / "queries.json").read_text(encoding="utf-8"))["texts"]
        return dict(zip(texts, np.load(path)))
    return stored_base_queries() if model == "base" else {}


def rrf_top(lexical_ids: Sequence[str], semantic_ids: Sequence[str], sw: float, kw: float, n: int) -> List[tuple]:
    """Production weighted RRF of ``search_bibliography.search_hybrid``, normalised so rank 1 in both legs is 1.0.

    :param lexical_ids: Keyword-leg ranking (already cut to the candidate size).
    :param semantic_ids: Semantic-leg ranking (already cut to the candidate size).
    :param sw: Semantic weight in [0, 1].
    :param kw: Keyword weight in [0, 1].
    :param n: ``num_results``.
    :returns: Top ``n`` ``(id, similarity_score)``.
    :rtype: List[tuple]
    """
    fused: Dict[str, float] = defaultdict(float)
    for weight, ranking in ((kw, lexical_ids), (sw, semantic_ids)):
        if weight > 0:
            for rank, doc in enumerate(ranking, start=1):
                fused[doc] += weight / (RRF_K + rank)
    top = sorted(fused.items(), key=lambda item: -item[1])[:n]
    return [(doc, score * (RRF_K + 1.0)) for doc, score in top]


def author_constrained(cos: np.ndarray, candidates: np.ndarray, book_of: Sequence[str], n: int = 8) -> List[tuple]:
    """``search_by_author``: best page per book (collapse, 4 inner pages), breadth-first, score ``(cos + 1) / 2``.

    :param cos: Cosine of every page to the query.
    :param candidates: Page rows that pass the author filter.
    :param book_of: Book title per page row (the collapse key, ``title.keyword``).
    :param n: ``num_results``.
    :returns: ``(row, similarity_score)`` in production order.
    :rtype: List[tuple]
    """
    by_book: Dict[str, List[int]] = defaultdict(list)
    for row in candidates[np.argsort(-cos[candidates])]:
        if len(by_book[book_of[row]]) < 4:
            by_book[book_of[row]].append(int(row))
    books = sorted(by_book.values(), key=lambda rows: -cos[rows[0]])[:n]
    hits: List[int] = []
    for depth in range(4):
        for rows in books:
            if depth < len(rows) and len(hits) < n:
                hits.append(rows[depth])
    return [(row, (float(cos[row]) + 1.0) / 2.0) for row in hits]


def bibliography_geometry() -> Optional[dict]:
    """Base vs tuned cosine geometry on the BIBLIOGRAPHY index, plus the two places where it can reach the chat.

    Uses the page and query vectors that ``bib_v9_check.py`` embedded with both models under the production contract
    (``BIB_DIR/vectors/{base,tuned}``) and its read-only production keyword rankings (``BIB_DIR/lexical.json``).

    * ``gate``: the 0.4 gate can only fire when the keyword leg is empty (router plans are 30/70); share of queries
      with no lexical hit.
    * ``author_merge``: a graph_scholar plan combined with one 30/70 ``bibliography_hybrid`` action (5 results):
      author-constrained pages ``(cos+1)/2`` and hybrid pages (RRF) are sorted together and cut to 8.

    :returns: The section, or ``None`` when either model's vectors are missing.
    :rtype: Optional[dict]
    """
    vec_dirs = {m: BIB_DIR / "vectors" / m for m in ("base", "tuned")}
    if not all((d / "meta.json").exists() and json.loads((d / "meta.json").read_text()).get("complete")
               for d in vec_dirs.values()):
        return None
    ids = json.loads((vec_dirs["base"] / "ids.json").read_text())
    assert ids == json.loads((vec_dirs["tuned"] / "ids.json").read_text()), "page order differs between models"
    texts = json.loads((vec_dirs["base"] / "queries.json").read_text())["texts"]
    assert texts == json.loads((vec_dirs["tuned"] / "queries.json").read_text())["texts"], "query order differs"
    page = {}
    with open(BIB_DIR / "pages.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            page[r["es_id"]] = r
    book_of = [page[d]["book"] for d in ids]
    authors_of = [set(page[d].get("authors") or []) for d in ids]
    row_of = {d: i for i, d in enumerate(ids)}
    q_of = {t: i for i, t in enumerate(texts)}
    lexical = json.loads((BIB_DIR / "lexical.json").read_text(encoding="utf-8"))
    ki = [json.loads(line) for line in BIB_KI.read_text(encoding="utf-8").splitlines() if line.strip()]
    ki = [q for q in ki if q["text"] in q_of and q["es_id"] in row_of]
    n_lex = np.array([len(lexical.get(t) or []) for t in texts])
    out = {"source": {"vectors": str(BIB_DIR / "vectors"), "lexical": str(BIB_DIR / "lexical.json"),
                      "known_item_queries": str(BIB_KI), "written_by": "bib_v9_check.py (same contract as this "
                      "module: documents raw, production query prefix, fp32, max_seq 8192, normalised)"},
           "n_pages": len(ids), "n_queries": len(texts), "n_known_item_queries": len(ki),
           "gate": {"share_queries_without_lexical_hit": round(float((n_lex == 0).mean()), 4),
                    "examples_without_lexical_hit": [t for t, k in zip(texts, n_lex) if k == 0][:12],
                    "note": "only these can drop a 30/70 plan below 0.4 (top = 0.3); the router's keyword_weight=70 "
                            "then triggers the 100/0 semantic fallback (top = 1.0), so the gate swaps the evidence "
                            "for a pure-semantic top 5 and never yields NO_RELEVANT_SOURCES"}}
    geo, merge = {}, {}
    rng = np.random.default_rng(7)
    a, b = rng.integers(0, len(ids), 200_000), rng.integers(0, len(ids), 200_000)
    keep = a != b
    for m, d in vec_dirs.items():
        P = eval_v3.normalise_rows(np.load(d / "pages.npy").astype(np.float32))
        Q = eval_v3.normalise_rows(np.load(d / "queries.npy").astype(np.float32))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious Accelerate matmul warnings
            S = Q @ P.T
        top = -np.sort(-S, axis=1)[:, :10]
        ki_rows = np.array([q_of[q["text"]] for q in ki])
        geo[m] = {"all_query_page_pairs": pct(S.ravel()),
                  "page_page_random": pct(np.einsum("ij,ij->i", P[a[keep]], P[b[keep]])),
                  "top1_all_queries": pct(top[:, 0]), "top8_all_queries": pct(top[:, 7]),
                  "top1_minus_top10_all_queries": pct(top[:, 0] - top[:, 9]),
                  "known_item_top1": pct(top[ki_rows, 0]),
                  "known_item_target": pct(np.array([S[q_of[q["text"]], row_of[q["es_id"]]] for q in ki])),
                  "raw_cos_0.4_percentile_of_all_pairs": round(float((S.ravel() < PROD_THRESHOLD).mean() * 100), 2)}
        n_author, target_in, author_top = [], [], []
        for q in ki:
            t_row = row_of[q["es_id"]]
            if not authors_of[t_row]:
                continue
            cand = np.array([i for i, au in enumerate(authors_of) if au & authors_of[t_row]])
            cos = S[q_of[q["text"]]]
            semantic = [ids[i] for i in np.argsort(-cos)[:40]]   # candidate_size for num_results 5 = 40
            hybrid = rrf_top((lexical.get(q["text"]) or [])[:40], semantic, ROUTER_SW, ROUTER_KW, 5)
            author = [(ids[r], s) for r, s in author_constrained(cos, cand, book_of)]
            best: Dict[str, tuple] = {}
            for src, rows in (("hybrid", hybrid), ("author", author)):
                for doc, score in rows:
                    if doc not in best or score > best[doc][0]:
                        best[doc] = (score, src)
            kept = sorted(best.items(), key=lambda item: -item[1][0])[:EVIDENCE_LIMIT]
            n_author.append(sum(1 for _, (_, src) in kept if src == "author"))
            target_in.append(any(doc == q["es_id"] for doc, _ in kept))
            author_top.append(author[0][1])
        n_author_arr = np.array(n_author)
        merge[m] = {"n_queries": len(n_author), "author_pages_in_top8": pct(n_author_arr, (5, 25, 50, 75, 95)),
                    "share_with_at_most_3_author_pages": round(float((n_author_arr <= 3).mean()), 4),
                    "target_page_in_top8": round(float(np.mean(target_in)), 4),
                    "best_author_page_score_(cos+1)/2": pct(author_top)}
    out["geometry"] = geo
    out["author_merge_sim"] = {"setup": "graph_scholar author = the known-item target page's author(s); one 30/70 "
                                        "bibliography_hybrid action (5 results, candidate size 40) with production "
                                        "keyword ranks; merge by max score, sort, keep 8",
                               **merge}
    return out


def analyze(args: argparse.Namespace) -> dict:
    """Compute every distribution, gate curve and mapping that the available vectors allow; write the report.

    :param args: Parsed CLI args (``smoke``, ``identity_test``, ``out``).
    :returns: The report.
    :rtype: dict
    """
    ev = eval_v3.load_eval_dir(EVAL_DIR)
    calib = SMOKE_DIR / "queries" if args.smoke else CALIB_DIR
    tuned_dir = SMOKE_DIR / "vectors" if args.smoke else TUNED_VECS
    tuned_ready = (tuned_dir / "meta.json").exists() and json.loads((tuned_dir / "meta.json").read_text()).get("complete")
    qv = {"base": query_vectors("base", calib)}
    if args.identity_test:
        qv["tuned"] = qv["base"]
    elif tuned_ready:
        qv["tuned"] = query_vectors("tuned", calib)
    models = [m for m in ("base", "tuned") if qv.get(m)]
    pool_ids = [r["doc_id"] for r in ev["eval_pool"]]
    if "tuned" in models and not args.identity_test:
        ids_t, mat_t = load_vectors(tuned_dir)
        have = set(ids_t)
        pool_ids = [d for d in pool_ids if d in have]
    pool_pos = {d: i for i, d in enumerate(pool_ids)}
    ids_b, mat_b = load_vectors(BASE_ORIG_VECS)
    row_b = {d: i for i, d in enumerate(ids_b)}
    P = {"base": eval_v3.normalise_rows(mat_b[[row_b[d] for d in pool_ids]])}
    del mat_b
    if "tuned" in models:
        if args.identity_test:
            P["tuned"] = P["base"]
        else:
            row_t = {d: i for i, d in enumerate(ids_t)}
            P["tuned"] = eval_v3.normalise_rows(mat_t[[row_t[d] for d in pool_ids]])
            del mat_t
    jobs = [j for j in build_jobs(ev, pool_pos) if all(j["text"] in qv[m] for m in models)]
    log(f"models {models}, pool {len(pool_ids)}, scored queries {len(jobs)}")
    stats, docdoc = {}, {}
    for m in models:
        Q = eval_v3.normalise_rows(np.stack([qv[m][j["text"]] for j in jobs]))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            stats[m] = per_query_stats(P[m], Q, jobs)
            docdoc[m] = doc_doc_random(P[m])
        log(f"{m}: per-query stats done")
    groups = np.array([j["group"] for j in jobs]).astype(str)
    ood = np.char.startswith(groups, "ood_")
    kinds = {"answerable_known_item": groups == "known_item",
             "answerable_focus": np.char.startswith(groups, "focus_"),
             "answerable_seen_name_in_distribution": groups == "seen_name",
             "unanswerable_ood": ood}
    kinds["answerable_clean"] = kinds["answerable_known_item"] | kinds["answerable_focus"]
    kinds["answerable_all"] = kinds["answerable_clean"] | kinds["answerable_seen_name_in_distribution"]
    for g in sorted(set(groups)):
        kinds[f"group:{g}"] = groups == g
    for lang in ("en", "he"):
        kinds[f"unanswerable_ood_{lang}"] = ood & np.array([j["sub"] == lang for j in jobs])
    removed = np.array([j["remove"] is not None for j in jobs])
    subj = kinds["answerable_focus"] | kinds["answerable_seen_name_in_distribution"]

    def vals(m: str, stat: str, mask: np.ndarray) -> np.ndarray:
        """Finite values of one statistic over a job mask.

        :param m: Model key.
        :param stat: Statistic key.
        :param mask: Job mask.
        :returns: Values.
        """
        v = stats[m][stat][mask]
        return v[np.isfinite(v)]

    unans = {"ood": ("top1", ood),
             "subject_queries_carriers_removed": ("top1_removed", subj & removed),
             "focus_queries_carriers_removed": ("top1_removed", kinds["answerable_focus"] & removed),
             "known_item_target_removed": ("top1_removed", kinds["answerable_known_item"] & removed)}
    unans = {u: v for u, v in unans.items() if v[1].any()}
    ans = {"known_item": kinds["answerable_known_item"], "focus": kinds["answerable_focus"],
           "clean (known_item + focus)": kinds["answerable_clean"],
           "seen_name (in-distribution)": kinds["answerable_seen_name_in_distribution"],
           "all answerable": kinds["answerable_all"]}

    dist = {}
    for m in models:
        d = {"random_query_doc_pairs": pct(stats[m]["random_pairs"]), "random_doc_doc_pairs": pct(docdoc[m])}
        for name, mask in kinds.items():
            if mask.any():
                d[name] = {s: pct(vals(m, s, mask)) for s in ("top1", "top10", "best_rel", "top1_removed", "median")
                           if len(vals(m, s, mask))}
        d["rank_of_best_relevant"] = {name: pct(vals(m, "rank_rel", mask), (25, 50, 75, 90)) for name, mask in ans.items()}
        # When the relevant record is rarely the top-1, "answerable top-1" and "relevant-removed top-1" are nearly the
        # same number (removing a record ranked ~100th cannot change the top-1), so their AUC says little about
        # absence detection; this share makes that visible.
        d["share_best_relevant_at_rank1"] = {name: round(float((vals(m, "rank_rel", mask) == 1).mean()), 4)
                                             for name, mask in ans.items()}
        dist[m] = d

    grid = np.round(np.arange(-0.05, 0.951, 0.005), 3)
    curves = {"thresholds": grid.tolist()}
    for m in models:
        curves[m] = {**{f"pass:{a}": [round(pass_rate(vals(m, "top1", mask), t), 4) for t in grid]
                        for a, mask in ans.items()},
                     **{f"reject:{u}": [round(1 - pass_rate(vals(m, stat, mask), t), 4) for t in grid]
                        for u, (stat, mask) in unans.items()}}

    separation = {}
    for m in models:
        separation[m] = {}
        for a in ("known_item", "clean (known_item + focus)", "all answerable"):
            for u, (stat, mask) in unans.items():
                pos, neg = vals(m, "top1", ans[a]), vals(m, stat, mask)
                separation[m][f"{a} vs {u}"] = {
                    "auc": round(auc(pos, neg), 4), "best": best_balanced(pos, neg, grid),
                    f"at_{PROD_THRESHOLD}": {"answerable_pass": round(pass_rate(pos, PROD_THRESHOLD), 4),
                                             "unanswerable_reject": round(1 - pass_rate(neg, PROD_THRESHOLD), 4)}}

    spread, hybrid_rows = {}, []
    answer = kinds["answerable_all"]
    for m in models:
        s = stats[m]
        spread[m] = {"top1_minus_top10": pct((s["top1"] - s["top10"])[answer], (25, 50, 75)),
                     "top1_minus_top100": pct((s["top1"] - s["top100"])[answer], (25, 50, 75)),
                     "top1_minus_median": pct((s["top1"] - s["median"])[answer], (25, 50, 75)),
                     "pool_std": pct(s["std"][answer], (25, 50, 75)),
                     "best_relevant_minus_median": pct((s["best_rel"] - s["median"])[answer], (25, 50, 75))}
    for sw, kw in HYBRID_WEIGHTS:
        bonus = kw / sw
        row = {"semantic": sw, "keyword": kw, "keyword_bonus_in_cosine_units": round(bonus, 4)}
        for m in models:
            s = stats[m]
            row[f"{m}_share_top1_minus_top100_gt_bonus"] = round(float(((s["top1"] - s["top100"])[answer] > bonus).mean()), 4)
            row[f"{m}_share_top1_minus_median_gt_bonus"] = round(float(((s["top1"] - s["median"])[answer] > bonus).mean()), 4)
        hybrid_rows.append(row)
    hybrid = {"formula": PRODUCTION_USAGE["primary_hybrid"]["formula"], "spread": spread, "weights": hybrid_rows}

    report = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete" if "tuned" in models and not args.identity_test else
                  ("identity_test" if args.identity_test else "base_only: v9 (tuned) vectors not embedded yet"),
        "smoke": bool(args.smoke),
        "production_usage": PRODUCTION_USAGE,
        "models": {"base": {**{k: BASE[k] for k in ("id", "revision")}, "text": "original production text",
                            "doc_vectors": str(BASE_ORIG_VECS),
                            "query_vectors": str(calib / "queries_base.npy") if (calib / "queries_base.npy").exists()
                            else str(BASE_SEM_QUERIES / "queries.npy")},
                   "tuned": {**{k: TUNED[k] for k in ("id", "revision")}, "text": "v3 semantic text",
                             "doc_vectors": str(tuned_dir), "available": "tuned" in models},
                   "query_prefix": QUERY_PREFIX, "max_seq": MAX_SEQ, "normalised": True},
        "pool": {"eval_dir": str(EVAL_DIR), "n": len(pool_ids)},
        "queries": {g: int((groups == g).sum()) for g in sorted(set(groups))},
        "distributions": dist,
        "separation": separation,
        "gate_curves": curves,
        "hybrid": hybrid,
    }
    if not args.smoke:
        bib = bibliography_geometry()
        report["bibliography_index"] = bib if bib is not None else "pending: v9_biblio vectors for both models missing"
        if bib is not None:
            log("bibliography index geometry done")
    if "tuned" in models:
        mapping = {"production_threshold_base": PROD_THRESHOLD, "same_answerable_pass_rate": {},
                   "same_unanswerable_reject_rate": {}}
        for a, mask in ans.items():
            mapping["same_answerable_pass_rate"][a] = match_threshold(vals("base", "top1", mask),
                                                                      vals("tuned", "top1", mask), PROD_THRESHOLD)
        for u, (stat, mask) in unans.items():
            r = match_threshold(vals("base", stat, mask), vals("tuned", stat, mask), PROD_THRESHOLD)
            r["base_reject_rate"] = round(1 - r.pop("base_pass_rate"), 4)
            mapping["same_unanswerable_reject_rate"][u] = r
        allmask = kinds["answerable_all"] | ood
        mapping["same_percentile_of_all_top1"] = match_threshold(vals("base", "top1", allmask),
                                                                 vals("tuned", "top1", allmask), PROD_THRESHOLD)
        rb, rt = stats["base"]["random_pairs"], stats["tuned"]["random_pairs"]
        z = (PROD_THRESHOLD - rb.mean()) / rb.std()
        mapping["same_z_over_random_pairs"] = {"z": round(float(z), 3),
                                               "tuned_threshold": round(float(rt.mean() + z * rt.std()), 4)}
        report["threshold_mapping"] = mapping
        tradeoff = []
        for t in np.round(np.arange(0.2, 0.801, 0.025), 3):
            row = {"t": float(t)}
            for a in ("known_item", "clean (known_item + focus)"):
                row[f"pass_{a}"] = round(pass_rate(vals("tuned", "top1", ans[a]), t), 4)
            for u, (stat, mask) in unans.items():
                row[f"reject_{u}"] = round(1 - pass_rate(vals("tuned", stat, mask), t), 4)
            tradeoff.append(row)
        report["tuned_tradeoff"] = tradeoff
        ratios = {k: round(spread["tuned"][k]["p50"] / spread["base"][k]["p50"], 3) for k in spread["base"]}
        scale = float(np.median([ratios[k] for k in ("top1_minus_top100", "top1_minus_median", "pool_std")]))
        hybrid["ratio_tuned_over_base_p50"] = ratios
        hybrid["scale_factor"] = round(scale, 3)
        for row in hybrid_rows:  # keep kw/sw constant in units of the cosine spread
            new_ratio = row["keyword_bonus_in_cosine_units"] * scale
            sw_new = 100.0 / (1.0 + new_ratio)
            row["tuned_equivalent"] = {"semantic": round(sw_new, 1), "keyword": round(100 - sw_new, 1)}
    report["per_query"] = {"text": [j["text"] for j in jobs], "group": groups.tolist(),
                           **{f"{m}_{s}": [None if not np.isfinite(v) else round(float(v), 4) for v in stats[m][s]]
                              for m in models for s in ("top1", "best_rel", "rank_rel", "top1_removed")}}
    path = Path(args.out) if args.out else (SMOKE_DIR / "v9_cosine_calibration.json" if args.smoke else RESULTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    s0.write_json(path, report)
    log(f"wrote {path}")
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("embed", "contract", "analyze"))
    parser.add_argument("--smoke", type=int, default=0, help="embed/analyze a ~N-record subset (AUDIT_ROOT/v9_calibration_smoke)")
    parser.add_argument("--smoke-layers", type=int, default=0, help="smoke only: truncate both models to N layers")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--shard-tokens", type=int, default=300_000)
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-attn", type=int, default=1 << 22)
    parser.add_argument("--soft-max-gb", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--canary-min-cos", type=float, default=0.999)
    parser.add_argument("--tolerance", type=float, default=0.01, help="contract: max |local - Colab| per metric")
    parser.add_argument("--identity-test", action="store_true", help="analyze: score base as if it were tuned")
    parser.add_argument("--out", default=None, help="analyze: report path override")
    args = parser.parse_args()
    if args.smoke_layers and not args.smoke:
        raise SystemExit("--smoke-layers is for --smoke runs only")
    if args.phase == "embed":
        run_embed(args)
    elif args.phase == "contract":
        run_contract(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
