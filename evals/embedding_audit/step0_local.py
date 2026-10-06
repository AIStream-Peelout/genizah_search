"""Step 0 (text track): does dropping the ID / shelf-mark lines alone improve the BASE embedder? No training.

Embeds the v3 *semantic* text (``v3/corpus_semantic.jsonl``: production text minus the ``Document ID:`` /
``Shelf Mark:`` lines and editor credits) of every eligible record with the production contract -- base
``Qwen3-Embedding-0.6B`` at its pinned revision (``embed_utils.TEXT_MODELS``), fp32, documents raw (no prefix),
L2-normalised, max_seq 8192 -- and scores it against the existing production-text vectors
(``corpus_vectors/qwen3-0.6b__all__8192``: same model, revision and window) with the frozen v3 eval
(``eval_v3.evaluate``) on the identical eligible pool, using the same query vectors (Qwen3 query instruction,
``embed_utils.DEFAULT_TASK``) for both representations.

Embedding is resumable (the guard may kill and relaunch it at any time):

* every unique text (``text_hash``) is embedded once and mapped back to all records sharing it;
* unique texts are sorted by token length and cut into shards of at most ``--shard-size`` texts and
  ``--shard-tokens`` tokens, written to ``<out>/unique/shard_XXXX.npy`` (tmp + rename). Existing shards are skipped;
  the plan is stored in ``<out>/unique/plan.json`` and checked on resume;
* inside a shard, contiguous batches are bounded by batch size, padded tokens and b * n^2 (attention), and padded
  to a multiple of ``--pad-multiple`` tokens so MPS sees few distinct shapes (padding is masked; the vectors do not
  change -- the contract canary checks this against the stored production vectors);
* after each shard the process measures its own footprint; above ``--soft-max-gb`` it exits with code 99 so the
  wrapper loop relaunches it fresh (the original-text run grew to 8-11 GB on MPS and was killed by the guard,
  whose hard footprint cap does not relaunch);
* finally the per-record matrix is written as ``<out>/ids.json`` + ``<out>/shard_XXXX.npy`` (the
  ``eval_corpus.load_vectors`` layout) plus ``meta.json``.

Outputs: ``AUDIT_ROOT/corpus_vectors/qwen3-0.6b__semantic__8192/`` (vectors, ``queries.{json,npy}``, ``canary.json``,
``eval_v3_{original,semantic}.json`` = full results incl. per-query lists, for later paired comparisons against
tuned models), ``results/step0_original_vs_semantic.json`` (headline numbers + paired bootstrap CIs) and a printed
side-by-side table.

Run (from this directory; ``PY`` = the historical-document-analysis venv python)::

    # smoke: ~200 records, CPU, 3 GB cap; everything goes to AUDIT_ROOT/v3/step0_smoke/
    python3 guard.py --name step0_smoke --small -- $PY step0_local.py --smoke 200 --shard-size 64
    # full run on MPS, relaunched while it exits 99 (soft footprint limit)
    nohup bash -c 'for i in $(seq 0 20); do /usr/bin/python3 guard.py --name gpu_step0_semantic --max-gb 8 -- \\
        $PY step0_local.py --soft-max-gb 6.5; rc=$?; [ $rc -eq 99 ] || break; done' >> AUDIT_ROOT/logs/step0.log 2>&1 &
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import eval_v3
from embed_utils import AUDIT_ROOT, DEFAULT_TASK, TEXT_MODELS, CachedTextEncoder, device
from eval_corpus import load_vectors
from guard import child_rss_gb

HERE = Path(__file__).parent
MODEL_KEY = "qwen3-0.6b"
MAX_SEQ = 8192
SEMANTIC_CORPUS = AUDIT_ROOT / "v3" / "corpus_semantic.jsonl"
ORIGINAL_CORPUS = AUDIT_ROOT / "corpus_v1.jsonl"
EVAL_DIR = AUDIT_ROOT / "v3" / "eval"
ORIGINAL_VECS = AUDIT_ROOT / "corpus_vectors" / f"{MODEL_KEY}__all__{MAX_SEQ}"
OUT_VECS = AUDIT_ROOT / "corpus_vectors" / f"{MODEL_KEY}__semantic__{MAX_SEQ}"
RESULTS = HERE / "results" / "step0_original_vs_semantic.json"
SMOKE_DIR = AUDIT_ROOT / "v3" / "step0_smoke"
RESTART_RC = 99          # "relaunch me": soft footprint limit reached between shards
RECORD_SHARD = 20000     # rows per per-record output shard
CANARY_MIN_COS = 0.999   # re-embedded production text vs the stored production vector


def log(msg: str) -> None:
    """Print a timestamped progress line.

    :param msg: Message text.
    """
    print(f"[step0 {datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _json_default(obj):
    """JSON fallback for numpy scalars and arrays.

    :param obj: Object json cannot serialise.
    :returns: A plain Python equivalent.
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(path: Path, obj) -> None:
    """Write JSON via a temp file and rename (the old file is removed first; SMB-safe).

    :param path: Target path.
    :param obj: JSON-serialisable object.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    path.unlink(missing_ok=True)
    tmp.rename(path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    """Save an array via ``<stem>.tmp.npy`` and rename (``load_vectors`` ignores ``.tmp`` files).

    :param path: Target ``.npy`` path.
    :param arr: Array to save.
    """
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    path.unlink(missing_ok=True)
    tmp.rename(path)


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------
def load_semantic(path: Path, keep: Optional[set] = None) -> Tuple[List[dict], Dict[str, str]]:
    """Load the eligible records of the semantic corpus and their unique texts.

    :param path: ``v3/corpus_semantic.jsonl``.
    :param keep: Optional doc-id subset (smoke test).
    :returns: (rows ``{doc_id, text_hash}`` in corpus order, ``text_hash -> text``).
    :rtype: Tuple[List[dict], Dict[str, str]]
    """
    rows: List[dict] = []
    texts: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["eligible"] or (keep is not None and r["doc_id"] not in keep):
                continue
            h = r["text_hash"]
            if h not in texts:
                if hashlib.md5(r["text"].encode("utf-8")).hexdigest() != h:
                    raise ValueError(f"text_hash does not match the text of {r['doc_id']}")
                texts[h] = r["text"]
            rows.append({"doc_id": r["doc_id"], "text_hash": h})
    return rows, texts


def query_texts(ev: dict) -> List[str]:
    """Every unique query text :func:`eval_v3.evaluate` will ask for, in its order.

    :param ev: Loaded eval dir.
    :returns: Unique query texts (subject queries incl. name queries, then known-item queries).
    :rtype: List[str]
    """
    texts = [q["text"] for e in ev["subject_queries"]["subjects"].values() for q in e["queries"]]
    texts += [q["text"] for q in ev["known_item"]]
    return list(dict.fromkeys(texts))


# ---------------------------------------------------------------------------------------------------------------
# planning and batching
# ---------------------------------------------------------------------------------------------------------------
def token_lengths(tokenizer, texts: Sequence[str], max_seq: int, chunk: int = 2000) -> np.ndarray:
    """Token counts (with special tokens) of each text, truncated at ``max_seq``.

    :param tokenizer: The model's tokenizer.
    :param texts: Texts.
    :param max_seq: Truncation length.
    :param chunk: Texts per tokenizer call.
    :returns: int64 array of lengths.
    :rtype: np.ndarray
    """
    out: List[int] = []
    for s in range(0, len(texts), chunk):
        ids = tokenizer(list(texts[s:s + chunk]), add_special_tokens=True, truncation=False)["input_ids"]
        out.extend(min(len(x), max_seq) for x in ids)
    return np.array(out, dtype=np.int64)


def shard_bounds(lengths: Sequence[int], shard_size: int, shard_tokens: int) -> List[List[int]]:
    """Cut length-sorted texts into contiguous shards bounded by text count and token count.

    :param lengths: Token lengths in embedding order.
    :param shard_size: Max texts per shard.
    :param shard_tokens: Max tokens per shard (a single longer text still gets its own shard).
    :returns: ``[start, end)`` pairs.
    :rtype: List[List[int]]
    """
    bounds, start, tok = [], 0, 0
    for i, n in enumerate(lengths):
        if i > start and (i - start >= shard_size or tok + int(n) > shard_tokens):
            bounds.append([start, i])
            start, tok = i, 0
        tok += int(n)
    if len(lengths):
        bounds.append([start, len(lengths)])
    return bounds


def make_batches(lengths: Sequence[int], pad_multiple: int, max_batch: int, max_tokens: int,
                 max_attn: int) -> List[Tuple[int, int]]:
    """Greedy contiguous batches over ascending lengths.

    A batch of b texts padded to n tokens satisfies ``b <= max_batch``, ``b * n <= max_tokens`` and
    ``b * n * n <= max_attn`` (bounds the attention buffers); a single text always forms a batch.

    :param lengths: Ascending token lengths.
    :param pad_multiple: Padding multiple.
    :param max_batch: Max texts per batch.
    :param max_tokens: Max padded tokens per batch.
    :param max_attn: Max ``b * n^2`` per batch.
    :returns: ``(start, end)`` pairs.
    :rtype: List[Tuple[int, int]]
    """
    out, start = [], 0
    while start < len(lengths):
        end = start + 1
        while end < len(lengths):
            n = -(-int(lengths[end]) // pad_multiple) * pad_multiple
            b = end - start + 1
            if b > max_batch or b * n > max_tokens or b * n * n > max_attn:
                break
            end += 1
        out.append((start, end))
        start = end
    return out


def build_plan(texts: Dict[str, str], tokenizer, args: argparse.Namespace) -> dict:
    """Deterministic embedding order (token length, then hash) and shard boundaries.

    :param texts: ``text_hash -> text``.
    :param tokenizer: The model's tokenizer.
    :param args: Parsed CLI arguments (``shard_size``, ``shard_tokens``).
    :returns: ``{hashes, lengths, bounds, max_seq, shard_size, shard_tokens}``.
    :rtype: dict
    """
    hashes = sorted(texts)
    lengths = token_lengths(tokenizer, [texts[h] for h in hashes], MAX_SEQ)
    order = sorted(range(len(hashes)), key=lambda i: (int(lengths[i]), hashes[i]))
    hashes = [hashes[i] for i in order]
    lengths = lengths[order]
    return {"hashes": hashes, "lengths": lengths.tolist(), "max_seq": MAX_SEQ, "shard_size": args.shard_size,
            "shard_tokens": args.shard_tokens,
            "bounds": shard_bounds(lengths, args.shard_size, args.shard_tokens)}


# ---------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------
def load_encoder(dtype_name: str, pad_multiple: int, n_layers: int = 0) -> CachedTextEncoder:
    """Build the cached encoder with the pinned base model preloaded.

    The model is loaded here (rather than lazily by ``CachedTextEncoder.model``) to pin the dtype and to pad
    batches to a multiple of ``pad_multiple`` tokens.

    :param dtype_name: torch dtype name (``float32`` = production contract).
    :param pad_multiple: Pad every batch to a multiple of this many tokens (1 = off).
    :param n_layers: Keep only the first ``n_layers`` transformer layers (smoke tests only: real tokenizer, embeddings
        and 1024-d output, a fraction of the memory and compute; vectors are NOT the contract). 0 = all layers.
    :returns: Encoder whose ``.model`` is ready.
    :rtype: CachedTextEncoder
    """
    import torch
    from sentence_transformers import SentenceTransformer

    enc = CachedTextEncoder(MODEL_KEY, max_seq_length=MAX_SEQ)
    spec = TEXT_MODELS[MODEL_KEY]
    dev = device()
    t0 = time.time()
    log(f"loading {spec['id']} ({dtype_name}, layers {n_layers or 'all'}) on {dev}; weights are read from the NAS")
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
    enc._model = model
    log(f"model {spec['id']}@{spec['revision'][:8]} on {dev}, dtype {next(model.parameters()).dtype}, "
        f"threads {torch.get_num_threads()}, footprint {child_rss_gb(os.getpid()):.2f} GB, load {time.time() - t0:.0f} s")
    return enc


def encode_texts(model, texts: Sequence[str], lengths: Sequence[int], args: argparse.Namespace) -> np.ndarray:
    """Encode documents (raw text, no prefix) in bounded batches; frees the MPS cache between batches.

    :param model: Loaded SentenceTransformer.
    :param texts: Texts, ascending by token length.
    :param lengths: Their token lengths.
    :param args: Parsed CLI arguments (batch bounds).
    :returns: (n, d) float32 L2-normalised vectors.
    :rtype: np.ndarray
    """
    import torch

    mps = model.device.type == "mps"
    parts = []
    for s, e in make_batches(lengths, args.pad_multiple, args.max_batch, args.max_batch_tokens, args.max_attn):
        parts.append(model.encode(list(texts[s:e]), batch_size=e - s, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False))
        if mps:
            torch.mps.empty_cache()
    return np.concatenate(parts).astype(np.float32)


def encode_with_backoff(model, texts: Sequence[str], lengths: Sequence[int], args: argparse.Namespace) -> np.ndarray:
    """:func:`encode_texts`, retried with halved batch bounds if the device runs out of memory.

    The MPS allocator is capped by the guard (high watermark); an out-of-memory error on one shard should shrink the
    batches rather than end the run. Any other error is raised.

    :param model: Loaded SentenceTransformer.
    :param texts: Texts, ascending by token length.
    :param lengths: Their token lengths.
    :param args: Parsed CLI arguments (batch bounds).
    :returns: (n, d) float32 L2-normalised vectors.
    :rtype: np.ndarray
    """
    import torch

    bounds = argparse.Namespace(**vars(args))
    while True:
        try:
            return encode_texts(model, texts, lengths, bounds)
        except RuntimeError as err:
            if "out of memory" not in str(err).lower() or bounds.max_batch_tokens <= 512:
                raise
            if model.device.type == "mps":
                torch.mps.empty_cache()
            bounds.max_batch_tokens //= 2
            bounds.max_attn //= 4
            log(f"out of memory; retrying the shard with max_batch_tokens={bounds.max_batch_tokens}, "
                f"max_attn={bounds.max_attn}")


def stored_rows(vec_dir: Path, wanted: Sequence[int]) -> np.ndarray:
    """Read selected rows of an ``embed_corpus`` vector dir without loading every shard.

    :param vec_dir: Vector dir (``shard_XXXX.npy`` files).
    :param wanted: Row indices into the concatenated matrix.
    :returns: (len(wanted), d) array.
    :rtype: np.ndarray
    """
    shards = sorted(p for p in vec_dir.glob("shard_*.npy") if ".tmp" not in p.name)
    maps = [np.load(p, mmap_mode="r") for p in shards]
    starts = np.cumsum([0] + [len(m) for m in maps])
    out = []
    for i in wanted:
        k = int(np.searchsorted(starts, i, side="right") - 1)
        out.append(np.array(maps[k][i - starts[k]]))
    return np.stack(out)


def contract_canary(enc: CachedTextEncoder, n: int, args: argparse.Namespace) -> dict:
    """Re-embed ``n`` production texts and compare them with the stored production vectors.

    Records are spread evenly over the stored (length-sorted) order, so the longest texts are included.

    :param enc: Loaded encoder.
    :param n: Number of records.
    :param args: Parsed CLI arguments (batch bounds, ``canary_min_cos``).
    :returns: ``{n, min_cos, mean_cos, max_abs_diff, ok}``.
    :rtype: dict
    """
    ids = json.loads((ORIGINAL_VECS / "ids.json").read_text())
    pick = sorted(set(np.linspace(0, len(ids) - 1, n).round().astype(int).tolist()))
    want = {ids[i]: i for i in pick}
    texts: Dict[str, str] = {}
    with open(ORIGINAL_CORPUS, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["doc_id"] in want:
                texts[r["doc_id"]] = r["text"]
    order = [ids[i] for i in pick]
    lengths = token_lengths(enc.model.tokenizer, [texts[d] for d in order], MAX_SEQ)
    srt = np.argsort(lengths, kind="stable")
    encoded = encode_texts(enc.model, [texts[order[j]] for j in srt], lengths[srt], args)
    vecs = np.empty_like(encoded)
    vecs[srt] = encoded
    ref = stored_rows(ORIGINAL_VECS, pick)
    cos = (vecs * ref).sum(1)
    out = {"n": len(pick), "min_cos": round(float(cos.min()), 6), "mean_cos": round(float(cos.mean()), 6),
           "max_abs_diff": round(float(np.abs(vecs - ref).max()), 6), "max_tokens": int(lengths.max()),
           "threshold": args.canary_min_cos, "ok": bool(cos.min() >= args.canary_min_cos)}
    log(f"contract canary vs stored production vectors: {out}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# embedding phases
# ---------------------------------------------------------------------------------------------------------------
def acquire_lock(out: Path) -> Path:
    """Take the one-writer lock of a vector dir (a stale lock of a dead pid is taken over).

    :param out: Vector dir.
    :returns: Lock path.
    :rtype: Path
    """
    lock = out / "LOCK"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            raise SystemExit(f"{out} is being embedded by pid {pid}")
        except (ProcessLookupError, ValueError):
            pass
    lock.write_text(str(os.getpid()))
    return lock


def embed_queries(enc: CachedTextEncoder, texts: List[str], out: Path, args: argparse.Namespace) -> np.ndarray:
    """Encode the eval queries with the Qwen3 query instruction and store them in ``out`` as one matrix.

    Uses ``CachedTextEncoder.prefix`` (``Instruct: <DEFAULT_TASK>\\nQuery: `` + text, as the production query path) but
    not its one-file-per-text cache: on the NAS each small file costs seconds (32 queries took 98 s in the smoke),
    so the whole set is encoded length-sorted in bounded batches (about a minute on MPS) and saved once.

    :param enc: Loaded encoder.
    :param texts: Unique query texts.
    :param out: Vector dir (``queries.json`` + ``queries.npy``).
    :param args: Parsed CLI arguments (batch bounds).
    :returns: (n, d) query matrix.
    :rtype: np.ndarray
    """
    t0 = time.time()
    prefix = enc.prefix("query", DEFAULT_TASK)
    full = [prefix + t for t in texts]
    lengths = token_lengths(enc.model.tokenizer, full, MAX_SEQ)
    order = np.argsort(lengths, kind="stable")
    encoded = encode_texts(enc.model, [full[j] for j in order], lengths[order], args)
    mat = np.empty_like(encoded)
    mat[order] = encoded
    save_npy(out / "queries.npy", mat)
    write_json(out / "queries.json", {"task": DEFAULT_TASK, "prefix": prefix, "texts": texts})
    log(f"queries: {len(texts)} ({int(lengths.sum())} tokens) in {time.time() - t0:.0f} s")
    return mat


def embed_unique(enc: CachedTextEncoder, texts: Dict[str, str], out: Path, args: argparse.Namespace) -> dict:
    """Embed every unique text in resumable shards; exits with :data:`RESTART_RC` above the soft footprint limit.

    :param enc: Loaded encoder.
    :param texts: ``text_hash -> text``.
    :param out: Vector dir (shards go to ``out/unique``).
    :param args: Parsed CLI arguments.
    :returns: The plan (hash order, lengths, shard bounds) plus run stats.
    :rtype: dict
    """
    udir = out / "unique"
    udir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(texts, enc.model.tokenizer, args)
    plan_path = udir / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old["hashes"] != plan["hashes"] or old["bounds"] != plan["bounds"]:
            raise SystemExit(f"stored plan differs from this run's (changed corpus or shard args); "
                             f"remove {udir} to start over")
    else:
        write_json(plan_path, plan)
    lengths = np.array(plan["lengths"])
    bounds = plan["bounds"]
    todo = [s for s in range(len(bounds)) if not (udir / f"shard_{s:04d}.npy").exists()]
    tok_left = int(sum(lengths[bounds[s][0]:bounds[s][1]].sum() for s in todo))
    log(f"unique texts {len(plan['hashes'])}, tokens {int(lengths.sum())}, shards {len(bounds)}, "
        f"to do {len(todo)} ({tok_left} tokens)")
    t_start, tok_done = time.time(), 0
    for k, s in enumerate(todo):
        a, b = bounds[s]
        t0 = time.time()
        vecs = encode_with_backoff(enc.model, [texts[h] for h in plan["hashes"][a:b]], lengths[a:b], args)
        save_npy(udir / f"shard_{s:04d}.npy", vecs)
        n_tok = int(lengths[a:b].sum())
        tok_done += n_tok
        tok_left -= n_tok
        rate = tok_done / (time.time() - t_start)
        fp = child_rss_gb(os.getpid())
        log(f"shard {s + 1}/{len(bounds)} n={b - a} tok={n_tok} max_len={int(lengths[b - 1])} "
            f"{n_tok / (time.time() - t0):.0f} tok/s ({(b - a) / (time.time() - t0):.1f} texts/s); "
            f"session {rate:.0f} tok/s, ETA {tok_left / rate / 60:.1f} min; footprint {fp:.2f} GB")
        if args.soft_max_gb and fp > args.soft_max_gb and k + 1 < len(todo):
            log(f"footprint {fp:.2f} GB > soft limit {args.soft_max_gb} GB: exiting {RESTART_RC} for a fresh relaunch")
            (out / "LOCK").unlink(missing_ok=True)
            sys.exit(RESTART_RC)
    plan["seconds_this_session"] = round(time.time() - t_start, 1)
    return plan


def expand(out: Path, rows: List[dict], plan: dict) -> None:
    """Map unique-text vectors back to every record and write the ``load_vectors`` layout.

    :param out: Vector dir.
    :param rows: Eligible rows ``{doc_id, text_hash}`` in corpus order.
    :param plan: Embedding plan (hash order and shard bounds).
    """
    uniq = np.concatenate([np.load(out / "unique" / f"shard_{s:04d}.npy") for s in range(len(plan["bounds"]))])
    if len(uniq) != len(plan["hashes"]):
        raise ValueError(f"{len(uniq)} unique vectors for {len(plan['hashes'])} texts")
    pos = {h: i for i, h in enumerate(plan["hashes"])}
    idx = np.array([pos[r["text_hash"]] for r in rows], dtype=np.int64)
    for old in out.glob("shard_*.npy"):
        old.unlink()
    for s in range(0, len(rows), RECORD_SHARD):
        save_npy(out / f"shard_{s // RECORD_SHARD:04d}.npy", uniq[idx[s:s + RECORD_SHARD]])
    write_json(out / "ids.json", [r["doc_id"] for r in rows])


# ---------------------------------------------------------------------------------------------------------------
# smoke-test subset
# ---------------------------------------------------------------------------------------------------------------
def smoke_selection(ev: dict, n: int, seed: int = 0) -> dict:
    """Pick ~``n`` eligible records that exercise every eval section, plus the subjects/queries to keep.

    One held-out, one linked and two seen subjects (one with memorisation pairs) with enough carriers, known-item
    targets, two identical-text groups, two long records, then short random fill.

    :param ev: Loaded full eval dir.
    :param n: Target number of records.
    :param seed: RNG seed.
    :returns: ``{doc_ids, subjects: {sid: role}, memorisation: {sid: {train, held_out}}, known_item_targets}``.
    :rtype: dict
    """
    rng = random.Random(seed)
    pool = {r["doc_id"]: r for r in ev["eval_pool"]}
    rel = ev["subject_relevance"]
    sq = ev["subject_queries"]["subjects"]
    chosen: Dict[str, None] = {}

    def short(ids: Sequence[str], max_chars: int = 1500) -> List[str]:
        return sorted(d for d in ids if d in pool and d not in chosen and pool[d]["n_chars"] <= max_chars)

    def take(ids: Sequence[str], k: int) -> List[str]:
        got = rng.sample(list(ids), min(k, len(ids)))
        chosen.update(dict.fromkeys(got))
        return got

    subjects: Dict[str, str] = {}
    for role, k in (("held_out", 12), ("linked", 8), ("seen", 8)):
        sid = next(s for s in sorted(sq) if sq[s]["status"] == role and len(short(rel[s]["strong"])) >= k)
        take(short(rel[sid]["strong"]), k)
        subjects[sid] = role
    mem_sid = next(s for s, p in sorted(ev["memorization_pairs"]["subjects"].items())
                   if s in sq and s not in subjects and len(short(p["train"])) >= 6 and len(short(p["held_out"])) >= 6)
    pair = ev["memorization_pairs"]["subjects"][mem_sid]
    mem = {mem_sid: {"train": take(short(pair["train"]), 6), "held_out": take(short(pair["held_out"]), 6)}}
    subjects[mem_sid] = "seen"
    targets = take(short({q["doc_id"] for q in ev["known_item"]}), 12)
    groups: Dict[str, List[str]] = defaultdict(list)
    for r in ev["eval_pool"]:
        if 2 <= r["dup_n"] <= 3:
            groups[r["text_hash"]].append(r["doc_id"])
    for h in sorted(groups)[:2]:
        chosen.update(dict.fromkeys(groups[h]))
    longs = sorted(d for d, r in pool.items() if 3000 <= r["n_chars"] <= 6000 and d not in chosen)
    take(longs, 2)
    rest = short(pool, 1200)
    take(rest, max(0, n - len(chosen)))
    return {"doc_ids": list(chosen), "subjects": subjects, "memorisation": mem, "known_item_targets": targets}


def write_smoke_eval_dir(ev: dict, sel: dict, path: Path) -> None:
    """Write a reduced eval dir for the smoke subset (few subjects, 3 queries each, 2 queries per target).

    :param ev: Loaded full eval dir.
    :param sel: Output of :func:`smoke_selection`.
    :param path: Target directory.
    """
    keep = set(sel["doc_ids"])
    sq = ev["subject_queries"]
    per_target: Dict[str, int] = defaultdict(int)
    ki = []
    for q in ev["known_item"]:
        if q["doc_id"] in sel["known_item_targets"] and per_target[q["doc_id"]] < 2:
            per_target[q["doc_id"]] += 1
            ki.append(q)
    pool = [r for r in ev["eval_pool"] if r["doc_id"] in keep]
    small = {
        "held_out_subjects": ev["held_out_subjects"],
        "held_out_records": ev["held_out_records"],
        "subject_queries": {**sq, "subjects": {s: {**sq["subjects"][s], "queries": sq["subjects"][s]["queries"][:3]}
                                               for s in sel["subjects"]}},
        "subject_relevance": {s: ev["subject_relevance"][s] for s in sel["subjects"]},
        "known_item": ki,
        "memorization_pairs": {**ev["memorization_pairs"], "subjects": sel["memorisation"]},
        "collection_sample": {"doc_ids": [r["doc_id"] for r in pool if r["series"]]},
        "eval_pool": pool,
    }
    eval_v3._write_eval_dir(path, small)


# ---------------------------------------------------------------------------------------------------------------
# evaluation and report
# ---------------------------------------------------------------------------------------------------------------
def known_item_clusters(res: dict, qtype: Optional[str], metric: str) -> Dict[str, List[float]]:
    """Per-target per-query known-item values, optionally for one query type.

    :param res: :func:`eval_v3.evaluate` result.
    :param qtype: Query type, or None for all.
    :param metric: ``RR``, ``R@10`` or ``R@100``.
    :returns: ``doc_id -> values`` (query order preserved).
    :rtype: Dict[str, List[float]]
    """
    pq = res["known_item"]["per_query"]
    out: Dict[str, List[float]] = defaultdict(list)
    for doc, t, rank in zip(pq["doc_id"], pq["type"], pq["rank"]):
        if qtype is None or t == qtype:
            out[doc].append({"RR": 1.0 / rank, "R@10": float(rank <= 10), "R@100": float(rank <= 100)}[metric])
    return out


def paired_comparisons(res_o: dict, res_s: dict, n_resamples: int) -> dict:
    """Paired bootstrap (semantic minus original) for every headline metric.

    :param res_o: Original-text result.
    :param res_s: Semantic-text result.
    :param n_resamples: Bootstrap resamples.
    :returns: Nested dict of :func:`eval_v3.paired_bootstrap` outputs.
    :rtype: dict
    """
    out: dict = {}
    for group in eval_v3.STATUSES:
        if res_o["subjects"][group]["per_query"]:
            out[group] = {m: eval_v3.compare(res_o, res_s, group, m, n_resamples) for m in ("AP", "P@10", "R@100")}
    out["held_out_per_subject"] = {
        sid: eval_v3.paired_bootstrap({sid: pq["AP"]}, {sid: res_s["subjects"]["held_out"]["per_query"][sid]["AP"]},
                                      n_resamples)
        for sid, pq in res_o["subjects"]["held_out"]["per_query"].items()}
    out["known_item"] = {m: eval_v3.compare(res_o, res_s, "known_item", m, n_resamples) for m in ("RR", "R@10", "R@100")}
    out["known_item_by_type"] = {
        t: {m: eval_v3.paired_bootstrap(known_item_clusters(res_o, t, m), known_item_clusters(res_s, t, m), n_resamples)
            for m in ("RR", "R@10")}
        for t in res_o["known_item"]["by_type"]}
    return out


def table_rows(res_o: dict, res_s: dict, paired: dict) -> List[Tuple[str, Optional[float], Optional[float], Optional[dict]]]:
    """Rows ``(label, original, semantic, paired CI or None)`` of the side-by-side table.

    :param res_o: Original-text result.
    :param res_s: Semantic-text result.
    :param paired: Output of :func:`paired_comparisons`.
    :returns: Table rows.
    :rtype: list
    """
    rows = []
    so, ss = res_o["subjects"], res_s["subjects"]
    for group, label in (("held_out", "held-out subjects (GATE)"), ("linked", "linked subjects"),
                         ("seen", "seen subjects")):
        if group not in paired:
            continue
        n = so[group]["macro"]["n_subjects"]
        for m in ("AP", "P@10", "R@100"):
            rows.append((f"{label} n={n} {m}", so[group]["macro"][m], ss[group]["macro"][m], paired[group][m]))
        if group == "held_out":
            for src in sorted(so[group]["macro_by_source"]):
                rows.append((f"  held-out by source {src} AP", so[group]["macro_by_source"][src]["AP"],
                             ss[group]["macro_by_source"][src]["AP"], None))
            for sid in sorted(so[group]["per_subject"]):
                rows.append((f"  {sid} AP", so[group]["per_subject"][sid]["AP"], ss[group]["per_subject"][sid]["AP"],
                             paired["held_out_per_subject"][sid]))
        if group == "seen":
            for kind in sorted(so[group]["macro_by_kind"]):
                rows.append((f"  seen {kind} AP (n={so[group]['macro_by_kind'][kind]['n_subjects']})",
                             so[group]["macro_by_kind"][kind]["AP"], ss[group]["macro_by_kind"][kind]["AP"], None))
    ko, ks = res_o["known_item"], res_s["known_item"]
    for m, pm in (("MRR", "RR"), ("R@10", "R@10"), ("R@100", "R@100")):
        rows.append((f"known-item all n={ko['all']['n']} {m}", ko["all"][m], ks["all"][m], paired["known_item"][pm]))
    for t in sorted(ko["by_type"]):
        rows.append((f"  known-item {t} n={ko['by_type'][t]['n']} MRR", ko["by_type"][t]["MRR"],
                     ks["by_type"][t]["MRR"], paired["known_item_by_type"][t]["RR"]))
    if ko["within_subject"].get("n"):
        rows.append(("  known-item within-subject MRR", ko["within_subject"]["MRR"], ks["within_subject"]["MRR"], None))
        rows.append(("  known-item within-subject median pct (lower=better)", ko["within_subject"]["median_percentile"],
                     ks["within_subject"]["median_percentile"], None))
    mo, ms = res_o["memorisation"]["macro"], res_s["memorisation"]["macro"]
    if mo.get("n_subjects"):
        rows.append((f"memorisation gap n={mo['n_subjects']} (calibration, ~0 for base)", mo["gap"], ms["gap"], None))
    for part in ("all", "long"):
        rows.append((f"collection lift {part} (1 = none)", res_o["collection"][part].get("lift"),
                     res_s["collection"][part].get("lift"), None))
    rows.append(("collection lift_vs_pool all", res_o["collection"]["all"].get("lift_vs_pool"),
                 res_s["collection"]["all"].get("lift_vs_pool"), None))
    rows.append(("hubness share dup/short in top-10", res_o["hubness"]["share"], res_s["hubness"]["share"], None))
    rows.append(("hubness ratio vs pool (1 = chance)", res_o["hubness"]["ratio"], res_s["hubness"]["ratio"], None))
    return rows


def format_table(rows: List[tuple]) -> str:
    """Render table rows as fixed-width text.

    :param rows: Output of :func:`table_rows`.
    :returns: Table text.
    :rtype: str
    """
    def num(x: Optional[float]) -> str:
        return "   -   " if x is None else f"{x:7.4f}"

    width = max(len(r[0]) for r in rows) + 2
    lines = [f"{'metric':<{width}}{'original':>9}{'semantic':>10}   delta (sem - orig) [95% CI]  p(sem better)"]
    for label, o, s, ci in rows:
        tail = "" if ci is None else (f"   {ci['delta']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]"
                                      f"  {ci['p_b_better']:.2f}")
        lines.append(f"{label:<{width}}{num(o):>9}{num(s):>10}{tail}")
    return "\n".join(lines)


def run_eval(out: Path, eval_dir: Path, rows: List[dict], results_path: Path, args: argparse.Namespace,
             extra: dict) -> dict:
    """Score both representations on the identical eligible pool and write the report.

    :param out: Semantic vector dir (also holds the query vectors).
    :param eval_dir: Eval dir (full or smoke subset).
    :param rows: Eligible rows ``{doc_id, text_hash}``.
    :param results_path: Report path.
    :param args: Parsed CLI arguments.
    :param extra: Extra report fields (embedding meta, canary).
    :returns: The report.
    :rtype: dict
    """
    q = json.loads((out / "queries.json").read_text(encoding="utf-8"))
    qmat = np.load(out / "queries.npy")
    if not np.isfinite(qmat).all():
        raise ValueError("non-finite query vectors")
    qvec = dict(zip(q["texts"], qmat))

    def qfn(texts: List[str]) -> np.ndarray:
        return np.stack([qvec[t] for t in texts])

    results = {}
    for name, vec_dir, corpus_rows in (
            ("original", ORIGINAL_VECS, None),
            ("semantic", out, {r["doc_id"]: {"text_hash": r["text_hash"]} for r in rows})):
        ids, mat = load_vectors(vec_dir)
        if not np.isfinite(mat).all():
            raise ValueError(f"non-finite values in {vec_dir}")
        # macOS Accelerate matmul raises spurious divide/overflow warnings on finite unit vectors; inputs are checked
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            results[name] = eval_v3.evaluate(ids, mat, qfn, eval_dir, corpus_rows=corpus_rows,
                                             n_resamples=args.n_resamples)
        del ids, mat
        log(f"eval {name}: {results[name]['seconds']} s, pool {results[name]['pool']}")
        write_json(out / f"eval_v3_{name}.json", results[name])
    res_o, res_s = results["original"], results["semantic"]
    for key in ("n_pool_scored", "n_missing_vectors", "n_query_texts"):
        if res_o["pool"][key] != res_s["pool"][key]:
            raise ValueError(f"pools differ on {key}: {res_o['pool'][key]} vs {res_s['pool'][key]}")
    if res_s["pool"]["text_hash_mismatch"]:
        raise ValueError(f"{res_s['pool']['text_hash_mismatch']} semantic vectors are not of the frozen pool's text")
    paired = paired_comparisons(res_o, res_s, args.n_resamples)
    rows_t = table_rows(res_o, res_s, paired)
    report = {"ran_at": datetime.now().isoformat(timespec="seconds"),
              "question": "Does removing the Document ID / Shelf Mark lines alone improve the base embedder?",
              "model": {**TEXT_MODELS[MODEL_KEY], "max_seq": MAX_SEQ, "query_task": DEFAULT_TASK},
              "vectors": {"original": str(ORIGINAL_VECS), "semantic": str(out)}, "eval_dir": str(eval_dir),
              "full_results": {k: str(out / f"eval_v3_{k}.json") for k in results},
              **extra,
              "original": eval_v3.summary(res_o), "semantic": eval_v3.summary(res_s), "paired_sem_minus_orig": paired,
              "table": [{"metric": r[0], "original": r[1], "semantic": r[2], "paired": r[3]} for r in rows_t]}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(results_path, report)
    print(format_table(rows_t), flush=True)
    log(f"wrote {results_path}")
    return report


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    """Embed the semantic text (resumable), embed the eval queries, score both representations."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", type=int, default=0, help="smoke test on ~N records (writes AUDIT_ROOT/v3/step0_smoke)")
    parser.add_argument("--out-dir", default=None, help=f"semantic vector dir (default {OUT_VECS})")
    parser.add_argument("--dtype", default="float32", help="torch dtype (float32 = the production contract)")
    parser.add_argument("--shard-size", type=int, default=2000, help="max unique texts per shard")
    parser.add_argument("--shard-tokens", type=int, default=300_000, help="max tokens per shard")
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-attn", type=int, default=1 << 22, help="max batch * padded_len^2")
    parser.add_argument("--soft-max-gb", type=float, default=0.0,
                        help=f"exit {RESTART_RC} between shards above this footprint (0 = off)")
    parser.add_argument("--cpu-threads", type=int, default=4, help="torch CPU threads unless AUDIT_THREADS is set")
    parser.add_argument("--canary", type=int, default=32, help="production texts re-embedded as a contract check")
    parser.add_argument("--canary-min-cos", type=float, default=CANARY_MIN_COS,
                        help="min cosine to the stored production vectors (lower it only for a non-fp32 smoke)")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--smoke-layers", type=int, default=0,
                        help="smoke only: truncate the model to its first N layers (fits the 3 GB --small cap; not the "
                             "contract, so pass --canary-min-cos -1)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(AUDIT_ROOT / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ev_full = eval_v3.load_eval_dir(EVAL_DIR)
    if args.smoke:
        sel = smoke_selection(ev_full, args.smoke, args.seed)
        eval_dir = SMOKE_DIR / "eval"
        write_smoke_eval_dir(ev_full, sel, eval_dir)
        out = Path(args.out_dir) if args.out_dir else SMOKE_DIR / "vectors"
        results_path = SMOKE_DIR / "results.json"
        rows, texts = load_semantic(SEMANTIC_CORPUS, set(sel["doc_ids"]))
        log(f"smoke: {len(rows)} records, {len(texts)} unique texts, subjects {sel['subjects']}")
    else:
        eval_dir, out, results_path = EVAL_DIR, Path(args.out_dir) if args.out_dir else OUT_VECS, RESULTS
        rows, texts = load_semantic(SEMANTIC_CORPUS)
        log(f"{len(rows)} eligible records, {len(texts)} unique texts")
    ev = eval_v3.load_eval_dir(eval_dir) if args.smoke else ev_full
    qtexts = query_texts(ev)
    del ev_full, ev
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    q_done = (out / "queries.npy").exists() and (out / "queries.json").exists() and \
        json.loads((out / "queries.json").read_text(encoding="utf-8"))["texts"] == qtexts
    docs_done = meta.get("complete") and meta.get("n_records") == len(rows) and (out / "ids.json").exists()
    canary = json.loads((out / "canary.json").read_text()) if (out / "canary.json").exists() else None
    if not (q_done and docs_done and canary):
        import torch

        if not os.environ.get("AUDIT_THREADS"):
            torch.set_num_threads(args.cpu_threads)
        lock = acquire_lock(out)
        if args.smoke_layers and not args.smoke:
            raise SystemExit("--smoke-layers is for --smoke runs only")
        enc = load_encoder(args.dtype, args.pad_multiple, args.smoke_layers)
        if canary is None:
            canary = contract_canary(enc, args.canary, args)
            write_json(out / "canary.json", canary)
        if not canary["ok"]:
            raise SystemExit(f"contract canary failed: {canary}")
        if not q_done:
            embed_queries(enc, qtexts, out, args)
        if not docs_done:
            t0 = time.time()
            plan = embed_unique(enc, texts, out, args)
            expand(out, rows, plan)
            meta = {"complete": True, "written": datetime.now().isoformat(timespec="seconds"),
                    "model": TEXT_MODELS[MODEL_KEY], "dtype": args.dtype, "max_seq": MAX_SEQ,
                    "document_prefix": "", "normalised": True, "text": str(SEMANTIC_CORPUS),
                    "n_records": len(rows), "n_unique_texts": len(plan["hashes"]),
                    "n_tokens": int(sum(plan["lengths"])), "n_shards_unique": len(plan["bounds"]),
                    "device_last_session": str(enc.model.device), "seconds_last_session": round(time.time() - t0, 1),
                    "batching": {k: getattr(args, k) for k in ("pad_multiple", "max_batch", "max_batch_tokens",
                                                               "max_attn", "shard_size", "shard_tokens")},
                    "smoke": bool(args.smoke), "smoke_layers": args.smoke_layers}
            write_json(meta_path, meta)
            log(f"expanded {len(plan['hashes'])} unique vectors to {len(rows)} records in {out}")
        del enc
        lock.unlink(missing_ok=True)
    run_eval(out, eval_dir, rows, results_path, args, {"embedding": meta, "canary": canary, "smoke": bool(args.smoke)})


if __name__ == "__main__":
    main()
