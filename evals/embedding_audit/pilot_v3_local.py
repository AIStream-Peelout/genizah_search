"""Local LoRA pilot of the v3 recipe on the real Qwen3-Embedding-0.6B (MPS): an early read on the held-out gate.

Two questions, answered before the Colab run: (1) does held-out-subject retrieval move at all versus the base model
when the v3 recipe trains the real weights, and (2) does the v3 trainer machinery (``colab/train_embedder_v3.py``:
per-kind datasets, nested prompts, ``label`` id masking, :class:`MaskedCachedMNRL`) work on the real model.

Design (what differs from the Colab run, on purpose):

* **LoRA** (peft) r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down of every layer; the frozen base runs in bf16
  (no autocast), the LoRA weights and Adam state are fp32. The loss upcasts the bf16 embeddings to fp32 before the
  masked InfoNCE (:class:`Fp32MaskedCachedMNRL`). Gradient checkpointing, GradCache mini-batch 8, batch 64, max_seq
  256, 1 epoch, lr 1e-4, 5 % warmup, cosine, InfoNCE scale 20, trainer seed 13, proportional multi-dataset sampler,
  NO_DUPLICATES batches -- the ``train_embedder_v3.train`` settings except LoRA, lr, batch, window, plain (non-fused)
  AdamW (fused kernels are not guaranteed on MPS) and padding to multiples of 32 tokens (few distinct MPS shapes).
* **Training subset**: a seeded sample of ``--n-train`` (default 12,000 = 189 steps) rows of ``train_mix_v3.jsonl``
  stratified by (kind, family, mask_level) with largest-remainder allocation, so every cell keeps its share of the
  mixture. 12,000 (the low end of the approved 12-16k) because MPS measured ~36 s/step (42 s for q2d batches) while the
  sibling's consensus pipeline shares the GPU (~15 s/step on a free GPU): 14,000 would push training alone to ~2.2 h.
* **No hard-negative mining** (in-batch masked negatives only): mining embeds the whole 43,904-record train pool with
  the base model (~20 min on MPS), a fifth of the time budget, for a refinement the gate read does not need.
* **Reduced eval pool** (fixed, identical for every model): every held-out eligible record (15,674; all carriers of
  the 6 gate subjects and the 78 linked ones) plus a seeded random sample of ``--n-other`` (10,000) of the 44,819
  other eligible records = 25,674 of 60,493. No change to ``eval_v3`` was needed: ``eval_v3.evaluate`` scores exactly
  the frozen-pool records present in ``doc_ids``, so each model is scored by passing only the reduced pool's vectors
  (``pool.n_missing_vectors`` = 34,819 is expected). Consequences: absolute AP is higher than on the full pool
  (fewer distractors) -- compare models with each other, not with full-pool numbers; seen subjects with < 5 carriers
  left in the pool drop out (454 of 662 remain), memorisation keeps 83 of 255 subjects, the collection sample keeps
  1,245 of 3,000, known-item 1,961 of 2,003 queries. ``pool_calibration`` in the results scores the base model on
  the original text over both pools (no compute) to show the size of the reduced-pool effect.
* **Three models**, each scored with ``eval_v3`` on the same pool and queries, documents raw at the production
  window (8192), queries with the production prompt (``Instruct: Given a search query, ...``), fp32, L2-normalised:
  ``base_original`` (the stored production-text vectors ``corpus_vectors/qwen3-0.6b__all__8192``, subset; no
  compute), ``base_semantic`` (the base model re-embedding the CURRENT v3 semantic text of the pool -- the stored
  semantic vectors predate the review that masked shelf marks in 3,947 descriptions) and ``tuned_semantic`` (fp32
  base + the LoRA merged in). ``train_embedder_v3.compare_reports`` gives the table and the paired, subject-clustered
  bootstrap CIs; :func:`extra_paired` adds gate P@10 / seen AP versus ``base_original`` and known-item RR by query
  type.

Phases, each resumable (the guard may kill and relaunch the process at any time):

1. ``prepare``     -- subset, reduced pool, hold-out guard (``train_embedder_v3.check_data``).
2. ``embed-base``  -- contract canary (``step0_local.contract_canary``: re-embedded production text vs the stored
   production vectors), base query vectors, base semantic vectors (``step0_local.embed_unique``: length-sorted
   shards, tmp + rename, OOM back-off, exit 99 above ``--soft-max-gb``).
3. ``train``       -- LoRA training; a checkpoint (LoRA weights, optimizer, scheduler, RNG, trainer state, mask stats)
   every ``--save-steps`` steps; a relaunch resumes from the newest complete checkpoint (HF ``resume_from_checkpoint``:
   same batches, restored RNG). The footprint is checked every ``--logging-steps`` steps; above the soft limit the
   trainer saves a checkpoint and the process exits 99. A SIGTERM (the guard's kill) also checkpoints the step in
   flight before exiting (:class:`PilotCallback`), so a guard kill usually costs one step, not ``--save-steps``.
4. ``embed-tuned`` -- fp32 base + LoRA, merge check (merged vs unmerged vectors), tuned query + semantic vectors.
5. ``eval``        -- the three reports, the comparison, ``results/pilot_v3_local.json``.

Outputs: ``AUDIT_ROOT/v3/pilot_local/`` (``subset/``, ``pool/``, ``vectors/{base,tuned}_semantic/``, ``checkpoints/``,
``lora_final/`` with ``training_info.json``, ``reports/`` = full eval results incl. per-query lists),
``results/pilot_v3_local.json`` (headline, table, paired CIs, training info, timings), log
``AUDIT_ROOT/logs/pilot_v3.log`` (via ``run_pilot_v3_local.sh``).

Run (from this directory; ``PY`` = the historical-document-analysis venv python)::

    # smoke: TINY random-init Qwen3 + a ~300-record eval subset, CPU, 3 GB cap -> AUDIT_ROOT/v3/pilot_local_smoke
    HF_HUB_OFFLINE=1 python3 guard.py --name pilot_smoke --small -- $PY pilot_v3_local.py --smoke 300
    # real-weight check: the 0.6B in bf16 + LoRA, 2 optimizer steps on CPU, checkpoint round trip, merge path
    HF_HUB_OFFLINE=1 python3 guard.py --name pilot_realcheck --small -- $PY pilot_v3_local.py --check-real-weights
    # optional MPS timing read (~5 min, own dir pilot_local/bench/): N steps -> s/step, projected training minutes
    python3 guard.py --name gpu_pilot_bench --max-gb 8 -- $PY pilot_v3_local.py --bench-steps 12
    # the pilot, relaunched while it exits 99 (soft footprint limit)
    nohup ./run_pilot_v3_local.sh > /dev/null 2>&1 &
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import signal
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "colab"))
# before huggingface_hub is imported (it reads HF_HUB_OFFLINE / HF_HOME once, at import): the weights are local
os.environ.setdefault("HF_HOME", "/Volumes/home/studio_offload/genizah_search_embedding_audit/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformerTrainer  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

import eval_v3  # noqa: E402
import step0_local as step0  # noqa: E402
import train_embedder_v3 as te  # noqa: E402
from embed_utils import AUDIT_ROOT, DEFAULT_TASK, TEXT_MODELS, CachedTextEncoder, device  # noqa: E402
from eval_corpus import load_vectors  # noqa: E402
from guard import child_rss_gb  # noqa: E402

PKG = AUDIT_ROOT / "hf_dataset" / "v3"
OUT_DIR = AUDIT_ROOT / "v3" / "pilot_local"
SMOKE_DIR = AUDIT_ROOT / "v3" / "pilot_local_smoke"
CHECK_DIR = AUDIT_ROOT / "v3" / "pilot_local_realcheck"
RESULTS = HERE / "results" / "pilot_v3_local.json"
ORIGINAL_VECS = step0.ORIGINAL_VECS
MODEL_KEY = step0.MODEL_KEY
MAX_SEQ = step0.MAX_SEQ
RESTART_RC = step0.RESTART_RC
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PHASES = ("prepare", "embed-base", "train", "embed-tuned", "eval")
MERGE_MIN_COS = {"float32": 0.9999, "bfloat16": 0.99}  # merged vs unmerged LoRA vectors (bf16 merges round)
CKPT_FILES = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "pilot_state.json")
SMOKE_OVERRIDES = {"tiny": True, "batch": 8, "mini_batch": 4, "max_seq": 128, "save_steps": 4, "logging_steps": 1,
                   "n_resamples": 100, "canary": 0, "shard_size": 64, "train_pad_multiple": 32}


def log(msg: str) -> None:
    """Print a timestamped progress line.

    :param msg: Message text.
    """
    print(f"[pilot {datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    """md5 of a file's bytes.

    :param path: File.
    :param chunk: Read size.
    :returns: Hex digest.
    :rtype: str
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    """Write JSONL via a temp file and rename.

    :param path: Target path.
    :param rows: Row dicts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    path.unlink(missing_ok=True)
    tmp.rename(path)


def run_paths(root: Path) -> SimpleNamespace:
    """Every output location of one pilot run.

    :param root: Run directory.
    :returns: Namespace of paths.
    :rtype: SimpleNamespace
    """
    return SimpleNamespace(root=root, subset=root / "subset", pool=root / "pool", eval_dir=root / "eval",
                           vec_base=root / "vectors" / "base_semantic", vec_tuned=root / "vectors" / "tuned_semantic",
                           vec_orig=root / "vectors" / "base_original", ckpt=root / "checkpoints",
                           lora=root / "lora_final", reports=root / "reports")


def release(dev: str) -> None:
    """Collect garbage and return cached MPS memory to the driver (CPU runs never touch Metal).

    :param dev: Device the process computes on.
    """
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()


def maybe_restart(args: argparse.Namespace, where: str, worked: bool) -> None:
    """Exit :data:`RESTART_RC` (relaunch fresh) when the footprint is above the soft limit at a phase boundary.

    Only after a phase that loaded a model in this process: a fresh process that merely skipped completed phases
    would otherwise relaunch forever.

    :param args: Parsed CLI arguments (``soft_max_gb``).
    :param where: Phase just finished (for the log).
    :param worked: Whether that phase loaded a model in this process.
    """
    fp = child_rss_gb(os.getpid())
    log(f"after {where}: footprint {fp:.2f} GB")
    if worked and args.soft_max_gb and fp > args.soft_max_gb:
        log(f"footprint {fp:.2f} GB > soft limit {args.soft_max_gb} GB: exiting {RESTART_RC} for a fresh relaunch")
        sys.exit(RESTART_RC)


def check_code(allow_drift: bool) -> dict:
    """The imported trainer / scorer must be the copies packaged with the v3 data (what Colab will run).

    :param allow_drift: Only record a mismatch instead of stopping.
    :returns: ``{file: {repo, package, same}}``.
    :rtype: dict
    :raises SystemExit: On a mismatch unless ``allow_drift``.
    """
    out = {}
    for name, mod in (("train_embedder_v3.py", te), ("eval_v3.py", eval_v3)):
        repo, pkg = md5_file(Path(mod.__file__)), md5_file(PKG / name)
        out[name] = {"repo": repo, "package": pkg, "same": repo == pkg}
        if repo != pkg and not allow_drift:
            raise SystemExit(f"{mod.__file__} differs from the packaged {PKG / name}; re-package or pass "
                             f"--allow-code-drift")
    return out


# ---------------------------------------------------------------------------------------------------------------
# phase 1: subset + pool
# ---------------------------------------------------------------------------------------------------------------
def stratified_indices(rows: Sequence[dict], n: int, seed: int) -> Tuple[List[int], Dict[str, dict]]:
    """Indices of a seeded sample of ``n`` rows keeping each (kind, family, mask_level) cell's share.

    Quotas are ``n * cell / total`` rounded down, the remainder going to the largest fractional parts (ties by cell
    name), then each cell is sampled with one seeded RNG in sorted cell order.

    :param rows: Mixture rows.
    :param n: Sample size (capped at ``len(rows)``).
    :param seed: RNG seed.
    :returns: (sorted row indices, ``"kind|family|mask_level" -> {rows, picked}``).
    :rtype: Tuple[List[int], Dict[str, dict]]
    """
    cells: Dict[tuple, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        cells[(r["kind"], r["family"], r["mask_level"])].append(i)
    n = min(n, len(rows))
    quota = {c: n * len(v) / len(rows) for c, v in cells.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    for c in sorted(cells, key=lambda c: (-(quota[c] - alloc[c]), c))[:n - sum(alloc.values())]:
        alloc[c] += 1
    rng = random.Random(seed)
    picked: List[int] = []
    for c in sorted(cells):
        picked += rng.sample(cells[c], alloc[c])
    return sorted(picked), {"|".join(c): {"rows": len(cells[c]), "picked": alloc[c]} for c in sorted(cells)}


def prepare_subset(paths: SimpleNamespace, args: argparse.Namespace, pool: Dict[str, dict],
                   held: set) -> Tuple[List[dict], dict]:
    """Write (or reopen) the stratified training subset and check it against the hold-out.

    :param paths: Run paths.
    :param args: Parsed CLI arguments (``n_train``, ``seed``, ``pkg``).
    :param pool: Train pool ``doc_id -> {text, text_hash, subjects}``.
    :param held: Held-out record ids.
    :returns: (subset rows, subset meta).
    :rtype: Tuple[List[dict], dict]
    :raises SystemExit: When a stored subset was drawn with other settings.
    """
    src = Path(args.pkg) / "train_mix_v3.jsonl"
    rows_path, meta_path = paths.subset / "train_rows.jsonl", paths.subset / "meta.json"
    want = {"source": str(src), "source_md5": md5_file(src), "n_train": args.n_train, "seed": args.seed}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if {k: meta.get(k) for k in want} != want:
            raise SystemExit(f"{paths.subset} was drawn with other settings ({meta_path}); remove it to start over")
        rows = te._read_jsonl(rows_path)
    else:
        mix = te._read_jsonl(src)
        idx, cells = stratified_indices(mix, args.n_train, args.seed)
        rows = [dict(mix[i], mix_index=i) for i in idx]
        write_jsonl(rows_path, rows)
        meta = {**want, "n_rows": len(rows), "n_mix": len(mix), "cells": cells,
                "by_kind": dict(Counter(r["kind"] for r in rows)),
                "by_family": dict(Counter(r["family"] for r in rows)),
                "by_mask_level": dict(Counter(r["mask_level"] for r in rows))}
        step0.write_json(meta_path, meta)
        del mix
    te.check_data({"held_out": held, "pool": pool, "train": rows})
    return rows, meta


def reduced_pool(eval_pool: Sequence[dict], n_other: int, seed: int) -> dict:
    """Every held-out eligible record plus a seeded sample of the other eligible records.

    :param eval_pool: Frozen pool rows (``held_out`` flag per record).
    :param n_other: Records sampled from the non-held-out side.
    :param seed: RNG seed.
    :returns: ``{doc_ids (frozen pool order), n_held_out, n_other, n_other_available, seed}``.
    :rtype: dict
    """
    held = [r["doc_id"] for r in eval_pool if r["held_out"]]
    other = sorted(r["doc_id"] for r in eval_pool if not r["held_out"])
    keep = set(held) | set(random.Random(seed).sample(other, min(n_other, len(other))))
    return {"doc_ids": [r["doc_id"] for r in eval_pool if r["doc_id"] in keep], "n_held_out": len(held),
            "n_other": len(keep) - len(held), "n_other_available": len(other), "n_frozen_pool": len(eval_pool),
            "seed": seed}


def prepare_pool(paths: SimpleNamespace, eval_dir: Path, args: argparse.Namespace) -> dict:
    """Write (or reopen) the reduced eval pool.

    :param paths: Run paths.
    :param eval_dir: Eval dir the pool is drawn from.
    :param args: Parsed CLI arguments (``n_other``, ``seed``).
    :returns: Output of :func:`reduced_pool` plus ``eval_dir``.
    :rtype: dict
    :raises SystemExit: When a stored pool was drawn with other settings.
    """
    path = paths.pool / "pool.json"
    if path.exists():
        pool = json.loads(path.read_text())
        if (pool["eval_dir"], pool["seed"], pool["n_other_requested"]) != (str(eval_dir), args.seed, args.n_other):
            raise SystemExit(f"{path} was drawn with other settings; remove {paths.pool} to start over")
        return pool
    ev = eval_v3.load_eval_dir(eval_dir)
    pool = {**reduced_pool(ev["eval_pool"], args.n_other, args.seed), "eval_dir": str(eval_dir),
            "n_other_requested": args.n_other}
    paths.pool.mkdir(parents=True, exist_ok=True)
    step0.write_json(path, pool)
    return pool


def smoke_eval_dir(paths: SimpleNamespace, n: int, seed: int) -> Path:
    """Reduced eval dir of the smoke test (``step0_local.smoke_selection``: every eval section on ~``n`` records).

    :param paths: Run paths.
    :param n: Target records.
    :param seed: RNG seed.
    :returns: The eval dir.
    :rtype: Path
    """
    if not (paths.eval_dir / "eval_pool.jsonl").exists():
        ev = eval_v3.load_eval_dir(PKG / "eval")
        step0.write_smoke_eval_dir(ev, step0.smoke_selection(ev, n, seed), paths.eval_dir)
    return paths.eval_dir


def load_original(path: Path, keep: set) -> Tuple[List[dict], Dict[str, str]]:
    """Original (production) text of selected records, hashed like :func:`step0_local.load_semantic` output.

    :param path: ``corpus_original.jsonl``.
    :param keep: Doc ids.
    :returns: (rows ``{doc_id, text_hash}`` in corpus order, ``text_hash -> text``).
    :rtype: Tuple[List[dict], Dict[str, str]]
    """
    rows, texts = [], {}
    for r in te._read_jsonl(path):
        if r["doc_id"] in keep:
            h = hashlib.md5(r["text"].encode("utf-8")).hexdigest()
            texts.setdefault(h, r["text"])
            rows.append({"doc_id": r["doc_id"], "text_hash": h})
    return rows, texts


# ---------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------
def load_model(dtype_name: str, tiny: bool, dev: str):
    """The pinned base model (or the tiny random smoke model) as a SentenceTransformer.

    The tiny model's random weights are rounded to bf16 first, like the real checkpoint (stored in bf16), so the bf16
    training base and the fp32 embedding base hold identical values in every phase.

    :param dtype_name: torch dtype name of the weights (``float32`` for embedding, ``bfloat16`` for the LoRA base).
    :param tiny: Use ``train_embedder_v3.tiny_model`` (smoke tests).
    :param dev: Device.
    :returns: SentenceTransformer on ``dev``.
    """
    from sentence_transformers import SentenceTransformer

    dtype = getattr(torch, dtype_name)
    t0 = time.time()
    if tiny:
        model = te.tiny_model(hidden=64, layers=2, seed=0, max_seq=MAX_SEQ)
        model.to(torch.bfloat16)
        model.to(device=dev, dtype=dtype)
    else:
        spec = TEXT_MODELS[MODEL_KEY]
        if (spec["id"], spec["revision"]) != (te.BASE_MODEL, te.BASE_REVISION):
            raise ValueError(f"embed_utils pins {spec}, the trainer {te.BASE_MODEL}@{te.BASE_REVISION}")
        model = SentenceTransformer(spec["id"], revision=spec["revision"], device=dev, model_kwargs={"dtype": dtype})
    te.check_text_path(model)
    log(f"loaded {'tiny random Qwen3' if tiny else te.BASE_MODEL} ({dtype_name}) on {dev} in {time.time() - t0:.0f} s, "
        f"footprint {child_rss_gb(os.getpid()):.2f} GB")
    return model


def set_pad_multiple(model, multiple: int) -> None:
    """Pad every batch to a multiple of ``multiple`` tokens (few distinct shapes on MPS; padding is masked).

    Same setting as ``step0_local.load_encoder``.

    :param model: SentenceTransformer.
    :param multiple: Padding multiple (1 = off).
    """
    if multiple <= 1:
        return
    module = model[0]
    kwargs = dict(module.processing_kwargs)
    kwargs["text"] = {**kwargs.get("text", {}), "pad_to_multiple_of": multiple}
    module.processing_kwargs = kwargs
    width = module.preprocess(["probe"])["input_ids"].shape[1]
    if width % multiple:
        raise AssertionError(f"pad_to_multiple_of not applied (width {width})")


def attach_lora(model, args: argparse.Namespace) -> dict:
    """Wrap the model's transformer in a trainable LoRA (``get_peft_model`` upcasts the adapter weights to fp32).

    :param model: SentenceTransformer (bf16 base).
    :param args: Parsed CLI arguments (``lora_r``, ``lora_alpha``, ``lora_dropout``, ``train_seed``).
    :returns: Output of :func:`lora_report`.
    :rtype: dict
    """
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(args.train_seed)  # LoRA A init: identical across relaunches (a resume then loads the weights)
    config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_r, lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout, target_modules=list(LORA_TARGETS), bias="none")
    model[0].model = get_peft_model(model[0].model, config)
    return lora_report(model)


def lora_report(model) -> dict:
    """Check the LoRA layout: only LoRA weights train, all in fp32, one adapter on every target of every layer.

    :param model: SentenceTransformer with a peft-wrapped transformer.
    :returns: Parameter counts and dtypes.
    :rtype: dict
    :raises ValueError: On any violation.
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise ValueError(f"trainable parameters other than LoRA: {[n for n, _ in trainable if 'lora_' not in n][:3]}")
    if any(p.dtype != torch.float32 for _, p in trainable):
        raise ValueError("LoRA weights must be fp32 (master weights of the adapter)")
    n_layers = model[0].model.get_base_model().config.num_hidden_layers
    adapted = [n for n, m in model.named_modules() if n.endswith(LORA_TARGETS) and hasattr(m, "lora_A")]
    if len(adapted) != n_layers * len(LORA_TARGETS):
        raise ValueError(f"{len(adapted)} adapted modules, expected {n_layers} x {len(LORA_TARGETS)}")
    return {"trainable_params": sum(p.numel() for _, p in trainable), "trainable_tensors": len(trainable),
            "frozen_params": sum(p.numel() for _, p in frozen),
            "frozen_dtypes": dict(Counter(str(p.dtype) for _, p in frozen)), "adapted_modules": len(adapted),
            "layers": n_layers, "targets": list(LORA_TARGETS)}


def load_lora_weights(model, path: Path) -> None:
    """Load a saved adapter (``adapter_model.safetensors``) into the model's LoRA weights.

    :param model: SentenceTransformer with a peft-wrapped transformer of the same LoRA config.
    :param path: Directory written by ``PeftModel.save_pretrained``.
    :raises ValueError: When the file does not fill every LoRA weight or holds unknown keys.
    """
    from peft import load_peft_weights, set_peft_model_state_dict

    result = set_peft_model_state_dict(model[0].model, load_peft_weights(str(path), device="cpu"))
    missing = [k for k in result.missing_keys if "lora_" in k]
    if missing or result.unexpected_keys:
        raise ValueError(f"adapter {path}: missing {missing[:3]}, unexpected {list(result.unexpected_keys)[:3]}")


def merge_lora(model, adapter_dir: Path, probe_texts: Sequence[str], dtype_name: str) -> dict:
    """Apply a saved adapter to a base model and merge it into the weights, checking merged == unmerged vectors.

    :param model: Base SentenceTransformer (not wrapped).
    :param adapter_dir: Saved adapter.
    :param probe_texts: A few texts encoded before and after merging.
    :param dtype_name: Base dtype (sets the tolerance, :data:`MERGE_MIN_COS`).
    :returns: ``{n, min_cos, ok, adapter_md5}``.
    :rtype: dict
    """
    from peft import PeftModel

    model[0].model = PeftModel.from_pretrained(model[0].model, str(adapter_dir), is_trainable=False)
    if any(p.dtype != torch.float32 for n, p in model.named_parameters() if "lora_" in n):
        raise ValueError("adapter weights were not loaded in fp32")
    before = model.encode(list(probe_texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=4)
    model[0].model = model[0].model.merge_and_unload()
    if any("lora_" in n for n, _ in model.named_parameters()):
        raise ValueError("LoRA weights left after merge_and_unload")
    after = model.encode(list(probe_texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=4)
    cos = (before * after).sum(1)
    out = {"n": len(probe_texts), "min_cos": round(float(cos.min()), 6), "threshold": MERGE_MIN_COS[dtype_name],
           "adapter_md5": md5_file(adapter_dir / "adapter_model.safetensors")}
    out["ok"] = bool(out["min_cos"] >= out["threshold"])
    log(f"merge check: {out}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------
class Fp32MaskedCachedMNRL(te.MaskedCachedMNRL):
    """:class:`train_embedder_v3.MaskedCachedMNRL` with the similarity / softmax in fp32.

    The frozen base runs in bf16 without autocast, so its sentence embeddings are bf16; bf16 cosines times the scale
    (20) would quantise the logits to ~0.08. The gradients still reach the cached bf16 embeddings (``.float()`` is
    differentiable), which is all GradCache needs.
    """

    def calculate_loss(self, reps: List[List[torch.Tensor]], with_backward: bool = False) -> torch.Tensor:
        """Upcast the embeddings, then the masked InfoNCE of the parent.

        :param reps: Embeddings per column, each a list of mini-batch tensors.
        :param with_backward: Backpropagate each mini-batch's loss to the cached embeddings.
        :returns: Loss.
        :rtype: torch.Tensor
        """
        return super().calculate_loss([[r.float() for r in col] for col in reps], with_backward)


class LoraCheckpointTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer whose checkpoints hold the LoRA weights only (the base never changes).

    HF Trainer writes optimizer, scheduler, RNG and trainer state next to them and restores all of it on
    ``resume_from_checkpoint``; MPS RNG (not covered by HF) is saved here too.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Save the adapter (+ the MPS RNG state).

        :param output_dir: Checkpoint directory.
        :param state_dict: Unused (the adapter is read from the model).
        """
        out = Path(output_dir or self.args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model[0].model.save_pretrained(str(out))
        if self.model.device.type == "mps":
            torch.save(torch.mps.get_rng_state(), out / "mps_rng_state.pt")

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """Load the adapter (+ the MPS RNG state) of a checkpoint.

        :param resume_from_checkpoint: Checkpoint directory.
        :param model: Unused.
        """
        load_lora_weights(self.model, Path(resume_from_checkpoint))
        rng = Path(resume_from_checkpoint) / "mps_rng_state.pt"
        if self.model.device.type == "mps" and rng.exists():
            torch.mps.set_rng_state(torch.load(rng))
        log(f"resumed LoRA weights from {resume_from_checkpoint}")


class PilotCallback(TrainerCallback):
    """Step timing + ETA, footprint watch (save and exit 99 above the soft limit), cumulative mask stats per checkpoint.

    Mask stats, training seconds and step times of earlier sessions come from the resumed checkpoint's
    ``pilot_state.json``; each save writes the cumulative values, so they always match the checkpoint's steps.

    While training runs, SIGTERM (the guard's kill: SIGTERM, then SIGKILL 20 s later) only sets a flag; the step in
    flight finishes, the trainer checkpoints it and stops, so a guard kill costs at most that step instead of up to
    ``save_steps`` steps. If the SIGKILL lands first, the half-written checkpoint lacks ``pilot_state.json`` and
    :func:`latest_checkpoint` falls back to the previous one. Outside training SIGTERM keeps its default action.
    """

    def __init__(self, losses: Dict[str, te.MaskedCachedMNRL], prior: dict, soft_max_gb: float,
                 stop_after_step: int, dev: str) -> None:
        """Build the callback.

        :param losses: Dataset name -> loss object (their ``stats`` count this session).
        :param prior: Cumulative state of the resumed checkpoint (``{}`` for a fresh run).
        :param soft_max_gb: Footprint above which the trainer saves and the process exits 99 (0 = off).
        :param stop_after_step: Testing: save and exit 99 once this step is reached (0 = off).
        :param dev: Device (MPS cache is emptied at every logging step).
        """
        self.dev = dev
        self.losses = losses
        self.prior = prior
        self.soft_max_gb = soft_max_gb
        self.stop_after_step = stop_after_step
        self.t_session = time.time()
        self.t_step = 0.0
        self.step_seconds: List[float] = []
        self.start_step = 0
        self.footprint = 0.0
        self.restart_reason: Optional[str] = None
        self.sigterm = False
        self._prev_sigterm = None

    def _on_sigterm(self, signum: int, frame) -> None:
        """SIGTERM handler while training: request a checkpoint at the end of the current step.

        :param signum: Signal number.
        :param frame: Current stack frame (unused).
        """
        self.sigterm = True
        log("SIGTERM received: checkpointing at the end of this step, then exiting")

    def _stop(self, control, reason: str) -> None:
        """Ask the trainer to checkpoint now and stop; the process then exits 99.

        :param control: TrainerControl.
        :param reason: Log text.
        """
        control.should_save = True
        control.should_training_stop = True
        self.restart_reason = reason

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Remember where this session starts.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control.
        :param kwargs: Unused.
        """
        self.start_step = state.global_step
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_sigterm)
        log(f"training session starts at step {state.global_step} of {state.max_steps}")

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """Give SIGTERM its previous handler back (embedding phases rely on the default: terminate).

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control.
        :param kwargs: Unused.
        """
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
            self._prev_sigterm = None

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        """Start the step clock.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control.
        :param kwargs: Unused.
        """
        self.t_step = time.time()

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Time the step; every logging interval check the footprint (before HF decides whether to save).

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control.
        :param kwargs: Unused.
        """
        self.step_seconds.append(time.time() - self.t_step)
        step, last = state.global_step, state.global_step >= state.max_steps
        if step % args.logging_steps == 0 or last:
            if self.dev == "mps":
                torch.mps.empty_cache()
            self.footprint = child_rss_gb(os.getpid())
            if self.soft_max_gb and self.footprint > self.soft_max_gb and not last:
                self._stop(control, f"footprint {self.footprint:.2f} GB > soft limit {self.soft_max_gb} GB")
        if self.stop_after_step and step >= self.stop_after_step > self.start_step and not last:
            self._stop(control, f"--stop-after-step {self.stop_after_step} (resume test)")
        if self.sigterm and not last:
            self._stop(control, f"SIGTERM (guard kill) during step {step}")

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """One status line per logged step: loss, s/step, ETA, footprint.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control.
        :param logs: The values HF just logged.
        :param kwargs: Unused.
        """
        if not logs or "loss" not in logs or not self.step_seconds:
            return
        recent = statistics.median(self.step_seconds[-10:])
        log(f"step {state.global_step}/{state.max_steps}: loss {logs['loss']:.4f}, lr {logs.get('learning_rate', 0):.2e},"
            f" {recent:.2f} s/step (median of last {min(10, len(self.step_seconds))}), "
            f"ETA {recent * (state.max_steps - state.global_step) / 60:.1f} min, footprint {self.footprint:.2f} GB")

    def cumulative(self) -> dict:
        """Prior + this session's mask stats, training seconds and step times.

        :returns: State dict (written as ``pilot_state.json``).
        :rtype: dict
        """
        stats = {name: Counter(self.prior.get("mask_stats", {}).get(name, {})) for name in self.losses}
        for name, loss in self.losses.items():
            stats[name].update(loss.stats)
        return {"mask_stats": {k: dict(v) for k, v in stats.items()},
                "train_seconds": round(self.prior.get("train_seconds", 0.0) + time.time() - self.t_session, 1),
                "step_seconds": self.prior.get("step_seconds", []) + [round(s, 3) for s in self.step_seconds],
                "sessions": self.prior.get("sessions", 0) + 1}

    def on_save(self, args, state, control, **kwargs) -> None:
        """Write the cumulative state into the checkpoint just saved (its completion marker).

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control.
        :param kwargs: Unused.
        """
        step0.write_json(Path(args.output_dir) / f"checkpoint-{state.global_step}" / "pilot_state.json",
                         self.cumulative())


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Newest complete checkpoint (every file of :data:`CKPT_FILES`); newer incomplete ones are removed.

    :param ckpt_dir: Trainer output dir.
    :returns: Checkpoint dir or None.
    :rtype: Optional[Path]
    """
    if not ckpt_dir.is_dir():
        return None
    dirs = sorted((p for p in ckpt_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
                  key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for d in dirs:
        if all((d / f).exists() for f in CKPT_FILES):
            return d
        log(f"removing incomplete checkpoint {d}")
        shutil.rmtree(d)
    return None


def expected_steps(datasets: dict, batch: int) -> int:
    """Optimizer steps of one epoch (one partial batch per dataset, as the batch samplers count them).

    :param datasets: Dataset name -> Dataset.
    :param batch: Batch size.
    :returns: Steps.
    :rtype: int
    """
    return sum(math.ceil(len(d) / batch) for d in datasets.values())


def train_lora(model, rows: List[dict], pool: Dict[str, dict], ckpt_dir: Path, args: argparse.Namespace, dev: str,
               max_steps: int = -1) -> dict:
    """Train the attached LoRA with the v3 datasets, prompts and masked GradCache loss; resume from checkpoints.

    :param model: SentenceTransformer with a trainable LoRA (:func:`attach_lora`).
    :param rows: Training rows (no mined negatives: in-batch masked negatives only).
    :param pool: Train pool (mask ids of d2d anchors).
    :param ckpt_dir: Trainer output dir (``checkpoint-<step>`` dirs).
    :param args: Parsed CLI arguments (training settings).
    :param dev: Device (``cpu`` sets ``use_cpu``).
    :param max_steps: Stop after this many steps (-1 = one epoch).
    :returns: Training info (``restart_reason`` set when the session stopped early for a relaunch).
    :rtype: dict
    """
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers, MultiDatasetBatchSamplers

    model.max_seq_length = args.max_seq
    set_pad_multiple(model, args.train_pad_multiple)
    datasets, prompts, ds_stats = te.build_datasets(rows, pool)
    losses = {name: Fp32MaskedCachedMNRL(model, symmetric=te.kind_of(name) in te.SYMMETRIC_KINDS,
                                         mini_batch_size=args.mini_batch, scale=args.scale) for name in datasets}
    log(json.dumps({"datasets": ds_stats["rows"], "label_width": ds_stats["label_width"],
                    "steps_per_epoch": expected_steps(datasets, args.batch),
                    "prompts": {k: sorted(v) for k, v in prompts.items()}}))
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=args.epochs, max_steps=max_steps,
        per_device_train_batch_size=args.batch, learning_rate=args.lr, warmup_ratio=args.warmup,
        lr_scheduler_type="cosine", bf16=False, optim="adamw_torch", batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL, prompts=prompts,
        logging_steps=args.logging_steps, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        report_to=[], seed=args.train_seed, use_cpu=(dev == "cpu"), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_num_workers=0,
        dataloader_pin_memory=False, disable_tqdm=True)
    resume = latest_checkpoint(ckpt_dir)
    prior = json.loads((resume / "pilot_state.json").read_text()) if resume else {}
    callback = PilotCallback(losses, prior, args.soft_max_gb, args.stop_after_step, dev)
    trainer = LoraCheckpointTrainer(model=model, args=targs, train_dataset=datasets, loss=losses, callbacks=[callback])
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    state = callback.cumulative()
    secs = state["step_seconds"][1:] or state["step_seconds"]  # the first step of a run includes MPS warm-up
    info = {"base_model": te.BASE_MODEL, "base_revision": te.BASE_REVISION, "query_prompt": te.QUERY_PROMPT,
            "global_steps": trainer.state.global_step, "max_steps": trainer.state.max_steps,
            "train_seconds": state["train_seconds"], "sessions": state["sessions"],
            "step_seconds": {"n": len(state["step_seconds"]), "median": round(statistics.median(secs), 3),
                             "mean": round(statistics.mean(secs), 3), "max": round(max(secs), 3)} if secs else {},
            "datasets": ds_stats, "prompts": prompts, "mining": "skipped: in-batch masked negatives only",
            "mask_stats": state["mask_stats"],
            "masked_share": {name: round(s.get("qd_masked", 0) / max(1, s.get("qd_candidates", 0)), 4)
                             for name, s in state["mask_stats"].items()},
            "loss_history": [h for h in trainer.state.log_history if "loss" in h],
            "restart_reason": callback.restart_reason,
            "resumed_from": str(resume) if resume else None}
    del trainer, losses
    release(dev)
    return info


def train_config(args: argparse.Namespace) -> dict:
    """Every setting that changes the trained adapter (a resume with other settings is refused).

    :param args: Parsed CLI arguments.
    :returns: Settings dict.
    :rtype: dict
    """
    keys = ("n_train", "seed", "batch", "mini_batch", "max_seq", "lr", "epochs", "warmup", "scale", "lora_r",
            "lora_alpha", "lora_dropout", "train_seed", "base_dtype", "train_pad_multiple", "save_steps", "tiny")
    return {k: getattr(args, k) for k in keys}


# ---------------------------------------------------------------------------------------------------------------
# embedding (step0_local machinery)
# ---------------------------------------------------------------------------------------------------------------
def queries_done(out: Path, qtexts: List[str]) -> bool:
    """Whether ``out`` already holds the query vectors of exactly these texts.

    :param out: Vector dir.
    :param qtexts: Query texts.
    :returns: True when ``queries.{json,npy}`` match.
    :rtype: bool
    """
    q = out / "queries.json"
    return q.exists() and (out / "queries.npy").exists() and json.loads(q.read_text(encoding="utf-8"))["texts"] == qtexts


def vectors_complete(out: Path, tag: dict, n_records: int) -> bool:
    """Whether ``out`` holds finished vectors of this tag and size.

    :param out: Vector dir.
    :param tag: Expected identity (model, dtype, adapter).
    :param n_records: Expected records.
    :returns: True when ``meta.json`` says complete with the same tag and record count.
    :rtype: bool
    """
    meta_path = out / "meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return bool(meta.get("complete") and meta.get("tag") == tag and meta.get("n_records") == n_records)


def embed_phase(out: Path, rows: List[dict], texts: Dict[str, str], qtexts: Optional[List[str]],
                load_fn: Callable[[], Tuple[object, dict]], tag: dict, args: argparse.Namespace,
                canary: bool) -> dict:
    """Embed documents (raw) and optionally the eval queries (production prompt) into a resumable vector dir.

    Reuses ``step0_local``: ``contract_canary``, ``embed_queries``, ``embed_unique`` (sharded, tmp + rename, OOM
    back-off, exit 99 above ``--soft-max-gb``) and ``expand`` (the ``load_vectors`` layout).

    :param out: Vector dir.
    :param rows: Records ``{doc_id, text_hash}``.
    :param texts: ``text_hash -> text``.
    :param qtexts: Query texts, or None (documents only).
    :param load_fn: Returns (model ready for the production window, load info).
    :param tag: Identity of the vectors (model, dtype, adapter); a dir holding another tag is refused.
    :param args: Parsed CLI arguments (batch bounds, ``canary``).
    :param canary: Run the contract canary against the stored production vectors first.
    :returns: Vector meta.
    :rtype: dict
    :raises SystemExit: Wrong tag, failed canary or failed merge check.
    """
    meta_path, tag_path = out / "meta.json", out / "tag.json"
    if vectors_complete(out, tag, len(rows)):
        return json.loads(meta_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    if tag_path.exists() and json.loads(tag_path.read_text()) != tag:
        raise SystemExit(f"{out} holds vectors of another model ({tag_path}); remove it to start over")
    step0.write_json(tag_path, tag)
    lock = step0.acquire_lock(out)
    model, load_info = load_fn()
    enc = CachedTextEncoder(MODEL_KEY, max_seq_length=MAX_SEQ)
    enc._model = model
    if enc.prefix("query", DEFAULT_TASK) != te.QUERY_PROMPT:
        raise AssertionError("embed_utils query prefix differs from the trainer's QUERY_PROMPT")
    canary_res = None
    if canary and args.canary:
        cpath = out / "canary.json"
        canary_res = json.loads(cpath.read_text()) if cpath.exists() else step0.contract_canary(enc, args.canary, args)
        step0.write_json(cpath, canary_res)
        if not canary_res["ok"]:
            raise SystemExit(f"contract canary failed: {canary_res}")
    if qtexts is not None and not queries_done(out, qtexts):
        step0.embed_queries(enc, qtexts, out, args)
    t0 = time.time()
    plan = step0.embed_unique(enc, texts, out, args)
    step0.expand(out, rows, plan)
    meta = {"complete": True, "written": datetime.now().isoformat(timespec="seconds"), "tag": tag,
            "n_records": len(rows), "n_unique_texts": len(plan["hashes"]), "n_tokens": int(sum(plan["lengths"])),
            "n_shards_unique": len(plan["bounds"]), "device_last_session": str(model.device),
            "seconds_last_session": round(time.time() - t0, 1), "canary": canary_res, **load_info,
            "batching": {k: getattr(args, k) for k in ("pad_multiple", "max_batch", "max_batch_tokens", "max_attn",
                                                       "shard_size", "shard_tokens")}}
    step0.write_json(meta_path, meta)
    lock.unlink(missing_ok=True)
    del enc, model
    release(str(meta["device_last_session"]).split(":")[0])
    return meta


def base_loader(args: argparse.Namespace, dev: str) -> Callable[[], Tuple[object, dict]]:
    """Loader of the fp32 base model at the production window.

    :param args: Parsed CLI arguments.
    :param dev: Device.
    :returns: Zero-argument loader.
    :rtype: Callable[[], Tuple[object, dict]]
    """
    def load() -> Tuple[object, dict]:
        model = load_model("float32", args.tiny, dev)
        model.max_seq_length = MAX_SEQ
        set_pad_multiple(model, args.pad_multiple)
        return model, {}

    return load


def tuned_loader(args: argparse.Namespace, dev: str, adapter_dir: Path,
                 probe_texts: Sequence[str]) -> Callable[[], Tuple[object, dict]]:
    """Loader of the fp32 base with the trained LoRA merged in (merge checked on ``probe_texts``).

    :param args: Parsed CLI arguments.
    :param dev: Device.
    :param adapter_dir: Saved adapter.
    :param probe_texts: Texts of the merge check.
    :returns: Zero-argument loader.
    :rtype: Callable[[], Tuple[object, dict]]
    """
    def load() -> Tuple[object, dict]:
        model = load_model("float32", args.tiny, dev)
        check = merge_lora(model, adapter_dir, probe_texts, "float32")
        if not check["ok"]:
            raise SystemExit(f"merge check failed: {check}")
        model.max_seq_length = MAX_SEQ
        set_pad_multiple(model, args.pad_multiple)
        return model, {"merge_check": check}

    return load


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def query_fn(vec_dir: Path) -> Callable[[List[str]], np.ndarray]:
    """Lookup of stored query vectors (``queries.{json,npy}``) as an ``eval_v3`` query function.

    :param vec_dir: Vector dir.
    :returns: Function mapping query texts to their vectors.
    :rtype: Callable[[List[str]], np.ndarray]
    """
    q = json.loads((vec_dir / "queries.json").read_text(encoding="utf-8"))
    mat = np.load(vec_dir / "queries.npy")
    if not np.isfinite(mat).all():
        raise ValueError(f"non-finite query vectors in {vec_dir}")
    vec = dict(zip(q["texts"], mat))

    def fn(texts: List[str]) -> np.ndarray:
        return np.stack([vec[t] for t in texts])

    return fn


def subset_vectors(vec_dir: Path, doc_ids: Sequence[str]) -> np.ndarray:
    """Rows of a stored vector dir (``ids.json`` + shards) for the given ids, in that order.

    :param vec_dir: Vector dir in the ``load_vectors`` layout.
    :param doc_ids: Wanted ids (all must be present).
    :returns: (len(doc_ids), d) matrix.
    :rtype: np.ndarray
    """
    pos = {d: i for i, d in enumerate(json.loads((vec_dir / "ids.json").read_text()))}
    missing = [d for d in doc_ids if d not in pos]
    if missing:
        raise ValueError(f"{len(missing)} ids missing from {vec_dir}, e.g. {missing[:3]}")
    mat = step0.stored_rows(vec_dir, [pos[d] for d in doc_ids])
    if not np.isfinite(mat).all():
        raise ValueError(f"non-finite values in {vec_dir}")
    return mat


def score(doc_ids: Sequence[str], mat: np.ndarray, qfn: Callable, eval_dir: Path, corpus_rows: Optional[dict],
          tsp: Dict[str, List[str]], n_resamples: int) -> dict:
    """``eval_v3.evaluate`` with the trained-positive memorisation section.

    :param doc_ids: Pool ids (one per row of ``mat``).
    :param mat: Document vectors.
    :param qfn: Query function.
    :param eval_dir: Eval dir.
    :param corpus_rows: ``doc_id -> {text_hash}`` of the embedded text (None for production text).
    :param tsp: ``train_embedder_v3.trained_subject_positives`` of the training subset.
    :param n_resamples: Bootstrap resamples.
    :returns: Result.
    :rtype: dict
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious Accelerate warnings on unit vectors
        return eval_v3.evaluate(list(doc_ids), mat, qfn, eval_dir, corpus_rows=corpus_rows, n_resamples=n_resamples,
                                trained_subject_positives=tsp)


def extra_paired(reports: Dict[str, dict], n_resamples: int) -> Dict[str, dict]:
    """Paired CIs that ``compare_reports`` leaves out: gate P@10 / seen AP vs production, known-item RR by type.

    :param reports: ``base_original``, ``base_semantic``, ``tuned_semantic`` results.
    :param n_resamples: Bootstrap resamples.
    :returns: ``name -> paired_bootstrap result``.
    :rtype: Dict[str, dict]
    """
    o, s, t = reports["base_original"], reports["base_semantic"], reports["tuned_semantic"]
    out: Dict[str, dict] = {}
    for group, metric in (("held_out", "P@10"), ("seen", "AP"), ("linked", "AP")):
        if t["subjects"][group]["per_query"]:
            out[f"tuned_semantic - base_original | {group} {metric}"] = eval_v3.compare(o, t, group, metric, n_resamples)
    if t["subjects"]["held_out"]["per_query"]:
        out["base_semantic - base_original | held_out P@10 (step 0)"] = eval_v3.compare(o, s, "held_out", "P@10",
                                                                                        n_resamples)
    for a_label, a in (("base_semantic", s), ("base_original", o)):
        for qtype in sorted(t["known_item"]["by_type"]):
            out[f"tuned_semantic - {a_label} | known_item {qtype} RR"] = eval_v3.paired_bootstrap(
                step0.known_item_clusters(a, qtype, "RR"), step0.known_item_clusters(t, qtype, "RR"), n_resamples)
    return out


def headline(reports: Dict[str, dict], paired: Dict[str, dict]) -> dict:
    """The numbers to read first: per-model headline metrics and the key paired differences.

    :param reports: Label -> result.
    :param paired: ``compare_reports`` paired CIs merged with :func:`extra_paired`.
    :returns: ``{models: {label: metrics}, paired: {name: CI}}``.
    :rtype: dict
    """
    models = {}
    for lb, r in reports.items():
        sub, ki = r["subjects"], r["known_item"]
        models[lb] = {"gate_AP": sub["gate"].get("AP"), "gate_P@10": sub["gate"].get("P@10"),
                      "gate_R@100": sub["gate"].get("R@100"),
                      "gate_per_subject_AP": {k: v["AP"] for k, v in sub["held_out"]["per_subject"].items()},
                      "linked_AP": sub["linked"]["macro"].get("AP"), "seen_AP": sub["seen"]["macro"].get("AP"),
                      "seen_n_subjects": sub["seen"]["macro"].get("n_subjects"),
                      "known_item_MRR": ki["all"].get("MRR"),
                      "known_item_MRR_by_type": {k: v.get("MRR") for k, v in ki["by_type"].items()},
                      "memorisation_gap": r["memorisation"]["macro"].get("gap"),
                      "memorisation_trained_gap": r.get("memorisation_trained", {}).get("macro", {}).get("gap"),
                      "collection_lift": r["collection"]["all"].get("lift"),
                      "collection_lift_long": r["collection"]["long"].get("lift"),
                      "hubness_ratio": r.get("hubness", {}).get("ratio")}
    keys = {"gate AP: tuned - base_semantic": "tuned_semantic - base_semantic | held_out AP",
            "gate P@10: tuned - base_semantic": "tuned_semantic - base_semantic | held_out P@10",
            "gate AP: tuned - base_original": "tuned_semantic - base_original | held_out AP (vs production today)",
            "gate P@10: tuned - base_original": "tuned_semantic - base_original | held_out P@10",
            "gate AP: base_semantic - base_original (step 0)": "base_semantic - base_original | held_out AP (step 0)",
            "seen AP: tuned - base_semantic": "tuned_semantic - base_semantic | seen AP",
            "seen AP: tuned - base_original": "tuned_semantic - base_original | seen AP",
            "known-item RR: tuned - base_semantic": "tuned_semantic - base_semantic | known_item RR",
            "known-item RR: tuned - base_original": "tuned_semantic - base_original | known_item RR (vs production today)",
            "memorisation (trained positives) gap: tuned - base_semantic":
                "tuned_semantic - base_semantic | memorisation_trained gap (net of baseline gap)"}
    return {"models": models, "paired": {k: paired[v] for k, v in keys.items() if v in paired}}


def vector_shift(base_dir: Path, tuned_dir: Path) -> dict:
    """How far the adapter moved the document vectors: cosine of each record's tuned vs base vector.

    :param base_dir: Base vector dir.
    :param tuned_dir: Tuned vector dir.
    :returns: ``{n, mean, p05, min}``.
    :rtype: dict
    """
    ids_b, mb = load_vectors(base_dir)
    ids_t, mt = load_vectors(tuned_dir)
    if ids_b != ids_t:
        raise ValueError("base and tuned vector dirs hold different records")
    cos = (mb * mt).sum(1)
    return {"n": len(cos), "mean": round(float(cos.mean()), 4), "p05": round(float(np.quantile(cos, 0.05)), 4),
            "min": round(float(cos.min()), 4)}


def eval_phase(paths: SimpleNamespace, pool: dict, rows: List[dict], sem_rows: List[dict], eval_dir: Path,
               orig_dir: Path, args: argparse.Namespace) -> dict:
    """Score the three models on the reduced pool, compare them, and calibrate the pool effect.

    :param paths: Run paths.
    :param pool: Reduced pool.
    :param rows: Training subset (trained subject positives).
    :param sem_rows: ``{doc_id, text_hash}`` of the semantic text the vectors were embedded from.
    :param eval_dir: Eval dir.
    :param orig_dir: Production-text vector dir (``base_original``).
    :param args: Parsed CLI arguments (``n_resamples``).
    :returns: ``{reports, comparison, extra, pool_calibration, vector_shift}``.
    :rtype: dict
    """
    ids = pool["doc_ids"]
    tsp = te.trained_subject_positives(rows)
    q_base, q_tuned = query_fn(paths.vec_base), query_fn(paths.vec_tuned)
    t0 = time.time()
    reports = {"base_original": score(ids, subset_vectors(orig_dir, ids), q_base, eval_dir, None, tsp,
                                      args.n_resamples)}
    corpus_rows = {r["doc_id"]: {"text_hash": r["text_hash"]} for r in sem_rows}  # eval_v3 checks them vs the pool
    for label, vec_dir, qfn in (("base_semantic", paths.vec_base, q_base), ("tuned_semantic", paths.vec_tuned, q_tuned)):
        vids, mat = load_vectors(vec_dir)
        reports[label] = score(vids, mat, qfn, eval_dir, corpus_rows, tsp, args.n_resamples)
        if reports[label]["pool"]["text_hash_mismatch"]:
            raise ValueError(f"{label}: {reports[label]['pool']['text_hash_mismatch']} vectors are not of the frozen "
                             f"semantic text")
    for key in ("n_pool_scored", "n_missing_vectors", "n_query_texts"):
        vals = {lb: r["pool"][key] for lb, r in reports.items()}
        if len(set(vals.values())) != 1:
            raise ValueError(f"pools differ on {key}: {vals}")
    if reports["base_original"]["pool"]["n_pool_scored"] != len(ids):
        raise ValueError(f"scored {reports['base_original']['pool']['n_pool_scored']} of {len(ids)} pool records")
    paths.reports.mkdir(parents=True, exist_ok=True)
    for lb, r in reports.items():
        step0.write_json(paths.reports / f"{lb}.json", r)
    comparison = te.compare_reports(reports, baseline="base_semantic", candidate="tuned_semantic",
                                    n_resamples=args.n_resamples)
    extra = extra_paired(reports, args.n_resamples)
    # pool calibration: the base model on production text over the FULL frozen pool vs the reduced one (no compute)
    full_ids = [r["doc_id"] for r in eval_v3.load_eval_dir(eval_dir)["eval_pool"]]
    full = score(full_ids, subset_vectors(orig_dir, full_ids), q_base, eval_dir, None, tsp, args.n_resamples)
    step0.write_json(paths.reports / "base_original_fullpool.json", full)
    calib = {"full_pool": eval_v3.summary(full), "reduced_pool": eval_v3.summary(reports["base_original"]),
             "note": "base model, production text: the same model on both pools; the difference is the pool effect"}
    log(f"eval: {time.time() - t0:.0f} s")
    return {"reports": reports, "comparison": comparison, "extra": extra, "pool_calibration": calib,
            "vector_shift": vector_shift(paths.vec_base, paths.vec_tuned)}


# ---------------------------------------------------------------------------------------------------------------
# real-weight check + MPS bench
# ---------------------------------------------------------------------------------------------------------------
def kind_sample(mix_path: Path, per_kind: Dict[str, int], seed: int) -> List[dict]:
    """A few seeded mixture rows per kind (the real-weight check).

    :param mix_path: ``train_mix_v3.jsonl``.
    :param per_kind: Kind -> rows.
    :param seed: RNG seed.
    :returns: Rows.
    :rtype: List[dict]
    """
    by_kind: Dict[str, List[dict]] = defaultdict(list)
    for r in te._read_jsonl(mix_path):
        by_kind[r["kind"]].append(r)
    rng = random.Random(seed)
    return [r for k, n in sorted(per_kind.items()) for r in rng.sample(by_kind[k], n)]


def check_real_weights(args: argparse.Namespace, dev: str) -> dict:
    """Two optimizer steps of the pilot's training path on the REAL 0.6B weights (bf16) + LoRA, then the merge path.

    Asserts: finite losses, every LoRA tensor moved (warmup is off here: with the pilot's warmup step 1 has lr 0, and
    ``lora_A`` only gets a gradient once the zero-initialised ``lora_B`` has moved), the frozen base did not, LoRA
    fp32 / base bf16, both checkpoints complete and their adapter round-trips into the live model, merged ==
    unmerged vectors on the adapter applied to a freshly loaded base (bf16 here, to stay far inside the 3 GB cap).

    :param args: Parsed CLI arguments (overridden to batch 4, mini-batch 2, max_seq 64, no warmup, a checkpoint every
        step).
    :param dev: Device (``cpu`` under ``guard --small``).
    :returns: Check result (also written to ``CHECK_DIR/check_result.json``).
    :rtype: dict
    """
    if CHECK_DIR.exists():
        shutil.rmtree(CHECK_DIR)
    CHECK_DIR.mkdir(parents=True)
    a = argparse.Namespace(**{**vars(args), "batch": 4, "mini_batch": 2, "max_seq": 64, "save_steps": 1,
                              "logging_steps": 1, "warmup": 0.0, "soft_max_gb": 0.0, "stop_after_step": 0})
    pool = {r["doc_id"]: r for r in te._read_jsonl(Path(args.pkg) / "train_pool.jsonl")}
    held = set(json.loads((Path(args.pkg) / "eval" / "held_out_records.json").read_text())["held_out"])
    rows = kind_sample(Path(args.pkg) / "train_mix_v3.jsonl", {"q2d": 8, "d2d": 4, "t2t": 4}, args.seed)
    te.check_data({"held_out": held, "pool": pool, "train": rows})
    fp = {"start": child_rss_gb(os.getpid())}
    model = load_model("bfloat16", False, dev)
    fp["loaded"] = child_rss_gb(os.getpid())
    lora = attach_lora(model, a)
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    base_name = next(n for n, p in model.named_parameters() if not p.requires_grad and "layers.27.mlp.down_proj" in n)
    base_before = dict(model.named_parameters())[base_name].detach().clone()
    info = train_lora(model, rows, pool, CHECK_DIR / "checkpoints", a, dev, max_steps=2)
    fp["trained"] = child_rss_gb(os.getpid())
    params = dict(model.named_parameters())
    moved = sum(int(not torch.equal(lora_before[n], params[n].detach())) for n in lora_before)
    n_lora = len(lora_before)
    losses = [h["loss"] for h in info["loss_history"]]
    ckpts = sorted(p.name for p in (CHECK_DIR / "checkpoints").glob("checkpoint-*"))
    live = {n: p.detach().clone() for n, p in params.items() if p.requires_grad}
    load_lora_weights(model, CHECK_DIR / "checkpoints" / "checkpoint-2")
    round_trip = all(torch.equal(live[n], params[n].detach()) for n in live)
    result = {"device": dev, "lora": lora, "global_steps": info["global_steps"], "losses": losses,
              "step_seconds": info["step_seconds"], "lora_tensors_moved": f"{moved}/{n_lora}",
              "base_unchanged": bool(torch.equal(base_before, params[base_name].detach())),
              "checkpoints": ckpts, "checkpoint_round_trip": round_trip, "datasets": info["datasets"]["rows"],
              "label_width": info["datasets"]["label_width"], "mask_stats": info["mask_stats"],
              "gradient_checkpointing": bool(model[0].model.get_base_model().is_gradient_checkpointing)}
    del model, params, live, lora_before
    release(dev)
    probe = [rows[0]["positive"], rows[-1]["positive"], te.QUERY_PROMPT + rows[0]["anchor"], "Description: x"]
    tuned = load_model("bfloat16", False, dev)
    result["merge_check_bf16"] = merge_lora(tuned, CHECK_DIR / "checkpoints" / "checkpoint-2", probe, "bfloat16")
    te.check_text_path(tuned)
    fp["merged"] = child_rss_gb(os.getpid())
    result["footprint_gb"] = {k: round(v, 2) for k, v in fp.items()}
    step0.write_json(CHECK_DIR / "check_result.json", result)
    log(json.dumps(result, ensure_ascii=False, default=str))
    assert info["global_steps"] == 2 and len(losses) == 2 and all(math.isfinite(x) for x in losses), losses
    assert moved == n_lora, result["lora_tensors_moved"]
    assert result["base_unchanged"] and round_trip and result["gradient_checkpointing"], result
    assert {"checkpoint-1", "checkpoint-2"} <= set(ckpts), ckpts
    assert all((CHECK_DIR / "checkpoints" / c / f).exists() for c in ckpts for f in CKPT_FILES), ckpts
    assert result["merge_check_bf16"]["ok"], result["merge_check_bf16"]
    assert set(lora["frozen_dtypes"]) == {"torch.bfloat16"}, lora["frozen_dtypes"]
    log("real-weight check: ok")
    return result


def bench(paths: SimpleNamespace, rows: List[dict], pool: Dict[str, dict], args: argparse.Namespace, dev: str) -> dict:
    """Time ``--bench-steps`` training steps and project the training phase (checkpoints are removed afterwards).

    :param paths: Paths of the bench run (``<run dir>/bench``, so the real run's subset is not fixed by the bench).
    :param rows: Training subset.
    :param pool: Train pool.
    :param args: Parsed CLI arguments.
    :param dev: Device.
    :returns: Timing summary.
    :rtype: dict
    """
    out = paths.ckpt
    if out.exists():
        shutil.rmtree(out)
    model = load_model(args.base_dtype, args.tiny, dev)
    attach_lora(model, args)
    a = argparse.Namespace(**{**vars(args), "save_steps": 10 ** 9, "stop_after_step": 0})
    info = train_lora(model, rows, pool, out, a, dev, max_steps=args.bench_steps)
    datasets, _, _ = te.build_datasets(rows, pool)
    steps = expected_steps(datasets, args.batch)
    med = info["step_seconds"]["median"]
    res = {"device": dev, "bench_steps": args.bench_steps, "step_seconds": info["step_seconds"], "n_train": len(rows),
           "steps_full_run": steps, "projected_train_minutes": round(steps * med / 60, 1),
           "rows_per_training_hour": int(3600 / med * args.batch), "footprint_gb": round(child_rss_gb(os.getpid()), 2)}
    shutil.rmtree(out)
    step0.write_json(paths.root / "bench.json", res)
    log(f"bench: {res}")
    return res


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """CLI.

    :returns: Parsed arguments.
    :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pkg", default=str(PKG), help="local v3 dataset package (identical to the HF tag v3)")
    p.add_argument("--out-dir", default=None, help=f"run dir (default {OUT_DIR}; smoke {SMOKE_DIR})")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--stop-after", choices=PHASES, default="eval", help="last phase to run")
    p.add_argument("--smoke", type=int, default=0, help="smoke test: tiny random model, ~N-record eval subset, CPU")
    p.add_argument("--smoke-train", type=int, default=96, help="smoke: training rows")
    p.add_argument("--smoke-other", type=int, default=150, help="smoke: non-held-out pool records")
    p.add_argument("--check-real-weights", action="store_true", help="2-step CPU check of the real 0.6B + LoRA")
    p.add_argument("--bench-steps", type=int, default=0, help="time N training steps in a scratch dir and exit")
    p.add_argument("--allow-code-drift", action="store_true", help="run although the repo trainer != packaged copy")
    # subset + pool
    p.add_argument("--n-train", type=int, default=12000)
    p.add_argument("--n-other", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0, help="subset + pool sampling seed")
    # training
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--mini-batch", type=int, default=8)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--train-seed", type=int, default=13)
    p.add_argument("--base-dtype", default="bfloat16", help="frozen base weights while training (float16 if MPS needs)")
    p.add_argument("--train-pad-multiple", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--stop-after-step", type=int, default=0, help="testing: checkpoint + exit 99 at this step")
    # embedding (step0_local names)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--shard-tokens", type=int, default=300_000)
    p.add_argument("--pad-multiple", type=int, default=64)
    p.add_argument("--max-batch", type=int, default=128)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--max-attn", type=int, default=1 << 22)
    p.add_argument("--canary", type=int, default=32)
    p.add_argument("--canary-min-cos", type=float, default=step0.CANARY_MIN_COS)
    # runtime
    p.add_argument("--soft-max-gb", type=float, default=0.0, help=f"exit {RESTART_RC} above this footprint (0 = off)")
    p.add_argument("--cpu-threads", type=int, default=4, help="torch CPU threads unless AUDIT_THREADS is set")
    p.add_argument("--n-resamples", type=int, default=1000)
    args = p.parse_args()
    args.tiny = False
    if args.smoke:
        for k, v in SMOKE_OVERRIDES.items():
            setattr(args, k, v)
        args.n_train, args.n_other = args.smoke_train, args.smoke_other
    return args


def main() -> None:
    """Run the pilot phases in order, each skipped when its output is complete (or the bench / real-weight check)."""
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(AUDIT_ROOT / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dev = device()
    if not os.environ.get("AUDIT_THREADS"):
        torch.set_num_threads(args.cpu_threads)
    code = check_code(args.allow_code_drift)
    log(f"device {dev}, threads {torch.get_num_threads()}, code {code}")
    if args.check_real_weights:
        check_real_weights(args, dev)
        return
    paths = run_paths(Path(args.out_dir) if args.out_dir else (SMOKE_DIR if args.smoke else OUT_DIR))
    if args.bench_steps:
        paths = run_paths(paths.root / "bench")
    paths.root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. prepare
    eval_dir = smoke_eval_dir(paths, args.smoke, args.seed) if args.smoke else Path(args.pkg) / "eval"
    train_pool = {r["doc_id"]: r for r in te._read_jsonl(Path(args.pkg) / "train_pool.jsonl")}
    held = set(json.loads((eval_dir / "held_out_records.json").read_text())["held_out"])
    rows, subset_meta = prepare_subset(paths, args, train_pool, held)
    pool = prepare_pool(paths, eval_dir, args)
    log(f"subset {len(rows)} rows {subset_meta['by_kind']}; pool {len(pool['doc_ids'])} records "
        f"({pool['n_held_out']} held out + {pool['n_other']} sampled of {pool['n_other_available']})")
    if args.bench_steps:
        bench(paths, rows, train_pool, args, dev)
        return
    if args.stop_after == "prepare":
        return
    sem_rows, sem_texts = step0.load_semantic(Path(args.pkg) / "corpus_semantic.jsonl", set(pool["doc_ids"]))
    if len(sem_rows) != len(pool["doc_ids"]):
        raise ValueError(f"{len(sem_rows)} semantic records for {len(pool['doc_ids'])} pool ids")
    qtexts = step0.query_texts(eval_v3.load_eval_dir(eval_dir))
    spec = TEXT_MODELS[MODEL_KEY]
    base_tag = {"model": "tiny-random-qwen3" if args.tiny else spec, "dtype": "float32", "max_seq": MAX_SEQ,
                "text": "semantic"}

    # 2. base vectors (+ production-text vectors of the tiny model in the smoke: the stored ones are the real model's)
    worked = not vectors_complete(paths.vec_base, base_tag, len(sem_rows))
    base_meta = embed_phase(paths.vec_base, sem_rows, sem_texts, qtexts, base_loader(args, dev), base_tag, args,
                            canary=not args.tiny)
    orig_dir = ORIGINAL_VECS
    if args.tiny:
        orig_rows, orig_texts = load_original(Path(args.pkg) / "corpus_original.jsonl",
                                              {r["doc_id"] for r in eval_v3.load_eval_dir(eval_dir)["eval_pool"]})
        orig_tag = {**base_tag, "text": "original"}
        worked = worked or not vectors_complete(paths.vec_orig, orig_tag, len(orig_rows))
        embed_phase(paths.vec_orig, orig_rows, orig_texts, None, base_loader(args, dev), orig_tag, args, canary=False)
        orig_dir = paths.vec_orig
    maybe_restart(args, "embed-base", worked)
    if args.stop_after == "embed-base":
        return

    # 3. train
    done = paths.lora / "training_info.json"
    if not done.exists():
        cfg_path = paths.root / "train_config.json"
        if cfg_path.exists() and json.loads(cfg_path.read_text()) != train_config(args):
            raise SystemExit(f"{paths.ckpt} was trained with other settings ({cfg_path}); remove both to start over")
        step0.write_json(cfg_path, train_config(args))
        model = load_model(args.base_dtype, args.tiny, dev)
        lora = attach_lora(model, args)
        info = train_lora(model, rows, train_pool, paths.ckpt, args, dev)
        if info["restart_reason"]:
            log(f"training stopped for a relaunch: {info['restart_reason']}; exiting {RESTART_RC}")
            sys.exit(RESTART_RC)
        tmp = paths.lora.with_name(paths.lora.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        model[0].model.save_pretrained(str(tmp))
        info.update(lora=lora, subset=subset_meta, args=train_config(args))
        step0.write_json(tmp / "training_info.json", info)
        if paths.lora.exists():
            shutil.rmtree(paths.lora)
        tmp.rename(paths.lora)
        log(f"training done: {info['global_steps']} steps in {info['train_seconds'] / 60:.1f} min "
            f"({info['sessions']} session(s)); adapter {paths.lora}")
        del model
        release(dev)
        maybe_restart(args, "train", True)
    train_info = json.loads(done.read_text())
    if args.stop_after == "train":
        return

    # 4. tuned vectors
    adapter_md5 = md5_file(paths.lora / "adapter_model.safetensors")
    by_len = sorted(sem_texts.values(), key=lambda t: (len(t), t))
    probe = [by_len[int(q * (len(by_len) - 1))] for q in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)]
    probe += [te.QUERY_PROMPT + q for q in qtexts[:2]]
    tuned_tag = {**base_tag, "adapter_md5": adapter_md5}
    worked = not vectors_complete(paths.vec_tuned, tuned_tag, len(sem_rows))
    tuned_meta = embed_phase(paths.vec_tuned, sem_rows, sem_texts, qtexts, tuned_loader(args, dev, paths.lora, probe),
                             tuned_tag, args, canary=False)
    maybe_restart(args, "embed-tuned", worked)
    if args.stop_after == "embed-tuned":
        return

    # 5. eval
    ev = eval_phase(paths, pool, rows, sem_rows, eval_dir, orig_dir, args)
    print(te.format_table(ev["comparison"]), flush=True)
    for name, ci in ev["extra"].items():
        print(f"  {name:<70} {ci['delta']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  {ci['p_b_better']:.2f}  "
              f"{ci['n_clusters']}/{ci['n_queries']}", flush=True)
    paired = {**ev["comparison"]["paired"], **ev["extra"]}
    report = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "question": "Does a small local LoRA run of the v3 recipe move held-out-subject retrieval vs the base model?",
        "smoke": bool(args.smoke),
        "headline": headline(ev["reports"], paired),
        "design": {"lora": train_info["lora"], "training": train_info["args"],
                   "mining": train_info["mining"], "eval_window": MAX_SEQ, "query_prompt": te.QUERY_PROMPT,
                   "documents": "raw (no prefix)", "embedding_dtype": "float32 (LoRA merged into the fp32 base)",
                   "base_original_vectors": str(orig_dir),
                   "pool": {k: v for k, v in pool.items() if k != "doc_ids"},
                   "pool_note": "eval_v3 scores only frozen-pool records present in doc_ids; every model gets exactly "
                                "the reduced pool's vectors, so n_missing_vectors = frozen pool - reduced pool"},
        "subset": subset_meta,
        "training": {k: v for k, v in train_info.items() if k not in ("subset", "args", "lora")},
        "embedding": {"base_semantic": base_meta, "tuned_semantic": tuned_meta},
        "vector_shift_tuned_vs_base": ev["vector_shift"],
        "summaries": {lb: eval_v3.summary(r) for lb, r in ev["reports"].items()},
        "comparison": ev["comparison"], "extra_paired": ev["extra"],
        "pool_calibration": ev["pool_calibration"],
        "full_results": {lb: str(paths.reports / f"{lb}.json") for lb in ev["reports"]},
        "code": code, "run_dir": str(paths.root), "seconds_this_session": round(time.time() - t_start, 1)}
    results_path = paths.root / "results.json" if args.smoke else Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    step0.write_json(results_path, report)
    log(f"wrote {results_path}")


if __name__ == "__main__":
    main()
