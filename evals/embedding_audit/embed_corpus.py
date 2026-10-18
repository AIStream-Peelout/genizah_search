"""Embed the audit corpus (document mode) with one model, in resumable shards.

Documents are encoded exactly as the index builder does for the production
model (raw text, no prefix, L2-normalised). Texts are processed in a fixed
length-sorted order and written in shards of ``--shard-size`` vectors to
``<out-dir>/<model>/shard_XXXX.npy`` plus one ``ids.json``; existing shards are
skipped, so the guard can kill and relaunch this at any time.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from embed_utils import AUDIT_ROOT, CachedTextEncoder


def load_corpus(path: Path, only_framed_or_labelled: bool) -> list:
    """Load corpus rows.

    :param path: corpus JSONL.
    :param only_framed_or_labelled: Keep only framed/topic-labelled rows (smaller pool).
    :returns: List of (doc_id, text).
    :rtype: list
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if only_framed_or_labelled and not (row["framed"] or row["topics"]):
                continue
            rows.append((row["doc_id"], row["text"]))
    return rows


def main() -> None:
    """Encode the corpus for one model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-path", default=None, help="fine-tuned checkpoint dir")
    parser.add_argument("--corpus", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--out-dir", default=str(AUDIT_ROOT / "corpus_vectors"))
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--subset", choices=["all", "framed"], default="all")
    args = parser.parse_args()

    rows = load_corpus(Path(args.corpus), args.subset == "framed")
    # Length-sorted so batches are homogeneous (much faster with padding); order is deterministic.
    rows.sort(key=lambda r: (len(r[1]), r[0]))
    out = Path(args.out_dir) / f"{args.model}__{args.subset}__{args.max_seq}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.json").write_text(json.dumps([r[0] for r in rows]))

    # One writer per vector set: the CPU queue and the GPU queue may both reach the same model.
    lock = out / "LOCK"
    if lock.exists():
        try:
            os.kill(int(lock.read_text().strip()), 0)
            raise SystemExit(f"{out.name} is being embedded by pid {lock.read_text().strip()}")
        except (ProcessLookupError, ValueError):
            pass
    lock.write_text(str(os.getpid()))
    enc = CachedTextEncoder(args.model, max_seq_length=args.max_seq, local_path=args.local_path)
    n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        path = out / f"shard_{s:04d}.npy"
        if path.exists():
            continue
        chunk = rows[s * args.shard_size:(s + 1) * args.shard_size]
        t0 = time.time()
        # Rows are length-sorted, so later shards hold long texts. Encode in small windows with a
        # length-dependent batch and release the MPS cache between windows, so the working set
        # (attention/activation buffers) stays well under the guard's footprint cap.
        parts = []
        for w in range(0, len(chunk), 32):
            window = [t for _, t in chunk[w:w + 32]]
            longest = max(len(t) for t in window)
            batch = args.batch_size if longest < 600 else 4 if longest < 1500 else 2 if longest < 3000 else 1
            parts.append(enc.model.encode(window, normalize_embeddings=True, batch_size=min(batch, args.batch_size),
                                          convert_to_numpy=True))
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.concatenate(parts).astype(np.float32))
        tmp.rename(path)
        print(f"shard {s + 1}/{n_shards} n={len(chunk)} {len(chunk) / (time.time() - t0):.1f} docs/s", flush=True)
    lock.unlink(missing_ok=True)
    print("done", out)


if __name__ == "__main__":
    main()
