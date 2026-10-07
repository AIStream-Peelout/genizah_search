"""Pilot I3: does a little supervision turn generic features into a handwriting encoder?

Trains a small projection head (MLP, supervised-contrastive loss) on FROZEN line features, with
"same manuscript" as the positive label, using KTIV Kraken line crops from manuscripts that are
disjoint from the I2 test set. Then re-scores I2 (same test lines: other page of the same
manuscript) with the head. If the head lifts held-out manuscript retrieval substantially, full
fine-tuning of the backbone on Genizah data is worth the GPU time.

``--encoder`` picks the frozen features (default ``dinov2-base``: mean of 4 tiles @224, the
original pilot): a Qwen3-VL tower spec (``qwen3vl-vit512[-<ckpt>][:<readout>]``, whole line at
``--height``; see :mod:`line_probe`) or a ``+`` fusion such as
``dinov2-base+qwen3vl-vit512-heb-v21b-step1200`` (normalised blocks concatenated). The DINOv2
component here is always the 4-tile feature (I2 uses 6 tiles), so the I3 "raw" row of a fusion
can differ slightly from the I2 fusion row; tower-only raw rows equal I2 (same test cache).

Feature extraction honours ``AUDIT_DEVICE`` (CPU during the sibling's GPU evals).
"""

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from embed_utils import AUDIT_ROOT
from line_probe import LINES, fuse, is_tower, load_tower, tiles

HERE = Path(__file__).parent


def train_selection(test_ms: set, per_page: int, max_pages: int, seed: int = 11) -> list:
    """Sample training lines from manuscripts not in the test set.

    :param test_ms: Manuscript ids held out for evaluation.
    :param per_page: Lines per page.
    :param max_pages: Pages per manuscript.
    :param seed: RNG seed.
    :returns: List of (ms, page, path).
    :rtype: list
    """
    rng = random.Random(seed)
    out = []
    for ms_dir in sorted(p for p in LINES.iterdir() if p.is_dir() and p.name not in test_ms):
        pages = defaultdict(list)
        for f in ms_dir.glob("*.jpg"):
            m = re.match(r"(.+_FL\d+)_c\d+_l\d+\.jpg$", f.name)
            if m:
                pages[m.group(1)].append(f)
        good = sorted(p for p, fs in pages.items() if len(fs) >= 2)
        if len(good) < 2:
            continue
        for page in rng.sample(good, min(max_pages, len(good))):
            for f in rng.sample(sorted(pages[page]), min(per_page, len(pages[page]))):
                out.append((ms_dir.name, page, str(f)))
    return out


def extract(sel: list, cache: Path) -> np.ndarray:
    """DINOv2-base line features (mean of tile CLS+mean-patch), cached.

    :param sel: (ms, page, path) list.
    :param cache: .npy cache path.
    :returns: (n, 1536) array.
    :rtype: np.ndarray
    """
    if cache.exists():
        return np.load(cache)
    from image_features import Encoder

    enc = Encoder("dinov2-base")
    feats = []
    for i, (_, _, path) in enumerate(sel):
        v = enc.embed(tiles(path, n=4), res=224).mean(0)
        feats.append(v / np.linalg.norm(v))
        if (i + 1) % 2000 == 0:
            print(f"extracted {i + 1}/{len(sel)}", flush=True)
    X = np.stack(feats).astype(np.float32)
    np.save(cache, X)
    return X


def features(name: str, sel: list, split: str, height: int, max_width: int, part_size: int = 500) -> np.ndarray:
    """Frozen line features for an encoder spec: dinov2-base (4 tiles), a tower readout, or a fusion.

    :param name: Encoder spec (``+`` fuses components).
    :param sel: (ms, page, path) list.
    :param split: ``train`` or ``test`` (cache name).
    :param height: Tower line height.
    :param max_width: Tower width cap.
    :param part_size: Lines per resumable tower-extraction part.
    :returns: (n, D) normalised features.
    :rtype: np.ndarray
    """
    feat_dir = AUDIT_ROOT / "cache" / "line_feats"
    feat_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for comp in name.split("+"):
        if is_tower(comp):
            parts.append(load_tower(comp, sel, split, height, max_width, part_size))
        elif comp == "dinov2-base":
            parts.append(extract(sel, feat_dir / f"dinov2-base_{split}_t4.npy"))
        else:
            raise ValueError(f"unsupported I3 encoder component {comp!r}")
    return fuse(parts)


def run_tag(name: str, height: int) -> str:
    """File-safe tag for results/head files (the default dinov2-base keeps its original names).

    :param name: Encoder spec.
    :param height: Tower line height (part of the tag when a tower is involved).
    :returns: Tag string.
    :rtype: str
    """
    tag = name.replace(":", "_")
    return f"{tag}_h{height}" if any(is_tower(c) for c in name.split("+")) else tag


def retrieval(X: np.ndarray, ms: list, page: list) -> dict:
    """I2 metric: other-page same-manuscript retrieval.

    :param X: Normalised features.
    :param ms: Manuscript id per line.
    :param page: Page id per line.
    :returns: P@1 and mAP.
    :rtype: dict
    """
    S = X @ X.T
    p1, aps = [], []
    for i in range(len(ms)):
        valid = np.array([j for j in range(len(ms)) if j != i and page[j] != page[i]])
        rel = np.array([ms[j] == ms[i] for j in valid])
        relo = rel[np.argsort(-S[i, valid])]
        p1.append(relo[0])
        hits = np.cumsum(relo)
        aps.append(float(((hits / np.arange(1, len(relo) + 1)) * relo).sum() / max(relo.sum(), 1)))
    return {"P@1": round(float(np.mean(p1)), 4), "mAP": round(float(np.mean(aps)), 4)}


def main() -> None:
    """Extract, train, evaluate."""
    import torch
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-page", type=int, default=6)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--encoder", default="dinov2-base", help="frozen features; tower ':<readout>', '+' fuses")
    parser.add_argument("--height", type=int, default=96, help="tower line height (multiple of 32)")
    parser.add_argument("--max-width", type=int, default=2048, help="tower width cap (multiple of 32)")
    parser.add_argument("--part-size", type=int, default=500, help="lines per resumable tower-extraction part")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    test = json.loads((AUDIT_ROOT / "line_selection_v1.json").read_text())
    test_ms = {t[0] for t in test}
    sel_path = AUDIT_ROOT / "line_train_selection_v1.json"
    if sel_path.exists():
        train = json.loads(sel_path.read_text())
    else:
        train = train_selection(test_ms, args.per_page, args.max_pages)
        sel_path.write_text(json.dumps(train))
    print(f"train lines {len(train)} from {len({t[0] for t in train})} MSS; test lines {len(test)} from {len(test_ms)} MSS", flush=True)
    # test first: tower test features are shared with I2, so a probe run may already have them
    Xte = features(args.encoder, test, "test", args.height, args.max_width, args.part_size)
    Xtr = features(args.encoder, train, "train", args.height, args.max_width, args.part_size)
    base = retrieval(Xte, [t[0] for t in test], [t[1] for t in test])
    print(f"raw {args.encoder} ({Xte.shape[1]}-d):", base, flush=True)

    torch.set_grad_enabled(True)  # Encoder() disables autograd globally for extraction
    torch.manual_seed(0)
    labels = {m: k for k, m in enumerate(sorted({t[0] for t in train}))}
    y = torch.tensor([labels[t[0]] for t in train])
    x = torch.from_numpy(Xtr)
    head = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 768), torch.nn.GELU(), torch.nn.Linear(768, 256))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    by_ms = defaultdict(list)
    for i, m in enumerate(y.tolist()):
        by_ms[m].append(i)
    ms_ids = [m for m, v in by_ms.items() if len(v) >= 2]
    rng = random.Random(0)
    for epoch in range(args.epochs):
        rng.shuffle(ms_ids)
        losses = []
        for b in range(0, len(ms_ids), 64):
            idx = [i for m in ms_ids[b:b + 64] for i in rng.sample(by_ms[m], min(4, len(by_ms[m])))]
            z = F.normalize(head(x[idx]), dim=1)
            lab = y[idx]
            sim = z @ z.T / args.temp
            sim.fill_diagonal_(-1e9)
            pos = (lab[:, None] == lab[None, :]).float()
            pos.fill_diagonal_(0)
            logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
            loss = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if epoch % 10 == 9:
            print(f"epoch {epoch + 1} supcon {np.mean(losses):.3f}", flush=True)
    with torch.no_grad():
        Zte = F.normalize(head(torch.from_numpy(Xte)), dim=1).numpy()
    tuned = retrieval(Zte, [t[0] for t in test], [t[1] for t in test])
    print("with trained head:", tuned, flush=True)
    tag = run_tag(args.encoder, args.height)
    torch.save(head.state_dict(), AUDIT_ROOT / "cache" / f"line_head_{tag}.pt")
    out = {"encoder": args.encoder, "feature_dim": int(Xte.shape[1]),
           "train_lines": len(train), "train_mss": len(labels), "test_lines": len(test), "test_mss": len(test_ms),
           "raw": base, "head": tuned, "args": vars(args), "ran_at": datetime.now().isoformat(timespec="seconds")}
    name = "i3_line_head_pilot.json" if args.encoder == "dinov2-base" else f"i3_line_head_pilot__{tag}.json"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / name).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
