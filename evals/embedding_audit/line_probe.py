"""Probe I2: line-level handwriting retrieval on the KTIV Kraken line crops (NAS, local).

Writer/manuscript identity is the signal joins and scribe attribution need, and line
crops remove most photographic context. For manuscripts with >= 2 pages, sample
``--per-page`` lines from each of two pages; every line queries all sampled lines and
its positives are lines of the SAME manuscript on a DIFFERENT page (same-page lines are
excluded, so layout/lighting of one photograph cannot score). Negatives from the same
holding library are reported separately to measure the imaging confound.

Line inputs per encoder family
------------------------------
* Square-input encoders (DINOv2, CLIP, SigLIP2): line vector = mean of L2-normalised
  embeddings of square tiles cut along the line (side = line height).
* Qwen3-VL vision towers (``qwen3vl-vit512`` = stock, ``qwen3vl-vit512-<ckpt>`` = a fine-tuned
  Hebrew VLM's tower, see :mod:`image_features`): the tower takes any aspect ratio, so the WHOLE
  line goes through one forward. The crop is resized to a fixed height (``--height``, default
  96 px, a multiple of 32 = patch 16 x spatial merge 2) keeping its aspect ratio, so the script
  scale is the same for every line (a 160-px Kraken crop has ~45-px letters; at 96 px they are
  ~27 px, close to letter size in the page images the VLM was fine-tuned on). Lines wider than
  ``--max-width`` are centre-cropped (not shrunk, to keep the scale fixed), and the width is
  padded to a multiple of 32 with the line's own border colour. The processor is called with
  ``do_resize=False`` so nothing is rescaled again. Readouts, all from the same forward:
  ``merger`` (final merged tokens, 4096-d, mean-pooled — the same pooling as the fragment-level
  probe) and the three deepstack mergers (``ds8``, ``ds16``, ``ds24``: mid-tower layers 8/16/24,
  also 4096-d, mean-pooled), which may carry more stroke-level style than the reading-tuned top.
  Select a readout with ``<encoder>:<readout>``, e.g. ``qwen3vl-vit512-heb-v21b-step1200:ds16``.
* Fusion: ``a+b`` concatenates the L2-normalised component vectors and renormalises, so the
  fused cosine is the mean of the component cosines (equal weight).

Tower features are extracted in resumable parts (the guard may kill a run on box pressure)
under ``cache/line_feats/<stem>_parts`` and assembled into one ``.npy`` per readout.
"""

import argparse
import json
import os
import random
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from embed_utils import AUDIT_ROOT
from image_features import Encoder, border_color

LINES = Path("/Volumes/home/studio_offload/datasets/kraken_ktiv_lines/lines")
# AUDIT_LINES_LOCAL: a local mirror of LINES (same relative layout), used when a file exists there
LINES_LOCAL = Path(os.environ["AUDIT_LINES_LOCAL"]) if os.environ.get("AUDIT_LINES_LOCAL") else None


def line_file(path) -> Path:
    """Prefer the local mirror of a NAS line crop.

    :param path: NAS path from a line selection.
    :returns: Local path if mirrored, else the NAS path.
    :rtype: Path
    """
    path = Path(path)
    if LINES_LOCAL is not None:
        try:
            local = LINES_LOCAL / path.relative_to(LINES)
        except ValueError:
            return path
        if local.exists():
            return local
    return path
HERE = Path(__file__).parent
FEAT_DIR = AUDIT_ROOT / "cache" / "line_feats"
TOWER_PREFIX = "qwen3vl-vit"
TOWER_GRID = 32  # Qwen3-VL patch 16 x spatial merge 2: tower inputs must be multiples of this
TOWER_READOUTS = ("merger", "ds8", "ds16", "ds24")  # final merger + deepstack mergers (layers 8/16/24)


def select_lines(n_ms: int, per_page: int, seed: int = 7) -> list:
    """Pick (ms, page, path) triples.

    :param n_ms: Manuscripts to sample.
    :param per_page: Lines per page (two pages per manuscript).
    :param seed: RNG seed.
    :returns: List of (ms, page, path).
    :rtype: list
    """
    rng = random.Random(seed)
    mss = sorted(p.name for p in LINES.iterdir() if p.is_dir())
    rng.shuffle(mss)
    out = []
    for ms in mss:
        pages = defaultdict(list)
        for f in (LINES / ms).glob("*.jpg"):
            m = re.match(r"(.+_FL\d+)_c\d+_l\d+\.jpg$", f.name)
            if m:
                pages[m.group(1)].append(f)
        good = [p for p, fs in pages.items() if len(fs) >= per_page]
        if len(good) < 2:
            continue
        for page in rng.sample(sorted(good), 2):
            for f in rng.sample(sorted(pages[page]), per_page):
                out.append((ms, page, str(f)))
        if len({o[0] for o in out}) >= n_ms:
            break
    return out


def tiles(path: str, n: int = 6) -> list:
    """Cut a line image into up to n square tiles (side = line height).

    :param path: Line image path.
    :param n: Max tiles (evenly spread).
    :returns: List of PIL tiles.
    :rtype: list
    """
    img = Image.open(line_file(path)).convert("RGB")
    h, w = img.height, img.width
    if w <= h:
        return [img]
    starts = np.linspace(0, w - h, num=min(n, max(1, w // h))).astype(int)
    return [img.crop((int(x), 0, int(x) + h, h)) for x in starts]


def is_tower(name: str) -> bool:
    """Whether an encoder spec names a Qwen3-VL vision tower (optionally with a ``:readout``).

    :param name: Encoder spec, e.g. ``qwen3vl-vit512-heb-v21b-step1200:ds16``.
    :returns: True for tower specs.
    :rtype: bool
    """
    return name.startswith(TOWER_PREFIX)


def split_readout(name: str) -> Tuple[str, str]:
    """Split ``<tower>[:<readout>]`` into (tower, readout); the default readout is ``merger``.

    :param name: Tower spec.
    :returns: (tower encoder name, readout).
    :rtype: Tuple[str, str]
    """
    base, _, readout = name.partition(":")
    readout = readout or "merger"
    if readout not in TOWER_READOUTS:
        raise ValueError(f"unknown tower readout {readout!r}; choose from {TOWER_READOUTS}")
    return base, readout


def tower_line_image(path: str, height: int, max_width: int) -> Image.Image:
    """Prepare a line crop for the tower: fixed height, aspect kept, width padded to the 32-px grid.

    :param path: Line image path.
    :param height: Target height in px (multiple of 32).
    :param max_width: Width cap in px (multiple of 32); longer lines are centre-cropped.
    :returns: RGB image whose sides are multiples of 32.
    :rtype: Image.Image
    """
    img = Image.open(line_file(path)).convert("RGB")
    width = max(TOWER_GRID, round(img.width * height / img.height))
    img = img.resize((width, height), Image.BICUBIC)
    if width > max_width:
        x0 = (width - max_width) // 2
        img, width = img.crop((x0, 0, x0 + max_width, height)), max_width
    target = -(-width // TOWER_GRID) * TOWER_GRID
    if target == width:
        return img
    canvas = Image.new("RGB", (target, height), tuple(int(c) for c in border_color(np.asarray(img))))
    canvas.paste(img, ((target - width) // 2, 0))
    return canvas


def tower_line_vectors(enc: Encoder, img: Image.Image) -> Dict[str, np.ndarray]:
    """One tower forward on a prepared line; mean-pooled merged tokens for every readout.

    :param enc: A loaded Qwen tower :class:`image_features.Encoder`.
    :param img: Output of :func:`tower_line_image`.
    :returns: {readout: L2-normalised 4096-d vector}.
    :rtype: Dict[str, np.ndarray]
    """
    import torch

    inp = enc.processor(images=[img], return_tensors="pt", do_resize=False)
    dtype = torch.bfloat16 if enc.dev != "cpu" else torch.float32
    hidden, deepstack = enc.model(inp["pixel_values"].to(enc.dev, dtype), grid_thw=inp["image_grid_thw"].to(enc.dev))
    pooled = [hidden] + list(deepstack)
    names = ["merger"] + [f"ds{i}" for i in enc.model.deepstack_visual_indexes]
    out = {}
    for name, h in zip(names, pooled):
        v = h.float().mean(0).cpu().numpy()
        out[name] = v / np.linalg.norm(v)
    return out


def tower_stem(tower: str, split: str, height: int, max_width: int) -> str:
    """Cache stem for one tower x line split x input geometry.

    :param tower: Tower encoder name (no readout).
    :param split: ``test`` (I2/I3 test lines) or ``train`` (I3 training lines).
    :param height: Line height in px.
    :param max_width: Width cap in px.
    :returns: File stem under :data:`FEAT_DIR`.
    :rtype: str
    """
    return f"{tower}_{split}_h{height}w{max_width}"


def extract_tower(tower: str, sel: list, split: str, height: int, max_width: int, part_size: int = 500) -> Dict[str, Path]:
    """Extract all tower readouts for a line selection, resumably, and assemble one .npy per readout.

    :param tower: Tower encoder name (no readout).
    :param sel: (ms, page, path) list.
    :param split: Split label used in the cache name.
    :param height: Line height in px.
    :param max_width: Width cap in px.
    :param part_size: Lines per resumable part.
    :returns: {readout: assembled .npy path}.
    :rtype: Dict[str, Path]
    """
    stem = tower_stem(tower, split, height, max_width)
    final = {r: FEAT_DIR / f"{stem}_{r}.npy" for r in TOWER_READOUTS}
    if all(p.exists() for p in final.values()):
        return final
    part_dir = FEAT_DIR / f"{stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = [part_dir / f"part_{k:04d}.npz" for k in range(-(-len(sel) // part_size))]
    enc, t0, done = None, time.time(), 0
    for k, path in enumerate(parts):
        if path.exists():
            continue
        enc = enc or Encoder(tower)
        vecs = defaultdict(list)
        for _, _, line in sel[k * part_size:(k + 1) * part_size]:
            for r, v in tower_line_vectors(enc, tower_line_image(line, height, max_width)).items():
                vecs[r].append(v)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{r: np.stack(v).astype(np.float32) for r, v in vecs.items()})
        tmp.rename(path)
        done += len(vecs["merger"])
        print(f"{stem} part {k + 1}/{len(parts)} done ({(time.time() - t0) / done:.2f} s/line)", flush=True)
    enc = None  # free the tower before assembling
    loaded = [np.load(p) for p in parts]
    for r, out in final.items():  # tmp + rename: a kill mid-write must not leave a "complete" cache
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, np.concatenate([z[r] for z in loaded]))
        tmp.rename(out)
    for p in parts:
        p.unlink()
    # SMB can leave AppleDouble/.DS_Store entries or lag on deletes: the parts are already gone, so a stubborn
    # empty-looking directory must not fail a run whose final caches are written
    shutil.rmtree(part_dir, ignore_errors=True)
    return final


def load_tower(name: str, sel: list, split: str, height: int, max_width: int, part_size: int = 500) -> np.ndarray:
    """Tower features for one readout (extracting all readouts first if needed).

    :param name: ``<tower>[:<readout>]``.
    :param sel: (ms, page, path) list.
    :param split: ``test`` or ``train``.
    :param height: Line height in px.
    :param max_width: Width cap in px.
    :param part_size: Lines per resumable extraction part.
    :returns: (n, 4096) normalised features.
    :rtype: np.ndarray
    """
    tower, readout = split_readout(name)
    X = np.load(extract_tower(tower, sel, split, height, max_width, part_size)[readout])
    assert len(X) == len(sel), (name, split, X.shape, len(sel))
    return X


def tile_features(name: str, sel: list) -> np.ndarray:
    """I2 tile features for a square-input encoder (6 tiles; DINOv2/CLIP at 224), cached.

    :param name: Encoder key from :mod:`image_features`.
    :param sel: (ms, page, path) list (the I2 test lines).
    :returns: (n, D) normalised features.
    :rtype: np.ndarray
    """
    cache = FEAT_DIR / f"{name}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return np.load(cache)
    enc = Encoder(name)
    feats = []
    for _, _, path in sel:
        v = enc.embed(tiles(path), res=224 if enc.family in ("dinov2", "clip") else None).mean(0)
        feats.append(v / np.linalg.norm(v))
    X = np.stack(feats).astype(np.float32)
    np.save(cache, X)
    return X


def fuse(parts: List[np.ndarray]) -> np.ndarray:
    """Concatenate L2-normalised feature blocks and renormalise (fused cosine = mean of cosines).

    :param parts: Feature arrays with the same row count.
    :returns: (n, sum D) normalised features.
    :rtype: np.ndarray
    """
    if len(parts) == 1:
        return parts[0]
    blocks = [p / np.linalg.norm(p, axis=1, keepdims=True) for p in parts]
    return (np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))).astype(np.float32)


def line_input_desc(name: str, height: int, max_width: int, dino_tiles: int = 6) -> str:
    """Human-readable description of how lines were fed to an encoder spec (stored with results).

    :param name: Encoder spec (may be a ``+`` fusion).
    :param height: Tower line height.
    :param max_width: Tower width cap.
    :param dino_tiles: Tile count used for square-input encoders.
    :returns: Description string.
    :rtype: str
    """
    descs = []
    for comp in name.split("+"):
        if is_tower(comp):
            descs.append(f"{comp}: full line h{height} (centre-crop >{max_width}, pad to 32), mean {split_readout(comp)[1]}")
        else:
            descs.append(f"{comp}: {dino_tiles} square tiles" + (" @224" if comp.startswith(("dinov2", "clip")) else ""))
    return " + ".join(descs)


def probe_features(name: str, sel: list, height: int, max_width: int, part_size: int = 500) -> np.ndarray:
    """I2 test-line features for an encoder spec (single encoder, tower readout, or ``+`` fusion).

    :param name: Encoder spec.
    :param sel: I2 test selection.
    :param height: Tower line height.
    :param max_width: Tower width cap.
    :param part_size: Lines per resumable tower-extraction part.
    :returns: (n, D) normalised features.
    :rtype: np.ndarray
    """
    parts = [load_tower(c, sel, "test", height, max_width, part_size) if is_tower(c) else tile_features(c, sel)
             for c in name.split("+")]
    return fuse(parts)


def time_towers(encoders: List[str], sel: list, n_lines: int, heights: List[int], max_width: int) -> None:
    """Timing smoke test: embed n evenly spaced test lines per tower x height; writes nothing.

    Prints seconds/line, mean token count, and the projected hours for the I2 test lines and the
    I3 training lines.

    :param encoders: Tower names (readouts ignored).
    :param sel: I2 test selection.
    :param n_lines: Lines to time.
    :param heights: Line heights to time.
    :param max_width: Width cap.
    """
    n_train = len(json.loads((AUDIT_ROOT / "line_train_selection_v1.json").read_text()))
    sample = [sel[i] for i in np.linspace(0, len(sel) - 1, n_lines).astype(int)]
    for tower in dict.fromkeys(split_readout(e)[0] for e in encoders if is_tower(e)):
        t0 = time.time()
        enc = Encoder(tower)
        load_s = time.time() - t0
        for height in heights:
            imgs = [tower_line_image(p, height, max_width) for _, _, p in sample]
            tower_line_vectors(enc, imgs[0])  # warm-up
            t0 = time.time()
            vecs = [tower_line_vectors(enc, im) for im in imgs]
            per = (time.time() - t0) / len(imgs)
            patches = np.mean([im.width * im.height / 256 for im in imgs])
            grid = enc.processor(images=[imgs[0]], return_tensors="pt", do_resize=False)["image_grid_thw"][0].tolist()
            r = {"tower": tower, "height": height, "lines": len(imgs), "load_s": round(load_s, 1),
                 "first_size": list(imgs[0].size), "first_grid_thw": grid,
                 "s_per_line": round(per, 3), "mean_patches": round(float(patches)),
                 "dims": {k: int(v.shape[0]) for k, v in vecs[0].items()},
                 "finite": bool(all(np.isfinite(v).all() for d in vecs for v in d.values())),
                 "proj_h_test": round(per * len(sel) / 3600, 2), "proj_h_train": round(per * n_train / 3600, 2)}
            print(json.dumps(r), flush=True)
        del enc


def library_by_ms(wanted: set) -> Dict[str, str]:
    """Holding library of each KTIV manuscript (sys_num -> merged record institution).

    :param wanted: KTIV sys_nums to look up.
    :returns: {sys_num: institution}.
    :rtype: Dict[str, str]
    """
    lib = {}
    merged = Path.home() / "Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
    with open(merged, encoding="utf-8") as fh:
        for line in fh:
            if '"sys_num"' not in line:
                continue
            rec = json.loads(line)
            sn = str(((rec.get("sources") or {}).get("ktiv") or {}).get("sys_num") or "")
            if sn in wanted:
                lib[sn] = rec.get("institution") or "?"
    return lib


def merge_results(path: Path, new: List[dict]) -> List[dict]:
    """Merge result rows into an existing results file, replacing rows with the same key.

    The key is (encoder, line_input). Rows written before ``line_input`` existed were all
    tile-based (DINOv2/CLIP), so their description is back-filled from the encoder name.

    :param path: Results JSON (a list of rows).
    :param new: Rows from this run.
    :returns: Merged rows.
    :rtype: List[dict]
    """
    old = json.loads(path.read_text()) if path.exists() else []
    for r in old:
        r.setdefault("line_input", line_input_desc(r["encoder"], 0, 0))
    keys = {(r["encoder"], r["line_input"]) for r in new}
    return [r for r in old if (r["encoder"], r["line_input"]) not in keys] + new


def main() -> None:
    """Embed sampled lines with each encoder and score manuscript retrieval."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoders", default="dinov2-base,siglip2-so400m,clip-b32",
                        help="comma list; towers take ':<readout>', '+' fuses specs")
    parser.add_argument("--n-ms", type=int, default=600)
    parser.add_argument("--per-page", type=int, default=3)
    parser.add_argument("--height", type=int, default=96, help="tower line height (multiple of 32)")
    parser.add_argument("--max-width", type=int, default=2048, help="tower width cap (multiple of 32)")
    parser.add_argument("--time-lines", type=int, default=0, help="timing smoke test on N lines; writes nothing")
    parser.add_argument("--time-heights", default="64,96")
    parser.add_argument("--part-size", type=int, default=500, help="lines per resumable tower-extraction part")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    assert args.height % TOWER_GRID == 0 and args.max_width % TOWER_GRID == 0, "height/max-width must be multiples of 32"
    sel_path = AUDIT_ROOT / "line_selection_v1.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
    else:
        sel = select_lines(args.n_ms, args.per_page)
        sel_path.write_text(json.dumps(sel))
    if args.time_lines:
        time_towers(args.encoders.split(","), sel, args.time_lines, [int(h) for h in args.time_heights.split(",")], args.max_width)
        return
    ms = [s[0] for s in sel]
    page = [s[1] for s in sel]
    # library of each manuscript for the confound split
    lib = library_by_ms(set(ms))
    from eval_images import pair_auc

    results = []
    for name in args.encoders.split(","):
        X = probe_features(name, sel, args.height, args.max_width, args.part_size)
        S = X @ X.T
        p1, aps, pos_s, neg_same_lib, neg_other_lib = [], [], [], [], []
        for i in range(len(sel)):
            valid = [j for j in range(len(sel)) if j != i and page[j] != page[i]]
            rel = np.array([ms[j] == ms[i] for j in valid])
            order = np.argsort(-S[i, valid])
            relo = rel[order]
            p1.append(relo[0])
            hits = np.cumsum(relo)
            aps.append(float(((hits / np.arange(1, len(relo) + 1)) * relo).sum() / max(relo.sum(), 1)))
            for j, r in zip(valid, rel):
                if r:
                    pos_s.append(S[i, j])
                elif lib.get(ms[j]) == lib.get(ms[i]):
                    neg_same_lib.append(S[i, j])
                else:
                    neg_other_lib.append(S[i, j])
        r = {"encoder": name, "line_input": line_input_desc(name, args.height, args.max_width),
             "n_lines": len(sel), "n_ms": len(set(ms)),
             "P@1_other_page_same_ms": round(float(np.mean(p1)), 4), "mAP": round(float(np.mean(aps)), 4),
             "chance_P@1": round((2 * args.per_page - args.per_page) / (len(sel) - args.per_page), 5),
             "auc_vs_same_library": round(pair_auc(np.array(pos_s), np.array(neg_same_lib)), 4) if neg_same_lib else None,
             "auc_vs_other_library": round(pair_auc(np.array(pos_s), np.array(neg_other_lib)), 4),
             "ran_at": datetime.now().isoformat(timespec="seconds")}
        results.append(r)
        print(json.dumps(r), flush=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "i2_lines.json"
    out.write_text(json.dumps(merge_results(out, results), indent=1))


if __name__ == "__main__":
    main()
