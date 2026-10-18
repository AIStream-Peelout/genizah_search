"""Package the image-track dataset for the Colab vision-tower fine-tune into ``<AUDIT_ROOT>/hf_image/<name>/``.

Training signal: KTIV Kraken line crops, "same manuscript" = positive. Evaluation: the audit's own held-out sets,
so the Colab run reports the same numbers the local probes do.

Files written
-------------
* ``lines.jsonl`` — one row per line crop: ``path`` (inside the tar shards), ``ms`` (KTIV sys number), ``page``,
  ``split`` (``train`` | ``val`` | ``test``), ``library`` (holding institution, for the I2 confound split and the
  library-blocked sampler), ``w``/``h`` of the stored crop.
* ``lines_<split>_NNN.tar`` — the crops as JPEG, resized to ``--height`` (96 px, the I2 protocol height) keeping the
  aspect ratio and centre-cropped to ``--max-width`` (2048 px), NOT padded (the trainer pads to the 32-px grid with
  the line's border colour exactly as :func:`line_probe.tower_line_image` does). ``test`` = the 3,600 I2 test lines
  of ``line_selection_v1.json`` (600 manuscripts x 2 pages x 3 lines); ``val`` = I2-style lines (2 pages x 3) of ~8 %
  of the training manuscripts, for monitoring; ``train`` = the rest, capped per page and per manuscript.
* ``fragments.jsonl`` + ``fragments_NNN.tar`` — the 5,176 fragment-eval images as the ``masked`` view of
  :func:`image_features.load_view` (backdrop blanked to grey, cropped, grey-padded square), resized to exactly the
  size the Qwen3-VL processor picks for the 512x512-pixel budget, so the processor does not resize them again.
* ``fragment_labels.json`` — ids (cache order), collection, FJP-clean and PGP join pairs, PGP scribe labels of the
  scribe selection, KTIV script attributes: everything :func:`eval_images.evaluate` reads, restricted to these ids.
* ``fragment_dinov2.npz`` — local DINOv2-base ``masked`` + ``mpatches`` features (for the DINOv2+tower fusion row).
* ``fragment_tower_local.npz`` — the local base-tower ``masked`` vectors (fp16), a Colab-vs-local sanity check.
* ``exclusions.json`` — what was removed from training and why.
* ``train_image_embedder.py``, ``README.md``.

Exclusions: no training line may come from an I2 test manuscript, or from a manuscript that is (via its merged
record) one of the fragment-eval images, a join partner of one, or a fragment of a scribe in the eval scribe sets.

Upload is NOT done here. Morning commands (private repos). The notebook pins ``DATA_REVISION = "v1"``, so the
dataset commit MUST be tagged ``v1`` (``hf_hub_download(..., revision="v1")`` fails otherwise); re-copy the trainer
first if it changed after packaging::

    cp colab/train_image_embedder.py <AUDIT_ROOT>/hf_image/dataset_v1/
    huggingface-cli upload isaacmg/genizah-image-train <AUDIT_ROOT>/hf_image/dataset_v1 . --repo-type dataset --private \\
        --exclude "*.sizes.json"
    huggingface-cli tag isaacmg/genizah-image-train v1 --repo-type dataset
    huggingface-cli upload isaacmg/qwen3-vl-8b-heb-v22b-vision-tower <AUDIT_ROOT>/hf_image/tower_heb-v22b-step1200 . --private

(A later re-upload of changed data needs a new tag, ``v2`` …, and ``DATA_REVISION`` bumped to match.)
"""

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))
from embed_utils import AUDIT_ROOT  # noqa: E402
from image_features import load_view  # noqa: E402
from line_probe import LINES  # noqa: E402

KTIV_ROOT = LINES.parent
MERGED = Path.home() / "Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
TOWER_CACHE = "qwen3vl-vit512-heb-v22b-step1200__masked"
LINE_RE = re.compile(r"(?P<page>(?P<ms>\d+)_FL\d+)_c\d+_l\d+\.jpg$")
FRAG_MIN_PIXELS, FRAG_MAX_PIXELS, GRID = 128 * 128, 512 * 512, 32

README = """---
license: other
---
# genizah-image-train ({name})

Private training/eval data for fine-tuning the Qwen3-VL-8B vision tower (our Hebrew VLM's, `{tower}`) as a
handwriting / fragment-similarity encoder. Built by `genizah_search/evals/embedding_audit/colab/` on {date}.

* `lines.jsonl` + `lines_*.tar` — KTIV Kraken line crops (height {height} px, width <= {max_width} px, unpadded):
  {n_train} train lines / {ms_train} manuscripts, {n_val} val lines / {ms_val} manuscripts,
  {n_test} test lines / {ms_test} manuscripts (= the audit's I2 test set `line_selection_v1.json`).
* `fragments.jsonl` + `fragments_*.tar` — {n_frag} masked fragment crops (the audit's I1 fragment-eval images,
  512x512-pixel budget) with `fragment_labels.json` (joins, scribes, script, collection).
* `fragment_dinov2.npz`, `fragment_tower_local.npz` — precomputed local features (fusion row; sanity check).
* `exclusions.json` — {n_excl_ms} manuscripts kept out of training: the 600 I2 test manuscripts plus every
  manuscript that is a fragment-eval image, a join partner of one, or in an eval scribe set.

Images: KTIV / National Library of Israel and partner libraries; private research use only, do not redistribute.
"""


def read_line_index() -> Dict[str, Dict[str, List[str]]]:
    """All written KTIV line crops, grouped by manuscript and page (from the dataset's own listings, no NAS walk).

    :returns: {ms: {page: [absolute path, ...]}} with paths sorted.
    :rtype: Dict[str, Dict[str, List[str]]]
    """
    out: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for listing in ("train.all.txt", "val.all.txt"):
        for rel in (KTIV_ROOT / listing).read_text().split():
            m = LINE_RE.search(rel)
            if m:
                out[m.group("ms")][m.group("page")].append(str(KTIV_ROOT / rel))
    return {ms: {p: sorted(v) for p, v in pages.items()} for ms, pages in out.items()}


def ktiv_records(wanted: Set[str]) -> Dict[str, dict]:
    """Merged records of KTIV manuscripts: canonical ids and holding institution per sys number.

    :param wanted: KTIV sys numbers.
    :returns: {sys_num: {"cids": set, "library": str}}; the library string is the merged record's institution, the
        same value :func:`line_probe.library_by_ms` uses for the I2 confound split.
    :rtype: Dict[str, dict]
    """
    out: Dict[str, dict] = {}
    with open(MERGED, encoding="utf-8") as fh:
        for line in fh:
            if '"sys_num"' not in line:
                continue
            rec = json.loads(line)
            sn = str(((rec.get("sources") or {}).get("ktiv") or {}).get("sys_num") or "")
            if sn in wanted:  # last record wins for the library, as in line_probe.library_by_ms
                entry = out.setdefault(sn, {"cids": set()})
                entry["cids"].add(rec["canonical_id"])
                entry["library"] = rec.get("institution") or "?"
    return out


def eval_fragment_ids() -> List[str]:
    """The fragment-eval ids in feature-cache order (the base-tower masked cache; DINOv2 caches must match).

    :returns: Ids (sorted, as :mod:`image_features` writes them).
    :rtype: List[str]
    """
    ids = []
    for part in sorted((AUDIT_ROOT / "cache" / "image_feats" / TOWER_CACHE).glob("part_*.npz")):
        ids += [str(c) for c in np.load(part)["ids"]]
    return ids


def exclusion_sets(manifest: dict, eval_ids: Set[str]) -> Dict[str, Set[str]]:
    """Canonical ids that must not reach training.

    :param manifest: ``image_manifest_v1.json``.
    :param eval_ids: Fragment-eval ids.
    :returns: {"eval": ..., "join_partner": ..., "scribe_set": ...} (disjoint, in that precedence).
    :rtype: Dict[str, Set[str]]
    """
    partners: Set[str] = set()
    for key in ("joins_fjp", "joins_fjp_clean", "joins_pgp"):
        for a, b in manifest[key]:
            if a in eval_ids or b in eval_ids:
                partners |= {a, b}
    scribes = {s for c, s in manifest["scribe"].items() if c in eval_ids}
    scribe_set = {c for c, s in manifest["scribe"].items() if s in scribes}
    return {"eval": set(eval_ids), "join_partner": partners - eval_ids, "scribe_set": scribe_set - eval_ids - partners}


def classify_manuscripts(line_ms: Set[str], records: Dict[str, dict], excl: Dict[str, Set[str]],
                         test_ms: Set[str]) -> Tuple[Dict[str, List[str]], dict]:
    """Decide which line manuscripts are excluded from training, with reasons.

    :param line_ms: KTIV manuscripts with line crops.
    :param records: Output of :func:`ktiv_records`.
    :param excl: Output of :func:`exclusion_sets`.
    :param test_ms: I2 test manuscripts.
    :returns: ({ms: [reasons]} for excluded manuscripts, overlap detail rows).
    :rtype: Tuple[Dict[str, List[str]], dict]
    """
    reasons: Dict[str, List[str]] = defaultdict(list)
    detail = []
    for ms in sorted(line_ms):
        if ms in test_ms:
            reasons[ms].append("i2_test")
        cids = records.get(ms, {}).get("cids", set()) | {ms}
        for why, ids in excl.items():
            hit = sorted(cids & ids)
            if hit:
                reasons[ms].append(why)
                detail.append({"ms": ms, "reason": why, "canonical_ids": hit, "also_i2_test": ms in test_ms})
    return dict(reasons), {"rows": detail}


def is_val(ms: str, frac: float) -> bool:
    """Deterministic hash split of training manuscripts into a small validation set.

    :param ms: KTIV sys number.
    :param frac: Validation fraction.
    :returns: True for validation manuscripts.
    :rtype: bool
    """
    return int(hashlib.md5(f"genizah-image-val:{ms}".encode()).hexdigest(), 16) % 10000 < frac * 10000


def select_lines(index: Dict[str, Dict[str, List[str]]], keep: List[str], test: list, val_frac: float,
                 per_page: int, per_ms: int, seed: int) -> List[dict]:
    """Choose the train / val / test line crops.

    Train manuscripts: up to ``per_page`` lines from each page, then pages are interleaved (round robin) until the
    ``per_ms`` cap, so a capped manuscript still contributes every page. Val manuscripts (>= 2 pages): 3 lines from
    each of 2 pages, the I2 protocol. Test: exactly the I2 selection.

    :param index: Output of :func:`read_line_index`.
    :param keep: Manuscripts allowed in training (sorted).
    :param test: ``line_selection_v1.json`` rows (ms, page, path).
    :param val_frac: Validation fraction of eligible manuscripts.
    :param per_page: Train lines per page cap.
    :param per_ms: Train lines per manuscript cap.
    :param seed: Sampling seed.
    :returns: Rows {path_src, ms, page, split}.
    :rtype: List[dict]
    """
    rows = [{"src": path, "ms": ms, "page": page, "split": "test"} for ms, page, path in test]
    for ms in keep:
        pages = index[ms]
        rng = random.Random(f"{seed}:{ms}")
        eligible = sorted(p for p, v in pages.items() if len(v) >= 3)
        if len(eligible) >= 2 and is_val(ms, val_frac):  # too few lines for the I2 protocol -> stays in train
            for page in rng.sample(eligible, 2):
                for src in rng.sample(pages[page], 3):
                    rows.append({"src": src, "ms": ms, "page": page, "split": "val"})
            continue
        queues = []
        for page in sorted(pages):
            lines = list(pages[page])
            rng.shuffle(lines)
            queues.append((page, lines[:per_page]))
        taken = 0
        for depth in range(per_page):
            for page, lines in queues:
                if depth < len(lines) and taken < per_ms:
                    rows.append({"src": lines[depth], "ms": ms, "page": page, "split": "train"})
                    taken += 1
    return rows


def line_crop(src: str, height: int, max_width: int) -> Image.Image:
    """Resize a line to ``height`` keeping its aspect ratio and centre-crop it to ``max_width`` (no padding).

    Identical to the first two steps of :func:`line_probe.tower_line_image`.

    :param src: Source crop path.
    :param height: Target height in px.
    :param max_width: Width cap in px.
    :returns: RGB image.
    :rtype: Image.Image
    """
    img = Image.open(src).convert("RGB")
    width = max(GRID, round(img.width * height / img.height))
    img = img.resize((width, height), Image.BICUBIC)
    if width > max_width:
        x0 = (width - max_width) // 2
        img = img.crop((x0, 0, x0 + max_width, height))
    return img


def smart_resize(height: int, width: int, factor: int = GRID, min_pixels: int = FRAG_MIN_PIXELS,
                 max_pixels: int = FRAG_MAX_PIXELS) -> Tuple[int, int]:
    """The Qwen2-VL/Qwen3-VL processor's target size (same arithmetic as transformers' ``smart_resize``).

    :param height: Input height.
    :param width: Input width.
    :param factor: Patch x merge (32).
    :param min_pixels: Pixel budget floor.
    :param max_pixels: Pixel budget ceiling.
    :returns: (height, width) multiples of ``factor``.
    :rtype: Tuple[int, int]
    """
    import math

    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def fragment_crop(cid: str) -> Image.Image:
    """The ``masked`` view of one fragment, resized to the processor's 512x512-budget target size.

    :param cid: Fragment id.
    :returns: RGB image whose sides are multiples of 32.
    :rtype: Image.Image
    """
    img = load_view(cid, "masked", 16, 320)[0]
    h, w = smart_resize(img.height, img.width)
    return img if (w, h) == img.size else img.resize((w, h), Image.BICUBIC)


def _encode_line(job: tuple) -> tuple:
    """Worker: one line crop -> JPEG bytes.

    :param job: (arcname, src, height, max_width, quality).
    :returns: (arcname, bytes, width, height).
    :rtype: tuple
    """
    arc, src, height, max_width, quality = job
    img = line_crop(src, height, max_width)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return arc, buf.getvalue(), img.width, img.height


def _encode_fragment(job: tuple) -> tuple:
    """Worker: one fragment -> masked, resized JPEG bytes.

    :param job: (arcname, fragment id, quality).
    :returns: (arcname, bytes, width, height).
    :rtype: tuple
    """
    arc, cid, quality = job
    img = fragment_crop(cid)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return arc, buf.getvalue(), img.width, img.height


def write_shard(path: Path, jobs: list, worker, workers: int) -> Dict[str, list]:
    """Encode jobs in a process pool and stream them into one tar shard (tmp + rename; skipped if present).

    :param path: Shard path.
    :param jobs: Worker jobs (first element = arcname).
    :param worker: :func:`_encode_line` or :func:`_encode_fragment`.
    :param workers: Pool size.
    :returns: {arcname: [width, height]} (also stored next to the shard as ``<shard>.sizes.json``).
    :rtype: Dict[str, list]
    """
    sizes_path = path.with_suffix(".sizes.json")
    if path.exists() and sizes_path.exists():
        return json.loads(sizes_path.read_text())
    tmp = path.with_suffix(".tar.tmp")
    sizes = {}
    with tarfile.open(tmp, "w") as tar, Pool(workers) as pool:
        for arc, data, w, h in pool.imap(worker, jobs, chunksize=8):
            info = tarfile.TarInfo(arc)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            sizes[arc] = [w, h]
    tmp.rename(path)
    sizes_path.write_text(json.dumps(sizes))
    return sizes


def fragment_labels(manifest: dict, sel: dict, ids: List[str]) -> dict:
    """Everything the fragment metrics need, restricted to the eval ids (pair/label order preserved).

    :param manifest: ``image_manifest_v1.json``.
    :param sel: ``image_selection_v1.json``.
    :param ids: Fragment-eval ids in cache order.
    :returns: Label dict.
    :rtype: dict
    """
    keep = set(ids)
    scribe_sel = set(sel["scribes"])
    return {
        "ids": ids,
        "collection": {c: manifest["collection"][c] for c in ids},
        "joins_fjp_clean": [p for p in manifest["joins_fjp_clean"] if p[0] in keep and p[1] in keep],
        "joins_pgp": [p for p in manifest["joins_pgp"] if p[0] in keep and p[1] in keep],
        "scribe": {c: s for c, s in manifest["scribe"].items() if c in keep and c in scribe_sel},
        "attrs": {c: {k: a[k] for k in ("script_style", "script_region") if k in a}
                  for c, a in manifest["attrs"].items() if c in keep and ("script_style" in a or "script_region" in a)},
    }


def load_cache(name: str, ids: List[str]) -> np.ndarray:
    """Load a local image-feature cache and order it like ``ids``.

    :param name: Cache folder under ``cache/image_feats``.
    :param ids: Wanted ids.
    :returns: (len(ids), D) float32.
    :rtype: np.ndarray
    """
    got, feats = [], []
    for part in sorted((AUDIT_ROOT / "cache" / "image_feats" / name).glob("part_*.npz")):
        z = np.load(part)
        got += [str(c) for c in z["ids"]]
        feats.append(z["feats"])
    X = np.concatenate(feats)
    pos = {c: i for i, c in enumerate(got)}
    return X[[pos[c] for c in ids]].astype(np.float32)


def smoke_subset(rows: List[dict], frag_ids: List[str], labels_full: dict, n_frag: int) -> Tuple[List[dict], List[str]]:
    """Tiny selection for smoke tests: 6 train manuscripts x 2 pages x 3 lines (enough for line stacks), 2 val,
    4 test manuscripts; ~n_frag fragments chosen so joins and scribes are non-empty.

    :param rows: Full line rows.
    :param frag_ids: Full fragment ids.
    :param labels_full: Full fragment labels.
    :param n_frag: Fragments wanted.
    :returns: (rows, fragment ids).
    :rtype: Tuple[List[dict], List[str]]
    """
    out, seen = [], Counter()
    by_split_ms = defaultdict(list)
    for r in rows:
        by_split_ms[(r["split"], r["ms"])].append(r)
    quota = {"train": 6, "val": 2, "test": 4}
    for (split, ms), rs in sorted(by_split_ms.items()):
        per_page = Counter(r["page"] for r in rs)
        if seen[split] >= quota[split] or (split == "train" and sum(n >= 3 for n in per_page.values()) < 2):
            continue
        seen[split] += 1
        if split == "train":
            pages = sorted(p for p, n in per_page.items() if n >= 3)[:2]
            out += [r for p in pages for r in [x for x in rs if x["page"] == p][:3]]
        else:
            out += rs
    pick: List[str] = []
    for a, b in labels_full["joins_fjp_clean"][:8] + labels_full["joins_pgp"][:5]:
        pick += [a, b]
    by_scribe = defaultdict(list)
    for c, s in labels_full["scribe"].items():
        by_scribe[s].append(c)
    for s, cs in sorted(by_scribe.items(), key=lambda kv: -len(kv[1]))[:3]:
        pick += cs[:4]
    pick += frag_ids[: n_frag]
    keep = set(list(dict.fromkeys(pick))[:n_frag])
    return out, [c for c in frag_ids if c in keep]


def main() -> None:
    """Select, exclude, encode, label, and write the dataset folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="dataset_v1")
    parser.add_argument("--out", type=Path, default=None, help="default <AUDIT_ROOT>/hf_image/<name>")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--per-page", type=int, default=32, help="train lines per page cap")
    parser.add_argument("--per-ms", type=int, default=128, help="train lines per manuscript cap")
    parser.add_argument("--val-frac", type=float, default=0.08)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--lines-per-shard", type=int, default=6000)
    parser.add_argument("--frags-per-shard", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--smoke", action="store_true", help="tiny subset (smoke tests)")
    parser.add_argument("--smoke-frags", type=int, default=50)
    parser.add_argument("--plan-only", action="store_true", help="selection + exclusion report only")
    args = parser.parse_args()
    out = args.out or AUDIT_ROOT / "hf_image" / args.name
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())
    sel = json.loads((AUDIT_ROOT / "image_selection_v1.json").read_text())
    test = json.loads((AUDIT_ROOT / "line_selection_v1.json").read_text())
    test_ms = {t[0] for t in test}
    frag_ids = eval_fragment_ids()
    index = read_line_index()
    records = ktiv_records(set(index))
    excl = exclusion_sets(manifest, set(frag_ids))
    reasons, detail = classify_manuscripts(set(index), records, excl, test_ms)
    keep = sorted(ms for ms in index if ms not in reasons)
    rows = select_lines(index, keep, test, args.val_frac, args.per_page, args.per_ms, args.seed)
    labels = fragment_labels(manifest, sel, frag_ids)
    if args.smoke:
        rows, frag_ids = smoke_subset(rows, frag_ids, labels, args.smoke_frags)
        labels = fragment_labels(manifest, sel, frag_ids)
    for r in rows:
        r["library"] = records.get(r["ms"], {}).get("library", "")
    n_lines_ms = {ms: sum(len(v) for v in pages.values()) for ms, pages in index.items()}
    by_reason = Counter(r for rs in reasons.values() for r in rs)
    beyond_test = sorted(ms for ms, rs in reasons.items() if "i2_test" not in rs)
    report = {
        "line_manuscripts": len(index), "line_crops": sum(n_lines_ms.values()),
        "manuscripts_without_merged_record": sum(1 for ms in index if ms not in records),
        "fragment_eval_ids": len(frag_ids) if not args.smoke else len(eval_fragment_ids()),
        "exclusion_canonical_ids": {k: len(v) for k, v in excl.items()},
        "excluded_manuscripts": len(reasons), "excluded_by_reason": dict(by_reason),
        "excluded_beyond_i2_test": len(beyond_test),
        "lines_removed_beyond_i2_test": sum(n_lines_ms[ms] for ms in beyond_test),
        "i2_test_manuscripts_also_fragment_eval": sum(1 for ms, rs in reasons.items() if "i2_test" in rs and len(rs) > 1),
        "kept_manuscripts": len(keep),
        "selected": {s: {"lines": sum(1 for r in rows if r["split"] == s), "manuscripts": len({r["ms"] for r in rows if r["split"] == s})}
                     for s in ("train", "val", "test")},
        "caps": {"per_page": args.per_page, "per_ms": args.per_ms, "val_frac": args.val_frac},
        "overlap_rows": detail["rows"],
    }
    (out / "exclusions.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "overlap_rows"}, indent=1), flush=True)
    assert not ({r["ms"] for r in rows if r["split"] != "test"} & set(reasons)), "excluded manuscript selected"
    if args.plan_only:
        return

    # line shards (per split, fixed chunks of the deterministic row order -> resumable)
    line_rows = []
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split]
        for k in range(0, len(rs), args.lines_per_shard):
            chunk = rs[k:k + args.lines_per_shard]
            shard = out / f"lines_{split}_{k // args.lines_per_shard:03d}.tar"
            jobs = [(f"lines/{split}/{r['ms']}/{Path(r['src']).name}", r["src"], args.height, args.max_width, args.quality)
                    for r in chunk]
            sizes = write_shard(shard, jobs, _encode_line, args.workers)
            for r, job in zip(chunk, jobs):
                w, h = sizes[job[0]]
                line_rows.append({"path": job[0], "ms": r["ms"], "page": r["page"], "split": split,
                                  "library": r["library"], "w": w, "h": h, "shard": shard.name})
            print(f"{shard.name}: {len(chunk)} lines", flush=True)
    with open(out / "lines.jsonl", "w", encoding="utf-8") as fh:
        for r in line_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    frag_rows = []
    for k in range(0, len(frag_ids), args.frags_per_shard):
        chunk = frag_ids[k:k + args.frags_per_shard]
        shard = out / f"fragments_{k // args.frags_per_shard:03d}.tar"
        jobs = [(f"fragments/{c.replace('/', '_')}.jpg", c, args.quality) for c in chunk]
        sizes = write_shard(shard, jobs, _encode_fragment, args.workers)
        for c, job in zip(chunk, jobs):
            w, h = sizes[job[0]]
            frag_rows.append({"id": c, "path": job[0], "collection": labels["collection"][c], "w": w, "h": h,
                              "shard": shard.name})
        print(f"{shard.name}: {len(chunk)} fragments", flush=True)
    with open(out / "fragments.jsonl", "w", encoding="utf-8") as fh:
        for r in frag_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "fragment_labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    np.savez(out / "fragment_dinov2.npz", ids=np.array(frag_ids), masked=load_cache("dinov2-base__masked", frag_ids),
             mpatches=load_cache("dinov2-base__mpatches", frag_ids))
    np.savez(out / "fragment_tower_local.npz", ids=np.array(frag_ids), feats=load_cache(TOWER_CACHE, frag_ids).astype(np.float16),
             note=np.array(f"local {TOWER_CACHE} (image_features, CPU fp32, from the full-resolution masked view)"))
    shutil.copy(HERE / "train_image_embedder.py", out / "train_image_embedder.py")
    sel_counts = report["selected"]
    (out / "README.md").write_text(README.format(
        name=args.name, tower=TOWER_CACHE.split("__")[0], date=date.today().isoformat(), height=args.height,
        max_width=args.max_width, n_train=sel_counts["train"]["lines"], ms_train=sel_counts["train"]["manuscripts"],
        n_val=sel_counts["val"]["lines"], ms_val=sel_counts["val"]["manuscripts"], n_test=sel_counts["test"]["lines"],
        ms_test=sel_counts["test"]["manuscripts"], n_frag=len(frag_ids), n_excl_ms=len(reasons)))
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(json.dumps({"dir": str(out), "bytes": total, "GB": round(total / 1e9, 3), "lines": len(line_rows),
                      "fragments": len(frag_rows)}), flush=True)


if __name__ == "__main__":
    main()
