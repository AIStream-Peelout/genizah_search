"""Package everything the gallery notebook needs into ``<AUDIT_ROOT>/hf_gallery/`` (no upload).

Layout (uploaded as-is to the private dataset repo ``isaacmg/genizah-image-gallery`` and tagged ``inputs-v1``)::

    README.md
    embed_genizah_gallery.ipynb          the notebook (convenience copy)
    inputs/manifest.jsonl                gallery_v1 manifest (build_gallery_manifest.py)
    inputs/known_joins.json              + manifest_stats.json
    inputs/benchmark.json                the 5,176 benchmark ids (cache order) and the image URL each was made from
    inputs/fragment_labels.json          labels for train_image_embedder.fragment_metrics (hf_image/dataset_v1 copy)
    inputs/benchmark_reference.json      the published tuned-tower I1 results (tuned-v1, model commit c45c8de4)
    inputs/parity_local.npz              200 benchmark fragments: local DINOv2 masked/mpatches vectors + geometry
    code/embed_gallery.py, code/gallery_preprocess.py, code/train_image_embedder.py

Parity fragments: a seeded sample of the benchmark plus the three probe images whose colour segmentation falls back
to the backdrop-distance crop. Their geometry (max1600 size, masked-crop size, mask path, ink-patch count) is
computed here with the ORIGINAL scikit-image pipeline (``image_features``) on ``images/max1600``; the vectors come
from the local caches ``cache/image_feats/dinov2-base__{masked,mpatches}``.

The upload commands are written to ``<AUDIT_ROOT>/hf_gallery_UPLOAD.txt`` (outside the package) and printed.
Run (NumPy/PIL/skimage only, no model): ``nice -n 10 <HDA .venv>/bin/python colab/package_gallery.py``.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))
from embed_utils import AUDIT_ROOT  # noqa: E402
from image_features import IMG_DIR, fragment_mask, load_view  # noqa: E402

GALLERY = AUDIT_ROOT / "gallery_v1"
DATASET_V1 = AUDIT_ROOT / "hf_image" / "dataset_v1"
TUNED_REPORT = AUDIT_ROOT / "colab_runs" / "image-v22b-lora" / "genizah_image_eval_report.json"
FALLBACK_EXAMPLES = ("Cambridge_CUL_T_S_AS_152_19", "New_York_JTS_ENA_NS_85_1312_2", "New_York_JTS_ENA_NS_85_663")
REPO = "isaacmg/genizah-image-gallery"
TAG = "inputs-v1"

README = """---
license: other
---
# genizah-image-gallery

Private. Image features of every imaged Cairo Genizah fragment in the merged catalogue ({n:,} records; one image each,
the first live one; {n_dead:,} records whose images were all HEAD-checked dead are skipped), for join discovery and
scribe attribution (`genizah_search/evals/embedding_audit`). Built {date}.

* `inputs/` (tag `inputs-v1`) — gallery manifest (labels: known join groups, PGP scribes/types/dates, KTIV
  material/script, benchmark and tower-training flags), benchmark ids/URLs/labels, published reference numbers and
  200 local parity vectors. `code/` — the exact code the notebook runs.
* `shards/part_XXXX.npz` — written by `embed_genizah_gallery.ipynb`: `ids`, `urls`, fp16 `merger` / `ds8` / `head`
  (tuned v22b tower, `isaacmg/genizah-image-tower-v22b-lora` @ c45c8de4) and `dino_masked` / `dino_mpatches`
  (DINOv2-base), all L2-normalised before the fp16 cast; `pooled` = `embed_gallery.pooled_from_parts(ds8, merger)`;
  provenance `sha1` (downloaded bytes), `max1600_wh`, `crop_wh`, `mask_ok`, `mask_frac`, `n_patches`, `dhash`
  (64-bit difference hash of the masked crop), `run_info`. `failures/part_XXXX.jsonl` — fragments that failed.
* `benchmark/` — the 5,176 benchmark fragments through the same pipeline and the gate report.

Images: public bucket `storage.googleapis.com/cairo-genizah-es-json` (FGP/KTIV/partner libraries); research use only.
"""


def parity_ids(bench_ids: List[str], n: int, seed: int) -> List[str]:
    """Seeded sample of benchmark ids with a local max1600 file, always including the fallback examples.

    :param bench_ids: Benchmark ids.
    :param n: Sample size.
    :param seed: Sampling seed.
    :returns: Sorted ids.
    :rtype: List[str]
    """
    have = [c for c in bench_ids if (IMG_DIR / f"{c.replace('/', '_')}.jpg").exists()]
    forced = [c for c in FALLBACK_EXAMPLES if c in set(have)]
    rest = sorted(set(have) - set(forced))
    return sorted(forced + random.Random(seed).sample(rest, n - len(forced)))


def load_cache(name: str, ids: List[str]) -> np.ndarray:
    """Rows of a local feature cache in ``ids`` order.

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
    pos = {c: i for i, c in enumerate(got)}
    return np.concatenate(feats)[[pos[c] for c in ids]].astype(np.float32)


def local_geometry(ids: List[str]) -> Dict[str, np.ndarray]:
    """max1600 size, masked-crop size, mask path and ink-patch count from the ORIGINAL local pipeline.

    :param ids: Fragment ids.
    :returns: {max1600_wh, crop_wh, mask_ok, n_patches} arrays.
    :rtype: Dict[str, np.ndarray]
    """
    out = {"max1600_wh": [], "crop_wh": [], "mask_ok": [], "n_patches": []}
    for cid in ids:
        img = Image.open(IMG_DIR / f"{cid.replace('/', '_')}.jpg").convert("RGB")
        masked, mask = fragment_mask(img)
        out["max1600_wh"].append(img.size)
        out["crop_wh"].append(masked.size)
        out["mask_ok"].append(mask is not None)
        out["n_patches"].append(len(load_view(cid, "mpatches", 16, 320)))
    return {"max1600_wh": np.array(out["max1600_wh"], np.int32), "crop_wh": np.array(out["crop_wh"], np.int32),
            "mask_ok": np.array(out["mask_ok"], bool), "n_patches": np.array(out["n_patches"], np.int16)}


def upload_text(out: Path) -> str:
    """The upload commands (not executed here).

    :param out: Package folder.
    :returns: Text.
    :rtype: str
    """
    return f"""# Upload the gallery-embedding inputs (private dataset repo) and tag them; run from the Studio.
# Python route (reads HF1_TOKEN from the sibling repo's .env, never prints it):
cd ~/Documents/GitHub/genizah_search/evals/embedding_audit
nice -n 10 ~/Documents/GitHub/historical-document-analysis/.venv/bin/python colab/push_gallery_inputs.py

# CLI route (same result; HF_TOKEN must hold a write token; keep every HF cache on the NAS):
export HF_HOME={AUDIT_ROOT / "hf"}
huggingface-cli upload {REPO} {out} . --repo-type dataset --private --commit-message "gallery inputs {TAG}"
huggingface-cli tag {REPO} {TAG} --repo-type dataset
# (newer CLI: hf upload {REPO} {out} . --repo-type dataset --private ; hf repo tag create {REPO} {TAG} --repo-type dataset)

# Re-packaging later: bump the tag (inputs-v2 ...) and INPUT_REVISION in the notebook.
"""


def main() -> None:
    """Assemble the package folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=AUDIT_ROOT / "hf_gallery")
    parser.add_argument("--n-parity", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    out = args.out
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)

    for name in ("manifest.jsonl", "known_joins.json", "manifest_stats.json"):
        shutil.copy(GALLERY / name, out / "inputs" / name)
    labels = json.loads((DATASET_V1 / "fragment_labels.json").read_text())
    bench_ids = labels["ids"]
    urls = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())["images"]
    assert all(c in urls for c in bench_ids), "benchmark id without a v1 image URL"
    (out / "inputs" / "benchmark.json").write_text(json.dumps({"ids": bench_ids, "urls": {c: urls[c] for c in bench_ids}}))
    shutil.copy(DATASET_V1 / "fragment_labels.json", out / "inputs" / "fragment_labels.json")
    report = json.loads(TUNED_REPORT.read_text())
    tuned = next(r for r in report if r["model"] == "tuned-v1")
    ref = {k: v for k, v in tuned.items() if k.startswith("i1_")}
    ref["_source"] = {"file": str(TUNED_REPORT), "model": tuned["model"], "readout": tuned.get("readout"),
                      "model_repo": "isaacmg/genizah-image-tower-v22b-lora",
                      "model_commit": "c45c8de439585c21e03f434c3b7f9c91a5a623dd"}
    (out / "inputs" / "benchmark_reference.json").write_text(json.dumps(ref, indent=1))

    pids = parity_ids(bench_ids, args.n_parity, args.seed)
    geo = local_geometry(pids)
    np.savez(out / "inputs" / "parity_local.npz", ids=np.array(pids), urls=np.array([urls[c] for c in pids]),
             dino_masked=load_cache("dinov2-base__masked", pids), dino_mpatches=load_cache("dinov2-base__mpatches", pids),
             **geo, note=np.array("local CPU fp32 DINOv2-base (image_features.py) on images/max1600; geometry from "
                                  "image_features.fragment_mask / load_view (scikit-image pipeline)"))
    for name in ("embed_gallery.py", "gallery_preprocess.py", "train_image_embedder.py"):
        shutil.copy(HERE / name, out / "code" / name)
    subprocess.run([sys.executable, str(HERE / "make_gallery_notebook.py")], check=True, cwd=HERE)
    shutil.copy(HERE / "embed_genizah_gallery.ipynb", out / "embed_genizah_gallery.ipynb")
    live = [json.loads(line).get("image_live") for line in open(out / "inputs" / "manifest.jsonl", encoding="utf-8")]
    n, n_dead = len(live), sum(1 for v in live if v is False)
    (out / "README.md").write_text(README.format(n=n, n_dead=n_dead, date=date.today().isoformat()))
    cmd = upload_text(out)
    (AUDIT_ROOT / "hf_gallery_UPLOAD.txt").write_text(cmd)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"dir": str(out), "MB": round(size / 1e6, 1), "manifest_rows": n, "benchmark": len(bench_ids),
                      "parity": len(pids), "parity_fallback_examples": int((~geo["mask_ok"]).sum()),
                      "parity_patches_lt16": int((geo["n_patches"] < 16).sum())}, indent=1))
    print(cmd)


if __name__ == "__main__":
    main()
