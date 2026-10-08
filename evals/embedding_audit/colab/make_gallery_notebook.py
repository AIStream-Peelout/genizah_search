"""Write ``embed_genizah_gallery.ipynb``: a thin Colab (A100) wrapper around ``embed_gallery.py``.

The notebook clones nothing: ``embed_gallery.py``, ``gallery_preprocess.py`` and ``train_image_embedder.py`` are
downloaded with the inputs from the private dataset repo (``package_gallery.py`` packs them), so the code that
produced a shard is versioned next to it. Order of cells: setup -> parity gate (200 fragments) -> benchmark gate
(5,176 fragments, published R@10) -> full gallery (resumable) -> retry failures -> verify.
"""

import json
from pathlib import Path

from make_notebook import code, md

HERE = Path(__file__).parent

CELLS = [
    md("""
# Genizah image gallery: tuned-tower + DINOv2 features for every imaged fragment

Runtime: **A100** (Runtime → Change runtime type; High-RAM helps the image workers). Colab secret `HF_TOKEN`: read
access to `isaacmg/genizah-image-tower-v22b-lora`, write access to `isaacmg/genizah-image-gallery`.

**Before the first run**, the inputs must be on the Hub (from the Studio; see `<AUDIT_ROOT>/hf_gallery_UPLOAD.txt`):
manifest (55,406 records; the ~2,400 whose images were all HEAD-checked dead are skipped, ~53,000 embedded),
benchmark ids/URLs/labels, published reference numbers, 200 parity vectors and the code, in
`isaacmg/genizah-image-gallery` tagged `inputs-v1`.

What it does, per fragment: download the embedded image (the first live one) from the public GCS bucket (thread
pool, retries; permanent 404s are logged, never retried, and do not count as an outage), resize to max
side 1600 exactly like the local `images/max1600` copies, mask the backdrop with the NumPy port of
`image_features.fragment_mask`, then compute
* the **tuned v22b tower** (`train_image_embedder.load_tuned` + `embed`, 512×512-pixel budget): `merger`, `ds8`,
  `pooled` (= `ds8+merger`, rebuilt from the stored parts) and `head`;
* **DINOv2-base** `masked` (CLS ‖ mean patch @448) and `mpatches` (16 ink patches inside the mask @224).

Shards of 2,000 fragments go to `shards/part_XXXX.npz` (fp16) with `failures/part_XXXX.jsonl`. **Resuming**: after a
disconnect re-run the setup cell and the gallery cell; shards already in the repo are skipped and the gate result
stored in `benchmark/benchmark_report.json` is accepted, so the gates need not be repeated.

Gates (the gallery cell refuses to start until they pass):
1. **Parity** (200 benchmark fragments, ~2 min): identical max1600 size / crop / mask path / patch count to the local
   pipeline, DINOv2 cosine > 0.99 to the local cache. Prints the measured throughput and an ETA.
2. **Benchmark** (5,176 fragments, ~6.5 GB download, ~5–15 min): tuned `merger` joins R@10 within 0.01 of the
   published 0.592 (also prints pooled / head / +DINOv2 / scribe P@1 next to the published values).

Time for the gallery (~66 GB of images: mean 1.24 MB per first image on a HEAD sample) is CPU-bound (full-size JPEG
decode + masking in the image workers): the parity cell's ETA is a first guess, the progress lines show the steady
rate (expect very roughly 1–2 h on 12 vCPUs).
"""),
    code("""
!pip -q install "transformers==4.57.6" "peft==0.17.1" "pillow==12.0.0" accelerate huggingface_hub requests
import os, torch, PIL
print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU", "| CPUs:", os.cpu_count())
# Pillow must match the Studio's (12.0.0: resampling + bundled libjpeg of the local features). If this prints an older
# version, Runtime -> Restart session, then run from the next cell (the install is kept).
print("Pillow", PIL.__version__, "OK" if PIL.__version__ == "12.0.0" else "-> RESTART THE SESSION NOW")
"""),
    code("""
import os, sys, json
from google.colab import userdata
from huggingface_hub import snapshot_download
HF_TOKEN = userdata.get("HF_TOKEN")
GALLERY_REPO = "isaacmg/genizah-image-gallery"     # private dataset repo: inputs (tag inputs-v1) + output shards
INPUT_REVISION = "inputs-v1"
TOWER_REPO = "isaacmg/genizah-image-tower-v22b-lora"
TOWER_REVISION = "c45c8de439585c21e03f434c3b7f9c91a5a623dd"   # the published tuned tower (merger joins R@10 0.592)
import shutil
SHM_GB = shutil.disk_usage("/dev/shm").total / 1e9
# image workers (download + decode + mask). Each returns ~150 MB chunks through /dev/shm, 2 prefetched per worker:
# keep that under half of /dev/shm (lower WORKERS further on "bus error" / shm errors)
WORKERS = max(2, min(os.cpu_count() or 8, int(0.5 * SHM_GB / (2 * 0.16))))
print(f"/dev/shm {SHM_GB:.1f} GB -> WORKERS = {WORKERS}")
THREADS = 8                               # concurrent downloads per worker
TOWER_BF16 = True                         # bf16 autocast, as in the published eval (the parity cell reports the effect)
OUT_DIR = "/content/gallery_out"

inp_dir = snapshot_download(GALLERY_REPO, repo_type="dataset", revision=INPUT_REVISION, token=HF_TOKEN,
                            local_dir="/content/gallery_inputs", allow_patterns=["inputs/*", "code/*", "README.md"])
sys.path.insert(0, os.path.join(inp_dir, "code"))
import embed_gallery as eg, gallery_preprocess as gp, train_image_embedder as tie
assert eg.TOWER_REVISION == TOWER_REVISION
inputs = eg.load_inputs(inp_dir)
print(len(inputs["rows"]), "manifest records,", len(eg.gallery_rows(inputs["rows"])), "to embed |", len(inputs["bench_ids"]),
      "benchmark |", len(inputs["parity"]["ids"]), "parity")
tower_dir = snapshot_download(TOWER_REPO, revision=TOWER_REVISION, token=HF_TOKEN, local_dir="/content/tower",
                              allow_patterns=["tower_merged/*", "head.pt", "head_config.json"])
emb = eg.GalleryEmbedder(tower_dir, bf16=TOWER_BF16)
GATES_PASSED = False
"""),
    md("""
## Gate 1 — parity with the local pipeline (200 benchmark fragments)
"""),
    code("""
rep_parity = eg.run_parity(emb, inputs, WORKERS, THREADS)
"""),
    md("""
## Gate 2 — the 5,176-fragment benchmark through this pipeline must reproduce the published I1 numbers

The published 0.592 came from the bf16 tower with the LoRA unmerged; here the fp32 merged tower runs under bf16
autocast (`TOWER_BF16 = True`). If the gate misses narrowly, set `TOWER_BF16 = False` in the setup cell (fp32 weights,
TF32 matmuls; the parity cell printed how far the two modes differ), re-run setup and both gates. A large miss means
the pipeline differs: do not run the gallery.
"""),
    code("""
rep_bench = eg.run_benchmark(emb, inputs, WORKERS, THREADS, out_dir=f"{OUT_DIR}/benchmark")
print("uploaded benchmark outputs:", eg.upload_benchmark(GALLERY_REPO, HF_TOKEN, f"{OUT_DIR}/benchmark"))
GATES_PASSED = True
"""),
    md("""
## Full gallery (resumable: re-run setup + this cell after a disconnect)

`LIMIT_SHARDS = 1` does one shard (2,000 fragments) as a timed trial; `None` runs everything still missing. A shard
stops the run (not uploaded) only when more than 20 % of it fails with TRANSIENT errors (an outage); permanent 404s
are logged in `failures/` and listed by the retry cell.
"""),
    code("""
LIMIT_SHARDS = None
assert GATES_PASSED or eg.gate_passed_in_repo(GALLERY_REPO, HF_TOKEN), "run both gates first"
summary = eg.run_gallery(emb, inputs, GALLERY_REPO, HF_TOKEN, OUT_DIR, WORKERS, THREADS, limit_shards=LIMIT_SHARDS)
print(json.dumps(summary))
"""),
    md("""
## Retry logged failures (one more round; permanent HTTP 4xx are listed, not retried)
"""),
    code("""
retry = eg.run_retry(emb, inputs, GALLERY_REPO, HF_TOKEN, OUT_DIR, WORKERS, THREADS)
print(json.dumps({k: v for k, v in retry.items() if k != "permanent"}), "| permanent:", len(retry["permanent"]))
for r in retry["permanent"][:20]:
    print(r)
"""),
    md("""
## Verify coverage and write `gallery_index.json`

Then, on the Studio: `discover_joins_v2.py --fetch` pulls the shards into the NAS cache and runs discovery.
"""),
    code("""
index = eg.verify_repo(GALLERY_REPO, HF_TOKEN, len(eg.gallery_rows(inputs["rows"])))
"""),
]


def main() -> None:
    """Write the notebook next to this script."""
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = HERE / "embed_genizah_gallery.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(out)


if __name__ == "__main__":
    main()
