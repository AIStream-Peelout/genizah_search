"""Write ``train_genizah_image_embedder.ipynb``: a thin Colab wrapper around ``train_image_embedder.py``.

Like the text notebook, it clones nothing: ``train_image_embedder.py`` is downloaded from the private dataset repo
(``package_image_dataset.py`` ships it with the data), so the code that trained a model is versioned with its data.
The tower comes from a separate private model repo (the folder ``export_tower.py`` wrote).
"""

import json
from pathlib import Path

from make_notebook import code, md

HERE = Path(__file__).parent

CELLS = [
    md("""
# Genizah handwriting / fragment image encoder (Qwen3-VL-8B vision tower, LoRA)

Runtime: **A100** (Runtime → Change runtime type). Needs the Colab secret `HF_TOKEN` (read access to the private
dataset and tower repos, write access to `isaacmg`). `DATA_REVISION` must be an existing tag of the dataset repo
(`huggingface-cli tag isaacmg/genizah-image-train v1 --repo-type dataset` after the upload; see
`package_image_dataset.py`).

What it does:
1. Downloads the dataset (KTIV line crops, masked fragment crops, labels, local DINOv2 features) and the exported
   vision tower of our Hebrew VLM (`heb-v22b-step1200`, bf16, 0.58B params).
2. Evaluates the **base** tower with the audit's pooling (mean of merged tokens) for every readout — the final
   `merger` and the deepstack taps `ds8`/`ds16`/`ds24`, one forward: I2 = line retrieval on the 600 held-out test
   manuscripts (other page, same manuscript), I1 = joins / scribes / collection confound on the 5,176 masked
   fragments, each also fused with DINOv2. Local anchors (I2 P@1, h96): merger 0.067, ds8 0.371, ds16 0.248,
   ds24 0.153, DINOv2 0.344; I1 merger: joins R@10 0.326, scribe P@1 0.457. The notebook also prints the cosine
   between its fragment merger vectors and the local cache (expect ≳ 0.99).
3. Trains LoRA (vision blocks' attention + MLP, and the mergers) + a projection head with a supervised contrastive
   loss on 48 manuscripts × 4 items per step (GradCache, bf16 autocast, gradient checkpointing). `READOUT` picks
   the pooled vector: `ds8+merger` (default: line-scale style tap + the fragment-scale best) or `ds8` alone, which
   truncates the tower after block 8 and makes steps ~3× cheaper — pick it if the base eval shows ds8 also leads
   at fragment level.
4. Evaluates the tuned model (the tower's own pooled vector and the head output) on the same held-out sets.
5. Saves adapter + head + an fp32-merged tower and pushes them with the reports to a private model repo.

Time: base eval ~5 min; training ~1 h for 1,000 steps (unmeasured: watch `s_per_step` in the first log lines).
`max_minutes` caps the run: after ~15 steps the cosine schedule's horizon shrinks to the steps that fit, so a slow
run still ends annealed; the best validation state (mean of pooled and head val mAP) is restored at the end. Tuned
eval ~5 min; save/push ~5 min (2.3 GB merged tower).
"""),
    code("""
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # before torch touches the GPU
!pip -q install "transformers==4.57.6" "peft==0.17.1" accelerate huggingface_hub
import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
"""),
    code("""
import os, sys, json
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
DATA_REPO = "isaacmg/genizah-image-train"                    # private dataset repo (package_image_dataset.py)
DATA_REVISION = "v1"                                          # dataset tag
TOWER_REPO = "isaacmg/qwen3-vl-8b-heb-v22b-vision-tower"     # private model repo (export_tower.py)
MODEL_REPO = "isaacmg/genizah-image-tower-v22b-lora"          # private output repo
RUN_NAME = "v1"
READOUT = "ds8+merger"                                        # or "ds8" (fast, truncated tower), "merger", "ds8+ds16"
TAPS = ("merger", "ds8", "ds16", "ds24")
TRAIN_ARGS = dict(steps=1000, P=48, K=4, lr=2e-4, head_lr=1e-3, temperature=0.07, pooled_weight=0.5,
                  stack_prob=0.3, library_block=8, chunk_patches=65536, lora_r=16, lora_alpha=32,
                  eval_every=200, num_workers=10, max_minutes=90, select_on="mean")
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.path.dirname(hf_hub_download(DATA_REPO, "train_image_embedder.py", repo_type="dataset",
                                                   token=HF_TOKEN, revision=DATA_REVISION)))
import train_image_embedder as tie
data = tie.load_data(DATA_REPO, HF_TOKEN, revision=DATA_REVISION)
from collections import Counter
print(Counter(r["split"] for r in data["lines"]), len(data["fragments"]), "fragments")
"""),
    code("""
# Base tower: the audit's pooling on the held-out sets
tower, processor = tie.load_tower(TOWER_REPO, token=HF_TOKEN)
model = tie.build_embedder(tower, readout=READOUT)
report_base = tie.evaluate(model, data, processor, "base-heb-v22b", kinds=("pooled",) + TAPS, splits=("test", "val"),
                           num_workers=8)
json.dump(report_base, open("/content/report_base.json", "w"), indent=1)   # survives a later OOM/restart
print(json.dumps(tie.summary_row(report_base), indent=1))
print("fragment merger vectors vs local cache (cos):", report_base.get("fragments_merger_cos_vs_local_base_cache"))
"""),
    code("""
# Train. chunk_patches is the memory knob (65536 is comfortable on 40 GB; halve it on OOM). Batch stays 48 x 4.
torch.cuda.reset_peak_memory_stats()
history = tie.train(model, data, processor, **TRAIN_ARGS)
print("peak GPU memory GB:", round(torch.cuda.max_memory_allocated() / 1e9, 1))
json.dump(history, open("/content/train_history.json", "w"), indent=1)
"""),
    code("""
report_tuned = tie.evaluate(model, data, processor, f"tuned-{RUN_NAME}", kinds=("pooled", "head") + TAPS,
                            splits=("test", "val"), num_workers=8)   # taps a truncated tower lacks are skipped
json.dump(report_tuned, open("/content/report_tuned.json", "w"), indent=1)
print(json.dumps(tie.summary_row(report_tuned), indent=1))
"""),
    code("""
# Side-by-side (val was used to pick the best step; test and fragments are held out)
b, t = tie.summary_row(report_base), tie.summary_row(report_tuned)
for k in sorted(set(b) | set(t)):
    if k != "model":
        print(f"{k:45s} base {str(b.get(k, '')):>8s}   tuned {str(t.get(k, '')):>8s}")
"""),
    code("""
out = tie.save_tuned(model, processor, f"/content/genizah-image-{RUN_NAME}", [report_base, report_tuned], history,
                     TRAIN_ARGS)
sha = tie.push(str(out), MODEL_REPO, HF_TOKEN)
print("pushed", MODEL_REPO, "commit", sha)
"""),
]


def main() -> None:
    """Write the notebook."""
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = HERE / "train_genizah_image_embedder.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(out)


if __name__ == "__main__":
    main()
