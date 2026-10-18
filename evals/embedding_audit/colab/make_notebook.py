"""Write ``train_genizah_embedder.ipynb``: a thin Colab wrapper around ``train_embedder.py``.

The notebook clones nothing: it downloads ``train_embedder.py`` from the private dataset repo (uploaded with the
data) so the code that trained a model is versioned with the data it used.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text: str) -> dict:
    """Markdown cell.

    :param text: Cell source.
    :returns: Cell dict.
    :rtype: dict
    """
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    """Code cell.

    :param text: Cell source.
    :returns: Cell dict.
    :rtype: dict
    """
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# Genizah embedder fine-tune (Qwen3-Embedding-0.6B)

Runtime: **A100** (Runtime → Change runtime type). Needs the Colab secret `HF_TOKEN` (read access to the private
dataset repo, write access to `isaacmg`). The run takes about 20–40 minutes, most of it corpus embedding for
hard-negative mining and the two evaluations.

What it does:
1. Downloads the train mix, corpus, eval queries and held-out ids.
2. Mines one hard negative per pair with the base model.
3. Fully fine-tunes the model.
4. Evaluates base vs tuned on held-out records only.
5. Pushes the model and its eval report to a private repo. Pin the printed commit in the embedding contract.
"""),
    code("""
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # before torch touches the GPU
!pip -q install "sentence-transformers==5.6.1" "transformers==4.57.6" datasets accelerate huggingface_hub
import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
"""),
    code("""
import os, sys, json
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
DATA_REPO = "isaacmg/genizah-embed-train"          # private dataset repo
MODEL_REPO = "isaacmg/genizah-embed-qwen3-0.6b"     # private model repo
DATA_REVISION = "v2"   # dataset tag: v1 = English queries + corpus + scholarship; v2 adds Hebrew/translit queries + cleaner labels; v3 adds Sefaria
RUN_NAME = DATA_REVISION
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.path.dirname(hf_hub_download(DATA_REPO, "train_embedder.py", repo_type="dataset", token=HF_TOKEN, revision=DATA_REVISION)))
import train_embedder as te
data = te.load_data(DATA_REPO, HF_TOKEN, revision=DATA_REVISION)
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in data.items() if k != "path"})
"""),
    code("""
# Baseline: production model on the held-out benchmarks
from sentence_transformers import SentenceTransformer
base = SentenceTransformer(te.BASE_MODEL, revision=te.BASE_REVISION, model_kwargs={"torch_dtype": torch.bfloat16})
base.max_seq_length = 8192   # evaluate() also forces the production window
report_base = te.evaluate(base, data, "base")
print(json.dumps(report_base, indent=1))
json.dump(report_base, open("/content/report_base.json", "w"))   # survives a later OOM/restart
"""),
    code("""
# Hard negatives with the base model, then free it
pairs = te.mine_negatives(base, data)
print(sum(1 for p in pairs if p.get("negative")), "of", len(pairs), "pairs got a hard negative")
del base; torch.cuda.empty_cache()
"""),
    code("""
# mini_batch is the memory knob (8 fits a 40 GB A100; drop to 4 if it still OOMs). batch stays 256.
tuned = te.train(data, pairs, out_dir=f"/content/genizah-embed-{RUN_NAME}", epochs=1.0, lr=2e-5, batch=256, mini_batch=8)
report_tuned = te.evaluate(tuned, data, f"tuned-{RUN_NAME}")
print(json.dumps(report_tuned, indent=1))
"""),
    code("""
# Side-by-side summary
def row(r):
    t3 = {k: v["R@10"] for k, v in r["t3_known_item"].items()}
    return {"t3_R@10": t3, "t2_macro": r["t2_macro"], "sukkot_gold": r["sukkot_gold"]}
print(json.dumps({"base": row(report_base), "tuned": row(report_tuned)}, indent=1))
"""),
    code("""
sha = te.push(f"/content/genizah-embed-{RUN_NAME}", MODEL_REPO, HF_TOKEN, [report_base, report_tuned])
print("pushed", MODEL_REPO, "commit", sha)
"""),
]


def main() -> None:
    """Write the notebook."""
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = HERE / "train_genizah_embedder.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(out)


if __name__ == "__main__":
    main()
