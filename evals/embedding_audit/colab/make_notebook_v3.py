"""Write ``train_genizah_embedder_v3.ipynb``: a thin Colab wrapper around ``train_embedder_v3.py``.

The notebook clones nothing: it downloads the dataset snapshot at ``DATA_REVISION`` (``v3``), which carries
``train_embedder_v3.py`` and ``eval_v3.py``, so the code that trained and scored a model is versioned with its data.
Every evaluation is written to ``REPORT_DIR`` as soon as it finishes, so a crash or runtime restart loses at most
the stage that was running.

``--final`` writes ``train_genizah_embedder_v3final.ipynb`` instead: the FINAL run on the ``v3-final`` dataset (the six
former held-out subjects trained on; no gate, the decision metric is the ``focus`` group over held-out records), with
an extra cell that scores the previous model (``MODEL_REPO`` at ``--previous-revision``, default ``v3-run1``) on the
same final eval, so base_original, base_semantic, v3-run1 and the new model are compared side by side::

    python3 colab/make_notebook_v3.py            # v3 notebook (unchanged)
    python3 colab/make_notebook_v3.py --final    # DATA_REVISION v3-final, RUN_NAME v3-final-run1
"""

import argparse
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
# Genizah embedder fine-tune v3 (Qwen3-Embedding-0.6B, semantic text)

Runtime: **A100** (Runtime -> Change runtime type). Needs the Colab secret `HF_TOKEN` (read access to the private
dataset repo, write access to `isaacmg`). Expect roughly 1-1.5 h: three full evaluations of the 60,493-record pool
at the production window (8192 tokens), hard-negative mining, and one training epoch.

What it does:
1. Downloads the dataset snapshot at tag `v3` (train mix, gated train pool, semantic + original corpus text, frozen
   eval set, and the training / scoring code).
2. Scores the **base model on the original production text** (what production serves today).
3. Scores the **base model on the semantic text** (no Document ID / Shelf Mark lines): the step-0 effect, no training.
4. Mines one hard negative per pair with the base model (train pool only; never held-out records).
5. Fine-tunes with a masked GradCache InfoNCE (in-batch candidates of the same subject are not used as negatives).
6. Scores the **tuned model on the semantic text** and prints all three side by side, with paired bootstrap CIs.
7. Pushes the model and every report to the private model repo, tagged `RUN_NAME`.

How to read the result: the decision metric is the **held-out-subject gate** (`GATE held-out subjects AP`, 6
subjects the model never saw in training), tuned vs base-on-semantic-text. Also check that the memorisation gap on the
records the model was actually trained on stays near the base model's (paired row `memorisation_trained gap (net of
baseline gap)`; the base gap itself is slightly negative by construction, so read the net value) and that collection
lift does not rise (the model must not learn shelf marks / collections). With only 6 held-out subjects the CI is wide
by design. The semantic text masks shelf marks / PGPIDs cited inside descriptions as `[shelfmark]` (as in training).
"""),
    code("""
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # before torch touches the GPU
!pip -q install "sentence-transformers==5.6.1" "transformers==4.57.6" datasets accelerate huggingface_hub
import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
# fp32 weights for every model under test (base and tuned alike); TF32 matmuls keep the 3 full evaluations fast
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
"""),
    code("""
import gc, json, sys
from google.colab import userdata
from huggingface_hub import snapshot_download
HF_TOKEN = userdata.get("HF_TOKEN")
DATA_REPO = "isaacmg/genizah-embed-train"        # private dataset repo
DATA_REVISION = "v3"                              # dataset tag: v3 = semantic text + subject gate + masked loss
MODEL_REPO = "isaacmg/genizah-embed-qwen3-0.6b"   # private model repo
RUN_NAME = "v3-run1"                              # also the git tag on MODEL_REPO: change it for every run
REPORT_DIR = f"/content/reports_{RUN_NAME}"
MODEL_DIR = f"/content/genizah-embed-{RUN_NAME}"
TUNED = f"tuned_{RUN_NAME}"
os.makedirs(REPORT_DIR, exist_ok=True)
DATA_DIR = snapshot_download(DATA_REPO, repo_type="dataset", token=HF_TOKEN, revision=DATA_REVISION,
                             local_dir="/content/genizah_embed_data_v3")
sys.path.insert(0, DATA_DIR)                      # train_embedder_v3.py and eval_v3.py ship with the data
import train_embedder_v3 as te, eval_v3
data = te.load_data(DATA_DIR)                     # also re-checks that nothing held out is in the train side
print({k: len(data[k]) for k in ("train", "pool", "semantic", "original", "eval_pool", "held_out")})
print(json.dumps(data["manifest"].get("summary"), indent=1))
"""),
    code("""
# 1. Base model on the ORIGINAL production text (today). Saved to REPORT_DIR as soon as it finishes.
base = te.load_base()
report_base_original = te.evaluate(base, data, "base_original", text="original",
                                   out_path=f"{REPORT_DIR}/base_original.json")
print(json.dumps(eval_v3.summary(report_base_original), indent=1, ensure_ascii=False)[:4000])
"""),
    code("""
# 2. Base model on the SEMANTIC text (step 0: representation change only, no training)
report_base_semantic = te.evaluate(base, data, "base_semantic", text="semantic",
                                   out_path=f"{REPORT_DIR}/base_semantic.json")
step0 = te.compare_reports({"base_original": report_base_original, "base_semantic": report_base_semantic},
                           baseline="base_original", candidate="base_semantic")
print(te.format_table(step0))
"""),
    code("""
# 3. Hard negatives from the base model (train pool only), then free the base model
pairs = te.mine_negatives(base, data)
mining = te.mining_stats(pairs)
json.dump(mining, open(f"{REPORT_DIR}/mining.json", "w"), indent=1)
print(json.dumps(mining, indent=1))
del base; gc.collect(); torch.cuda.empty_cache()
"""),
    code("""
# 4. Fine-tune. mini_batch is the memory knob (8 fits a 40 GB A100; drop to 4 if it OOMs). batch stays 256.
tuned = te.train(data, pairs, out_dir=MODEL_DIR, epochs=1.0, lr=2e-5, batch=256, mini_batch=8)
info = json.load(open(f"{MODEL_DIR}/training_info.json"))
print(json.dumps({k: info[k] for k in ("global_steps", "train_seconds", "datasets", "masked_share")}, indent=1))
"""),
    code("""
# 5. Tuned model on the SEMANTIC text
report_tuned = te.evaluate(tuned, data, TUNED, text="semantic", out_path=f"{REPORT_DIR}/{TUNED}.json")
"""),
    code("""
# 6. Side by side (read back from REPORT_DIR, so this cell also works after a runtime restart + the setup cells)
reports = te.load_reports(REPORT_DIR, ["base_original", "base_semantic", TUNED])
comparison = te.compare_reports(reports, baseline="base_semantic", candidate=TUNED)
json.dump(comparison, open(f"{REPORT_DIR}/comparison.json", "w"), indent=1)
print(te.format_table(comparison))
print("PRIMARY (gate AP, tuned - base on semantic text):", comparison["primary"])
"""),
    code("""
# 7. Push model + reports (private repo, tagged RUN_NAME). Pin the printed commit in the embedding contract.
sha = te.push(MODEL_DIR, MODEL_REPO, HF_TOKEN, reports, comparison, run_name=RUN_NAME, data_revision=DATA_REVISION)
print("pushed", MODEL_REPO, "tag", RUN_NAME, "commit", sha)
"""),
]


FINAL_INTRO = """
# Genizah embedder fine-tune {data_revision} (Qwen3-Embedding-0.6B, semantic text, FINAL run)

Runtime: **A100** (Runtime -> Change runtime type). Needs the Colab secret `HF_TOKEN` (read access to the private
dataset repo and to `MODEL_REPO`, write access to `isaacmg`). Expect roughly 1.5-2 h: four full evaluations of the
60,493-record pool at the production window (8192 tokens), hard-negative mining, and one training epoch on a mixture
about a quarter larger than v3's.

This is the FINAL run of the v3 recipe (same hyperparameters, families, masked loss and mining as `v3-run1`). The
six subjects that v3 held out to prove generalisation -- Sukkot, Pesach, kashrut, partnership, slavery, the geonic
academies -- and their linked subjects are now TRAINED ON. Honest evaluation is kept by records, not subjects: the
same 20 % record split, the Sukkot gold seed and the known-item eval pool never appear in any training pair (nor
does any record whose text matches one of them up to punctuation / spacing), and no eval query (probe incl. "laws of
lulav", llm_t2, name, known-item) is a training anchor or a near-copy of one. One-word queries ("Haggadah") do occur
inside longer training texts: these subjects are trained on.

What it does:
1. Downloads the dataset snapshot at tag `{data_revision}` (train mix, gated train pool, semantic + original corpus
   text, frozen final eval set, and the training / scoring code).
2. Scores the **base model on the original production text** (what production serves today).
3. Scores the **base model on the semantic text** (step 0, no training).
4. Scores the **previous model `{previous_revision}`** on the same final eval (it never trained on the six subjects).
5. Mines one hard negative per pair with the base model (train pool only; never held-out records).
6. Fine-tunes with the masked GradCache InfoNCE.
7. Scores the **tuned model** and prints all four side by side, with paired bootstrap CIs.
8. Pushes the model and every report to the private model repo, tagged `RUN_NAME`.

How to read the result: there is no held-out-subject gate in this run. The decision metric is the **focus group**
(`FOCUS subjects (held-out records only) AP`): the six subjects scored only on their held-out records (train-side
carriers are left out of the ranking), tuned vs base-on-semantic-text (`primary`), and tuned vs `{previous_label}`
(did training on these topics help on records of them it never saw?). Also read the focus group by query source
(probe / llm_t2 / name) and per subject (`pgp:geonic-academies` has only ~15 held-out positives: its CI is wide), seen
subjects, known-item, the memorisation gap net of the base model's, collection lift (must not rise) and hubness.
`probe` / `llm_t2` are the paraphrase test; read `name` as near-in-distribution (spelling variants of the name queries,
e.g. "Succos", "חג פסח", are training labels of the same subjects).
"""


def final_cells(data_revision: str, run_name: str, previous_revision: str, previous_label: str) -> list:
    """Cells of the final-run notebook (shared stages reuse the v3 cells unchanged).

    :param data_revision: Dataset tag (``v3-final``).
    :param run_name: Run name / model tag (``v3-final-run1``).
    :param previous_revision: Tag of the previous model on ``MODEL_REPO`` (``v3-run1``).
    :param previous_label: Report label of the previous model (``v3_run1``).
    :returns: Cell dicts.
    :rtype: list
    """
    local = "/content/genizah_embed_data_" + data_revision.replace("-", "")
    setup = f"""
import gc, json, sys
from google.colab import userdata
from huggingface_hub import snapshot_download
HF_TOKEN = userdata.get("HF_TOKEN")
DATA_REPO = "isaacmg/genizah-embed-train"        # private dataset repo
DATA_REVISION = "{data_revision}"                      # dataset tag: final run, every subject trained, focus-group eval
MODEL_REPO = "isaacmg/genizah-embed-qwen3-0.6b"   # private model repo
RUN_NAME = "{run_name}"                        # also the git tag on MODEL_REPO: change it for every run
PREV_REVISION = "{previous_revision}"                        # previous model on MODEL_REPO, scored on the same final eval
PREV_LABEL = "{previous_label}"
REPORT_DIR = f"/content/reports_{{RUN_NAME}}"
MODEL_DIR = f"/content/genizah-embed-{{RUN_NAME}}"
TUNED = f"tuned_{{RUN_NAME}}"
os.makedirs(REPORT_DIR, exist_ok=True)
DATA_DIR = snapshot_download(DATA_REPO, repo_type="dataset", token=HF_TOKEN, revision=DATA_REVISION,
                             local_dir="{local}")
sys.path.insert(0, DATA_DIR)                      # train_embedder_v3.py and eval_v3.py ship with the data
import train_embedder_v3 as te, eval_v3
data = te.load_data(DATA_DIR)                     # also re-checks that nothing held out is in the train side
print({{k: len(data[k]) for k in ("train", "pool", "semantic", "original", "eval_pool", "held_out")}})
print(json.dumps(data["manifest"].get("summary"), indent=1))
"""
    previous = """
# 2b. The previous model (PREV_REVISION, trained with the six subjects held out) on the SAME final eval
prev = te.load_previous(MODEL_REPO, revision=PREV_REVISION, token=HF_TOKEN)
report_prev = te.evaluate(prev, data, PREV_LABEL, text="semantic", out_path=f"{REPORT_DIR}/{PREV_LABEL}.json")
print(json.dumps(eval_v3.summary(report_prev)["focus"], indent=1))
del prev; gc.collect(); torch.cuda.empty_cache()
"""
    compare = """
# 6. Side by side (read back from REPORT_DIR, so this cell also works after a runtime restart + the setup cells)
reports = te.load_reports(REPORT_DIR, ["base_original", "base_semantic", PREV_LABEL, TUNED])
comparison = te.compare_reports(reports, baseline="base_semantic", candidate=TUNED, also_vs=[PREV_LABEL])
json.dump(comparison, open(f"{REPORT_DIR}/comparison.json", "w"), indent=1)
print(te.format_table(comparison))
print(f"PRIMARY ({comparison['primary_name']}):", comparison["primary"])
print(f"{TUNED} - {PREV_LABEL} | focus AP:", comparison["paired"].get(f"{TUNED} - {PREV_LABEL} | focus AP"))
for lb in (PREV_LABEL, TUNED):
    print(lb, json.dumps(eval_v3.summary(reports[lb])["focus_per_subject"], indent=1))
"""
    push = """
# 7. Push model + reports (private repo, tagged RUN_NAME). Pin the printed commit in the embedding contract.
sha = te.push(MODEL_DIR, MODEL_REPO, HF_TOKEN, reports, comparison, run_name=RUN_NAME, data_revision=DATA_REVISION,
              meta_extra={"previous_model": {"repo": MODEL_REPO, "revision": PREV_REVISION, "label": PREV_LABEL}})
print("pushed", MODEL_REPO, "tag", RUN_NAME, "commit", sha)
"""
    intro = FINAL_INTRO.format(data_revision=data_revision, previous_revision=previous_revision,
                               previous_label=previous_label)
    # v3 cells: 0 intro, 1 install, 2 setup, 3 base original, 4 base semantic, 5 mining, 6 train, 7 tuned eval,
    # 8 compare, 9 push
    return [md(intro), CELLS[1], code(setup), CELLS[3], CELLS[4], code(previous), CELLS[5], CELLS[6], CELLS[7],
            code(compare), code(push)]


def main() -> None:
    """Write the v3 notebook, or with ``--final`` the final-run notebook."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true", help="final-run notebook (v3-final data, previous-model cell)")
    parser.add_argument("--data-revision", default="v3-final")
    parser.add_argument("--run-name", default="v3-final-run1")
    parser.add_argument("--previous-revision", default="v3-run1")
    parser.add_argument("--previous-label", default="v3_run1")
    parser.add_argument("--out", default=None, help="default colab/train_genizah_embedder_v3[final].ipynb")
    args = parser.parse_args()
    cells = final_cells(args.data_revision, args.run_name, args.previous_revision, args.previous_label) \
        if args.final else CELLS
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = Path(args.out) if args.out else HERE / ("train_genizah_embedder_v3final.ipynb" if args.final
                                                  else "train_genizah_embedder_v3.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print(out)


if __name__ == "__main__":
    main()
