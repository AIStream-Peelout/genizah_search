"""Fine-tune Qwen3-Embedding-0.6B on the semantic-text mixture (text track step 4 of 4; v3 recipe and the v10 arms).

Runs on a Colab A100 (``train_genizah_embedder_v4final.ipynb``; older runs: ``..._v3[final].ipynb``); importable
locally for the CPU smoke test (``smoke_train_v3.py``) and the unit tests (``python train_embedder_v3.py --self-test``,
torch + a tiny random model; the torch-free half is :func:`self_test_pure`).

Stages (one function each, so the notebook runs them cell by cell):

1. :func:`load_data`      -- the private dataset repo at a tag (``package_dataset_v3.py``): train mix, train pool
   (the gated, shelf-mark-masked train records -- the ONLY source of mined negatives), the embedded corpus text
   (``corpus_semantic.jsonl``: semantic_v4 for v4 packages), the v8 production text (``corpus_original.jsonl``), the
   optional v9 text (``corpus_semantic_v3.jsonl`` or the rows' ``text_v3``) and no-frames ablation text
   (``corpus_semantic_nf.jsonl``), the frozen eval dir (with ``gate_v1/`` for v4), ``eval_v3.py`` and ``eval_gate.py``.
   Eval-only packages (``v4-ablation``) have no train files.
2. :func:`evaluate`       -- ``eval_v3.evaluate`` of one model on one text over the whole eligible pool; when the eval
   dir holds ``gate_v1/`` also ``eval_gate.evaluate_gate`` (``res["gate"]``); optionally the document and query
   vectors go to ``vectors_dir`` (Drive) as ``vectors_<label>.f16.npy``. Reference models of the v10 gate (plan E):
   **P** = stock model on the v8 text, **B3** = stock on the v9 text (semantic_v3), **B** = stock on the v4 text, **V**
   = v9 on semantic_v3 with the ids v9 trained on ignored (:func:`load_v9_trained`).
3. :func:`mine_negatives` -- one hard negative per q2d / d2d row from the base model's ranking of the train pool at the
   FULL text window (:data:`EVAL_MAX_SEQ`). Policy ``v4`` (plan C5): the same-subject preference only for
   ``synthetic_specific`` rows; every doc-level row also rejects candidates carrying one of the positive's festival
   topics and candidates whose label text is a near-duplicate (token Jaccard >= :data:`NEAR_DUP_JACCARD`) of the
   positive's; rejections are logged per reason, family and subject. Policy ``v3`` reproduces the v9 mining.
4. :func:`train`          -- ``arm="lora"`` (r16 / alpha 32 / dropout 0.05 on q/k/v/o/gate/up/down, lr 1e-4, merged
   into the fp32 weights before saving after a merged-vs-adapter cosine check) or ``arm="full"`` (lr 1e-5); warmup
   10 %, cosine, <= 1 epoch, checkpoints every 25 % of the steps (``save_only_model``), seed 13, clipping 1.0, training
   window :data:`TRAIN_MAX_SEQ` (1024) with a per-mini-batch padding trim; ``arm=None`` = the v3 recipe. fp32 master
   weights + bf16 autocast, gradient checkpointing, GradCache (:class:`MaskedCachedMNRL`): in-batch candidates whose ids
   overlap the row's ids get logit -inf; the loss is logged per dataset and per family. The saved model carries the
   serving contract: max_seq_length 8192, no tokenizer truncation / padding baked into ``tokenizer.json``, prompts
   ``{"query": QUERY_PROMPT, "document": ""}``; windows recorded in ``training_info.json`` and the model card.
5. :func:`evaluate_checkpoints` + :func:`compare_reports` (side by side, paired clustered bootstrap CIs incl. the
   holiday section clustered by festival, and the ``eval_gate.verdict`` when ``refs`` are given) +
   :func:`select_checkpoint` (plan E pre-registration: among checkpoints passing every hard criterion the highest mean
   H1 delta vs B; within +-:data:`SELECTION_TOLERANCE` the earlier checkpoint, then arm lora; none passing -> no push).
6. :func:`push`           -- model + reports to a private HF model repo (the owner runs it); ``require_pass`` refuses
   a model whose gate verdict does not pass.

Masking contract (``build_training_mix_v3.py``). Each row is turned into three id segments, hashed to int64 and
carried in a numeric ``label`` column of width ``3 * K + 1`` (segments padded with -1, then the row's family code
for the per-family loss log):

* ``seg0`` = relevance of the ANCHOR side: what counts as "the same thing" for this anchor.
  doc-level q2d: ``D:<doc>`` + ``T:<text_hash>`` of the positive (only the record itself / identical text) + the
  row's ``anchor_topic_ids`` (festivals the query names, ``topic:<f>`` and ``M:topic:<f>``, so another record of
  that festival -- labelled or merely naming it -- is never its negative);
  subject-level q2d: the positive's (non-broad) subject ids + ``anchor_subject`` + ``anchor_mention_id``
  (``M:<anchor_subject>``: records whose catalogue text names the subject without carrying it) +
  ``anchor_topic_ids`` + ``D:`` + ``T:``;
  d2d: the anchor record's subject ids + its ``D:`` / ``T:``; t2t: the pair's Sefaria ``concept_ids`` + ``T:`` of the
  anchor string. Broad ids (:data:`BROAD_PREFIXES`: the KTIV ``domain:`` genres and the ``M:`` mention ids) are
  dropped from relevance unless the row names them (``anchor_subject`` / ``anchor_mention_id`` /
  ``anchor_topic_ids``): sharing a genre or a passing mention does not make two records the same subject.
* ``seg1`` = identity of the POSITIVE (all subject ids incl. broad and ``M:`` ones + ``D:`` + ``T:``; t2t: concepts +
  ``T:``).
* ``seg2`` = identity of the mined hard NEGATIVE (empty when the row has none).

query_to_doc: candidate column j (positive j, or negative j) is masked for row i when ``seg0_i`` shares an id with
``seg1_j`` (``seg2_j``). doc_to_query (symmetric kinds t2t and d2d): row i's positive against anchor j is masked when
``seg1_i`` shares an id with ``seg0_j``. The diagonal (the row's own positive) is never masked. Batches never mix
kinds (one dataset per kind and negative layout), so no cross-kind rule is needed.

The embedding contract is unchanged from production: 1024-d, last-token pooling, L2-normalised, queries prefixed
with :data:`QUERY_PROMPT`, documents raw.
"""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch touches the GPU

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BASE_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
V9_MODEL = "isaacmg/genizah-embed-qwen3-0.6b"
V9_REVISION = "dbfab73f1e91a6b95eba728541cc5a391a88d933"  # tag v3-final-run1 (the v9 index embedder)
QUERY_PROMPT = "Instruct: Given a search query, retrieve relevant passages\nQuery: "
SAVED_PROMPTS = {"query": QUERY_PROMPT, "document": ""}  # written into the saved model (plan B5)
KIND_PROMPTS = {"q2d": {"anchor": QUERY_PROMPT},
                "t2t": {"anchor": QUERY_PROMPT, "positive": QUERY_PROMPT},
                "d2d": {}}  # d2d: both sides are records -> documents raw
SYMMETRIC_KINDS = ("t2t", "d2d")
BROAD_PREFIXES = ("domain:", "M:")  # KTIV genres + catalogue-text mentions: never relevance unless the row names them
N_SEGMENTS = 3
PAD_ID = -1
EVAL_MAX_SEQ = 8192  # production window (embedding_service MAX_SEQ_LENGTH); also saved with the tuned model
TRAIN_MAX_SEQ = 1024  # v10 (plan B5): truncated positives 19 % -> 0.44 %; v3-final trained at V3_TRAIN_MAX_SEQ
V3_TRAIN_MAX_SEQ = 384
# the hub checkpoint's input path (legacy ST config -> plain text, no chat template); see tiny_model / check_text_path
TEXT_MODALITY = {"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
DATA_FILES = ("train_mix_v3.jsonl", "train_pool.jsonl", "corpus_semantic.jsonl", "corpus_original.jsonl")
TRAIN_FILES = ("train_mix_v3.jsonl", "train_pool.jsonl")  # absent from eval-only packages (v4-ablation)
# optional text variants (dataset root or ablation/): v9 text for B3 / V, no-frames v4 text for the H0 ablation
OPTIONAL_TEXT_FILES = {"semantic_v3": "corpus_semantic_v3.jsonl", "semantic_nf": "corpus_semantic_nf.jsonl"}
# eval-pool field holding the md5 of each text variant (eval_v3 ``text_hash_key``); other texts are not hash-checked
TEXT_HASH_KEYS = {"semantic": "text_hash", "semantic_v3": "text_hash_v3"}
V9_TRAINED_FILES = ("v9_trained.json", "eval/v9_trained.json", "eval/gate_v1/v9_trained.json")
GATE_SUBDIR = "gate_v1"
# festival topics (build_subjects.FESTIVAL_TOPICS; hard-coded because the trainer cannot import build_subjects on
# Colab -- self_test_pure compares the two when build_subjects is importable)
FESTIVAL_TOPICS = ("pesach", "sukkot", "yom_kippur", "rosh_hashanah", "purim", "hanukkah", "shavuot", "simhat_torah",
                   "fast_days", "rosh_hodesh")
SAME_SUBJECT_FAMILIES = ("synthetic_specific",)  # plan C5: the only family that prefers same-subject negatives
NEAR_DUP_JACCARD = 0.6  # plan C5: label-text token Jaccard at or above which a doc-level candidate is no negative
NEGATIVE_POLICIES = ("v4", "v3")
LABEL_TOKEN_RE = re.compile(r"[\w֐-׿]+")  # the training-config audit's tokeniser (audit_recipe/q4.py)
LABEL_STOPWORDS = frozenset("the of and in a an to for on from with by about at as or is are was were this that "
                            "these those its his her".split())
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RECIPES = {"lora": {"lr": 1e-4, "warmup_ratio": 0.10, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
           "full": {"lr": 1e-5, "warmup_ratio": 0.10}}
V3_RECIPE = {"lr": 2e-5, "warmup_ratio": 0.05}  # arm=None: the v3 / v3-final full fine-tune
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
MERGE_MIN_COSINE = 0.9999
MERGE_CHECK_TEXTS = 100
SELECTION_TOLERANCE = 0.005
DEFERRED_CRITERIA = ("H0", "H10", "H12")  # per-checkpoint ranking: H0 (session 1), H10 (bib check), H12 (LCO run)
REF_ROLES = ("P", "B", "B3", "V")
REF_TEXT = {"P": "original", "B": "semantic", "B3": "semantic_v3", "V": "semantic_v3", "Bf": "semantic",
            "Bnf": "semantic_nf"}
H12_AP_FLOOR = 0.182
MODEL_CARD_BEGIN = "<!-- genizah-training -->"
MODEL_CARD_END = "<!-- /genizah-training -->"


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
def _read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file.

    :param path: File path.
    :returns: Rows.
    :rtype: List[dict]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _find_file(root: Path, name: str, subdirs: Sequence[str] = ("", "ablation")) -> Optional[Path]:
    """First existing ``root/<subdir>/<name>``.

    :param root: Dataset folder.
    :param name: File name.
    :param subdirs: Sub-folders to try, in order.
    :returns: Path or None.
    :rtype: Optional[Path]
    """
    for sub in subdirs:
        p = root / sub / name
        if p.exists():
            return p
    return None


def load_texts(path: Path, rows: Dict[str, dict]) -> Dict[str, Dict[str, str]]:
    """The optional text variants of a dataset folder (:data:`OPTIONAL_TEXT_FILES`).

    ``semantic_v3`` falls back to the ``text_v3`` field of ``corpus_semantic.jsonl`` rows (semantic_v4 packages carry
    the v9 text there) when no ``corpus_semantic_v3.jsonl`` was shipped.

    :param path: Dataset folder.
    :param rows: ``corpus_semantic.jsonl`` rows by doc id.
    :returns: ``name -> {doc_id: text}``.
    :rtype: Dict[str, Dict[str, str]]
    """
    out: Dict[str, Dict[str, str]] = {}
    for name, fname in OPTIONAL_TEXT_FILES.items():
        f = _find_file(path, fname)
        if f is not None:
            out[name] = {r["doc_id"]: r["text"] for r in _read_jsonl(f)}
    if "semantic_v3" not in out and rows and all("text_v3" in r for r in rows.values()):
        out["semantic_v3"] = {d: r["text_v3"] for d, r in rows.items()}
    return out


def load_data(repo_id: str, token: Optional[str] = None, local_dir: str = "/content/genizah_embed_data_v3",
              revision: Optional[str] = None) -> Dict[str, object]:
    """Download (or open) the dataset folder and parse it.

    The folder also holds ``eval_v3.py`` (and ``eval_gate.py`` for v4 packages); it is put on ``sys.path`` so
    :func:`evaluate` scores with the copies that were packaged with the data.

    :param repo_id: Private HF dataset repo id, or a local directory (smoke tests).
    :param token: HF token (Colab secret ``HF_TOKEN``).
    :param local_dir: Download directory.
    :param revision: Dataset tag/commit to pin (``v3``, ``v4-final``); default = latest.
    :returns: Dict with ``train`` (mixture rows; empty for eval-only packages), ``pool`` (doc_id -> {text, text_hash,
        subjects}), ``semantic`` (doc_id -> corpus_semantic row), ``original`` (doc_id -> v8 production text),
        ``texts`` (optional variants, :func:`load_texts`), ``eval_pool`` (frozen pool rows), ``held_out`` (ids),
        ``eval_dir``, ``path``, ``manifest``, ``eval_only``.
    :rtype: Dict[str, object]
    """
    if Path(repo_id).is_dir():
        path = Path(repo_id)
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", token=token, local_dir=local_dir,
                                      revision=revision))
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    eval_dir = path / "eval"
    eval_only = not all((path / f).exists() for f in TRAIN_FILES)
    semantic = {r["doc_id"]: r for r in _read_jsonl(path / "corpus_semantic.jsonl")}
    data = {
        "train": [] if eval_only else _read_jsonl(path / "train_mix_v3.jsonl"),
        "pool": {} if eval_only else {r["doc_id"]: r for r in _read_jsonl(path / "train_pool.jsonl")},
        "semantic": semantic,
        "original": {r["doc_id"]: r["text"] for r in _read_jsonl(path / "corpus_original.jsonl")},
        "texts": load_texts(path, semantic),
        "eval_pool": _read_jsonl(eval_dir / "eval_pool.jsonl"),
        "held_out": set(json.loads((eval_dir / "held_out_records.json").read_text())["held_out"]),
        "eval_dir": str(eval_dir),
        "path": path,
        "manifest": json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else {},
        "eval_only": eval_only,
    }
    check_data(data)
    return data


def check_data(data: dict) -> None:
    """Cheap hold-out guards on the loaded data (the packager verified the same; this catches a wrong revision).

    :param data: Output of :func:`load_data`.
    :raises ValueError: When a train row or pool record touches the held-out side.
    """
    held, pool = data["held_out"], data["pool"]
    bad_pool = held & set(pool)
    if bad_pool:
        raise ValueError(f"{len(bad_pool)} train-pool records are held out, e.g. {sorted(bad_pool)[:3]}")
    for r in data["train"]:
        for key in ("doc_id", "anchor_doc_id"):
            d = r.get(key)
            if d is not None and (d in held or d not in pool):
                raise ValueError(f"train row {key}={d} is held out or missing from the train pool")


def _ids_from_json(obj) -> List[str]:
    """Doc ids of a ``v9_trained`` JSON value: a list, or a dict with ``doc_ids`` / ``v9_trained`` / ``ids``.

    :param obj: Parsed JSON.
    :returns: Sorted unique ids.
    :rtype: List[str]
    :raises ValueError: For any other shape.
    """
    if isinstance(obj, dict):
        for key in ("doc_ids", "v9_trained", "ids"):
            if isinstance(obj.get(key), list):
                return sorted(set(obj[key]))
        raise ValueError(f"v9_trained JSON has none of doc_ids / v9_trained / ids: keys {sorted(obj)[:10]}")
    if isinstance(obj, list):
        return sorted(set(obj))
    raise ValueError(f"v9_trained JSON must be a list or a dict, not {type(obj).__name__}")


def load_v9_trained(data: dict) -> Optional[List[str]]:
    """Ids v9 trained on (anchors, positives, negatives of ``v3/train_mix_v3final.jsonl``), listed in the dataset.

    Passed as ``ignore_ids`` when scoring the reference V (v9): text regrouping moves records v9 trained on into the
    v4 held-out side (plan E / H9). Looked up at :data:`V9_TRAINED_FILES`.

    :param data: Output of :func:`load_data`.
    :returns: Sorted ids, or None when the package lists none (see :func:`v9_trained_from_mix`).
    :rtype: Optional[List[str]]
    """
    for rel in V9_TRAINED_FILES:
        p = Path(data["path"]) / rel
        if p.exists():
            return _ids_from_json(json.loads(p.read_text(encoding="utf-8")))
    return None


def v9_trained_from_mix(mix_path: str) -> List[str]:
    """Fallback for :func:`load_v9_trained`: anchor and positive record ids of the v3-final mixture.

    Mined negatives were never written to a file, so they are missing here (they came from the same train pool).

    :param mix_path: ``train_mix_v3.jsonl`` of the ``v3-final`` dataset tag.
    :returns: Sorted ids.
    :rtype: List[str]
    """
    ids: Set[str] = set()
    for r in _read_jsonl(Path(mix_path)):
        ids.update(d for d in (r.get("doc_id"), r.get("anchor_doc_id")) if d)
    return sorted(ids)


def gate_dir(data: dict) -> Optional[Path]:
    """The eval dir's ``gate_v1/`` folder (written by ``build_gate_v1.py``), or None (v3 eval dirs).

    :param data: Output of :func:`load_data`.
    :returns: Path or None.
    :rtype: Optional[Path]
    """
    p = Path(data["eval_dir"]) / GATE_SUBDIR
    return p if p.is_dir() else None


def gate_thresholds(data: dict) -> Tuple[dict, str]:
    """The frozen gate thresholds and the sha256 of their file (printed before training: pre-registration).

    :param data: Output of :func:`load_data`.
    :returns: (thresholds dict, sha256 hex of ``thresholds.json``).
    :rtype: Tuple[dict, str]
    :raises FileNotFoundError: When the eval dir has no ``gate_v1/thresholds.json``.
    """
    gd = gate_dir(data)
    if gd is None or not (gd / "thresholds.json").exists():
        raise FileNotFoundError(f"no {GATE_SUBDIR}/thresholds.json under {data['eval_dir']}")
    raw = (gd / "thresholds.json").read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def is_broad(sid: str) -> bool:
    """Whether a subject id is broad (a KTIV ``domain:`` genre or an ``M:`` mention id; dropped from relevance).

    :param sid: Subject id.
    :returns: True for :data:`BROAD_PREFIXES` ids.
    :rtype: bool
    """
    return sid.startswith(BROAD_PREFIXES)


def text_token(text: str) -> str:
    """``T:<md5>`` id of a string (identical text is never a negative).

    :param text: Text.
    :returns: Id string.
    :rtype: str
    """
    return "T:" + hashlib.md5(text.encode("utf-8")).hexdigest()


def record_identity(doc_id: str, subjects: Iterable[str], text_hash: str) -> List[str]:
    """Identity ids of a record as a candidate: every subject id (incl. ``M:`` mentions), ``D:<doc_id>``, ``T:<hash>``.

    :param doc_id: Record id.
    :param subjects: Its subject ids (strong, weak and ``M:`` mention ids; ``D:`` ids are ignored).
    :param text_hash: md5 of its embedded text.
    :returns: Sorted unique ids.
    :rtype: List[str]
    """
    ids = {s for s in subjects if not s.startswith("D:")}
    return sorted(ids | {f"D:{doc_id}", f"T:{text_hash}"})


def relevance_of(identity: Iterable[str], keep: Iterable[Optional[str]] = ()) -> List[str]:
    """Relevance ids from identity ids: broad ids (genres, mentions) dropped, ``keep`` ids always added.

    :param identity: Identity ids.
    :param keep: Ids kept even when broad (the anchor's own subject, its mention id, the festivals it names).
    :returns: Sorted unique ids.
    :rtype: List[str]
    """
    keep_set = {k for k in keep if k}
    return sorted({s for s in identity if not is_broad(s)} | keep_set)


def row_segments(row: dict, pool: Dict[str, dict]) -> Tuple[List[str], List[str], List[str]]:
    """The three id segments of a training row (see the module docstring).

    :param row: Mixture row (optionally with ``negative_doc_id`` from :func:`mine_negatives`).
    :param pool: Train pool ``doc_id -> {text, text_hash, subjects}``.
    :returns: (seg0 anchor relevance, seg1 positive identity, seg2 negative identity).
    :rtype: Tuple[List[str], List[str], List[str]]
    """
    if row["kind"] == "t2t":
        concepts = sorted(set(row.get("concept_ids") or []))
        return concepts + [text_token(row["anchor"])], concepts + [text_token(row["positive"])], []
    pos = record_identity(row["doc_id"], row["mask_ids"], row["text_hash"])
    topics = sorted(row.get("anchor_topic_ids") or [])
    if row["kind"] == "d2d":
        a = row["anchor_doc_id"]
        seg0 = relevance_of(record_identity(a, row["anchor_mask_ids"], pool[a]["text_hash"]))
    elif row["mask_level"] == "doc":
        # + the festivals the query names: another Passover record is no negative for "Passover Haggadah"
        seg0 = [f"D:{row['doc_id']}", f"T:{row['text_hash']}"] + topics
    else:
        seg0 = relevance_of(pos, keep=[row.get("anchor_subject"), row.get("anchor_mention_id")] + topics)
    neg = row.get("negative_doc_id")
    seg2 = record_identity(neg, pool[neg]["subjects"], pool[neg]["text_hash"]) if neg else []
    return seg0, pos, seg2


def id_int(token: str) -> int:
    """Stable non-negative 60-bit integer for an id string.

    :param token: Id string.
    :returns: Integer id.
    :rtype: int
    """
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:15], 16)


def encode_label(segments: Sequence[Sequence[str]], width: int) -> List[int]:
    """Flatten id segments into the fixed-width numeric segment part of the ``label`` value.

    :param segments: :data:`N_SEGMENTS` lists of id strings.
    :param width: Ids per segment (K); shorter segments are padded with :data:`PAD_ID`.
    :returns: ``N_SEGMENTS * width`` ints.
    :rtype: List[int]
    """
    out: List[int] = []
    for seg in segments:
        ids = sorted({id_int(s) for s in seg})
        if len(ids) > width:
            raise ValueError(f"segment of {len(ids)} ids exceeds the label width {width}")
        out += ids + [PAD_ID] * (width - len(ids))
    return out


def label_rows(segments: Sequence[Sequence[Sequence[str]]], width: int, family_codes: Sequence[int]) -> List[List[int]]:
    """``label`` values of a dataset: the encoded segments of each row plus its family code as the last column.

    :param segments: Per row, the :data:`N_SEGMENTS` id lists (:func:`row_segments`).
    :param width: Ids per segment (K).
    :param family_codes: Per row, the index of its family in the dataset stats' ``families`` list.
    :returns: Rows of ``N_SEGMENTS * width + 1`` ints.
    :rtype: List[List[int]]
    """
    return [encode_label(s, width) + [int(c)] for s, c in zip(segments, family_codes)]


def label_part(text: str) -> str:
    """A record text without its ``Transcription:`` / ``Translation:`` lines (the content family's positive form).

    :param text: Record text.
    :returns: The remaining lines.
    :rtype: str
    """
    return "\n".join(ln for ln in text.split("\n") if not ln.startswith(("Transcription:", "Translation:")))


def label_tokens(text: str) -> frozenset:
    """Token set of a record's label text (:func:`label_part`) for the near-duplicate guard.

    Same tokeniser as the training-config audit that measured the near-duplicate rates (lower-cased word / Hebrew
    runs, a small English stop list, tokens of 2+ characters), so :data:`NEAR_DUP_JACCARD` means what it measured.

    :param text: Record text.
    :returns: Tokens.
    :rtype: frozenset
    """
    return frozenset(t for t in (x.lower() for x in LABEL_TOKEN_RE.findall(label_part(text)))
                     if len(t) > 1 and t not in LABEL_STOPWORDS)


def token_jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two token sets (0 when either is empty).

    :param a: Tokens.
    :param b: Tokens.
    :returns: |a & b| / |a | b|.
    :rtype: float
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def festival_names(subjects: Iterable[str]) -> Set[str]:
    """Festivals a record carries: ``topic:<f>`` (strong or weak) or ``M:topic:<f>`` (its catalogue names it).

    :param subjects: Subject ids of the record.
    :returns: Festival names from :data:`FESTIVAL_TOPICS`.
    :rtype: Set[str]
    """
    s = set(subjects)
    return {f for f in FESTIVAL_TOPICS if f"topic:{f}" in s or f"M:topic:{f}" in s}


def dataset_name(row: dict) -> str:
    """Dataset a row trains in: its kind, ``_hn`` suffix when it carries a mined negative.

    :param row: Training row.
    :returns: ``q2d``, ``q2d_hn``, ``d2d``, ``d2d_hn`` or ``t2t``.
    :rtype: str
    """
    return row["kind"] + ("_hn" if row.get("negative") else "")


def kind_of(name: str) -> str:
    """Kind of a dataset name.

    :param name: Output of :func:`dataset_name`.
    :returns: ``q2d``, ``t2t`` or ``d2d``.
    :rtype: str
    """
    return name.split("_")[0]


def build_datasets(pairs: List[dict], pool: Dict[str, dict]) -> Tuple[dict, Dict[str, dict], dict]:
    """One ``datasets.Dataset`` per kind and negative layout, the nested prompts, and size stats.

    Columns: ``anchor``, ``positive`` (+ ``negative``), ``label`` (fixed-width int ids + family code, see the module
    docstring).

    :param pairs: Training rows (output of :func:`mine_negatives`, or raw mixture rows).
    :param pool: Train pool.
    :returns: (``name -> Dataset``, ``name -> {column: prompt}``, stats incl. ``families`` = the family code table).
    :rtype: Tuple[dict, Dict[str, dict], dict]
    """
    from datasets import Dataset

    groups: Dict[str, List[dict]] = defaultdict(list)
    segs: Dict[int, tuple] = {}
    for p in pairs:
        groups[dataset_name(p)].append(p)
        segs[id(p)] = row_segments(p, pool)
    width = max(1, max(len(set(s)) for v in segs.values() for s in v))
    families = sorted({p["family"] for p in pairs})
    code = {f: i for i, f in enumerate(families)}
    datasets, prompts = {}, {}
    for name in sorted(groups):
        rows = groups[name]
        cols = {"anchor": [r["anchor"] for r in rows], "positive": [r["positive"] for r in rows]}
        if name.endswith("_hn"):
            cols["negative"] = [r["negative"] for r in rows]
        cols["label"] = label_rows([segs[id(r)] for r in rows], width, [code[r["family"]] for r in rows])
        datasets[name] = Dataset.from_dict(cols)
        prompts[name] = dict(KIND_PROMPTS[kind_of(name)])
    if set(prompts) != set(datasets):
        raise AssertionError(f"prompt keys {sorted(prompts)} != dataset keys {sorted(datasets)}")
    for name, ds in datasets.items():
        if not set(prompts[name]) <= set(ds.column_names):
            raise AssertionError(f"{name}: prompt columns {sorted(prompts[name])} not in {ds.column_names}")
    stats = {"label_width": width, "families": families, "rows": {k: len(v) for k, v in datasets.items()},
             "by_family": {k: dict(Counter(r["family"] for r in v)) for k, v in sorted(groups.items())}}
    return datasets, prompts, stats


# ---------------------------------------------------------------------------------------------------------------
# masked GradCache loss
# ---------------------------------------------------------------------------------------------------------------
def id_overlap(a: Tensor, b: Tensor) -> Tensor:
    """Which rows of two padded id matrices share an id.

    :param a: (n, K) int64 ids, padded with :data:`PAD_ID`.
    :param b: (m, K2) int64 ids, padded with :data:`PAD_ID`.
    :returns: (n, m) bool; True where row i of ``a`` and row j of ``b`` share a non-pad id.
    :rtype: Tensor
    """
    n, m = a.shape[0], b.shape[0]
    flat = torch.cat([a.reshape(-1), b.reshape(-1)])
    uniq, inv = torch.unique(flat, return_inverse=True)
    inv_a, inv_b = inv[: a.numel()].view(a.shape), inv[a.numel():].view(b.shape)
    inc_a = torch.zeros(n, len(uniq), device=a.device, dtype=torch.float32)
    inc_b = torch.zeros(m, len(uniq), device=b.device, dtype=torch.float32)
    va, vb = a >= 0, b >= 0
    inc_a[torch.arange(n, device=a.device).unsqueeze(1).expand_as(a)[va], inv_a[va]] = 1.0
    inc_b[torch.arange(m, device=b.device).unsqueeze(1).expand_as(b)[vb], inv_b[vb]] = 1.0
    with torch.autocast(device_type=a.device.type, enabled=False):
        return (inc_a @ inc_b.T) > 0.5


def split_label(labels: Tensor) -> List[Tensor]:
    """Split the ``label`` tensor into its :data:`N_SEGMENTS` (B, K) id segments (a trailing family column is skipped).

    :param labels: (B, N_SEGMENTS * K) or (B, N_SEGMENTS * K + 1) int64.
    :returns: List of (B, K) tensors.
    :rtype: List[Tensor]
    """
    width = (labels.shape[1] - labels.shape[1] % N_SEGMENTS) // N_SEGMENTS
    return [labels[:, i * width:(i + 1) * width] for i in range(N_SEGMENTS)]


def label_family(labels: Tensor) -> Optional[Tensor]:
    """The family-code column of a ``label`` tensor (:func:`label_rows`), or None when it has none.

    :param labels: (B, N_SEGMENTS * K [+ 1]) int64.
    :returns: (B,) codes or None.
    :rtype: Optional[Tensor]
    """
    return labels[:, -1] if labels.shape[1] % N_SEGMENTS == 1 else None


def trim_minibatch(features: dict, begin: int, end: int) -> Tuple[dict, int, int]:
    """Slice rows ``begin:end`` out of a column's padded features and drop the token columns that are padding for
    every row of the slice.

    Verified in sentence-transformers 5.6.1 (``CachedMultipleNegativesRankingLoss`` ``_create_minibatch``): GradCache
    mini-batches are plain batch-dimension slices, so every mini-batch keeps the column's padding to the longest text
    of the whole (256-row) batch (the collator tokenises each column once, ``padding=True``). Works for left or right
    padding (the kept columns are the contiguous span covering every real token). Last-token pooling reads the
    attention mask, so the pooled token is unchanged; with left padding the real tokens' absolute positions shift by
    the trimmed amount, which RoPE attention does not see (scores depend on relative positions; asserted numerically
    in :func:`self_test`). Accepts torch tensors or numpy arrays (unit tests).

    :param features: Column features (``input_ids``, ``attention_mask``, non-tensor entries such as ``modality``).
    :param begin: First row.
    :param end: End row (clamped to the batch size).
    :returns: (mini-batch features, padded token slots before the trim, after the trim).
    :rtype: Tuple[dict, int, int]
    """
    mask = features["attention_mask"]
    size, seq_len = mask.shape[0], mask.shape[1]
    end = min(end, size)
    keep = (mask[begin:end] > 0).any(0).tolist()
    cols = [i for i, k in enumerate(keep) if k]
    lo, hi = (cols[0], cols[-1] + 1) if cols else (0, seq_len)
    out = {}
    for key, value in features.items():
        if hasattr(value, "shape") and len(value.shape) >= 1 and value.shape[0] == size:
            value = value[begin:end]
            if len(value.shape) >= 2 and value.shape[1] == seq_len:
                value = value[:, lo:hi]
        out[key] = value
    return out, (end - begin) * seq_len, (end - begin) * (hi - lo)


class MaskedCachedMNRL(CachedMultipleNegativesRankingLoss):
    """GradCache InfoNCE whose in-batch candidates sharing ids with the row get logit -inf.

    The ids arrive as the numeric ``label`` column (see the module docstring). Supported: one optional hard negative
    per row, ``query_to_doc`` (+ ``doc_to_query`` when symmetric, one softmax per direction), no hardness weighting,
    single device. ``stats`` counts batches, masked off-diagonal candidates and padded vs trimmed token slots;
    ``history`` holds one entry per batch (global batch number from the shared ``clock``, loss, per-family loss sums)
    for the per-dataset / per-family loss log.
    """

    def __init__(self, model, symmetric: bool = False, family_names: Optional[Sequence[str]] = None,
                 clock: Optional[Counter] = None, trim_padding: bool = True, **kwargs) -> None:
        """Build the loss.

        :param model: SentenceTransformer being trained.
        :param symmetric: Add the ``doc_to_query`` direction (t2t, d2d) with a per-direction softmax.
        :param family_names: Family code table (``build_datasets`` stats ``families``) for the per-family loss log.
        :param clock: Counter shared by every loss of a run (``clock["batches"]`` = global batch number).
        :param trim_padding: Trim each GradCache mini-batch to its own longest text (:func:`trim_minibatch`).
        :param kwargs: Forwarded to :class:`CachedMultipleNegativesRankingLoss` (``scale``, ``mini_batch_size`` ...).
        """
        directions = ("query_to_doc", "doc_to_query") if symmetric else ("query_to_doc",)
        super().__init__(model, directions=directions, partition_mode="per_direction" if symmetric else "joint",
                         **kwargs)
        if self.gather_across_devices or self.hardness_mode is not None:
            raise ValueError("MaskedCachedMNRL supports a single device and no hardness weighting")
        self.labels: Optional[Tensor] = None
        self.stats: Counter = Counter()
        self.family_names = list(family_names or [])
        self.clock = clock if clock is not None else Counter()
        self.trim_padding = trim_padding
        self.history: List[dict] = []

    def forward(self, sentence_features: Iterable[dict], labels: Optional[Tensor]) -> Tensor:
        """Keep the batch's id labels for :meth:`calculate_loss`, then run GradCache as usual.

        :param sentence_features: Tokenised columns (anchor, positive[, negative]).
        :param labels: (B, N_SEGMENTS * K + 1) id tensor from the ``label`` column.
        :returns: Loss.
        :rtype: Tensor
        """
        self.labels = labels
        return super().forward(sentence_features, labels)

    def embed_minibatch(self, sentence_feature: dict, begin: int, end: int, with_grad: bool, copy_random_state: bool,
                        random_state=None):
        """Embed one mini-batch after trimming its padding (both GradCache passes go through here).

        :param sentence_feature: Column features of the whole batch.
        :param begin: First row.
        :param end: End row.
        :param with_grad: Second (gradient) pass.
        :param copy_random_state: Record the RNG state (first pass).
        :param random_state: RNG state to replay (second pass).
        :returns: (embeddings, random state) as the parent.
        """
        if self.trim_padding and "attention_mask" in sentence_feature and "cu_seq_lens_q" not in sentence_feature:
            sentence_feature, before, after = trim_minibatch(sentence_feature, begin, end)
            if not with_grad:
                self.stats["tokens_padded"] += before
                self.stats["tokens_after_trim"] += after
            begin, end = 0, end - begin
        return super().embed_minibatch(sentence_feature=sentence_feature, begin=begin, end=end, with_grad=with_grad,
                                       copy_random_state=copy_random_state, random_state=random_state)

    def candidate_masks(self, n_docs: int, device: torch.device) -> Tuple[Tensor, Optional[Tensor]]:
        """Masks of the current batch: (B, B * n_docs) for query_to_doc, (B, B) for doc_to_query (or None).

        :param n_docs: 1 (positives) or 2 (positives + hard negatives).
        :param device: Device of the embeddings.
        :returns: (query_to_doc mask, doc_to_query mask or None); True = logit set to -inf.
        :rtype: Tuple[Tensor, Optional[Tensor]]
        """
        if n_docs > 2:
            raise ValueError("MaskedCachedMNRL supports at most one hard negative per row")
        seg0, seg1, seg2 = split_label(self.labels.to(device))
        size = seg0.shape[0]
        eye = torch.arange(size, device=device)
        mask_qd = id_overlap(seg0, torch.cat([seg1, seg2][:n_docs]))
        mask_qd[eye, eye] = False
        mask_dq = None
        if "doc_to_query" in self.directions:
            mask_dq = id_overlap(seg1, seg0)
            mask_dq[eye, eye] = False
        return mask_qd, mask_dq

    def masked_logits(self, queries: Tensor, docs_all: Tensor, docs_pos: Tensor, idx: Tensor, mask_qd: Tensor,
                      mask_dq: Optional[Tensor]) -> Dict[str, Tensor]:
        """Scaled similarity matrices of one loss mini-batch with masked candidates at -inf.

        :param queries: (B, d) anchor embeddings.
        :param docs_all: (B * n_docs, d) positives then negatives.
        :param docs_pos: (B, d) positives.
        :param idx: Row indices of this mini-batch.
        :param mask_qd: Output of :meth:`candidate_masks`.
        :param mask_dq: Output of :meth:`candidate_masks` (None when not symmetric).
        :returns: ``{"query_to_doc": (b, B * n_docs)[, "doc_to_query": (b, B)]}``.
        :rtype: Dict[str, Tensor]
        """
        sims = {"query_to_doc": (self.similarity_fct(queries[idx], docs_all) * self.scale)
                .masked_fill(mask_qd[idx], float("-inf"))}
        if mask_dq is not None:
            sims["doc_to_query"] = (self.similarity_fct(queries, docs_pos[idx]).T * self.scale) \
                .masked_fill(mask_dq[idx], float("-inf"))
        return sims

    def record_batch(self, total: Tensor, per_row: Tensor) -> dict:
        """Append the batch's loss (and per-family loss sums, when the labels carry family codes) to ``history``.

        :param total: Batch loss (detached).
        :param per_row: (B,) per-row losses (detached).
        :returns: The history entry.
        :rtype: dict
        """
        self.clock["batches"] += 1
        entry = {"batch": self.clock["batches"], "loss": round(float(total), 5)}
        codes = label_family(self.labels) if self.labels is not None else None
        if codes is not None and self.family_names:
            codes = codes.to(per_row.device).long()
            n_fam = len(self.family_names)
            sums = torch.zeros(n_fam, device=per_row.device, dtype=torch.float32).index_add_(0, codes,
                                                                                            per_row.float())
            cnts = torch.bincount(codes, minlength=n_fam)
            entry["families"] = {self.family_names[i]: [round(s, 5), int(c)]
                                 for i, (s, c) in enumerate(zip(sums.tolist(), cnts.tolist())) if c}
        self.history.append(entry)
        return entry

    def calculate_loss(self, reps: List[List[Tensor]], with_backward: bool = False) -> Tensor:
        """InfoNCE over the batch with masked candidates (same mini-batch / backward scheme as the parent).

        :param reps: Embeddings per column, each a list of mini-batch tensors.
        :param with_backward: Backpropagate each mini-batch's loss to the cached embeddings.
        :returns: Loss.
        :rtype: Tensor
        """
        if self.labels is None:
            return super().calculate_loss(reps, with_backward)
        queries = torch.cat(reps[0])
        docs = [torch.cat(r) for r in reps[1:]]
        size = len(queries)
        docs_all, docs_pos = torch.cat(docs, dim=0), docs[0]
        mask_qd, mask_dq = self.candidate_masks(len(docs), queries.device)
        self.stats["batches"] += 1
        self.stats["rows"] += size
        self.stats["qd_masked"] += int(mask_qd.sum())
        self.stats["qd_candidates"] += mask_qd.numel() - size
        if mask_dq is not None:
            self.stats["dq_masked"] += int(mask_dq.sum())
            self.stats["dq_candidates"] += mask_dq.numel() - size
        losses, per_rows = [], []
        for begin in range(0, size, self.mini_batch_size):
            end = min(begin + self.mini_batch_size, size)
            idx = torch.arange(begin, end, device=queries.device)
            rows = torch.arange(end - begin, device=queries.device)
            sims = self.masked_logits(queries, docs_all, docs_pos, idx, mask_qd, mask_dq)
            positive = sims["query_to_doc"][rows, idx]
            if self.partition_mode == "joint":
                log_z = torch.logsumexp(torch.cat(list(sims.values()), dim=1), dim=1)
            else:
                log_z = sum(torch.logsumexp(s, dim=1) for s in sims.values()) / len(sims)
            per_row = -(positive - log_z)
            loss_mb = per_row.mean() * len(idx) / size
            per_rows.append(per_row.detach())
            if with_backward:
                loss_mb.backward()
                loss_mb = loss_mb.detach()
            losses.append(loss_mb)
        total = sum(losses)
        self.record_batch(total.detach(), torch.cat(per_rows))
        return total


def loss_breakdown(losses: Dict[str, object], total_steps: int) -> dict:
    """Per-dataset and per-family loss summaries from the losses' ``history`` (plan D: loss per dataset and family).

    :param losses: ``dataset name -> loss`` objects with a ``history`` list (:meth:`MaskedCachedMNRL.record_batch`).
    :param total_steps: Optimizer steps of the run (the quarter of a batch = its global batch number / steps).
    :returns: ``{by_dataset: {name: {batches, mean, by_quarter}}, by_family: {family: {rows, mean, by_quarter}}}``;
        quarters with no data are None.
    :rtype: dict
    """
    def quarter(batch: int) -> int:
        return min(3, int(4 * (batch - 1) / max(1, total_steps)))

    by_dataset = {}
    fam_sum: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 4)
    fam_n: Dict[str, List[int]] = defaultdict(lambda: [0] * 4)
    for name, loss in losses.items():
        q_vals: List[List[float]] = [[], [], [], []]
        for h in loss.history:
            q = quarter(h["batch"])
            q_vals[q].append(h["loss"])
            for fam, (s, n) in (h.get("families") or {}).items():
                fam_sum[fam][q] += s
                fam_n[fam][q] += n
        vals = [v for q in q_vals for v in q]
        by_dataset[name] = {"batches": len(vals), "mean": round(float(np.mean(vals)), 4) if vals else None,
                            "by_quarter": [round(float(np.mean(q)), 4) if q else None for q in q_vals]}
    by_family = {fam: {"rows": sum(fam_n[fam]),
                       "mean": round(sum(fam_sum[fam]) / sum(fam_n[fam]), 4) if sum(fam_n[fam]) else None,
                       "by_quarter": [round(s / n, 4) if n else None for s, n in zip(fam_sum[fam], fam_n[fam])]}
                 for fam in sorted(fam_sum)}
    return {"by_dataset": by_dataset, "by_family": by_family}


# ---------------------------------------------------------------------------------------------------------------
# encoding + negatives
# ---------------------------------------------------------------------------------------------------------------
def encode(model, texts: List[str], is_query: bool, batch_size: int = 128) -> np.ndarray:
    """Encode with the production contract; identical texts are embedded once.

    :param model: SentenceTransformer.
    :param texts: Texts.
    :param is_query: Prefix with :data:`QUERY_PROMPT`.
    :param batch_size: Batch size.
    :returns: (len(texts), d) L2-normalised float32 matrix.
    :rtype: np.ndarray
    """
    uniq = list(dict.fromkeys(texts))
    vecs = model.encode(uniq, prompt=QUERY_PROMPT if is_query else None, batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=len(uniq) > 5000).astype(np.float32)
    if len(uniq) == len(texts):
        return vecs
    where = {t: i for i, t in enumerate(uniq)}
    return vecs[[where[t] for t in texts]]


def choose_negative(row: dict, ranked: Sequence[str], pool: Dict[str, dict], rng: random.Random, skip_top: int,
                    pick_from: int, policy: str = "v4", rejections: Optional[Counter] = None,
                    token_cache: Optional[Dict[str, frozenset]] = None) -> Tuple[Optional[str], str]:
    """Pick one hard negative record for a row from the base model's ranking of the train pool.

    Ranks ``< skip_top`` are skipped (too likely relevant, labels are incomplete). A candidate is rejected, in this
    order (the first reason is counted in ``rejections``): it is the positive or anchor record (``banned_doc``),
    shares their text (``same_text``), shares an id with the row's relevance segment (``relevance``; subject rows: any
    non-broad subject, the anchor subject's ``M:`` id and the festivals the anchor names), (content family) has no
    text left once its transcription is dropped (``content_empty``). Policy ``v4`` (plan C5), doc-level rows only:
    a candidate carrying one of the positive's festival topics (``topic:<f>`` or ``M:topic:<f>``) is rejected
    (``festival``, also counted per festival as ``festival:topic:<f>``), and so is one whose label text has token
    Jaccard >= :data:`NEAR_DUP_JACCARD` with the positive's (``near_duplicate``); ``synthetic_specific`` rows then pick
    among valid candidates sharing a non-broad subject with the positive when there are any. Policy ``v3`` (v9's
    mining): no festival / near-duplicate guard, the same-subject preference for every doc-level row.

    :param row: Mixture row.
    :param ranked: Pool doc ids in the base model's order for this anchor.
    :param pool: Train pool.
    :param rng: RNG.
    :param skip_top: Raw ranks to skip.
    :param pick_from: Choose uniformly among the first this-many eligible candidates.
    :param policy: ``v4`` or ``v3``.
    :param rejections: Counter filled with the rejection reasons (optional).
    :param token_cache: ``doc_id -> label tokens`` cache shared across rows (optional).
    :returns: (negative doc id or None, how it was chosen: ``same_subject`` / ``any`` / ``none``).
    :rtype: Tuple[Optional[str], str]
    """
    if policy not in NEGATIVE_POLICIES:
        raise ValueError(f"policy must be one of {NEGATIVE_POLICIES}, not {policy!r}")
    rej = rejections if rejections is not None else Counter()
    cache = token_cache if token_cache is not None else {}

    def tokens(d: str) -> frozenset:
        if d not in cache:
            cache[d] = label_tokens(pool[d]["text"])
        return cache[d]

    relevance = set(row_segments(row, pool)[0])
    banned_docs = {row["doc_id"], row.get("anchor_doc_id")}
    banned_hash = {row["text_hash"]} | ({pool[row["anchor_doc_id"]]["text_hash"]} if row.get("anchor_doc_id") else set())
    pos_record = pool[row["doc_id"]]
    pos_subjects = {s for s in pos_record["subjects"] if not is_broad(s)}
    doc_level = row["mask_level"] == "doc"
    if policy == "v4":
        prefer_same = doc_level and row["family"] in SAME_SUBJECT_FAMILIES
        pos_fests = festival_names(list(pos_record["subjects"]) + list(row.get("mask_ids") or [])) if doc_level \
            else set()
        pos_tokens = tokens(row["doc_id"]) if doc_level else None
    else:
        prefer_same, pos_fests, pos_tokens = doc_level, set(), None
    valid, preferred = [], []
    for d in ranked[skip_top:]:
        c = pool[d]
        if d in banned_docs:
            rej["banned_doc"] += 1
            continue
        if c["text_hash"] in banned_hash:
            rej["same_text"] += 1
            continue
        if relevance & set(record_identity(d, c["subjects"], c["text_hash"])):
            rej["relevance"] += 1
            continue
        if row["family"] == "content" and not label_part(c["text"]).strip():
            rej["content_empty"] += 1
            continue
        if pos_fests:
            hit = pos_fests & festival_names(c["subjects"])
            if hit:
                rej["festival"] += 1
                for f in sorted(hit):
                    rej[f"festival:topic:{f}"] += 1
                continue
        if pos_tokens is not None and token_jaccard(pos_tokens, tokens(d)) >= NEAR_DUP_JACCARD:
            rej["near_duplicate"] += 1
            continue
        valid.append(d)
        if prefer_same and pos_subjects & set(c["subjects"]):
            preferred.append(d)
    if preferred:
        return rng.choice(preferred[:pick_from]), "same_subject"
    if valid:
        return rng.choice(valid[:pick_from]), "any"
    return None, "none"


def mine_negatives(model, data: dict, top_k: int = 50, skip_top: int = 3, pick_from: int = 10,
                   max_seq: int = EVAL_MAX_SEQ, batch_size: int = 16, seed: int = 0, policy: str = "v4") -> List[dict]:
    """Attach one hard negative to every q2d / d2d row whose candidates allow it (t2t keeps in-batch negatives).

    Candidates are the train pool only (eligible, gated, masked train records -- never held out), ranked by the
    BASE model at the full text (``max_seq`` = :data:`EVAL_MAX_SEQ`; plan B5: every v4 text is under 2,048 tokens, so
    this is the text training and serving see): q2d anchors as queries, d2d anchors (records) by their own document
    vector. See :func:`choose_negative` for the filters. Content-family negatives use the candidate's text without its
    transcription, matching that family's positive form. Rows get ``negative_rejections`` (reason counts) when a
    candidate was rejected; the mining parameters go to ``data["mining_params"]`` (copied into training_info.json).

    :param model: Base SentenceTransformer.
    :param data: Output of :func:`load_data`.
    :param top_k: Ranking depth.
    :param skip_top: Raw ranks skipped.
    :param pick_from: Choose among the first this-many eligible candidates.
    :param max_seq: Token window while mining.
    :param batch_size: Document encode batch size (16 = the batch :func:`evaluate` uses at 8192 tokens).
    :param seed: RNG seed.
    :param policy: Negative policy (:data:`NEGATIVE_POLICIES`).
    :returns: Copies of the mixture rows, some with ``negative``, ``negative_doc_id``, ``negative_choice``.
    :rtype: List[dict]
    """
    if policy not in NEGATIVE_POLICIES:
        raise ValueError(f"policy must be one of {NEGATIVE_POLICIES}, not {policy!r}")
    rng = random.Random(seed)
    pool = data["pool"]
    ids = sorted(pool, key=lambda d: (len(pool[d]["text"]), d))
    row_of = {d: i for i, d in enumerate(ids)}
    prev = model.max_seq_length
    model.max_seq_length = max_seq
    t0 = time.time()
    mat = encode(model, [pool[d]["text"] for d in ids], is_query=False, batch_size=batch_size)
    pairs = [dict(r) for r in data["train"]]
    q_idx = [i for i, p in enumerate(pairs) if p["kind"] == "q2d"]
    d_idx = [i for i, p in enumerate(pairs) if p["kind"] == "d2d"]
    qv = encode(model, [pairs[i]["anchor"] for i in q_idx], is_query=True, batch_size=max(batch_size, 128))
    dv = mat[[row_of[pairs[i]["anchor_doc_id"]] for i in d_idx]]
    model.max_seq_length = prev
    seconds = time.time() - t0
    print(f"mining: embedded {len(ids)} pool records + {len(q_idx)} queries in {seconds:.0f} s", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = torch.from_numpy(mat).to(dev)
    k = min(top_k, len(ids))
    token_cache: Dict[str, frozenset] = {}
    for idx_list, vecs in ((q_idx, qv), (d_idx, dv)):
        for start in range(0, len(idx_list), 1024):
            Q = torch.from_numpy(np.ascontiguousarray(vecs[start:start + 1024])).to(dev)
            top = torch.topk(Q @ M.T, k, dim=1).indices.cpu().numpy()
            for r, i in enumerate(idx_list[start:start + 1024]):
                rej: Counter = Counter()
                neg, how = choose_negative(pairs[i], [ids[j] for j in top[r]], pool, rng, skip_top, pick_from,
                                           policy=policy, rejections=rej, token_cache=token_cache)
                pairs[i]["negative_choice"] = how
                if rej:
                    pairs[i]["negative_rejections"] = dict(rej)
                if neg:
                    text = pool[neg]["text"]
                    pairs[i]["negative"] = label_part(text) if pairs[i]["family"] == "content" else text
                    pairs[i]["negative_doc_id"] = neg
    del M
    data["mining_params"] = {"policy": policy, "max_seq": max_seq, "top_k": top_k, "skip_top": skip_top,
                             "pick_from": pick_from, "seed": seed, "batch_size": batch_size,
                             "same_subject_families": list(SAME_SUBJECT_FAMILIES) if policy == "v4" else "doc-level",
                             "near_dup_jaccard": NEAR_DUP_JACCARD if policy == "v4" else None,
                             "festival_topics": list(FESTIVAL_TOPICS) if policy == "v4" else None,
                             "embed_seconds": round(seconds, 1)}
    return pairs


def mining_stats(pairs: List[dict], top_subjects: int = 50) -> dict:
    """How many rows got a negative (by family and choice) and why candidates were rejected.

    :param pairs: Output of :func:`mine_negatives`.
    :param top_subjects: Subjects kept per reason in ``rejections.by_subject``.
    :returns: ``{total, by_family, rejections: {total, by_family, by_subject, rows_with}}``. ``by_subject``:
        ``festival`` by festival topic, ``near_duplicate`` by the positive's non-broad subjects, ``relevance`` by the
        row's anchor subject.
    :rtype: dict
    """
    out: Dict[str, Counter] = defaultdict(Counter)
    rej_total: Counter = Counter()
    rej_family: Dict[str, Counter] = defaultdict(Counter)
    rej_subject: Dict[str, Counter] = defaultdict(Counter)
    rows_with: Counter = Counter()
    for p in pairs:
        out[p["family"]][p.get("negative_choice", "not_mined")] += 1
        rej = p.get("negative_rejections") or {}
        for reason, n in rej.items():
            if reason.startswith("festival:"):
                rej_subject["festival"][reason.split(":", 1)[1]] += n
                continue
            rej_total[reason] += n
            rej_family[p["family"]][reason] += n
            rows_with[reason] += 1
        if rej.get("near_duplicate"):
            for s in p.get("mask_ids") or []:
                if not s.startswith("D:") and not is_broad(s):
                    rej_subject["near_duplicate"][s] += rej["near_duplicate"]
        if rej.get("relevance") and p.get("anchor_subject"):
            rej_subject["relevance"][p["anchor_subject"]] += rej["relevance"]
    total = Counter()
    for c in out.values():
        total.update(c)
    return {"total": dict(total), "by_family": {f: dict(c) for f, c in sorted(out.items())},
            "rejections": {"total": dict(rej_total), "by_family": {f: dict(c) for f, c in sorted(rej_family.items())},
                           "by_subject": {r: dict(c.most_common(top_subjects)) for r, c in sorted(rej_subject.items())},
                           "rows_with": dict(rows_with)}}


# ---------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------
def load_base(device: Optional[str] = None):
    """The pinned base model with fp32 weights.

    :param device: Device override.
    :returns: SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BASE_MODEL, revision=BASE_REVISION, device=device,
                                model_kwargs={"dtype": torch.float32})
    check_text_path(model)
    return model


def load_previous(model_id: str, revision: Optional[str] = None, token: Optional[str] = None,
                  device: Optional[str] = None):
    """An earlier tuned model (HF repo at a tag, or a local folder) in fp32, checked against the production input path.

    Used to score e.g. v9 (:data:`V9_MODEL` at :data:`V9_REVISION`) on the same eval set next to the new model, and to
    load full fine-tune checkpoints (:func:`load_checkpoint`).

    :param model_id: Model repo id or local directory.
    :param revision: Tag / commit (``v3-run1``); None for a local folder.
    :param token: HF token for a private repo.
    :param device: Device override.
    :returns: SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, revision=revision, token=token, device=device,
                                model_kwargs={"dtype": torch.float32})
    check_text_path(model)
    return model


def resolve_recipe(arm: Optional[str], lr: Optional[float] = None, warmup_ratio: Optional[float] = None) -> dict:
    """Training hyperparameters of an arm (plan D), with explicit overrides.

    :param arm: ``lora`` (arm A), ``full`` (arm B) or None (the v3 / v3-final full fine-tune).
    :param lr: Learning-rate override.
    :param warmup_ratio: Warmup override.
    :returns: ``{arm, lr, warmup_ratio[, lora_r, lora_alpha, lora_dropout]}``.
    :rtype: dict
    :raises ValueError: For an unknown arm.
    """
    if arm is None:
        recipe = dict(V3_RECIPE)
    elif arm in RECIPES:
        recipe = dict(RECIPES[arm])
    else:
        raise ValueError(f"arm must be one of {sorted(RECIPES)} or None, not {arm!r}")
    recipe["arm"] = arm
    if lr is not None:
        recipe["lr"] = lr
    if warmup_ratio is not None:
        recipe["warmup_ratio"] = warmup_ratio
    return recipe


def checkpoint_save_steps(fractions: Sequence[float]) -> Optional[float]:
    """HF ``save_steps`` ratio for evenly spaced checkpoint fractions (e.g. 0.25 for 25/50/75/100 %).

    The last fraction must be 1.0 (the final model, saved by :func:`train` itself; HF saves at multiples of
    ``ceil(ratio * total_steps)``, so the 100 % checkpoint exists only when the steps divide evenly).

    :param fractions: Checkpoint fractions of the total steps; empty = no checkpoints.
    :returns: The ratio, or None.
    :rtype: Optional[float]
    :raises ValueError: When the fractions are not ``k * f`` for ``k = 1..n`` ending at 1.0.
    """
    if not fractions:
        return None
    fr = sorted(float(f) for f in fractions)
    step = fr[0]
    if not 0 < step < 1 or abs(fr[-1] - 1.0) > 1e-9 or any(abs(f - step * (k + 1)) > 1e-9 for k, f in enumerate(fr)):
        raise ValueError(f"checkpoint fractions must be k * f for k = 1..n ending at 1.0, got {fractions}")
    return step


def list_checkpoints(out_dir: str) -> List[Tuple[int, str]]:
    """Trainer checkpoints of a run (``<out_dir>_trainer/checkpoint-<step>``), by step.

    :param out_dir: :func:`train`'s ``out_dir``.
    :returns: ``[(step, path), ...]`` ascending.
    :rtype: List[Tuple[int, str]]
    """
    root = Path(str(out_dir) + "_trainer")
    out = []
    for p in root.glob("checkpoint-*"):
        tail = p.name.rsplit("-", 1)[-1]
        if p.is_dir() and tail.isdigit():
            out.append((int(tail), str(p)))
    return sorted(out)


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
               targets: Sequence[str] = LORA_TARGETS) -> dict:
    """Wrap the model's transformer in a LoRA adapter in place (``model[0].model`` becomes a ``PeftModel``).

    Same wrapping as the local LoRA trial (``train_text_lora.py``); the base weights stay fp32 and frozen.

    :param model: SentenceTransformer (fp32).
    :param r: LoRA rank.
    :param alpha: LoRA alpha.
    :param dropout: LoRA dropout.
    :param targets: Target module names.
    :returns: ``{trainable, total, r, alpha, dropout, targets}``.
    :rtype: dict
    """
    from peft import LoraConfig, TaskType, get_peft_model

    model[0].model = get_peft_model(model[0].model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(targets)))
    model[0].model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total), "r": r, "alpha": alpha, "dropout": dropout,
            "targets": list(targets)}


def merge_check_texts(data: dict, n: int = MERGE_CHECK_TEXTS) -> Tuple[List[str], List[str]]:
    """Fixed texts for the merged-vs-adapter check: half eval-pool documents, half training anchors (queries).

    :param data: Output of :func:`load_data`.
    :param n: Total texts (eval-only packages: all documents).
    :returns: (documents, queries).
    :rtype: Tuple[List[str], List[str]]
    """
    def key(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    anchors = sorted({r["anchor"] for r in data["train"] if r["kind"] == "q2d"}, key=key)
    n_q = min(n // 2, len(anchors))
    ids = sorted((r["doc_id"] for r in data["eval_pool"]), key=key)[: n - n_q]
    return [data["semantic"][d]["text"] for d in ids], anchors[:n_q]


def lora_merge(model, docs: List[str], queries: List[str], min_cosine: float = MERGE_MIN_COSINE) -> dict:
    """Merge the LoRA adapter into the fp32 base weights in place, after checking the merge changes no embedding.

    :param model: SentenceTransformer whose ``model[0].model`` is a ``PeftModel``.
    :param docs: Check documents.
    :param queries: Check queries.
    :param min_cosine: Smallest allowed cosine between the adapter's and the merged model's embedding of a text.
    :returns: ``{n_texts, min_cosine, mean_cosine, threshold}``.
    :rtype: dict
    :raises ValueError: When the model holds no adapter or a cosine falls below ``min_cosine`` (nothing is saved).
    """
    from peft import PeftModel

    if not isinstance(model[0].model, PeftModel):
        raise ValueError("lora_merge: the model has no LoRA adapter")
    before = [encode(model, docs, False, 16), encode(model, queries, True, 64)] if queries else \
        [encode(model, docs, False, 16)]
    model[0].model = model[0].model.merge_and_unload()
    after = [encode(model, docs, False, 16), encode(model, queries, True, 64)] if queries else \
        [encode(model, docs, False, 16)]
    cos = np.concatenate([(a * b).sum(axis=1) for a, b in zip(before, after)])
    out = {"n_texts": int(len(cos)), "min_cosine": round(float(cos.min()), 7), "mean_cosine": round(float(cos.mean()), 7),
           "threshold": min_cosine}
    if cos.min() < min_cosine:
        raise ValueError(f"merged LoRA model differs from the adapter model: {out}")
    check_text_path(model)
    return out


def finalize_for_save(model) -> dict:
    """Put the serving contract on a model before ``model.save`` (plan B5).

    * ``max_seq_length`` = :data:`EVAL_MAX_SEQ` (a reload without an explicit window would otherwise truncate at the
      training window);
    * the fast tokenizer's backend loses its truncation / padding state (``tokenizer.json`` otherwise records the last
      call's ``max_length``: a TEI or raw ``tokenizers`` consumer would truncate every document at the training window);
    * ``prompts`` = :data:`SAVED_PROMPTS` (``prompt_name="query"`` gives the production prefix, documents raw).

    In-memory encoding is unaffected: transformers re-applies truncation / padding on every call.

    :param model: SentenceTransformer.
    :returns: What was set.
    :rtype: dict
    """
    model.max_seq_length = EVAL_MAX_SEQ
    backend = getattr(model.tokenizer, "backend_tokenizer", None)
    if backend is not None:
        backend.no_truncation()
        backend.no_padding()
    model.prompts = dict(SAVED_PROMPTS)
    model.default_prompt_name = None
    return {"max_seq_length": EVAL_MAX_SEQ, "prompts": dict(SAVED_PROMPTS), "backend_tokenizer_reset": backend is not None}


def check_saved_model(model_dir: str) -> dict:
    """Read a saved model folder's config files and assert the serving contract of :func:`finalize_for_save`.

    :param model_dir: Saved SentenceTransformer folder.
    :returns: The values checked.
    :rtype: dict
    :raises AssertionError: When a value differs from the contract.
    """
    root = Path(model_dir)
    found, problems = {}, []

    def load(name: str) -> Optional[dict]:
        p = root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    st_cfg = load("config_sentence_transformers.json")
    found["prompts"] = (st_cfg or {}).get("prompts")
    if found["prompts"] != SAVED_PROMPTS:
        problems.append(f"prompts {found['prompts']!r} != {SAVED_PROMPTS!r}")
    tok_cfg = load("tokenizer_config.json")
    found["model_max_length"] = (tok_cfg or {}).get("model_max_length")
    if found["model_max_length"] != EVAL_MAX_SEQ:
        problems.append(f"tokenizer_config model_max_length {found['model_max_length']} != {EVAL_MAX_SEQ}")
    sbert = load("sentence_bert_config.json") or {}
    if "max_seq_length" in sbert:
        found["sentence_bert_max_seq_length"] = sbert["max_seq_length"]
        if sbert["max_seq_length"] != EVAL_MAX_SEQ:
            problems.append(f"sentence_bert_config max_seq_length {sbert['max_seq_length']} != {EVAL_MAX_SEQ}")
    tok = load("tokenizer.json")
    if tok is None:
        problems.append("tokenizer.json missing")
    else:
        found["tokenizer_truncation"], found["tokenizer_padding"] = tok.get("truncation"), tok.get("padding")
        if tok.get("truncation") is not None or tok.get("padding") is not None:
            problems.append(f"tokenizer.json bakes in truncation={tok.get('truncation')} padding={tok.get('padding')}")
    if problems:
        raise AssertionError(f"{model_dir}: " + "; ".join(problems))
    return found


def model_card_section(info: dict) -> str:
    """Markdown block describing a run's recipe and token windows for the saved model's README.

    :param info: ``training_info.json`` content.
    :returns: Markdown wrapped in :data:`MODEL_CARD_BEGIN` / :data:`MODEL_CARD_END`.
    :rtype: str
    """
    w, rec = info.get("windows") or {}, info.get("recipe") or {}
    sel = info.get("selected_checkpoint")
    lines = [MODEL_CARD_BEGIN, "## Genizah embedder training", "",
             f"- base: `{info.get('base_model')}@{info.get('base_revision')}`",
             f"- arm: `{rec.get('arm') or 'v3 full fine-tune'}`, lr {rec.get('lr')}, warmup {rec.get('warmup_ratio')}, "
             f"cosine, epochs {(info.get('args') or {}).get('epochs')}, steps {info.get('global_steps')}, "
             f"seed {(info.get('args') or {}).get('seed')}",
             f"- token windows: train {w.get('train_max_seq')}, mining {w.get('mining_max_seq')}, "
             f"eval / serving {w.get('eval_max_seq')} (per-mini-batch padding trim: {w.get('padding_trim')})",
             f"- query prompt: `{json.dumps(QUERY_PROMPT)}`; documents raw; `tokenizer.json` carries no truncation or "
             "padding; 1024-d, last-token pooling, L2-normalised",
             f"- data: `{(info.get('run_meta') or {}).get('data_revision')}`; gate thresholds sha256 "
             f"`{(info.get('run_meta') or {}).get('gate_thresholds_sha256')}`"]
    if rec.get("arm") == "lora":
        lines.append(f"- LoRA merged into fp32 weights; merged-vs-adapter check {info.get('lora_merge')}")
    if sel:
        lines.append(f"- selected checkpoint: step {sel.get('step')} ({sel.get('fraction')} of the run)")
    return "\n".join(lines + [MODEL_CARD_END]) + "\n"


def write_model_card(model_dir: str, info: dict) -> Path:
    """Add (or replace) the training block of the model folder's ``README.md``.

    :param model_dir: Saved model folder.
    :param info: ``training_info.json`` content.
    :returns: README path.
    :rtype: Path
    """
    path = Path(model_dir) / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = model_card_section(info)
    if MODEL_CARD_BEGIN in text and MODEL_CARD_END in text:
        head, rest = text.split(MODEL_CARD_BEGIN, 1)
        text = head + block + rest.split(MODEL_CARD_END, 1)[1].lstrip("\n")
    else:
        text = (text.rstrip("\n") + "\n\n" if text else "") + block
    path.write_text(text, encoding="utf-8")
    return path


def save_finalized(model, out_dir: str, info: dict) -> dict:
    """Finalise, save, verify and document a model folder (shared by :func:`train` and :func:`finalize_checkpoint`).

    :param model: SentenceTransformer (merged when it was a LoRA run).
    :param out_dir: Target folder.
    :param info: ``training_info.json`` content (gains ``saved_contract`` / ``saved_model_check``).
    :returns: ``info`` as written.
    :rtype: dict
    """
    info["saved_contract"] = finalize_for_save(model)
    model.save(out_dir)
    info["saved_model_check"] = check_saved_model(out_dir)
    Path(out_dir, "training_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=_json_default))
    write_model_card(out_dir, info)
    return info


def train(data: dict, pairs: List[dict], out_dir: str, epochs: float = 1.0, lr: Optional[float] = None,
          batch: int = 256, mini_batch: int = 8, max_seq: int = TRAIN_MAX_SEQ, bf16: bool = True,
          device: Optional[str] = None, model=None, scale: float = 20.0, max_steps: int = -1,
          sampler: str = "proportional", seed: int = 13, logging_steps: int = 10, arm: Optional[str] = None,
          warmup_ratio: Optional[float] = None, checkpoint_fractions: Sequence[float] = (),
          max_grad_norm: float = 1.0, trim_padding: bool = True, run_meta: Optional[dict] = None):
    """Fine-tune with the masked GradCache InfoNCE, one dataset per kind and negative layout.

    ``arm`` picks the recipe (:func:`resolve_recipe`): ``lora`` / ``full`` are the v10 arms (warmup 10 %, <= 1 epoch;
    pass ``checkpoint_fractions=CHECKPOINT_FRACTIONS`` for the 25 % checkpoints), None the v3 full fine-tune. The
    final model is saved to ``out_dir`` with the serving contract (:func:`save_finalized`); a LoRA run is merged into
    the fp32 weights first (:func:`lora_merge`). Checkpoints stay in ``<out_dir>_trainer`` (LoRA: adapter only);
    :func:`load_checkpoint` turns one into a plain model.

    :param data: Output of :func:`load_data` (``pool`` is used for the mask ids).
    :param pairs: Training rows (output of :func:`mine_negatives`).
    :param out_dir: Where the final model is saved (``training_info.json`` goes next to the weights).
    :param epochs: Epochs (<= 1 for the v10 arms).
    :param lr: Learning rate (default: the arm's).
    :param batch: Effective batch size (in-batch negatives).
    :param mini_batch: GradCache mini-batch (memory knob: 8 fits a 40 GB A100 with checkpointing).
    :param max_seq: Max tokens per text while training.
    :param bf16: bf16 autocast (A100); weights and optimizer state stay fp32. False for CPU smoke tests.
    :param device: Force a device ("cpu" for smoke tests); default picks the accelerator.
    :param model: Model to train in place (smoke tests: a tiny random model); default loads the base model in fp32.
    :param scale: InfoNCE scale (1 / temperature).
    :param max_steps: Stop after this many optimizer steps (-1 = full epochs).
    :param sampler: ``proportional`` (default) or ``round_robin`` over the datasets.
    :param seed: Trainer seed.
    :param logging_steps: Loss logging interval.
    :param arm: ``lora``, ``full`` or None (v3 recipe).
    :param warmup_ratio: Warmup override (default: the arm's).
    :param checkpoint_fractions: Evenly spaced fractions ending at 1.0 (:func:`checkpoint_save_steps`); empty = none.
    :param max_grad_norm: Gradient clipping.
    :param trim_padding: Per-mini-batch padding trim (:func:`trim_minibatch`).
    :param run_meta: Run metadata copied into training_info.json (run name, data revision, thresholds sha ...).
    :returns: Trained SentenceTransformer (merged for LoRA).
    """
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers, MultiDatasetBatchSamplers

    recipe = resolve_recipe(arm, lr, warmup_ratio)
    if arm is not None and epochs > 1.0:
        raise ValueError(f"the v10 arms train at most one epoch, got epochs={epochs}")
    save_ratio = checkpoint_save_steps(checkpoint_fractions)
    # fp32 master weights + bf16 autocast: with bf16 weights an lr-2e-5 Adam step is below half a bf16 ulp for any
    # |w| >= ~0.008, so most updates would round to zero and the "fine-tune" would barely move the model
    if model is None:
        model = load_base(device)
    check_text_path(model)
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError(f"master weights must be fp32, got {next(model.parameters()).dtype}")
    lora_info = None
    if arm == "lora":
        lora_info = apply_lora(model, recipe["lora_r"], recipe["lora_alpha"], recipe["lora_dropout"])
        print(json.dumps({"lora": lora_info}), flush=True)
    model.max_seq_length = max_seq
    datasets, prompts, ds_stats = build_datasets(pairs, data["pool"])
    clock: Counter = Counter()
    losses = {name: MaskedCachedMNRL(model, symmetric=kind_of(name) in SYMMETRIC_KINDS, mini_batch_size=mini_batch,
                                     scale=scale, family_names=ds_stats["families"], clock=clock,
                                     trim_padding=trim_padding) for name in datasets}
    print(json.dumps({"datasets": ds_stats["rows"], "label_width": ds_stats["label_width"], "recipe": recipe,
                      "prompts": {k: sorted(v) for k, v in prompts.items()}}), flush=True)
    save_kwargs = ({"save_strategy": "steps", "save_steps": save_ratio, "save_only_model": True} if save_ratio
                   else {"save_strategy": "no"})
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir + "_trainer", num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch, learning_rate=recipe["lr"], warmup_ratio=recipe["warmup_ratio"],
        lr_scheduler_type="cosine", weight_decay=0.0, max_grad_norm=max_grad_norm,
        bf16=bf16, batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler={"proportional": MultiDatasetBatchSamplers.PROPORTIONAL,
                                     "round_robin": MultiDatasetBatchSamplers.ROUND_ROBIN}[sampler],
        prompts=prompts, logging_steps=logging_steps, report_to=[], seed=seed,
        use_cpu=(device == "cpu"), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, **save_kwargs,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=datasets, loss=losses)
    t0 = time.time()
    trainer.train()
    seconds = time.time() - t0
    steps = trainer.state.global_step
    merge_info = None
    if arm == "lora":
        model.max_seq_length = EVAL_MAX_SEQ
        merge_info = lora_merge(model, *merge_check_texts(data))
    mining = data.get("mining_params") or {}
    padded = sum(loss.stats["tokens_padded"] for loss in losses.values())
    trimmed = sum(loss.stats["tokens_after_trim"] for loss in losses.values())
    info = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION, "query_prompt": QUERY_PROMPT,
            "train_seconds": round(seconds, 1), "global_steps": steps, "recipe": recipe, "lora": lora_info,
            "lora_merge": merge_info, "datasets": ds_stats, "prompts": prompts, "mining": mining_stats(pairs),
            "mining_params": mining,
            "windows": {"train_max_seq": max_seq, "mining_max_seq": mining.get("max_seq"),
                        "eval_max_seq": EVAL_MAX_SEQ, "serving_max_seq": EVAL_MAX_SEQ, "padding_trim": trim_padding,
                        "token_slots_padded": padded, "token_slots_after_trim": trimmed,
                        "trim_saving": round(1 - trimmed / padded, 4) if padded else None},
            "mask_stats": {name: dict(loss.stats) for name, loss in losses.items()},
            "masked_share": {name: round(loss.stats["qd_masked"] / max(1, loss.stats["qd_candidates"]), 4)
                             for name, loss in losses.items()},
            "loss_history": [h for h in trainer.state.log_history if "loss" in h],
            "loss_breakdown": loss_breakdown(losses, steps),
            "checkpoints": [{"step": s, "fraction": round(s / max(1, steps), 4), "path": p}
                            for s, p in list_checkpoints(out_dir)] + [
                {"step": steps, "fraction": 1.0, "path": out_dir, "final": True}],
            "args": {"epochs": epochs, "lr": recipe["lr"], "warmup_ratio": recipe["warmup_ratio"], "batch": batch,
                     "mini_batch": mini_batch, "max_seq": max_seq, "bf16": bf16, "scale": scale,
                     "max_steps": max_steps, "sampler": sampler, "seed": seed, "max_grad_norm": max_grad_norm,
                     "weight_decay": 0.0, "lr_scheduler": "cosine", "checkpoint_fractions": list(checkpoint_fractions),
                     "broad_prefixes": list(BROAD_PREFIXES)},
            "run_meta": run_meta or {},
            "data_manifest": data.get("manifest", {}).get("summary")}
    save_finalized(model, out_dir, info)
    # release optimizer state / caches before the evaluation re-embeds the corpus
    del trainer, losses
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def load_checkpoint(path: str, arm: Optional[str], device: Optional[str] = None,
                    check_texts: Optional[Tuple[List[str], List[str]]] = None, merge_log: Optional[dict] = None):
    """A trainer checkpoint as a plain fp32 model (LoRA: the pinned base + the checkpoint's adapter, merged).

    The base is always :func:`load_base` (pinned revision), never the ``base_model_name_or_path`` recorded in the
    adapter config (it names no revision).

    :param path: ``checkpoint-<step>`` folder.
    :param arm: ``lora`` or ``full`` / None.
    :param device: Device override.
    :param check_texts: (documents, queries) for the merged-vs-adapter check (LoRA; default: no check).
    :param merge_log: Dict updated with the :func:`lora_merge` result (LoRA with ``check_texts``).
    :returns: SentenceTransformer.
    :raises FileNotFoundError: When a LoRA checkpoint holds no ``adapter_config.json``.
    """
    adapters = sorted(Path(path).rglob("adapter_config.json"))
    if not adapters:
        if arm == "lora":
            raise FileNotFoundError(f"{path}: LoRA checkpoint without adapter_config.json")
        return load_previous(path, device=device)
    from peft import PeftModel

    model = load_base(device)
    model[0].model = PeftModel.from_pretrained(model[0].model, str(adapters[0].parent))
    model.max_seq_length = EVAL_MAX_SEQ
    if check_texts is not None:
        result = lora_merge(model, *check_texts)
        if merge_log is not None:
            merge_log.update(result)
    else:
        model[0].model = model[0].model.merge_and_unload()
        check_text_path(model)
    return model


def checkpoint_label(prefix: str, step: int) -> str:
    """Report label of a checkpoint (``<prefix>_s0051``).

    :param prefix: Usually the arm (``lora`` / ``full``).
    :param step: Optimizer step.
    :returns: Label.
    :rtype: str
    """
    return f"{prefix}_s{step:04d}"


def evaluate_checkpoints(data: dict, out_dir: str, arm: Optional[str], report_dir: str,
                         vectors_dir: Optional[str] = None, final_model=None, prefix: Optional[str] = None,
                         skip_existing: bool = True, **eval_kwargs) -> Dict[str, dict]:
    """Score every checkpoint of a run (eval_v3 + gate) and write the reports as they finish.

    The checkpoints are those listed in ``<out_dir>/training_info.json`` (a trainer checkpoint at the final step is
    the final model and is not scored twice). Each report goes to ``<report_dir>/<label>.json``; the index
    ``<report_dir>/checkpoints.json`` maps labels to ``{arm, step, fraction, path}`` (:func:`load_checkpoint_index`).

    :param data: Output of :func:`load_data`.
    :param out_dir: :func:`train`'s ``out_dir``.
    :param arm: The run's arm.
    :param report_dir: Report folder (Drive).
    :param vectors_dir: Vector folder (Drive), or None.
    :param final_model: The trained model still in memory (else the final folder is loaded).
    :param prefix: Label prefix (default: the arm, or ``v3``).
    :param skip_existing: Keep reports already in ``report_dir``.
    :param eval_kwargs: Forwarded to :func:`evaluate`.
    :returns: ``label -> {arm, step, fraction, path}``.
    :rtype: Dict[str, dict]
    """
    info = json.loads(Path(out_dir, "training_info.json").read_text())
    total = info["global_steps"]
    prefix = prefix or arm or "v3"
    check = merge_check_texts(data) if arm == "lora" else None
    index: Dict[str, dict] = {}
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    for ck in info["checkpoints"]:
        if ck["step"] == total and not ck.get("final"):
            continue  # the trainer saved the last step too: same weights as the final model
        label = checkpoint_label(prefix, ck["step"])
        index[label] = {"arm": arm, "step": ck["step"], "fraction": ck["fraction"], "path": ck["path"],
                        "final": bool(ck.get("final"))}
        out_path = Path(report_dir) / f"{label}.json"
        if not (skip_existing and out_path.exists()):
            if ck.get("final") and final_model is not None:
                model = final_model
            elif ck.get("final"):
                model = load_previous(ck["path"])
            else:
                model = load_checkpoint(ck["path"], arm, check_texts=check)
            evaluate(model, data, label, out_path=str(out_path), vectors_dir=vectors_dir, **eval_kwargs)
            if model is not final_model:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        (Path(report_dir) / "checkpoints.json").write_text(json.dumps(index, indent=1))
    return index


def load_checkpoint_index(report_dir: str) -> Dict[str, dict]:
    """Re-read the index written by :func:`evaluate_checkpoints` (after a Colab runtime restart).

    :param report_dir: Report folder.
    :returns: ``label -> {arm, step, fraction, path, final}``.
    :rtype: Dict[str, dict]
    """
    return json.loads(Path(report_dir, "checkpoints.json").read_text())


def finalize_checkpoint(data: dict, out_dir: str, ckpt: dict, dest: str) -> dict:
    """Save one scored checkpoint as a servable model folder (merged, serving contract, training_info, model card).

    :param data: Output of :func:`load_data` (LoRA merge-check texts).
    :param out_dir: The run's :func:`train` ``out_dir`` (its ``training_info.json`` is copied).
    :param ckpt: Entry of :func:`evaluate_checkpoints`' index.
    :param dest: Target folder (Drive).
    :returns: The written ``training_info.json`` content.
    :rtype: dict
    """
    info = json.loads(Path(out_dir, "training_info.json").read_text())
    info["selected_checkpoint"] = {k: ckpt.get(k) for k in ("arm", "step", "fraction", "path", "final")}
    if ckpt.get("final"):
        shutil.copytree(out_dir, dest, dirs_exist_ok=True)
        Path(dest, "training_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False,
                                                               default=_json_default))
        write_model_card(dest, info)
        check_saved_model(dest)
        return info
    check = merge_check_texts(data) if ckpt.get("arm") == "lora" else None
    merge_log: dict = {}
    model = load_checkpoint(ckpt["path"], ckpt.get("arm"), check_texts=check, merge_log=merge_log)
    info["lora_merge"] = merge_log or None
    info = save_finalized(model, dest, info)
    del model
    gc.collect()
    return info


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def corpus_texts(data: dict, text: str, ids: Sequence[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """Ids present in a text variant, their texts, and the eval-pool hash field that variant is checked against.

    :param data: Output of :func:`load_data`.
    :param text: ``semantic`` (the eval dir's embedded text), ``original`` (v8 production text), or a key of
        ``data["texts"]`` (``semantic_v3``, ``semantic_nf``).
    :param ids: Eval-pool ids in order.
    :returns: (ids with a text, texts, hash field or None).
    :rtype: Tuple[List[str], List[str], Optional[str]]
    :raises ValueError: For an unknown variant, or a pool record missing from the ``semantic`` corpus.
    """
    if text == "semantic":
        src = {d: r["text"] for d, r in data["semantic"].items()}
    elif text == "original":
        src = data["original"]
    elif text in data.get("texts", {}):
        src = data["texts"][text]
    else:
        raise ValueError(f"text must be semantic, original or one of {sorted(data.get('texts', {}))}, not {text!r}")
    present = [d for d in ids if d in src]
    if text == "semantic" and len(present) != len(ids):
        raise ValueError(f"{len(ids) - len(present)} eval-pool records have no semantic text")
    return present, [src[d] for d in present], TEXT_HASH_KEYS.get(text)


def save_vectors(vectors_dir: str, label: str, ids: Sequence[str], mat: np.ndarray,
                 queries: Dict[str, np.ndarray]) -> dict:
    """Write a model's document and query vectors (float16) next to the reports, for re-scoring without a GPU.

    Files: ``vectors_<label>.f16.npy`` + ``vectors_<label>.ids.json`` (row order) and ``queries_<label>.f16.npy`` +
    ``queries_<label>.json`` (query texts, row order; vectors WITHOUT re-normalisation after the float16 cast).

    :param vectors_dir: Folder (Drive, not the Hub).
    :param label: Report label.
    :param ids: Document ids, one per row of ``mat``.
    :param mat: (n, d) document vectors.
    :param queries: Query text -> vector.
    :returns: ``{dir, docs, n_docs, n_queries}``.
    :rtype: dict
    """
    out = Path(vectors_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"vectors_{label}.f16.npy", np.asarray(mat, dtype=np.float16))
    (out / f"vectors_{label}.ids.json").write_text(json.dumps(list(ids)))
    qtexts = list(queries)
    qmat = np.stack([queries[t] for t in qtexts]) if qtexts else np.zeros((0, mat.shape[1]))
    np.save(out / f"queries_{label}.f16.npy", qmat.astype(np.float16))
    (out / f"queries_{label}.json").write_text(json.dumps(qtexts, ensure_ascii=False))
    return {"dir": str(out), "docs": f"vectors_{label}.f16.npy", "n_docs": len(ids), "n_queries": len(qtexts)}


def evaluate(model, data: dict, label: str, text: str = "semantic", max_seq: int = EVAL_MAX_SEQ,
             batch_size: int = 16, n_resamples: int = 1000, out_path: Optional[str] = None,
             ignore_ids: Optional[Iterable[str]] = None, vectors_dir: Optional[str] = None, gate: bool = True) -> dict:
    """Score one model with the frozen eval (``eval_v3.evaluate``, + ``eval_gate.evaluate_gate`` when the eval dir has
    ``gate_v1/``) over the whole eligible pool.

    :param model: SentenceTransformer.
    :param data: Output of :func:`load_data`.
    :param label: Name for the report (``base_original``, ``P``, ``B``, ``lora_s0051`` ...).
    :param text: Document text to embed (:func:`corpus_texts`): ``semantic`` (the frozen embedded text; hash-checked),
        ``original`` (v8 production text), ``semantic_v3`` (v9 text; hash-checked against the pool's
        ``text_hash_v3`` when the eval dir and eval_v3 carry it), ``semantic_nf`` (no-frames ablation text).
    :param max_seq: Token window; 8192 = production, so every model is scored on the vectors production would serve.
    :param batch_size: Document encode batch size.
    :param n_resamples: Bootstrap resamples (memorisation-gap CI).
    :param out_path: Write the full result (incl. per-query lists, needed for paired comparisons) here as JSON.
    :param ignore_ids: Records removed from the pool for this model (V: the ids v9 trained on).
    :param vectors_dir: Also write the document / query vectors there (:func:`save_vectors`).
    :param gate: Run the gate when the eval dir has ``gate_v1/`` (result in ``res["gate"]``).
    :returns: ``eval_v3.evaluate`` result plus ``meta`` (and ``gate``).
    :rtype: dict
    """
    import eval_v3

    pool_ids = [r["doc_id"] for r in data["eval_pool"]]
    ids, texts, hash_key = corpus_texts(data, text, pool_ids)
    checked = hash_key == "text_hash" or (
        hash_key is not None and "text_hash_key" in inspect.signature(eval_v3.evaluate).parameters
        and all(hash_key in r for r in data["eval_pool"]))
    kwargs = {"text_hash_key": hash_key} if checked and hash_key != "text_hash" else {}
    corpus_rows = {d: {"text": t} for d, t in zip(ids, texts)} if checked else None
    ignore = sorted(set(ignore_ids)) if ignore_ids else None
    prev = model.max_seq_length
    model.max_seq_length = max_seq
    t0 = time.time()
    mat = encode(model, texts, is_query=False, batch_size=batch_size)
    t_docs = time.time() - t0
    qcache: Dict[str, np.ndarray] = {}

    def qfn(queries: List[str]) -> np.ndarray:
        todo = [q for q in dict.fromkeys(queries) if q not in qcache]
        if todo:
            qcache.update(zip(todo, encode(model, todo, is_query=True, batch_size=64)))
        return np.stack([qcache[q] for q in queries]) if queries else np.zeros((0, mat.shape[1]), dtype=np.float32)

    tsp = trained_subject_positives(data["train"]) if data["train"] else None
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious BLAS warnings on unit vectors
        res = eval_v3.evaluate(ids, mat, qfn, data["eval_dir"], corpus_rows=corpus_rows, n_resamples=n_resamples,
                               trained_subject_positives=tsp, ignore_ids=ignore, **kwargs)
        t_gate = time.time()
        gd = gate_dir(data) if gate else None
        if gd is not None:
            import eval_gate

            # H5 needs a document encoder and the text version; the hash check (H11) runs on the frozen texts only
            probe_text = {"semantic": "semantic_v4", "semantic_v3": "semantic_v3"}.get(text)
            hashed = text in ("semantic", "semantic_v3")
            res["gate"] = eval_gate.evaluate_gate(
                ids, mat, qfn, str(gd), ignore_ids=ignore,
                doc_vectors_fn=lambda docs: encode(model, docs, is_query=False, batch_size=batch_size),
                probe_text=probe_text,
                text_hashes={d: hashlib.md5(t.encode("utf-8")).hexdigest() for d, t in zip(ids, texts)} if hashed else None,
                hash_field="text_hash_v3" if text == "semantic_v3" else "text_hash")
        t_gate = time.time() - t_gate
    model.max_seq_length = prev
    if checked and res["pool"]["text_hash_mismatch"]:
        raise ValueError(f"{res['pool']['text_hash_mismatch']} pool vectors are not of the frozen {text} text")
    res["meta"] = {"label": label, "text": text, "max_seq": max_seq, "embed_seconds": round(t_docs, 1),
                   "gate_seconds": round(t_gate, 1), "total_seconds": round(time.time() - t0, 1), "n_docs": len(ids),
                   "n_pool_missing_text": len(pool_ids) - len(ids), "text_hash_checked": hash_key if checked else None,
                   "n_ignored_ids": len(ignore or []), "gate": gd is not None,
                   "dtype": str(next(model.parameters()).dtype), "query_prompt": QUERY_PROMPT}
    if vectors_dir:
        res["meta"]["vectors"] = save_vectors(vectors_dir, label, ids, mat, qcache)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, ensure_ascii=False, default=_json_default))
    return res


def cached_report(report_dir: str, label: str) -> Optional[dict]:
    """A report already written to ``<report_dir>/<label>.json`` (reference models are scored once per dataset tag).

    :param report_dir: Report folder.
    :param label: Label.
    :returns: The report or None.
    :rtype: Optional[dict]
    """
    p = Path(report_dir, f"{label}.json")
    return json.loads(p.read_text()) if p.exists() else None


def trained_subject_positives(rows: Iterable[dict]) -> Dict[str, List[str]]:
    """Records the mixture trains as positives of each subject's name anchors (``subject`` family).

    Passed to ``eval_v3.evaluate`` for every model of a comparison, so ``memorisation_trained`` measures the gap on
    records the tuned model actually saw with that subject's names (the frozen sample is only train-SIDE).

    :param rows: Mixture rows.
    :returns: ``subject -> sorted doc ids``.
    :rtype: Dict[str, List[str]]
    """
    out: Dict[str, set] = defaultdict(set)
    for r in rows:
        if r["family"] == "subject" and r.get("anchor_subject") and r.get("doc_id"):
            out[r["anchor_subject"]].add(r["doc_id"])
    return {s: sorted(v) for s, v in sorted(out.items())}


def _json_default(obj):
    """JSON fallback for numpy scalars / arrays and sets.

    :param obj: Object.
    :returns: JSON-serialisable value.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))


def load_reports(report_dir: str, labels: Sequence[str]) -> Dict[str, dict]:
    """Re-read reports written by :func:`evaluate` (after a Colab runtime restart).

    :param report_dir: Directory holding ``<label>.json``.
    :param labels: Labels to read.
    :returns: ``label -> result``.
    :rtype: Dict[str, dict]
    """
    return {lb: json.loads(Path(report_dir, f"{lb}.json").read_text()) for lb in labels}


def primary_group(res: dict) -> str:
    """The subject group that decides a comparison: ``held_out`` (the gate) or, without held-out subjects, ``focus``.

    :param res: ``eval_v3.evaluate`` result (reports written before the focus group existed default to ``held_out``).
    :returns: Group name.
    :rtype: str
    """
    return res["subjects"].get("primary_group") or "held_out"


def _subject_groups(res: dict) -> List[Tuple[str, str]]:
    """``(group, display name)`` of the subject groups shown in the table, in order.

    The v3 layout (gate, linked, seen) when the eval dir has held-out subjects; otherwise the focus group and seen
    subjects (empty held-out / linked groups are left out).

    :param res: ``eval_v3.evaluate`` result.
    :returns: Groups.
    :rtype: List[Tuple[str, str]]
    """
    sub = res["subjects"]
    if primary_group(res) == "held_out":
        groups = [("held_out", "GATE held-out subjects"), ("linked", "linked subjects")]
    else:
        groups = [(g, n) for g, n in (("held_out", "held-out subjects"), ("linked", "linked subjects"))
                  if sub[g]["macro"].get("n_subjects")]
    if "focus" in sub:
        groups.append(("focus", "FOCUS subjects (held-out records only)"))
    return groups + [("seen", "seen subjects")]


def _holiday_sections(res: dict) -> List[Tuple[str, str, dict]]:
    """``(compare group, display name, section)`` of the holiday sections present in a result.

    :param res: ``eval_v3.evaluate`` result.
    :returns: The whole-pool section and (when present) its held-out-only variant.
    :rtype: List[Tuple[str, str, dict]]
    """
    hol = res.get("holiday")
    if not hol:
        return []
    out = [("holiday", "holiday", hol)]
    held = hol.get("held_out")
    if isinstance(held, dict) and held.get("per_query"):
        out.append(("holiday_held_out", "holiday held-out", held))
    return out


def _holiday_rows(res: dict) -> List[Tuple[str, Optional[float]]]:
    """Holiday table rows of one result (finding 16): macro P@10 / MRR@100 / canon_share@20, alt-only lift, the
    worst festival's P@10 and every festival's P@10, for the whole pool and the held-out-only variant.

    :param res: ``eval_v3.evaluate`` result.
    :returns: Rows (none when the eval dir has no holiday section).
    :rtype: List[Tuple[str, Optional[float]]]
    """
    rows: List[Tuple[str, Optional[float]]] = []
    for _, name, sec in _holiday_sections(res):
        mac = sec.get("macro") or {}
        rows.append((f"{name} macro P@10", mac.get("P@10")))
        rows.append((f"{name} macro MRR@100", mac.get("MRR@100")))
        if "canon_share@20" in mac:
            rows.append((f"{name} macro canon_share@20 (lower=better)", mac.get("canon_share@20")))
        elif "weak_share@20" in mac:  # holiday reports written before the typed weak labels
            rows.append((f"{name} macro weak_share@20 (lower=better)", mac.get("weak_share@20")))
        if "alt_only_lift" in mac:
            rows.append((f"{name} macro alt_only_lift (1 = none)", mac.get("alt_only_lift")))
        rows.append((f"{name} macro n_queries", mac.get("n_queries")))
        worst = sec.get("worst_festival") or {}
        rows.append((f"  {name} worst festival P@10", worst.get("P@10")))
        for fest, v in sorted((sec.get("per_festival") or {}).items()):
            rows.append((f"  {name} {fest} P@10", v.get("P@10")))
    return rows


def _metric_rows(res: dict) -> List[Tuple[str, Optional[float]]]:
    """Headline ``(metric, value)`` rows of one result, in table order.

    :param res: ``eval_v3.evaluate`` result.
    :returns: Rows.
    :rtype: List[Tuple[str, Optional[float]]]
    """
    rows: List[Tuple[str, Optional[float]]] = []
    sub = res["subjects"]
    for group, name in _subject_groups(res):
        mac = sub[group]["macro"]
        for m in ("AP", "P@10", "R@100"):
            rows.append((f"{name} (n={mac.get('n_subjects', 0)}) {m}", mac.get(m)))
        if group == "held_out":
            for src, v in sorted(sub[group]["macro_by_source"].items()):
                rows.append((f"  held-out by source {src} AP", v.get("AP")))
            for sid, v in sorted(sub[group]["per_subject"].items()):
                rows.append((f"  {sid} AP", v.get("AP")))
        if group == "focus":
            for src, v in sorted(sub[group]["macro_by_source"].items()):
                for m in ("AP", "P@10", "R@100"):
                    rows.append((f"  focus by source {src} (n={v.get('n_subjects', 0)}) {m}", v.get(m)))
            for sid, v in sorted(sub[group]["per_subject"].items()):
                rows.append((f"  {sid} (held-out positives={v.get('n_relevant')}) AP", v.get("AP")))
                for src, sv in sorted((v.get("by_source") or {}).items()):
                    rows.append((f"    {sid} {src} AP", sv.get("AP")))
        if group == "seen":
            for kind, v in sorted(sub[group]["macro_by_kind"].items()):
                rows.append((f"  seen {kind} (n={v['n_subjects']}) AP", v.get("AP")))
    for key, name in (("memorisation", "memorisation (train-side sample)"),
                      ("memorisation_trained", "memorisation (trained positives)")):
        if key not in res:
            continue
        mem = res[key]["macro"]
        for m in ("AP_train", "AP_held", "gap"):
            rows.append((f"{name} {m} (n={mem.get('n_subjects', 0)})", mem.get(m)))
        ci = mem.get("gap_ci") or [None, None]
        rows.append((f"{name} gap CI low", ci[0]))
        rows.append((f"{name} gap CI high", ci[1]))
    ki = res["known_item"]
    for m in ("MRR", "R@10", "R@100"):
        rows.append((f"known-item all (n={ki['all'].get('n', 0)}) {m}", ki["all"].get(m)))
    for t, v in sorted(ki["by_type"].items()):
        rows.append((f"  known-item {t} (n={v['n']}) MRR", v.get("MRR")))
    ws = ki["within_subject"]
    rows.append(("  known-item within-subject MRR", ws.get("MRR")))
    rows.append(("  known-item within-subject median pct (lower=better)", ws.get("median_percentile")))
    for part in ("all", "long"):
        rows.append((f"collection lift {part} (1 = none)", res["collection"][part].get("lift")))
    rows.append(("collection lift_vs_pool all", res["collection"]["all"].get("lift_vs_pool")))
    rows.append(("hubness share dup/short in top-10", res.get("hubness", {}).get("share")))
    rows.append(("hubness ratio vs pool (1 = chance)", res.get("hubness", {}).get("ratio")))
    return rows + _holiday_rows(res)


def holiday_compare(res_a: dict, res_b: dict, group: str, metric: str, n_resamples: int = 1000) -> Optional[dict]:
    """Paired bootstrap of B minus A on one holiday metric, clustered by festival (``eval_v3.paired_bootstrap``).

    Only the queries both results scored are paired (a model scored with ``ignore_ids`` may lose a festival below the
    relevance minimum).

    :param res_a: Baseline result.
    :param res_b: Candidate result.
    :param group: ``holiday`` or ``holiday_held_out``.
    :param metric: Per-query metric (``P@10``, ``MRR@100``, ``canon_share@20`` ...).
    :param n_resamples: Bootstrap resamples.
    :returns: Bootstrap dict, or None when no query is shared.
    :rtype: Optional[dict]
    """
    import eval_v3

    secs = []
    for res in (res_a, res_b):
        sec = {g: s for g, _, s in _holiday_sections(res)}.get(group)
        secs.append((sec or {}).get("per_query") or {})
    pa, pb = secs
    ca: Dict[str, List[float]] = defaultdict(list)
    cb: Dict[str, List[float]] = defaultdict(list)
    for key in sorted(set(pa) & set(pb)):
        va, vb = pa[key].get(metric), pb[key].get(metric)
        if va is None or vb is None:
            continue
        fest = pa[key].get("festival") or key.split("|", 1)[0]
        ca[fest].append(va)
        cb[fest].append(vb)
    if not ca:
        return None
    return eval_v3.paired_bootstrap(ca, cb, n_resamples)


def known_item_compare(res_a: dict, res_b: dict, metric: str, n_resamples: int = 1000) -> dict:
    """Known-item paired bootstrap (clustered by target record) over the queries both results scored.

    ``eval_v3.compare`` refuses two results whose known-item query lists differ; a model scored with ``ignore_ids``
    (V) drops the queries of ignored targets, so the comparison is restricted to the shared ``qid`` s.

    :param res_a: Baseline result.
    :param res_b: Candidate result.
    :param metric: ``RR``, ``R@10`` or ``R@100``.
    :param n_resamples: Bootstrap resamples.
    :returns: Bootstrap dict (``n_queries`` = shared queries).
    :rtype: dict
    """
    import eval_v3

    pa, pb = res_a["known_item"]["per_query"], res_b["known_item"]["per_query"]
    if pa["qid"] == pb["qid"]:
        return eval_v3.compare(res_a, res_b, "known_item", metric, n_resamples)
    rank_b = dict(zip(pb["qid"], pb["rank"]))

    def val(rank: float) -> float:
        return {"RR": 1.0 / rank, "R@10": float(rank <= 10), "R@100": float(rank <= 100)}[metric]

    ca: Dict[str, List[float]] = defaultdict(list)
    cb: Dict[str, List[float]] = defaultdict(list)
    for qid, doc, rank in zip(pa["qid"], pa["doc_id"], pa["rank"]):
        if qid in rank_b:
            ca[doc].append(val(rank))
            cb[doc].append(val(rank_b[qid]))
    return eval_v3.paired_bootstrap(ca, cb, n_resamples)


def compare_reports(reports: Dict[str, dict], baseline: str = "base_semantic", candidate: Optional[str] = None,
                    n_resamples: int = 1000, also_vs: Sequence[str] = (), refs: Optional[Dict[str, str]] = None,
                    thresholds: Optional[dict] = None, extra: Optional[Dict[str, dict]] = None) -> dict:
    """Side-by-side headline metrics of several results plus paired, clustered bootstrap CIs (and the gate verdict).

    The primary group is the held-out-subject gate, or -- when the eval dir has no held-out subject (final run) --
    the ``focus`` group (:func:`primary_group`). Paired tests: candidate (default: the last report) vs ``baseline`` on
    every subject group present (AP, P@10, R@100), every primary-group subject's AP, known-item RR / R@10, the holiday
    section (P@10 / MRR@100 / canon_share@20, clustered by festival, whole pool and held-out only); plus the step-0
    effect ``base_semantic`` vs ``base_original`` and the candidate vs ``base_original`` on the primary AP and
    known-item RR when those reports are present. Each label in ``also_vs`` (e.g. P / V / B3 or an earlier tuned
    model scored on the same eval) gets the candidate-vs-it CIs on the primary group, its subjects, seen subjects,
    known-item and holiday, and its own primary AP vs ``baseline``. With ``refs`` (``{"P", "B", "B3", "V"} -> label``)
    and ``thresholds``, ``verdict`` = ``eval_gate.verdict`` of the candidate (:func:`run_verdict`; every report
    involved must carry ``gate``), printed by :func:`format_table`.

    :param reports: ``label -> eval result`` in display order (same eval dir).
    :param baseline: Baseline label.
    :param candidate: Candidate label (default: last key of ``reports``).
    :param n_resamples: Bootstrap resamples.
    :param also_vs: Further labels to compare the candidate against.
    :param refs: Gate reference roles -> labels (enables the verdict).
    :param thresholds: Frozen gate thresholds (:func:`gate_thresholds`).
    :param extra: ``label -> extra`` inputs of the verdict (H0 / H10 / H12 results, :func:`load_extra`).
    :returns: ``{labels, baseline, candidate, primary_group, primary_name, primary, table: [{metric, values}],
        paired: {name: CI dict}[, verdict]}``.
    :rtype: dict
    """
    import eval_v3

    labels = list(reports)
    candidate = candidate or labels[-1]
    pg = primary_group(reports[candidate])
    per_label = {lb: dict(_metric_rows(r)) for lb, r in reports.items()}
    order = [m for m, _ in _metric_rows(reports[labels[0]])]
    known = set(order)
    order += [m for m, _ in _metric_rows(reports[candidate]) if m not in known]  # e.g. holiday rows a ref lacks
    table = [{"metric": m, "values": {lb: per_label[lb].get(m) for lb in labels}} for m in order]
    paired: Dict[str, dict] = {}
    done: set = set()

    def has_queries(label: str, group: str) -> bool:
        return bool(reports[label]["subjects"].get(group, {}).get("per_query"))

    def add(name: str, a: str, b: str, group: str, metric: str) -> None:
        if a not in reports or b not in reports or a == b or (a, b, group, metric) in done:
            return  # missing report, self-comparison, or already reported under another name
        if group in ("holiday", "holiday_held_out"):
            ci = holiday_compare(reports[a], reports[b], group, metric, n_resamples)
            if ci is not None:
                done.add((a, b, group, metric))
                paired[name] = ci
            return
        if group == "known_item":
            done.add((a, b, group, metric))
            paired[name] = known_item_compare(reports[a], reports[b], metric, n_resamples)
            return
        if not (has_queries(a, group) and has_queries(b, group)):
            return  # empty group (e.g. no held-out subject in a final-run eval)
        done.add((a, b, group, metric))
        paired[name] = eval_v3.compare(reports[a], reports[b], group, metric, n_resamples)

    def per_subject(a: str, b: str, group: str) -> None:
        if a not in reports or b not in reports or a == b or not has_queries(a, group):
            return
        pb = reports[b]["subjects"].get(group, {}).get("per_query", {})
        for sid, pq in reports[a]["subjects"][group]["per_query"].items():
            if sid in pb:
                paired[f"{b} - {a} | {sid} AP"] = eval_v3.paired_bootstrap({sid: pq["AP"]}, {sid: pb[sid]["AP"]},
                                                                            n_resamples)

    def add_holiday(a: str, b: str) -> None:
        for group, name, _ in _holiday_sections(reports[b]):
            for m in ("P@10", "MRR@100", "canon_share@20"):
                add(f"{b} - {a} | {name} {m}", a, b, group, m)

    a, b = baseline, candidate
    for group in ("held_out", "linked", "focus", "seen"):
        for m in ("AP", "P@10", "R@100"):
            add(f"{b} - {a} | {group} {m}", a, b, group, m)
    for m in ("RR", "R@10"):
        add(f"{b} - {a} | known_item {m}", a, b, "known_item", m)
    for group in ("memorisation", "memorisation_trained"):  # gap net of the baseline's calibration gap
        if a in reports and b in reports and all(group in reports[x] and "per_query_gap" in reports[x][group]
                                                 for x in (a, b)):
            done.add((a, b, group, "gap"))
            paired[f"{b} - {a} | {group} gap (net of baseline gap)"] = eval_v3.compare(reports[a], reports[b], group,
                                                                                      "gap", n_resamples)
    add_holiday(a, b)
    per_subject(a, b, pg)
    add(f"base_semantic - base_original | {pg} AP (step 0)", "base_original", "base_semantic", pg, "AP")
    add("base_semantic - base_original | known_item RR (step 0)", "base_original", "base_semantic", "known_item", "RR")
    add(f"{b} - base_original | {pg} AP (vs production today)", "base_original", b, pg, "AP")
    add(f"{b} - base_original | known_item RR (vs production today)", "base_original", b, "known_item", "RR")
    for other in also_vs:
        for m in ("AP", "P@10", "R@100"):
            add(f"{b} - {other} | {pg} {m}", other, b, pg, m)
        add(f"{b} - {other} | seen AP", other, b, "seen", "AP")
        add(f"{b} - {other} | known_item RR", other, b, "known_item", "RR")
        if other in reports:
            add_holiday(other, b)
        per_subject(other, b, pg)
        add(f"{other} - {a} | {pg} AP", a, other, pg, "AP")
    primary_name = f"{b} - {a} | {pg} AP"
    out = {"labels": labels, "baseline": a, "candidate": b, "table": table, "paired": paired,
           "primary_group": pg, "primary_name": primary_name, "primary": paired.get(primary_name)}
    if refs and thresholds is not None:
        out["verdict"] = run_verdict(reports, b, refs, thresholds, extra)
        out["refs"] = dict(refs)
    return out


def format_table(comparison: dict, deferred: Sequence[str] = ()) -> str:
    """Fixed-width text of :func:`compare_reports` output (values side by side, the paired CIs, the gate verdict).

    :param comparison: Output of :func:`compare_reports`.
    :param deferred: Criteria shown as deferred in the verdict block (:func:`format_verdict`).
    :returns: Table text.
    :rtype: str
    """
    labels = comparison["labels"]
    width = max(len(r["metric"]) for r in comparison["table"]) + 2
    col = max(12, max(len(lb) for lb in labels) + 2)
    lines = [f"{'metric':<{width}}" + "".join(f"{lb:>{col}}" for lb in labels)]
    for r in comparison["table"]:
        vals = "".join(f"{'-':>{col}}" if r["values"][lb] is None else f"{r['values'][lb]:>{col}.4f}" for lb in labels)
        lines.append(f"{r['metric']:<{width}}{vals}")
    lines.append("")
    lines.append("paired bootstrap (B minus A): delta [95% CI]  p(B better)  clusters/queries")
    for name, ci in comparison["paired"].items():
        lines.append(f"  {name:<70} {ci['delta']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
                     f"{ci['p_b_better']:.2f}  {ci['n_clusters']}/{ci['n_queries']}")
    if comparison.get("verdict") is not None:
        lines.append("")
        lines.append(format_verdict(comparison["verdict"], comparison["candidate"], comparison.get("refs"), deferred))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# gate verdict + checkpoint selection (plan E)
# ---------------------------------------------------------------------------------------------------------------
def criterion_base(cid: str) -> str:
    """The ``H<n>`` a criterion id belongs to (``H1``, ``H1.G3`` and ``H1_vs_P`` -> ``H1``; ``H10`` stays ``H10``).

    :param cid: Criterion id.
    :returns: Base id (the id itself when it does not start with ``H<digits>``).
    :rtype: str
    """
    m = re.match(r"H\d+", cid)
    return m.group(0) if m else cid


def gate_inputs(reports: Dict[str, dict], extra: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """``eval_gate.verdict`` input: ``label -> {gate, eval_v3, extra}`` from :func:`evaluate` reports.

    :param reports: ``label -> report`` (``report["gate"]`` = the gate report).
    :param extra: ``label -> extra`` (H0 / H10 / H12 inputs).
    :returns: Verdict input.
    :rtype: Dict[str, dict]
    """
    return {lb: {"gate": r.get("gate"), "eval_v3": r, "extra": dict((extra or {}).get(lb) or {})}
            for lb, r in reports.items()}


def run_verdict(reports: Dict[str, dict], cand: str, refs: Dict[str, str], thresholds: dict,
                extra: Optional[Dict[str, dict]] = None) -> dict:
    """``eval_gate.verdict`` of one candidate against the reference roles.

    :param reports: ``label -> report`` (must hold the candidate and every reference label).
    :param cand: Candidate label.
    :param refs: ``{"P", "B", "B3", "V"} -> label``.
    :param thresholds: Frozen thresholds.
    :param extra: ``label -> extra`` inputs.
    :returns: ``{"pass", "criteria": {id: {pass, measured, value, threshold, detail}}}``.
    :rtype: dict
    :raises ValueError: When a label is missing or a role is not one of :data:`REF_ROLES`.
    """
    import eval_gate

    if set(refs) - set(REF_ROLES):
        raise ValueError(f"refs roles must be among {REF_ROLES}, got {sorted(refs)}")
    missing = [lb for lb in [cand, *refs.values()] if lb not in reports]
    if missing:
        raise ValueError(f"reports missing for {missing}")
    return eval_gate.verdict(gate_inputs(reports, extra), cand, dict(refs), thresholds)


def failing_criteria(verdict: dict, deferred: Sequence[str] = ()) -> Dict[str, dict]:
    """Criteria that do not pass (NOT MEASURED counts as FAIL), skipping the ``deferred`` ``H<n>`` ids.

    :param verdict: Output of ``eval_gate.verdict``.
    :param deferred: Base ids evaluated later (:data:`DEFERRED_CRITERIA`).
    :returns: ``criterion id -> criterion``.
    :rtype: Dict[str, dict]
    """
    out = {}
    for cid, c in (verdict.get("criteria") or {}).items():
        if criterion_base(cid) in deferred:
            continue
        if c.get("pass") is not True or c.get("measured") is False:
            out[cid] = c
    return out


def verdict_passes(verdict: Optional[dict], deferred: Sequence[str] = ()) -> bool:
    """Whether a verdict passes: no failing criterion outside ``deferred``; with nothing deferred also ``pass`` True.

    :param verdict: Output of ``eval_gate.verdict`` (None = not measured = fail).
    :param deferred: Base ids evaluated later.
    :returns: True when it passes.
    :rtype: bool
    """
    if not verdict or not verdict.get("criteria"):
        return False
    if failing_criteria(verdict, deferred):
        return False
    return bool(deferred) or verdict.get("pass") is True


def criterion_passes(verdict: dict, cid: str) -> Optional[bool]:
    """Pass state of every criterion of one ``H<n>`` (None when the verdict has none).

    :param verdict: Output of ``eval_gate.verdict``.
    :param cid: Base id (``H0``).
    :returns: True / False / None.
    :rtype: Optional[bool]
    """
    crits = [c for k, c in (verdict.get("criteria") or {}).items() if criterion_base(k) == cid]
    if not crits:
        return None
    return all(c.get("pass") is True and c.get("measured") is not False for c in crits)


def h1_mean_delta_vs_b(verdict: dict) -> Optional[float]:
    """Mean H1 delta vs B of a verdict (the selection score of plan E).

    Read from the H1 criterion's ``detail`` or ``value``: a number under ``mean_delta_vs_B``; else the mean of the
    numbers under ``vs_B`` (a ``family -> delta`` or ``family -> {"delta": x}`` mapping); else, when the H1 criteria
    are split per reference (``H1_vs_B`` / ``H1.vs_B``), the mean of their numeric ``value`` s.

    :param verdict: Output of ``eval_gate.verdict``.
    :returns: The mean delta, or None when the verdict does not expose it.
    :rtype: Optional[float]
    """
    def num(x) -> Optional[float]:
        return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    def from_mapping(m) -> Optional[float]:
        if not isinstance(m, dict):
            return None
        direct = num(m.get("mean_delta_vs_B"))
        if direct is not None:
            return direct
        vs_b = m.get("vs_B")
        if isinstance(vs_b, dict):
            vals = [num(v.get("delta")) if isinstance(v, dict) else num(v) for v in vs_b.values()]
            vals = [v for v in vals if v is not None]
            if vals:
                return float(np.mean(vals))
        return None

    crits = {k: c for k, c in (verdict.get("criteria") or {}).items() if criterion_base(k) == "H1"}
    for c in crits.values():
        for src in (c.get("detail"), c.get("value")):
            got = from_mapping(src)
            if got is not None:
                return got
    split = [num(c.get("value")) for k, c in crits.items() if re.search(r"vs_?B$", k)]
    split = [v for v in split if v is not None]
    return float(np.mean(split)) if split else None


def selection_candidate(verdict: dict, ckpt: dict, run_dir: Optional[str] = None) -> dict:
    """One checkpoint's entry for :func:`select_checkpoint`.

    :param verdict: Its ``eval_gate.verdict``.
    :param ckpt: Its :func:`evaluate_checkpoints` index entry (``arm``, ``step``, ``fraction``).
    :param run_dir: Its report folder (for cross-arm selection).
    :returns: ``{verdict, arm, step, fraction, h1, run_dir}``.
    :rtype: dict
    """
    return {"verdict": verdict, "arm": ckpt.get("arm"), "step": ckpt.get("step"), "fraction": ckpt.get("fraction"),
            "h1": h1_mean_delta_vs_b(verdict), "run_dir": run_dir}


def select_checkpoint(candidates: Dict[str, dict], deferred: Sequence[str] = DEFERRED_CRITERIA,
                      tol: float = SELECTION_TOLERANCE) -> dict:
    """The pre-registered checkpoint choice (plan E).

    Among candidates whose verdict passes every hard criterion (``deferred`` ids aside: they are judged on the winner
    before any push), take the highest mean H1 delta vs B; every passing candidate within ``tol`` of the best is
    tied, and ties go to the earlier checkpoint (smaller fraction), then to arm ``lora``. When nothing passes, or a
    passing candidate's H1 delta cannot be read, ``selected`` is None and nothing may be pushed.

    :param candidates: ``label -> selection_candidate(...)``.
    :param deferred: Base criterion ids not required here.
    :param tol: Tie tolerance on the H1 delta.
    :returns: ``{selected, eligible, h1, failing: {label: [criterion ids]}, deferred, tolerance, reason}``.
    :rtype: dict
    """
    def failing_ids(verdict: Optional[dict]) -> List[str]:
        ids = sorted(failing_criteria(verdict or {}, deferred))
        if not ids and not verdict_passes(verdict, deferred):
            ids = ["verdict (empty or pass=False)"]
        return ids

    failing = {lb: failing_ids(c["verdict"]) for lb, c in candidates.items()}
    eligible = sorted(lb for lb, c in candidates.items() if verdict_passes(c["verdict"], deferred))
    out = {"selected": None, "eligible": eligible, "h1": {lb: c.get("h1") for lb, c in candidates.items()},
           "failing": {lb: f for lb, f in failing.items() if f}, "deferred": list(deferred), "tolerance": tol,
           "rule": "pass every hard criterion; max mean H1 delta vs B; within tolerance earlier, then lora"}
    if not eligible:
        out["reason"] = "no checkpoint passes every hard criterion"
        return out
    unreadable = [lb for lb in eligible if candidates[lb].get("h1") is None]
    if unreadable:
        out["reason"] = f"mean H1 delta vs B not readable from the verdict of {unreadable}"
        return out
    best = max(candidates[lb]["h1"] for lb in eligible)
    tied = [lb for lb in eligible if candidates[lb]["h1"] >= best - tol]
    pick = min(tied, key=lambda lb: (candidates[lb].get("fraction") or 1.0, candidates[lb].get("arm") != "lora",
                                     candidates[lb].get("step") or 0, lb))
    out.update({"selected": pick, "best_h1": best, "tied": sorted(tied), "reason": "selected"})
    return out


def select_across_runs(run_dirs: Sequence[str], deferred: Sequence[str] = DEFERRED_CRITERIA,
                       tol: float = SELECTION_TOLERANCE) -> dict:
    """:func:`select_checkpoint` over the checkpoints of several runs (the two arms), from their ``selection.json``.

    :param run_dirs: Report folders of the runs (one ``selection.json`` each, written by the notebook).
    :param deferred: Base criterion ids not required here.
    :param tol: Tie tolerance.
    :returns: :func:`select_checkpoint` output plus ``run_dirs`` (label -> folder) and ``missing_runs``.
    :rtype: dict
    """
    candidates, where, missing = {}, {}, []
    for rd in run_dirs:
        p = Path(rd, "selection.json")
        if not p.exists():
            missing.append(str(rd))
            continue
        for lb, c in json.loads(p.read_text())["candidates"].items():
            if lb in candidates:
                raise ValueError(f"checkpoint label {lb} appears in two runs")
            candidates[lb] = c
            where[lb] = str(rd)
    out = select_checkpoint(candidates, deferred, tol)
    out.update({"run_dirs": where, "missing_runs": missing,
                "candidates": {lb: {k: v for k, v in c.items() if k != "verdict"} for lb, c in candidates.items()}})
    return out


def format_verdict(verdict: dict, cand: str = "", refs: Optional[Dict[str, str]] = None,
                   deferred: Sequence[str] = ()) -> str:
    """Text block of a gate verdict: overall result and every criterion (NOT MEASURED shown as such, = FAIL).

    :param verdict: Output of ``eval_gate.verdict``.
    :param cand: Candidate label.
    :param refs: Reference roles -> labels.
    :param deferred: Base ids shown as deferred.
    :returns: Text.
    :rtype: str
    """
    ok = verdict_passes(verdict, deferred)
    head = f"GATE VERDICT {cand} vs {refs or {}}: {'PASS' if ok else 'FAIL'}"
    if deferred:
        head += f" (deferred: {', '.join(deferred)})"
    lines = [head]
    for cid, c in sorted((verdict.get("criteria") or {}).items(), key=lambda kv: (len(criterion_base(kv[0])), kv[0])):
        if criterion_base(cid) in deferred:
            state = "deferred"
        elif c.get("measured") is False:
            state = "NOT MEASURED"
        else:
            state = "pass" if c.get("pass") is True else "FAIL"
        detail = json.dumps(c.get("detail"), ensure_ascii=False, default=_json_default)
        lines.append(f"  {cid:<12} {state:<13} value={json.dumps(c.get('value'), default=_json_default)[:80]} "
                     f"threshold={json.dumps(c.get('threshold'), default=_json_default)[:60]} {detail[:160]}")
    return "\n".join(lines)


def format_selection(selection: dict) -> str:
    """Text of a :func:`select_checkpoint` / :func:`select_across_runs` result.

    :param selection: Selection dict.
    :returns: Text.
    :rtype: str
    """
    lines = [f"SELECTION: {selection.get('selected') or 'NONE -- do not push'} ({selection.get('reason')})",
             f"  rule: {selection.get('rule')}; deferred {selection.get('deferred')}; tolerance "
             f"{selection.get('tolerance')}"]
    for lb in sorted(selection.get("h1") or {}):
        fails = (selection.get("failing") or {}).get(lb)
        lines.append(f"  {lb:<16} H1 mean delta vs B {selection['h1'][lb]}  "
                     f"{'eligible' if lb in selection.get('eligible', []) else 'failing: ' + ', '.join(fails or [])}")
    if selection.get("missing_runs"):
        lines.append(f"  runs without a selection.json yet: {selection['missing_runs']}")
    return "\n".join(lines)


def gate_headlines(report: Optional[dict], max_depth: int = 2) -> Dict[str, float]:
    """Numeric leaves of a gate report up to ``max_depth`` levels (per-query lists skipped), for a side-by-side view.

    :param report: Gate report (``res["gate"]``).
    :param max_depth: Nesting depth.
    :returns: ``"a/b" -> value``.
    :rtype: Dict[str, float]
    """
    out: Dict[str, float] = {}

    def walk(node, path: Tuple[str, ...]) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            out["/".join(path)] = float(node)
        elif isinstance(node, dict) and len(path) < max_depth:
            for k, v in node.items():
                if "per_query" not in str(k) and "queries" != str(k):
                    walk(v, path + (str(k),))

    walk(report or {}, ())
    return out


def format_gate_table(gate_reports: Dict[str, Optional[dict]], max_depth: int = 2) -> str:
    """Side-by-side numeric headlines of several gate reports (:func:`gate_headlines`).

    :param gate_reports: ``label -> gate report``.
    :param max_depth: Nesting depth.
    :returns: Text.
    :rtype: str
    """
    heads = {lb: gate_headlines(r, max_depth) for lb, r in gate_reports.items()}
    keys = sorted(set().union(*heads.values())) if heads else []
    if not keys:
        return "(no numeric gate headlines)"
    width = max(len(k) for k in keys) + 2
    col = max(10, max(len(lb) for lb in heads) + 2)
    lines = [f"{'gate':<{width}}" + "".join(f"{lb:>{col}}" for lb in heads)]
    for k in keys:
        lines.append(f"{k:<{width}}" + "".join(f"{'-':>{col}}" if k not in heads[lb] else f"{heads[lb][k]:>{col}.4f}"
                                               for lb in heads))
    return "\n".join(lines)


def load_extra(extra_dir: str) -> Dict[str, dict]:
    """Verdict extras written outside the training session, one ``<criterion>.json`` per file (``H0.json`` from the
    ablation session, ``H10.json`` from ``bib_v9_check``, ``H12.json`` from the leave-concept-out session).

    :param extra_dir: Folder (Drive).
    :returns: ``criterion id -> content``.
    :rtype: Dict[str, dict]
    """
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(extra_dir).glob("H*.json"))}


def h12_summary(comparison: dict, reports: Dict[str, dict], candidate: str, baseline: str,
                floor: float = H12_AP_FLOOR) -> dict:
    """H12 (leave-concept-out) from a comparison on the LCO eval dir: held-out subject AP vs B, CI low > 0 and AP >=
    ``floor`` - 1 SE (SE approximated by the paired CI half-width / 1.96).

    :param comparison: :func:`compare_reports` with ``baseline`` = B on the LCO eval dir.
    :param reports: The reports compared.
    :param candidate: LCO-trained model label.
    :param baseline: B label.
    :param floor: AP floor (plan E: 0.182).
    :returns: ``{measured, pass, value, threshold, detail}``.
    :rtype: dict
    """
    ci = comparison["paired"].get(f"{candidate} - {baseline} | held_out AP")
    ap = ((reports[candidate].get("subjects") or {}).get("held_out") or {}).get("macro", {}).get("AP")
    if not ci or not ci.get("n_clusters") or ap is None:
        return {"measured": False, "pass": None, "value": None, "threshold": {"ci_low_gt": 0.0, "AP_floor": floor},
                "detail": "no held-out subject comparison in this eval dir"}
    se = (ci["ci_high"] - ci["ci_low"]) / (2 * 1.96)
    ok = ci["ci_low"] > 0 and ap >= floor - se
    return {"measured": True, "pass": bool(ok),
            "value": {"AP": ap, "delta_vs_B": ci["delta"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                      "se": round(se, 4)},
            "threshold": {"ci_low_gt": 0.0, "AP_ge": round(floor - se, 4), "AP_floor": floor},
            "detail": {"candidate": candidate, "baseline": baseline, "n_subjects": ci["n_clusters"]}}


# ---------------------------------------------------------------------------------------------------------------
# reports + push
# ---------------------------------------------------------------------------------------------------------------
def save_reports(model_dir: str, reports: Dict[str, dict], comparison: dict, meta: Optional[dict] = None) -> Path:
    """Write the eval reports next to the weights (``<model_dir>/genizah_eval/``).

    :param model_dir: Saved model directory.
    :param reports: ``label -> full eval result``.
    :param comparison: Output of :func:`compare_reports`.
    :param meta: Run metadata (run name, data revision ...).
    :returns: The report directory.
    :rtype: Path
    """
    import eval_v3

    out = Path(model_dir) / "genizah_eval"
    out.mkdir(parents=True, exist_ok=True)
    for lb, res in reports.items():
        (out / f"{lb}.json").write_text(json.dumps(res, ensure_ascii=False, default=_json_default))
    summary = {"meta": meta or {}, "summaries": {lb: eval_v3.summary(r) for lb, r in reports.items()},
               "comparison": comparison}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=_json_default))
    (out / "comparison.txt").write_text(format_table(comparison) + "\n")
    return out


def push(model_dir: str, repo_id: str, token: str, reports: Dict[str, dict], comparison: dict, run_name: str,
         data_revision: str, meta_extra: Optional[dict] = None, verdict: Optional[dict] = None,
         require_pass: bool = False) -> str:
    """Upload the model and its eval reports to a private HF model repo, tag the commit; return the commit sha.

    :param model_dir: Saved model directory (``training_info.json`` already inside).
    :param repo_id: Target model repo.
    :param token: HF token.
    :param reports: ``label -> full eval result``.
    :param comparison: Output of :func:`compare_reports`.
    :param run_name: Run name (also the git tag; pin the sha in the embedding contract).
    :param data_revision: Dataset tag the run trained on.
    :param meta_extra: More run metadata for ``summary.json`` (e.g. the previous model scored alongside).
    :param verdict: The model's final gate verdict (nothing deferred); recorded in ``summary.json``.
    :param require_pass: Refuse unless ``verdict`` passes every criterion (v10: always True).
    :returns: Commit sha.
    :rtype: str
    :raises ValueError: When ``require_pass`` and the verdict is missing or fails, or the tag exists.
    """
    from huggingface_hub import HfApi

    if require_pass and not verdict_passes(verdict):
        raise ValueError(f"gate verdict does not pass; NOT pushing. Failing: {sorted(failing_criteria(verdict or {}))}")
    meta = {"run_name": run_name, "data_revision": data_revision, "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION, **(meta_extra or {})}
    if verdict is not None:
        meta["gate_verdict"] = verdict
    save_reports(model_dir, reports, comparison, meta)
    api = HfApi(token=token)
    api.create_repo(repo_id, private=True, exist_ok=True)
    # create_tag(exist_ok=True) would silently leave an existing tag on the OLD commit: refuse a reused run name
    if any(t.name == run_name for t in api.list_repo_refs(repo_id).tags):
        raise ValueError(f"tag {run_name!r} already exists on {repo_id}; set a new RUN_NAME")
    info = api.upload_folder(folder_path=model_dir, repo_id=repo_id,
                             commit_message=f"genizah embedder {run_name} (data {data_revision})")
    api.create_tag(repo_id, tag=run_name, revision=info.oid)
    return info.oid


# ---------------------------------------------------------------------------------------------------------------
# smoke helpers + self-test
# ---------------------------------------------------------------------------------------------------------------
def tiny_model(hidden: int = 64, layers: int = 2, seed: int = 0, max_seq: int = TRAIN_MAX_SEQ):
    """A random-init 2-layer Qwen3 SentenceTransformer with the real Qwen3-Embedding tokenizer (smoke tests only).

    Same module stack AND input path as the base model: Transformer -> last-token Pooling -> Normalize, with the
    Transformer's modality config pinned to plain text (:data:`TEXT_MODALITY`). The hub checkpoint loads that way
    (its legacy ST config has no modality config), so prompts are prepended as plain text and ``<|endoftext|>`` is
    appended. A Transformer built from a local folder instead INFERS its modalities, finds the tokenizer's chat
    template and wraps every input in ``<|im_start|>user ...`` with the prompt as a system message -- a different
    tokenisation from production, which would make the smoke tests exercise the wrong path.

    :param hidden: Hidden size.
    :param layers: Decoder layers.
    :param seed: Init seed.
    :param max_seq: Max sequence length.
    :returns: SentenceTransformer on CPU.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize, Pooling, Transformer
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory(prefix="tiny_qwen3_") as tmp:  # weights + tokenizer are in memory once loaded
        Qwen3Model(cfg).save_pretrained(tmp)
        tok.save_pretrained(tmp)
        module = Transformer(tmp, max_seq_length=max_seq, model_kwargs={"dtype": torch.float32},
                             modality_config=TEXT_MODALITY, module_output_name="token_embeddings")
        model = SentenceTransformer(modules=[module, Pooling(hidden, pooling_mode="lasttoken"), Normalize()],
                                    device="cpu")
    check_text_path(model)
    return model


def check_text_path(model) -> None:
    """Assert the production tokenisation contract: prompt prepended as plain text, ``<|endoftext|>`` appended,
    no chat template (see :func:`tiny_model`).

    :param model: SentenceTransformer (base, tuned or tiny).
    :raises AssertionError: When the model would tokenise differently from production.
    """
    probe = "laws of lulav"
    ids = model.preprocess([probe], prompt=QUERY_PROMPT)["input_ids"][0]
    text = model.tokenizer.decode(ids)
    want = QUERY_PROMPT + probe + "<|endoftext|>"
    if text != want:
        raise AssertionError(f"tokenisation differs from the production contract: {text!r} != {want!r}")
    doc = model.tokenizer.decode(model.preprocess(["Description: x"])["input_ids"][0])
    if doc != "Description: x<|endoftext|>":
        raise AssertionError(f"document tokenisation differs from the production contract: {doc!r}")


def _self_test_rows() -> Tuple[List[dict], Dict[str, dict]]:
    """Hand-made rows + pool exercising every masking rule.

    :returns: (rows, pool).
    :rtype: Tuple[List[dict], Dict[str, dict]]
    """
    def rec(d: str, subjects: List[str]) -> dict:
        return {"doc_id": d, "text": f"text of {d}", "text_hash": hashlib.md5(d.encode()).hexdigest(),
                "subjects": subjects}

    pool = {r["doc_id"]: r for r in (rec("A", ["pgp:x", "domain:letters"]), rec("B", ["pgp:x"]),
                                     rec("C", ["domain:letters"]), rec("C2", ["domain:letters"]), rec("N", ["pgp:x"]),
                                     rec("E", []),
                                     rec("F", ["pgp:y"]), rec("G", ["pgp:y", "pgp:z"]), rec("H", ["pgp:z"]))}

    def q2d(d: str, family: str, level: str, **extra) -> dict:
        r = pool[d]
        return {"anchor": f"query for {d} {family}", "positive": r["text"], "family": family, "kind": "q2d",
                "mask_level": level, "doc_id": d, "mask_ids": r["subjects"] + [f"D:{d}"], "anchor_mask_ids": [],
                "text_hash": r["text_hash"], **extra}

    rows = [q2d("A", "synthetic_implicit", "subject"),                        # 0: relevance {pgp:x}
            q2d("B", "subject", "subject", anchor_subject="pgp:x"),           # 1: shares pgp:x with row 0
            q2d("N", "synthetic_specific", "doc"),                            # 2: doc-level, pgp:x positive
            q2d("C", "synthetic_topical", "subject"),                         # 3: only a broad (domain) subject
            q2d("C2", "subject", "subject", anchor_subject="domain:letters"),  # 4: domain anchor -> masks A, C
            q2d("E", "content", "doc")]                                       # 5: no subjects
    rows[5]["positive"] = "label part of E"
    d2d = [{"anchor": pool[a]["text"], "positive": pool[b]["text"], "family": "d2d_cross_genre", "kind": "d2d",
            "mask_level": "subject", "doc_id": b, "mask_ids": pool[b]["subjects"] + [f"D:{b}"],
            "anchor_mask_ids": pool[a]["subjects"] + [f"D:{a}"], "anchor_doc_id": a, "text_hash": pool[b]["text_hash"],
            "shared_subject": s} for a, b, s in (("F", "G", "pgp:y"), ("H", "B", "none"), ("E", "C", "none"))]
    t2t = [{"anchor": a, "positive": p, "family": "t2t_xling", "kind": "t2t", "mask_level": "subject", "doc_id": None,
            "mask_ids": [], "anchor_mask_ids": [], "concept_ids": c}
           for a, p, c in (("Sukkah", "סוכה", ["sefaria:s1"]), ("Sukka booth", "Sukkah", ["sefaria:s1"]),
                           ("Lulav", "לולב", ["sefaria:s2"]))]
    return rows + d2d + t2t, pool


def _fake_holiday(weight: float, rng: np.random.Generator, festivals: Sequence[str] = ("topic:pesach", "topic:sukkot",
                                                                                       "topic:purim")) -> dict:
    """A holiday section in the eval_v3 layout whose quality grows with ``weight`` (compare tests).

    :param weight: Model strength.
    :param rng: RNG.
    :param festivals: Festival ids.
    :returns: ``res["holiday"]``-shaped dict with a held-out variant.
    :rtype: dict
    """
    def section(scale: float) -> dict:
        per_query = {}
        for f in festivals:
            for j, form in enumerate(("english", "hebrew", "translit")):
                p10 = float(np.clip(0.2 + scale * weight + rng.normal(0, 0.02), 0, 1))
                per_query[f"{f}|{f} q{j}"] = {"festival": f, "form": form, "text": f"{f} q{j}", "P@10": p10,
                                              "P@20": p10, "MRR@100": min(1.0, p10 + 0.1),
                                              "canon_share@20": max(0.0, 0.3 - 0.1 * weight)}
        per_festival = {}
        for f in festivals:
            rows = [r for r in per_query.values() if r["festival"] == f]
            per_festival[f] = {k: float(np.mean([r[k] for r in rows])) for k in ("P@10", "MRR@100", "canon_share@20")}
            per_festival[f]["n_queries"] = len(rows)
        macro = {k: float(np.mean([r[k] for r in per_query.values()])) for k in ("P@10", "MRR@100", "canon_share@20")}
        macro.update({"alt_only_lift": 1.0 + 0.1 * weight, "n_queries": len(per_query)})
        worst = min(per_festival.items(), key=lambda kv: kv[1]["P@10"])
        return {"macro": macro, "per_festival": per_festival, "worst_festival": {"id": worst[0], **worst[1]},
                "per_query": per_query}

    out = section(0.4)
    out["held_out"] = section(0.3)
    return out


def _self_test_compare(final: bool, seed: int = 0) -> dict:
    """:func:`compare_reports` on a tiny synthetic eval dir: v3 layout (gate) or final layout (focus, no held-out).

    Four "models" of increasing signal stand for base_original, base_semantic, a previous tuned model and the
    candidate; checks the primary group, the table layout, the ``also_vs`` CIs, the holiday rows and their
    festival-clustered paired CIs, and a known-item comparison against a model that lost queries (ignore_ids).

    :param final: Final-run layout (``focus`` + ``seen`` statuses, no held-out subject).
    :param seed: RNG seed.
    :returns: The comparison.
    :rtype: dict
    """
    import eval_v3

    rng = np.random.default_rng(seed)
    d, n = 16, 300
    sids = ["topic:a", "pgp:b", "work:c", "domain:d"]
    status = dict(zip(sids, ("focus", "focus", "seen", "seen") if final else ("held_out", "held_out", "linked",
                                                                               "seen")))
    centres = eval_v3.normalise_rows(rng.normal(size=(4, d)))
    ids = [f"S{i % 5}_{i}" for i in range(n)]
    label = rng.integers(-1, 4, size=n)
    held = [ids[i] for i in range(n) if i % 3 == 0]
    strong = {s: [ids[i] for i in range(n) if label[i] == k] for k, s in enumerate(sids)}
    ev = {"held_out_subjects": {"subjects": []}, "held_out_records": {"held_out": held},
          "subject_queries": {"subjects": {s: {"status": status[s], "kind": s.split(":")[0],
                                               "queries": [{"text": f"{s} q{j}", "source": ("probe", "name")[j % 2],
                                                            "lang": "en", "rel_mode": ("law", "bare")[j % 2]}
                                                           for j in range(4)]} for s in sids}},
          "subject_relevance": {s: {"strong": strong[s], "weak": [], "canon": [], "unjudged": []} for s in sids},
          "known_item": [{"qid": f"k{i}", "doc_id": ids[i], "type": "implicit", "text": f"known {i}",
                          "shelfmark_query": False} for i in range(0, 60, 6)],
          "memorization_pairs": {"subjects": {"domain:d": {"train": [x for x in strong["domain:d"] if x not in held][:6],
                                                           "held_out": [x for x in strong["domain:d"] if x in held][:6]}}},
          "collection_sample": {"doc_ids": ids[:60]},
          "eval_pool": [{"doc_id": x, "text_hash": f"h{i}", "series": f"S{i % 5}", "n_chars": 300, "dup_n": 1,
                         "held_out": x in held} for i, x in enumerate(ids)]}
    base = eval_v3.normalise_rows(rng.normal(size=(n, d)))
    qtext = {f"{s} q{j}": centres[k] for k, s in enumerate(sids) for j in range(4)}
    reports = {}
    with tempfile.TemporaryDirectory() as tmp:
        eval_v3._write_eval_dir(Path(tmp), ev)
        for lb, w in (("base_original", 0.0), ("base_semantic", 0.3), ("prev", 0.6), ("tuned", 1.2)):
            mat = base.copy()
            for i in range(n):
                if label[i] >= 0:
                    mat[i] = base[i] + w * centres[label[i]]
            q = dict(qtext, **{f"known {i}": mat[i] for i in range(0, 60, 6)})
            ignore = [ids[0]] if lb == "prev" else None  # loses known-item query k0 (V-style ignore_ids)
            reports[lb] = eval_v3.evaluate(ids, mat, lambda t, q=q: np.stack([q[x] for x in t]), tmp,
                                           n_resamples=50, ignore_ids=ignore)
            reports[lb]["holiday"] = _fake_holiday(w, rng)
    del reports["prev"]["holiday"]["per_query"]["topic:purim|topic:purim q0"]  # one query fewer: paired on the rest
    comp = compare_reports(reports, baseline="base_semantic", candidate="tuned", n_resamples=100, also_vs=["prev"])
    metrics = [r["metric"] for r in comp["table"]]
    pg = "focus" if final else "held_out"
    assert comp["primary_group"] == pg and comp["primary_name"] == f"tuned - base_semantic | {pg} AP", comp
    assert comp["primary"] is not None and comp["primary"]["n_clusters"] == 2 and comp["primary"]["delta"] > 0
    assert any(m.startswith("FOCUS subjects") for m in metrics) == final
    assert any(m.startswith("GATE held-out") for m in metrics) != final
    assert f"tuned - prev | {pg} AP" in comp["paired"] and f"prev - base_semantic | {pg} AP" in comp["paired"]
    assert "tuned - prev | topic:a AP" in comp["paired"] and "tuned - base_semantic | pgp:b AP" in comp["paired"]
    assert f"base_semantic - base_original | {pg} AP (step 0)" in comp["paired"]
    # holiday rows + festival-clustered paired CIs (finding 16)
    for m in ("holiday macro P@10", "holiday macro MRR@100", "holiday macro canon_share@20 (lower=better)",
              "  holiday worst festival P@10", "  holiday topic:sukkot P@10", "holiday held-out macro P@10"):
        assert m in metrics, (m, [x for x in metrics if "holiday" in x])
    hol = comp["paired"]["tuned - base_semantic | holiday P@10"]
    assert hol["n_clusters"] == 3 and hol["n_queries"] == 9 and hol["delta"] > 0, hol
    assert "tuned - base_semantic | holiday held-out MRR@100" in comp["paired"]
    assert comp["paired"]["tuned - prev | holiday P@10"]["n_queries"] == 8, comp["paired"]["tuned - prev | holiday P@10"]
    ki = comp["paired"]["tuned - prev | known_item RR"]
    assert ki["n_queries"] == len(reports["prev"]["known_item"]["per_query"]["qid"]) == 9, ki
    if final:
        assert not any("| held_out" in k or "| linked" in k for k in comp["paired"]), list(comp["paired"])
        assert any(m.startswith("    topic:a probe AP") for m in metrics)
    format_table(comp)
    return comp


def _fake_verdict(passing: Dict[str, bool], h1: Optional[float]) -> dict:
    """A verdict in the ``eval_gate.verdict`` layout (selection tests).

    :param passing: ``criterion id -> pass`` (False = fail; missing ``measured`` defaults to measured).
    :param h1: Mean H1 delta vs B exposed under ``criteria["H1"]["detail"]["mean_delta_vs_B"]`` (None: not exposed).
    :returns: Verdict dict.
    :rtype: dict
    """
    crit = {cid: {"pass": ok, "measured": True, "value": None, "threshold": None, "detail": {}}
            for cid, ok in passing.items()}
    if "H1" in crit and h1 is not None:
        crit["H1"]["detail"] = {"mean_delta_vs_B": h1}
    return {"pass": all(passing.values()), "criteria": crit}


def check_festival_topics() -> Optional[bool]:
    """Compare :data:`FESTIVAL_TOPICS` with ``build_subjects.FESTIVAL_TOPICS`` when that module is importable (repo
    checkout; not on Colab).

    :returns: True when equal (as sets), False when they differ, None when build_subjects is not importable.
    :rtype: Optional[bool]
    """
    import importlib.util

    if importlib.util.find_spec("build_subjects") is None:
        return None
    import build_subjects

    return set(build_subjects.FESTIVAL_TOPICS) == set(FESTIVAL_TOPICS)


def self_test_pure(strict_festivals: bool = False) -> dict:
    """Torch-free unit tests (runnable with torch / sentence-transformers stubbed): segments with ``M:`` ids and
    festival tags, the C5 negative policy and its rejection log, padding trim, label layout, holiday / known-item
    comparisons, checkpoint selection, the saved-model contract check, the model card, recipes and the v9-trained
    lookup.

    :param strict_festivals: Fail when :data:`FESTIVAL_TOPICS` differs from an importable ``build_subjects``.
    :returns: Summary of what ran.
    :rtype: dict
    """
    def rec(d: str, subjects: List[str], text: Optional[str] = None) -> dict:
        t = text if text is not None else f"Description: record {d} about {d.lower()} things"
        return {"doc_id": d, "text": t, "text_hash": hashlib.md5(t.encode()).hexdigest(), "subjects": subjects}

    # 1. segments: subject rows keep anchor_subject + anchor_mention_id + anchor_topic_ids; M: ids are broad
    pool = {r["doc_id"]: r for r in (
        rec("S1", ["topic:sukkot", "pgp:x", "domain:letters", "M:pgp:y"]), rec("S2", ["M:topic:sukkot"]),
        rec("S3", ["topic:sukkot"]), rec("Y1", ["M:pgp:y"]), rec("Z1", ["pgp:z"]), rec("Q1", ["domain:letters"]))}

    def q2d(d: str, family: str, level: str, **extra) -> dict:
        r = pool[d]
        return {"anchor": f"query {d}", "positive": r["text"], "family": family, "kind": "q2d", "mask_level": level,
                "doc_id": d, "mask_ids": r["subjects"] + [f"D:{d}"], "anchor_mask_ids": [],
                "text_hash": r["text_hash"], **extra}

    subj_row = q2d("S1", "subject", "subject", anchor_subject="pgp:x", anchor_mention_id="M:pgp:x",
                   anchor_topic_ids=["topic:sukkot", "M:topic:sukkot"])
    seg0, seg1, seg2 = row_segments(subj_row, pool)
    assert {"pgp:x", "M:pgp:x", "topic:sukkot", "M:topic:sukkot", "D:S1"} <= set(seg0), seg0
    assert "domain:letters" not in seg0 and "M:pgp:y" not in seg0, seg0
    assert "M:pgp:y" in seg1 and "domain:letters" in seg1 and seg2 == []
    doc_row = q2d("S3", "synthetic_specific", "doc", anchor_topic_ids=["M:topic:sukkot", "topic:sukkot"])
    assert row_segments(doc_row, pool)[0] == ["D:S3", f"T:{pool['S3']['text_hash']}", "M:topic:sukkot",
                                              "topic:sukkot"]
    plain_doc = q2d("Z1", "synthetic_specific", "doc")
    assert row_segments(plain_doc, pool)[0] == ["D:Z1", f"T:{pool['Z1']['text_hash']}"]
    d2d = {"anchor": pool["Y1"]["text"], "positive": pool["Z1"]["text"], "family": "d2d_cross_genre", "kind": "d2d",
           "mask_level": "subject", "doc_id": "Z1", "mask_ids": ["pgp:z", "D:Z1"], "anchor_doc_id": "Y1",
           "anchor_mask_ids": ["M:pgp:y", "D:Y1"], "text_hash": pool["Z1"]["text_hash"]}
    assert "M:pgp:y" not in row_segments(d2d, pool)[0]
    # pure-python mask simulation (seg0_i x seg1_j): the festival-tagged subject row masks the M:topic:sukkot record
    rows = [subj_row, q2d("S2", "synthetic_topical", "subject"), q2d("Y1", "subject", "subject",
                                                                       anchor_subject="pgp:y",
                                                                       anchor_mention_id="M:pgp:y"),
            q2d("Z1", "synthetic_implicit", "subject")]
    segs = [row_segments(r, pool) for r in rows]
    masked = {(i, j) for i in range(len(rows)) for j in range(len(rows)) if i != j and set(segs[i][0]) & set(segs[j][1])}
    assert (0, 1) in masked, masked                 # S2 names Sukkot (M:topic:sukkot) -> no negative for row 0
    assert (2, 0) in masked                          # row 2 anchors pgp:y; S1 mentions it (M:pgp:y)
    assert (1, 0) not in masked and (3, 0) not in masked and (0, 3) not in masked, masked
    width = max(len(s) for sg in segs for s in sg)
    lab = label_rows(segs, width, [0, 1, 2, 3])
    arr = np.array(lab)
    assert arr.shape == (4, N_SEGMENTS * width + 1)
    assert [s.shape for s in split_label(arr)] == [(4, width)] * 3 and label_family(arr).tolist() == [0, 1, 2, 3]
    assert label_family(arr[:, :-1]) is None and split_label(arr[:, :-1])[2].shape == (4, width)

    # 2. C5 negatives
    neg_pool = {r["doc_id"]: r for r in (
        rec("P", ["topic:pesach", "pgp:h"], "Description: Passover Haggadah with illuminations\nTranscription: x y z"),
        rec("F1", ["topic:pesach"], "Description: Seder liturgy fragment"),
        rec("F2", ["M:topic:pesach", "pgp:h"], "Description: letter mentioning Passover"),
        rec("H1", ["pgp:h"], "Description: Haggadah commentary in Judaeo-Arabic"),
        rec("ND", ["pgp:w"], "Description: Passover Haggadah with illuminations\nTranscription: other text"),
        rec("O1", ["pgp:o"], "Description: account list of wheat"),
        rec("O2", [], "Description: marriage contract"),
        rec("EM", ["pgp:o"], "Transcription: only a transcription"))}

    def nrow(family: str, level: str, **extra) -> dict:
        r = neg_pool["P"]
        return {"anchor": "a query", "positive": r["text"], "family": family, "kind": "q2d", "mask_level": level,
                "doc_id": "P", "mask_ids": r["subjects"] + ["D:P"], "anchor_mask_ids": [], "text_hash": r["text_hash"],
                **extra}

    ranked = ["P", "F1", "F2", "H1", "ND", "O1", "O2", "EM"]
    rej: Counter = Counter()
    picks = {choose_negative(nrow("synthetic_specific", "doc"), ranked, neg_pool, random.Random(s), 0, 10,
                             rejections=rej) for s in range(10)}
    assert picks == {("H1", "same_subject")}, picks  # F1 / F2 festival-banned, ND near-duplicate, H1 shares pgp:h
    assert rej["festival"] == 20 and rej["festival:topic:pesach"] == 20 and rej["near_duplicate"] == 10, rej
    assert rej["banned_doc"] == 10 and token_jaccard(label_tokens(neg_pool["P"]["text"]),
                                                     label_tokens(neg_pool["ND"]["text"])) == 1.0
    heb = {choose_negative(nrow("synthetic_hebrew", "doc"), ranked, neg_pool, random.Random(s), 0, 10)
           for s in range(20)}
    assert all(how == "any" for _, how in heb) and {d for d, _ in heb} <= {"H1", "O1", "O2", "EM"}, heb
    v3 = {choose_negative(nrow("synthetic_hebrew", "doc"), ranked, neg_pool, random.Random(s), 0, 10, policy="v3")
          for s in range(20)}
    assert {how for _, how in v3} == {"same_subject"} and {d for d, _ in v3} == {"F1", "F2", "H1"}, v3  # v9 mining
    sub_rej: Counter = Counter()
    sub = choose_negative(nrow("synthetic_implicit", "subject"), ranked, neg_pool, random.Random(0), 0, 10,
                          rejections=sub_rej)
    assert sub[1] == "any" and sub_rej["relevance"] == 3 and not sub_rej["festival"] and not sub_rej["near_duplicate"]
    content_rej: Counter = Counter()
    choose_negative(nrow("content", "doc"), ["EM", "O1"], neg_pool, random.Random(0), 0, 10, rejections=content_rej)
    assert content_rej["content_empty"] == 1, content_rej
    assert choose_negative(nrow("synthetic_specific", "doc"), ["P", "F1"], neg_pool, random.Random(0), 0, 10) == \
        (None, "none")
    stats = mining_stats([dict(nrow("synthetic_specific", "doc"), negative_choice="same_subject",
                               negative_rejections=dict(rej)),
                          dict(nrow("synthetic_implicit", "subject", anchor_subject="pgp:h"), negative_choice="any",
                               negative_rejections=dict(sub_rej))])
    assert stats["rejections"]["by_subject"]["festival"] == {"topic:pesach": 20}, stats
    assert stats["rejections"]["by_subject"]["near_duplicate"] == {"pgp:h": 10, "topic:pesach": 10}, stats
    assert stats["rejections"]["by_subject"]["relevance"] == {"pgp:h": 3} and "festival:topic:pesach" not in \
        stats["rejections"]["total"], stats
    assert stats["total"] == {"same_subject": 1, "any": 1}

    # 3. padding trim (left and right padding)
    left = {"input_ids": np.arange(24).reshape(4, 6),
            "attention_mask": np.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]]),
            "modality": "text", "prompt_length": 3}
    mb, before, after = trim_minibatch(left, 0, 2)
    assert mb["input_ids"].tolist() == [[2, 3, 4, 5], [8, 9, 10, 11]] and (before, after) == (12, 8)
    assert mb["modality"] == "text" and mb["prompt_length"] == 3
    mb, before, after = trim_minibatch(left, 2, 10)
    assert mb["input_ids"].shape == (2, 6) and (before, after) == (12, 12)
    right = {"input_ids": np.arange(12).reshape(2, 6), "attention_mask": np.array([[1, 1, 0, 0, 0, 0],
                                                                                    [1, 1, 1, 0, 0, 0]])}
    mb, _, after = trim_minibatch(right, 0, 2)
    assert mb["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]] and after == 6

    # 4. compare_reports with holiday rows (v3 + final layouts) and known-item / holiday intersections
    cmp_v3, cmp_final = _self_test_compare(final=False), _self_test_compare(final=True)

    # 5. checkpoint selection (plan E)
    crit = {"H1": True, "H3": True, "H0": False, "H10": False, "H12": False}
    cands = {"lora_s0051": {"verdict": _fake_verdict(crit, 0.020), "arm": "lora", "step": 51, "fraction": 0.25},
             "lora_s0102": {"verdict": _fake_verdict(crit, 0.024), "arm": "lora", "step": 102, "fraction": 0.5},
             "full_s0051": {"verdict": _fake_verdict(crit, 0.026), "arm": "full", "step": 51, "fraction": 0.25},
             "full_s0102": {"verdict": _fake_verdict(dict(crit, H3=False), 0.090), "arm": "full", "step": 102,
                            "fraction": 0.5}}
    for c in cands.values():
        c["h1"] = h1_mean_delta_vs_b(c["verdict"])
    sel = select_checkpoint(cands)
    assert sel["selected"] == "full_s0051" and sel["tied"] == ["full_s0051", "lora_s0102"], sel  # 0.026 vs 0.024
    assert sel["failing"]["full_s0102"] == ["H3"] and "lora_s0051" not in sel["tied"], sel
    cands["lora_s0051"]["h1"] = 0.0215  # within 0.005 of 0.026 and earlier+lora -> wins
    cands["full_s0051"]["h1"] = 0.0260
    assert select_checkpoint(cands)["selected"] == "lora_s0051"
    cands["lora_s0051"]["fraction"] = 0.5  # same fraction as full? no: full_s0051 is 0.25 -> full wins
    assert select_checkpoint(cands)["selected"] == "full_s0051"
    assert select_checkpoint(cands, deferred=())["selected"] is None  # H0/H10/H12 fail when not deferred
    none = select_checkpoint({k: dict(v, verdict=_fake_verdict(dict(crit, H1=False), 0.1)) for k, v in cands.items()})
    assert none["selected"] is None and none["reason"].startswith("no checkpoint"), none
    unread = select_checkpoint({"x": {"verdict": _fake_verdict(crit, None), "arm": "lora", "step": 1, "fraction": 1.0,
                                      "h1": None}})
    assert unread["selected"] is None and "not readable" in unread["reason"], unread
    nm = _fake_verdict({"H1": True}, 0.01)
    nm["criteria"]["H4"] = {"pass": None, "measured": False}
    assert not verdict_passes(nm, DEFERRED_CRITERIA) and "H4" in failing_criteria(nm)
    assert criterion_base("H10") == "H10" and criterion_base("H1.G3") == "H1" and criterion_base("H1_vs_P") == "H1"
    assert abs(h1_mean_delta_vs_b({"criteria": {"H1": {"value": {"vs_B": {"G1": 0.01, "G2": {"delta": 0.03}}}}}})
               - 0.02) < 1e-12
    assert h1_mean_delta_vs_b({"criteria": {"H1_vs_B": {"value": 0.04}, "H1_vs_P": {"value": 0.0}}}) == 0.04
    assert criterion_passes(_fake_verdict(crit, 0.0), "H0") is False and criterion_passes({"criteria": {}}, "H0") is None
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ("lora", "full"):
            Path(tmp, arm).mkdir()
            Path(tmp, arm, "selection.json").write_text(json.dumps(
                {"candidates": {k: v for k, v in cands.items() if v["arm"] == arm}}))
        across = select_across_runs([str(Path(tmp, "lora")), str(Path(tmp, "full")), str(Path(tmp, "missing"))])
        assert across["selected"] == "full_s0051" and across["run_dirs"]["full_s0051"].endswith("full"), across
        assert across["missing_runs"] == [str(Path(tmp, "missing"))]
        format_selection(across)
    format_verdict(_fake_verdict(crit, 0.02), "lora_s0051", {"B": "B"}, DEFERRED_CRITERIA)

    # 6. saved-model contract check, model card, vectors, extras, v9-trained lookup
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp)
        (md / "config_sentence_transformers.json").write_text(json.dumps({"prompts": SAVED_PROMPTS}))
        (md / "tokenizer_config.json").write_text(json.dumps({"model_max_length": EVAL_MAX_SEQ}))
        (md / "tokenizer.json").write_text(json.dumps({"truncation": None, "padding": None, "model": {}}))
        assert check_saved_model(tmp)["model_max_length"] == EVAL_MAX_SEQ
        (md / "tokenizer.json").write_text(json.dumps({"truncation": {"max_length": 1024}, "padding": None}))
        bad = None
        try:  # the check must refuse a tokenizer.json that bakes in the training window
            check_saved_model(tmp)
        except AssertionError as exc:
            bad = str(exc)
        assert bad and "truncation" in bad, bad
        info = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION, "global_steps": 203,
                "recipe": resolve_recipe("lora"), "args": {"epochs": 1.0, "seed": 13},
                "windows": {"train_max_seq": TRAIN_MAX_SEQ, "mining_max_seq": EVAL_MAX_SEQ,
                            "eval_max_seq": EVAL_MAX_SEQ, "padding_trim": True},
                "run_meta": {"data_revision": "v4-final", "gate_thresholds_sha256": "abc"},
                "lora_merge": {"min_cosine": 0.99999}}
        (md / "README.md").write_text("# auto card\n")
        write_model_card(tmp, info)
        write_model_card(tmp, dict(info, selected_checkpoint={"step": 51, "fraction": 0.25}))
        card = (md / "README.md").read_text()
        assert card.count(MODEL_CARD_BEGIN) == 1 and "train 1024, mining 8192" in card and "step 51" in card, card
        assert card.startswith("# auto card")
        meta = save_vectors(tmp, "B", ["a", "b"], np.eye(2, 4, dtype=np.float32), {"q1": np.ones(4) / 2})
        assert np.load(md / "vectors_B.f16.npy").dtype == np.float16 and meta["n_queries"] == 1
        (md / "H12.json").write_text(json.dumps({"pass": True}))
        assert load_extra(tmp) == {"H12": {"pass": True}}
        fake = {"path": md}
        assert load_v9_trained(fake) is None
        (md / "eval").mkdir()
        (md / "eval" / "v9_trained.json").write_text(json.dumps({"doc_ids": ["b", "a", "a"]}))
        assert load_v9_trained(fake) == ["a", "b"]
        (md / "mix.jsonl").write_text("\n".join(json.dumps(r) for r in (
            {"doc_id": "x", "anchor_doc_id": None}, {"doc_id": "y", "anchor_doc_id": "z"}, {"doc_id": None})))
        assert v9_trained_from_mix(str(md / "mix.jsonl")) == ["x", "y", "z"]
        gd = {"eval_dir": str(md / "eval")}
        assert gate_dir(gd) is None
        (md / "eval" / GATE_SUBDIR).mkdir()
        (md / "eval" / GATE_SUBDIR / "thresholds.json").write_text('{"H1": {"delta": 0.03}}')
        thr, sha = gate_thresholds(gd)
        assert thr == {"H1": {"delta": 0.03}} and sha == hashlib.sha256(b'{"H1": {"delta": 0.03}}').hexdigest()
        (md / "ck_trainer" / "checkpoint-51").mkdir(parents=True)
        (md / "ck_trainer" / "checkpoint-102").mkdir()
        (md / "ck_trainer" / "runs").mkdir()
        assert [s for s, _ in list_checkpoints(str(md / "ck"))] == [51, 102]

    # 7. recipes, checkpoint ratio, loss breakdown, H12, corpus texts, festival list
    assert resolve_recipe("lora")["lr"] == 1e-4 and resolve_recipe("full")["lr"] == 1e-5
    assert resolve_recipe(None) == {"lr": 2e-5, "warmup_ratio": 0.05, "arm": None}
    assert resolve_recipe("full", lr=3e-6)["lr"] == 3e-6 and resolve_recipe("lora")["warmup_ratio"] == 0.10
    assert checkpoint_save_steps(CHECKPOINT_FRACTIONS) == 0.25 and checkpoint_save_steps(()) is None
    for bad_fr in ((0.3, 0.6, 1.0), (0.25, 0.5), (0.5, 1.0, 1.5)):
        err = None
        try:  # invalid fraction lists must be refused
            checkpoint_save_steps(bad_fr)
        except ValueError as exc:
            err = exc
        assert err is not None, bad_fr

    class _Hist:
        def __init__(self, history: List[dict]) -> None:
            self.history = history

    lb = loss_breakdown({"q2d_hn": _Hist([{"batch": 1, "loss": 4.0, "families": {"subject": [8.0, 2]}},
                                          {"batch": 4, "loss": 2.0, "families": {"subject": [2.0, 2],
                                                                                 "content": [3.0, 1]}}]),
                         "t2t": _Hist([{"batch": 2, "loss": 3.0}])}, total_steps=4)
    assert lb["by_dataset"]["q2d_hn"] == {"batches": 2, "mean": 3.0, "by_quarter": [4.0, None, None, 2.0]}, lb
    assert lb["by_family"]["subject"] == {"rows": 4, "mean": 2.5, "by_quarter": [4.0, None, None, 1.0]}, lb
    h12 = h12_summary({"paired": {"m - B | held_out AP": {"delta": 0.05, "ci_low": 0.01, "ci_high": 0.09,
                                                           "n_clusters": 6}}},
                      {"m": {"subjects": {"held_out": {"macro": {"AP": 0.17}}}}}, "m", "B")
    assert h12["measured"] and h12["pass"] and h12["threshold"]["AP_ge"] < 0.182, h12
    assert h12_summary({"paired": {}}, {"m": {"subjects": {}}}, "m", "B")["measured"] is False
    sd = {"semantic": {"a": {"text": "A4", "text_v3": "A3"}, "b": {"text": "B4", "text_v3": "B3"}},
          "original": {"a": "A8"}, "texts": load_texts(Path(tempfile.gettempdir()) / "no_such_dir_xyz",
                                                       {"a": {"text_v3": "A3"}, "b": {"text_v3": "B3"}})}
    assert corpus_texts(sd, "semantic_v3", ["a", "b"]) == (["a", "b"], ["A3", "B3"], "text_hash_v3")
    assert corpus_texts(sd, "original", ["a", "b"]) == (["a"], ["A8"], None)
    assert corpus_texts(sd, "semantic", ["a", "b"])[2] == "text_hash"
    assert festival_names(["topic:pesach", "M:topic:sukkot", "M:pgp:x", "topic:rosh_hodesh"]) == \
        {"pesach", "sukkot", "rosh_hodesh"}
    fest = check_festival_topics()
    assert fest is not False or not strict_festivals, "trainer FESTIVAL_TOPICS != build_subjects.FESTIVAL_TOPICS"
    return {"self_test_pure": "ok", "festival_topics_vs_build_subjects": fest,
            "compare_v3_primary": [cmp_v3["primary_name"], cmp_v3["primary"]["delta"]],
            "compare_final_primary": [cmp_final["primary_name"], cmp_final["primary"]["delta"]],
            "holiday_paired": cmp_final["paired"]["tuned - base_semantic | holiday P@10"],
            "selected": sel["selected"]}


def self_test() -> None:
    """Unit tests of the masking (pure torch + a tiny random model for the loss object; CPU, seconds), the padding
    trim's numerical equivalence, plus :func:`self_test_pure`."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # 1. id_overlap against brute-force sets, with padding
    for _ in range(100):
        n, m, k = rng.integers(1, 7), rng.integers(1, 7), rng.integers(1, 5)
        a = torch.tensor(rng.integers(-1, 6, size=(n, k)))
        b = torch.tensor(rng.integers(-1, 6, size=(m, k + 1)))
        want = torch.tensor([[bool({x for x in ra.tolist() if x >= 0} & {x for x in rb.tolist() if x >= 0})
                              for rb in b] for ra in a])
        assert torch.equal(id_overlap(a, b), want)
    assert not id_overlap(torch.full((2, 3), PAD_ID), torch.full((2, 3), PAD_ID)).any()

    # 2. segment semantics on hand-made rows
    rows, pool = _self_test_rows()
    rows[1]["negative_doc_id"] = "F"
    q = rows[:6]
    segs = [row_segments(r, pool) for r in q]
    assert "domain:letters" not in segs[0][0] and "pgp:x" in segs[0][0]
    assert segs[2][0] == ["D:N", f"T:{pool['N']['text_hash']}"]
    assert "domain:letters" in segs[4][0] and "domain:letters" in segs[0][1]
    assert segs[1][2] == record_identity("F", ["pgp:y"], pool["F"]["text_hash"]) and segs[0][2] == []
    # festival guard: a doc-level query naming a festival never mines another record of that festival
    hag = dict(pool, **{d: {"doc_id": d, "text": f"text of {d}", "text_hash": hashlib.md5(d.encode()).hexdigest(),
                            "subjects": ["topic:pesach"]} for d in ("P1", "P2")})
    hrow = {"anchor": "Passover Haggadah", "positive": hag["P2"]["text"], "family": "synthetic_specific",
            "kind": "q2d", "mask_level": "doc", "doc_id": "P2", "mask_ids": ["topic:pesach", "D:P2"],
            "anchor_mask_ids": [], "text_hash": hag["P2"]["text_hash"]}
    ranked = ["P2", "P1", "A", "E"]
    assert choose_negative(hrow, ranked, hag, random.Random(0), 0, 10, policy="v3") == ("P1", "same_subject")
    assert choose_negative(hrow, ranked, hag, random.Random(0), 0, 10)[0] != "P1"  # v4: festival-banned
    hrow["anchor_topic_ids"] = ["topic:pesach"]
    assert "topic:pesach" in row_segments(hrow, hag)[0]
    assert all(choose_negative(hrow, ranked, hag, random.Random(s), 0, 10, policy="v3")[0] in ("A", "E")
               for s in range(20))

    # 3. the loss object: masks, -inf cross logits, equality with the parent loss when nothing overlaps
    model = tiny_model(hidden=32, layers=1)
    width = max(len(s) for sg in segs for s in sg)
    labels = torch.tensor(label_rows(segs, width, list(range(len(segs)))))
    loss = MaskedCachedMNRL(model, mini_batch_size=4, family_names=[f"f{i}" for i in range(len(segs))])
    loss.labels = labels
    mask_qd, _ = loss.candidate_masks(1, torch.device("cpu"))
    expect = {(0, 1), (1, 0), (0, 2), (1, 2), (4, 0), (4, 3)}  # pgp:x shared by A,B,N; row 4's domain anchor -> A, C
    got = {(i, j) for i, j in zip(*torch.where(mask_qd)) for i, j in [(int(i), int(j))]}
    assert got == expect, got
    B, d = len(q), 8
    reps = [[torch.nn.functional.normalize(torch.randn(B, d), dim=1).requires_grad_()] for _ in range(2)]
    sims = loss.masked_logits(torch.cat(reps[0]), torch.cat(reps[1]), torch.cat(reps[1]), torch.arange(B), mask_qd,
                              None)["query_to_doc"]
    assert torch.isinf(sims[0, 1]) and torch.isinf(sims[1, 0]) and sims[0, 1] < 0, "shared-subject rows not masked"
    assert torch.isfinite(sims.diagonal()).all() and torch.isfinite(sims[2, 0]) and torch.isfinite(sims[3, 0])
    masked_val = loss.calculate_loss(reps)
    manual = []
    for i in range(B):
        s = sims[i].detach()
        manual.append(-(s[i] - torch.logsumexp(s, dim=0)))
    assert torch.allclose(masked_val, torch.stack(manual).mean(), atol=1e-5), (masked_val, manual)
    fam = loss.history[-1]["families"]  # per-family loss log: one row per family here, sums = per-row losses
    assert len(fam) == B and abs(sum(v[0] for v in fam.values()) / B - float(masked_val)) < 1e-4, fam
    loss.labels = torch.tensor([encode_label([[f"u{i}a"], [f"u{i}b"], []], 1) for i in range(B)])  # no overlaps
    parent = CachedMultipleNegativesRankingLoss.calculate_loss(loss, reps)
    assert torch.allclose(loss.calculate_loss(reps), parent, atol=1e-6)
    # hard-negative column: row 1's own negative F is a candidate for every row; row 1 masks nothing on F
    reps3 = reps + [[torch.nn.functional.normalize(torch.randn(B, d), dim=1)]]
    loss.labels = labels
    mqd3, _ = loss.candidate_masks(2, torch.device("cpu"))
    assert mqd3.shape == (B, 2 * B) and not mqd3[:, B + 1].any()
    assert torch.isfinite(loss.calculate_loss(reps3))
    # subject-level festival-tagged row (A6): the anchor's mention / festival ids mask an untagged festival record
    tag_pool = dict(pool, S1=dict(pool["B"], doc_id="S1", subjects=["pgp:q"]),
                    S2=dict(pool["B"], doc_id="S2", subjects=["M:topic:sukkot"],
                            text_hash=hashlib.md5(b"S2").hexdigest()))
    tagged = {"anchor": "laws of the booth", "positive": "x", "family": "subject", "kind": "q2d",
              "mask_level": "subject", "doc_id": "S1", "mask_ids": ["pgp:q", "D:S1"], "anchor_mask_ids": [],
              "text_hash": tag_pool["S1"]["text_hash"], "anchor_subject": "pgp:q", "anchor_mention_id": "M:pgp:q",
              "anchor_topic_ids": ["topic:sukkot", "M:topic:sukkot"]}
    other = dict(tagged, doc_id="S2", mask_ids=["M:topic:sukkot", "D:S2"], text_hash=tag_pool["S2"]["text_hash"],
                 anchor_subject="pgp:r", anchor_mention_id="M:pgp:r", anchor_topic_ids=[])
    tsegs = [row_segments(r, tag_pool) for r in (tagged, other)]
    loss.labels = torch.tensor(label_rows(tsegs, max(len(s) for sg in tsegs for s in sg), [0, 0]))
    tm, _ = loss.candidate_masks(1, torch.device("cpu"))
    assert bool(tm[0, 1]) and not bool(tm[1, 0]), tm

    # 4. symmetric kinds: d2d and t2t, both directions
    sym = MaskedCachedMNRL(model, symmetric=True, mini_batch_size=2)
    d2d_rows, t2t_rows = rows[6:9], rows[9:12]
    s_d = [row_segments(r, pool) for r in d2d_rows]
    sym.labels = torch.tensor([encode_label(s, max(len(x) for sg in s_d for x in sg)) for s in s_d])
    mqd, mdq = sym.candidate_masks(1, torch.device("cpu"))
    # rows: F->G (pgp:y), H->B, E->C. F shares nothing with B or C; H (pgp:z) shares pgp:z with G (row 0's positive)
    assert {(int(i), int(j)) for i, j in zip(*torch.where(mqd))} == {(1, 0)}
    assert {(int(i), int(j)) for i, j in zip(*torch.where(mdq))} == {(0, 1)}  # G vs anchor H (pgp:z)
    s_t = [row_segments(r, pool) for r in t2t_rows]
    sym.labels = torch.tensor([encode_label(s, max(len(x) for sg in s_t for x in sg)) for s in s_t])
    mqd, mdq = sym.candidate_masks(1, torch.device("cpu"))
    assert {(int(i), int(j)) for i, j in zip(*torch.where(mqd))} == {(0, 1), (1, 0)}
    assert torch.equal(mqd, mdq)
    r3 = [[torch.nn.functional.normalize(torch.randn(3, d), dim=1).requires_grad_()] for _ in range(2)]
    sims_dq = sym.masked_logits(torch.cat(r3[0]), torch.cat(r3[1]), torch.cat(r3[1]), torch.arange(3), mqd, mdq)
    assert torch.isinf(sims_dq["query_to_doc"][0, 1]) and torch.isinf(sims_dq["doc_to_query"][1, 0])
    sym.labels = torch.tensor([encode_label([[f"v{i}a"], [f"v{i}b"], []], 1) for i in range(3)])
    assert torch.allclose(sym.calculate_loss(r3), CachedMultipleNegativesRankingLoss.calculate_loss(sym, r3),
                          atol=1e-6)
    val = sym.calculate_loss(r3, with_backward=True)
    assert torch.isfinite(val) and r3[0][0].grad is not None and r3[1][0].grad is not None

    # 5. padding trim: a mini-batch embeds the same with and without its all-padding columns
    feats = model.preprocess(["short", "a much longer text " * 12, "mid length text here"])
    trim_loss = MaskedCachedMNRL(model, mini_batch_size=2, trim_padding=True)
    plain_loss = MaskedCachedMNRL(model, mini_batch_size=2, trim_padding=False)
    model.eval()
    for begin in (0, 2):
        e_trim, _ = trim_loss.embed_minibatch(feats, begin, begin + 2, False, False)
        e_full, _ = plain_loss.embed_minibatch(feats, begin, begin + 2, False, False)
        assert torch.allclose(e_trim, e_full, atol=1e-4), (begin, (e_trim - e_full).abs().max())
    assert trim_loss.stats["tokens_after_trim"] < trim_loss.stats["tokens_padded"], trim_loss.stats

    # 6. datasets + prompts
    rows[1]["negative"] = pool["F"]["text"]
    ds, prompts, stats = build_datasets(rows, pool)
    assert set(ds) == {"q2d", "q2d_hn", "d2d", "t2t"} and set(prompts) == set(ds)
    assert prompts["t2t"] == {"anchor": QUERY_PROMPT, "positive": QUERY_PROMPT} and prompts["d2d"] == {}
    assert ds["q2d_hn"].column_names == ["anchor", "positive", "negative", "label"]
    assert len(ds["q2d"][0]["label"]) == N_SEGMENTS * stats["label_width"] + 1
    assert stats["families"] == sorted({r["family"] for r in rows})
    # the ST collator resolves the nested prompts per dataset and lifts "label" out of the text columns
    from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator

    seen_prompts: Dict[str, Optional[str]] = {}

    def fake_preprocess(inputs: list, prompt: Optional[str] = None, task: Optional[str] = None) -> dict:
        seen_prompts[inputs[0]] = prompt
        return {"input_ids": torch.zeros(len(inputs), 1, dtype=torch.long)}

    collator = SentenceTransformerDataCollator(preprocess_fn=fake_preprocess, prompts=prompts)
    for name, d_set in ds.items():
        seen_prompts.clear()
        batch = collator([{**d_set[i], "dataset_name": name} for i in range(len(d_set))])
        assert batch["label"].shape == (len(d_set), N_SEGMENTS * stats["label_width"] + 1), name
        first = d_set[0]
        assert seen_prompts[first["anchor"]] == prompts[name].get("anchor"), name
        assert seen_prompts[first["positive"]] == prompts[name].get("positive"), name
        if "negative" in first:
            assert seen_prompts[first["negative"]] is None, name
    assert [k for k in batch if k.endswith("input_ids")] == ["anchor_input_ids", "positive_input_ids"]
    # 7. the serving contract on a saved tiny model
    with tempfile.TemporaryDirectory() as tmp:
        model.max_seq_length = 128
        model.preprocess(["x" * 50])  # leaves truncation state on the backend tokenizer, as training does
        finalize_for_save(model)
        model.save(tmp)
        check_saved_model(tmp)
    pure = self_test_pure(strict_festivals=True)
    print(json.dumps(pure, default=_json_default))
    print(json.dumps({"self_test": "ok", "q2d_masked_pairs": sorted(expect), "label_width": stats["label_width"],
                      "masked_loss": round(float(masked_val), 4)}))


def main() -> None:
    """CLI: ``--self-test`` (mask unit tests; needs torch + sentence-transformers)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    # repo checkout: eval_v3.py sits one level up (in a packaged dataset folder it sits next to this file, which wins)
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    if args.self_test:
        self_test()


if __name__ == "__main__":
    main()
