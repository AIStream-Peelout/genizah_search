"""Fine-tune Qwen3-Embedding-0.6B on the v3 semantic-text mixture (text track step 4 of 4).

Runs on a Colab A100 (``train_genizah_embedder_v3.ipynb``); importable locally for the CPU smoke test
(``smoke_train_v3.py``) and the mask unit tests (``python train_embedder_v3.py --self-test``, tiny random model).

Stages (one function each, so the notebook runs them cell by cell):

1. :func:`load_data`      -- the private dataset repo at tag ``v3`` (``package_dataset_v3.py``): train mix, train pool
   (the gated, shelf-mark-masked train records -- the ONLY source of mined negatives), semantic and original corpus
   text, the frozen eval dir and ``eval_v3.py``.
2. :func:`evaluate`       -- ``eval_v3.evaluate`` for the BASE model on the ORIGINAL production text (what production
   serves today) and on the SEMANTIC text (step 0: the representation change alone, no training).
3. :func:`mine_negatives` -- one hard negative per q2d / d2d row from the base model's ranking of the train pool
   (keyed by doc_id, never a held-out record, never identical text). ``mask_level == "subject"`` rows skip
   candidates sharing any (non-broad) subject with the row; ``mask_level == "doc"`` rows prefer a DIFFERENT record of
   the SAME subject (keeps within-subject discrimination).
4. :func:`train`          -- full fine-tune; fp32 master weights + bf16 autocast, gradient checkpointing, GradCache
   (:class:`MaskedCachedMNRL`): in-batch candidates whose ids overlap the row's ids get logit -inf.
5. :func:`evaluate` (tuned model, semantic text) + :func:`compare_reports` (three models side by side, paired
   subject-clustered bootstrap CIs, tuned vs base-on-semantic on the held-out-subject gate).
6. :func:`push`           -- model + reports to a private HF model repo (the owner runs it).

Masking contract (``build_training_mix_v3.py``). Each row is turned into three id segments, hashed to int64 and
carried in a numeric ``label`` column of fixed width ``3 * K`` (padding -1):

* ``seg0`` = relevance of the ANCHOR side: what counts as "the same thing" for this anchor.
  doc-level q2d: ``D:<doc>`` + ``T:<text_hash>`` of the positive (only the record itself / identical text);
  subject-level q2d: the positive's subject ids + ``anchor_subject`` + ``D:`` + ``T:``;
  d2d: the anchor record's subject ids + its ``D:`` / ``T:``; t2t: the pair's Sefaria ``concept_ids`` + ``T:`` of the
  anchor string. Broad ids (:data:`BROAD_PREFIXES`, the KTIV ``domain:`` genres such as documentary / letters /
  bible) are dropped from relevance unless the anchor names that domain (``anchor_subject``): sharing a genre does
  not make two records the same subject, and keeping it would mask ~6 % of in-batch candidates instead of ~1.6 %
  (T3's measurement) and forbid every same-genre hard negative.
* ``seg1`` = identity of the POSITIVE (all subject ids incl. broad ones + ``D:`` + ``T:``; t2t: concepts + ``T:``).
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
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch touches the GPU

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BASE_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_PROMPT = "Instruct: Given a search query, retrieve relevant passages\nQuery: "
KIND_PROMPTS = {"q2d": {"anchor": QUERY_PROMPT},
                "t2t": {"anchor": QUERY_PROMPT, "positive": QUERY_PROMPT},
                "d2d": {}}  # d2d: both sides are records -> documents raw
SYMMETRIC_KINDS = ("t2t", "d2d")
BROAD_PREFIXES = ("domain:",)
N_SEGMENTS = 3
PAD_ID = -1
EVAL_MAX_SEQ = 8192  # production window (embedding_service MAX_SEQ_LENGTH); also saved with the tuned model
TRAIN_MAX_SEQ = 384
# the hub checkpoint's input path (legacy ST config -> plain text, no chat template); see tiny_model / check_text_path
TEXT_MODALITY = {"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
DATA_FILES = ("train_mix_v3.jsonl", "train_pool.jsonl", "corpus_semantic.jsonl", "corpus_original.jsonl")


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


def load_data(repo_id: str, token: Optional[str] = None, local_dir: str = "/content/genizah_embed_data_v3",
              revision: Optional[str] = None) -> Dict[str, object]:
    """Download (or open) the v3 dataset folder and parse it.

    The folder also holds ``eval_v3.py``; it is put on ``sys.path`` so :func:`evaluate` scores with the copy that
    was packaged with the data.

    :param repo_id: Private HF dataset repo id, or a local directory (smoke tests).
    :param token: HF token (Colab secret ``HF_TOKEN``).
    :param local_dir: Download directory.
    :param revision: Dataset tag/commit to pin (``v3``); default = latest.
    :returns: Dict with ``train`` (mixture rows), ``pool`` (doc_id -> {text, text_hash, subjects}), ``semantic``
        (doc_id -> {text, eligible, text_hash, series}), ``original`` (doc_id -> production text), ``eval_pool``
        (frozen pool rows), ``held_out`` (ids), ``eval_dir``, ``path``, ``manifest``.
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
    data = {
        "train": _read_jsonl(path / "train_mix_v3.jsonl"),
        "pool": {r["doc_id"]: r for r in _read_jsonl(path / "train_pool.jsonl")},
        "semantic": {r["doc_id"]: r for r in _read_jsonl(path / "corpus_semantic.jsonl")},
        "original": {r["doc_id"]: r["text"] for r in _read_jsonl(path / "corpus_original.jsonl")},
        "eval_pool": _read_jsonl(eval_dir / "eval_pool.jsonl"),
        "held_out": set(json.loads((eval_dir / "held_out_records.json").read_text())["held_out"]),
        "eval_dir": str(eval_dir),
        "path": path,
        "manifest": json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else {},
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


def is_broad(sid: str) -> bool:
    """Whether a subject id is a broad genre id (dropped from relevance sets).

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
    """Identity ids of a record as a candidate: every subject id, ``D:<doc_id>`` and ``T:<text_hash>``.

    :param doc_id: Record id.
    :param subjects: Its subject ids (strong and weak; ``D:`` ids are ignored).
    :param text_hash: md5 of its semantic text.
    :returns: Sorted unique ids.
    :rtype: List[str]
    """
    ids = {s for s in subjects if not s.startswith("D:")}
    return sorted(ids | {f"D:{doc_id}", f"T:{text_hash}"})


def relevance_of(identity: Iterable[str], keep: Iterable[Optional[str]] = ()) -> List[str]:
    """Relevance ids from identity ids: broad genre ids dropped unless listed in ``keep``.

    :param identity: Identity ids.
    :param keep: Ids kept even when broad (the anchor's own subject).
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
    if row["kind"] == "d2d":
        a = row["anchor_doc_id"]
        seg0 = relevance_of(record_identity(a, row["anchor_mask_ids"], pool[a]["text_hash"]))
    elif row["mask_level"] == "doc":
        seg0 = [f"D:{row['doc_id']}", f"T:{row['text_hash']}"]
    else:
        seg0 = relevance_of(pos, keep=[row.get("anchor_subject")])
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
    """Flatten id segments into the fixed-width numeric ``label`` value.

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


def label_part(text: str) -> str:
    """A record text without its ``Transcription:`` / ``Translation:`` lines (the content family's positive form).

    :param text: Record text.
    :returns: The remaining lines.
    :rtype: str
    """
    return "\n".join(ln for ln in text.split("\n") if not ln.startswith(("Transcription:", "Translation:")))


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

    Columns: ``anchor``, ``positive`` (+ ``negative``), ``label`` (fixed-width int ids, see the module docstring).

    :param pairs: Training rows (output of :func:`mine_negatives`, or raw mixture rows).
    :param pool: Train pool.
    :returns: (``name -> Dataset``, ``name -> {column: prompt}``, stats).
    :rtype: Tuple[dict, Dict[str, dict], dict]
    """
    from datasets import Dataset

    groups: Dict[str, List[dict]] = defaultdict(list)
    segs: Dict[int, tuple] = {}
    for i, p in enumerate(pairs):
        groups[dataset_name(p)].append(p)
        segs[id(p)] = row_segments(p, pool)
    width = max(1, max(len(set(s)) for v in segs.values() for s in v))
    datasets, prompts = {}, {}
    for name in sorted(groups):
        rows = groups[name]
        cols = {"anchor": [r["anchor"] for r in rows], "positive": [r["positive"] for r in rows]}
        if name.endswith("_hn"):
            cols["negative"] = [r["negative"] for r in rows]
        cols["label"] = [encode_label(segs[id(r)], width) for r in rows]
        datasets[name] = Dataset.from_dict(cols)
        prompts[name] = dict(KIND_PROMPTS[kind_of(name)])
    if set(prompts) != set(datasets):
        raise AssertionError(f"prompt keys {sorted(prompts)} != dataset keys {sorted(datasets)}")
    for name, ds in datasets.items():
        if not set(prompts[name]) <= set(ds.column_names):
            raise AssertionError(f"{name}: prompt columns {sorted(prompts[name])} not in {ds.column_names}")
    stats = {"label_width": width, "rows": {k: len(v) for k, v in datasets.items()},
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
    """Split the ``label`` tensor into its :data:`N_SEGMENTS` (B, K) id segments.

    :param labels: (B, N_SEGMENTS * K) int64.
    :returns: List of (B, K) tensors.
    :rtype: List[Tensor]
    """
    width = labels.shape[1] // N_SEGMENTS
    return [labels[:, i * width:(i + 1) * width] for i in range(N_SEGMENTS)]


class MaskedCachedMNRL(CachedMultipleNegativesRankingLoss):
    """GradCache InfoNCE whose in-batch candidates sharing ids with the row get logit -inf.

    The ids arrive as the numeric ``label`` column (see the module docstring). Supported: one optional hard negative
    per row, ``query_to_doc`` (+ ``doc_to_query`` when symmetric, one softmax per direction), no hardness weighting,
    single device. ``stats`` counts batches and masked off-diagonal candidates for the training report.
    """

    def __init__(self, model, symmetric: bool = False, **kwargs) -> None:
        """Build the loss.

        :param model: SentenceTransformer being trained.
        :param symmetric: Add the ``doc_to_query`` direction (t2t, d2d) with a per-direction softmax.
        :param kwargs: Forwarded to :class:`CachedMultipleNegativesRankingLoss` (``scale``, ``mini_batch_size`` ...).
        """
        directions = ("query_to_doc", "doc_to_query") if symmetric else ("query_to_doc",)
        super().__init__(model, directions=directions, partition_mode="per_direction" if symmetric else "joint",
                         **kwargs)
        if self.gather_across_devices or self.hardness_mode is not None:
            raise ValueError("MaskedCachedMNRL supports a single device and no hardness weighting")
        self.labels: Optional[Tensor] = None
        self.stats: Counter = Counter()

    def forward(self, sentence_features: Iterable[dict], labels: Optional[Tensor]) -> Tensor:
        """Keep the batch's id labels for :meth:`calculate_loss`, then run GradCache as usual.

        :param sentence_features: Tokenised columns (anchor, positive[, negative]).
        :param labels: (B, N_SEGMENTS * K) id tensor from the ``label`` column.
        :returns: Loss.
        :rtype: Tensor
        """
        self.labels = labels
        return super().forward(sentence_features, labels)

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
        losses = []
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
            loss_mb = -(positive - log_z).mean() * len(idx) / size
            if with_backward:
                loss_mb.backward()
                loss_mb = loss_mb.detach()
            losses.append(loss_mb)
        return sum(losses)


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
                    pick_from: int) -> Tuple[Optional[str], str]:
    """Pick one hard negative record for a row from the base model's ranking of the train pool.

    Ranks ``< skip_top`` are skipped (too likely relevant, labels are incomplete). A candidate is invalid when it is
    the positive or anchor record, shares their text, shares an id with the row's relevance segment (subject rows:
    any non-broad subject), or (content family) has no text left once its transcription is dropped. Doc-level rows
    pick among valid candidates sharing a non-broad subject with the positive when there are any.

    :param row: Mixture row.
    :param ranked: Pool doc ids in the base model's order for this anchor.
    :param pool: Train pool.
    :param rng: RNG.
    :param skip_top: Raw ranks to skip.
    :param pick_from: Choose uniformly among the first this-many eligible candidates.
    :returns: (negative doc id or None, how it was chosen: ``same_subject`` / ``any`` / ``none``).
    :rtype: Tuple[Optional[str], str]
    """
    relevance = set(row_segments(row, pool)[0])
    banned_docs = {row["doc_id"], row.get("anchor_doc_id")}
    banned_hash = {row["text_hash"]} | ({pool[row["anchor_doc_id"]]["text_hash"]} if row.get("anchor_doc_id") else set())
    pos_subjects = {s for s in pool[row["doc_id"]]["subjects"] if not is_broad(s)}
    valid, preferred = [], []
    for d in ranked[skip_top:]:
        c = pool[d]
        if d in banned_docs or c["text_hash"] in banned_hash:
            continue
        if relevance & set(record_identity(d, c["subjects"], c["text_hash"])):
            continue
        if row["family"] == "content" and not label_part(c["text"]).strip():
            continue
        valid.append(d)
        if row["mask_level"] == "doc" and pos_subjects & set(c["subjects"]):
            preferred.append(d)
    if preferred:
        return rng.choice(preferred[:pick_from]), "same_subject"
    if valid:
        return rng.choice(valid[:pick_from]), "any"
    return None, "none"


def mine_negatives(model, data: dict, top_k: int = 50, skip_top: int = 3, pick_from: int = 10, max_seq: int = 512,
                   batch_size: int = 64, seed: int = 0) -> List[dict]:
    """Attach one hard negative to every q2d / d2d row whose candidates allow it (t2t keeps in-batch negatives).

    Candidates are the train pool only (eligible, gated, masked train records -- never held out), ranked by the
    BASE model: q2d anchors as queries, d2d anchors (records) by their own document vector. See
    :func:`choose_negative` for the filters. Content-family negatives use the candidate's text without its
    transcription, matching that family's positive form.

    :param model: Base SentenceTransformer.
    :param data: Output of :func:`load_data`.
    :param top_k: Ranking depth.
    :param skip_top: Raw ranks skipped.
    :param pick_from: Choose among the first this-many eligible candidates.
    :param max_seq: Token window while mining (training sees 384).
    :param batch_size: Encode batch size.
    :param seed: RNG seed.
    :returns: Copies of the mixture rows, some with ``negative``, ``negative_doc_id``, ``negative_choice``.
    :rtype: List[dict]
    """
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
    print(f"mining: embedded {len(ids)} pool records + {len(q_idx)} queries in {time.time() - t0:.0f} s", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = torch.from_numpy(mat).to(dev)
    k = min(top_k, len(ids))
    for idx_list, vecs in ((q_idx, qv), (d_idx, dv)):
        for start in range(0, len(idx_list), 1024):
            Q = torch.from_numpy(np.ascontiguousarray(vecs[start:start + 1024])).to(dev)
            top = torch.topk(Q @ M.T, k, dim=1).indices.cpu().numpy()
            for r, i in enumerate(idx_list[start:start + 1024]):
                neg, how = choose_negative(pairs[i], [ids[j] for j in top[r]], pool, rng, skip_top, pick_from)
                pairs[i]["negative_choice"] = how
                if neg:
                    text = pool[neg]["text"]
                    pairs[i]["negative"] = label_part(text) if pairs[i]["family"] == "content" else text
                    pairs[i]["negative_doc_id"] = neg
    del M
    return pairs


def mining_stats(pairs: List[dict]) -> dict:
    """How many rows got a negative, by family and choice.

    :param pairs: Output of :func:`mine_negatives`.
    :returns: ``{family: {same_subject, any, none}}`` plus totals.
    :rtype: dict
    """
    out: Dict[str, Counter] = defaultdict(Counter)
    for p in pairs:
        out[p["family"]][p.get("negative_choice", "not_mined")] += 1
    total = Counter()
    for c in out.values():
        total.update(c)
    return {"total": dict(total), "by_family": {f: dict(c) for f, c in sorted(out.items())}}


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


def train(data: dict, pairs: List[dict], out_dir: str, epochs: float = 1.0, lr: float = 2e-5, batch: int = 256,
          mini_batch: int = 8, max_seq: int = TRAIN_MAX_SEQ, bf16: bool = True, device: Optional[str] = None,
          model=None, scale: float = 20.0, max_steps: int = -1, sampler: str = "proportional", seed: int = 13,
          logging_steps: int = 10):
    """Full fine-tune with the masked GradCache InfoNCE, one dataset per kind and negative layout.

    :param data: Output of :func:`load_data` (``pool`` is used for the mask ids).
    :param pairs: Training rows (output of :func:`mine_negatives`).
    :param out_dir: Where the final model is saved (``training_info.json`` goes next to the weights).
    :param epochs: Epochs.
    :param lr: Learning rate.
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
    :returns: Trained SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers, MultiDatasetBatchSamplers

    # fp32 master weights + bf16 autocast: with bf16 weights an lr-2e-5 Adam step is below half a bf16 ulp for any
    # |w| >= ~0.008, so most updates would round to zero and the "fine-tune" would barely move the model
    if model is None:
        model = load_base(device)
    check_text_path(model)
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError(f"master weights must be fp32, got {next(model.parameters()).dtype}")
    model.max_seq_length = max_seq
    datasets, prompts, ds_stats = build_datasets(pairs, data["pool"])
    losses = {name: MaskedCachedMNRL(model, symmetric=kind_of(name) in SYMMETRIC_KINDS, mini_batch_size=mini_batch,
                                     scale=scale) for name in datasets}
    print(json.dumps({"datasets": ds_stats["rows"], "label_width": ds_stats["label_width"],
                      "prompts": {k: sorted(v) for k, v in prompts.items()}}), flush=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir + "_trainer", num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch, learning_rate=lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
        bf16=bf16, batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler={"proportional": MultiDatasetBatchSamplers.PROPORTIONAL,
                                     "round_robin": MultiDatasetBatchSamplers.ROUND_ROBIN}[sampler],
        prompts=prompts, logging_steps=logging_steps, save_strategy="no", report_to=[], seed=seed,
        use_cpu=(device == "cpu"), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=datasets, loss=losses)
    t0 = time.time()
    trainer.train()
    seconds = time.time() - t0
    # save with the production window, not the training one (a reload without an explicit max_seq_length would
    # otherwise truncate every document at TRAIN_MAX_SEQ tokens)
    model.max_seq_length = EVAL_MAX_SEQ
    model.save(out_dir)
    info = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION, "query_prompt": QUERY_PROMPT,
            "train_seconds": round(seconds, 1), "global_steps": trainer.state.global_step,
            "datasets": ds_stats, "prompts": prompts, "mining": mining_stats(pairs),
            "mask_stats": {name: dict(loss.stats) for name, loss in losses.items()},
            "masked_share": {name: round(loss.stats["qd_masked"] / max(1, loss.stats["qd_candidates"]), 4)
                             for name, loss in losses.items()},
            "loss_history": [h for h in trainer.state.log_history if "loss" in h],
            "args": {"epochs": epochs, "lr": lr, "batch": batch, "mini_batch": mini_batch, "max_seq": max_seq,
                     "bf16": bf16, "scale": scale, "max_steps": max_steps, "sampler": sampler, "seed": seed,
                     "broad_prefixes": list(BROAD_PREFIXES)},
            "data_manifest": data.get("manifest", {}).get("summary")}
    Path(out_dir, "training_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    # release optimizer state / caches before the evaluation re-embeds the corpus
    del trainer, losses
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def evaluate(model, data: dict, label: str, text: str = "semantic", max_seq: int = EVAL_MAX_SEQ,
             batch_size: int = 16, n_resamples: int = 1000, out_path: Optional[str] = None) -> dict:
    """Score one model with the frozen v3 eval (``eval_v3.evaluate``) over the whole eligible pool.

    :param model: SentenceTransformer.
    :param data: Output of :func:`load_data`.
    :param label: Name for the report (``base_original``, ``base_semantic``, ``tuned_<run>``).
    :param text: Document text to embed: ``semantic`` (v3, no ID / shelf-mark lines) or ``original`` (production).
    :param max_seq: Token window; 8192 = production, so every model is scored on the vectors production would serve.
    :param batch_size: Document encode batch size.
    :param n_resamples: Bootstrap resamples (memorisation-gap CI).
    :param out_path: Write the full result (incl. per-query lists, needed for paired comparisons) here as JSON.
    :returns: ``eval_v3.evaluate`` result plus ``meta``.
    :rtype: dict
    """
    import eval_v3

    ids = [r["doc_id"] for r in data["eval_pool"]]
    if text == "semantic":
        texts = [data["semantic"][d]["text"] for d in ids]
        corpus_rows = {d: {"text": t} for d, t in zip(ids, texts)}  # eval_v3 hashes the texts actually embedded
    elif text == "original":
        texts = [data["original"][d] for d in ids]
        corpus_rows = None  # production text: its hashes differ from the pool's by design
    else:
        raise ValueError(f"text must be 'semantic' or 'original', not {text!r}")
    prev = model.max_seq_length
    model.max_seq_length = max_seq
    t0 = time.time()
    mat = encode(model, texts, is_query=False, batch_size=batch_size)
    t_docs = time.time() - t0

    def qfn(queries: List[str]) -> np.ndarray:
        return encode(model, queries, is_query=True, batch_size=64)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # spurious BLAS warnings on unit vectors
        res = eval_v3.evaluate(ids, mat, qfn, data["eval_dir"], corpus_rows=corpus_rows, n_resamples=n_resamples,
                               trained_subject_positives=trained_subject_positives(data["train"]))
    model.max_seq_length = prev
    if text == "semantic" and res["pool"]["text_hash_mismatch"]:
        raise ValueError(f"{res['pool']['text_hash_mismatch']} pool vectors are not of the frozen semantic text")
    res["meta"] = {"label": label, "text": text, "max_seq": max_seq, "embed_seconds": round(t_docs, 1),
                   "total_seconds": round(time.time() - t0, 1), "n_docs": len(ids),
                   "dtype": str(next(model.parameters()).dtype), "query_prompt": QUERY_PROMPT}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, ensure_ascii=False, default=_json_default))
    return res


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
    """JSON fallback for numpy scalars / arrays.

    :param obj: Object.
    :returns: JSON-serialisable value.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def load_reports(report_dir: str, labels: Sequence[str]) -> Dict[str, dict]:
    """Re-read reports written by :func:`evaluate` (after a Colab runtime restart).

    :param report_dir: Directory holding ``<label>.json``.
    :param labels: Labels to read.
    :returns: ``label -> result``.
    :rtype: Dict[str, dict]
    """
    return {lb: json.loads(Path(report_dir, f"{lb}.json").read_text()) for lb in labels}


def _metric_rows(res: dict) -> List[Tuple[str, Optional[float]]]:
    """Headline ``(metric, value)`` rows of one result, in table order.

    :param res: ``eval_v3.evaluate`` result.
    :returns: Rows.
    :rtype: List[Tuple[str, Optional[float]]]
    """
    rows: List[Tuple[str, Optional[float]]] = []
    sub = res["subjects"]
    for group, name in (("held_out", "GATE held-out subjects"), ("linked", "linked subjects"), ("seen", "seen subjects")):
        mac = sub[group]["macro"]
        for m in ("AP", "P@10", "R@100"):
            rows.append((f"{name} (n={mac.get('n_subjects', 0)}) {m}", mac.get(m)))
        if group == "held_out":
            for src, v in sorted(sub[group]["macro_by_source"].items()):
                rows.append((f"  held-out by source {src} AP", v.get("AP")))
            for sid, v in sorted(sub[group]["per_subject"].items()):
                rows.append((f"  {sid} AP", v.get("AP")))
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
    return rows


def compare_reports(reports: Dict[str, dict], baseline: str = "base_semantic", candidate: Optional[str] = None,
                    n_resamples: int = 1000) -> dict:
    """Side-by-side headline metrics of several results plus paired, subject-clustered bootstrap CIs.

    Paired tests (``eval_v3.compare``): candidate (default: the last report) vs ``baseline`` on held-out (the gate),
    linked and seen subjects (AP, P@10, R@100), every held-out subject's AP, and known-item RR / R@10; plus the
    step-0 effect ``base_semantic`` vs ``base_original`` and the candidate vs ``base_original`` (production today) on
    the gate AP and known-item RR when those reports are present.

    :param reports: ``label -> eval result`` in display order (same eval dir).
    :param baseline: Baseline label.
    :param candidate: Candidate label (default: last key of ``reports``).
    :param n_resamples: Bootstrap resamples.
    :returns: ``{labels, table: [{metric, values}], paired: {name: CI dict}}``.
    :rtype: dict
    """
    import eval_v3

    labels = list(reports)
    candidate = candidate or labels[-1]
    per_label = {lb: dict(_metric_rows(r)) for lb, r in reports.items()}
    order = [m for m, _ in _metric_rows(reports[labels[0]])]
    table = [{"metric": m, "values": {lb: per_label[lb].get(m) for lb in labels}} for m in order]
    paired: Dict[str, dict] = {}
    done: set = set()

    def add(name: str, a: str, b: str, group: str, metric: str) -> None:
        if a not in reports or b not in reports or a == b or (a, b, group, metric) in done:
            return  # missing report, self-comparison, or already reported under another name
        if group != "known_item" and not reports[a]["subjects"][group]["per_query"]:
            return
        done.add((a, b, group, metric))
        paired[name] = eval_v3.compare(reports[a], reports[b], group, metric, n_resamples)

    a, b = baseline, candidate
    for group in ("held_out", "linked", "seen"):
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
    if a in reports and b in reports:
        for sid, pq in reports[a]["subjects"]["held_out"]["per_query"].items():
            paired[f"{b} - {a} | {sid} AP"] = eval_v3.paired_bootstrap(
                {sid: pq["AP"]}, {sid: reports[b]["subjects"]["held_out"]["per_query"][sid]["AP"]}, n_resamples)
    add("base_semantic - base_original | held_out AP (step 0)", "base_original", "base_semantic", "held_out", "AP")
    add("base_semantic - base_original | known_item RR (step 0)", "base_original", "base_semantic", "known_item", "RR")
    add(f"{b} - base_original | held_out AP (vs production today)", "base_original", b, "held_out", "AP")
    add(f"{b} - base_original | known_item RR (vs production today)", "base_original", b, "known_item", "RR")
    return {"labels": labels, "baseline": a, "candidate": b, "table": table, "paired": paired,
            "primary": paired.get(f"{b} - {a} | held_out AP")}


def format_table(comparison: dict) -> str:
    """Fixed-width text of :func:`compare_reports` output (values side by side, then the paired CIs).

    :param comparison: Output of :func:`compare_reports`.
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
    return "\n".join(lines)


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
         data_revision: str) -> str:
    """Upload the model and its eval reports to a private HF model repo, tag the commit; return the commit sha.

    :param model_dir: Saved model directory (``training_info.json`` already inside).
    :param repo_id: Target model repo.
    :param token: HF token.
    :param reports: ``label -> full eval result``.
    :param comparison: Output of :func:`compare_reports`.
    :param run_name: Run name (also the git tag; pin the sha in the embedding contract).
    :param data_revision: Dataset tag the run trained on.
    :returns: Commit sha.
    :rtype: str
    """
    from huggingface_hub import HfApi

    save_reports(model_dir, reports, comparison, {"run_name": run_name, "data_revision": data_revision,
                                                  "base_model": BASE_MODEL, "base_revision": BASE_REVISION})
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


def self_test() -> None:
    """Unit tests of the masking (pure torch + a tiny random model for the loss object; CPU, seconds)."""
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

    # 3. the loss object: masks, -inf cross logits, equality with the parent loss when nothing overlaps
    model = tiny_model(hidden=32, layers=1)
    width = max(len(s) for sg in segs for s in sg)
    labels = torch.tensor([encode_label(s, width) for s in segs])
    loss = MaskedCachedMNRL(model, mini_batch_size=4)
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
    loss.labels = torch.tensor([encode_label([[f"u{i}a"], [f"u{i}b"], []], 1) for i in range(B)])  # no overlaps
    parent = CachedMultipleNegativesRankingLoss.calculate_loss(loss, reps)
    assert torch.allclose(loss.calculate_loss(reps), parent, atol=1e-6)
    # hard-negative column: row 1's own negative F is a candidate for every row; row 1 masks nothing on F
    reps3 = reps + [[torch.nn.functional.normalize(torch.randn(B, d), dim=1)]]
    loss.labels = labels
    mqd3, _ = loss.candidate_masks(2, torch.device("cpu"))
    assert mqd3.shape == (B, 2 * B) and not mqd3[:, B + 1].any()
    assert torch.isfinite(loss.calculate_loss(reps3))

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

    # 5. datasets + prompts
    rows[1]["negative"] = pool["F"]["text"]
    ds, prompts, stats = build_datasets(rows, pool)
    assert set(ds) == {"q2d", "q2d_hn", "d2d", "t2t"} and set(prompts) == set(ds)
    assert prompts["t2t"] == {"anchor": QUERY_PROMPT, "positive": QUERY_PROMPT} and prompts["d2d"] == {}
    assert ds["q2d_hn"].column_names == ["anchor", "positive", "negative", "label"]
    assert len(ds["q2d"][0]["label"]) == N_SEGMENTS * stats["label_width"]
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
        assert batch["label"].shape == (len(d_set), N_SEGMENTS * stats["label_width"]), name
        first = d_set[0]
        assert seen_prompts[first["anchor"]] == prompts[name].get("anchor"), name
        assert seen_prompts[first["positive"]] == prompts[name].get("positive"), name
        if "negative" in first:
            assert seen_prompts[first["negative"]] is None, name
    assert [k for k in batch if k.endswith("input_ids")] == ["anchor_input_ids", "positive_input_ids"]
    print(json.dumps({"self_test": "ok", "q2d_masked_pairs": sorted(expect), "label_width": stats["label_width"],
                      "masked_loss": round(float(masked_val), 4)}))


def main() -> None:
    """CLI: ``--self-test`` (mask unit tests)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()


if __name__ == "__main__":
    main()
