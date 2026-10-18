"""Fine-tune Qwen3-Embedding-0.6B for Cairo Genizah retrieval (runs on a Colab A100; also importable locally).

Pipeline (each stage is a function so the notebook can run them cell by cell):
1. ``load_data``      — pull the private dataset repo (train mix, corpus, eval queries, held-out ids).
2. ``mine_negatives`` — one hard negative per pair, mined with the BASE model over the corpus, skipping
                        candidates that share the positive's topic label or catalogue frame (false-negative guard).
3. ``train``          — full fine-tune, CachedMultipleNegativesRankingLoss (GradCache), production query prompt.
4. ``evaluate``       — the audit's held-out metrics for base vs tuned: known-item retrieval of eval-pool synthetic
                        queries by type (T3), topic retrieval over the identified-record pool with the 130 concept
                        queries (T2: AP / nDCG@10), and recall of the human-verified Sukkot records.
5. ``push``           — model + eval report to a private HF model repo, pinned by commit for the embedding contract.

The embedding contract (prefix, normalisation, 1024-d, last-token pooling) is unchanged from production.
"""

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BASE_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_PROMPT = "Instruct: Given a search query, retrieve relevant passages\nQuery: "


def load_data(repo_id: str, token: Optional[str], local_dir: str = "/content/genizah_embed_data",
              revision: Optional[str] = None) -> Dict[str, object]:
    """Download and parse the dataset repo.

    :param repo_id: Private HF dataset repo id (or a local directory for smoke tests).
    :param token: HF token (Colab secret ``HF_TOKEN``).
    :param revision: Dataset tag/commit to pin (``v1``, ``v2`` …); default = latest.
    :param local_dir: Download directory.
    :returns: Dict with ``train`` (list of pairs), ``corpus`` (doc_id -> row), ``eval_queries``, ``concepts``, ``held_out``.
    :rtype: Dict[str, object]
    """
    if Path(repo_id).is_dir():  # local smoke tests
        path = Path(repo_id)
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", token=token, local_dir=local_dir,
                                      revision=revision))
    read = lambda name: [json.loads(l) for l in open(path / name, encoding="utf-8")]  # noqa: E731
    corpus = {r["doc_id"]: r for r in read("corpus.jsonl")}
    return {
        "train": read("train_mix.jsonl"),
        "corpus": corpus,
        "eval_queries": read("eval_queries.jsonl"),
        "concepts": json.loads((path / "concept_probe_v1.json").read_text()),
        "held_out": set(json.loads((path / "held_out_ids.json").read_text())),
        "path": path,
    }


def encode(model, texts: List[str], is_query: bool, batch_size: int = 128) -> np.ndarray:
    """Encode with the production contract.

    :param model: SentenceTransformer.
    :param texts: Texts.
    :param is_query: Prefix with the query instruction.
    :param batch_size: Batch size.
    :returns: L2-normalised float32 matrix.
    :rtype: np.ndarray
    """
    return model.encode(texts, prompt=QUERY_PROMPT if is_query else None, batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def corpus_matrix(model, corpus: Dict[str, dict]) -> tuple:
    """Embed the whole corpus (documents in length order for speed).

    :param model: SentenceTransformer.
    :param corpus: doc_id -> row.
    :returns: (ids, matrix).
    :rtype: tuple
    """
    ids = sorted(corpus, key=lambda d: len(corpus[d]["text"]))
    return ids, encode(model, [corpus[d]["text"] for d in ids], is_query=False, batch_size=16)


def mine_negatives(model, data: dict, top_k: int = 30, skip_top: int = 3, seed: int = 0) -> List[dict]:
    """Attach one hard negative to each corpus-positive pair.

    Candidates come from ranks ``skip_top..top_k`` of the base model's corpus ranking for the anchor; a candidate is
    rejected if it is the positive, a held-out record, or shares a topic label / first catalogue frame with the
    positive's record (likely an unlabelled true positive). Pairs whose positive is not a corpus record
    (Sefaria passages, scholarship pages) keep in-batch negatives only.

    :param model: Base SentenceTransformer.
    :param data: Output of :func:`load_data`.
    :param top_k: Candidate depth.
    :param skip_top: Ranks to skip (too likely to be relevant).
    :param seed: RNG seed.
    :returns: Pairs, some with a ``negative`` field.
    :rtype: List[dict]
    """
    rng = random.Random(seed)
    corpus, held = data["corpus"], data["held_out"]
    ids, mat = corpus_matrix(model, corpus)
    pairs = data["train"]
    # keyed by the pair's doc_id, not its text: identical record texts would otherwise resolve to the wrong record.
    # scholar_doc2page positives are scholarship pages, not records.
    idx = [i for i, p in enumerate(pairs) if p.get("doc_id") in corpus and p["family"] != "scholar_doc2page"]
    qv = encode(model, [pairs[i]["anchor"] for i in idx], is_query=True)
    import torch

    M = torch.from_numpy(mat).cuda() if torch.cuda.is_available() else torch.from_numpy(mat)
    for start in range(0, len(idx), 1024):
        Q = torch.from_numpy(qv[start:start + 1024]).to(M.device)
        top = torch.topk(Q @ M.T, top_k, dim=1).indices.cpu().numpy()
        for row, i in enumerate(idx[start:start + 1024]):
            pos_id = pairs[i]["doc_id"]
            pos = corpus[pos_id]
            pos_topics, pos_frame = set(pos.get("topics") or []), (pos.get("frames") or [None])[0]
            cands = []
            for j in top[row][skip_top:]:
                d = ids[j]
                c = corpus[d]
                if d == pos_id or d in held or (pos_topics & set(c.get("topics") or [])):
                    continue
                if c["text"] in (pos["text"], pairs[i]["positive"]):  # duplicate text is not a negative
                    continue
                if pos_frame and (c.get("frames") or [None])[0] == pos_frame:
                    continue
                cands.append(d)
            if cands:
                pairs[i]["negative"] = corpus[rng.choice(cands[:10])]["text"]
    return pairs


def train(data: dict, pairs: List[dict], out_dir: str, epochs: float = 1.0, lr: float = 2e-5, batch: int = 256,
          mini_batch: int = 8, max_seq: int = 384, bf16: bool = True, device: Optional[str] = None):
    """Full fine-tune with GradCache InfoNCE.

    :param data: Output of :func:`load_data` (unused fields ignored).
    :param pairs: Training pairs (``anchor``, ``positive``, optional ``negative``).
    :param out_dir: Where to save the final model.
    :param epochs: Epochs.
    :param lr: Learning rate.
    :param batch: Effective batch size.
    :param mini_batch: GradCache mini-batch (memory knob: 8 fits a 40 GB A100 with checkpointing).
    :param max_seq: Max tokens per text (median record is ~210 chars; 384 keeps the long tail).
    :param bf16: bf16 autocast (A100); weights and optimizer state stay fp32. False for CPU smoke tests.
    :param device: Force a device ("cpu" for smoke tests); default picks the accelerator.
    :returns: Trained SentenceTransformer.
    """
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    # fp32 master weights + bf16 autocast: with bf16 weights an lr-2e-5 Adam step is below half a bf16 ulp for any
    # |w| >= ~0.008, so most updates would round to zero and the "fine-tune" would barely move the model
    model = SentenceTransformer(BASE_MODEL, revision=BASE_REVISION, device=device,
                                model_kwargs={"torch_dtype": torch.float32})
    model.max_seq_length = max_seq
    with_neg = [p for p in pairs if p.get("negative")]
    without = [p for p in pairs if not p.get("negative")]
    # two datasets: GradCache MNRL accepts (anchor, positive[, negative]); keep columns consistent per dataset
    ds = {
        "with_negative": Dataset.from_dict({"anchor": [p["anchor"] for p in with_neg], "positive": [p["positive"] for p in with_neg],
                                            "negative": [p["negative"] for p in with_neg]}),
        "in_batch_only": Dataset.from_dict({"anchor": [p["anchor"] for p in without], "positive": [p["positive"] for p in without]}),
    }
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch)
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir + "_trainer", num_train_epochs=epochs, per_device_train_batch_size=batch,
        learning_rate=lr, warmup_ratio=0.05, lr_scheduler_type="cosine", bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES, prompts={"anchor": QUERY_PROMPT},
        logging_steps=10, save_strategy="no", report_to=[], seed=13, use_cpu=(device == "cpu"),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss={k: loss for k in ds})
    trainer.train()
    model.save(out_dir)
    # release optimizer state / caches before the evaluation re-embeds the corpus
    import gc
    del trainer, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _ap(rel: np.ndarray) -> float:
    """Average precision of a ranked binary relevance vector.

    :param rel: Relevance in rank order.
    :returns: AP.
    :rtype: float
    """
    n = rel.sum()
    if not n:
        return 0.0
    hits = np.cumsum(rel)
    return float(((hits / np.arange(1, len(rel) + 1)) * rel).sum() / n)


def evaluate(model, data: dict, label: str, max_seq: int = 8192) -> dict:
    """The audit's held-out metrics for one model.

    :param model: SentenceTransformer.
    :param data: Output of :func:`load_data`.
    :param label: Name for the report.
    :param max_seq: Token window for every model under test; 8192 = production, so base and tuned are compared on the
        vectors production would actually serve.
    :returns: Metrics dict.
    :rtype: dict
    """
    model.max_seq_length = max_seq
    corpus = data["corpus"]
    ids, mat = corpus_matrix(model, corpus)
    pos = {d: i for i, d in enumerate(ids)}
    out = {"model": label, "max_seq": max_seq}
    # T3 known-item by query type (eval-pool synthetic queries, optionally Isaac-validated). Ties count against the
    # target (pessimistic), and a record whose text is identical to the target's counts as a hit, not a competitor.
    eq = [q for q in data["eval_queries"] if q["doc_id"] in pos]
    qv = encode(model, [q["text"] for q in eq], is_query=True)
    same_text = defaultdict(list)
    for i, d in enumerate(ids):
        same_text[corpus[d]["text"]].append(i)
    ranks = defaultdict(list)
    for q, v in zip(eq, qv):
        s = mat @ v
        twins = same_text[corpus[q["doc_id"]]["text"]]
        ahead = s >= s[twins].max()
        ahead[twins] = False
        ranks[q["type"]].append(int(ahead.sum()) + 1)
    out["t3_known_item"] = {t: {"n": len(r), "MRR": round(float(np.mean(1 / np.array(r))), 4),
                                "R@10": round(float(np.mean(np.array(r) <= 10)), 4),
                                "R@100": round(float(np.mean(np.array(r) <= 100)), 4),
                                "median_rank": int(np.median(r))} for t, r in sorted(ranks.items())}
    # T2 topic retrieval within the identified (framed) pool, held-out records only
    framed = np.array([i for i, d in enumerate(ids) if corpus[d].get("framed") and d in data["held_out"]])
    topics = data["concepts"]["concepts"]
    # a probe query that also appears verbatim as a training anchor is not a held-out query: score only unseen ones
    seen = {" ".join(p["anchor"].lower().split()) for p in data["train"]}
    t2, n_seen = {}, 0
    for topic, spec in topics.items():
        rel_all = np.array([topic in (corpus[ids[i]].get("topics") or []) for i in framed])
        if rel_all.sum() < 5:
            continue
        queries = [q for q in spec["queries"] if " ".join(q.lower().split()) not in seen]
        n_seen += len(spec["queries"]) - len(queries)
        if not queries:
            continue
        qv = encode(model, queries, is_query=True)
        aps, ndcg = [], []
        for v in qv:
            order = np.argsort(-(mat[framed] @ v))
            rel = rel_all[order].astype(float)
            aps.append(_ap(rel))
            disc = 1 / np.log2(np.arange(2, 12))
            ndcg.append(float((rel[:10] * disc).sum() / disc[: int(min(10, rel.sum()))].sum()))
        t2[topic] = {"AP": round(float(np.mean(aps)), 4), "nDCG@10": round(float(np.mean(ndcg)), 4), "n_pos": int(rel_all.sum())}
    out["t2_topics_heldout"] = t2
    out["t2_queries_skipped_seen_in_train"] = n_seen
    out["t2_macro"] = {k: round(float(np.mean([v[k] for v in t2.values()])), 4) for k in ("AP", "nDCG@10")} if t2 else {}
    # Sukkot gold recall (all sukkot queries, max-sim over the full corpus)
    gold = [pos[d] for d in ids if corpus[d].get("sukkot_gold")]
    qv = encode(model, topics["sukkot"]["queries"], is_query=True)
    best = (qv @ mat.T).max(0)
    rank_of = np.empty(len(ids), dtype=int)
    rank_of[np.argsort(-best)] = np.arange(len(ids))
    out["sukkot_gold"] = {"n": len(gold), "R@100": round(float(np.mean(rank_of[gold] < 100)), 4),
                          "R@1000": round(float(np.mean(rank_of[gold] < 1000)), 4),
                          "median_rank": int(np.median(rank_of[gold]))}
    return out


def push(model_dir: str, repo_id: str, token: str, reports: List[dict]) -> str:
    """Upload the model and its eval report to a private HF repo; return the commit sha.

    :param model_dir: Saved model directory.
    :param repo_id: Target model repo.
    :param token: HF token.
    :param reports: Eval reports to store next to the weights.
    :returns: Commit sha (pin this in the embedding contract).
    :rtype: str
    """
    from huggingface_hub import HfApi

    Path(model_dir, "genizah_eval_report.json").write_text(json.dumps(reports, indent=1))
    api = HfApi(token=token)
    api.create_repo(repo_id, private=True, exist_ok=True)
    info = api.upload_folder(folder_path=model_dir, repo_id=repo_id, commit_message="genizah embedder fine-tune")
    return info.oid
