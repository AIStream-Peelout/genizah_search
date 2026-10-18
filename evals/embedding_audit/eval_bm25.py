"""Probe T2-lexical: the same topic queries scored with plain BM25 (the keyword leg's ceiling).

Uses the identical corpus text, queries, labels and metrics as ``eval_corpus.py`` so the
semantic numbers can be read against what string matching alone achieves. Tokenisation is
deliberately simple (lowercase word characters incl. Hebrew); no stemming or prefix
stripping, roughly what an analyzer without Hebrew morphology does.
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from embed_utils import AUDIT_ROOT
from eval_corpus import EVAL_TOPICS, average_precision, ndcg_at_k

HERE = Path(__file__).parent


def bm25_matrix(texts: list, k1: float = 1.2, b: float = 0.75) -> tuple:
    """Build a BM25-weighted doc-term matrix.

    :param texts: Documents.
    :param k1: BM25 k1.
    :param b: BM25 b.
    :returns: (vectorizer, weighted CSR matrix).
    :rtype: tuple
    """
    vec = CountVectorizer(token_pattern=r"(?u)[\w֐-׿']{2,}", lowercase=True, min_df=1)
    tf = vec.fit_transform(texts).tocsr().astype(np.float32)
    dl = np.asarray(tf.sum(1)).ravel()
    avgdl = dl.mean()
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((tf.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
    denom = tf.data + k1 * (1 - b + b * dl[rows] / avgdl)
    tf.data = (tf.data * (k1 + 1) / denom) * idf[tf.indices]
    return vec, tf


def main() -> None:
    """Score BM25 on the T2 topic queries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--probe", default=str(HERE / "concept_probe_v1.json"))
    args = parser.parse_args()
    rows, texts = [], []
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            texts.append(r.pop("text"))
            rows.append(r)
    vec, W = bm25_matrix(texts)
    topics = [set(r["topics"]) for r in rows]
    lexical = [set(r["lexical"]) for r in rows]
    framed_idx = np.where(np.array([r["framed"] for r in rows]))[0]
    probe = json.loads(Path(args.probe).read_text())
    per_topic = {}
    for topic in EVAL_TOPICS:
        q = vec.transform(probe["concepts"][topic]["queries"])
        q.data[:] = 1.0
        S = (q @ W.T).toarray()
        strict = np.array([topic in t for t in topics])
        lenient = strict | np.array([topic in l for l in lexical])
        res = defaultdict(list)
        for qi in range(S.shape[0]):
            order = np.argsort(-S[qi])
            res["p10_strict"].append(strict[order[:10]].mean())
            res["p10_lenient"].append(lenient[order[:10]].mean())
            res["zero_hit_query"].append(float(S[qi].max() == 0))
            fo = framed_idx[np.argsort(-S[qi, framed_idx])]
            rel = strict[fo].astype(float)
            npos = int(rel.sum())
            res["framed_ap"].append(average_precision(rel, npos))
            res["framed_ndcg10"].append(ndcg_at_k(rel, 10, npos))
        per_topic[topic] = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    macro = {k: round(float(np.mean([pt[k] for pt in per_topic.values()])), 4) for k in next(iter(per_topic.values()))}
    out = {"model": "bm25", "macro": macro, "per_topic": per_topic, "ran_at": datetime.now().isoformat(timespec="seconds")}
    (HERE / "results" / "t2_corpus_bm25.json").write_text(json.dumps(out, indent=1))
    print("bm25", json.dumps(macro), flush=True)


if __name__ == "__main__":
    main()
