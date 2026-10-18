"""Package the private training dataset repo contents into ``<AUDIT_ROOT>/hf_dataset/<name>/``.

Files:
* ``train_mix.jsonl``        — from ``build_training_mix.py`` (anchor, positive, family, …)
* ``corpus.jsonl``           — every record: doc_id, production text, topics, frames, framed, sukkot_gold
* ``eval_queries.jsonl``     — eval-pool synthetic queries (doc_id, type, text); restricted to Isaac's kept
                               queries when ``--reviews`` is given
* ``concept_probe_v1.json``  — the 130 nuanced topic queries (T1/T2)
* ``held_out_ids.json``      — records that must never be trained on
* ``train_embedder.py``      — the training/eval code that consumes these files
* ``README.md``              — provenance and the embedding contract
"""

import argparse
import json
import shutil
from pathlib import Path

from build_query_batches import is_test
from embed_utils import AUDIT_ROOT

HERE = Path(__file__).parent
AUDIT = HERE.parent

README = """---
license: other
---
# genizah-embed-train ({name})

Private training/eval data for fine-tuning the Cairo Genizah retrieval embedder
(base `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1024-d, cosine, last-token pooling,
queries prefixed `Instruct: Given a search query, retrieve relevant passages\\nQuery: `, documents raw).

* `train_mix.jsonl` — {n_train} pairs; families: {families}
* `corpus.jsonl` — {n_corpus} records = `genizah_merged_v8` production text (verified identical vectors)
* `eval_queries.jsonl` — {n_eval} held-out synthetic queries ({eval_note})
* `held_out_ids.json` — {n_held} records excluded from training (20 % hash split + Sukkot seed + eval pools)

Built by `genizah_search/evals/embedding_audit/` on {date}. Synthetic queries were written by Claude sub-agents
following `query_writing_spec.md` (v2); Sefaria-derived pairs follow Sefaria's per-text licences.
"""


def main() -> None:
    """Assemble the dataset folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v1")
    parser.add_argument("--mix", default=str(AUDIT_ROOT / "train_mix_v1.jsonl"))
    parser.add_argument("--rounds", default="r1")
    parser.add_argument("--reviews", default=None)
    args = parser.parse_args()
    out = AUDIT_ROOT / "hf_dataset" / args.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.mix, out / "train_mix.jsonl")
    shutil.copy(AUDIT / "concept_probe_v1.json", out / "concept_probe_v1.json")
    shutil.copy(HERE / "train_embedder.py", out / "train_embedder.py")
    held, n_corpus = set(), 0
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for line in open(AUDIT_ROOT / "corpus_v1.jsonl", encoding="utf-8"):
            r = json.loads(line)
            n_corpus += 1
            if is_test(r["doc_id"]) or r.get("sukkot_gold"):
                held.add(r["doc_id"])
            fh.write(json.dumps({k: r.get(k) for k in ("doc_id", "text", "topics", "frames", "framed", "sukkot_gold")},
                                ensure_ascii=False) + "\n")
    kept = None
    if args.reviews:
        kept = {}
        for rv in json.loads(Path(args.reviews).read_text()):
            for q in rv.get("queries", []):
                if q.get("verdict") in ("good", "fixed") and q.get("eval", True):
                    kept[(rv["doc_id"], q["qid"].split(":")[-1])] = q["text"]
    n_eval = 0
    with open(out / "eval_queries.jsonl", "w", encoding="utf-8") as fh:
        for rnd in args.rounds.split(","):
            base = AUDIT_ROOT / "synthetic_queries" / rnd
            pool = {}
            for f in (base / "batches").glob("*.jsonl"):
                for line in open(f, encoding="utf-8"):
                    b = json.loads(line)
                    pool[b["doc_id"]] = b["pool"]
                    if b["pool"] == "eval":
                        held.add(b["doc_id"])
            for f in sorted((base / "out").glob("*.jsonl")) + sorted((base / "out_he").glob("*.jsonl")):
                for line in open(f, encoding="utf-8"):
                    o = json.loads(line)
                    if pool.get(o["doc_id"]) != "eval":
                        continue
                    he_round = "out_he" in str(f)
                    for i, q in enumerate(o.get("queries") or []):
                        text = q["text"]
                        if kept is not None:
                            if (o["doc_id"], f"{'he' if he_round else ''}{i}") not in kept:
                                continue
                            text = kept[(o["doc_id"], f"{'he' if he_round else ''}{i}")]
                        fh.write(json.dumps({"doc_id": o["doc_id"], "type": q["type"], "text": text}, ensure_ascii=False) + "\n")
                        n_eval += 1
    (out / "held_out_ids.json").write_text(json.dumps(sorted(held)))
    stats = json.loads(Path(args.mix).with_suffix(".stats.json").read_text()) if Path(args.mix).with_suffix(".stats.json").exists() else {}
    from datetime import date
    (out / "README.md").write_text(README.format(
        name=args.name, n_train=sum(1 for _ in open(out / "train_mix.jsonl")), families=", ".join(stats.get("by_family", {})),
        n_corpus=n_corpus, n_eval=n_eval, eval_note="Isaac-validated" if kept is not None else "not yet validated",
        n_held=len(held), date=date.today().isoformat()))
    print(json.dumps({"dir": str(out), "corpus": n_corpus, "eval_queries": n_eval, "held_out": len(held)}))


if __name__ == "__main__":
    main()
