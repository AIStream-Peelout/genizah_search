"""Assemble the v3 training / eval dataset folder ``AUDIT_ROOT/hf_dataset/v3/`` (does NOT upload).

Uploading is a separate, owner-run step: ``python3 colab/push_dataset.py --name v3`` (tags the commit ``v3``, which
the notebook's ``DATA_REVISION`` pins).

Files (consumed by ``train_embedder_v3.load_data``):

* ``train_mix_v3.jsonl``    -- the T3 mixture, copied unchanged.
* ``train_pool.jsonl``      -- the gated train records ``{doc_id, text, text_hash, subjects}``: eligible, not held out,
  no held-out / linked subject, no text shared with a held-out record, leak filter passed; ``text`` = semantic text
  with cited shelf marks / PGPIDs masked as ``[shelfmark]``. Reconstructed with ``build_training_mix_v3.MixBuilder``
  and the parameters recorded in ``train_mix_v3.stats.json``; every mixture positive / d2d anchor is checked to be
  byte-identical to its pool text. The ONLY source of mined negatives.
* ``corpus_semantic.jsonl`` -- every record ``{doc_id, text, eligible, text_hash, series}`` (v3 semantic text, shelf marks
  already masked by ``semantic_text.py``: what the eval embeds and what production would embed).
* ``corpus_original.jsonl`` -- every record ``{doc_id, text}`` = production embedding text (``corpus_v1.jsonl``, with
  ``Document ID:`` / ``Shelf Mark:`` lines): the "as today" baseline.
* ``eval/``                 -- the frozen eval set (``AUDIT_ROOT/v3/eval``), copied unchanged.
* ``eval_v3.py``, ``train_embedder_v3.py`` -- the scoring and training code, versioned with the data.
* ``README.md`` (dataset card), ``manifest.json`` (md5 / bytes / rows per file, counts, verification results).

JSON only, no model (~1-3 min, < 1 GB)::

    python3 colab/package_dataset_v3.py            # -> AUDIT_ROOT/hf_dataset/v3

Final run (all subjects trained; the six former held-out subjects scored as the ``focus`` group): the mixture built
from ``v3/eval_final`` is packaged under its own name, with ``eval_final`` as the package's ``eval/`` (plus
``focus_subjects.json``). Inside the package the mixture keeps the name ``train_mix_v3.jsonl`` (the trainer's
contract); ``manifest.json`` records its source file::

    python3 colab/package_dataset_v3.py --name v3-final --eval-dir eval_final --mix train_mix_v3final
    # -> AUDIT_ROOT/hf_dataset/v3-final (not uploaded; push_dataset.py --name v3-final tags it)

A package folder is never replaced by one built from a different mixture file (``--overwrite`` to force).
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))

from build_subjects import iter_jsonl  # noqa: E402
from build_train_pairs import split_text  # noqa: E402
from build_training_mix_v3 import (MixBuilder, build_leak_filter, held_norm_keys, load_eval,  # noqa: E402
                                   load_semantic, load_subjects, norm_key, own_shelfmarks)
from build_training_mix_v3 import verify as verify_mix  # noqa: E402
from embed_utils import AUDIT_ROOT  # noqa: E402
from semantic_text import EXTRA_ID_RE, PROD_SHELFMARK_RE  # noqa: E402

V3 = AUDIT_ROOT / "v3"
EVAL_FILES = ("held_out_subjects.json", "held_out_records.json", "subject_queries.json", "subject_relevance.json",
              "known_item.jsonl", "memorization_pairs.json", "collection_sample.json", "eval_pool.jsonl",
              "README.md", "build_stats.json")
OPTIONAL_EVAL_FILES = ("focus_subjects.json",)  # final-run eval dirs only
FULL_TEXT_FAMILIES_PREFIX = ("synthetic_", "subject", "d2d_")  # positive = the record's whole pool text

README = """---
license: other
---
# genizah-embed-train (v3: semantic text, subject-generalisation gate)

Private training / evaluation data for fine-tuning the Cairo Genizah retrieval embedder (base
`Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1024-d, cosine, last-token pooling; queries
prefixed `Instruct: Given a search query, retrieve relevant passages\\nQuery: `, documents raw).

Goal: put related SUBJECTS close together (e.g. "laws of lulav" near Sukkot records; Hebrew and English names of the
same thing together) without learning literal text -> record or shelf-mark matching.

| file | content |
|---|---|
| `train_mix_v3.jsonl` | {n_train} pairs: q2d {n_q2d}, t2t {n_t2t}, d2d {n_d2d}; mask levels subject {n_subject} / doc {n_doc}. Families: {families} |
| `train_pool.jsonl` | {n_pool} gated train records (masked semantic text + subject ids): the only negative source |
| `corpus_semantic.jsonl` | {n_corpus} records, v3 semantic text ({n_eligible} eligible = the eval pool) |
| `corpus_original.jsonl` | {n_corpus} records, production embedding text (with Document ID / Shelf Mark lines) |
| `eval/` | frozen eval set (6 held-out subjects = the gate, linked + seen subjects, known-item, memorisation, collection, hubness); see `eval/README.md` |
| `eval_v3.py` | scorer (pure numpy) |
| `train_embedder_v3.py` | trainer: hard-negative mining, masked GradCache InfoNCE, 3-way evaluation, push |
| `manifest.json` | md5 / size / rows per file, build counts and verification results |

Hold-out guarantees (verified when this folder was built, re-checked by `train_embedder_v3.load_data`): no train
pair or pool record is a held-out record ({n_held} records), carries a held-out or linked subject, or shares text
with a held-out record; mixture positives are byte-identical to their pool text.

Masking contract: rows carry `mask_level` (`subject` / `doc`), `mask_ids` (positive's subject ids + `D:<doc_id>`),
`anchor_mask_ids` (d2d anchor record), `anchor_subject` (subject family), `concept_ids` (t2t). The trainer turns
them into a numeric `label` column and sets in-batch candidates that share ids with the row to -inf (KTIV `domain:`
genre ids are not used for relevance unless the anchor names that domain). See `train_embedder_v3.py`.

Built by `genizah_search/evals/embedding_audit/colab/package_dataset_v3.py` on {date} from
`AUDIT_ROOT/v3` (T1 `semantic_text.py` / `build_subjects.py`, T2 `build_eval_v3.py`, T3 `build_training_mix_v3.py`).
Synthetic queries were written by Claude sub-agents following `query_writing_spec*.md`; Sefaria-derived pairs follow
Sefaria's per-text licences. This split is for the EVALUATION run; a production model is retrained on everything.
"""

README_FINAL = """---
license: other
---
# genizah-embed-train ({name}: semantic text, FINAL run, every subject trained)

Private training / evaluation data for the FINAL fine-tune of the Cairo Genizah retrieval embedder (base
`Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1024-d, cosine, last-token pooling; queries
prefixed `Instruct: Given a search query, retrieve relevant passages\\nQuery: `, documents raw). Same recipe as the
`v3` package (families, masking contract, mining, trainer); the difference is the hold-out: the six subjects that `v3`
held out to prove generalisation (topic:sukkot, topic:pesach, topic:kashrut, pgp:partnership, pgp:slavery,
pgp:geonic-academies) and their linked subjects are now TRAINED ON.

| file | content |
|---|---|
| `train_mix_v3.jsonl` | {n_train} pairs (the v3-final mixture, source `{mix_name}`): q2d {n_q2d}, t2t {n_t2t}, d2d {n_d2d}; mask levels subject {n_subject} / doc {n_doc}. Families: {families} |
| `train_pool.jsonl` | {n_pool} gated train records (masked semantic text + subject ids): the only negative source |
| `corpus_semantic.jsonl` | {n_corpus} records, v3 semantic text ({n_eligible} eligible = the eval pool) |
| `corpus_original.jsonl` | {n_corpus} records, production embedding text (with Document ID / Shelf Mark lines) |
| `eval/` | frozen final eval set: no held-out subject (no gate); the six as the `focus` group scored over held-out records only (`eval/focus_subjects.json`), seen subjects, known-item, memorisation, collection, hubness; see `eval/README.md` |
| `eval_v3.py` | scorer (pure numpy) |
| `train_embedder_v3.py` | trainer: hard-negative mining, masked GradCache InfoNCE, evaluation (incl. a previous model), push |
| `manifest.json` | md5 / size / rows per file, build counts and verification results |

Hold-out guarantees (verified when this folder was built, re-checked by `train_embedder_v3.load_data`): no train
pair or pool record is a held-out record ({n_held} records: the v3 20 % split, Sukkot gold, synthetic-query eval
pool, identical-text closure; a subset of v3's, so v3-trained models never saw them either) or shares its text, even
up to niqqud / case / punctuation / spacing; no focus-subject query (probe / llm_t2 / name) or known-item query is a
training anchor or a near-copy of one on either side of a pair (leak filter: exact, also ignoring spacing inside
words; containment of queries of >= 3 tokens, and of two-token focus probe / llm_t2 queries inside anchors; content
Jaccard >= 0.5). One-word focus queries ("Haggadah", "סוכות") do occur inside longer anchors and record texts: the
subjects are trained on. Mixture positives are byte-identical to their pool text.

Built by `genizah_search/evals/embedding_audit/colab/package_dataset_v3.py` on {date} from `AUDIT_ROOT/v3`
(`build_eval_v3.py --final`, `build_training_mix_v3.py --eval-dir v3/eval_final`).
"""


def resolve(value: str, base: Path, suffix: str = "") -> Path:
    """Resolve a short name (``eval_final``, ``train_mix_v3final``) under ``base``; paths are kept as given.

    :param value: Name or path.
    :param base: Directory for bare names (``AUDIT_ROOT/v3``).
    :param suffix: Suffix added to a bare name without one (``.jsonl`` for mixtures).
    :returns: Path.
    :rtype: Path
    """
    p = Path(value)
    if p.is_absolute() or p.exists():
        return p
    return base / (value + suffix if suffix and not value.endswith(suffix) else value)


def md5_file(path: Path) -> str:
    """md5 of a file's bytes.

    :param path: File.
    :returns: Hex digest.
    :rtype: str
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write rows as JSONL.

    :param path: Target file.
    :param rows: Row dicts.
    :returns: Rows written.
    :rtype: int
    """
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def gate_pool(params: dict) -> tuple:
    """Rebuild T3's gated, masked train records with the parameters T3 recorded.

    :param params: ``train_mix_v3.stats.json["params"]``.
    :returns: (pool rows ``doc_id -> {doc_id, text, text_hash, subjects}``, gate stats, eval info from ``load_eval``,
        semantic rows, leak filter).
    :rtype: tuple
    """
    ev = load_eval(Path(params["eval_dir"]))
    strict = params.get("final_strict", "off") == "on"
    leak, _ = build_leak_filter(ev, params["probe_scope"], params["min_contain"], params["min_shared"],
                                params.get("focus_names", "block"), params.get("he_prefix_contain", "off") == "on",
                                strict)
    rows, _ = load_semantic(Path(params["semantic"]))
    subj, weak = load_subjects(Path(params["subjects"]))
    marks = own_shelfmarks(Path(params["corpus_v1"]))
    builder = MixBuilder(ev, leak, rows, subj, weak, marks, random.Random(params["seed"]), norm_gate=strict)
    pool = {d: {"doc_id": d, "text": builder.train[d], "text_hash": rows[d]["text_hash"],
                "subjects": sorted(subj.get(d, set()))} for d in sorted(builder.train)}
    return pool, dict(builder.gate_stats), ev, rows, leak


def verify(mix: List[dict], pool: Dict[str, dict], ev: dict, sem_rows: Dict[str, dict], eval_pool: List[dict],
           original_ids: set, norm_gate: bool = False) -> Counter:
    """Package checks; every counter must be zero.

    :param mix: Mixture rows.
    :param pool: Gated pool.
    :param ev: ``build_training_mix_v3.load_eval`` output.
    :param sem_rows: Semantic rows (``load_semantic``).
    :param eval_pool: Frozen eval pool rows.
    :param original_ids: Doc ids present in ``corpus_original.jsonl``.
    :param norm_gate: Final run (``final_strict``): a pool record whose text equals a held-out record's up to niqqud /
        case / punctuation / spacing is a violation too.
    :returns: Violation counts (only non-zero keys mean failure) plus ``checked_*`` info counts.
    :rtype: Counter
    """
    out: Counter = Counter()
    held = ev["held"]
    held_hashes = {r["text_hash"] for d, r in sem_rows.items() if d in held and r["eligible"]}
    held_norms = held_norm_keys(sem_rows, held) if norm_gate else set()
    blocked = ev["held_subjects"] | ev["linked"]
    for d, r in pool.items():
        out["pool_held_out"] += d in held
        out["pool_held_text"] += r["text_hash"] in held_hashes
        if norm_gate:
            out["pool_held_norm_text"] += norm_key(r["text"]) in held_norms
        out["pool_blocked_subject"] += bool(set(r["subjects"]) & blocked)
        out["pool_not_eligible"] += not sem_rows[d]["eligible"]
        out["pool_id_line"] += ("Document ID:" in r["text"]) or ("Shelf Mark:" in r["text"])
        # the semantic text is already masked, so the pool text IS the semantic text and T:<text_hash> ids match it
        out["pool_text_not_semantic_text"] += hashlib.md5(r["text"].encode("utf-8")).hexdigest() != r["text_hash"]
        out["pool_shelfmark_regex"] += bool(PROD_SHELFMARK_RE.search(r["text"]) or EXTRA_ID_RE.search(r["text"]))
    for r in mix:
        if r["kind"] == "t2t":
            out["t2t_with_doc"] += r["doc_id"] is not None
            continue
        d = r["doc_id"]
        if d not in pool:
            out["positive_not_in_pool"] += 1
            continue
        out["text_hash_mismatch"] += r["text_hash"] != pool[d]["text_hash"]
        if r["family"] == "content":
            out["content_positive_mismatch"] += r["positive"] != split_text(pool[d]["text"])[0]
        elif r["family"].startswith(FULL_TEXT_FAMILIES_PREFIX):
            out["positive_mismatch"] += r["positive"] != pool[d]["text"]
        else:
            out["unknown_family"] += 1
        if r["kind"] == "d2d":
            a = r.get("anchor_doc_id")
            if a not in pool:
                out["anchor_not_in_pool"] += 1
            else:
                out["d2d_anchor_mismatch"] += r["anchor"] != pool[a]["text"]
        out["checked_rows"] += 1
    for r in eval_pool:
        s = sem_rows.get(r["doc_id"])
        out["eval_pool_not_eligible"] += s is None or not s["eligible"]
        if s is not None:
            out["eval_pool_shelfmark_regex"] += bool(PROD_SHELFMARK_RE.search(s["text"]) or EXTRA_ID_RE.search(s["text"]))
            out["eval_pool_id_line"] += ("Document ID:" in s["text"]) or ("Shelf Mark:" in s["text"])
        out["eval_pool_hash_mismatch"] += s is not None and s["text_hash"] != r["text_hash"]
        out["eval_pool_no_original"] += r["doc_id"] not in original_ids
    out["checked_pool"] = len(pool)
    out["checked_eval_pool"] = len(eval_pool)
    return out


def main() -> None:
    """Build, verify, then move the folder into place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v3")
    parser.add_argument("--mix", default=str(V3 / "train_mix_v3.jsonl"),
                        help="mixture file, or a bare name under v3/ (train_mix_v3final)")
    parser.add_argument("--eval-dir", default=None,
                        help="eval dir, or a bare name under v3/ (eval_final); must be the one the mixture was built "
                             "from (default: read from the mixture's stats)")
    parser.add_argument("--corpus-v1", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing package folder built from a different mixture file")
    args = parser.parse_args()
    out = AUDIT_ROOT / "hf_dataset" / args.name
    mix_path = resolve(args.mix, V3, ".jsonl")
    params = json.loads(mix_path.with_suffix(".stats.json").read_text())["params"]
    eval_dir = Path(params["eval_dir"])
    if args.eval_dir and resolve(args.eval_dir, V3).resolve() != eval_dir.resolve():
        raise SystemExit(f"--eval-dir {args.eval_dir} is not the eval dir {eval_dir} that {mix_path.name} was built from")
    if out.exists() and not args.overwrite:
        old = json.loads((out / "manifest.json").read_text())["sources"]["train_mix_v3"]["path"]
        if Path(old).resolve() != mix_path.resolve():
            raise SystemExit(f"{out} was built from {old}, not {mix_path}: use another --name (or --overwrite)")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "eval").mkdir(parents=True)
    final = (eval_dir / "focus_subjects.json").exists()
    eval_files = EVAL_FILES + tuple(f for f in OPTIONAL_EVAL_FILES if (eval_dir / f).exists())
    pool, gate_stats, ev, sem_rows, leak = gate_pool(params)
    print(json.dumps({"gate": gate_stats}), flush=True)
    mix = list(iter_jsonl(mix_path))
    eval_pool = list(iter_jsonl(eval_dir / "eval_pool.jsonl"))

    counts = {"train_pool": write_jsonl(tmp / "train_pool.jsonl", pool.values())}
    shutil.copyfile(mix_path, tmp / "train_mix_v3.jsonl")
    counts["train_mix_v3"] = len(mix)
    counts["corpus_semantic"] = write_jsonl(tmp / "corpus_semantic.jsonl", (
        {k: r[k] for k in ("doc_id", "text", "eligible", "text_hash", "series")}
        for r in iter_jsonl(Path(params["semantic"]))))
    original_ids = set()

    def originals() -> Iterable[dict]:
        for r in iter_jsonl(Path(args.corpus_v1)):
            original_ids.add(r["doc_id"])
            yield {"doc_id": r["doc_id"], "text": r["text"]}

    counts["corpus_original"] = write_jsonl(tmp / "corpus_original.jsonl", originals())
    for name in eval_files:
        shutil.copyfile(eval_dir / name, tmp / "eval" / name)
    shutil.copyfile(AUDIT / "eval_v3.py", tmp / "eval_v3.py")
    shutil.copyfile(HERE / "train_embedder_v3.py", tmp / "train_embedder_v3.py")

    strict = params.get("final_strict", "off") == "on"
    checks = verify(mix, pool, ev, sem_rows, eval_pool, original_ids, norm_gate=strict)
    # re-scan every pair with T3's own verifier (held-out records / texts / subjects, ID lines, shelf marks, leak filter
    # on both sides); its info_ counters are informational
    for k, v in verify_mix(mix, ev, sem_rows, leak, norm_gate=strict).items():
        checks[f"checked_mix_{k}" if k.startswith("info_") else f"mix_{k}"] += v
    for name in eval_files:
        checks["eval_copy_mismatch"] += md5_file(eval_dir / name) != md5_file(tmp / "eval" / name)
    checks["corpus_count_mismatch"] += counts["corpus_semantic"] != counts["corpus_original"]
    violations = {k: v for k, v in checks.items() if v and not k.startswith("checked_")}
    stats = json.loads(mix_path.with_suffix(".stats.json").read_text())
    kinds, levels = Counter(r["kind"] for r in mix), Counter(r["mask_level"] for r in mix)
    fmt = dict(n_train=len(mix), n_q2d=kinds["q2d"], n_t2t=kinds["t2t"], n_d2d=kinds["d2d"], n_subject=levels["subject"],
               n_doc=levels["doc"], families=", ".join(f"{k} {v}" for k, v in stats["by_family"].items()),
               n_pool=len(pool), n_corpus=counts["corpus_semantic"], n_eligible=len(eval_pool),
               n_held=len(ev["held"]), date=datetime.now().date().isoformat())
    readme = README_FINAL.format(name=args.name, mix_name=mix_path.name, **fmt) if final else README.format(**fmt)
    (tmp / "README.md").write_text(readme)
    files = {}
    for p in sorted(tmp.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = str(p.relative_to(tmp))
            files[rel] = {"md5": md5_file(p), "bytes": p.stat().st_size}
            if p.suffix == ".jsonl":
                files[rel]["rows"] = sum(1 for _ in open(p, encoding="utf-8"))
    gate_match = gate_stats.get("train_records") == stats["records"]["train_records"]
    manifest = {"name": args.name, "built_at": datetime.now().isoformat(timespec="seconds"),
                "sources": {"train_mix_v3": {"path": str(mix_path), "md5": md5_file(mix_path)},
                            "eval_dir": str(eval_dir), "semantic": params["semantic"], "corpus_v1": args.corpus_v1,
                            "subjects": params["subjects"], "gate_params": params},
                "summary": {"train_pairs": len(mix), "by_kind": dict(kinds), "by_mask_level": dict(levels),
                            "train_pool": len(pool), "corpus": counts["corpus_semantic"],
                            "eval_pool": len(eval_pool), "held_out_records": len(ev["held"]),
                            "mode": "final" if final else "v3",
                            "held_out_subjects": sorted(ev["held_subjects"]), "focus_subjects": sorted(ev["focus"])},
                "gate_stats": gate_stats, "gate_matches_t3_stats": gate_match,
                "verification": dict(checks), "violations": violations, "files": files}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({"counts": counts, "verification": dict(checks), "violations": violations,
                      "gate_matches_t3_stats": gate_match}, indent=1), flush=True)
    if violations or not gate_match:
        raise SystemExit(f"package NOT moved into place (left at {tmp}): violations={violations}, "
                         f"gate_matches_t3_stats={gate_match}")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
