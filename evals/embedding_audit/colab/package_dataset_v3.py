"""Assemble the v3 training / eval dataset folder ``AUDIT_ROOT/hf_dataset/v3/`` (does NOT upload).

Uploading is a separate, owner-run step: ``python3 colab/push_dataset.py --name v3`` (tags the commit ``v3``, which
the notebook's ``DATA_REVISION`` pins).

Files (consumed by ``train_embedder_v3.load_data``):

* ``train_mix_v3.jsonl``    -- the T3 mixture, copied unchanged.
* ``train_pool.jsonl``      -- the gated train records ``{doc_id, text, text_hash, subjects}``: eligible, not held out,
  no held-out / linked subject, no text shared with a held-out record, leak filter passed; ``text`` = semantic text
  with cited shelf marks / PGPIDs masked as ``[shelfmark]``; ``subjects`` = every subject id of the record plus
  ``M:<sid>`` for each subject its catalogue mentions (``subjects_v1`` ``mentions``). Reconstructed with
  ``build_training_mix_v3.MixBuilder`` and the parameters recorded in ``train_mix_v3.stats.json``; every mixture
  positive / d2d anchor is checked to be byte-identical to its pool text, and its record mask ids to equal the pool
  record's ids. The ONLY source of mined negatives.
* ``corpus_semantic.jsonl`` -- every record ``{doc_id, text, eligible, text_hash, series}`` (+ ``text_sha256`` /
  ``text_version`` when the corpus has them): the EMBEDDED text (``semantic_v4`` or ``semantic_v3``), what the eval
  embeds and what production would embed.
* ``corpus_semantic_v3.jsonl`` -- the same records with the ``semantic_v3`` text (the corpus column ``text_v3``; the
  v9 text): for the reference models V (v9) and B3 (base on the v9 text). ``text_hash`` = md5 of that text (the eval
  pool's ``text_hash_v3``); ``eligible`` = the embedded text's eligibility, so every model is scored on one pool.
* ``corpus_original.jsonl`` -- every record ``{doc_id, text}`` = production embedding text (``corpus_v1.jsonl``, with
  ``Document ID:`` / ``Shelf Mark:`` lines): the "as today" baseline (P).
* ``v9_trained_ids.json``   -- every doc id that is a positive or a d2d anchor of the v3-final mixture
  (``v3/train_mix_v3final.jsonl``, what v9 was trained on): passed as ``ignore_ids`` when scoring V (text regrouping
  can move records v9 trained on onto the held-out side). The mixture stores no negatives (they were mined in Colab).
* ``eval/``                 -- the frozen eval set (``AUDIT_ROOT/v3/eval``), copied unchanged, incl.
  ``holiday_queries.json`` and the acceptance-gate dir ``eval/gate_v1/`` when present.
* ``eval_v3.py``, ``eval_gate.py``, ``train_embedder_v3.py`` -- the scoring and training code, versioned with the data
  (``eval_gate.py`` when it exists; ``--require-gate`` makes it, ``gate_v1/`` and ``holiday_queries.json`` mandatory).
* ``corpus_<name>.jsonl``   -- optional extra text variants (``--extra-corpus name=path``, e.g. the no-frames v4 text).
* ``README.md`` (dataset card), ``manifest.json`` (md5 / bytes / rows per file, counts, guard stats, verification).

JSON only, no model (~1-3 min, < 1 GB)::

    python3 colab/package_dataset_v3.py            # -> AUDIT_ROOT/hf_dataset/v3

Final run (all subjects trained; the six former held-out subjects scored as the ``focus`` group): the mixture built
from ``v3/eval_final`` is packaged under its own name, with ``eval_final`` as the package's ``eval/`` (plus
``focus_subjects.json``). Inside the package the mixture keeps the name ``train_mix_v3.jsonl`` (the trainer's
contract); ``manifest.json`` records its source file::

    python3 colab/package_dataset_v3.py --name v3-final --eval-dir eval_final --mix train_mix_v3final
    # -> AUDIT_ROOT/hf_dataset/v3-final (not uploaded; push_dataset.py --name v3-final tags it)
    python3 colab/package_dataset_v3.py --name v4-final --mix $V4/train_mix_v4final.jsonl --eval-dir $V4/eval_final \\
        --corpus-v1 $V4/corpus_v1.jsonl --require-gate

Eval-only package (the text-function ablation, plan step 7: stock and v9 models only, no mixture and no train pool;
``train_embedder_v3.load_data`` needs the training files, so the notebook scores such a package with
``eval_v3`` / ``eval_gate`` directly)::

    python3 colab/package_dataset_v3.py --eval-only --name v4-ablation --semantic $V4/corpus_semantic.jsonl \\
        --eval-dir $V4/eval_final --corpus-v1 $V4/corpus_v1.jsonl \\
        --extra-corpus semantic_nf=$V4/ablation/corpus_semantic_nf.jsonl

An existing package folder is never replaced (``--overwrite`` to force); a failed build is left in ``<name>.tmp``.
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))

from build_subjects import iter_jsonl  # noqa: E402
from build_train_pairs import split_text  # noqa: E402
from build_training_mix_v3 import (MixBuilder, build_leak_filter, held_norm_keys, load_eval,  # noqa: E402
                                   load_gate_strings, load_semantic, load_subject_labels, load_subjects, norm_key,
                                   own_shelfmarks)
from build_training_mix_v3 import verify as verify_mix  # noqa: E402
from embed_utils import AUDIT_ROOT  # noqa: E402
from semantic_text import EXTRA_ID_RE, PROD_SHELFMARK_RE  # noqa: E402

V3 = AUDIT_ROOT / "v3"
EVAL_FILES = ("held_out_subjects.json", "held_out_records.json", "subject_queries.json", "subject_relevance.json",
              "known_item.jsonl", "memorization_pairs.json", "collection_sample.json", "eval_pool.jsonl",
              "README.md", "build_stats.json")
OPTIONAL_EVAL_FILES = ("focus_subjects.json",  # final-run eval dirs only
                       "holiday_queries.json")  # eval dirs built after 2026-10-08 (eval_v3 "holiday" section)
GATE_DIR = "gate_v1"  # acceptance gate (build_gate_v1.py), shipped as eval/gate_v1/ when present
GATE_FILES = ("queries.jsonl", "concepts.json", "records.jsonl", "probes.json", "thresholds.json")
GATE_REQUIRED_EVAL_FILES = ("holiday_queries.json",)  # also mandatory with --require-gate
CODE_FILES = ((AUDIT / "eval_v3.py", True), (AUDIT / "eval_gate.py", False),
              (HERE / "train_embedder_v3.py", True))  # (source, always required)
SEMANTIC_V3_FILE = "corpus_semantic_v3.jsonl"
V9_TRAINED_FILE = "v9_trained_ids.json"
V9_MIX = V3 / "train_mix_v3final.jsonl"
SEMANTIC_KEYS = ("doc_id", "text", "eligible", "text_hash", "series")
SEMANTIC_OPTIONAL_KEYS = ("text_sha256", "text_version")
BROAD_ID_PREFIXES = ("domain:", "M:")  # train_embedder_v3.BROAD_PREFIXES (contract item 5)
FULL_TEXT_FAMILIES_PREFIX = ("synthetic_", "subject", "d2d_")  # positive = the record's whole pool text

MASKING = """Masking contract: rows carry `mask_level` (`subject` / `doc`), `mask_ids` (the positive record's ids: every
subject id, strong and weak, `M:<sid>` for each subject its catalogue mentions without carrying it strongly, and
`D:<doc_id>`), `anchor_mask_ids` (d2d anchor record, same id set), `anchor_subject` (subject family),
`anchor_mention_id` (`M:<anchor_subject>` on subject-family rows), `anchor_topic_ids` (non-content query rows whose
anchor names a festival: the `topic:<f>` ids it names and the matching `M:topic:<f>` ids; absent when the anchor is a
canon / known-item reference) and `concept_ids` (t2t). `train_pool.jsonl` `subjects` holds the same ids per record
(incl. `M:`), so mined negatives are masked like in-batch ones. The trainer turns the rows into a numeric `label`
column and sets every in-batch or mined candidate that shares a relevance id with the row to -inf: relevance =
the positive's ids without the broad prefixes `domain:` / `M:`, except the ones the row keeps (`anchor_subject`,
`anchor_mention_id`, `anchor_topic_ids`). So a festival-name query never treats another record of that festival, or
one whose catalogue names it, as a negative. See `train_embedder_v3.py` and `manifest.json["guards"]`."""

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
| `train_pool.jsonl` | {n_pool} gated train records (masked semantic text + subject ids incl. `M:` mention ids): the only negative source |
| `corpus_semantic.jsonl` | {n_corpus} records, embedded text `{text_version}` ({n_eligible} eligible = the eval pool) |
| `corpus_semantic_v3.jsonl` | {n_corpus} records, semantic_v3 text (the v9 text, for v9 and the base model on it) |
| `corpus_original.jsonl` | {n_corpus} records, production embedding text (with Document ID / Shelf Mark lines) |
| `v9_trained_ids.json` | {n_v9} doc ids v9 was trained on (positives / d2d anchors of the v3-final mixture): `ignore_ids` when scoring v9 |
| `eval/` | frozen eval set (6 held-out subjects = the gate, linked + seen subjects, known-item, memorisation, collection, hubness{eval_extra}); see `eval/README.md` |
| `eval_v3.py`{gate_code} | scorer (pure numpy) |
| `train_embedder_v3.py` | trainer: hard-negative mining, masked GradCache InfoNCE, 3-way evaluation, push |
| `manifest.json` | md5 / size / rows per file, build counts, guard stats and verification results |

Hold-out guarantees (verified when this folder was built, re-checked by `train_embedder_v3.load_data`): no train
pair or pool record is a held-out record ({n_held} records), carries a held-out or linked subject, or shares text
with a held-out record; mixture positives are byte-identical to their pool text.

{masking}

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
| `train_mix_v3.jsonl` | {n_train} pairs (the mixture, source `{mix_name}`): q2d {n_q2d}, t2t {n_t2t}, d2d {n_d2d}; mask levels subject {n_subject} / doc {n_doc}. Families: {families} |
| `train_pool.jsonl` | {n_pool} gated train records (masked semantic text + subject ids incl. `M:` mention ids): the only negative source |
| `corpus_semantic.jsonl` | {n_corpus} records, embedded text `{text_version}` ({n_eligible} eligible = the eval pool) |
| `corpus_semantic_v3.jsonl` | {n_corpus} records, semantic_v3 text (the v9 text, for v9 and the base model on it) |
| `corpus_original.jsonl` | {n_corpus} records, production embedding text (with Document ID / Shelf Mark lines) |
| `v9_trained_ids.json` | {n_v9} doc ids v9 was trained on (positives / d2d anchors of the v3-final mixture): `ignore_ids` when scoring v9 |
| `eval/` | frozen final eval set: no held-out subject (no gate); the six as the `focus` group scored over held-out records only (`eval/focus_subjects.json`), seen subjects, known-item, memorisation, collection, hubness{eval_extra}; see `eval/README.md` |
| `eval_v3.py`{gate_code} | scorer (pure numpy) |
| `train_embedder_v3.py` | trainer: hard-negative mining, masked GradCache InfoNCE, evaluation (incl. a previous model), push |
| `manifest.json` | md5 / size / rows per file, build counts, guard stats and verification results |

Hold-out guarantees (verified when this folder was built, re-checked by `train_embedder_v3.load_data`): no train
pair or pool record is a held-out record ({n_held} records: the 20 % split, Sukkot gold, synthetic-query eval
pool, identical-text closure) or shares its text, even up to niqqud / case / punctuation / spacing; no focus-subject
query (probe / llm_t2 / name), known-item query or blocked acceptance-gate query is a training anchor or a near-copy
of one on either side of a pair (leak filter: exact, also ignoring spacing inside words; containment of queries of
>= 3 tokens, and of two-token focus probe / llm_t2 queries inside anchors; content Jaccard >= 0.5). One-word focus
queries ("Haggadah", "סוכות") do occur inside longer anchors and record texts: the subjects are trained on. Mixture
positives are byte-identical to their pool text.

{masking}

Built by `genizah_search/evals/embedding_audit/colab/package_dataset_v3.py` on {date}
(`build_eval_v3.py --final`, `build_training_mix_v3.py`; sources in `manifest.json`).
"""

README_EVAL_ONLY = """---
license: other
---
# genizah-embed-eval ({name}: eval-only package, no training data)

Frozen evaluation data for scoring stock / previously tuned Cairo Genizah retrieval embedders on several text
variants (base `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; queries prefixed
`Instruct: Given a search query, retrieve relevant passages\\nQuery: `, documents raw). No mixture, no train pool.

| file | content |
|---|---|
| `corpus_semantic.jsonl` | {n_corpus} records, embedded text `{text_version}` ({n_eligible} eligible = the eval pool) |
| `corpus_semantic_v3.jsonl` | {n_corpus} records, semantic_v3 text (the v9 text) |
| `corpus_original.jsonl` | {n_corpus} records, production embedding text (with Document ID / Shelf Mark lines) |
{extra_rows}| `v9_trained_ids.json` | {n_v9} doc ids v9 was trained on: `ignore_ids` when scoring v9 |
| `eval/` | frozen eval set{eval_extra}; see `eval/README.md` |
| `eval_v3.py`{gate_code} | scorer (pure numpy) |
| `train_embedder_v3.py` | encoder / evaluation helpers (its `load_data` expects a training package) |
| `manifest.json` | md5 / size / rows per file and verification results |

Every model is scored on the eval pool (`eval/eval_pool.jsonl`) with its own text variant.

Built by `genizah_search/evals/embedding_audit/colab/package_dataset_v3.py --eval-only` on {date}.
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


# ---------------------------------------------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------------------------------------------
def semantic_rows(path: Path) -> Iterable[dict]:
    """Rows of a semantic corpus as shipped: the embedded text and its keys (``text_v3`` is shipped separately).

    :param path: ``corpus_semantic.jsonl`` (any text version).
    :yields: ``{doc_id, text, eligible, text_hash, series}`` plus ``text_sha256`` / ``text_version`` when present.
    """
    for r in iter_jsonl(path):
        out = {k: r[k] for k in SEMANTIC_KEYS}
        out.update({k: r[k] for k in SEMANTIC_OPTIONAL_KEYS if k in r})
        yield out


def semantic_v3_rows(path: Path) -> Iterable[dict]:
    """Rows of the semantic_v3 (v9) text variant, from the corpus column ``text_v3`` (``text`` in older corpora).

    :param path: ``corpus_semantic.jsonl``.
    :yields: ``{doc_id, text, eligible, text_hash, text_sha256, series, text_version}``; ``text_hash`` / ``text_sha256``
        are of the semantic_v3 text, ``eligible`` is the embedded text's flag (one pool for every model).
    """
    for r in iter_jsonl(path):
        text = r.get("text_v3") or r["text"]
        yield {"doc_id": r["doc_id"], "text": text, "eligible": r["eligible"],
               "text_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
               "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "series": r["series"],
               "text_version": "semantic_v3"}


def corpus_text_version(path: Path) -> str:
    """Text version of a corpus (first row's ``text_version``; corpora written before it existed are semantic_v3).

    :param path: ``corpus_semantic.jsonl``.
    :returns: Version name.
    :rtype: str
    """
    first = next(iter(iter_jsonl(path)), {})
    return first.get("text_version") or "semantic_v3"


def parse_extra_corpora(specs: List[str]) -> Dict[str, Path]:
    """``name=path`` specs of extra text variants.

    :param specs: ``--extra-corpus`` values.
    :returns: ``name -> path`` (shipped as ``corpus_<name>.jsonl``).
    :rtype: Dict[str, Path]
    :raises SystemExit: On a malformed spec, a reserved or repeated name, or a missing file.
    """
    out: Dict[str, Path] = {}
    reserved = {"semantic", "semantic_v3", "original"}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name.replace("_", "").isalnum() or name in reserved or name in out:
            raise SystemExit(f"--extra-corpus {spec!r}: expected a new name=path (reserved: {sorted(reserved)})")
        if not Path(path).exists():
            raise SystemExit(f"--extra-corpus {spec!r}: {path} does not exist")
        out[name] = Path(path)
    return out


def v9_trained_ids(mix_path: Path, eval_pool: List[dict]) -> dict:
    """Doc ids v9 trained on: positives (``doc_id``) and d2d anchors (``anchor_doc_id``) of the v3-final mixture.

    :param mix_path: ``v3/train_mix_v3final.jsonl``.
    :param eval_pool: Rows of the package's eval pool (for the overlap counts).
    :returns: ``{source, source_md5, rule, n, n_in_eval_pool, n_held_out_in_eval_pool, doc_ids}``.
    :rtype: dict
    :raises SystemExit: When the mixture file does not exist.
    """
    if not mix_path.exists():
        raise SystemExit(f"--v9-mix {mix_path} does not exist (needed for {V9_TRAINED_FILE})")
    ids: Set[str] = set()
    for r in iter_jsonl(mix_path):
        ids.update(d for d in (r.get("doc_id"), r.get("anchor_doc_id")) if d)
    pool = {r["doc_id"]: r for r in eval_pool}
    return {"source": str(mix_path), "source_md5": md5_file(mix_path),
            "rule": "doc_id (positive) and anchor_doc_id (d2d anchor) of every mixture row; the mixture stores no "
                    "negatives (mined in Colab), so they are not included",
            "n": len(ids), "n_in_eval_pool": sum(d in pool for d in ids),
            "n_held_out_in_eval_pool": sum(bool(pool[d].get("held_out")) for d in ids if d in pool),
            "doc_ids": sorted(ids)}


def record_mentions(subjects_path: Path, blocked: Set[str]) -> Dict[str, List[str]]:
    """``M:<sid>`` mention ids per record (``subjects_v1`` ``mentions``; empty for files written before v10).

    Mentions of held-out / linked subjects are left out, as in ``build_training_mix_v3.MixBuilder.mention_ids``.

    :param subjects_path: ``subjects_v1.jsonl``.
    :param blocked: Held-out and linked subject ids.
    :returns: ``doc_id -> sorted M: ids`` (records without mentions omitted).
    :rtype: Dict[str, List[str]]
    """
    out = {}
    for r in iter_jsonl(subjects_path):
        ids = sorted(f"M:{s}" for s in r.get("mentions") or [] if s not in blocked)
        if ids:
            out[r["doc_id"]] = ids
    return out


# ---------------------------------------------------------------------------------------------------------------
# eval dir and code
# ---------------------------------------------------------------------------------------------------------------
def eval_file_list(eval_dir: Path, require_gate: bool) -> List[str]:
    """Eval files to ship, relative to the eval dir (``gate_v1/...`` included when present).

    :param eval_dir: Eval dir.
    :param require_gate: Fail unless ``gate_v1/`` (with :data:`GATE_FILES`) and ``holiday_queries.json`` exist.
    :returns: Relative paths.
    :rtype: List[str]
    :raises SystemExit: When a required file is missing.
    """
    missing = [f for f in EVAL_FILES if not (eval_dir / f).exists()]
    if require_gate:
        missing += [f for f in GATE_REQUIRED_EVAL_FILES if not (eval_dir / f).exists()]
        missing += [f"{GATE_DIR}/{f}" for f in GATE_FILES if not (eval_dir / GATE_DIR / f).exists()]
    if missing:
        raise SystemExit(f"{eval_dir} lacks {missing}")
    files = list(EVAL_FILES) + [f for f in OPTIONAL_EVAL_FILES if (eval_dir / f).exists()]
    gate = eval_dir / GATE_DIR
    if gate.is_dir():
        files += sorted(str(p.relative_to(eval_dir)) for p in gate.rglob("*") if p.is_file())
    return files


def code_files(require_gate: bool) -> List[Path]:
    """Code files to ship next to the data.

    :param require_gate: Fail unless ``eval_gate.py`` exists.
    :returns: Source paths.
    :rtype: List[Path]
    :raises SystemExit: When a required file is missing.
    """
    out = []
    for src, always in CODE_FILES:
        if src.exists():
            out.append(src)
        elif always or require_gate:
            raise SystemExit(f"{src} is missing (required{' by --require-gate' if not always else ''})")
    return out


def copy_eval_and_code(eval_dir: Path, tmp: Path, eval_files: List[str], code: List[Path]) -> Counter:
    """Copy the eval files to ``tmp/eval`` and the code next to the data; check the copies byte for byte.

    :param eval_dir: Eval dir.
    :param tmp: Package build folder.
    :param eval_files: Relative eval file paths.
    :param code: Code source paths.
    :returns: ``eval_copy_mismatch`` / ``code_copy_mismatch`` counts.
    :rtype: Counter
    """
    out: Counter = Counter()
    for name in eval_files:
        dst = tmp / "eval" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(eval_dir / name, dst)
        out["eval_copy_mismatch"] += md5_file(eval_dir / name) != md5_file(dst)
    for src in code:
        shutil.copyfile(src, tmp / src.name)
        out["code_copy_mismatch"] += md5_file(src) != md5_file(tmp / src.name)
    return out


# ---------------------------------------------------------------------------------------------------------------
# training data
# ---------------------------------------------------------------------------------------------------------------
def gate_pool(params: dict) -> tuple:
    """Rebuild T3's gated, masked train records with the parameters T3 recorded.

    The leak filter includes the blocked acceptance-gate queries the mixture was built with (``gate_dir`` /
    ``gate_spec``), so the pool's text gating is the mixture's. Pool ``subjects`` = every subject id plus the
    record's ``M:`` mention ids, unless the mixture was built with ``--no-mention-masks``.

    :param params: ``train_mix_v3.stats.json["params"]``.
    :returns: (pool rows ``doc_id -> {doc_id, text, text_hash, subjects}``, gate stats, eval info from
        ``load_eval``, semantic rows, leak filter).
    :rtype: tuple
    """
    ev = load_eval(Path(params["eval_dir"]))
    strict = params.get("final_strict", "off") == "on"
    gate_strings = load_gate_strings(Path(params["gate_dir"]) if params.get("gate_dir") else None,
                                     Path(params["gate_spec"]) if params.get("gate_spec") else None)
    leak, _ = build_leak_filter(ev, params["probe_scope"], params["min_contain"], params["min_shared"],
                                params.get("focus_names", "block"), params.get("he_prefix_contain", "off") == "on",
                                strict, gate_strings)
    rows, _ = load_semantic(Path(params["semantic"]))
    subj, weak = load_subjects(Path(params["subjects"]))
    blocked = ev["held_subjects"] | ev["linked"]
    mentions = {} if params.get("no_mention_masks") else record_mentions(Path(params["subjects"]), blocked)
    marks = own_shelfmarks(Path(params["corpus_v1"]))
    builder = MixBuilder(ev, leak, rows, subj, weak, marks, random.Random(params["seed"]), norm_gate=strict)
    pool = {d: {"doc_id": d, "text": builder.train[d], "text_hash": rows[d]["text_hash"],
                "subjects": sorted(subj.get(d, set()) | set(mentions.get(d, [])))} for d in sorted(builder.train)}
    return pool, dict(builder.gate_stats), ev, rows, leak


def record_mask_ids(ids: Iterable[str]) -> Set[str]:
    """A row's record mask ids without the ``D:`` id (comparable with the pool record's ``subjects``).

    :param ids: ``mask_ids`` or ``anchor_mask_ids``.
    :returns: Id set.
    :rtype: Set[str]
    """
    return {x for x in ids if not x.startswith("D:")}


def verify(mix: List[dict], pool: Dict[str, dict], ev: dict, sem_rows: Dict[str, dict],
           norm_gate: bool = False) -> Counter:
    """Training-data package checks; every counter except ``checked_*`` must be zero.

    :param mix: Mixture rows.
    :param pool: Gated pool.
    :param ev: ``build_training_mix_v3.load_eval`` output.
    :param sem_rows: Semantic rows (``load_semantic``).
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
        # the trainer masks mined negatives with the pool's ids and in-batch candidates with the rows' ids: same set
        out["mask_ids_not_pool_subjects"] += record_mask_ids(r["mask_ids"]) != set(pool[d]["subjects"])
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
                out["anchor_mask_ids_not_pool_subjects"] += record_mask_ids(r["anchor_mask_ids"]) != set(
                    pool[a]["subjects"])
        out["checked_rows"] += 1
    out["checked_pool"] = len(pool)
    return out


def verify_eval_pool(eval_pool: List[dict], sem_rows: Dict[str, dict], original_ids: Set[str],
                     v3_hashes: Dict[str, str]) -> Counter:
    """Eval-pool checks against the shipped corpora; every counter except ``checked_*`` must be zero.

    :param eval_pool: Frozen eval pool rows.
    :param sem_rows: Semantic rows (``doc_id -> {eligible, text_hash, text}``) of the embedded text.
    :param original_ids: Doc ids present in ``corpus_original.jsonl``.
    :param v3_hashes: ``doc_id -> md5`` of the shipped semantic_v3 text.
    :returns: Counts.
    :rtype: Counter
    """
    out: Counter = Counter()
    for r in eval_pool:
        s = sem_rows.get(r["doc_id"])
        out["eval_pool_not_eligible"] += s is None or not s["eligible"]
        if s is not None:
            out["eval_pool_shelfmark_regex"] += bool(PROD_SHELFMARK_RE.search(s["text"]) or EXTRA_ID_RE.search(s["text"]))
            out["eval_pool_id_line"] += ("Document ID:" in s["text"]) or ("Shelf Mark:" in s["text"])
        out["eval_pool_hash_mismatch"] += s is not None and s["text_hash"] != r["text_hash"]
        out["eval_pool_no_original"] += r["doc_id"] not in original_ids
        out["eval_pool_no_semantic_v3"] += r["doc_id"] not in v3_hashes
        if "text_hash_v3" in r:
            out["eval_pool_v3_hash_mismatch"] += v3_hashes.get(r["doc_id"]) != r["text_hash_v3"]
            out["checked_eval_pool_v3_hashes"] += 1
    out["checked_eval_pool"] = len(eval_pool)
    return out


def mask_overlap(rows: List[dict], rng: random.Random, n: int = 20000) -> dict:
    """Share of random row pairs that share mask ids (diagnostic for batch construction).

    Ids per row = record ids of the positive and the d2d anchor (no ``D:``) plus ``anchor_topic_ids`` and
    ``anchor_mention_id``.

    :param rows: Mixture rows.
    :param rng: RNG.
    :param n: Sampled pairs.
    :returns: ``subject_overlap`` (any id), ``subject_overlap_no_broad`` (without ``domain:`` / ``M:`` ids),
        ``mention_overlap`` (shared ``M:`` ids), ``guard_overlap`` (one row's guard ids among the other's ids) and
        ``doc_overlap`` (same ``D:`` id).
    :rtype: dict
    """
    rec = [r for r in rows if r["mask_ids"]]
    if len(rec) < 2:
        return {}

    def ids(r: dict) -> Tuple[Set[str], Set[str]]:
        guard = set(r.get("anchor_topic_ids") or [])
        if r.get("anchor_mention_id"):
            guard.add(r["anchor_mention_id"])
        return record_mask_ids(r["mask_ids"] + r["anchor_mask_ids"]) | guard, guard

    c: Counter = Counter()
    for _ in range(n):
        a, b = rng.sample(rec, 2)
        (sa, ga), (sb, gb) = ids(a), ids(b)
        shared = sa & sb
        c["subject_overlap"] += bool(shared)
        c["subject_overlap_no_broad"] += bool({x for x in shared if not x.startswith(BROAD_ID_PREFIXES)})
        c["mention_overlap"] += bool({x for x in shared if x.startswith("M:")})
        c["guard_overlap"] += bool((ga & sb) or (gb & sa))
        c["doc_overlap"] += bool({x for x in a["mask_ids"] if x.startswith("D:")} & set(b["mask_ids"]))
    return {k: round(v / n, 4) for k, v in c.items()}


def guard_stats(mix: List[dict], pool: Dict[str, dict]) -> dict:
    """Festival and mention guards of the mixture and the pool.

    :param mix: Mixture rows.
    :param pool: Gated pool.
    :returns: ``{festival_guard: {rows_tagged, by_topic, by_family, by_mask_level}, mention_guard: {...}}``.
    :rtype: dict
    """
    fest: Dict[str, Counter] = defaultdict(Counter)
    ment: Dict[str, Counter] = defaultdict(Counter)
    for r in mix:
        topics = [x for x in r.get("anchor_topic_ids") or [] if x.startswith("topic:")]
        mtopics = [x for x in r.get("anchor_topic_ids") or [] if x.startswith("M:")]
        if topics:
            fest["totals"]["rows_tagged"] += 1
            fest["by_family"][r["family"]] += 1
            fest["by_mask_level"][r["mask_level"]] += 1
            fest["by_topic"].update(topics)
        if mtopics:
            ment["totals"]["rows_with_mention_topic_ids"] += 1
            ment["by_family_topic_ids"][r["family"]] += 1
        if r.get("anchor_mention_id"):
            ment["totals"]["rows_with_anchor_mention_id"] += 1
            ment["by_family_anchor_mention_id"][r["family"]] += 1
        ment["totals"]["rows_with_mention_mask_ids"] += any(x.startswith("M:") for x in r["mask_ids"])
    pool_m = [sum(x.startswith("M:") for x in r["subjects"]) for r in pool.values()]
    ment["totals"]["pool_records_with_mentions"] = sum(bool(m) for m in pool_m)
    ment["totals"]["pool_mention_ids"] = sum(pool_m)

    def fold(groups: Dict[str, Counter]) -> dict:
        totals = dict(groups.pop("totals", {}))
        return {**totals, **{k: dict(v.most_common()) for k, v in sorted(groups.items())}}

    return {"festival_guard": fold(fest) or {"rows_tagged": 0}, "mention_guard": fold(ment)}


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------
def write_corpora(tmp: Path, semantic: Path, corpus_v1: Path,
                  extra: Dict[str, Path]) -> Tuple[dict, Set[str], Dict[str, str]]:
    """Write the corpus files of a package.

    :param tmp: Package build folder.
    :param semantic: Embedded-text corpus (``corpus_semantic.jsonl``).
    :param corpus_v1: Production-text corpus.
    :param extra: Extra text variants ``name -> path``.
    :returns: (row counts per file stem, doc ids of ``corpus_original.jsonl``, ``doc_id -> md5`` of the semantic_v3
        text).
    :rtype: Tuple[dict, Set[str], Dict[str, str]]
    """
    counts = {"corpus_semantic": write_jsonl(tmp / "corpus_semantic.jsonl", semantic_rows(semantic))}
    v3_hashes: Dict[str, str] = {}

    def v3_rows() -> Iterable[dict]:
        for r in semantic_v3_rows(semantic):
            v3_hashes[r["doc_id"]] = r["text_hash"]
            yield r

    counts["corpus_semantic_v3"] = write_jsonl(tmp / SEMANTIC_V3_FILE, v3_rows())
    original_ids: Set[str] = set()

    def originals() -> Iterable[dict]:
        for r in iter_jsonl(corpus_v1):
            original_ids.add(r["doc_id"])
            yield {"doc_id": r["doc_id"], "text": r["text"]}

    counts["corpus_original"] = write_jsonl(tmp / "corpus_original.jsonl", originals())
    for name, path in extra.items():
        counts[f"corpus_{name}"] = write_jsonl(tmp / f"corpus_{name}.jsonl", semantic_rows(path))
    return counts, original_ids, v3_hashes


def file_table(tmp: Path) -> dict:
    """md5 / bytes / rows of every file in the build folder (``manifest.json`` excluded).

    :param tmp: Package build folder.
    :returns: ``relative path -> {md5, bytes[, rows]}``.
    :rtype: dict
    """
    files = {}
    for p in sorted(tmp.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = str(p.relative_to(tmp))
            files[rel] = {"md5": md5_file(p), "bytes": p.stat().st_size}
            if p.suffix == ".jsonl":
                files[rel]["rows"] = sum(1 for _ in open(p, encoding="utf-8"))
    return files


def main() -> None:
    """Build, verify, then move the folder into place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v3")
    parser.add_argument("--mix", default=str(V3 / "train_mix_v3.jsonl"),
                        help="mixture file, or a bare name under v3/ (train_mix_v3final)")
    parser.add_argument("--eval-dir", default=None,
                        help="eval dir, or a bare name under v3/ (eval_final); must be the one the mixture was built "
                             "from (default: read from the mixture's stats; required with --eval-only)")
    parser.add_argument("--corpus-v1", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--semantic", default=None,
                        help="--eval-only: the embedded-text corpus (default for training packages: the mixture's)")
    parser.add_argument("--eval-only", action="store_true",
                        help="package corpora + eval + scorer only (no mixture / train pool): the text ablation")
    parser.add_argument("--extra-corpus", action="append", default=[],
                        help="name=path of an extra text variant, shipped as corpus_<name>.jsonl (repeatable)")
    parser.add_argument("--v9-mix", default=str(V9_MIX),
                        help=f"the mixture v9 was trained on (-> {V9_TRAINED_FILE})")
    parser.add_argument("--require-gate", action="store_true",
                        help="fail unless the eval dir has gate_v1/ and holiday_queries.json and eval_gate.py exists")
    parser.add_argument("--out-root", default=str(AUDIT_ROOT / "hf_dataset"), help="parent folder of the package")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing package folder")
    args = parser.parse_args()
    out = Path(args.out_root) / args.name
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists: use another --name (or --overwrite to replace it)")
    extra = parse_extra_corpora(args.extra_corpus)
    if args.eval_only:
        if not (args.eval_dir and args.semantic):
            raise SystemExit("--eval-only needs --eval-dir and --semantic")
        eval_dir, semantic, params, mix_path = resolve(args.eval_dir, V3), Path(args.semantic), None, None
    else:
        mix_path = resolve(args.mix, V3, ".jsonl")
        params = json.loads(mix_path.with_suffix(".stats.json").read_text())["params"]
        eval_dir, semantic = Path(params["eval_dir"]), Path(params["semantic"])
        if args.eval_dir and resolve(args.eval_dir, V3).resolve() != eval_dir.resolve():
            raise SystemExit(f"--eval-dir {args.eval_dir} is not the eval dir {eval_dir} that {mix_path.name} was "
                             "built from")
        if args.semantic and Path(args.semantic).resolve() != semantic.resolve():
            raise SystemExit(f"--semantic {args.semantic} is not the corpus {semantic} that {mix_path.name} was "
                             "built from")
    eval_files = eval_file_list(eval_dir, args.require_gate)
    code = code_files(args.require_gate)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "eval").mkdir(parents=True)
    final = (eval_dir / "focus_subjects.json").exists()
    text_version = corpus_text_version(semantic)
    eval_pool = list(iter_jsonl(eval_dir / "eval_pool.jsonl"))

    checks: Counter = Counter()
    if args.eval_only:
        sem_rows, _ = load_semantic(semantic)
        pool, mix, gate_stats, ev, leak = {}, [], {}, load_eval(eval_dir), None
    else:
        pool, gate_stats, ev, sem_rows, leak = gate_pool(params)
        print(json.dumps({"gate": gate_stats}), flush=True)
        mix = list(iter_jsonl(mix_path))
    counts, original_ids, v3_hashes = write_corpora(tmp, semantic, Path(args.corpus_v1), extra)
    if not args.eval_only:
        counts["train_pool"] = write_jsonl(tmp / "train_pool.jsonl", pool.values())
        shutil.copyfile(mix_path, tmp / "train_mix_v3.jsonl")
        counts["train_mix_v3"] = len(mix)
    v9 = v9_trained_ids(Path(args.v9_mix), eval_pool)
    (tmp / V9_TRAINED_FILE).write_text(json.dumps(v9, ensure_ascii=False), encoding="utf-8")
    checks.update(copy_eval_and_code(eval_dir, tmp, eval_files, code))

    strict = bool(params) and params.get("final_strict", "off") == "on"
    if not args.eval_only:
        checks.update(verify(mix, pool, ev, sem_rows, norm_gate=strict))
        # re-scan every pair with T3's own verifier (held-out records / texts / subjects, ID lines, shelf marks, leak
        # filter on both sides, bare festival anchors on canon positives, mask levels) with the rules the mixture was
        # built with (a mixture whose params predate a rule was built without it); its info_ counters are informational
        labels = load_subject_labels(Path(params["subjects"]))
        for k, v in verify_mix(mix, ev, sem_rows, leak, norm_gate=strict, labels=labels,
                               festival_canon=not params.get("no_festival_canon_drops", True),
                               short_anchor_subject=not params.get("legacy_mask_level", True)).items():
            checks[f"checked_mix_{k}" if k.startswith("info_") else f"mix_{k}"] += v
    checks.update(verify_eval_pool(eval_pool, sem_rows, original_ids, v3_hashes))
    checks["corpus_count_mismatch"] += len({n for k, n in counts.items() if k.startswith("corpus_")}) != 1
    violations = {k: v for k, v in checks.items() if v and not k.startswith("checked_")}

    guards = guard_stats(mix, pool) if mix else {}
    overlap = mask_overlap(mix, random.Random(0)) if mix else {}
    gate_code = ", `eval_gate.py`" if any(p.name == "eval_gate.py" for p in code) else ""
    eval_extra = "".join(s for s, f in ((", holiday", "holiday_queries.json"),
                                        (", acceptance gate `gate_v1/`", GATE_DIR))
                         if any(e == f or e.startswith(f + "/") for e in eval_files))
    common = dict(n_corpus=counts["corpus_semantic"], n_eligible=len(eval_pool), text_version=text_version,
                  n_v9=v9["n"], gate_code=gate_code, eval_extra=eval_extra, date=datetime.now().date().isoformat())
    if args.eval_only:
        extra_rows = "".join(f"| `corpus_{n}.jsonl` | {counts[f'corpus_{n}']} records, extra text variant (`{p}`) |\n"
                             for n, p in extra.items())
        readme = README_EVAL_ONLY.format(name=args.name, extra_rows=extra_rows, **common)
        stats, kinds, levels = {}, Counter(), Counter()
    else:
        stats = json.loads(mix_path.with_suffix(".stats.json").read_text())
        kinds, levels = Counter(r["kind"] for r in mix), Counter(r["mask_level"] for r in mix)
        fmt = dict(n_train=len(mix), n_q2d=kinds["q2d"], n_t2t=kinds["t2t"], n_d2d=kinds["d2d"],
                   n_subject=levels["subject"], n_doc=levels["doc"],
                   families=", ".join(f"{k} {v}" for k, v in stats["by_family"].items()), n_pool=len(pool),
                   n_held=len(ev["held"]), masking=MASKING, **common)
        readme = README_FINAL.format(name=args.name, mix_name=mix_path.name, **fmt) if final else README.format(**fmt)
    (tmp / "README.md").write_text(readme)
    files = file_table(tmp)
    # None for an eval-only package (no train pool to compare with T3's gate counts)
    gate_match = None if args.eval_only else gate_stats.get("train_records") == stats["records"]["train_records"]
    manifest = {"name": args.name, "built_at": datetime.now().isoformat(timespec="seconds"),
                "sources": {"train_mix_v3": {"path": str(mix_path), "md5": md5_file(mix_path)} if mix_path else None,
                            "eval_dir": str(eval_dir), "semantic": str(semantic), "text_version": text_version,
                            "corpus_v1": args.corpus_v1, "extra_corpora": {n: str(p) for n, p in extra.items()},
                            "v9_mix": v9["source"], "subjects": params["subjects"] if params else None,
                            "gate_params": params},
                "summary": {"mode": "eval_only" if args.eval_only else "final" if final else "v3",
                            "train_pairs": len(mix), "by_kind": dict(kinds), "by_mask_level": dict(levels),
                            "train_pool": len(pool), "corpus": counts["corpus_semantic"],
                            "eval_pool": len(eval_pool), "held_out_records": len(ev["held"]),
                            "held_out_subjects": sorted(ev["held_subjects"]), "focus_subjects": sorted(ev["focus"]),
                            "v9_trained": {k: v for k, v in v9.items() if k != "doc_ids"},
                            "eval_files": eval_files, "code_files": [p.name for p in code]},
                "guards": guards, "mask_overlap": overlap,
                "gate_stats": gate_stats, "gate_matches_t3_stats": gate_match,
                "verification": dict(checks), "violations": violations, "files": files}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({"counts": counts, "guards": guards, "mask_overlap": overlap, "verification": dict(checks),
                      "violations": violations, "gate_matches_t3_stats": gate_match}, indent=1), flush=True)
    if violations or gate_match is False:
        raise SystemExit(f"package NOT moved into place (left at {tmp}): violations={violations}, "
                         f"gate_matches_t3_stats={gate_match}")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
