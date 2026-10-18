"""v9 bibliography check: does the tuned text embedder hurt or help scholarship-page retrieval?

The v3-final tuned model (``isaacmg/genizah-embed-qwen3-0.6b``) was trained on catalogue records only (v3 dropped the
scholarship pairs), yet production embeds BOTH indexes with one model: ``embedding_client`` serves the query vector
for the merged index and for the bibliography index, and the startup canary gate (``app.py``
``verify_embedding_compatibility``) checks both indexes' ``_meta`` canaries against that one embedder. So index v9
either re-embeds the bibliography pages with the tuned model too, or production needs a second query embedder. This
script measures what the first option costs or gains.

Pool: every page of the production bibliography index (``bibliography_text_only_0.7``, read-only via ``es_read``),
embedded from ``full_text_content`` -- exactly the text the indexer embedded (``BibliographyDocument.
create_text_representation``: Source PDF / Title / Authors / Subjects / Shelf marks mentioned / Summary / Main text /
Transcriptions lines, kept as production will).

Queries:

* known-item: hand-written natural queries (one English, one Hebrew) for a seeded sample of 150 pages
  (``sample``; the queries live in ``results/v9_bib_known_item_queries.jsonl``); tie-aware rank of the target page over
  the whole pool (identical-text twins count as hits), RR / R@10, plus a lenient rank where the target's
  neighbouring pages (same book, page_seq +-1) also count;
* subject: the frozen v3-final eval's subject queries (name, probe, llm_t2) scored against pages with WEAK
  relevance = pages whose text names the subject (``label``; rules in :func:`page_match_text` and
  :func:`subject_matchers`).

Both models embed the same unique page texts and the same queries with the production contract (documents raw,
queries ``Instruct: Given a search query, retrieve relevant passages\\nQuery: `` + text, L2-normalised, max_seq 8192,
fp32), sharded and resumable through ``step0_local.embed_unique``. Pages longer than ``--mps-max-tokens`` are
embedded one by one on the CPU at the end (bounded attention memory on MPS).

``lexical`` (optional, read-only ES) runs production's bibliography keyword query (``search_bibliography.py``
``search_hybrid``: multi_match + phrase boosts + alias expansion) for every query, so ``score`` can also simulate the
production hybrid ranking (weighted RRF, k=60, semantic 60 / keyword 40) for base vs tuned.

Run (from this directory; ``PY`` = the historical-document-analysis venv python)::

    $PY bib_v9_check.py fetch            # ES -> AUDIT_ROOT/v9_biblio/pages.jsonl (+ stored vectors of a sample)
    $PY bib_v9_check.py sample           # seeded known-item sample -> AUDIT_ROOT/v9_biblio/ki_sample.jsonl
    $PY bib_v9_check.py queries --tsv q.tsv   # hand-written queries -> results/v9_bib_known_item_queries.jsonl
    $PY bib_v9_check.py label            # weak subject relevance -> AUDIT_ROOT/v9_biblio/subject_labels.json
    $PY bib_v9_check.py lexical          # production keyword ranks (read-only ES) -> AUDIT_ROOT/v9_biblio/lexical.json
    python3 guard.py --name gpu_bib_v9_base --max-gb 8 -- $PY bib_v9_check.py embed --model base --soft-max-gb 6.5
    python3 guard.py --name gpu_bib_v9_tuned --max-gb 8 -- $PY bib_v9_check.py embed --model tuned --soft-max-gb 6.5
    $PY bib_v9_check.py score            # -> results/v9_bibliography_check.json
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embed_utils import AUDIT_ROOT, DEFAULT_TASK, TEXT_MODELS

HERE = Path(__file__).parent
REPO = HERE.parent.parent
BIB_INDEX = "bibliography_text_only_0.7"
WORK = AUDIT_ROOT / "v9_biblio"
PAGES = WORK / "pages.jsonl"
STORED_SAMPLE = WORK / "stored_vectors_sample.npz"
INDEX_META = WORK / "index_meta.json"
KI_SAMPLE = WORK / "ki_sample.jsonl"
KI_QUERIES = HERE / "results" / "v9_bib_known_item_queries.jsonl"
SUBJECT_LABELS = WORK / "subject_labels.json"
LEXICAL = WORK / "lexical.json"
VEC_ROOT = WORK / "vectors"
EVAL_FINAL = AUDIT_ROOT / "v3" / "eval_final"
SUBJECT_VOCAB = AUDIT_ROOT / "v3" / "subject_vocab.json"
RESULTS = HERE / "results" / "v9_bibliography_check.json"
MAX_SEQ = 8192
QUERY_PREFIX = f"Instruct: {DEFAULT_TASK}\nQuery: "
MODELS = {
    "base": {"id": TEXT_MODELS["qwen3-0.6b"]["id"], "revision": TEXT_MODELS["qwen3-0.6b"]["revision"]},
    "tuned": {"id": "isaacmg/genizah-embed-qwen3-0.6b", "revision": "dbfab73f1e91a6b95eba728541cc5a391a88d933"},
}
KI_N_PAGES = 150
KI_MAX_PER_BOOK = 8
KI_MIN_MAIN_CHARS = 600
HE_PAGE_SHARE = 0.3          # a page whose OCR body is >= 30 % Hebrew letters counts as a Hebrew page
SUBJ_MIN_REL = 3             # a subject is scored when at least this many pages name it ...
SUBJ_MAX_REL_SHARE = 0.10    # ... and at most this share of the pool (generic words are not a subject signal)
MIN_NAME_CHARS = 4           # shorter subject names are too ambiguous to label pages with
HE_LETTERS = "א-ת"
HE_PREFIX = "[בהולשכמ]{0,2}"
RRF_K = 60.0
HYBRID_WEIGHTS = (0.6, 0.4)  # HTTP endpoint default: semanticWeight 60, keywordWeight 40
LEX_SIZE = 50                # search_hybrid candidate_size for num_results=10: min(200, max(40, 10 * 5))
# The chat agent (lms_agentic_search, origin/prod-mbp) calls search_hybrid with SearchAction defaults: 50/50 weights
# (normalize_weights / SearchAction.semantic_weight=keyword_weight=50), num_results=5 -> candidate_size
# min(200, max(40, 5 * 5)) = 40, and the synthesis sees the top 5; several injected actions force 30/70.
CHAT_WEIGHTS = (0.5, 0.5)
CHAT_CAND = 40
CHAT_TOP = 5
TRAIN_MIX = AUDIT_ROOT / "v3" / "train_mix_v3final.jsonl"  # the tuned model's training pairs (anchor-leak check)
RESTART_RC = 99


def log(msg: str) -> None:
    """Print a timestamped progress line.

    :param msg: Message text.
    """
    print(f"[bib_v9 {datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    """Write JSON via a temp file and rename (SMB-safe).

    :param path: Target path.
    :param obj: JSON-serialisable object (numpy values are converted).
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1,
                              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    path.unlink(missing_ok=True)
    tmp.rename(path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    """Save an array via ``<stem>.tmp.npy`` and rename.

    :param path: Target ``.npy`` path.
    :param arr: Array.
    """
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    path.unlink(missing_ok=True)
    tmp.rename(path)


def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file.

    :param path: File path.
    :returns: Parsed rows.
    :rtype: List[dict]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------------------------------------------
# fetch (read-only ES)
# ---------------------------------------------------------------------------------------------------------------
FETCH_FIELDS = ["doc_id", "title", "author", "authors", "page_number", "extracted_page_number", "page_seq",
                "book_uuid", "shelf_marks_mentioned", "description", "full_text_content", "subject_keywords"]


def hebrew_share(text: str) -> float:
    """Share of letters in ``text`` that are Hebrew.

    :param text: Text.
    :returns: Hebrew letters / all letters (0 for no letters).
    :rtype: float
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "א" <= c <= "ת") / len(letters)


def main_text(full: str) -> str:
    """The ``Main text:`` part of a page representation (up to the first ``Transcription for`` line).

    :param full: ``full_text_content``.
    :returns: The OCR'd page body ('' when absent).
    :rtype: str
    """
    m = re.search(r"^Main text: (.*?)(?=^Transcription for |\Z)", full, re.S | re.M)
    return m.group(1).strip() if m else ""


def fetch(args: argparse.Namespace) -> None:
    """Scroll the bibliography index (read-only) into ``pages.jsonl`` and keep stored vectors of a sample.

    :param args: Parsed CLI arguments (``n_stored``).
    """
    from es_read import ES_URL, SESSION, get_docs, scan

    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for hit in scan(BIB_INDEX, {"match_all": {}}, FETCH_FIELDS, size=500):
        s = hit["_source"]
        text = s.get("full_text_content") or ""
        sms = s.get("shelf_marks_mentioned") or []
        rows.append({"es_id": hit["_id"], "doc_id": s.get("doc_id"), "book": s.get("title") or "",
                     "book_uuid": s.get("book_uuid"), "page_seq": s.get("page_seq"),
                     "page_number": s.get("page_number"), "authors": s.get("authors") or s.get("author"),
                     "n_shelf_marks": len(sms) if isinstance(sms, list) else 1,
                     "description": s.get("description") or "", "text": text,
                     "text_hash": hashlib.md5(text.encode("utf-8")).hexdigest(), "n_chars": len(text),
                     "n_main_chars": len(main_text(text)), "he_share": round(hebrew_share(main_text(text)), 3)})
    rows.sort(key=lambda r: r["es_id"])
    with open(PAGES.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    PAGES.unlink(missing_ok=True)
    PAGES.with_suffix(".jsonl.tmp").rename(PAGES)
    # stored production vectors of a length-spread sample: the base re-embedding must reproduce them (contract check)
    by_len = sorted(rows, key=lambda r: (r["n_chars"], r["es_id"]))
    pick = [by_len[i]["es_id"] for i in np.linspace(0, len(by_len) - 1, args.n_stored).round().astype(int)]
    pick = list(dict.fromkeys(pick))
    stored = get_docs(BIB_INDEX, pick, ["embedding_vector"])
    ids = [i for i in pick if stored.get(i, {}).get("embedding_vector")]
    np.savez(STORED_SAMPLE, ids=np.array(ids), vectors=np.array([stored[i]["embedding_vector"] for i in ids],
                                                                 dtype=np.float32))
    mapping = SESSION.get(f"{ES_URL}/{BIB_INDEX}/_mapping", timeout=60).json()
    write_json(INDEX_META, list(mapping.values())[0]["mappings"].get("_meta", {}))
    books = Counter(r["book"] for r in rows)
    log(f"{len(rows)} pages, {len({r['text_hash'] for r in rows})} unique texts, {len(books)} books, "
        f"{sum(r['n_chars'] for r in rows)} chars; stored vectors for {len(ids)} sample pages; wrote {PAGES}")


# ---------------------------------------------------------------------------------------------------------------
# known-item sample and queries
# ---------------------------------------------------------------------------------------------------------------
SKIP_DESC = re.compile(r"\b(bibliograph|index|table of contents|contents page|blank|list of abbreviations|"
                       r"abbreviations|title page|copyright|acknowledg|plate|figure|facsimile)", re.I)


def ki_eligible(r: dict) -> bool:
    """Whether a page can carry hand-written known-item queries.

    Needs at least :data:`KI_MIN_MAIN_CHARS` of OCR'd body text and a summary that does not describe a reference list,
    index, contents, blank, front-matter or plate page (those answer no natural question of their own).

    :param r: Page row.
    :returns: Eligibility.
    :rtype: bool
    """
    return r["n_main_chars"] >= KI_MIN_MAIN_CHARS and not SKIP_DESC.search(r["description"][:300])


def sample(args: argparse.Namespace) -> None:
    """Seeded known-item sample: shuffled eligible pages, at most :data:`KI_MAX_PER_BOOK` per book.

    :param args: Parsed CLI arguments (``seed``, ``n``).
    """
    rows = read_jsonl(PAGES)
    dup = Counter(r["text_hash"] for r in rows)
    elig = [r for r in rows if ki_eligible(r) and dup[r["text_hash"]] == 1]
    rng = random.Random(args.seed)
    rng.shuffle(elig)
    per_book: Counter = Counter()
    picked = []
    for r in elig:
        if per_book[r["book"]] >= KI_MAX_PER_BOOK:
            continue
        per_book[r["book"]] += 1
        picked.append(r)
        if len(picked) == args.n:
            break
    with open(KI_SAMPLE, "w", encoding="utf-8") as fh:
        for i, r in enumerate(picked):
            fh.write(json.dumps({"k": i, **r}, ensure_ascii=False) + "\n")
    log(f"{len(elig)} eligible of {len(rows)}; picked {len(picked)} from {len(per_book)} books; "
        f"he_share>={HE_PAGE_SHARE}: {sum(r['he_share'] >= HE_PAGE_SHARE for r in picked)}; wrote {KI_SAMPLE}")


def build_queries(args: argparse.Namespace) -> None:
    """Turn the hand-written ``k <TAB> English <TAB> Hebrew`` TSV into the known-item query file.

    :param args: Parsed CLI arguments (``tsv``).
    """
    pages = {r["k"]: r for r in read_jsonl(KI_SAMPLE)}
    out = []
    for line in Path(args.tsv).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        k, en, he = line.split("\t")
        p = pages[int(k)]
        for lang, text in (("en", en.strip()), ("he", he.strip())):
            if lang == "he" and hebrew_share(text) < 0.5:
                raise ValueError(f"k={k}: the Hebrew query is not Hebrew: {text!r}")
            out.append({"qid": f"bib{int(k):03d}:{lang}", "k": int(k), "es_id": p["es_id"], "lang": lang,
                        "text": text, "page_lang": "he" if p["he_share"] >= HE_PAGE_SHARE else "non-he",
                        "page_he_share": p["he_share"], "book": p["book"]})
    if sorted({q["k"] for q in out}) != sorted(pages):
        raise ValueError("every sampled page needs exactly one query pair")
    KI_QUERIES.parent.mkdir(parents=True, exist_ok=True)
    with open(KI_QUERIES, "w", encoding="utf-8") as fh:
        for q in sorted(out, key=lambda q: q["qid"]):
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    log(f"{len(out)} queries for {len(pages)} pages -> {KI_QUERIES}")


def show(args: argparse.Namespace) -> None:
    """Print sampled pages ``start..end`` (for writing queries).

    :param args: Parsed CLI arguments (``start``, ``end``, ``chars``).
    """
    for r in read_jsonl(KI_SAMPLE)[args.start:args.end]:
        print(f"=== k={r['k']} es_id={r['es_id']} he_share={r['he_share']} book={r['book'][:80]}")
        print(r["text"][:args.chars])
        print()


# ---------------------------------------------------------------------------------------------------------------
# weak subject labels
# ---------------------------------------------------------------------------------------------------------------
def page_match_text(text: str) -> str:
    """The part of a page representation a subject must be named in to label the page.

    Drops the ``Source PDF:`` line (file names) and the ``Shelf marks mentioned:`` line (shelf marks name no
    subject); keeps Title, Authors, Subjects, Summary, Main text and Transcriptions.

    :param text: ``full_text_content``.
    :returns: Text to match against.
    :rtype: str
    """
    return "\n".join(l for l in text.split("\n")
                     if not l.startswith("Source PDF:") and not l.startswith("Shelf marks mentioned:"))


def name_pattern(name: str) -> Optional[str]:
    """Whole-word regex for one subject name (Hebrew names may carry 0-2 prefix letters ב ה ו ל ש כ מ).

    :param name: Subject name (English or Hebrew).
    :returns: Regex source, or None for names shorter than :data:`MIN_NAME_CHARS` letters.
    :rtype: Optional[str]
    """
    name = name.strip()
    if sum(c.isalpha() for c in name) < MIN_NAME_CHARS:
        return None
    body = r"[\s\-]+".join(re.escape(w) for w in name.split())
    if hebrew_share(name) >= 0.5:
        return rf"(?<![{HE_LETTERS}]){HE_PREFIX}{body}(?![{HE_LETTERS}])"
    return rf"(?<![^\W\d_]){body}(?![^\W\d_])"


def subject_matchers(sq: dict, vocab: dict,
                     focus: Dict[str, dict]) -> Dict[str, Tuple[str, re.Pattern, Optional[List[str]]]]:
    """One compiled matcher per subject: its concept pattern (focus subjects) or its names (all others).

    :param sq: ``subject_queries.json["subjects"]``.
    :param vocab: ``subject_vocab.json`` (names_en / names_he per subject).
    :param focus: Focus-subject descriptions by id (``focus_subjects.json``).
    :returns: ``sid -> (rule, compiled regex, lowercase literal needles or None)``. A page can only match a name
        regex if it contains one of the needles (each name's first word), which lets :func:`label` skip the regex on
        most pages. Subjects without a usable name are left out.
    :rtype: Dict[str, Tuple[str, re.Pattern, Optional[List[str]]]]
    """
    out = {}
    for sid, entry in sq.items():
        if sid in focus and focus[sid].get("concept_pattern"):
            out[sid] = ("concept_pattern", re.compile(focus[sid]["concept_pattern"], re.I), None)
            continue
        v = vocab.get(sid, {})
        names = list(dict.fromkeys(list(v.get("names_en") or []) + list(v.get("names_he") or [])
                                   + [q["text"] for q in entry["queries"] if q["source"] == "name"]))
        usable = [n for n in names if name_pattern(n)]
        if usable:
            needles = sorted({n.split()[0].lower() for n in usable})
            out[sid] = ("names", re.compile("|".join(name_pattern(n) for n in usable), re.I), needles)
    return out


def label(args: argparse.Namespace) -> None:
    """Label pages with the frozen eval's subjects (weak: the page text names the subject) and keep the scorable ones.

    :param args: Parsed CLI arguments (unused).
    """
    pages = read_jsonl(PAGES)
    sq = json.loads((EVAL_FINAL / "subject_queries.json").read_text(encoding="utf-8"))["subjects"]
    vocab = json.loads(SUBJECT_VOCAB.read_text(encoding="utf-8"))
    focus = {e["id"]: e for e in json.loads((EVAL_FINAL / "focus_subjects.json").read_text(encoding="utf-8"))["subjects"]}
    texts = [page_match_text(p["text"]) for p in pages]
    lowered = [t.lower() for t in texts]
    matchers = subject_matchers(sq, vocab, focus)
    max_rel = int(SUBJ_MAX_REL_SHARE * len(pages))
    subjects, skipped = {}, Counter()
    for sid, (rule, rx, needles) in sorted(matchers.items()):
        cand = range(len(pages)) if needles is None else \
            [i for i, low in enumerate(lowered) if any(nd in low for nd in needles)]
        rel = [pages[i]["es_id"] for i in cand if rx.search(texts[i])]
        status = sq[sid]["status"]
        if len(rel) < SUBJ_MIN_REL:
            skipped["too_few_pages"] += 1
            continue
        if len(rel) > max_rel and status != "focus":
            skipped["too_many_pages"] += 1
            continue
        subjects[sid] = {"status": status, "kind": sq[sid]["kind"], "rule": rule, "n_rel": len(rel), "rel": rel,
                         "queries": sq[sid]["queries"]}
    skipped["no_usable_name"] = len(sq) - len(matchers)
    write_json(SUBJECT_LABELS, {
        "built": datetime.now().isoformat(timespec="seconds"), "n_pages": len(pages),
        "rule": ("WEAK relevance = the page text (minus its Source PDF and Shelf marks mentioned lines) names the "
                 "subject: focus subjects by their frozen concept_pattern (focus_subjects.json, case-insensitive); "
                 "every other subject by any of its names (subject_vocab names_en/names_he + its name queries) "
                 f"as a whole word/phrase, case-insensitive, Hebrew with 0-2 prefix letters, names under "
                 f"{MIN_NAME_CHARS} letters ignored. Scored when {SUBJ_MIN_REL} <= pages <= {max_rel} "
                 f"({SUBJ_MAX_REL_SHARE:.0%} of the pool; focus subjects are kept above the cap)."),
        "skipped": dict(skipped), "subjects": subjects})
    by = Counter((v["status"], v["kind"]) for v in subjects.values())
    log(f"{len(subjects)} subjects labelled ({dict(by)}); skipped {dict(skipped)}; wrote {SUBJECT_LABELS}")


# ---------------------------------------------------------------------------------------------------------------
# production keyword ranks (read-only ES)
# ---------------------------------------------------------------------------------------------------------------
def prod_alias_expander() -> Callable[[str], List[str]]:
    """Load ``expand_query_aliases`` from production's ``genizah_terminology.py`` (origin/prod-mbp, read-only).

    :returns: The production alias expansion function.
    :rtype: Callable[[str], List[str]]
    """
    src = subprocess.run(["git", "-C", str(REPO), "show", "origin/prod-mbp:src/backend/genizah_terminology.py"],
                         capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("prod_genizah_terminology")
    exec(compile(src, "prod_genizah_terminology.py", "exec"), mod.__dict__)
    return mod.expand_query_aliases


def lexical_body(query: str, expand: Callable[[str], List[str]]) -> dict:
    """Production's bibliography keyword query (``search_bibliography.search_hybrid``, origin/prod-mbp).

    :param query: User query.
    :param expand: Alias expansion function.
    :returns: Query DSL.
    :rtype: dict
    """
    fields = ["author^6.0", "authors^6.0", "title^4.0", "full_text_content^2.5", "description^2.0",
              "subject_keywords^1.5", "shelf_marks_mentioned^1.0"]
    should = [
        {"multi_match": {"query": query, "fields": fields, "type": "best_fields", "fuzziness": "AUTO",
                         "prefix_length": 2}},
        {"match_phrase": {"author": {"query": query, "boost": 10.0}}},
        {"match_phrase": {"title": {"query": query, "boost": 6.0}}},
    ]
    for form in expand(query):
        should.append({"match_phrase": {"full_text_content": {"query": form, "boost": 3.0}}})
        should.append({"match_phrase": {"title": {"query": form, "boost": 4.0}}})
    return {"bool": {"should": should, "minimum_should_match": 1}}


def all_query_texts() -> List[str]:
    """Every unique query text scored here: known-item queries, then the labelled subjects' queries.

    :returns: Unique query texts.
    :rtype: List[str]
    """
    texts = [q["text"] for q in read_jsonl(KI_QUERIES)]
    labels = json.loads(SUBJECT_LABELS.read_text(encoding="utf-8"))["subjects"]
    texts += [q["text"] for s in labels.values() for q in s["queries"]]
    return list(dict.fromkeys(texts))


def lexical(args: argparse.Namespace) -> None:
    """Top-``LEX_SIZE`` production keyword ranking of every query (read-only ``_search``), cached and resumable.

    :param args: Parsed CLI arguments (``sleep``).
    """
    from es_read import search

    expand = prod_alias_expander()
    cache = json.loads(LEXICAL.read_text(encoding="utf-8")) if LEXICAL.exists() else {}
    todo = [t for t in all_query_texts() if t not in cache]
    log(f"lexical: {len(cache)} cached, {len(todo)} to run")
    for i, text in enumerate(todo):
        resp = search(BIB_INDEX, {"query": lexical_body(text, expand), "size": LEX_SIZE, "_source": False})
        cache[text] = [h["_id"] for h in resp["hits"]["hits"]]
        if (i + 1) % 200 == 0:
            write_json(LEXICAL, cache)
            log(f"lexical {i + 1}/{len(todo)}")
        time.sleep(args.sleep)
    write_json(LEXICAL, cache)
    log(f"lexical: wrote {len(cache)} rankings to {LEXICAL}")


# ---------------------------------------------------------------------------------------------------------------
# embedding (GPU job: run through guard.py)
# ---------------------------------------------------------------------------------------------------------------
def load_model(name: str, dev: str, pad_multiple: int):
    """Load one of :data:`MODELS` with the production contract (fp32, max_seq 8192, no default prompt).

    :param name: ``base`` or ``tuned``.
    :param dev: torch device.
    :param pad_multiple: Pad batches to a multiple of this many tokens (1 = off).
    :returns: SentenceTransformer.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    spec = MODELS[name]
    t0 = time.time()
    model = SentenceTransformer(spec["id"], revision=spec["revision"], device=dev,
                                model_kwargs={"dtype": torch.float32})
    model.max_seq_length = MAX_SEQ
    model.default_prompt_name = None  # the tuned card's "query" prompt is the stock web-search one; never apply it
    if pad_multiple > 1:
        module = model[0]
        kwargs = dict(module.processing_kwargs)
        kwargs["text"] = {**kwargs.get("text", {}), "pad_to_multiple_of": pad_multiple}
        module.processing_kwargs = kwargs
    log(f"loaded {spec['id']}@{spec['revision'][:8]} on {dev} ({next(model.parameters()).dtype}) "
        f"in {time.time() - t0:.0f} s")
    return model


def encode_sorted(model, texts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    """Encode texts length-sorted in bounded batches and return them in input order.

    :param model: SentenceTransformer.
    :param texts: Texts (already prefixed for queries).
    :param args: Batch bounds (``step0_local`` names).
    :returns: (n, d) float32 normalised vectors.
    :rtype: np.ndarray
    """
    from step0_local import encode_with_backoff, token_lengths

    lengths = token_lengths(model.tokenizer, list(texts), MAX_SEQ)
    order = np.argsort(lengths, kind="stable")
    enc = encode_with_backoff(model, [texts[j] for j in order], lengths[order], args)
    out = np.empty_like(enc)
    out[order] = enc
    return out


def contract_check(name: str, model, args: argparse.Namespace, out: Path, pages: Dict[str, dict],
                   tok: Dict[str, int]) -> dict:
    """Embed the index ``_meta`` canary string and (for base) re-embed the stored sample pages short enough for MPS.

    For ``base`` both must reproduce production's stored vectors (cos >= ``canary_min_cos``), which proves this run
    embeds the pages exactly as the production index did; the long sample pages are checked by
    :func:`long_contract_check` once the CPU phase has embedded them. For ``tuned`` the canary vector is the one a v9
    index ``_meta`` would carry.

    :param name: ``base`` or ``tuned``.
    :param model: Loaded model.
    :param args: Batch bounds, ``mps_max_tokens``, ``canary_min_cos``.
    :param out: Vector dir.
    :param pages: ``es_id -> page row``.
    :param tok: ``text_hash -> token count``.
    :returns: Check record (also written to ``out/canary.json``).
    :rtype: dict
    """
    meta = json.loads(INDEX_META.read_text(encoding="utf-8"))
    canary = encode_sorted(model, [meta["canary"]["string"]], args)[0]
    stored_canary = np.asarray(meta["canary"]["vector"], dtype=np.float32)
    rec = {"model": name, "canary_cos_vs_index_meta": round(float(canary @ stored_canary), 6),
           "canary_vector": canary.round(6).tolist()}
    if name == "base":
        z = np.load(STORED_SAMPLE)
        keep = [j for j, i in enumerate(z["ids"]) if tok[pages[str(i)]["text_hash"]] <= args.mps_max_tokens]
        ids = [str(z["ids"][j]) for j in keep]
        fresh = encode_sorted(model, [pages[i]["text"] for i in ids], args)
        cos = (fresh * z["vectors"][keep]).sum(1)
        rec.update({"stored_sample_mps_n": len(ids), "stored_sample_mps_min_cos": round(float(cos.min()), 6),
                    "stored_sample_mps_mean_cos": round(float(cos.mean()), 6),
                    "stored_sample_mps_max_tokens": max(tok[pages[i]["text_hash"]] for i in ids)})
        rec["ok"] = bool(cos.min() >= args.canary_min_cos and rec["canary_cos_vs_index_meta"] >= args.canary_min_cos)
    else:
        rec["ok"] = True
    write_json(out / "canary.json", rec)
    log(f"contract check {name}: { {k: v for k, v in rec.items() if k != 'canary_vector'} }")
    return rec


def long_contract_check(name: str, out: Path, pages: Dict[str, dict], tok: Dict[str, int],
                        args: argparse.Namespace) -> dict:
    """Compare the CPU-embedded long sample pages with production's stored vectors (base only).

    :param name: ``base`` or ``tuned``.
    :param out: Vector dir (``canary.json`` is updated).
    :param pages: ``es_id -> page row``.
    :param tok: ``text_hash -> token count``.
    :param args: ``mps_max_tokens``, ``canary_min_cos``.
    :returns: Updated check record.
    :rtype: dict
    """
    rec = json.loads((out / "canary.json").read_text())
    if name != "base":
        return rec
    z = np.load(STORED_SAMPLE)
    keep = [j for j, i in enumerate(z["ids"]) if tok[pages[str(i)]["text_hash"]] > args.mps_max_tokens]
    if keep:
        fresh = np.stack([np.load(out / "long" / f"{pages[str(z['ids'][j])]['text_hash']}.npy") for j in keep])
        cos = (fresh * z["vectors"][keep]).sum(1)
        rec.update({"stored_sample_cpu_long_n": len(keep), "stored_sample_cpu_long_min_cos": round(float(cos.min()), 6),
                    "stored_sample_cpu_long_max_tokens": max(tok[pages[str(z["ids"][j])]["text_hash"]] for j in keep)})
        rec["ok"] = bool(rec["ok"] and cos.min() >= args.canary_min_cos)
    write_json(out / "canary.json", rec)
    log(f"long contract check {name}: { {k: v for k, v in rec.items() if k != 'canary_vector'} }")
    return rec


def embed_long(model, texts: Dict[str, str], hashes: List[str], out: Path, args: argparse.Namespace) -> None:
    """Embed the long texts one by one on the CPU with flash attention (resumable: one ``.npy`` per text hash).

    Both MPS SDPA and transformers' CPU path materialise the n x n attention (16 heads x 7.6k^2 fp32 is ~4 GB per
    tensor, over the guard's cap). transformers asks torch for ``enable_gqa=True`` (Qwen3 has 8 KV heads for 16
    query heads), which sends CPU SDPA to the math kernel; with GQA left to transformers (``repeat_kv``) and one
    unpadded sequence (no mask, ``is_causal=True``) CPU SDPA uses its flash kernel, O(n) memory. Outputs agree with
    the math kernel to ~1e-6 (checked on a random model); the base contract check compares the longest sample pages
    with production's stored vectors.

    :param model: SentenceTransformer (moved to the CPU here; batch padding switched off).
    :param texts: ``text_hash -> text``.
    :param hashes: Hashes of the long texts.
    :param out: Vector dir (writes ``out/long/<hash>.npy``).
    :param args: Parsed CLI arguments (``cpu_threads``, ``soft_max_gb``).
    """
    import torch
    import transformers.integrations.sdpa_attention as sdpa_attention

    from guard import child_rss_gb

    ldir = out / "long"
    ldir.mkdir(exist_ok=True)
    todo = [h for h in hashes if not (ldir / f"{h}.npy").exists()]
    if not todo:
        return
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
    if model.device.type != "cpu":
        model.to("cpu")
        torch.mps.empty_cache()
    module = model[0]
    kwargs = dict(module.processing_kwargs)
    kwargs["text"] = {k: v for k, v in kwargs.get("text", {}).items() if k != "pad_to_multiple_of"}
    module.processing_kwargs = kwargs
    torch.set_num_threads(args.cpu_threads)
    t_start, tok_done = time.time(), 0
    for i, h in enumerate(todo):
        t0 = time.time()
        vec = model.encode([texts[h]], batch_size=1, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)[0].astype(np.float32)
        save_npy(ldir / f"{h}.npy", vec)
        n_tok = len(model.tokenizer(texts[h], truncation=True, max_length=MAX_SEQ)["input_ids"])
        tok_done += n_tok
        fp = child_rss_gb(os.getpid())
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            log(f"long {i + 1}/{len(todo)} tok={n_tok} {time.time() - t0:.1f} s; session "
                f"{tok_done / (time.time() - t_start):.0f} tok/s; footprint {fp:.2f} GB")
        if args.soft_max_gb and fp > args.soft_max_gb and i + 1 < len(todo):
            log(f"footprint {fp:.2f} GB > soft limit {args.soft_max_gb} GB: exiting {RESTART_RC} for a fresh relaunch")
            (out / "LOCK").unlink(missing_ok=True)
            sys.exit(RESTART_RC)


def embed(args: argparse.Namespace) -> None:
    """Embed the pool (MPS shards + long texts on CPU) and all queries with one model; resumable.

    :param args: Parsed CLI arguments.
    """
    os.environ.setdefault("HF_HOME", str(AUDIT_ROOT / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    import step0_local
    from embed_utils import device

    out = VEC_ROOT / args.model
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    qjson = out / "queries.json"
    qtexts = all_query_texts()
    docs_done = meta_path.exists() and json.loads(meta_path.read_text())["complete"]
    q_fresh = qjson.exists() and json.loads(qjson.read_text(encoding="utf-8"))["texts"] == qtexts
    if docs_done and q_fresh:
        log(f"{out} already complete")
        return
    pages = read_jsonl(PAGES)
    by_id = {p["es_id"]: p for p in pages}
    texts = {p["text_hash"]: p["text"] for p in pages}
    lock = step0_local.acquire_lock(out)
    if not os.environ.get("AUDIT_THREADS"):
        torch.set_num_threads(args.cpu_threads)
    dev = device()
    model = load_model(args.model, dev, args.pad_multiple)
    lengths = step0_local.token_lengths(model.tokenizer, [texts[h] for h in sorted(texts)], MAX_SEQ)
    tok = dict(zip(sorted(texts), lengths.tolist()))
    canary_path = out / "canary.json"
    canary = json.loads(canary_path.read_text()) if canary_path.exists() else contract_check(args.model, model, args,
                                                                                           out, by_id, tok)
    if not canary["ok"]:
        raise SystemExit(f"contract check failed: { {k: v for k, v in canary.items() if k != 'canary_vector'} }")
    if not q_fresh:
        t0 = time.time()
        qmat = encode_sorted(model, [QUERY_PREFIX + t for t in qtexts], args)
        save_npy(out / "queries.npy", qmat)
        write_json(qjson, {"prefix": QUERY_PREFIX, "texts": qtexts})
        log(f"queries: {len(qtexts)} in {time.time() - t0:.0f} s")
    if docs_done:
        lock.unlink(missing_ok=True)
        log(f"{out}: page vectors already complete; queries refreshed")
        return
    short ={h: t for h, t in texts.items() if tok[h] <= args.mps_max_tokens}
    long_hashes = sorted((h for h in texts if tok[h] > args.mps_max_tokens), key=lambda h: (tok[h], h))
    log(f"{len(texts)} unique page texts: {len(short)} on {dev} (<= {args.mps_max_tokens} tokens, "
        f"{sum(tok[h] for h in short)} tokens), {len(long_hashes)} long on cpu "
        f"({sum(tok[h] for h in long_hashes)} tokens, max {max(tok.values())})")
    t0 = time.time()
    plan = step0_local.embed_unique(types.SimpleNamespace(model=model), short, out, args)  # exits 99 above soft limit
    embed_long(model, texts, long_hashes, out, args)
    canary = long_contract_check(args.model, out, by_id, tok, args)
    if not canary["ok"]:
        raise SystemExit(f"long contract check failed: { {k: v for k, v in canary.items() if k != 'canary_vector'} }")
    uniq = np.concatenate([np.load(out / "unique" / f"shard_{s:04d}.npy") for s in range(len(plan["bounds"]))])
    vec = {h: uniq[i] for i, h in enumerate(plan["hashes"])}
    vec.update({h: np.load(out / "long" / f"{h}.npy") for h in long_hashes})
    mat = np.stack([vec[p["text_hash"]] for p in pages]).astype(np.float32)
    if not np.isfinite(mat).all() or not np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-3):
        raise ValueError("page vectors are not finite unit vectors")
    save_npy(out / "pages.npy", mat)
    write_json(out / "ids.json", [p["es_id"] for p in pages])
    write_json(meta_path, {"complete": True, "written": datetime.now().isoformat(timespec="seconds"),
                           "model": MODELS[args.model], "dtype": "float32", "max_seq": MAX_SEQ,
                           "document_prefix": "", "query_prefix": QUERY_PREFIX, "normalised": True,
                           "n_pages": len(pages), "n_unique_texts": len(texts), "n_tokens": int(sum(tok.values())),
                           "n_long_cpu": len(long_hashes), "mps_max_tokens": args.mps_max_tokens,
                           "device": dev, "seconds_last_session": round(time.time() - t0, 1),
                           "batching": {k: getattr(args, k) for k in ("pad_multiple", "max_batch", "max_batch_tokens",
                                                                      "max_attn", "shard_size", "shard_tokens")}})
    lock.unlink(missing_ok=True)
    log(f"wrote {out}/pages.npy ({mat.shape})")


# ---------------------------------------------------------------------------------------------------------------
# scoring (numpy only)
# ---------------------------------------------------------------------------------------------------------------
def load_model_vectors(name: str, pages: List[dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray], dict]:
    """Page matrix (pool order), query vectors by text and the canary record of one model.

    :param name: ``base`` or ``tuned``.
    :param pages: Pool rows (``pages.jsonl`` order).
    :returns: (P, query text -> vector, canary record).
    :rtype: Tuple[np.ndarray, Dict[str, np.ndarray], dict]
    """
    out = VEC_ROOT / name
    if json.loads((out / "ids.json").read_text()) != [p["es_id"] for p in pages]:
        raise ValueError(f"{out}: page order differs from pages.jsonl")
    P = np.load(out / "pages.npy")
    q = json.loads((out / "queries.json").read_text(encoding="utf-8"))
    if q["prefix"] != QUERY_PREFIX:
        raise ValueError(f"{out}: queries were not encoded with the production instruction")
    Q = np.load(out / "queries.npy")
    return P, dict(zip(q["texts"], Q)), json.loads((out / "canary.json").read_text())


def rrf_rank(sem_order: np.ndarray, lex_ids: List[str], pos: Dict[str, int], hits: np.ndarray,
             weights: Tuple[float, float] = HYBRID_WEIGHTS, cand: int = LEX_SIZE) -> int:
    """Rank of the best hit in production's weighted-RRF fusion of the top-``cand`` semantic and keyword lists.

    A hit absent from both candidate lists gets rank ``cand * 2 + 1`` (not retrievable by the hybrid endpoint).
    Ties count against the hit.

    :param sem_order: Pool indices by descending cosine.
    :param lex_ids: Keyword ranking (es ids, best first).
    :param pos: es id -> pool index.
    :param hits: Pool indices that count as hits.
    :param weights: (semantic, keyword) weights.
    :param cand: Candidates per list.
    :returns: 1-based fused rank.
    :rtype: int
    """
    score: Dict[int, float] = defaultdict(float)
    for r, i in enumerate(sem_order[:cand], start=1):
        score[int(i)] += weights[0] / (RRF_K + r)
    for r, d in enumerate(lex_ids[:cand], start=1):
        score[pos[d]] += weights[1] / (RRF_K + r)
    best = max((score.get(int(h), 0.0) for h in hits), default=0.0)
    if best <= 0:
        return cand * 2 + 1
    hitset = {int(h) for h in hits}
    return 1 + sum(1 for i, s in score.items() if s >= best and i not in hitset)


def known_item_ranks(P: np.ndarray, qvec: Dict[str, np.ndarray], queries: List[dict], pages: List[dict],
                     lex: Optional[dict]) -> Dict[str, List[int]]:
    """Strict, lenient (+-1 page) and hybrid ranks of every known-item query.

    :param P: Page matrix.
    :param qvec: Query text -> vector.
    :param queries: Known-item queries.
    :param pages: Pool rows.
    :param lex: Keyword rankings by query text (None = skip hybrid).
    :returns: ``{"strict": [...], "lenient": [...], "hybrid": [...], "cos_target": [...], "cos_top1": [...]}``.
    :rtype: Dict[str, List[int]]
    """
    from eval_v3 import hit_rank

    pos = {p["es_id"]: i for i, p in enumerate(pages)}
    by_hash: Dict[str, List[int]] = defaultdict(list)
    by_book: Dict[str, Dict[int, int]] = defaultdict(dict)
    for i, p in enumerate(pages):
        by_hash[p["text_hash"]].append(i)
        if p["book_uuid"] and p["page_seq"] is not None:
            by_book[p["book_uuid"]][int(p["page_seq"])] = i
    out = defaultdict(list)
    book_of = np.array([p["book"] for p in pages])
    books = sorted(set(book_of.tolist()))
    book_idx = np.array([books.index(b) for b in book_of])
    Q = np.stack([qvec[q["text"]] for q in queries])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = Q @ P.T
    for q, s in zip(queries, S):
        t = pos[q["es_id"]]
        strict = np.array(by_hash[pages[t]["text_hash"]])
        nb = [by_book[pages[t]["book_uuid"]].get(int(pages[t]["page_seq"]) + d) for d in (-1, 1)] \
            if pages[t]["book_uuid"] and pages[t]["page_seq"] is not None else []
        lenient = np.array(sorted(set(strict.tolist()) | {i for i in nb if i is not None}))
        out["strict"].append(hit_rank(s, strict))
        out["lenient"].append(hit_rank(s, lenient))
        # diagnostic split of the strict rank: is the right BOOK found (books ranked by their best page), and where
        # does the target page sit among its own book's pages?
        book_best = np.full(len(books), -np.inf)
        np.maximum.at(book_best, book_idx, s)
        out["book_rank"].append(int((book_best >= book_best[book_idx[t]]).sum()))
        same = np.flatnonzero(book_idx == book_idx[t])
        out["within_book_rank"].append(hit_rank(s, strict, candidates=same))
        out["cos_target"].append(float(s[t]))
        out["cos_top1"].append(float(s.max()))
        if lex is not None:
            order = np.argsort(-s, kind="stable")
            out["hybrid"].append(rrf_rank(order, lex[q["text"]], pos, strict))
            out["hybrid_kw70"].append(rrf_rank(order, lex[q["text"]], pos, strict, weights=(0.3, 0.7)))
            out["chat_hybrid"].append(rrf_rank(order, lex[q["text"]], pos, strict, weights=CHAT_WEIGHTS, cand=CHAT_CAND))
            out["chat_hybrid_kw70"].append(rrf_rank(order, lex[q["text"]], pos, strict, weights=(0.3, 0.7),
                                                    cand=CHAT_CAND))
            lex_pos = [r for r, d in enumerate(lex[q["text"]], start=1) if pos[d] in set(strict.tolist())]
            out["keyword_only"].append(lex_pos[0] if lex_pos else LEX_SIZE + 1)
    return dict(out)


def subject_scores(P: np.ndarray, qvec: Dict[str, np.ndarray], labels: dict, pages: List[dict]) -> dict:
    """AP / P@10 / R@100 of every labelled subject query against the page pool.

    :param P: Page matrix.
    :param qvec: Query text -> vector.
    :param labels: ``subject_labels.json["subjects"]``.
    :param pages: Pool rows.
    :returns: ``sid -> {"status", "kind", "n_rel", "queries": [{"text", "source", "lang", AP, P@10, R@100}]}``.
    :rtype: dict
    """
    from eval_v3 import positive_ranks, ranking_metrics

    pos = {p["es_id"]: i for i, p in enumerate(pages)}
    out = {}
    for sid, s in labels.items():
        rel = np.array(sorted(pos[d] for d in s["rel"]))
        mask = np.ones(len(pages), bool)
        mask[rel] = False
        Q = np.stack([qvec[q["text"]] for q in s["queries"]])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            S = Q @ P.T
        rows = []
        for q, sc in zip(s["queries"], S):
            m = ranking_metrics(positive_ranks(sc[rel], sc[mask]), len(rel))
            rows.append({"text": q["text"], "source": q["source"], "lang": q["lang"], **m})
        out[sid] = {"status": s["status"], "kind": s["kind"], "n_rel": s["n_rel"], "queries": rows}
    return out


def rr_clusters(queries: List[dict], ranks: Sequence[int], metric: str, keep: Callable[[dict], bool]) -> Dict[str, List[float]]:
    """Per-page known-item values for the paired bootstrap (cluster = target page).

    :param queries: Known-item queries.
    :param ranks: Their ranks (same order).
    :param metric: ``RR``, ``R@1``, ``R@5``, ``R@10`` or ``R@50``.
    :param keep: Query filter.
    :returns: ``es_id -> values``.
    :rtype: Dict[str, List[float]]
    """
    out: Dict[str, List[float]] = defaultdict(list)
    for q, r in zip(queries, ranks):
        if keep(q):
            out[q["es_id"]].append(1.0 / r if metric == "RR" else float(r <= int(metric[2:])))
    return out


def training_anchor_set() -> set:
    """Normalised anchors of the tuned model's training pairs (``train_mix_v3final.jsonl``).

    :returns: Anchors lower-cased, niqqud-stripped, punctuation-collapsed (as :func:`norm_query`).
    :rtype: set
    """
    out = set()
    with open(TRAIN_MIX, encoding="utf-8") as fh:
        for line in fh:
            out.add(norm_query(json.loads(line)["anchor"]))
    return out


def norm_query(text: str) -> str:
    """Lower-case, strip niqqud, keep word characters only (anchor comparison key).

    :param text: Query or anchor.
    :returns: Normalised key.
    :rtype: str
    """
    return " ".join(re.findall(r"[^\W_]+", re.sub(r"[֑-ׇ]", "", text.lower())))


def subject_clusters(scores: dict, metric: str, keep: Callable[[str, dict, dict], bool]) -> Dict[str, List[float]]:
    """Per-subject per-query values for the paired bootstrap (cluster = subject).

    :param scores: Output of :func:`subject_scores`.
    :param metric: ``AP``, ``P@10`` or ``R@100``.
    :param keep: Filter ``(sid, subject, query row) -> bool``.
    :returns: ``sid -> values``.
    :rtype: Dict[str, List[float]]
    """
    out = {}
    for sid, s in scores.items():
        vals = [q[metric] for q in s["queries"] if keep(sid, s, q)]
        if vals:
            out[sid] = vals
    return out


def macro(cl: Dict[str, List[float]]) -> Optional[float]:
    """Mean over clusters of the per-cluster mean.

    :param cl: Cluster -> values.
    :returns: Macro mean (None when empty).
    :rtype: Optional[float]
    """
    return round(float(np.mean([np.mean(v) for v in cl.values()])), 4) if cl else None


def score(args: argparse.Namespace) -> None:
    """Score base and tuned on the same pool and queries; write the report with paired bootstrap CIs.

    :param args: Parsed CLI arguments (``n_resamples``).
    """
    from eval_v3 import paired_bootstrap, rank_stats

    pages = read_jsonl(PAGES)
    queries = read_jsonl(KI_QUERIES)
    labels = json.loads(SUBJECT_LABELS.read_text(encoding="utf-8"))
    lex = json.loads(LEXICAL.read_text(encoding="utf-8")) if LEXICAL.exists() else None
    if lex is not None and any(q["text"] not in lex for q in queries):
        lex = None
    res = {}
    for name in ("base", "tuned"):
        P, qvec, canary = load_model_vectors(name, pages)
        res[name] = {"P": P, "ki": known_item_ranks(P, qvec, queries, pages, lex),
                     "subj": subject_scores(P, qvec, labels["subjects"], pages),
                     "canary": {k: v for k, v in canary.items() if k != "canary_vector"},
                     "canary_vector": np.asarray(canary["canary_vector"])}
    b, t = res["base"], res["tuned"]
    n = args.n_resamples

    def ki_block(rank_key: str, keep: Callable[[dict], bool], metrics: Sequence[str] = ("RR", "R@10", "R@50")) -> dict:
        sel = [i for i, q in enumerate(queries) if keep(q)]
        blk = {"n_queries": len(sel), "base": rank_stats([b["ki"][rank_key][i] for i in sel]),
               "tuned": rank_stats([t["ki"][rank_key][i] for i in sel])}
        for name, r in (("base", b), ("tuned", t)):
            blk[name][f"R@{CHAT_TOP}"] = round(float(np.mean([r["ki"][rank_key][i] <= CHAT_TOP for i in sel])), 4)
        for m in metrics:
            blk[f"paired_{m}_tuned_minus_base"] = paired_bootstrap(
                rr_clusters(queries, b["ki"][rank_key], m, keep), rr_clusters(queries, t["ki"][rank_key], m, keep), n)
        return blk

    slices = {"all": lambda q: True, "en": lambda q: q["lang"] == "en", "he": lambda q: q["lang"] == "he",
              "en_on_he_page": lambda q: q["lang"] == "en" and q["page_lang"] == "he",
              "en_on_non_he_page": lambda q: q["lang"] == "en" and q["page_lang"] == "non-he",
              "he_on_he_page": lambda q: q["lang"] == "he" and q["page_lang"] == "he",
              "he_on_non_he_page (cross-lingual)": lambda q: q["lang"] == "he" and q["page_lang"] == "non-he"}
    known_item = {"dense_strict": {k: ki_block("strict", f) for k, f in slices.items()},
                  "dense_lenient_pm1_page": {k: ki_block("lenient", slices[k]) for k in ("all", "en", "he")}}
    if lex is not None:
        known_item["prod_hybrid_rrf_60_40_strict"] = {k: ki_block("hybrid", slices[k]) for k in ("all", "en", "he")}
        known_item["prod_hybrid_rrf_30_70_strict"] = {k: ki_block("hybrid_kw70", slices[k]) for k in ("all", "en", "he")}
        chat_metrics = ("RR", f"R@{CHAT_TOP}", "R@10")
        known_item["chat_default_hybrid_rrf_50_50_cand40_strict"] = {
            "note": ("what the chat agent actually runs: SearchAction defaults 50/50 and num_results=5 -> candidate_size "
                     f"40, synthesis sees the top {CHAT_TOP} (so R@{CHAT_TOP} is the user-visible recall)"),
            **{k: ki_block("chat_hybrid", slices[k], chat_metrics) for k in ("all", "en", "he")}}
        known_item["chat_kw70_hybrid_rrf_30_70_cand40_strict"] = {
            k: ki_block("chat_hybrid_kw70", slices[k], chat_metrics) for k in ("all", "en", "he")}
        known_item["keyword_only_reference (same for both models)"] = {
            "note": f"production keyword query, top-{LEX_SIZE} only: a target outside it gets rank {LEX_SIZE + 1}, "
                    "so R@100 is not meaningful here",
            **{k: rank_stats([b["ki"]["keyword_only"][i] for i, q in enumerate(queries) if slices[k](q)])
               for k in ("all", "en", "he")}}
    known_item["diagnostic_book_vs_page"] = {
        "how": ("book_rank = rank of the target's book when the 41 books are ranked by their best-scoring page; "
                "within_book_rank = rank of the target page among the pages of its own book (ties count against)."),
        **{f"{part}_{k}": {"base": rank_stats([b["ki"][part][i] for i, q in enumerate(queries) if slices[k](q)]),
                           "tuned": rank_stats([t["ki"][part][i] for i, q in enumerate(queries) if slices[k](q)]),
                           "paired_RR_tuned_minus_base": paired_bootstrap(
                               rr_clusters(queries, b["ki"][part], "RR", slices[k]),
                               rr_clusters(queries, t["ki"][part], "RR", slices[k]), n)}
           for part in ("book_rank", "within_book_rank") for k in ("all", "en", "he")}}
    # per-book: does the tuned model lose a whole book (e.g. Hebrew-language monographs)?
    per_book = {}
    for book in sorted({q["book"] for q in queries}):
        keep = (lambda bk: (lambda q: q["book"] == bk))(book)
        sel = [i for i, q in enumerate(queries) if keep(q)]
        per_book[book[:90]] = {"n_queries": len(sel),
                               "base_MRR": rank_stats([b["ki"]["strict"][i] for i in sel])["MRR"],
                               "tuned_MRR": rank_stats([t["ki"]["strict"][i] for i in sel])["MRR"]}
    per_query = [{"qid": q["qid"], "lang": q["lang"], "page_lang": q["page_lang"], "text": q["text"],
                  "rank_base": b["ki"]["strict"][i], "rank_tuned": t["ki"]["strict"][i],
                  **({"hybrid_rank_base": b["ki"]["hybrid"][i], "hybrid_rank_tuned": t["ki"]["hybrid"][i],
                      "keyword_rank": b["ki"]["keyword_only"][i]} if lex is not None else {})}
                 for i, q in enumerate(queries)]

    def subj_block(keep: Callable[[str, dict, dict], bool]) -> dict:
        blk = {}
        for m in ("AP", "P@10", "R@100"):
            cb, ct = subject_clusters(b["subj"], m, keep), subject_clusters(t["subj"], m, keep)
            blk[m] = {"base": macro(cb), "tuned": macro(ct), "paired_tuned_minus_base": paired_bootstrap(cb, ct, n)}
        blk["n_subjects"] = blk["AP"]["paired_tuned_minus_base"]["n_clusters"]
        blk["n_queries"] = blk["AP"]["paired_tuned_minus_base"]["n_queries"]
        return blk

    subj = {"focus_all_sources": subj_block(lambda sid, s, q: s["status"] == "focus"),
            "focus_probe_llm_t2 (paraphrase)": subj_block(lambda sid, s, q: s["status"] == "focus"
                                                          and q["source"] in ("probe", "llm_t2")),
            "focus_name": subj_block(lambda sid, s, q: s["status"] == "focus" and q["source"] == "name"),
            "named_subjects_name_queries": subj_block(lambda sid, s, q: s["status"] != "focus"),
            "named_subjects_en": subj_block(lambda sid, s, q: s["status"] != "focus" and q["lang"] == "en"),
            "named_subjects_he": subj_block(lambda sid, s, q: s["status"] != "focus" and q["lang"] == "he")}
    # The FINAL model trains on every subject, and most named-subject name queries are literally training anchors
    # (subject name -> catalogue record): split them so the in-distribution share of the gain is visible.
    anchors = training_anchor_set() if TRAIN_MIX.exists() else None
    if anchors is not None:
        subj["named_subjects_query_is_training_anchor"] = subj_block(
            lambda sid, s, q: s["status"] != "focus" and norm_query(q["text"]) in anchors)
        subj["named_subjects_query_not_training_anchor"] = subj_block(
            lambda sid, s, q: s["status"] != "focus" and norm_query(q["text"]) not in anchors)
        subj["training_anchor_note"] = (
            f"{TRAIN_MIX.name}: exact (normalised) training anchors among the scored queries -- focus: "
            f"{sum(norm_query(q['text']) in anchors for s in labels['subjects'].values() if s['status'] == 'focus' for q in s['queries'])}"
            f"/{sum(len(s['queries']) for s in labels['subjects'].values() if s['status'] == 'focus')}, named: "
            f"{sum(norm_query(q['text']) in anchors for s in labels['subjects'].values() if s['status'] != 'focus' for q in s['queries'])}"
            f"/{sum(len(s['queries']) for s in labels['subjects'].values() if s['status'] != 'focus')}. Every subject "
            "(focus included) was trained on in the FINAL run, so named-subject results are in-distribution query "
            "text applied to an unseen document type, not a generalisation test; the focus probe / llm_t2 block is "
            "the paraphrase test.")
    for kind in sorted({s["kind"] for s in labels["subjects"].values() if s["status"] != "focus"}):
        subj[f"named_subjects_kind={kind}"] = subj_block(
            (lambda kd: (lambda sid, s, q: s["status"] != "focus" and s["kind"] == kd))(kind))
    focus_per_subject = {sid: {"n_rel_pages": s["n_rel"],
                               "base_AP": round(float(np.mean([q["AP"] for q in s["queries"]])), 4),
                               "tuned_AP": round(float(np.mean([q["AP"] for q in t["subj"][sid]["queries"]])), 4)}
                         for sid, s in b["subj"].items() if s["status"] == "focus"}
    # vector-space geometry of the pool under each model (calibration context, not a quality metric)
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(pages), size=(20000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    geometry = {name: {"random_page_pair_cos_median": round(float(np.median((r["P"][pairs[:, 0]] * r["P"][pairs[:, 1]]).sum(1))), 4),
                       "known_item_target_cos_median": round(float(np.median(r["ki"]["cos_target"])), 4),
                       "known_item_top1_cos_median": round(float(np.median(r["ki"]["cos_top1"])), 4)}
                for name, r in res.items()}
    report = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "question": ("Does the v3-final tuned text embedder hurt or help retrieval of BIBLIOGRAPHY (scholarship) pages, "
                     "which it was not trained on? Decides whether index v9 can re-embed the bibliography index with "
                     "it (production serves one query embedder for both indexes)."),
        "models": MODELS, "contract": {"document": "raw text (full_text_content), no prefix", "query_prefix": QUERY_PREFIX,
                                       "max_seq": MAX_SEQ, "dtype": "float32", "normalised": True},
        "pool": {"index": BIB_INDEX, "n_pages": len(pages), "n_unique_texts": len({p["text_hash"] for p in pages}),
                 "n_books": len({p["book"] for p in pages}),
                 "n_hebrew_pages": sum(p["he_share"] >= HE_PAGE_SHARE for p in pages),
                 "text": "full_text_content as stored (= BibliographyDocument.create_text_representation, incl. "
                         "'Shelf marks mentioned' lines)"},
        "contract_checks": {"base": b["canary"], "tuned": t["canary"],
                            "canary_cos_base_vs_tuned": round(float(np.sum(b["canary_vector"] * t["canary_vector"])), 6),
                            "base_longest_pages_vs_stored_prod_vectors": (
                                json.loads((WORK / "long_tail_contract_check.json").read_text())
                                if (WORK / "long_tail_contract_check.json").exists() else None),
                            "base_full_pool_vs_stored_prod_vectors (independent verification, read-only scroll)": (
                                json.loads((AUDIT_ROOT / "v9_biblio_verify" / "full_contract.json").read_text())
                                if (AUDIT_ROOT / "v9_biblio_verify" / "full_contract.json").exists() else None),
                            "tokenizer": ("the tuned repo's tokenizer triggers transformers' 'incorrect regex pattern / "
                                          "fix_mistral_regex' warning on load; checked: its token ids are identical to "
                                          "the base tokenizer's on all 4944 pages (do NOT set fix_mistral_regex=True "
                                          "in the embedding service; that would change tokenization).")},
        "known_item": {"n_pages": len({q["es_id"] for q in queries}), "n_queries": len(queries),
                       "query_file": str(KI_QUERIES),
                       "how": ("Seeded (seed 9) sample of 150 pages with >= 600 chars of OCR body whose summary is not a "
                               "reference list / index / contents / blank / front matter, at most 8 per book; two "
                               "hand-written queries per page (one English, one Hebrew) that the page answers, "
                               "paraphrased (no shelf marks, no copied distinctive strings). Rank of the target over all "
                               "pages (ties count against); lenient = target or its +-1 neighbour page in the same book; "
                               "hybrid = production weighted RRF (k=60, semantic 0.6 / keyword 0.4) of the top-50 dense "
                               "list and production's keyword query run read-only on ES (unretrievable = rank 101); "
                               "chat_default = the chat agent's own call (50/50, top-40 candidate lists, top 5 shown; "
                               "unretrievable = rank 81). Paired bootstrap clustered by target page."),
                       **known_item, "per_book_strict": per_book, "per_query": per_query},
        "subjects": {"labels": labels["rule"], "skipped": labels["skipped"], **subj,
                     "focus_per_subject": focus_per_subject},
        "geometry": geometry,
        "n_resamples": n,
    }
    report["decision"] = decision(report)
    write_json(RESULTS, report)
    log(f"wrote {RESULTS}")
    print_summary(report)


KEEP_BASE_REQUIRES = [
    "Production has ONE query embedder: src/backend/embedding_client.py builds a module-level `embedding_client` "
    "(EMBEDDING_SERVICE_URL -> one embedding container loading EMBEDDING_MODEL_NAME@EMBEDDING_MODEL_REVISION), and "
    "both search_service.py (merged index) and search_bibliography.py (`search`, `search_hybrid`, and the third "
    "call near l.555) embed the user query through it.",
    "The startup gate (app.py `verify_embedding_compatibility`, ~l.193-231) re-embeds each index's `_meta` canary "
    "through that one client; any cosine <= 0.99 sets the GLOBAL `embedding_client.drift_reason`, after which "
    "`get_embedding` raises for every caller, i.e. semantic search is switched off for BOTH indexes (lexical "
    "survives). A tuned v9 merged index next to the base-embedded bibliography index fails it: the tuned model "
    "reproduces the bibliography canary at cosine 0.41.",
    "Keeping base for bibliography therefore needs: (1) a second embedding container (or a model selector in "
    "/embed) serving the pinned base model; (2) a second EmbeddingClient (e.g. BIBLIOGRAPHY_EMBEDDING_SERVICE_URL) "
    "used by search_bibliography.py; (3) per-client drift state, with the startup gate checking each index against "
    "its own client so one mismatch does not disable the other index; (4) one extra query embedding per chat turn "
    "that searches both indexes (parallelisable), and ~2.4 GB more RAM for a second fp32 0.6B model on the MBP; "
    "(5) two pinned contracts / canaries to maintain from then on.",
]


def decision(rep: dict) -> dict:
    """Rule-based verdict from the paired CIs (no metric judged by eye).

    Harm = a primary comparison whose 95 % paired CI lies entirely below 0. Watch = a primary point estimate below
    0 whose CI still includes 0.

    :param rep: Report (known_item and subjects sections filled).
    :returns: ``{"verdict", "harm", "watch", "gains", "rule", "keep_base_would_require"}``.
    :rtype: dict
    """
    ki, sj = rep["known_item"], rep["subjects"]
    primary = {f"known-item dense RR {k}": ki["dense_strict"][k]["paired_RR_tuned_minus_base"]
               for k in ("all", "en", "he")}
    primary.update({f"known-item dense R@10 {k}": ki["dense_strict"][k]["paired_R@10_tuned_minus_base"]
                    for k in ("all", "en", "he")})
    if "prod_hybrid_rrf_60_40_strict" in ki:
        primary.update({f"known-item prod hybrid 60/40 RR {k}": ki["prod_hybrid_rrf_60_40_strict"][k]["paired_RR_tuned_minus_base"]
                        for k in ("all", "en", "he")})
        primary.update({f"known-item prod hybrid 30/70 RR {k}": ki["prod_hybrid_rrf_30_70_strict"][k]["paired_RR_tuned_minus_base"]
                        for k in ("all", "en", "he")})
    if "chat_default_hybrid_rrf_50_50_cand40_strict" in ki:
        chat = ki["chat_default_hybrid_rrf_50_50_cand40_strict"]
        primary.update({f"known-item chat default hybrid 50/50 RR {k}": chat[k]["paired_RR_tuned_minus_base"]
                        for k in ("all", "en", "he")})
        primary.update({f"known-item chat default hybrid 50/50 R@{CHAT_TOP} {k}":
                        chat[k][f"paired_R@{CHAT_TOP}_tuned_minus_base"] for k in ("all", "en", "he")})
    primary.update({f"subject {g} AP": sj[g]["AP"]["paired_tuned_minus_base"]
                    for g in ("focus_all_sources", "focus_probe_llm_t2 (paraphrase)", "named_subjects_name_queries")})
    harm = {k: v for k, v in primary.items() if v["ci_high"] < 0}
    watch = {k: v for k, v in primary.items() if v["delta"] < 0 and v["ci_high"] >= 0}
    gains = {k: v for k, v in primary.items() if v["ci_low"] > 0}
    return {"verdict": "safe_to_reembed_bibliography_with_tuned" if not harm else "keep_base_for_bibliography",
            "rule": "harm = a primary paired 95% CI entirely below 0; watch = negative point estimate, CI includes 0",
            "harm": harm, "watch": watch, "gains": gains, "keep_base_would_require": KEEP_BASE_REQUIRES}


def print_summary(rep: dict) -> None:
    """Print the headline numbers.

    :param rep: Report.
    """
    def ci(p: dict) -> str:
        return f"{p['delta']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] p(t>b)={p['p_b_better']:.2f}"

    for sec in ("dense_strict", "dense_lenient_pm1_page", "prod_hybrid_rrf_60_40_strict", "prod_hybrid_rrf_30_70_strict",
                "chat_default_hybrid_rrf_50_50_cand40_strict", "chat_kw70_hybrid_rrf_30_70_cand40_strict"):
        for k, blk in rep["known_item"].get(sec, {}).items():
            if not isinstance(blk, dict):
                continue
            print(f"KI {sec:30s} {k:34s} n={blk['n_queries']:3d} MRR {blk['base']['MRR']:.4f} -> "
                  f"{blk['tuned']['MRR']:.4f} {ci(blk['paired_RR_tuned_minus_base'])} | R@{CHAT_TOP} "
                  f"{blk['base'][f'R@{CHAT_TOP}']:.3f} -> {blk['tuned'][f'R@{CHAT_TOP}']:.3f} | R@10 "
                  f"{blk['base']['R@10']:.3f} -> {blk['tuned']['R@10']:.3f}")
    for k, blk in rep["subjects"].items():
        if isinstance(blk, dict) and "AP" in blk:
            print(f"SUBJ {k:40s} subj={blk['n_subjects']:3d} q={blk['n_queries']:4d} AP {blk['AP']['base']} -> "
                  f"{blk['AP']['tuned']} {ci(blk['AP']['paired_tuned_minus_base'])}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--n-stored", type=int, default=48)
    p = sub.add_parser("sample")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--n", type=int, default=KI_N_PAGES)
    p = sub.add_parser("show")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--chars", type=int, default=3500)
    p = sub.add_parser("queries")
    p.add_argument("--tsv", required=True)
    sub.add_parser("label")
    p = sub.add_parser("lexical")
    p.add_argument("--sleep", type=float, default=0.02, help="pause between read-only ES searches")
    p = sub.add_parser("embed")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--mps-max-tokens", type=int, default=2048, help="longer pages are embedded one by one on the CPU")
    p.add_argument("--shard-size", type=int, default=400)
    p.add_argument("--shard-tokens", type=int, default=300_000)
    p.add_argument("--pad-multiple", type=int, default=64)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--max-attn", type=int, default=1 << 22, help="max batch * padded_len^2")
    p.add_argument("--soft-max-gb", type=float, default=0.0, help=f"exit {RESTART_RC} between shards above this")
    p.add_argument("--cpu-threads", type=int, default=6)
    p.add_argument("--canary-min-cos", type=float, default=0.999)
    p = sub.add_parser("score")
    p.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()
    {"fetch": fetch, "sample": sample, "show": show, "queries": build_queries, "label": label, "lexical": lexical,
     "embed": embed, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
