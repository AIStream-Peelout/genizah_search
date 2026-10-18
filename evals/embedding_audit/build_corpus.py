"""Build the offline audit corpus: production embedding text + silver topic labels.

Streams a frozen ``merged_shelfmarks.jsonl`` and, for every record, reproduces exactly the text the fragment index
embeds as ``v8`` (``GenizahDocument.from_merged_format(...).create_text_representation()``, imported from a
historical-document-analysis tree, ``--hda-root``), then attaches labels we can evaluate retrieval against without
touching ES:

* ``topics`` — festival / halakhic topics derived from KTIV scholarly "frames"
  (canonical identifications from the Sussmann/Lieberman/Zulay catalogues, e.g.
  ``[Talmud Bavli]: Sukkah 29a`` or ``(Hoshanot)``), plus the human-verified
  Sukkot seed (``sukkot_seed.json``) and the KTIV Sukkot scrape queues.
* ``framed`` — the record carries at least one KTIV frame (a fully identified text).
* ``domain`` / ``pgp_type`` / ``lang`` / ``sources`` — genre, language and source groups, for the
  "what does the vector space organise by?" diagnostic and the population cuts.
* ``lexical_<topic>`` — whether the embedded text itself names the topic
  (separates real semantic matches from string matches).

When the HDA copy has the scholarly-frames render switch (``create_text_representation(include_scholarly_frames=...)``,
plan B1a), ``text`` lists the KTIV scholarly frames in its ``Alternate titles:`` line and ``text_no_frames`` holds the
rendering without them wherever the two differ, so one corpus serves both ``semantic_v4`` and ``semantic_v4_nf``
(``semantic_text.py --scholarly-frames on|off``). ``--no-scholarly-frames`` writes only the rendering without frames.

The text depends on the HDA document model, which changes in place (2026-10-09: 28,321 texts changed under one
label), so the run records what it was built from in ``<out>.stats.json`` ``provenance``: the HDA commit (or
``working-copy-dirty``), a sha256 over every HDA source file the import loaded, and the merged file's sha256. A dirty
HDA tree is refused unless ``--allow-dirty-hda``. A frozen tree extracted with ``git archive`` has no ``.git``:
declare its commit with ``--hda-commit`` and every loaded file is compared with ``git show <commit>:<path>`` in
``--hda-git-dir`` (read-only).

Run with the HDA venv (it needs pydantic; no torch, no models)::

    $HDA/.venv/bin/python -B build_corpus.py --hda-root $SCR/hda_$HDA_SHA --hda-commit $HDA_SHA \\
        --merged $V4/merged_shelfmarks_20261008.jsonl --expect-merged-sha256 da28... \\
        --seed $AR/sukkot_seed_feature_branch.json --out $V4/corpus_v1.jsonl
"""

import argparse
import csv
import hashlib
import inspect
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
MERGED_REL = Path("src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl")
MERGED = HDA / MERGED_REL
KTIV_QUEUES = [Path.home() / "Documents/GitHub/ktiv-scraper/scraping_csv" / n
               for n in ("ktiv_queue_sukkot.csv", "ktiv_queue_sukkot_bodleian.csv")]
# keyword argument of GenizahDocument.create_text_representation that leaves the KTIV scholarly frames out of the
# Alternate titles line (plan B1a); absent from HDA copies that predate the frames attribute
FRAMES_RENDER_KWARG = "include_scholarly_frames"
# corpus field with the rendering without frames, read by semantic_text.py --scholarly-frames off
TEXT_NO_FRAMES_FIELD = "text_no_frames"
SOURCE_KEYS = ("pgp", "fjp", "ktiv")

# KTIV frame -> topic. Strict on purpose: only frames whose subject is unambiguous.
TOPIC_FRAME_RULES: Dict[str, str] = {
    "sukkot": r"\]:\s*Sukkah\b|Hoshanot|Shofar ve-Sukkah|\(Sukkot\)|Lulav",
    "pesach": r"\]:\s*Pesahim\b|Hametz u-Mat|Haggadah|\(Pesach\)|\(Passover\)",
    "yom_kippur": r"\]:\s*Yoma\b|Shevitat Asor|Avodat Yom|Kippur",
    "rosh_hashanah": r"\]:\s*Rosh ha-Shanah|Shofar ve-Sukkah",
    "shabbat": r"\]:\s*Shabbat\b|\]:\s*Eruvin\b",
    "purim": r"\]:\s*Megillah\b|Megillah va-Hanukkah|\[Bible\]:\s*Esther|\(Purim\)",
    "hanukkah": r"Megillah va-Hanukkah|\(Hanukkah\)",
    "marriage": r"\]:\s*Ketubbot\b|\]:\s*Qiddushin\b|\]:\s*Ishut\b",
    "divorce": r"\]:\s*Gittin\b|\]:\s*Gerushin\b",
    "kashrut": r"\]:\s*Hullin\b|\]:\s*Shehitah\b|Ma'akhalot Asurot",
    "niddah": r"\]:\s*Niddah\b|Mikva'ot|Issurei Bi'ah",
    "tefillin_mezuzah": r"Tefillin Mezuzah|\]:\s*Tefillin\b",
    "mourning": r"\]:\s*Mo'ed Qatan\b|\]:\s*Evel\b|Semahot",
}
TOPIC_FRAME_RE = {t: re.compile(p, re.I) for t, p in TOPIC_FRAME_RULES.items()}

# Does the embedded text itself name the topic? (lexical-gap flag)
TOPIC_NAME_RE = {
    "sukkot": re.compile(r"sukk|succ|tabernacle|booths|hosha|lulav|etrog|סוכ|הושענ|לולב|אתרוג|ערבה", re.I),
    "pesach": re.compile(r"pesa|passover|haggad|hamet|matz|פסח|הגדה|חמץ|מצה", re.I),
    "yom_kippur": re.compile(r"kippur|atonement|yoma|כפור|כיפור|יומא", re.I),
    "rosh_hashanah": re.compile(r"rosh ha|new year|shofar|ראש השנה|שופר", re.I),
    "shabbat": re.compile(r"shabbat|sabbath|eruv|שבת|עירוב|ערוב", re.I),
    "purim": re.compile(r"purim|megill|esther|פורים|מגילה|אסתר", re.I),
    "hanukkah": re.compile(r"hanuk|chanuk|חנוכה", re.I),
    "marriage": re.compile(r"ketub|marriage|betroth|qiddush|kiddush|ishut|כתוב|קידוש|נישוא|אישות", re.I),
    "divorce": re.compile(r"divorc|gittin|\bget\b|gerush|גט|גיטין|גירוש", re.I),
    "kashrut": re.compile(r"hullin|shehit|slaughter|kosher|kashr|חולין|שחיט|טרפ|טריפ", re.I),
    "niddah": re.compile(r"niddah|nidda|mikv|immersion|נדה|מקו", re.I),
    "tefillin_mezuzah": re.compile(r"tefill|mezuz|phylact|תפילין|מזוז", re.I),
    "mourning": re.compile(r"mourn|mo'ed qatan|evel|burial|אבל|מועד קטן", re.I),
}


def as_list(value) -> List[str]:
    """Normalise a scalar-or-list catalogue field to a list of strings.

    :param value: Field value.
    :returns: List of strings.
    :rtype: List[str]
    """
    if not value:
        return []
    return [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]


def ktiv_frames_and_domains(record: dict) -> tuple:
    """Extract KTIV scholarly frames and level-1 domains.

    :param record: Merged record.
    :returns: (frames, domains) lists.
    :rtype: tuple
    """
    ktiv = (record.get("sources") or {}).get("ktiv") or {}
    frames, domains = [], []
    for entry in ktiv.get("scholarly_entries") or []:
        wc = ((entry.get("subsections") or {}).get("writing_characteristics") or {})
        frames += as_list(wc.get("frame"))
        domains += [d.split(" # ")[0].strip() for d in as_list(wc.get("domain"))]
    return frames, domains


def load_sukkot_ids(seed_path: Path) -> Dict[str, Set[str]]:
    """Load gold and weak Sukkot ids.

    :param seed_path: Path to the Sukkot seed JSON (from origin/feature/sukkot-page).
    :returns: {"gold": ids, "gold_no_mention": ids, "weak": ids}.
    :rtype: Dict[str, Set[str]]
    """
    seed = json.loads(seed_path.read_text())
    gold, no_mention = set(), set()
    for frag in seed["fragments"]:
        ids = [frag["doc_id"]] + list(frag.get("old_ids") or [])
        gold.update(ids)
        mentions = frag.get("record_mentions_sukkot") or {}
        if not any(mentions.values()):
            no_mention.update(ids)
    weak = set()
    for queue in KTIV_QUEUES:
        if queue.exists():
            with open(queue, newline="") as fh:
                weak.update(row["canonical_id"] for row in csv.DictReader(fh))
    return {"gold": gold, "gold_no_mention": no_mention, "weak": weak}


def iter_records(path: Path) -> Iterable[dict]:
    """Yield merged records (streamed; the file is about 1 GB).

    :param path: JSONL path.
    :yields: Record dicts.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a (large) file.

    :param path: File path.
    :param chunk: Read size in bytes.
    :returns: Hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> Optional[bytes]:
    """Run a read-only git command in ``repo`` (``--no-optional-locks``: never rewrites the index).

    :param repo: Repository or work-tree directory.
    :param args: git arguments.
    :returns: stdout bytes, or ``None`` when git fails (not a repository, unknown object ...).
    :rtype: Optional[bytes]
    """
    proc = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, check=False)
    return proc.stdout if proc.returncode == 0 else None


def import_genizah_document(hda_root: Path) -> Any:
    """Import ``GenizahDocument`` from an HDA tree without writing bytecode into it.

    :param hda_root: HDA checkout or ``git archive`` extraction.
    :returns: The ``GenizahDocument`` class.
    :rtype: Any
    :raises SystemExit: When another ``src`` package is already imported (it would shadow the HDA one).
    """
    loaded = sys.modules.get("src")
    if loaded is not None and not str(getattr(loaded, "__file__", "") or "").startswith(str(hda_root)):
        raise SystemExit(f"a different 'src' package is already imported ({loaded}); run build_corpus.py directly")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(hda_root))
    from src.datasets.document_models.genizah_document import GenizahDocument  # noqa: E402

    return GenizahDocument


def loaded_hda_files(hda_root: Path) -> List[Path]:
    """Every source file of an imported module that lives under the HDA tree.

    :param hda_root: HDA tree.
    :returns: Paths relative to ``hda_root``, sorted.
    :rtype: List[Path]
    """
    root = hda_root.resolve()
    files = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().is_relative_to(root):
            files.add(Path(path).resolve().relative_to(root))
    return sorted(files)


def source_digest(hda_root: Path, files: List[Path]) -> str:
    """sha256 over the loaded HDA source files (path, NUL, bytes, NUL for each, in path order).

    :param hda_root: HDA tree.
    :param files: Relative paths (:func:`loaded_hda_files`).
    :returns: Hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    for rel in files:
        digest.update(str(rel).encode("utf-8") + b"\0" + (hda_root / rel).read_bytes() + b"\0")
    return digest.hexdigest()


def hda_tree_state(hda_root: Path, files: List[Path], declared_commit: Optional[str],
                   git_dir: Path) -> Dict[str, Any]:
    """Which HDA code the corpus text came from, and whether that code exists in git exactly.

    A work tree is dirty when a loaded file is modified / untracked or any tracked file under ``src/`` is modified.
    A tree without ``.git`` is clean only when ``declared_commit`` is given and every loaded file equals
    ``git show <commit>:<file>`` in ``git_dir``.

    :param hda_root: HDA tree the import used.
    :param files: Loaded files (relative paths).
    :param declared_commit: Commit a ``git archive`` tree was extracted from, if any.
    :param git_dir: Repository used to verify ``declared_commit``.
    :returns: ``{hda_commit, hda_head, hda_tree, hda_dirty, hda_dirty_files}``.
    :rtype: Dict[str, Any]
    """
    top = git(hda_root, "rev-parse", "--show-toplevel")
    if top is not None and Path(top.decode().strip()).resolve() == hda_root.resolve():
        head = (git(hda_root, "rev-parse", "HEAD") or b"").decode().strip()
        rels = [str(f) for f in files]
        loaded = (git(hda_root, "status", "--porcelain", "--untracked-files=all", "--", *rels) or b"").decode()
        src = (git(hda_root, "status", "--porcelain", "--untracked-files=no", "--", "src") or b"").decode()
        dirty_files = sorted({ln[3:] for ln in (loaded + src).splitlines() if ln.strip()})
        return {"hda_commit": "working-copy-dirty" if dirty_files else head, "hda_head": head, "hda_tree": "git",
                "hda_dirty": bool(dirty_files), "hda_dirty_files": dirty_files[:50]}
    if not declared_commit:
        return {"hda_commit": "working-copy-dirty", "hda_head": None, "hda_tree": "unversioned",
                "hda_dirty": True, "hda_dirty_files": ["<no .git and no --hda-commit>"]}
    mismatched = [str(rel) for rel in files
                  if git(git_dir, "show", f"{declared_commit}:{rel}") != (hda_root / rel).read_bytes()]
    return {"hda_commit": "working-copy-dirty" if mismatched else declared_commit, "hda_head": declared_commit,
            "hda_tree": "archive", "hda_dirty": bool(mismatched), "hda_dirty_files": mismatched[:50]}


def frames_render_kwargs(doc_cls: Any, scholarly_frames: bool) -> Tuple[Dict[str, bool], bool]:
    """Keyword arguments that switch the HDA scholarly frames on or off in the ``Alternate titles:`` line.

    :param doc_cls: ``GenizahDocument``.
    :param scholarly_frames: Whether the line is wanted.
    :returns: (kwargs for ``create_text_representation``, whether this HDA copy has the switch).
    :rtype: Tuple[Dict[str, bool], bool]
    :raises SystemExit: When frames must be off but the HDA copy cannot switch them.
    """
    has_flag = FRAMES_RENDER_KWARG in inspect.signature(doc_cls.create_text_representation).parameters
    if has_flag:
        return {FRAMES_RENDER_KWARG: scholarly_frames}, True
    if not scholarly_frames:
        raise SystemExit(f"this HDA copy's create_text_representation() has no {FRAMES_RENDER_KWARG!r} switch "
                         "(plan B1a): it cannot render a corpus without scholarly frames")
    return {}, False


def source_groups(record: dict) -> List[str]:
    """Which of PGP / FJP / KTIV contributed to a merged record.

    :param record: Merged record.
    :returns: Present source names in :data:`SOURCE_KEYS` order.
    :rtype: List[str]
    """
    sources = record.get("sources") or {}
    pgp_docs = (sources.get("pgp") or {}).get("documents") or []
    present = {"pgp": bool(pgp_docs), "fjp": bool(sources.get("fjp")), "ktiv": bool(sources.get("ktiv"))}
    return [k for k in SOURCE_KEYS if present[k]]


def corpus_row(record: dict, doc_cls: Any, render: Dict[str, bool], write_no_frames: bool,
               sukkot: Dict[str, Set[str]]) -> dict:
    """One corpus row: the record's v8 text (and, where it differs, its rendering without scholarly frames) plus labels.

    :param record: Merged record.
    :param doc_cls: ``GenizahDocument``.
    :param render: ``create_text_representation`` keyword arguments (:func:`frames_render_kwargs`).
    :param write_no_frames: Also render without frames into ``text_no_frames`` (only when it differs from ``text``),
        so one corpus serves both ``semantic_v4`` and ``semantic_v4_nf``.
    :param sukkot: :func:`load_sukkot_ids` output.
    :returns: Row dict.
    :rtype: dict
    """
    doc = doc_cls.from_merged_format(record)
    text = doc.create_text_representation(**render)
    extra: Dict[str, str] = {}
    if write_no_frames:
        no_frames = doc.create_text_representation(**{FRAMES_RENDER_KWARG: False})
        if no_frames != text:
            extra[TEXT_NO_FRAMES_FIELD] = no_frames
    frames, domains = ktiv_frames_and_domains(record)
    frame_blob = " || ".join(frames)
    topics = sorted(t for t, rx in TOPIC_FRAME_RE.items() if rx.search(frame_blob))
    cid = record["canonical_id"]
    labels: Dict[str, Any] = {"topics": topics, "framed": bool(frames)}
    if cid in sukkot["gold"]:
        labels["sukkot_gold"] = True
        labels["sukkot_gold_no_mention"] = cid in sukkot["gold_no_mention"]
        if "sukkot" not in topics:
            labels["topics"] = sorted(topics + ["sukkot"])
    if cid in sukkot["weak"]:
        labels["sukkot_weak"] = True
    pgp_docs = ((record.get("sources") or {}).get("pgp") or {}).get("documents") or []
    return {
        "doc_id": cid,
        "text": text,
        **extra,
        "description": (record.get("description") or "")[:400],
        "domain": domains[0] if domains else None,
        "pgp_type": pgp_docs[0].get("type") if pgp_docs else None,
        "lang": getattr(doc, "language", None),
        "sources": source_groups(record),
        "frames": frames[:6],
        "lexical": sorted(t for t, rx in TOPIC_NAME_RE.items() if rx.search(text)),
        **labels,
    }


def text_shape(row: dict, stats: Counter) -> None:
    """Count the line kinds of a v8 text (which HDA rendering produced the corpus).

    :param row: Corpus row.
    :param stats: Counter updated in place.
    """
    text = f"\n{row['text']}"
    group = "+".join(row["sources"]) or "other"
    stats[f"group:{group}"] += 1
    stats[f"group:{group}:language_line"] += "\nLanguage:" in text
    stats["records_with_text_no_frames"] += TEXT_NO_FRAMES_FIELD in row
    stats["records_with_subjects_line"] += "\nSubjects:" in text
    alt = next((ln for ln in text.split("\n") if ln.startswith("Alternate titles:")), "")
    stats["records_with_bracketed_alt_part"] += any(p.strip().startswith("[") for p in alt.split(";"))


def main() -> None:
    """Build and write the corpus JSONL and its ``.stats.json`` (stats + provenance)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", required=True, help="sukkot_seed.json path")
    parser.add_argument("--hda-root", default=str(HDA), help="HDA tree to import GenizahDocument from")
    parser.add_argument("--merged", default=str(MERGED), help="frozen merged_shelfmarks.jsonl")
    parser.add_argument("--expect-merged-sha256", default=None, help="refuse a merged file with another sha256")
    parser.add_argument("--hda-commit", default=None, help="commit a git-archive --hda-root was extracted from")
    parser.add_argument("--hda-git-dir", default=str(HDA), help="repository that verifies --hda-commit (read-only)")
    parser.add_argument("--allow-dirty-hda", action="store_true", help="build from uncommitted HDA code (recorded)")
    parser.add_argument("--no-scholarly-frames", action="store_true",
                        help="render the text without KTIV scholarly frames (needs the HDA switch); by default "
                             "both renderings are written (text, text_no_frames)")
    parser.add_argument("--limit", type=int, default=None, help="first N merged records only (smoke runs)")
    args = parser.parse_args()

    hda_root = Path(args.hda_root).resolve()
    merged = Path(args.merged)
    merged_sha = sha256_file(merged)
    if args.expect_merged_sha256 and merged_sha != args.expect_merged_sha256:
        raise SystemExit(f"{merged}: sha256 {merged_sha} is not the expected {args.expect_merged_sha256}")
    doc_cls = import_genizah_document(hda_root)
    files = loaded_hda_files(hda_root)
    state = hda_tree_state(hda_root, files, args.hda_commit, Path(args.hda_git_dir))
    if state["hda_dirty"] and not args.allow_dirty_hda:
        raise SystemExit(f"HDA tree {hda_root} is not a committed state ({state['hda_tree']}; e.g. "
                         f"{state['hda_dirty_files'][:5]}); commit / archive it or pass --allow-dirty-hda")
    render, has_flag = frames_render_kwargs(doc_cls, not args.no_scholarly_frames)
    provenance = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "hda_root": str(hda_root), **state,
        "hda_source_sha256": source_digest(hda_root, files), "hda_source_files": [str(f) for f in files],
        "hda_frames_render_flag": has_flag, "scholarly_frames": not args.no_scholarly_frames,
        "text_no_frames_written": has_flag and not args.no_scholarly_frames,
        "merged": str(merged), "merged_sha256": merged_sha, "seed": args.seed,
        "seed_sha256": sha256_file(Path(args.seed)), "limit": args.limit,
    }
    sukkot = load_sukkot_ids(Path(args.seed))
    stats: Counter = Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for i, record in enumerate(iter_records(merged)):
            if args.limit is not None and i >= args.limit:
                break
            row = corpus_row(record, doc_cls, render, provenance["text_no_frames_written"], sukkot)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["docs"] += 1
            stats["framed"] += row["framed"]
            text_shape(row, stats)
            for t in row["topics"]:
                stats[f"topic:{t}"] += 1
                stats[f"topic:{t}:no_lexical"] += t not in row["lexical"]
    provenance["merged_records_read"] = stats["docs"]
    summary = dict(sorted(stats.items()))
    print(json.dumps({"provenance": {k: v for k, v in provenance.items() if k != "hda_source_files"}, **summary},
                     indent=1))
    Path(args.out).with_suffix(".stats.json").write_text(
        json.dumps({"provenance": provenance, "stats": summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
