"""Build the offline audit corpus: production embedding text + silver topic labels.

Streams the sibling repo's ``merged_shelfmarks.jsonl`` and, for every record,
reproduces exactly the text the fragment index embeds
(``GenizahDocument.from_merged_format(...).create_text_representation()``),
then attaches labels we can evaluate retrieval against without touching ES:

* ``topics`` — festival / halakhic topics derived from KTIV scholarly "frames"
  (canonical identifications from the Sussmann/Lieberman/Zulay catalogues, e.g.
  ``[Talmud Bavli]: Sukkah 29a`` or ``(Hoshanot)``), plus the human-verified
  Sukkot seed (``sukkot_seed.json``) and the KTIV Sukkot scrape queues.
* ``framed`` — the record carries at least one KTIV frame (a fully identified text).
* ``domain`` / ``pgp_type`` / ``lang`` — genre and language, for the
  "what does the vector space organise by?" diagnostic.
* ``lexical_<topic>`` — whether the embedded text itself names the topic
  (separates real semantic matches from string matches).

Run from anywhere with the HDA venv::

    python build_corpus.py --out /Volumes/.../corpus_v1.jsonl
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
sys.path.insert(0, str(HDA))
from src.datasets.document_models.genizah_document import GenizahDocument  # noqa: E402

MERGED = HDA / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
KTIV_QUEUES = [Path.home() / "Documents/GitHub/ktiv-scraper/scraping_csv" / n
               for n in ("ktiv_queue_sukkot.csv", "ktiv_queue_sukkot_bodleian.csv")]

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
    """Yield merged records.

    :param path: JSONL path.
    :yields: Record dicts.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def main() -> None:
    """Build and write the corpus JSONL."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", required=True, help="sukkot_seed.json path")
    args = parser.parse_args()

    sukkot = load_sukkot_ids(Path(args.seed))
    stats: Counter = Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for record in iter_records(MERGED):
            doc = GenizahDocument.from_merged_format(record)
            text = doc.create_text_representation()
            frames, domains = ktiv_frames_and_domains(record)
            frame_blob = " || ".join(frames)
            topics = sorted(t for t, rx in TOPIC_FRAME_RE.items() if rx.search(frame_blob))
            cid = record["canonical_id"]
            labels = {"topics": topics, "framed": bool(frames)}
            if cid in sukkot["gold"]:
                labels["sukkot_gold"] = True
                labels["sukkot_gold_no_mention"] = cid in sukkot["gold_no_mention"]
                if "sukkot" not in topics:
                    labels["topics"] = sorted(topics + ["sukkot"])
            if cid in sukkot["weak"]:
                labels["sukkot_weak"] = True
            pgp_docs = ((record.get("sources") or {}).get("pgp") or {}).get("documents") or []
            row = {
                "doc_id": cid,
                "text": text,
                "description": (record.get("description") or "")[:400],
                "domain": domains[0] if domains else None,
                "pgp_type": pgp_docs[0].get("type") if pgp_docs else None,
                "lang": getattr(doc, "language", None),
                "frames": frames[:6],
                "lexical": sorted(t for t, rx in TOPIC_NAME_RE.items() if rx.search(text)),
                **labels,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["docs"] += 1
            stats["framed"] += bool(frames)
            for t in row["topics"]:
                stats[f"topic:{t}"] += 1
                stats[f"topic:{t}:no_lexical"] += t not in row["lexical"]
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
