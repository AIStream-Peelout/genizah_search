"""Pilot training pairs for the text embedder, built ONLY from corpus data we already have.

The point of the pilot is to measure how far corpus-internal supervision moves the
topic/concept metrics; it is deliberately not the production data recipe.

Split: KTIV-framed records are split 80/20 by a stable hash of ``doc_id``; every
human-verified Sukkot seed record is forced into the test side. Only train-side
records produce pairs; the test ids are written out so eval can exclude train docs.

Pair families (anchor -> positive):

* ``label``   — catalogue identification phrased as a query (KTIV frame "Talmud Bavli
  Sukkah", the Hebrew KTIV title, and for topic-mapped frames a topic phrase such as
  "laws of Sukkot") -> the record's full embedded text.
* ``content`` — a short span of the record's own transcription (Hebrew/Aramaic content
  words, e.g. a line about a stolen lulav) -> the record's *label-only* text (description +
  alternate titles, transcription removed), so the model must learn content->topic
  rather than copy strings.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from embed_utils import AUDIT_ROOT

TOPIC_PHRASES = {
    "sukkot": ["Sukkot", "laws of the festival of Sukkot", "הלכות סוכות", "Feast of Tabernacles"],
    "pesach": ["Passover", "laws of Pesach", "הלכות פסח"],
    "yom_kippur": ["Yom Kippur", "Day of Atonement", "יום הכפורים"],
    "rosh_hashanah": ["Rosh Hashanah", "laws of the New Year", "ראש השנה"],
    "shabbat": ["Sabbath laws", "Shabbat", "הלכות שבת"],
    "purim": ["Purim", "Scroll of Esther", "מגילת אסתר"],
    "marriage": ["marriage law", "betrothal and marriage", "הלכות אישות"],
    "divorce": ["divorce law", "bill of divorce", "הלכות גירושין"],
    "kashrut": ["ritual slaughter and dietary law", "kashrut", "הלכות שחיטה"],
    "niddah": ["laws of menstrual purity", "niddah", "הלכות נדה"],
    "tefillin_mezuzah": ["tefillin and mezuzah", "הלכות תפילין"],
    "mourning": ["mourning rites", "הלכות אבלות"],
}
GENERIC_TITLES = {"תחום לא מזוהה", "מכתבים", "קטע גניזה", "not define"}


def is_test(doc_id: str) -> bool:
    """Stable 20% test split.

    :param doc_id: Record id.
    :returns: True if the record belongs to the test side.
    :rtype: bool
    """
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 5 == 0


def split_text(text: str) -> tuple:
    """Split an embedded text representation into (label part, transcription part).

    :param text: ``create_text_representation`` output.
    :returns: (text without Transcription/Translation lines, transcription text).
    :rtype: tuple
    """
    label, trans = [], []
    for line in text.split("\n"):
        (trans if line.startswith(("Transcription:", "Translation:")) else label).append(line)
    return "\n".join(label), "\n".join(trans)


def frame_query(frame: str) -> str:
    """Turn a KTIV frame into a short label query.

    :param frame: e.g. ``"[Talmud Bavli]: Sukkah 29 a – 30 b"``.
    :returns: e.g. ``"Talmud Bavli Sukkah"``.
    :rtype: str
    """
    m = re.match(r"\[([^\]]+)\]:\s*([^\d@\[]+)", frame)
    if m:
        return f"{m.group(1).split(',')[-1].strip()} {m.group(2).strip(' ,;:')}".strip()
    return re.sub(r"[@\[\]]", "", frame).strip()[:80]


def content_spans(trans: str, rng: random.Random, n: int) -> list:
    """Sample short Hebrew-script spans (6-12 words) from transcription text.

    :param trans: Transcription text.
    :param rng: RNG.
    :param n: Spans wanted.
    :returns: List of spans.
    :rtype: list
    """
    words = [w for w in re.findall(r"[֐-׿'\"]+", trans) if len(w) > 1]
    if len(words) < 12:
        return []
    out = []
    for _ in range(n):
        size = rng.randint(6, 12)
        start = rng.randint(0, len(words) - size)
        out.append(" ".join(words[start:start + size]))
    return out


def main() -> None:
    """Write pairs + split files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(AUDIT_ROOT / "corpus_v1.jsonl"))
    parser.add_argument("--out-dir", default=str(AUDIT_ROOT / "pilot_text_v1"))
    args = parser.parse_args()
    rng = random.Random(13)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs, train_ids, test_ids, stats = [], [], [], Counter()
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if not row["framed"]:
                continue
            if row.get("sukkot_gold") or is_test(row["doc_id"]):
                test_ids.append(row["doc_id"])
                continue
            train_ids.append(row["doc_id"])
            label_text, trans = split_text(row["text"])
            queries = {frame_query(f) for f in row["frames"] if frame_query(f)}
            title = (row.get("description") or "").strip()
            if title and title not in GENERIC_TITLES and len(title) < 120:
                queries.add(title)
            for topic in row["topics"]:
                queries.add(rng.choice(TOPIC_PHRASES.get(topic, [topic])))
            for q in sorted(queries)[:4]:
                pairs.append({"anchor": q, "positive": row["text"], "family": "label", "doc_id": row["doc_id"]})
                stats["label"] += 1
            for span in content_spans(trans, rng, 2):
                pairs.append({"anchor": span, "positive": label_text, "family": "content", "doc_id": row["doc_id"]})
                stats["content"] += 1
    rng.shuffle(pairs)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "train_ids.json").write_text(json.dumps(train_ids))
    (out / "test_ids.json").write_text(json.dumps(test_ids))
    print(json.dumps({"pairs": len(pairs), **stats, "train_docs": len(train_ids), "test_docs": len(test_ids)}))


if __name__ == "__main__":
    main()
