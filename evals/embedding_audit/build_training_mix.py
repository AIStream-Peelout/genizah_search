"""Assemble the fine-tuning mixture (anchor → positive pairs, family-tagged) with strict split hygiene.

Families and sources:

* ``synthetic_<type>``  — LLM-written queries for TRAIN-pool records → the record's production text: English
  implicit/topical/specific + Hebrew keyword (``out``), and Hebrew implicit/topical + transliterated (``out_he``).
  Eval-pool records never enter training.
* ``label`` / ``content`` — corpus pairs from the LoRA pilot builder (frames/titles → record; Hebrew
  transcription span → label-only text), train split only.
* ``scholar_page2doc`` / ``scholar_doc2page`` — scholarship page ↔ fragment record (``scholarship_pairs.jsonl``).
* ``sefaria_*`` — concept-bridge pairs from ``sefaria/derived/pairs.jsonl``.

Split rule: a record is held out if it is on the 20 % test side of the stable hash, in the Sukkot seed, or in any
eval pool; held-out records are never anchors or positives. The 130 concept-probe queries (T2 eval) are never
anchors either; truncated catalogue titles are cleaned and one generic title anchors at most ``LABEL_ANCHOR_CAP``
records. Sefaria pairs are opt-in (``--with-sefaria``). Every pair is tagged with its family so mixture
ablations are one filter away. Writes ``train_mix_<name>.jsonl`` and a stats JSON.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from build_query_batches import is_test
from build_train_pairs import TOPIC_PHRASES, content_spans, frame_query, split_text  # noqa: F401
from embed_utils import AUDIT_ROOT

CAPS = {  # max pairs per family (keeps one source from dominating); None = all
    "synthetic_implicit": None, "synthetic_topical": None, "synthetic_hebrew": None, "synthetic_specific": None,
    "synthetic_he_implicit": None, "synthetic_he_topical": None, "synthetic_translit": None,
    "label": 15000, "content": None, "scholar_page2doc": None, "scholar_doc2page": None,
    "sefaria_title": None, "sefaria_topic": 15000, "sefaria_parent": 10000, "sefaria_xling": 10000,
}
LABEL_ANCHOR_CAP = 25  # one generic catalogue title ("Bible Genesis") may anchor at most this many records
PROBE = Path(__file__).parent / "concept_probe_v1.json"


def norm_query(text: str) -> str:
    """Normalise a query for leak checks (case, quotes/geresh, punctuation, whitespace).

    :param text: Query text.
    :returns: Normalised key.
    :rtype: str
    """
    return " ".join(re.sub(r"[\"'״׳.,;:!?()\[\]\-–]", " ", text.lower()).split())


def clean_label(anchor: str) -> str:
    """Drop a truncated trailing ``[…`` from a catalogue-title query (``"מקרא [טקסט"`` -> ``"מקרא"``).

    :param anchor: Label query.
    :returns: Cleaned query (may be empty).
    :rtype: str
    """
    if anchor.count("[") > anchor.count("]"):
        anchor = anchor[: anchor.rfind("[")]
    return anchor.strip(" ;,:")


def main() -> None:
    """Build the mixture."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v1")
    parser.add_argument("--rounds", default="r1")
    parser.add_argument("--with-sefaria", action="store_true", help="include sefaria/derived/pairs.jsonl")
    args = parser.parse_args()
    rng = random.Random(42)
    corpus = {}
    for line in open(AUDIT_ROOT / "corpus_v1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        corpus[r["doc_id"]] = r
    held = {d for d, r in corpus.items() if is_test(d) or r.get("sukkot_gold")}
    # the 130 concept-probe queries are the T2 eval queries: none may appear as a training anchor
    probe = {norm_query(q) for s in json.loads(PROBE.read_text())["concepts"].values() for q in s["queries"]}
    pairs, dropped = [], Counter()

    def add(anchor, positive, family, **kw):
        if family == "label":
            anchor = clean_label(anchor or "")
        if not (anchor and positive and anchor.strip() and positive.strip()):
            return
        if norm_query(anchor) in probe:
            dropped["probe_query_leak"] += 1
            return
        pairs.append({"anchor": anchor.strip(), "positive": positive.strip(), "family": family, **kw})

    for rnd in args.rounds.split(","):
        base = AUDIT_ROOT / "synthetic_queries" / rnd
        pool = {}
        for f in (base / "batches").glob("*.jsonl"):
            for line in open(f, encoding="utf-8"):
                b = json.loads(line)
                pool[b["doc_id"]] = b["pool"]
                if b["pool"] == "eval":
                    held.add(b["doc_id"])
        for f in list((base / "out").glob("*.jsonl")) + list((base / "out_he").glob("*.jsonl")):
            for line in open(f, encoding="utf-8"):
                o = json.loads(line)
                d = o["doc_id"]
                if d in held or pool.get(d) != "train" or d not in corpus:
                    continue
                for q in o.get("queries") or []:
                    add(q.get("text"), corpus[d]["text"], f"synthetic_{q.get('type')}", doc_id=d, grade=q.get("grade"))
    pilot = AUDIT_ROOT / "pilot_text_v1" / "pairs.jsonl"
    for line in open(pilot, encoding="utf-8"):
        p = json.loads(line)
        if p["doc_id"] not in held:
            add(p["anchor"], p["positive"], p["family"], doc_id=p["doc_id"])
    for line in open(AUDIT_ROOT / "scholarship_pairs.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s["doc_id"] in held or s["doc_id"] not in corpus:
            continue
        add(s["page_text"][:1500], corpus[s["doc_id"]]["text"], "scholar_page2doc", doc_id=s["doc_id"], page_id=s["page_id"])
        add(corpus[s["doc_id"]]["text"][:1500], s["page_text"], "scholar_doc2page", doc_id=s["doc_id"], page_id=s["page_id"])
    sef = AUDIT_ROOT / "sefaria" / "derived" / "pairs.jsonl"
    if args.with_sefaria and sef.exists():
        for line in open(sef, encoding="utf-8"):
            s = json.loads(line)
            add(s["anchor"], s["positive"], s["family"], ref=s.get("ref"))
    # cap families and repeated label anchors, dedupe exact (anchor, positive)
    rng.shuffle(pairs)
    seen, kept, per, per_label = set(), [], Counter(), Counter()
    for p in pairs:
        key = (p["anchor"], p["positive"][:200])
        cap = CAPS.get(p["family"])
        if key in seen or (cap is not None and per[p["family"]] >= cap):
            continue
        if p["family"] == "label" and per_label[p["anchor"]] >= LABEL_ANCHOR_CAP:
            dropped["label_anchor_cap"] += 1
            continue
        seen.add(key)
        per[p["family"]] += 1
        per_label[p["anchor"]] += p["family"] == "label"
        kept.append(p)
    out = AUDIT_ROOT / f"train_mix_{args.name}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for p in kept:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats = {"pairs": len(kept), "held_out_records": len(held), "by_family": dict(per.most_common()),
             "dropped": dict(dropped)}
    (AUDIT_ROOT / f"train_mix_{args.name}.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
