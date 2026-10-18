"""Turn the pulled Sefaria topic graph (+ fetched texts) into concept-bridge training data.

Outputs (under ``<AUDIT_ROOT>/sefaria/derived/``):

* ``concept_graph.json`` — per topic: English/Hebrew names and alternate titles, and its links
  (``participates-in``, ``member-of``, ``is-a``, ``related-to``…). This is the narrower-term → topic lexicon
  (lulav → Sukkot) for query expansion and for generating parent-topic training pairs.
* ``curated_refs.jsonl`` — every non-sheet source ref attached to a topic that carries a Sefaria-written title
  or prompt: {topic, ref, title, prompt}. Titles/prompts are natural-language descriptions of the passage,
  i.e. query-like anchors already paired with a source.
* ``curated_refs.txt`` — unique refs to fetch with ``sefaria_pull.py texts`` (filtered to corpora that matter
  for the Genizah: Tanakh, Mishnah, Tosefta, Talmud, Midrash, halakhic codes, liturgy, Geonim).
* ``pairs.jsonl`` (only refs whose text has been fetched) — anchor → positive pairs, family-tagged:
  ``sefaria_title`` (Sefaria title → passage HE+EN), ``sefaria_topic`` (topic name/alt title → passage),
  ``sefaria_parent`` (a parent topic's name → a child topic's passage, e.g. "Sukkot" → a lulav passage),
  ``sefaria_xling`` (English passage → Hebrew passage).
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from embed_utils import AUDIT_ROOT
from sefaria_pull import safe_name

SEF = AUDIT_ROOT / "sefaria"
DER = SEF / "derived"
KEEP_CORPORA = re.compile(
    r"^(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|I |II |Isaiah|Jeremiah|Ezekiel|Hosea|Joel|Amos|"
    r"Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Psalms|Proverbs|Job|Song of Songs|Ruth|"
    r"Lamentations|Ecclesiastes|Esther|Daniel|Ezra|Nehemiah|Chronicles|Mishnah |Tosefta |Jerusalem Talmud |"
    r"Mishneh Torah|Shulchan Arukh|Tur,|Sifra|Sifrei|Mekhilta|Bereshit Rabbah|Shemot Rabbah|Vayikra Rabbah|"
    r"Bamidbar Rabbah|Devarim Rabbah|Eichah Rabbah|Esther Rabbah|Ruth Rabbah|Kohelet Rabbah|Shir HaShirim Rabbah|"
    r"Pesikta|Tanchuma|Midrash Tehillim|Pirkei DeRabbi Eliezer|Siddur|Machzor|Seder Rav Amram|Halakhot Gedolot|"
    r"Sheiltot|Saadia|Rif |Berakhot|Shabbat|Eruvin|Pesachim|Shekalim|Yoma|Sukkah|Beitzah|Rosh Hashanah|Taanit|"
    r"Megillah|Moed Katan|Chagigah|Yevamot|Ketubot|Nedarim|Nazir|Sotah|Gittin|Kiddushin|Bava Kamma|Bava Metzia|"
    r"Bava Batra|Sanhedrin|Makkot|Shevuot|Avodah Zarah|Horayot|Zevachim|Menachot|Chullin|Bekhorot|Arakhin|Temurah|"
    r"Keritot|Meilah|Tamid|Niddah)")
LINK_UP = {"participates-in", "member-of", "is-a", "temporally-contained-in"}
PER_TOPIC = 10  # refs per topic to fetch: every Sefaria-titled one, then the topic's top-ranked others


def load_topics() -> dict:
    """Load all pulled topic JSONs keyed by slug.

    :returns: slug -> topic dict.
    :rtype: dict
    """
    out = {}
    for f in (SEF / "topics").glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("slug"):
            out[d["slug"]] = d
    return out


def names(t: dict) -> list:
    """All English/Hebrew titles of a topic.

    :param t: Topic dict.
    :returns: Unique titles.
    :rtype: list
    """
    seen, res = set(), []
    for x in t.get("titles") or []:
        txt = (x.get("text") or "").strip()
        if txt and txt.lower() not in seen:
            seen.add(txt.lower())
            res.append(txt)
    return res


def passage_text(ref: str) -> tuple:
    """Hebrew and English text of a fetched ref (empty strings if absent).

    :param ref: Sefaria ref.
    :returns: (he, en).
    :rtype: tuple
    """
    f = SEF / "texts" / f"{safe_name(ref)}.json"
    if not f.exists():
        return "", ""
    d = json.loads(f.read_text() or "{}")
    he = en = ""
    for v in d.get("versions") or []:
        txt = v.get("text")
        flat = " ".join(_flatten(txt)) if txt else ""
        flat = re.sub(r"<[^>]+>", "", flat).strip()
        if v.get("language") == "he" and not he:
            he = flat
        elif v.get("language") == "en" and not en:
            en = flat
    return he[:1500], en[:1500]


def _flatten(x):
    """Flatten nested text arrays.

    :param x: str or nested list.
    :yields: strings.
    """
    if isinstance(x, str):
        yield x
    elif isinstance(x, list):
        for y in x:
            yield from _flatten(y)


def main() -> None:
    """Build graph, curated refs and (where texts exist) pairs."""
    DER.mkdir(parents=True, exist_ok=True)
    topics = load_topics()
    graph = {}
    for slug, t in topics.items():
        links = {k: [l.get("topic") for l in (v.get("links") or [])] for k, v in (t.get("links") or {}).items()
                 if "sheets" not in k}
        graph[slug] = {"en": (t.get("primaryTitle") or {}).get("en"), "he": (t.get("primaryTitle") or {}).get("he"),
                       "titles": names(t), "links": links}
    (DER / "concept_graph.json").write_text(json.dumps(graph, ensure_ascii=False))
    curated, refs, stats = [], Counter(), Counter()
    for slug, t in topics.items():
        cands = []
        for r in ((t.get("refs") or {}).get("about") or {}).get("refs") or []:
            if r.get("is_sheet"):
                continue
            desc = ((r.get("descriptions") or {}).get("en") or {})
            ref = r.get("ref")
            stats["refs"] += 1
            if ref and KEEP_CORPORA.match(ref):
                cands.append({"topic": slug, "ref": ref, "title": desc.get("title"), "prompt": desc.get("prompt")})
        # Sefaria lists a topic's sources best-first; keep every titled one, then fill to PER_TOPIC
        titled = [c for c in cands if c["title"]]
        rest = [c for c in cands if not c["title"]][: max(0, PER_TOPIC - len(titled))]
        for c in titled + rest:
            curated.append(c)
            refs[c["ref"]] += 1
            stats["kept"] += 1
            stats["with_title"] += bool(c["title"])
    with open(DER / "curated_refs.jsonl", "w", encoding="utf-8") as fh:
        for c in curated:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (DER / "curated_refs.txt").write_text("\n".join(sorted(refs)) + "\n")
    # pairs for refs whose text is already fetched
    parents = defaultdict(set)
    for slug, g in graph.items():
        for kind, targets in g["links"].items():
            if kind in LINK_UP:
                parents[slug].update(x for x in targets if x in graph)
    n = Counter()
    with open(DER / "pairs.jsonl", "w", encoding="utf-8") as fh:
        def emit(anchor, positive, family, **kw):
            if anchor and positive:
                fh.write(json.dumps({"anchor": anchor, "positive": positive, "family": family, **kw}, ensure_ascii=False) + "\n")
                n[family] += 1
        for c in curated:
            he, en = passage_text(c["ref"])
            if not (he or en):
                continue
            passage = (he + "\n" + en).strip()
            emit(c["title"], passage, "sefaria_title", ref=c["ref"], topic=c["topic"])
            g = graph.get(c["topic"], {})
            for nm in g.get("titles", [])[:3]:
                emit(nm, passage, "sefaria_topic", ref=c["ref"], topic=c["topic"])
            for p in list(parents.get(c["topic"], []))[:2]:
                emit(graph[p]["en"], passage, "sefaria_parent", ref=c["ref"], topic=c["topic"], parent=p)
            if he and en:
                emit(en, he, "sefaria_xling", ref=c["ref"])
    print(json.dumps({"topics": len(topics), **stats, "unique_curated_refs": len(refs), "pairs": dict(n)}))


if __name__ == "__main__":
    main()
