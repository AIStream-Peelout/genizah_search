"""Build the gate_v1 eval dir (plan section E): json in, json out, stdlib only, no model.

Reads the gate spec written by the gate-spec session (``gate_v1_spec.json``: families, concepts with relevance rules
and queries, probes, thresholds), the semantic corpus (fields parsed from ``text_v3``, never from the embedded v4
text), the frozen merged file (people / places / languages / document types / titles per source), ``subjects_v1``
(typed weak labels, ``canon``, ``mentions``), the frozen eval dir (held-out records, known-item queries) and optional
extras, and writes ``gate_dir``::

    queries.jsonl       {qid, text, family, concept, form, blocked, lang, source, report_only}
    concepts.json       {cid: {family, label_ids, rel_mode, report_only, relevant, canon, unjudged, alt_only,
                         held_out_relevant, n_relevant_text, n_relevant_label, n_desc_mention, n_by_group,
                         too_small, audit}}        (eligible / pool records only)
    records.jsonl       {doc_id, source_group (KTIV|FJP|PGP|other), desc_lang (he|en|ar|mixed), len_bucket,
                         held_out, new_since_v9, has_alt_titles, text_hash, text_hash_v3, sources, canon_kind,
                         v9_trained, v9_trained_by, v3_text_changed, n_chars}
    probes.json         {rank_regressions, rank_regressions_dropped, title_position, rank_sweep, seed_present}
    thresholds.json     eval_gate.DEFAULT_THRESHOLDS deep-merged with the spec's thresholds ("proposed - Isaac
                        freezes"; its sha256 is printed)
    review_sample.csv   rule audit: 20 matched + 20 unmatched records per concept (text-only reading)
    build_stats.json    inputs (+ sha256), counts, size checks, warnings, unread threshold keys, audit status

Relevance per concept (label precedence first, then the audited text rule):

1. ``label_ids`` carried strong and not listed in the record's ``canon`` -> relevant;
2. label canon (weak_reason ``canon`` or listed in ``canon``) -> canon (a miss); ``rel_mode: "law"`` -> relevant;
3. typed weak labels (``title``, ``incidental``; a ``gold_override`` record whose label is canon) -> unjudged;
4. otherwise the ``rule`` over its ``fields`` -> relevant, unless ``canon_rule`` also matches (-> canon);
5. ``canon_rule`` alone -> canon; the rule matching only in other fields (e.g. transcription), or a label
   ``mention`` -> unjudged (out of the ranking).

Rule shape: ``{"any": [regex], "all": [regex], "none": [regex], "fields": [...], "flags": "i"}``; ``{HB}`` / ``{HE}``
expand to the Hebrew word start (up to two prefix letters) / word end of ``build_subjects.py``. Fields:
description / alt_titles / doc_type / language / people / places / title / transcription (folded ``text_v3`` lines
plus the merged record's own fields; see :func:`merged_fields`).

Run (plan step 6)::

    python3 build_gate_v1.py --eval-dir $V4/eval_final --semantic $V4/corpus_semantic.jsonl \\
        --subjects $V4/subjects_v1.jsonl --seed $GS/data/holiday_eval_seed.json \\
        --chat-sessions evals/results/rag_audit_20261007/chat_sessions.jsonl \\
        --v9-mix $AR/v3/train_mix_v3final.jsonl --out-dir $V4/eval_final/gate_v1
    python3 build_gate_v1.py --self-test [--work-dir DIR]
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
AUDIT_ROOT = Path(os.environ.get("AUDIT_ROOT", "/Volumes/home/studio_offload/genizah_search_embedding_audit"))
V4 = AUDIT_ROOT / "v4"
EVAL_GATE = HERE / "eval_gate.py"
DEFAULT_SPEC = HERE / "gate_v1_spec.json"
DEFAULT_SEED = HERE / "data" / "holiday_eval_seed.json"
DEFAULT_CHAT = REPO / "evals" / "results" / "rag_audit_20261007" / "chat_sessions.jsonl"

# Hebrew word start / end for the {HB} / {HE} rule placeholders (copied from build_subjects.py :61-62 so this file
# stays stdlib-only: build_subjects imports embed_utils, which imports numpy).
HB = r"(?<![\u05D0-\u05EA])[בהוכלמשד]{0,2}"
HE = r"(?![\u05D0-\u05EA])"

# Field labels of the semantic texts (text-pipeline audit, semantic_v4 FIELD_PREFIXES): a line that starts with none
# of them continues the field before it (1,828 records have such continuation lines).
FIELD_PREFIXES = ("Document ID:", "Shelf Mark:", "Description:", "Alternate titles:", "Subjects:", "Title:",
                  "Summary:", "Language:", "Document Type:", "Date:", "Transcription:", "Translation:",
                  "Related People:", "Related Places:", "Physical:", "Additional Info:")
TEXT_FIELDS = {"description": ("Description",), "alt_titles": ("Alternate titles",), "doc_type": ("Document Type",),
               "language": ("Language",), "people": ("Related People",), "places": ("Related Places",),
               "title": ("Title",), "transcription": ("Transcription", "Translation")}
RULE_FIELDS = tuple(TEXT_FIELDS)
DEFAULT_RULE_FIELDS = ("description", "alt_titles")
DEFAULT_CANON_FIELDS = ("description", "alt_titles", "title")
FAMILIES = tuple(f"G{i}" for i in range(1, 14))
REPORT_ONLY_FAMILIES = ("G12", "G13")
TEI_LANG = {"he": "Hebrew", "heb": "Hebrew", "arc": "Aramaic", "ar": "Arabic", "ara": "Arabic", "jrb": "Judaeo-Arabic",
            "jpr": "Judaeo-Persian", "lad": "Ladino", "yi": "Yiddish", "el": "Greek", "grc": "Greek", "la": "Latin",
            "syr": "Syriac", "fa": "Persian", "es": "Spanish", "de": "German", "it": "Italian", "en": "English",
            "fr": "French", "cop": "Coptic", "tr": "Turkish"}
PGP_PLACE_KEYS = ("origin", "destination", "location", "mentioned", "possibly_mentioned")
# canon kinds by subject-id prefix (slugs of build_subjects TRACTATE_WORKS / LAW_CODE_WORKS / BOOK_WORKS)
CANON_KIND_PREFIXES = (
    ("tractate", ("work:talmud-bavli", "work:talmud-yerushalmi", "work:mishnah", "work:tosefta", "work:hilkhot-ha-rif",
                  "work:rabbenu-hananel", "domain:talmud-bavli", "domain:talmud-yerushalmi", "domain:mishnah",
                  "domain:tosefta")),
    ("law_code", ("work:mishneh-torah", "work:halakhot-gedolot", "work:sefer-ha-mitzvot")),
    ("bible", ("work:bible", "work:targum", "work:aramaic-targumim", "work:arabic-tafsir", "work:tafsir-saadya",
               "work:haftarot", "work:biblical-exegesis", "work:rashi-on-the-bible", "work:massorah",
               "work:yefet-b-eli", "domain:bible")),
)
LEN_BUCKETS = ((100, "lt100"), (300, "100-300"), (1000, "300-1k"))
CHAT_TAGS = ("organic", "eval", "eval_question")
CHAT_SEARCH_TYPES = ("bibliography_hybrid", "bibliography_semantic", "primary_hybrid", "primary_semantic",
                     "fragment_hybrid", "fragment_semantic", "primary_shelfmark")
DESCRIPTIVE_FORMS = ("descriptive", "nuanced", "probe", "seed_descriptive")
REVIEW_N = 20


# ---------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield rows of a JSONL file (streamed).

    :param path: File path.
    :yields: Row dicts.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file (streamed).

    :param path: File path.
    :returns: Hex digest.
    :rtype: str
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5(text: str) -> str:
    """md5 hex of a text (the corpus ``text_hash`` rule).

    :param text: Text.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def seeded_key(*parts: str) -> str:
    """Deterministic sort key (md5 of the joined parts).

    :param parts: Key parts.
    :returns: Hex digest.
    :rtype: str
    """
    return hashlib.md5("\x1f".join(parts).encode("utf-8")).hexdigest()


def norm_text(text: str) -> str:
    """Case- and space-folded text (query de-duplication).

    :param text: Text.
    :returns: Normalised text.
    :rtype: str
    """
    return " ".join(text.casefold().split())


def query_lang(text: str) -> str:
    """``he`` when the text holds Hebrew letters, else ``latin`` (same rule as eval_gate.query_lang).

    :param text: Query text.
    :returns: Language tag.
    :rtype: str
    """
    return "he" if re.search(r"[\u05D0-\u05EA]", text) else "latin"


def as_list(value) -> List[str]:
    """A scalar-or-list catalogue value as a list of non-empty strings.

    :param value: Field value.
    :returns: Strings.
    :rtype: List[str]
    """
    if not value:
        return []
    vals = value if isinstance(value, list) else [value]
    return [v.strip() for v in vals if isinstance(v, str) and v.strip()]


def deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge (``over`` wins; non-dict values replaced).

    :param base: Defaults.
    :param over: Overrides.
    :returns: New merged dict.
    :rtype: dict
    """
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def unknown_keys(base: dict, over: dict, prefix: str = "") -> List[str]:
    """Keys of ``over`` that ``base`` does not have (spec threshold keys eval_gate.verdict never reads).

    :param base: Defaults.
    :param over: Overrides.
    :param prefix: Key path so far.
    :returns: Dotted key paths.
    :rtype: List[str]
    """
    out = []
    for k, v in over.items():
        path = f"{prefix}{k}"
        if k not in base:
            out.append(path)
        elif isinstance(v, dict) and isinstance(base[k], dict) and k not in ("min_queries", "primary_metric",
                                                                            "metrics", "delta"):
            out += unknown_keys(base[k], v, path + ".")
    return out


def default_thresholds(path: Path = EVAL_GATE) -> dict:
    """``DEFAULT_THRESHOLDS`` of ``eval_gate.py``, read with ``ast`` (no numpy import).

    :param path: ``eval_gate.py``.
    :returns: Thresholds dict.
    :rtype: dict
    :raises ValueError: When the assignment is missing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_THRESHOLDS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"DEFAULT_THRESHOLDS not found in {path}")


# ---------------------------------------------------------------------------------------------------------------
# record fields
# ---------------------------------------------------------------------------------------------------------------
def fold_field_lines(text: str) -> List[str]:
    """Lines of a semantic text, each line that starts no field label appended (one space) to the line before it.

    Same rule as ``semantic_text.fold_field_lines`` (text-pipeline audit); re-implemented to keep this file
    stdlib-only.

    :param text: semantic_v3 text.
    :returns: One line per field.
    :rtype: List[str]
    """
    out: List[str] = []
    for ln in text.split("\n"):
        if out and not ln.startswith(FIELD_PREFIXES):
            out[-1] = f"{out[-1]} {ln.strip()}"
        else:
            out.append(ln)
    return out


def text_labels(text: str) -> Dict[str, str]:
    """``Label -> value`` of the folded field lines (repeated labels joined with " ; ").

    :param text: semantic_v3 text.
    :returns: Label values.
    :rtype: Dict[str, str]
    """
    out: Dict[str, List[str]] = defaultdict(list)
    for ln in fold_field_lines(text):
        if ":" in ln and ln.startswith(FIELD_PREFIXES):
            label, value = ln.split(":", 1)
            if value.strip():
                out[label].append(value.strip())
    return {k: " ; ".join(v) for k, v in out.items()}


def merged_fields(record: dict) -> Dict[str, List[str]]:
    """People, places, languages, document types and titles of a merged record, per source.

    * KTIV: ``basic_catalog`` title / author / language / place_of_writing, ``full_catalog.additional_persons``.
    * FJP (list): language, other_languages, related_people[].name, related_places[].name.
    * PGP: ``documents[]`` type, languages_primary / _secondary, origin / destination / location / (possibly_)
      mentioned (places).
    * Bodleian TEI: titles, items[].title / author, languages (codes, also mapped to names), orig_place.

    :param record: Merged record.
    :returns: ``{people, places, language, doc_type, title, sources}`` lists.
    :rtype: Dict[str, List[str]]
    """
    out: Dict[str, List[str]] = {"people": [], "places": [], "language": [], "doc_type": [], "title": []}
    src = record.get("sources") or {}
    ktiv = src.get("ktiv") or {}
    if ktiv:
        bc = ktiv.get("basic_catalog") or {}
        out["title"] += as_list(bc.get("title"))
        out["people"] += as_list(bc.get("author")) + as_list((ktiv.get("full_catalog") or {}).get("additional_persons"))
        out["language"] += as_list(bc.get("language"))
        out["places"] += as_list(bc.get("place_of_writing"))
    for doc in src.get("fjp") or []:
        out["language"] += as_list(doc.get("language")) + as_list(doc.get("other_languages"))
        out["people"] += [p.get("name", "") for p in doc.get("related_people") or [] if isinstance(p, dict)]
        out["places"] += [p.get("name", "") for p in doc.get("related_places") or [] if isinstance(p, dict)]
    for doc in (src.get("pgp") or {}).get("documents") or []:
        out["doc_type"] += as_list(doc.get("type"))
        out["language"] += as_list(doc.get("languages_primary")) + as_list(doc.get("languages_secondary"))
        for key in PGP_PLACE_KEYS:
            out["places"] += as_list(doc.get(key))
    tei = (src.get("bodleian") or {}).get("tei") or {}
    if tei:
        out["title"] += as_list(tei.get("titles"))
        for item in tei.get("items") or []:
            out["title"] += as_list(item.get("title"))
            out["people"] += as_list(item.get("author"))
        for code in as_list(tei.get("languages")):
            out["language"] += [code] + ([TEI_LANG[code]] if code in TEI_LANG else [])
        out["places"] += as_list(tei.get("orig_place"))
    out = {k: list(dict.fromkeys(" ".join(v.split()) for v in vals if v and v.strip())) for k, vals in out.items()}
    out["sources"] = sorted(record.get("sources_present") or [s for s, v in src.items() if v])
    return out


def record_fields(text_v3: str, merged: Optional[Dict[str, List[str]]]) -> Dict[str, str]:
    """Rule fields of one record: folded ``text_v3`` field values plus the merged record's fields.

    :param text_v3: semantic_v3 text of the record.
    :param merged: :func:`merged_fields` output (None when the record is absent from the merged file).
    :returns: ``field -> text`` for every name in :data:`RULE_FIELDS`, plus ``__all__`` (every field joined).
    :rtype: Dict[str, str]
    """
    labels = text_labels(text_v3)
    out = {}
    for field, names in TEXT_FIELDS.items():
        parts = [labels[n] for n in names if labels.get(n)]
        if merged and field in merged:
            parts += [v for v in merged[field] if v not in parts]
        out[field] = "\n".join(parts)
    out["__all__"] = "\n".join(out[f] for f in RULE_FIELDS if out[f])
    return out


def source_group(sources: Iterable[str]) -> str:
    """Population source group: KTIV if KTIV catalogued it, else PGP (whose description wins the merge), else FJP
    (FJP-only), else other (Bodleian-only and the rest).

    :param sources: Merged ``sources_present``.
    :returns: ``KTIV`` / ``PGP`` / ``FJP`` / ``other``.
    :rtype: str
    """
    s = set(sources)
    if "ktiv" in s:
        return "KTIV"
    if "pgp" in s:
        return "PGP"
    if "bodleian" in s:
        return "other"
    return "FJP" if "fjp" in s else "other"


def script_lang(text: str) -> Optional[str]:
    """Script of a text: he / en (Latin) / ar when one script has >= 80 % of the letters, mixed otherwise.

    :param text: Text.
    :returns: Language tag, or None when the text has no letters.
    :rtype: Optional[str]
    """
    counts = {"he": len(re.findall(r"[\u05D0-\u05EA]", text)), "ar": len(re.findall(r"[\u0600-\u06FF]", text)),
              "en": len(re.findall(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]", text))}
    total = sum(counts.values())
    if not total:
        return None
    best = max(counts, key=counts.get)
    return best if counts[best] >= 0.8 * total else "mixed"


def len_bucket(n_chars: int) -> str:
    """Length bucket of the embedded text.

    :param n_chars: Characters.
    :returns: ``lt100`` / ``100-300`` / ``300-1k`` / ``ge1k``.
    :rtype: str
    """
    return next((name for lim, name in LEN_BUCKETS if n_chars < lim), "ge1k")


def canon_kind(subjects: Iterable[str]) -> Optional[str]:
    """Canon kind of a record from its subject ids (tractate > law_code > bible).

    :param subjects: Subject ids.
    :returns: Kind or None.
    :rtype: Optional[str]
    """
    subs = list(subjects)
    for kind, prefixes in CANON_KIND_PREFIXES:
        if any(s == p or s.startswith(p + "-") for s in subs for p in prefixes):
            return kind
    return None


# ---------------------------------------------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------------------------------------------
class Rule:
    """A compiled relevance rule ``{any, all, none, fields, flags}``."""

    FLAGS = {"i": re.I, "m": re.M, "s": re.S, "x": re.X}

    def __init__(self, spec: dict, default_fields: Tuple[str, ...], where: str) -> None:
        """Compile the rule.

        :param spec: Rule dict.
        :param default_fields: Fields when the rule names none.
        :param where: Location for error messages.
        :raises ValueError: On an unknown field or flag, or a rule with neither ``any`` nor ``all``.
        """
        flags = 0
        for ch in spec.get("flags", "i"):
            if ch not in self.FLAGS:
                raise ValueError(f"{where}: unknown regex flag {ch!r}")
            flags |= self.FLAGS[ch]
        self.fields = tuple(spec.get("fields") or default_fields)
        bad = [f for f in self.fields if f not in RULE_FIELDS]
        if bad:
            raise ValueError(f"{where}: unknown field(s) {bad}; allowed {list(RULE_FIELDS)}")
        comp = lambda key: [re.compile(p.replace("{HB}", HB).replace("{HE}", HE), flags) for p in spec.get(key) or []]
        self.any, self.all, self.none = comp("any"), comp("all"), comp("none")
        if not self.any and not self.all:
            raise ValueError(f"{where}: rule has neither 'any' nor 'all' patterns")

    def matches(self, text: str) -> bool:
        """Whether the rule matches a text.

        :param text: Text (fields joined).
        :returns: True on a match.
        :rtype: bool
        """
        if self.any and not any(r.search(text) for r in self.any):
            return False
        if not all(r.search(text) for r in self.all):
            return False
        return not any(r.search(text) for r in self.none)

    def quick(self, text: str) -> bool:
        """Cheap pre-filter: could the rule match anywhere in this text?

        :param text: All fields joined.
        :returns: False when the rule cannot match.
        :rtype: bool
        """
        return any(r.search(text) for r in self.any) if self.any else bool(self.all[0].search(text))

    def named(self, text: str) -> bool:
        """Whether one field names the concept (an ``any`` pattern, no ``none`` pattern).

        :param text: One field's text.
        :returns: True when named.
        :rtype: bool
        """
        return bool(text) and any(r.search(text) for r in (self.any or self.all)) and \
            not any(r.search(text) for r in self.none)

    def on(self, fields: Dict[str, str], names: Optional[Tuple[str, ...]] = None) -> bool:
        """Match over a record's fields.

        :param fields: :func:`record_fields` output.
        :param names: Fields to use (default: the rule's own).
        :returns: True on a match.
        :rtype: bool
        """
        return self.matches("\n".join(fields[f] for f in (names or self.fields) if fields[f]))


def load_spec(path: Path) -> dict:
    """Load and validate the gate spec (contract item 7).

    :param path: ``gate_v1_spec.json``.
    :returns: Spec dict.
    :rtype: dict
    :raises ValueError: On a structural problem (all problems listed).
    """
    spec = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if spec.get("version") != 1:
        errors.append(f"version must be 1, got {spec.get('version')!r}")
    if not isinstance(spec.get("concepts"), dict) or not spec["concepts"]:
        errors.append("'concepts' must be a non-empty object")
    for cid, c in (spec.get("concepts") or {}).items():
        if c.get("family") not in FAMILIES:
            errors.append(f"concept {cid}: family {c.get('family')!r} not in G1..G13")
        if not isinstance(c.get("queries", []), list):
            errors.append(f"concept {cid}: queries must be a list")
        for i, q in enumerate(c.get("queries") or []):
            if not isinstance(q, dict) or not str(q.get("text", "")).strip():
                errors.append(f"concept {cid}: query {i} has no text")
        if not c.get("report_only", c.get("family") in REPORT_ONLY_FAMILIES) and not c.get("rule") \
                and not c.get("label_ids"):
            errors.append(f"concept {cid}: neither rule nor label_ids (no relevance)")
        if c.get("rel_mode", "bare") not in ("bare", "law"):
            errors.append(f"concept {cid}: rel_mode must be 'bare' or 'law'")
    if errors:
        raise ValueError(f"{path}: invalid gate spec:\n  " + "\n  ".join(errors))
    return spec


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------
def load_corpus(path: Path) -> Tuple[Dict[str, dict], List[str], Counter]:
    """Pool (eligible) records of ``corpus_semantic.jsonl`` with their v3 text and hashes.

    :param path: Corpus file (``text`` = embedded text, ``text_v3`` = semantic_v3 text; v3-only corpora have no
        ``text_v3`` and use ``text``).
    :returns: (``doc_id -> {text, text_v3, text_hash, text_version, n_chars}`` for eligible rows, eligible ids in file
        order, counts).
    :rtype: Tuple[Dict[str, dict], List[str], Counter]
    """
    rows, order, n = {}, [], Counter()
    for r in iter_jsonl(path):
        n["rows"] += 1
        if not r.get("eligible"):
            continue
        n["eligible"] += 1
        v3 = r.get("text_v3") if r.get("text_v3") is not None else r["text"]
        n["has_text_v3"] += r.get("text_v3") is not None
        rows[r["doc_id"]] = {"text": r["text"], "text_v3": v3, "text_hash": r.get("text_hash") or md5(r["text"]),
                             "text_version": r.get("text_version") or "semantic_v3", "n_chars": len(r["text"])}
        order.append(r["doc_id"])
    n["text_versions"] = Counter(r["text_version"] for r in rows.values())
    return rows, order, n


def load_merged(path: Path, wanted: Set[str]) -> Dict[str, Dict[str, List[str]]]:
    """:func:`merged_fields` of the pool records (the 1 GB file is streamed).

    :param path: Frozen merged file.
    :param wanted: Pool ids.
    :returns: ``doc_id -> fields``.
    :rtype: Dict[str, Dict[str, List[str]]]
    :raises ValueError: When a pool id occurs twice.
    """
    out = {}
    for rec in iter_jsonl(path):
        cid = rec.get("canonical_id")
        if cid in wanted:
            if cid in out:
                raise ValueError(f"{cid} occurs twice in {path}")
            out[cid] = merged_fields(rec)
    return out


def load_subjects(path: Path, wanted: Set[str]) -> Dict[str, dict]:
    """``subjects_v1`` rows of the pool records (missing keys of older schemas -> empty).

    :param path: ``subjects_v1.jsonl``.
    :param wanted: Pool ids.
    :returns: ``doc_id -> {subjects, weak, weak_reason, canon, mentions, gold_override}``.
    :rtype: Dict[str, dict]
    """
    out = {}
    for r in iter_jsonl(path):
        if r["doc_id"] in wanted:
            out[r["doc_id"]] = {"subjects": set(r.get("subjects") or []), "weak": set(r.get("weak") or []),
                                "weak_reason": dict(r.get("weak_reason") or {}), "canon": set(r.get("canon") or []),
                                "mentions": set(r.get("mentions") or []),
                                "gold_override": bool(r.get("gold_override"))}
    return out


def load_v9_mix(path: Optional[Path]) -> Tuple[Set[str], Set[str], Counter]:
    """Record ids and positive-text hashes the v9 training mix used.

    Ids: ``doc_id`` (positive), ``anchor_doc_id`` (d2d anchors) and every ``D:<id>`` mask id. Mined negatives are not
    stored in the mix file (they were mined on Colab), so they are not covered.

    :param path: ``v3/train_mix_v3final.jsonl`` or None.
    :returns: (ids, positive text hashes, counts).
    :rtype: Tuple[Set[str], Set[str], Counter]
    """
    ids, hashes, n = set(), set(), Counter()
    if path is None:
        return ids, hashes, n
    for r in iter_jsonl(path):
        n["rows"] += 1
        for key in ("doc_id", "anchor_doc_id"):
            if r.get(key):
                ids.add(r[key])
        for m in list(r.get("mask_ids") or []) + list(r.get("anchor_mask_ids") or []):
            if m.startswith("D:"):
                ids.add(m[2:])
        if r.get("text_hash"):
            hashes.add(r["text_hash"])
    n["ids"], n["hashes"] = len(ids), len(hashes)
    return ids, hashes, n


def load_prev_corpus(path: Optional[Path]) -> Optional[Dict[str, str]]:
    """``doc_id -> text_hash`` of the corpus v9 was trained on (for ``new_since_v9`` / ``v3_text_changed``).

    :param path: Previous ``corpus_semantic.jsonl`` or None.
    :returns: Hash map or None.
    :rtype: Optional[Dict[str, str]]
    """
    if path is None:
        return None
    return {r["doc_id"]: r.get("text_hash") or md5(r["text"]) for r in iter_jsonl(path)}


def chat_queries(path: Path) -> List[dict]:
    """Real-user turns and router search queries of the RAG-audit chat sessions (G11 material).

    User turn = the ``LATEST USER MESSAGE`` of ``resolver_input`` when present, else ``question``; truncated log
    texts are skipped. Router queries come from ``router_plan`` (malformed plans are skipped and counted).

    :param path: ``chat_sessions.jsonl``.
    :returns: ``[{text, form ("user_turn" | "router" | "router_shelfmark"), source}]`` de-duplicated.
    :rtype: List[dict]
    """
    out, seen = [], set()

    def add(text: str, form: str, source: str) -> None:
        text = " ".join(text.split())
        key = norm_text(text)
        if text and "<Truncated" not in text and key not in seen:
            seen.add(key)
            out.append({"text": text, "form": form, "source": source})

    for r in iter_jsonl(path):
        if r.get("tag") not in CHAT_TAGS:
            continue
        ri = r.get("resolver_input") or ""
        turn = ri.split("LATEST USER MESSAGE:", 1)[1] if "LATEST USER MESSAGE:" in ri else (r.get("question") or "")
        add(turn, "user_turn", f"chat:{r.get('session_id')}:{r.get('tag')}")
        plans = r.get("router_plan")
        if not plans:
            continue
        try:  # the logged plans are JSON strings of JSON strings; some were cut mid-string in the logs
            actions = [a for p in json.loads(plans) for a in json.loads(p).get("actions", [])]
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
        for a in actions:
            if a.get("search_type") in CHAT_SEARCH_TYPES and a.get("query"):
                form = "router_shelfmark" if a["search_type"] == "primary_shelfmark" else "router"
                add(a["query"], form, f"chat:{r.get('session_id')}:{a['search_type']}")
    return out


def load_seed(path: Optional[Path]) -> Optional[dict]:
    """The handoff's ``holiday_eval_seed.json`` in a normalised shape (format not in the repo; aliases accepted).

    Accepted: query lists under ``queries`` / ``sets`` / ``items`` (items ``{text|query, festival|concept|set|topic,
    form|type|kind}``, or a dict ``set name -> [str | item]``); rank cases under ``rank_regressions`` /
    ``regressions`` / ``m2`` / ``rank_cases`` (``{query|text, doc_id|target|expected_doc_id, max_rank}``); label-swap
    pairs under ``label_swap_pairs`` / ``m1`` (kept for diagnosis).

    :param path: Seed path (None or absent -> None).
    :returns: ``{queries: [{text, festival, form}], rank_regressions: [...], label_swap_pairs: [...], keys}`` or None.
    :rtype: Optional[dict]
    :raises ValueError: When the file exists but no known key is present.
    """
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    pick = lambda *keys: next((raw[k] for k in keys if k in raw), None)
    qsrc, rsrc, lsrc = pick("queries", "sets", "items"), pick("rank_regressions", "regressions", "m2", "rank_cases"), \
        pick("label_swap_pairs", "m1")
    if qsrc is None and rsrc is None and lsrc is None:
        raise ValueError(f"{path}: unrecognised seed format; top-level keys {sorted(raw)}")
    queries = []
    items = [(None, x) for x in qsrc] if isinstance(qsrc, list) else \
        [(k, x) for k, xs in (qsrc or {}).items() for x in (xs if isinstance(xs, list) else [xs])]
    for set_name, x in items:
        if isinstance(x, str):
            x = {"text": x}
        text = x.get("text") or x.get("query")
        if text:
            queries.append({"text": text, "festival": x.get("festival") or x.get("concept") or x.get("set") or
                            x.get("topic") or set_name, "form": x.get("form") or x.get("type") or x.get("kind") or "seed"})
    cases = []
    for i, x in enumerate(rsrc or []):
        doc = x.get("doc_id") or x.get("target") or x.get("expected_doc_id")
        text = x.get("query") or x.get("text")
        if doc and text:
            cases.append({"case_id": x.get("case_id") or f"seed:{i}", "text": text, "doc_id": doc,
                          "max_rank": x.get("max_rank", 50), "source": "seed"})
    return {"queries": queries, "rank_regressions": cases, "label_swap_pairs": lsrc or [], "keys": sorted(raw)}


# ---------------------------------------------------------------------------------------------------------------
# relevance
# ---------------------------------------------------------------------------------------------------------------
def label_sets(subjects: Dict[str, dict], label_ids: Set[str]) -> Dict[str, Dict[str, Set[str]]]:
    """Per label id: strong (bare) carriers, canon carriers, typed-unjudged carriers and mention records.

    :param subjects: Pool subjects rows.
    :param label_ids: Label ids named by the spec.
    :returns: ``sid -> {strong, canon, unjudged, mention}``.
    :rtype: Dict[str, Dict[str, Set[str]]]
    """
    out = {sid: {"strong": set(), "canon": set(), "unjudged": set(), "mention": set()} for sid in label_ids}
    for d, r in subjects.items():
        for sid in (r["subjects"] | r["canon"]) & label_ids:
            reason = r["weak_reason"].get(sid) or ("weak" if sid in r["weak"] else None)
            in_canon = sid in r["canon"]
            if r["gold_override"] and in_canon and sid not in r["weak"]:
                out[sid]["unjudged"].add(d)            # documented override: a canon-only gold record is unjudged
            elif reason == "canon" or in_canon:
                out[sid]["canon"].add(d)
            elif reason is not None:                   # title / incidental / untyped weak (older schema)
                out[sid]["unjudged"].add(d)
            elif sid in r["subjects"]:
                out[sid]["strong"].add(d)
        for sid in r["mentions"] & label_ids:
            out[sid]["mention"].add(d)
    return out


def concept_relevance(c: dict, cid: str, pool: List[str], fields: Dict[str, Dict[str, str]],
                      labels: Dict[str, Dict[str, Set[str]]]) -> dict:
    """Relevant / canon / unjudged / alt-only records of one concept (precedence in the module docstring).

    :param c: Spec concept.
    :param cid: Concept id.
    :param pool: Pool ids.
    :param fields: ``doc_id -> record_fields``.
    :param labels: :func:`label_sets` output.
    :returns: ``{relevant, canon, unjudged, alt_only, text_relevant (None without a rule), n_relevant_text,
        n_relevant_label, n_desc_mention}`` (sets).
    :rtype: dict
    """
    law = c.get("rel_mode", "bare") == "law"
    l_strong, l_canon, l_unj, l_mention = set(), set(), set(), set()
    for sid in c.get("label_ids") or []:
        ls = labels.get(sid, {})
        l_strong |= ls.get("strong", set())
        l_canon |= ls.get("canon", set())
        l_unj |= ls.get("unjudged", set())
        l_mention |= ls.get("mention", set())
    l_canon -= l_strong
    l_unj -= l_strong | l_canon
    rule = Rule(c["rule"], DEFAULT_RULE_FIELDS, f"{cid}.rule") if c.get("rule") else None
    canon_rule = Rule(c["canon_rule"], DEFAULT_CANON_FIELDS, f"{cid}.canon_rule") if c.get("canon_rule") else None
    strict, loose, canon_hit, named_desc, named_alt = set(), set(), set(), set(), set()
    for d in pool:
        f = fields[d]
        if rule is not None and rule.quick(f["__all__"]):
            if rule.on(f):
                strict.add(d)
            if rule.on(f, RULE_FIELDS):
                loose.add(d)
            if rule.named(f["description"]):
                named_desc.add(d)
            if rule.named(f["alt_titles"]):
                named_alt.add(d)
        if canon_rule is not None and canon_rule.quick(f["__all__"]) and canon_rule.on(f):
            canon_hit.add(d)
    labelled = l_strong | l_canon | l_unj
    text_rel = (strict - labelled) if law else (strict - labelled - canon_hit)
    relevant = l_strong | text_rel | (l_canon if law else set())
    canon = ((canon_hit - labelled) | (set() if law else l_canon)) - relevant
    unjudged = (l_unj | ((loose | l_mention) - labelled)) - relevant - canon
    return {"relevant": relevant, "canon": canon, "unjudged": unjudged,
            "alt_only": {d for d in relevant if d in named_alt and d not in named_desc},
            "text_relevant": text_rel if rule is not None else None,
            "n_relevant_text": len(text_rel), "n_relevant_label": len(relevant & (l_strong | l_canon)),
            "n_desc_mention": len(relevant & named_desc)}


# ---------------------------------------------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------------------------------------------
def v4_position_variant(text_v4: str, desc: str, parts_named: callable, direction: str) -> Optional[str]:
    """Move the concept text inside the v4 ``Description:`` line (``desc; alt part; alt part``).

    :param text_v4: Embedded semantic_v4 text.
    :param desc: The record's own description (``text_v3`` Description, folded).
    :param parts_named: Callable telling whether a text names the concept.
    :param direction: ``tail_to_head`` (concept only in appended alt parts: move those parts to the head) or
        ``head_to_tail`` (concept only in the description: move it behind the alt parts).
    :returns: Variant text, or None when the record does not fit the direction.
    :rtype: Optional[str]
    """
    lines = text_v4.split("\n")
    i = next((k for k, ln in enumerate(lines) if ln.startswith("Description:")), None)
    if i is None or not desc:
        return None
    value = lines[i][len("Description:"):].strip()
    if not value.startswith(desc + "; "):
        return None
    parts = [p for p in value[len(desc) + 2:].split("; ") if p]
    if not parts:
        return None
    hit = [p for p in parts if parts_named(p)]
    if direction == "tail_to_head":
        if not hit or parts_named(desc):
            return None
        new = hit + [desc] + [p for p in parts if p not in hit]
    elif direction == "head_to_tail":
        if hit or not parts_named(desc):
            return None
        new = parts + [desc]
    else:
        raise ValueError(f"unknown direction {direction!r}")
    lines[i] = "Description: " + "; ".join(new)
    return "\n".join(lines)


def v3_label_swap(text_v3: str, parts_named: callable) -> Optional[Tuple[str, str, str]]:
    """Label-swap diagnostic on the v3 layout (V / B3): the concept text keeps its words, only its label changes.

    ``alt_to_desc``: concept named under ``Alternate titles:`` only -> the two labels swap. ``desc_to_alt``: no alt
    line and the concept named in ``Description:`` -> that line is relabelled ``Alternate titles:``.

    :param text_v3: semantic_v3 text.
    :param parts_named: Callable telling whether a text names the concept.
    :returns: (direction, folded original text, variant text) or None.
    :rtype: Optional[Tuple[str, str, str]]
    """
    lines = fold_field_lines(text_v3)
    di = next((k for k, ln in enumerate(lines) if ln.startswith("Description:")), None)
    ai = next((k for k, ln in enumerate(lines) if ln.startswith("Alternate titles:")), None)
    desc = lines[di][len("Description:"):].strip() if di is not None else ""
    alt = lines[ai][len("Alternate titles:"):].strip() if ai is not None else ""
    orig = "\n".join(lines)
    var = list(lines)
    if alt and parts_named(alt) and not parts_named(desc):
        if di is None:
            var[ai] = f"Description: {alt}"
        else:
            var[di], var[ai] = f"Description: {alt}", f"Alternate titles: {desc}"
        return "alt_to_desc", orig, "\n".join(var)
    if ai is None and desc and parts_named(desc):
        var[di] = f"Alternate titles: {desc}"
        return "desc_to_alt", orig, "\n".join(var)
    return None


def build_title_probes(concepts: Dict[str, dict], rel: Dict[str, dict], corpus: Dict[str, dict],
                       fields: Dict[str, Dict[str, str]], qids_by_concept: Dict[str, List[str]],
                       params: dict) -> List[dict]:
    """H5 probes: v4 position moves (tail_to_head, head_to_tail) and v3 label swaps, sampled deterministically and
    spread over concepts round-robin.

    :param concepts: Spec concepts.
    :param rel: ``cid -> concept_relevance`` output.
    :param corpus: Pool rows (``text``, ``text_v3``, ``text_version``).
    :param fields: ``doc_id -> record_fields``.
    :param qids_by_concept: Scorable query ids per concept.
    :param params: ``thresholds["H5"]["probe"]``.
    :returns: Probe rows ``{pid, doc_id, concept, qids, direction, text_version, kind, orig_text, variant_text}``.
    :rtype: List[dict]
    """
    fams = set(params.get("families", []))
    n_per, max_q, seed = params.get("n_per_direction", 150), params.get("max_qids", 6), str(params.get("seed", "gate"))
    cands: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for cid, c in sorted(concepts.items()):
        if c["family"] not in fams or not c.get("rule") or not qids_by_concept.get(cid):
            continue
        rule = Rule(c["rule"], DEFAULT_RULE_FIELDS, f"{cid}.rule")
        for d in sorted(rel[cid]["relevant"]):
            row = corpus[d]
            if row["text_version"].startswith("semantic_v4"):  # incl. semantic_v4_nf (no frames)
                for direction in ("tail_to_head", "head_to_tail"):
                    var = v4_position_variant(row["text"], fields[d]["description"], rule.named, direction)
                    if var is not None:
                        cands[direction][cid].append({"doc_id": d, "concept": cid, "direction": direction,
                                                      "text_version": "semantic_v4", "kind": "position",
                                                      "orig_text": row["text"], "variant_text": var})
            swap = v3_label_swap(row["text_v3"], rule.named)
            if swap is not None:
                cands[swap[0]][cid].append({"doc_id": d, "concept": cid, "direction": swap[0],
                                            "text_version": "semantic_v3", "kind": "label_swap",
                                            "orig_text": swap[1], "variant_text": swap[2]})
    out = []
    for direction, by_c in sorted(cands.items()):
        queues = {cid: sorted(v, key=lambda p: seeded_key(seed, direction, cid, p["doc_id"]))
                  for cid, v in by_c.items()}
        used: Set[str] = set()
        picked: List[dict] = []
        while len(picked) < n_per and any(queues.values()):
            for cid in sorted(queues):
                while queues[cid] and queues[cid][0]["doc_id"] in used:
                    queues[cid].pop(0)
                if queues[cid] and len(picked) < n_per:
                    p = queues[cid].pop(0)
                    used.add(p["doc_id"])
                    picked.append(p)
        for p in picked:
            p["qids"] = qids_by_concept[p["concept"]][:max_q]
            p["pid"] = f"{direction}:{p['concept']}:{p['doc_id']}"
            out.append(p)
    return out


# ---------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------
def build_queries(spec: dict, known: List[dict], pool_set: Set[str], seed: Optional[dict], chat: List[dict],
                  concept_rules: Dict[str, Rule], warnings: List[str]) -> Tuple[List[dict], Dict[str, dict]]:
    """All gate queries: spec concepts, known-item (G10), seed (G1/G2) and chat sessions (G11).

    :param spec: Gate spec.
    :param known: Eval-dir known-item rows.
    :param pool_set: Pool ids.
    :param seed: :func:`load_seed` output or None.
    :param chat: :func:`chat_queries` output.
    :param concept_rules: ``cid -> Rule`` (G1-G7 concepts with a rule; maps chat queries to concepts).
    :param warnings: Warning sink.
    :returns: (query rows, extra G10 concepts ``ki:<doc_id> -> {family, target}``).
    :rtype: Tuple[List[dict], Dict[str, dict]]
    """
    rows, seen = [], set()

    def add(qid: str, text: str, family: str, concept: Optional[str], form: str, blocked: bool, source: str,
            report_only: bool) -> None:
        key = (family, concept, norm_text(text))
        if key in seen:
            return
        seen.add(key)
        rows.append({"qid": qid, "text": text.strip(), "family": family, "concept": concept, "form": form,
                     "blocked": bool(blocked), "lang": query_lang(text), "source": source,
                     "report_only": bool(report_only)})

    for cid, c in spec["concepts"].items():
        ro = c.get("report_only", c["family"] in REPORT_ONLY_FAMILIES)
        for i, q in enumerate(c.get("queries") or []):
            add(f"{c['family']}:{cid}:{i}", q["text"], c["family"], cid, q.get("form") or "unspecified",
                q.get("blocked", True), "spec", ro)
    ki_concepts = {}
    for q in known:
        if q.get("shelfmark_query"):
            add(f"G13:{q['qid']}", q["text"], "G13", None, "shelfmark", True, "known_item", True)
        elif q["doc_id"] in pool_set:
            cid = f"ki:{q['doc_id']}"
            ki_concepts[cid] = {"family": "G10", "target": q["doc_id"]}
            add(f"G10:{q['qid']}", q["text"], "G10", cid, q.get("type", "known_item"), True, "known_item", False)
    if seed is not None:
        by_label: Dict[str, List[str]] = defaultdict(list)
        for cid, c in spec["concepts"].items():
            for sid in c.get("label_ids") or []:
                by_label[sid].append(cid)
                by_label[sid.split(":", 1)[-1]].append(cid)
            by_label[cid].append(cid)
        for i, q in enumerate(seed["queries"]):
            cands = by_label.get(str(q["festival"]), []) + by_label.get(f"topic:{q['festival']}", [])
            if not cands:
                warnings.append(f"seed query {q['text']!r}: festival {q['festival']!r} maps to no spec concept")
                continue
            want = "G2" if q["form"] in DESCRIPTIVE_FORMS or "descr" in str(q["form"]) else "G1"
            cid = next((x for x in cands if spec["concepts"][x]["family"] == want), cands[0])
            fam = spec["concepts"][cid]["family"]
            add(f"{fam}:seed:{i}", q["text"], fam, cid, f"seed_{q['form']}" if want == "G2" else str(q["form"]),
                want == "G2", "seed", False)
    for i, q in enumerate(chat):
        if q["form"] == "router_shelfmark":
            add(f"G13:chat:{i}", q["text"], "G13", None, q["form"], True, q["source"], True)
            continue
        hits = [cid for cid, r in concept_rules.items() if r.matches(q["text"])][:3]
        if not hits:
            add(f"G11:chat:{i}", q["text"], "G11", None, q["form"], True, q["source"], True)
        for cid in hits:
            add(f"G11:chat:{i}:{cid}", q["text"], "G11", cid, q["form"], True, q["source"], False)
    return rows, ki_concepts


def review_rows(cid: str, c: dict, r: dict, pool: List[str], groups: Dict[str, str], langs: Dict[str, str],
                fields: Dict[str, Dict[str, str]]) -> List[dict]:
    """Rule-audit rows of one concept: 20 matched (text-rule relevant, spread over source groups) + 20 unmatched
    (label-only relevant, unjudged, canon, then random pool records).

    :param cid: Concept id.
    :param c: Spec concept.
    :param r: :func:`concept_relevance` output.
    :param pool: Pool ids.
    :param groups: ``doc_id -> source_group``.
    :param langs: ``doc_id -> desc_lang``.
    :param fields: ``doc_id -> record_fields``.
    :returns: CSV rows (``matched`` = text-rule relevant; label-only concepts sample ``relevant`` instead).
    :rtype: List[dict]
    """
    rng = random.Random(seeded_key("review", cid))
    text_rel = r["text_relevant"]
    by_g: Dict[str, List[str]] = defaultdict(list)
    for d in sorted(r["relevant"] if text_rel is None else text_rel):
        by_g[groups[d]].append(d)
    for v in by_g.values():
        rng.shuffle(v)
    matched: List[str] = []
    while len(matched) < REVIEW_N and any(by_g.values()):
        for g in sorted(by_g):
            if by_g[g] and len(matched) < REVIEW_N:
                matched.append(by_g[g].pop())
    unmatched: List[Tuple[str, str]] = []
    label_only = [] if text_rel is None else sorted(r["relevant"] - text_rel)
    for kind, ids in (("label_only", label_only), ("unjudged", sorted(r["unjudged"])), ("canon", sorted(r["canon"]))):
        ids = list(ids)
        rng.shuffle(ids)
        unmatched += [(kind, d) for d in ids[:max(0, REVIEW_N - len(unmatched))]]
    judged = r["relevant"] | r["unjudged"] | r["canon"]
    while len(unmatched) < REVIEW_N and len(judged) + len(unmatched) < len(pool):
        d = pool[rng.randrange(len(pool))]
        if d not in judged and d not in {x for _, x in unmatched}:
            unmatched.append(("random", d))
    out = []
    for kind, d in [("relevant" if text_rel is None else "matched", x) for x in matched] + unmatched:
        f = fields[d]
        out.append({"concept": cid, "family": c["family"], "sample_kind": kind, "doc_id": d,
                    "source_group": groups[d], "desc_lang": langs[d],
                    "in_relevant": d in r["relevant"], "in_canon": d in r["canon"], "in_unjudged": d in r["unjudged"],
                    "description": f["description"][:400], "alt_titles": f["alt_titles"][:300],
                    "doc_type": f["doc_type"][:120], "language": f["language"][:120], "title": f["title"][:200],
                    "people": f["people"][:200], "places": f["places"][:200], "verdict": "", "notes": ""})
    return out


def build(args: argparse.Namespace) -> dict:
    """Build the gate dir (see the module docstring).

    :param args: Parsed CLI arguments.
    :returns: Build stats.
    :rtype: dict
    :raises ValueError: On an eval-pool / corpus mismatch (unless ``--allow-pool-mismatch``) or a bad spec.
    """
    warnings: List[str] = []
    out_dir = Path(args.out_dir)
    spec = load_spec(Path(args.spec))
    th_default = default_thresholds()
    th = deep_merge(th_default, spec.get("thresholds") or {})
    th.setdefault("status", "proposed - Isaac freezes")
    unread = unknown_keys(th_default, spec.get("thresholds") or {})
    for k in unread:
        warnings.append(f"spec threshold key {k} is not read by eval_gate.verdict")

    corpus, pool, n_corpus = load_corpus(Path(args.semantic))
    pool_set = set(pool)
    print(f"corpus: {n_corpus['rows']} rows, {len(pool)} eligible (pool), text versions {dict(n_corpus['text_versions'])}",
          flush=True)
    if n_corpus["has_text_v3"] < len(pool):
        warnings.append(f"{len(pool) - n_corpus['has_text_v3']} pool rows have no text_v3 (fields parsed from text)")
    eval_dir = Path(args.eval_dir)
    ev_pool = {r["doc_id"] for r in iter_jsonl(eval_dir / "eval_pool.jsonl")}
    if ev_pool != pool_set:
        msg = (f"eval pool ({len(ev_pool)}) and corpus eligible ids ({len(pool)}) differ: "
               f"{len(ev_pool - pool_set)} only in the eval pool, {len(pool_set - ev_pool)} only in the corpus")
        if not args.allow_pool_mismatch:
            raise ValueError(msg + " (rebuild the eval dir from this corpus, or pass --allow-pool-mismatch)")
        warnings.append(msg)
    held = set(json.loads((eval_dir / "held_out_records.json").read_text(encoding="utf-8"))["held_out"])
    known = list(iter_jsonl(eval_dir / "known_item.jsonl"))
    merged = load_merged(Path(args.merged), pool_set)
    if len(merged) < len(pool):
        warnings.append(f"{len(pool) - len(merged)} pool records are absent from the merged file (text fields only)")
    print(f"merged: {len(merged)} pool records matched", flush=True)
    fields = {d: record_fields(corpus[d]["text_v3"], merged.get(d)) for d in pool}
    subjects = load_subjects(Path(args.subjects), pool_set)
    if len(subjects) < len(pool):
        warnings.append(f"{len(pool) - len(subjects)} pool records have no subjects_v1 row")
    schema = Counter(k for r in iter_jsonl(Path(args.subjects)) for k in r.keys()) if args.subjects else Counter()
    for key in ("weak_reason", "canon", "mentions", "gold_override"):
        if not schema.get(key):
            warnings.append(f"subjects_v1 has no '{key}' key (older schema): treated as empty")
    v9_ids, v9_hashes, n_mix = load_v9_mix(Path(args.v9_mix) if args.v9_mix else None)
    if args.v9_mix is None:
        warnings.append("no --v9-mix: v9_trained is unknown (V would be over-scored; plan H9)")
    prev = load_prev_corpus(Path(args.prev_corpus) if args.prev_corpus else None)
    if prev is None:
        warnings.append("no --prev-corpus: new_since_v9 is null (H7's new-since-v9 cell is NOT MEASURED)")

    # records ------------------------------------------------------------------------------------------------
    groups, langs, records = {}, {}, []
    for d in pool:
        row, f, m = corpus[d], fields[d], merged.get(d) or {}
        groups[d] = source_group(m.get("sources", []))
        langs[d] = script_lang(f["description"]) or script_lang(f["alt_titles"]) or \
            script_lang(row["text_v3"]) or "mixed"
        h3 = md5(row["text_v3"])
        by = "id" if d in v9_ids else ("text" if h3 in v9_hashes else None)
        records.append({"doc_id": d, "source_group": groups[d], "desc_lang": langs[d],
                        "len_bucket": len_bucket(row["n_chars"]), "held_out": d in held,
                        "new_since_v9": None if prev is None else d not in prev,
                        "has_alt_titles": bool(f["alt_titles"]),
                        "text_hash": row["text_hash"], "text_hash_v3": h3, "sources": "+".join(m.get("sources", [])),
                        "canon_kind": canon_kind(subjects.get(d, {}).get("subjects", ())),
                        "v9_trained": by is not None if args.v9_mix else None, "v9_trained_by": by,
                        "v3_text_changed": None if prev is None or d not in prev else prev[d] != h3,
                        "n_chars": row["n_chars"]})

    # concepts -----------------------------------------------------------------------------------------------
    label_ids = {sid for c in spec["concepts"].values() for sid in c.get("label_ids") or []}
    labels = label_sets(subjects, label_ids)
    for sid in sorted(label_ids):
        if not any(labels[sid].values()):
            warnings.append(f"label id {sid} has no carrier in the pool")
    rel: Dict[str, dict] = {}
    for cid, c in sorted(spec["concepts"].items()):
        if c.get("rule") or c.get("label_ids"):
            rel[cid] = concept_relevance(c, cid, pool, fields, labels)
        else:
            rel[cid] = {"relevant": set(), "canon": set(), "unjudged": set(), "alt_only": set(), "text_relevant": None,
                        "n_relevant_text": 0, "n_relevant_label": 0, "n_desc_mention": 0}
    concept_rules = {cid: Rule(c["rule"], DEFAULT_RULE_FIELDS, f"{cid}.rule") for cid, c in spec["concepts"].items()
                     if c.get("rule") and c["family"] in ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
                     and not c.get("report_only")}
    seed = load_seed(Path(args.seed) if args.seed else None)
    if seed is None:
        warnings.append(f"seed {args.seed} absent: seed G1/G2 queries and the H6 seed cases are skipped "
                        "(H6 seed part NOT MEASURED unless the spec lists rank_regressions)")
    chat = chat_queries(Path(args.chat_sessions)) if args.chat_sessions and Path(args.chat_sessions).exists() else []
    if not chat:
        warnings.append("no chat-session queries (G11 has only spec queries)")
    queries, ki_concepts = build_queries(spec, known, pool_set, seed, chat, concept_rules, warnings)
    twins: Dict[str, List[str]] = defaultdict(list)
    for d in pool:
        twins[corpus[d]["text_hash"]].append(d)
    min_rel, exempt = th["H11"]["min_relevant"], set(th["H11"]["min_relevant_exempt"])
    concepts_out = {}
    for cid, c in sorted(spec["concepts"].items()):
        r = rel[cid]
        ro = c.get("report_only", c["family"] in REPORT_ONLY_FAMILIES)
        too_small = not ro and c["family"] not in exempt and len(r["relevant"]) < min_rel
        if too_small:
            warnings.append(f"concept {cid} ({c['family']}): {len(r['relevant'])} relevant < {min_rel}")
        concepts_out[cid] = {"family": c["family"], "label_ids": c.get("label_ids") or [],
                             "rel_mode": c.get("rel_mode", "bare"), "report_only": ro,
                             "relevant": sorted(r["relevant"]), "canon": sorted(r["canon"]),
                             "unjudged": sorted(r["unjudged"]), "alt_only": sorted(r["alt_only"]),
                             "held_out_relevant": sorted(r["relevant"] & held),
                             "n_relevant_text": r["n_relevant_text"], "n_relevant_label": r["n_relevant_label"],
                             "n_desc_mention": r["n_desc_mention"],
                             "n_by_group": dict(Counter(groups[d] for d in r["relevant"])), "too_small": too_small,
                             "audit": c.get("audit") or {"precision_matched": None, "n_checked": 0, "notes": ""}}
    for cid, k in sorted(ki_concepts.items()):
        rel_ids = sorted(twins[corpus[k["target"]]["text_hash"]])
        concepts_out[cid] = {"family": "G10", "label_ids": [], "rel_mode": "bare", "report_only": False,
                             "relevant": rel_ids, "canon": [], "unjudged": [], "alt_only": [],
                             "held_out_relevant": [], "target": k["target"], "n_relevant_text": 0,
                             "n_relevant_label": 0, "n_desc_mention": 0,
                             "n_by_group": dict(Counter(groups[d] for d in rel_ids)), "too_small": False,
                             "audit": {"precision_matched": None, "n_checked": 0, "notes": "known-item target + twins"}}

    # probes -------------------------------------------------------------------------------------------------
    qids_by_concept: Dict[str, List[str]] = defaultdict(list)
    for q in queries:
        if q["concept"] and not q["report_only"] and q["source"] == "spec":
            qids_by_concept[q["concept"]].append(q["qid"])
    title = build_title_probes(spec["concepts"], rel, corpus, fields, qids_by_concept, th["H5"]["probe"])
    if not any(p["text_version"] == th["H5"].get("text_version") for p in title):
        warnings.append(f"no title-position probe for text version {th['H5'].get('text_version')} (H5 NOT MEASURED)")
    cases, dropped = [], []
    spec_probes = spec.get("probes") or {}
    for i, x in enumerate(list(spec_probes.get("rank_regressions") or []) + (seed or {}).get("rank_regressions", [])):
        text, doc = x.get("text") or x.get("query"), x.get("doc_id") or x.get("target") or x.get("expected_doc_id")
        case = {"case_id": x.get("case_id") or f"spec:{i}", "text": text, "doc_id": doc,
                "max_rank": x.get("max_rank", th["H6"]["seed_max_rank"]), "source": x.get("source", "spec")}
        (cases if doc in pool_set and text else dropped).append(case)
    for case in dropped:
        warnings.append(f"rank-regression case {case['case_id']} dropped: {case['doc_id']} not in the pool")
    probes = {"rank_regressions": cases, "rank_regressions_dropped": dropped, "title_position": title,
              "label_swap_pairs": (seed or {}).get("label_swap_pairs", []),
              "rank_sweep": {"pairs": "held-out relevant records of every scored concept of the sweep families",
                             "families": th["H6"]["sweep_families"], "ref": th["H6"]["sweep_ref"]},
              "seed_present": seed is not None, "spec_title_position": spec_probes.get("title_position")}

    # size checks / audit ---------------------------------------------------------------------------------------
    scorable = Counter(q["family"] for q in queries if not q["report_only"] and q["concept"]
                       and concepts_out[q["concept"]]["relevant"])
    size = {f: {"n_queries": scorable.get(f, 0), "min": mn, "ok": scorable.get(f, 0) >= mn}
            for f, mn in th["H11"]["min_queries"].items()}
    for f, s in size.items():
        if not s["ok"]:
            warnings.append(f"family {f}: {s['n_queries']} scorable queries < minimum {s['min']}")
    audit_pending = sorted(cid for cid, c in concepts_out.items() if c["family"] != "G10" and not c["report_only"]
                           and (c["audit"].get("precision_matched") is None or c["audit"]["precision_matched"] < 0.9))

    # write --------------------------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "concepts.json").write_text(json.dumps(concepts_out, ensure_ascii=False), encoding="utf-8")
    (out_dir / "probes.json").write_text(json.dumps(probes, ensure_ascii=False), encoding="utf-8")
    (out_dir / "thresholds.json").write_text(json.dumps(th, ensure_ascii=False, indent=1), encoding="utf-8")
    review = []
    for cid, c in sorted(spec["concepts"].items()):
        if not concepts_out[cid]["report_only"] and (c.get("rule") or c.get("label_ids")):
            review += review_rows(cid, c, rel[cid], pool, groups, langs, fields)
    with open(out_dir / "review_sample.csv", "w", encoding="utf-8", newline="") as fh:
        cols = ["concept", "family", "sample_kind", "doc_id", "source_group", "desc_lang", "in_relevant", "in_canon",
                "in_unjudged", "description", "alt_titles", "doc_type", "language", "title", "people", "places",
                "verdict", "notes"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(review)
    th_sha = sha256_file(out_dir / "thresholds.json")
    inputs = {k: getattr(args, k) for k in ("spec", "semantic", "merged", "subjects", "eval_dir", "seed",
                                            "chat_sessions", "v9_mix", "prev_corpus")}
    shas = {k: sha256_file(Path(v)) for k, v in inputs.items()
            if v and Path(v).is_file() and k in ("spec", "semantic", "subjects", "seed", "v9_mix")}
    if args.hash_merged:
        shas["merged"] = sha256_file(Path(args.merged))
    stats = {"inputs": inputs, "inputs_sha256": shas, "thresholds_sha256": th_sha,
             "thresholds_status": th.get("status"), "families": spec.get("families", {}),
             "counts": {"pool": len(pool), "records_held_out": sum(r["held_out"] for r in records),
                        "records_new_since_v9": sum(bool(r["new_since_v9"]) for r in records),
                        "records_v9_trained": sum(bool(r["v9_trained"]) for r in records),
                        "records_v9_trained_by_text_only": sum(r["v9_trained_by"] == "text" for r in records),
                        "source_group": dict(Counter(groups.values())), "desc_lang": dict(Counter(langs.values())),
                        "len_bucket": dict(Counter(r["len_bucket"] for r in records)),
                        "has_alt_titles": sum(r["has_alt_titles"] for r in records),
                        "queries": len(queries), "queries_by_family": dict(Counter(q["family"] for q in queries)),
                        "queries_by_source": dict(Counter(q["source"].split(":")[0] for q in queries)),
                        "queries_blocked": sum(q["blocked"] for q in queries),
                        "concepts": len(concepts_out), "known_item_concepts": len(ki_concepts),
                        "chat_queries_extracted": len(chat),
                        "title_probes": dict(Counter(f"{p['text_version']}:{p['direction']}" for p in title)),
                        "rank_regressions": len(cases), "rank_regressions_dropped": len(dropped),
                        "review_rows": len(review), "v9_mix": dict(n_mix)},
             "size_checks": size, "rule_audit_pending": audit_pending,
             "concept_sizes": {cid: {"family": c["family"], "relevant": len(c["relevant"]), "canon": len(c["canon"]),
                                     "unjudged": len(c["unjudged"]), "alt_only": len(c["alt_only"]),
                                     "held_out_relevant": len(c["held_out_relevant"])}
                               for cid, c in concepts_out.items() if c["family"] != "G10"},
             "unread_threshold_keys": unread, "seed_keys": (seed or {}).get("keys"), "warnings": warnings}
    (out_dir / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    for w_ in warnings:
        print("WARNING", w_, flush=True)
    print(json.dumps({k: stats["counts"][k] for k in ("pool", "queries", "queries_by_family", "concepts",
                                                      "title_probes", "rank_regressions")}, ensure_ascii=False))
    print(f"thresholds.json sha256 {th_sha} ({th.get('status')})")
    return stats


# ---------------------------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows as JSONL (self-test helper).

    :param path: Target file.
    :param rows: Row dicts.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fixture(root: Path) -> argparse.Namespace:
    """Tiny synthetic inputs covering every relevance path, source and probe kind.

    :param root: Directory for the fixture files.
    :returns: CLI arguments pointing at them.
    :rtype: argparse.Namespace
    """
    root.mkdir(parents=True, exist_ok=True)
    recs = {  # doc_id: (text_v3, text_v4 or None (= text_v3 folded), eligible, sources, merged extras)
        "KTIV_1": ("Description: פיוט לפורים\nAlternate titles: Piyyut: Purim; Liturgy\nLanguage: Hebrew", True, ["ktiv"]),
        "KTIV_2": ("Description: קטע תפילה\nAlternate titles: Common Prayers: Purim Amidah; תפילות קבע", True, ["ktiv"]),
        "FJP_3": ("Description: Poem for Purim. Liturgy\nLanguage: Hebrew", True, ["fjp"]),
        "KTIV_4": ("Description: מגילת אסתר\nAlternate titles: Bible: Esther 3:1-5", True, ["ktiv"]),
        "PGP_5": ("Description: Letter mentioning a payment after Purim.\nLanguage: Judaeo-Arabic\nDocument Type: letter",
                  True, ["fjp", "pgp"]),
        "TEI_6": ("Description: A fragment of a marriage contract from Fustat\nLanguage: Aramaic", True,
                  ["bodleian", "fjp"]),
        "FJP_7": ("Description: Letter to the Nagid\nRelated People: Moses Maimonides", True, ["fjp"]),
        "FJP_8": ("Description: Liturgical fragment\nTranscription: ... לפורים ...", True, ["fjp"]),
        "FJP_9": ("Description: Purim", False, ["fjp"]),
        "KTIV_10": ("Description: קטע\nAlternate titles: Hoshanot; reading customs for the 14th of Adar\nLanguage: Hebrew",
                    True, ["ktiv"]),
        "KTIV_11": ("Description: Talmud Bavli Sukkah 29a\nAlternate titles: Talmud Bavli: Sukkah 29 a", True, ["ktiv"]),
        "FJP_12": ("Description: Hoshanot for Sukkot\ncontinuation line about the willow", True, ["fjp"]),
        "PGP_13": ("Description: Ketubba of the daughter of Moses\nDocument Type: ketubah", True, ["pgp"]),
    }
    for i in range(14, 40):
        recs[f"FJP_{i}"] = (f"Description: Fragment number {i} of a commentary on Genesis", True, ["fjp"])
    subjects = {"KTIV_1": {"subjects": ["topic:purim"], "weak": []},
                "KTIV_2": {"subjects": ["topic:purim"], "weak": []},
                "KTIV_4": {"subjects": ["topic:purim", "work:bible-esther"], "weak": ["topic:purim"],
                           "weak_reason": {"topic:purim": "canon"}, "canon": ["topic:purim"]},
                "PGP_5": {"subjects": ["topic:purim"], "weak": ["topic:purim"],
                          "weak_reason": {"topic:purim": "incidental"}},
                "KTIV_10": {"subjects": [], "weak": [], "mentions": ["topic:purim"]},
                "KTIV_11": {"subjects": ["topic:sukkot", "work:talmud-bavli-sukkah"], "weak": [],
                            "canon": ["topic:sukkot"], "gold_override": True},
                "FJP_12": {"subjects": ["topic:sukkot"], "weak": []}}
    corpus, merged, subj = [], [], []
    for d, (t3, elig, srcs) in recs.items():
        lines = fold_field_lines(t3)
        alt = next((ln for ln in lines if ln.startswith("Alternate titles:")), None)
        v4 = [ln for ln in lines if not ln.startswith(("Alternate titles:", "Document Type:"))]
        if alt:
            k = next(i for i, ln in enumerate(v4) if ln.startswith("Description:"))
            v4[k] = v4[k] + "; " + "; ".join(p.strip() for p in alt[len("Alternate titles:"):].split(";"))
        text = "\n".join(v4)
        corpus.append({"doc_id": d, "text": text, "text_v3": t3, "text_version": "semantic_v4", "eligible": elig,
                       "text_hash": md5(text)})
        src = {}
        if "ktiv" in srcs:
            src["ktiv"] = {"basic_catalog": {"title": "קטע גניזה", "language": "Hebrew"}, "scholarly_entries": []}
        if "fjp" in srcs:
            src["fjp"] = [{"language": "Hebrew", "other_languages": [],
                           "related_people": [{"name": "Maimonides"}] if d == "FJP_7" else [],
                           "related_places": [{"name": "Fustat"}]}]
        if "pgp" in srcs:
            src["pgp"] = {"documents": [{"type": "Letter" if d == "PGP_5" else "Legal document",
                                         "languages_primary": "Judaeo-Arabic", "origin": "Alexandria"}]}
        if "bodleian" in srcs:
            src["bodleian"] = {"tei": {"titles": ["A fragment of a marriage contract"], "languages": ["arc"],
                                       "orig_place": "Fostat", "items": [{"title": "marriage contract", "author": None}]}}
        merged.append({"canonical_id": d, "sources_present": srcs, "sources": src})
        s = subjects.get(d, {"subjects": [], "weak": []})
        subj.append({"doc_id": d, "subjects": s["subjects"], "weak": s["weak"],
                     "weak_reason": s.get("weak_reason", {}), "canon": s.get("canon", []),
                     "mentions": s.get("mentions", []), "gold_override": s.get("gold_override", False)})
    corpus.append({"doc_id": "DUP_A", "text": "Description: Fragment number 14 of a commentary on Genesis",
                   "text_v3": "Description: Fragment number 14 of a commentary on Genesis",
                   "text_version": "semantic_v4", "eligible": True,
                   "text_hash": md5("Description: Fragment number 14 of a commentary on Genesis")})
    merged.append({"canonical_id": "DUP_A", "sources_present": ["fjp"], "sources": {"fjp": [{"language": "Hebrew"}]}})
    _write_jsonl(root / "corpus_semantic.jsonl", corpus)
    _write_jsonl(root / "merged.jsonl", merged)
    _write_jsonl(root / "subjects_v1.jsonl", subj)
    eligible = [r["doc_id"] for r in corpus if r["eligible"]]
    ev = root / "eval"
    ev.mkdir(exist_ok=True)
    (ev / "held_out_records.json").write_text(json.dumps({"held_out": ["KTIV_10", "FJP_3", "KTIV_2", "FJP_14"]}))
    _write_jsonl(ev / "eval_pool.jsonl", [{"doc_id": d} for d in eligible])
    _write_jsonl(ev / "known_item.jsonl", [
        {"qid": "r1:FJP_14:0", "doc_id": "FJP_14", "type": "specific", "text": "commentary on Genesis fragment 14",
         "shelfmark_query": False},
        {"qid": "r1:FJP_9:0", "doc_id": "FJP_9", "type": "specific", "text": "ineligible target", "shelfmark_query": False}])
    spec = {"version": 1, "families": {"G1": "holiday names", "G3": "canon", "G4": "documentary", "G6": "people",
                                       "G7": "languages", "G12": "out of domain"},
            "concepts": {
                "fest:purim": {"family": "G1", "label_ids": ["topic:purim"],
                               "rule": {"any": [r"\bpurim\b", "{HB}פורים{HE}"], "fields": ["description", "alt_titles"]},
                               "canon_rule": {"any": [r"esther|אסתר|megill"]},
                               "queries": [{"text": "Purim", "form": "translit", "blocked": False},
                                           {"text": "הפורים", "form": "hebrew_prefixed", "blocked": True}],
                               "audit": {"precision_matched": 0.95, "n_checked": 20, "notes": "fixture"}},
                "fest:sukkot": {"family": "G1", "label_ids": ["topic:sukkot"],
                                "rule": {"any": [r"\bsukkot\b", "{HB}סוכות{HE}"]},
                                "queries": [{"text": "Sukkot", "form": "translit", "blocked": False}]},
                "canon:esther": {"family": "G3", "label_ids": ["work:bible-esther"],
                                 "queries": [{"text": "Book of Esther", "form": "canon", "blocked": True}]},
                "doc:marriage": {"family": "G4", "rule": {"any": ["marriage contract|ketub"],
                                                          "fields": ["description", "title"]},
                                 "queries": [{"text": "marriage contract", "form": "english", "blocked": True}]},
                "people:maimonides": {"family": "G6", "rule": {"any": ["maimonides"], "fields": ["people"]},
                                      "queries": [{"text": "Maimonides", "form": "english", "blocked": True}]},
                "lang:aramaic": {"family": "G7", "rule": {"any": ["aramaic"], "fields": ["language"]},
                                 "queries": [{"text": "Aramaic documents", "form": "english", "blocked": True}]},
                "ood": {"family": "G12", "report_only": True,
                        "queries": [{"text": "how to bake sourdough bread", "form": "ood", "blocked": False}]}},
            "probes": {"rank_regressions": [{"text": "Purim", "doc_id": "FJP_3", "max_rank": 50},
                                            {"text": "Purim", "doc_id": "FJP_9"}]},
            "thresholds": {"H1": {"se_mult": 2.0}, "H99": {"foo": 1},
                           "H5": {"probe": {"families": ["G1", "G4"], "n_per_direction": 10}}}}
    (root / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(root / "chat.jsonl", [
        {"session_id": 1, "tag": "eval", "question": "Purim fragments in the Cairo Genizah", "resolver_input": None,
         "router_plan": json.dumps([json.dumps({"actions": [{"search_type": "bibliography_hybrid", "query": "Purim"},
                                                            {"search_type": "primary_shelfmark", "query": "T-S 1"}]})])},
        {"session_id": 2, "tag": "organic", "question": "follow-up",
         "resolver_input": "CONVERSATION SO FAR:\nUser: x\n\nLATEST USER MESSAGE:\nWho is S.D. Goitein?",
         "router_plan": "[\"{\\\"actions\\\": [{\\\"query"},
        {"session_id": 3, "tag": "audit_probe", "question": "Sukkot", "router_plan": None}])
    _write_jsonl(root / "mix.jsonl", [
        {"anchor": "Poem for Purim", "doc_id": "FJP_3", "mask_ids": ["D:FJP_3"], "anchor_mask_ids": [],
         "text_hash": "x"},
        {"anchor": "letter", "doc_id": "GONE", "mask_ids": ["D:GONE"], "anchor_mask_ids": ["D:FJP_7"],
         "anchor_doc_id": "FJP_7", "text_hash": md5(recs["FJP_12"][0])}])
    _write_jsonl(root / "prev_corpus.jsonl", [{"doc_id": r["doc_id"], "text_hash": md5(r["text_v3"]),
                                              "text": r["text_v3"]} for r in corpus if r["doc_id"] != "KTIV_10"])
    return argparse.Namespace(spec=str(root / "spec.json"), semantic=str(root / "corpus_semantic.jsonl"),
                              merged=str(root / "merged.jsonl"), subjects=str(root / "subjects_v1.jsonl"),
                              eval_dir=str(ev), seed=str(root / "absent_seed.json"),
                              chat_sessions=str(root / "chat.jsonl"), v9_mix=str(root / "mix.jsonl"),
                              prev_corpus=str(root / "prev_corpus.jsonl"), out_dir=str(root / "gate_v1"),
                              allow_pool_mismatch=False, hash_merged=False)


def self_test(work_dir: Optional[str] = None) -> None:
    """Build the gate from :func:`_fixture` and check every relevance path, record attribute, query source and probe.

    :param work_dir: Directory for the fixture and outputs (default: a temporary directory).
    """
    assert fold_field_lines("Description: a\nmore\nLanguage: x") == ["Description: a more", "Language: x"]
    assert text_labels("Description: a\nmore\nTranscription: t\nTranslation: u")["Description"] == "a more"
    assert script_lang("פיוט לפורים") == "he" and script_lang("Poem for Purim") == "en" and script_lang("") is None
    assert script_lang("Piyyut פיוט לשבת") == "mixed" and len_bucket(99) == "lt100" and len_bucket(1000) == "ge1k"
    assert canon_kind(["work:talmud-bavli-sukkah"]) == "tractate" and canon_kind(["work:bible-esther"]) == "bible"
    assert canon_kind(["work:mishneh-torah-hametz"]) == "law_code" and canon_kind(["topic:purim"]) is None
    assert source_group(["fjp", "pgp"]) == "PGP" and source_group(["bodleian", "fjp"]) == "other"
    assert v4_position_variant("Description: d; Common Prayers: Purim; x", "d", lambda t: "Purim" in t,
                               "tail_to_head") == "Description: Common Prayers: Purim; d; x"
    assert v4_position_variant("Description: Purim d; x", "Purim d", lambda t: "Purim" in t,
                               "head_to_tail") == "Description: x; Purim d"
    assert default_thresholds()["H1"]["se_mult"] == 2.0
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        args = _fixture(Path(tmp) / "fixture")
        stats = build(args)
        out = Path(args.out_dir)
        con = json.loads((out / "concepts.json").read_text(encoding="utf-8"))
        recs = {r["doc_id"]: r for r in iter_jsonl(out / "records.jsonl")}
        qs = list(iter_jsonl(out / "queries.jsonl"))
        probes = json.loads((out / "probes.json").read_text(encoding="utf-8"))
        th = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
        pur = con["fest:purim"]
        assert set(pur["relevant"]) == {"KTIV_1", "KTIV_2", "FJP_3"}, pur["relevant"]          # label + text
        assert pur["canon"] == ["KTIV_4"], pur["canon"]                                         # canon label
        assert set(pur["unjudged"]) == {"PGP_5", "FJP_8", "KTIV_10"}, pur["unjudged"]           # incidental / transcr. / mention
        assert "FJP_9" not in json.dumps(con)                                                   # ineligible: nowhere
        assert pur["alt_only"] == ["KTIV_2"] and pur["held_out_relevant"] == ["FJP_3", "KTIV_2"]
        assert con["fest:sukkot"]["relevant"] == ["FJP_12"] and con["fest:sukkot"]["unjudged"] == ["KTIV_11"]
        assert con["canon:esther"]["relevant"] == ["KTIV_4"]
        assert set(con["doc:marriage"]["relevant"]) == {"TEI_6", "PGP_13"}                     # TEI title + description
        assert con["people:maimonides"]["relevant"] == ["FJP_7"]                               # merged FJP people
        assert set(con["lang:aramaic"]["relevant"]) == {"TEI_6"}                               # text_v3 Language + TEI
        assert con["ki:FJP_14"]["relevant"] == ["DUP_A", "FJP_14"]                             # known-item twins
        assert recs["KTIV_1"]["source_group"] == "KTIV" and recs["PGP_5"]["source_group"] == "PGP"
        assert recs["FJP_3"]["source_group"] == "FJP" and recs["TEI_6"]["source_group"] == "other"
        assert recs["KTIV_1"]["desc_lang"] == "he" and recs["FJP_3"]["desc_lang"] == "en"
        assert recs["KTIV_10"]["new_since_v9"] and not recs["FJP_3"]["new_since_v9"]
        assert recs["KTIV_10"]["held_out"] and recs["KTIV_2"]["has_alt_titles"] and not recs["FJP_3"]["has_alt_titles"]
        assert recs["FJP_3"]["v9_trained_by"] == "id" and recs["FJP_7"]["v9_trained_by"] == "id"
        assert recs["FJP_12"]["v9_trained_by"] == "text" and recs["KTIV_1"]["v9_trained"] is False
        assert recs["KTIV_11"]["canon_kind"] == "tractate" and recs["KTIV_4"]["canon_kind"] == "bible"
        fams = Counter(q["family"] for q in qs)
        assert fams["G10"] == 1 and fams["G12"] == 1 and fams["G1"] == 3, fams                # ineligible KI target skipped
        g11 = [q for q in qs if q["family"] == "G11"]
        assert {(q["text"], q["concept"]) for q in g11 if q["concept"]} == {
            ("Purim fragments in the Cairo Genizah", "fest:purim"), ("Purim", "fest:purim")}, g11
        assert any(q["text"] == "Who is S.D. Goitein?" and q["report_only"] for q in g11)
        assert any(q["family"] == "G13" and q["text"] == "T-S 1" for q in qs)
        assert not any(q["text"] == "Sukkot" and q["family"] == "G11" for q in qs)            # audit_probe tag skipped
        assert next(q for q in qs if q["text"] == "Purim" and q["family"] == "G1")["blocked"] is False
        assert [c["doc_id"] for c in probes["rank_regressions"]] == ["FJP_3"]
        assert probes["rank_regressions_dropped"][0]["doc_id"] == "FJP_9" and not probes["seed_present"]
        kinds = Counter((p["text_version"], p["direction"], p["doc_id"]) for p in probes["title_position"])
        assert ("semantic_v4", "tail_to_head", "KTIV_2") in kinds, kinds
        assert ("semantic_v3", "alt_to_desc", "KTIV_2") in kinds and ("semantic_v3", "desc_to_alt", "FJP_3") in kinds
        assert all(p["qids"] for p in probes["title_position"])
        assert th["status"].startswith("proposed") and th["H5"]["probe"]["n_per_direction"] == 10
        assert "H99" in stats["unread_threshold_keys"] and stats["thresholds_sha256"] == sha256_file(out / "thresholds.json")
        assert any("seed" in w and "absent" in w for w in stats["warnings"])
        assert stats["rule_audit_pending"] and "fest:purim" not in stats["rule_audit_pending"]
        with open(out / "review_sample.csv", encoding="utf-8") as fh:
            review = list(csv.DictReader(fh))
        kinds_rev = Counter((r["concept"], r["sample_kind"]) for r in review)
        assert kinds_rev[("fest:purim", "matched")] == 1 and kinds_rev[("fest:purim", "label_only")] == 2, kinds_rev
        assert kinds_rev[("fest:purim", "unjudged")] == 3 and kinds_rev[("fest:purim", "canon")] == 1
        assert sum(n for (c, _), n in kinds_rev.items() if c == "fest:purim") == 1 + 20
        assert len(set(open(out / "thresholds.json", encoding="utf-8").read())) > 0
        bad = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        bad["concepts"]["x"] = {"family": "G99", "queries": []}
        Path(args.spec).write_text(json.dumps(bad), encoding="utf-8")
        try:
            load_spec(Path(args.spec))
            raise AssertionError("invalid spec accepted")
        except ValueError as e:
            assert "G99" in str(e) and "neither rule nor label_ids" in str(e)
    print(json.dumps({"self_test": "ok", "concept_sizes": stats["concept_sizes"], "counts": stats["counts"]},
                     ensure_ascii=False, indent=1))


def main() -> None:
    """CLI (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--semantic", default=str(V4 / "corpus_semantic.jsonl"))
    parser.add_argument("--merged", default=str(V4 / "merged_shelfmarks_20261008.jsonl"))
    parser.add_argument("--subjects", default=str(V4 / "subjects_v1.jsonl"))
    parser.add_argument("--eval-dir", default=str(V4 / "eval_final"))
    parser.add_argument("--seed", default=str(DEFAULT_SEED), help="holiday_eval_seed.json (absent: skipped)")
    parser.add_argument("--chat-sessions", default=str(DEFAULT_CHAT))
    parser.add_argument("--v9-mix", default=str(AUDIT_ROOT / "v3" / "train_mix_v3final.jsonl"))
    parser.add_argument("--prev-corpus", default=str(AUDIT_ROOT / "v3" / "corpus_semantic.jsonl"),
                        help="corpus v9 was trained on (new_since_v9); 'none' to skip")
    parser.add_argument("--out-dir", default=str(V4 / "eval_final" / "gate_v1"))
    parser.add_argument("--allow-pool-mismatch", action="store_true")
    parser.add_argument("--hash-merged", action="store_true", help="also sha256 the 1 GB merged file")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--work-dir", default=None, help="self-test: parent dir of the temporary fixture")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.work_dir)
        return
    for key in ("v9_mix", "prev_corpus", "seed", "chat_sessions"):
        if str(getattr(args, key)).lower() == "none":
            setattr(args, key, None)
    build(args)


if __name__ == "__main__":
    main()
