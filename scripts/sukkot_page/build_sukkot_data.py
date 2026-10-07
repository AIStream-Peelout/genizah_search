"""Build the public JSON for the /sukkot page from the curated seed.

Reads ``docs/planned_features/sukkot_page/sukkot_seed.json`` and writes the one
prebuilt JSON the page ships (``src/frontend/src/sukkot/data/sukkot_5787.json``)
plus a build report that is never shipped (``scripts/sukkot_page/out/build_report.json``).

The JSON is public: everything in it reaches every visitor. The builder copies
only whitelisted fields, then validates the result and exits non-zero on any
hit (forbidden project names, machine-read claims, card sources, references,
hedges, size and, with ``--check-images``, image HTTP status). Internal
wording, raw shelfmarks and every exclusion go to the report instead.

Usage::

    python3 scripts/sukkot_page/build_sukkot_data.py --check-images
    python3 scripts/sukkot_page/build_sukkot_data.py --check-images --thumbs src/frontend/public/sukkot/thumbs

``--thumbs`` downloads each card image once (about 2.8 MB for KTIV images) and
writes a 320 px JPEG; it is opt-in and needs the owner's OK.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from festival_days import DAY_IDS, FESTIVAL_DAYS, LISTED_NOT_IN_SEED, NAMED_PLACEMENTS, explain_placement  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = REPO_ROOT / "docs/planned_features/sukkot_page/sukkot_seed.json"
DEFAULT_OUT = REPO_ROOT / "src/frontend/src/sukkot/data/sukkot_5787.json"
DEFAULT_REPORT = REPO_ROOT / "scripts/sukkot_page/out/build_report.json"

SCHEMA = "sukkot-page/1"
EDITION = "Sukkot 5787 (from sundown 25 September to 4 October 2026)"
CATALOGUE_INDEX = "genizah_merged_v8"
AI_INDEX = "genizah_ai_transcriptions_v2"
HEBREW_YEAR = 5787
MAX_BYTES = 300 * 1024
THUMB_WIDTH = 320
THUMB_QUALITY = 72
THUMB_URL_PREFIX = "/sukkot/thumbs/"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
REQUEST_PAUSE_S = 0.3

#: Names that must never appear anywhere in the public JSON.
FORBIDDEN_TERMS_RE = re.compile(r"friedberg|fjms|fjp|fgp|genazim|genizah\.org", re.IGNORECASE)
#: Words that would present a machine reading as checked.
CHECK_CLAIM_RE = re.compile(r"\b(checked|confirmed|verified)\b", re.IGNORECASE)
#: Seed fields (and derived keys) that must never be shipped.
INTERNAL_KEYS: Set[str] = {
    "card_source", "card_source_text", "scholarship_note", "notes", "readiness", "readiness_flags",
    "spot_check", "old_ids", "zero_agreed_reads_public", "ai_read", "record_mentions_sukkot",
    "origin", "source_group", "ktiv_sys_num", "ktiv_page_url", "embedding", "embeddings", "vector", "xy",
    "verified_claims", "outside_seed",
}
#: Seed ``card_source_public`` values accepted as human sources -> short category.
HUMAN_SOURCE_LABELS: Dict[str, str] = {
    "KTIV": "KTIV", "KTIV (National Library of Israel)": "KTIV",
    "PGP": "PGP", "Princeton Geniza Project": "PGP",
    "OPenn": "OPenn", "OPenn (Penn Libraries)": "OPenn",
    "catalogue note": "catalogue note",
    "catalogue record": "catalogue record", "catalogue record on this site": "catalogue record",
    "published scholarship": "published scholarship",
}
#: Words that disqualify a card source as a human source.
NON_HUMAN_SOURCE_RE = re.compile(r"\b(machine|ai|vlm|model|claude|automatic|ocr|htr)\b", re.IGNORECASE)
#: Public rewrites of curated date caveats whose seed wording addresses an editor
#: ("Do not reuse ...", "CONTESTED day:", "Computed check:"). Exact-string
#: replacements applied to ``date.caveat`` before validation.
DATE_CAVEAT_REWRITES = {
    "Do not reuse PGP's 'Sept. 29, 1219' for the verso (part b): ":
        "PGP gives 29 September 1219 for the verso (part b); its ",
    "CONTESTED day: Stern 2019 second day of Sukkot (Fri 21 Sep 921) vs Laufer 2025 Hoshana Rabbah":
        "Contested day: Stern 2019, the second day of Sukkot (Fri 21 Sep 921); Laufer 2025, Hoshana Rabbah",
    "Computed check: ": "Computed: ",
}


def rewrite_date_caveat(text):
    """Apply the public rewrites to a date caveat.

    :param text: Caveat text after internal clauses were filtered, or None.
    :return: The caveat with editor-facing wording replaced, or None.
    """
    if not text:
        return text
    for old, new in DATE_CAVEAT_REWRITES.items():
        text = text.replace(old, new)
    return text


#: Clauses of curated caveats that are internal instructions or name a zero-agreed read.
INTERNAL_CLAUSE_RE = re.compile(
    r"\b0 of \d+|\bzero\b|never link|link only|do not use|do not quote|say so if linked|build the /read"
    r"|old id|\bv8\b|available=false|ai search|machine read|checked|confirmed|verified",
    re.IGNORECASE,
)
#: Internal markers in a card_source_note.
INTERNAL_NOTE_RE = re.compile(r"re-check|before quoting|credit as|internal|todo", re.IGNORECASE)
#: Wording worth a human look in a shipped string (warning only).
INSTRUCTION_WORDING_RE = re.compile(r"\b(do not|don't|never)\b", re.IGNORECASE)
EXCLUDED_SHELFMARKS: Set[str] = {"ENA NS 5.29"}
#: Catalogue prefixes ("Paris AIU: IV.B.46") whose acronym is part of the display shelfmark.
KEEP_PREFIX_ACRONYMS: Set[str] = {"AIU", "JRL", "BL", "RNL", "NLI", "BNU"}


# --------------------------------------------------------------------------- text helpers


def natural_key(text: str) -> List[Any]:
    """Split a string into text and integer parts for natural sorting.

    :param text: String such as a shelfmark.
    :return: Sort key in which ``"T-S 8J"`` sorts before ``"T-S 24"``.
    """
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text)]


def normalise_shelfmark(raw: str) -> str:
    """Normalise a shelfmark label to the page convention (tech_notes section 3.6).

    Strips institution prefixes ("Cambridge University Library, Cambridge, England Ms. ",
    "Cambridge CUL: ", "Manchester: Rylands " ...), closes KTIV's spaced T-S classmarks
    ("T-S 10 H 4.5" -> "T-S 10H4.5") and writes Bodleian shelfmarks as "Bodl. MS heb. d 76/30".
    Already-short forms are returned unchanged.

    :param raw: Shelfmark as stored in the seed or the catalogue.
    :return: Display shelfmark.
    """
    text = " ".join(raw.split())
    text = re.sub(r"^.*?,\s*England\s+Ms\.?\s+", "", text)
    text = re.sub(r"^Manchester:\s*Rylands\s+", "JRL ", text)
    prefixed = re.match(r"^([A-Z][A-Za-z .-]*?):\s+(.+)$", text)
    if prefixed:
        prefix, rest = prefixed.groups()
        acronym = prefix.split()[-1]
        keep = acronym in KEEP_PREFIX_ACRONYMS and not rest.startswith(acronym)
        text = f"{acronym} {rest}" if keep else rest
    text = re.sub(r"^T-S (\d+) ([A-Z]{1,2}) (\d)", r"T-S \1\2\3", text)
    bodl = re.match(r"^(?:Bodl\.?\s+)?(?:MS\.?\s+)?heb\.?\s*([a-g])\.?\s+(\d+)[./](\S+)$", text, re.IGNORECASE)
    if bodl:
        text = f"Bodl. MS heb. {bodl.group(1).lower()} {bodl.group(2)}/{bodl.group(3)}"
    return text


def split_clauses(text: str) -> List[str]:
    """Split a caveat into clauses at ``;``/``:`` and at sentence ends.

    A full stop only ends a clause when a capital letter follows, so "Sept. 29" stays whole.

    :param text: Caveat text.
    :return: Non-empty clauses, each with its trailing punctuation.
    """
    parts = re.split(r"(?<=[;:])\s+|(?<=\.)\s+(?=[A-Z])", text.strip())
    return [part for part in parts if part.strip()]


def filter_public_clauses(text: Optional[str], internal_re: re.Pattern = INTERNAL_CLAUSE_RE) -> Optional[str]:
    """Drop clauses that are internal instructions or mention zero-agreed reads.

    :param text: Curated caveat, or None.
    :param internal_re: Pattern marking a clause as internal.
    :return: The remaining text ending with a full stop, or None when nothing public remains.
    """
    if not text:
        return None
    kept = [clause for clause in split_clauses(text) if not internal_re.search(clause)]
    if not kept:
        return None
    joined = " ".join(kept).strip().rstrip(";:,").strip()
    if not joined:
        return None
    return joined if joined.endswith((".", ")", "'", '"')) else joined + "."


def public_card_source_note(note: Optional[str], card_source_public: Optional[str]) -> Optional[str]:
    """Return a card_source_note fit for the public page, or None.

    Removes "(credit as ...)" instructions, rejects notes with internal markers or forbidden
    names, and drops notes that only repeat the card's public source label.

    :param note: Seed ``card_source_note``.
    :param card_source_public: The row's public source label.
    :return: Public note or None.
    """
    if not note:
        return None
    text = re.sub(r"\s*\((?:credit|cite) as [^)]*\)", "", note).strip()
    if not text or INTERNAL_NOTE_RE.search(text) or FORBIDDEN_TERMS_RE.search(text):
        return None
    if text.rstrip(".").strip().casefold() == (card_source_public or "").casefold():
        return None
    return text


def clean_citation(entry: str) -> Tuple[Optional[str], Optional[str]]:
    """Turn a seed bibliography entry into a plain one-line citation.

    Fixes the missing spaces the catalogue export leaves ("Gil,Palestine", "CE(Leiden",
    "athttps://"), trims an entry truncated with "…" back to its last closing parenthesis and
    rejects entries that are not citations.

    :param entry: Raw bibliography string.
    :return: ``(citation, None)`` when usable, else ``(None, reason)``.
    """
    text = " ".join(str(entry).split())
    if not text:
        return None, "empty"
    if FORBIDDEN_TERMS_RE.search(text):
        return None, "names a forbidden project"
    if re.search(r"\bunknown\b", text, re.IGNORECASE):
        return None, "incomplete citation ('unknown')"
    text = re.sub(r"\bat(https?://)", r"at \1", text)
    text = re.sub(r",(?=[^\s\d\"'”’])", ", ", text)
    text = re.sub(r";(?=\S)", "; ", text)
    text = re.sub(r"(?<=[\w\"”)])\((?=[A-Za-z])", " (", text)
    if text.endswith("…"):
        cut = text.rfind(")")
        if cut < 40:
            return None, "truncated in the seed"
        text = text[:cut + 1]
    return text, None


def public_bibliography_line(entries: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return the first seed bibliography entry as a public citation.

    :param entries: Seed ``bibliography`` list.
    :return: ``(citation, None)`` or ``(None, reason)``.
    """
    if not entries:
        return None, "no bibliography in the seed"
    return clean_citation(entries[0])


# --------------------------------------------------------------------------- selection


def exclusion_reasons(fragment: Dict[str, Any], excluded_ids: Set[str], excluded_shelfmarks: Set[str]) -> List[str]:
    """List every reason a seed row stays off the page (empty list = include).

    :param fragment: Seed fragment row.
    :param excluded_ids: doc_ids from ``excluded_machine_only``.
    :param excluded_shelfmarks: Shelfmarks excluded by name (Salonika, ``not_on_site``).
    :return: Reasons, in a stable order.
    """
    reasons: List[str] = []
    if not fragment.get("usable_now"):
        reasons.append(f"usable_now is false (readiness: {fragment.get('readiness')})")
    if "needs_human_check" in (fragment.get("readiness_flags") or []):
        reasons.append("readiness_flags contain needs_human_check")
    if fragment.get("doc_id") in excluded_ids:
        reasons.append("listed in excluded_machine_only")
    if fragment.get("shelfmark") in excluded_shelfmarks:
        reasons.append("excluded by shelfmark (Salonika / not_on_site)")
    if not (fragment.get("images") or {}).get("ok_url"):
        reasons.append("no working image (images.ok_url is empty)")
    return reasons


def select_fragments(seed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split the seed's fragments into included rows and reported exclusions.

    :param seed: Parsed seed JSON.
    :return: ``(included, excluded)``; each exclusion is ``{doc_id, shelfmark, theme, reasons}``.
    """
    excluded_ids = {row["doc_id"] for row in seed.get("excluded_machine_only", []) if row.get("doc_id")}
    excluded_shelfmarks = set(EXCLUDED_SHELFMARKS) | {row["shelfmark"] for row in seed.get("not_on_site", [])}
    included: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for fragment in seed["fragments"]:
        reasons = exclusion_reasons(fragment, excluded_ids, excluded_shelfmarks)
        if reasons:
            excluded.append({"doc_id": fragment["doc_id"], "shelfmark": fragment["shelfmark"],
                             "theme": fragment.get("theme"), "reasons": reasons})
        else:
            included.append(fragment)
    return included, excluded


# --------------------------------------------------------------------------- per-fragment fields


def machine_read_label(n_agreed: int, n_lines: int) -> str:
    """Return the only wording the page uses for a machine reading.

    :param n_agreed: Lines on which the two machine readers agree.
    :param n_lines: Lines in the read.
    :return: Label text.
    """
    return (f"Machine reading, not a human transcription: {n_agreed} of {n_lines} lines "
            f"agreed by two machine readers")


def build_machine_read(ai_read: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build the public ``machine_read`` block, or None when the read must not be linked.

    A read is linked only when the seed marks it linkable and it has at least one agreed line.
    No read text is copied; the note is the curated caveat with internal clauses removed.

    :param ai_read: Seed ``ai_read`` block.
    :return: ``{n_agreed, n_lines, image_index, image_url, href, label, note}`` or None.
    """
    if not ai_read or not ai_read.get("linkable"):
        return None
    n_agreed = ai_read.get("n_agreed") or 0
    if n_agreed <= 0 or not ai_read.get("ai_read_doc_id") or ai_read.get("image_index") is None:
        return None
    return {
        "n_agreed": n_agreed,
        "n_lines": ai_read["n_lines"],
        "image_index": ai_read["image_index"],
        "image_url": ai_read.get("image_url"),
        "href": f"/read?doc={ai_read['ai_read_doc_id']}&image={ai_read['image_index']}",
        "label": machine_read_label(n_agreed, ai_read["n_lines"]),
        "note": filter_public_clauses(ai_read.get("caveat")),
    }


def build_date(fragment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the public ``date`` block from the human-sourced date fields.

    :param fragment: Seed fragment row.
    :return: ``{display, ce_start, ce_end, kind, precise, caveat}`` or None without ``date_text``.
    """
    if not fragment.get("date_text"):
        return None
    return {
        "display": fragment["date_text"],
        "ce_start": fragment.get("date_ce_start"),
        "ce_end": fragment.get("date_ce_end"),
        "kind": fragment.get("date_kind"),
        "precise": fragment.get("date_precise"),
        "caveat": rewrite_date_caveat(filter_public_clauses(fragment.get("date_caveat"), re.compile(r"machine read|do not quote", re.I))),
    }


def text_status(has_human_transcription: bool, machine_read: Optional[Dict[str, Any]]) -> str:
    """Classify what text a visitor can read for the card.

    :param has_human_transcription: Seed flag.
    :param machine_read: Public machine_read block or None.
    :return: ``"human"``, ``"machine"`` or ``"none"``.
    """
    if has_human_transcription:
        return "human"
    return "machine" if machine_read else "none"


API_BASE = "https://api.cairogenizah.ai"
#: Seconds between public API calls when refreshing reads.
API_PACE_SECONDS = 0.3


def best_read_item(items: List[Dict[str, Any]], image_ok=None) -> Optional[Dict[str, Any]]:
    """Pick the image whose read has the most agreed lines (ties: higher agreed share).

    Items whose image file is missing are skipped: some records list scans that
    were never uploaded to the bucket, and a link to a read on a missing image
    opens an empty viewer.

    :param items: ``items`` of the ``/ai-transcriptions/{id}`` response.
    :param image_ok: Optional predicate ``url -> bool``; items failing it are skipped.
    :return: The best item, or None when no item has an agreed line on a live image.
    """
    best = None
    for item in items:
        read = item.get("ai_read") or {}
        n_agreed = read.get("n_agreed") or 0
        n_lines = read.get("n_lines") or 0
        if n_agreed <= 0:
            continue
        if image_ok is not None and not image_ok(item.get("image_url")):
            continue
        key = (n_agreed, n_agreed / n_lines if n_lines else 0.0)
        if best is None or key > best[0]:
            best = (key, item)
    return best[1] if best else None


def live_machine_read(fragment: Dict[str, Any], session: Any) -> Optional[Dict[str, Any]]:
    """Refresh a fragment's machine read from the public reads endpoint.

    Tries the catalogue id first, then the ids the seed knows reads live under
    (``ai_read.ai_read_doc_id`` and ``old_ids``). Only a read with at least one
    agreed line becomes a link; the curated caveat is kept as the public note.

    :param fragment: Seed fragment row.
    :param session: ``requests.Session`` with the browser-like User-Agent.
    :return: Seed-shaped ``ai_read`` block (linkable) or None.
    """
    seed_read = fragment.get("ai_read") or {}
    candidates: List[str] = [fragment["doc_id"]]
    for extra in [seed_read.get("ai_read_doc_id")] + list(fragment.get("old_ids") or []):
        if extra and extra not in candidates:
            candidates.append(extra)
    for read_id in candidates:
        time.sleep(API_PACE_SECONDS)
        response = session.get(f"{API_BASE}/ai-transcriptions/{read_id}", timeout=30)
        if response.status_code != 200:
            continue
        data = response.json()
        if not data.get("available"):
            continue
        item = best_read_item(data.get("items") or [], image_ok=lambda url: image_status(session, url) == 200)
        if item is None:
            return None
        read = item["ai_read"]
        return {
            "linkable": True, "ai_read_doc_id": read_id, "image_index": item["image_index"],
            "image_url": item.get("image_url"), "n_agreed": read["n_agreed"], "n_lines": read["n_lines"],
            "vlm_model": read.get("vlm_model"), "caveat": seed_read.get("caveat"),
        }
    return None


def refresh_reads(included: List[Dict[str, Any]], session: Any) -> Dict[str, Any]:
    """Replace each included fragment's ``ai_read`` with the live read summary.

    :param included: Selected seed rows (mutated in place).
    :param session: ``requests.Session``.
    :return: Report ``{refreshed, linkable_before, linkable_after, changed: [doc_ids]}``.
    """
    before = {f["doc_id"] for f in included if build_machine_read(f.get("ai_read"))}
    changed: List[str] = []
    for fragment in included:
        live = live_machine_read(fragment, session)
        old = build_machine_read(fragment.get("ai_read"))
        fragment["ai_read"] = live
        new = build_machine_read(live)
        if (old or {}).get("n_agreed") != (new or {}).get("n_agreed") or (old or {}).get("href") != (new or {}).get("href"):
            changed.append(fragment["doc_id"])
    after = {f["doc_id"] for f in included if build_machine_read(f.get("ai_read"))}
    return {"refreshed": len(included), "linkable_before": len(before), "linkable_after": len(after), "changed": changed}


def build_fragment(fragment: Dict[str, Any], shipped_clusters: Set[str], thumbs: Dict[str, str]) -> Dict[str, Any]:
    """Build one public fragment card from a seed row (whitelisted fields only).

    :param fragment: Seed fragment row (already selected).
    :param shipped_clusters: Cluster ids that ship (two or more included members).
    :param thumbs: doc_id -> thumbnail URL for thumbnails written in this run.
    :return: Public card dict.
    """
    machine_read = build_machine_read(fragment.get("ai_read"))
    bibliography_line, _reason = public_bibliography_line(fragment.get("bibliography") or [])
    return {
        "doc_id": fragment["doc_id"],
        "ai_read_doc_id": fragment["ai_read"]["ai_read_doc_id"] if machine_read else None,
        "shelfmark": normalise_shelfmark(fragment["shelfmark"]),
        "holding": fragment.get("institution"),
        "holding_credit": fragment.get("holding_credit"),
        "theme": fragment["theme"],
        "subtheme": fragment.get("subtheme"),
        "genre": fragment["genre"],
        "clusters": [cid for cid in fragment.get("clusters") or [] if cid in shipped_clusters],
        "tags": list(fragment.get("tags") or []),
        "contested": bool(fragment.get("contested")),
        "card_line": fragment.get("card_line"),
        "card_source_public": fragment.get("card_source_public"),
        "card_source_note": public_card_source_note(fragment.get("card_source_note"),
                                                    fragment.get("card_source_public")),
        "date": build_date(fragment),
        "place": fragment.get("place") or None,
        "festival_day": explain_placement(fragment)[0],
        "image": {"url": fragment["images"]["ok_url"], "thumb": thumbs.get(fragment["doc_id"])},
        "text_status": text_status(bool(fragment.get("has_human_transcription")), machine_read),
        "has_translation": bool(fragment.get("has_translation")),
        "machine_read": machine_read,
        "bibliography_line": bibliography_line,
        "usable_now": True,
    }


# --------------------------------------------------------------------------- page-level blocks


def build_themes(seed_themes: Dict[str, Dict[str, Any]], fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the themes that have at least one included fragment, in theme order.

    :param seed_themes: Seed ``themes`` dict (id -> theme).
    :param fragments: Public fragments, already sorted.
    :return: Public theme list.
    """
    themes: List[Dict[str, Any]] = []
    for theme_id, theme in sorted(seed_themes.items(), key=lambda item: item[1]["order"]):
        members = [frag["doc_id"] for frag in fragments if frag["theme"] == theme_id]
        if not members:
            continue
        themes.append({
            "id": theme_id, "title": theme["title"], "order": theme["order"], "genre": theme["genre"],
            "intro": theme["intro"], "hedges": list(theme.get("hedges") or []),
            "contested": json.loads(json.dumps(theme.get("contested") or [])),
            "refs": list(theme.get("refs") or []), "fragment_ids": members,
        })
    return themes


def build_clusters(seed_clusters: List[Dict[str, Any]], included_ids: Set[str]) -> List[Dict[str, Any]]:
    """Build the clusters that keep two or more included members.

    :param seed_clusters: Seed ``clusters`` list.
    :param included_ids: doc_ids of included fragments.
    :return: Public cluster list (``outside_seed`` is not shipped).
    """
    clusters: List[Dict[str, Any]] = []
    for cluster in seed_clusters:
        members = [doc_id for doc_id in cluster.get("members") or [] if doc_id in included_ids]
        if len(members) >= 2:
            clusters.append({"id": cluster["id"], "title": cluster["title"], "basis": cluster.get("basis"),
                             "caveat": cluster.get("caveat"), "members": members})
    return clusters


def build_genres(seed_genres: Dict[str, Dict[str, Any]], used: Set[str]) -> List[Dict[str, Any]]:
    """Build the genre colour buckets in use, sorted by order.

    :param seed_genres: Seed ``genres`` dict.
    :param used: Genre ids used by shipped themes or fragments.
    :return: Public genre list.
    """
    return [{"id": gid, "label": genre["label"], "order": genre["order"]}
            for gid, genre in sorted(seed_genres.items(), key=lambda item: item[1]["order"]) if gid in used]


def build_output(seed: Dict[str, Any], generated_at: str,
                 thumbs: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the public page JSON and the (unshipped) build report.

    :param seed: Parsed seed JSON.
    :param generated_at: ISO timestamp for the output.
    :param thumbs: doc_id -> thumbnail URL (only for thumbnails written in this run).
    :return: ``(output, report)``.
    """
    thumbs = thumbs or {}
    included, excluded = select_fragments(seed)
    included_ids = {frag["doc_id"] for frag in included}
    clusters = build_clusters(seed["clusters"], included_ids)
    shipped_clusters = {cluster["id"] for cluster in clusters}
    theme_order = {tid: theme["order"] for tid, theme in seed["themes"].items()}
    included.sort(key=lambda frag: (theme_order.get(frag["theme"], 999), natural_key(normalise_shelfmark(frag["shelfmark"]))))
    fragments = [build_fragment(frag, shipped_clusters, thumbs) for frag in included]
    themes = build_themes(seed["themes"], fragments)
    used_genres = {frag["genre"] for frag in fragments} | {theme["genre"] for theme in themes}
    output = {
        "schema": SCHEMA, "edition": EDITION, "generated_at": generated_at,
        "catalogue_index": CATALOGUE_INDEX, "ai_index": AI_INDEX,
        "festival": {"hebrew_year": HEBREW_YEAR, "days": [dict(day) for day in FESTIVAL_DAYS]},
        "genres": build_genres(seed["genres"], used_genres),
        "themes": themes,
        "clusters": clusters,
        "refs": dict(seed["refs"]),
        "fragments": fragments,
    }
    report = build_report(seed, included, excluded, fragments, clusters)
    return output, report


def build_report(seed: Dict[str, Any], included: List[Dict[str, Any]], excluded: List[Dict[str, Any]],
                 fragments: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect the build decisions that are not shipped (raw shelfmarks, dropped fields, counts).

    :param seed: Parsed seed JSON.
    :param included: Included seed rows, in output order.
    :param excluded: Exclusions from :func:`select_fragments`.
    :param fragments: Public fragments (same order as ``included``).
    :param clusters: Public clusters.
    :return: Report dict.
    """
    per_theme: Dict[str, int] = {}
    per_day: Dict[str, Dict[str, int]] = {}
    placement_problems: List[Dict[str, str]] = []
    decisions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for row, frag in zip(included, fragments):
        per_theme[frag["theme"]] = per_theme.get(frag["theme"], 0) + 1
        day = frag["festival_day"]
        key = day["day"] if day else "none"
        basis = day["basis"] if day else "none"
        per_day.setdefault(key, {})
        per_day[key][basis] = per_day[key].get(basis, 0) + 1
        _placement, problem = explain_placement(row)
        if problem:
            placement_problems.append({"doc_id": row["doc_id"], "shelfmark": row["shelfmark"], "problem": problem})
        bib_line, bib_reason = public_bibliography_line(row.get("bibliography") or [])
        decisions.append({
            "doc_id": row["doc_id"],
            "shelfmark_raw": row["shelfmark"], "shelfmark": frag["shelfmark"],
            "card_source_internal": row.get("card_source"),
            "card_source_note_raw": row.get("card_source_note"), "card_source_note": frag["card_source_note"],
            "bibliography_first_raw": (row.get("bibliography") or [None])[0],
            "bibliography_line": bib_line, "bibliography_dropped_reason": bib_reason if bib_line is None else None,
            "ai_read_caveat_raw": (row.get("ai_read") or {}).get("caveat"),
            "machine_read_note": (frag["machine_read"] or {}).get("note"),
            "date_caveat_raw": row.get("date_caveat"), "date_caveat": (frag["date"] or {}).get("caveat"),
            "festival_day": frag["festival_day"],
        })
        if frag["card_source_note"] and INSTRUCTION_WORDING_RE.search(frag["card_source_note"]):
            warnings.append(f"{row['doc_id']}.card_source_note: instruction-like wording: {frag['card_source_note']!r}")
        if frag["date"] and frag["date"]["caveat"] and INSTRUCTION_WORDING_RE.search(frag["date"]["caveat"]):
            warnings.append(f"{row['doc_id']}.date.caveat: instruction-like wording: {frag['date']['caveat']!r}")
        if frag["card_source_public"] == "published scholarship":
            warnings.append(f"{row['doc_id']}: card_source_public is the generic 'published scholarship'")
    seed_ids = {row["doc_id"] for row in seed["fragments"]}
    included_ids = {row["doc_id"] for row in included}
    curated_not_included = [
        {"doc_id": doc_id, "day": entry["day"], "group": entry["group"],
         "why": "not in seed" if doc_id not in seed_ids else "row not included on the page"}
        for doc_id, entry in NAMED_PLACEMENTS.items() if doc_id not in included_ids
    ]
    return {
        "counts": {
            "seed_fragments": len(seed["fragments"]), "included": len(fragments), "excluded": len(excluded),
            "per_theme": per_theme,
            "machine_read_links": sum(1 for frag in fragments if frag["machine_read"]),
            "festival_days": per_day,
            "clusters_shipped": len(clusters), "clusters_seed": len(seed["clusters"]),
            "bibliography_lines": sum(1 for frag in fragments if frag["bibliography_line"]),
            "dated": sum(1 for frag in fragments if frag["date"]),
        },
        "excluded": excluded,
        "excluded_machine_only_seed": seed.get("excluded_machine_only", []),
        "not_on_site_seed": seed.get("not_on_site", []),
        "festival_placement": {
            "rejected_curated_entries": placement_problems,
            "curated_not_on_page": curated_not_included,
            "tech_notes_items_not_in_seed": LISTED_NOT_IN_SEED,
        },
        "clusters_dropped": [c["id"] for c in seed["clusters"] if c["id"] not in {x["id"] for x in clusters}],
        "decisions": decisions,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- validators (pure)


def walk_strings(obj: Any, path: str = "$") -> Iterable[Tuple[str, str]]:
    """Yield every string in a JSON-like object, dict keys included, with its path.

    :param obj: JSON-like value.
    :param path: Path of ``obj``.
    :return: Iterator of ``(path, string)``.
    """
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield f"{path}.{key}<key>", str(key)
            yield from walk_strings(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from walk_strings(value, f"{path}[{index}]")


def find_forbidden_terms(output: Dict[str, Any]) -> List[str]:
    """Find forbidden project names anywhere in the output (check a).

    :param output: Public JSON.
    :return: One message per hit.
    """
    return [f"forbidden term {match.group(0)!r} at {path}"
            for path, text in walk_strings(output) for match in [FORBIDDEN_TERMS_RE.search(text)] if match]


def find_internal_keys(output: Any, path: str = "$") -> List[str]:
    """Find internal seed fields that leaked into the output.

    :param output: Public JSON (or any sub-value).
    :param path: Path of ``output``.
    :return: One message per leaked key.
    """
    hits: List[str] = []
    if isinstance(output, dict):
        for key, value in output.items():
            if key in INTERNAL_KEYS:
                hits.append(f"internal key {key!r} at {path}")
            hits.extend(find_internal_keys(value, f"{path}.{key}"))
    elif isinstance(output, list):
        for index, value in enumerate(output):
            hits.extend(find_internal_keys(value, f"{path}[{index}]"))
    return hits


def validate_machine_reads(fragments: List[Dict[str, Any]]) -> List[str]:
    """Check machine-read blocks: no zero-agreed read, no "checked" wording (check b).

    :param fragments: Public fragments.
    :return: Problems found.
    """
    problems: List[str] = []
    for frag in fragments:
        read = frag.get("machine_read")
        if read is None:
            if frag.get("text_status") == "machine":
                problems.append(f"{frag['doc_id']}: text_status 'machine' without a machine_read")
            continue
        if not read.get("n_agreed") or read["n_agreed"] <= 0:
            problems.append(f"{frag['doc_id']}: machine_read with n_agreed == 0")
        for field in ("label", "note"):
            if read.get(field) and CHECK_CLAIM_RE.search(read[field]):
                problems.append(f"{frag['doc_id']}: machine_read.{field} claims checking: {read[field]!r}")
        if not frag.get("ai_read_doc_id") or f"doc={frag['ai_read_doc_id']}&" not in read.get("href", ""):
            problems.append(f"{frag['doc_id']}: machine_read.href does not use ai_read_doc_id")
    return problems


def is_human_source(label: Optional[str]) -> bool:
    """Tell whether a public card source names a human source.

    Accepted: the fixed labels in :data:`HUMAN_SOURCE_LABELS`, or a named scholar/edition
    (a string with a capitalised word and no machine/AI wording).

    :param label: ``card_source_public`` value.
    :return: True for a human source.
    """
    if not label or not label.strip():
        return False
    if label in HUMAN_SOURCE_LABELS:
        return True
    if NON_HUMAN_SOURCE_RE.search(label):
        return False
    return bool(re.search(r"\b[A-Z][a-zà-ÿ]+", label))


def validate_card_sources(fragments: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Check every card has a card_line and a human card source (check c).

    :param fragments: Public fragments.
    :return: ``(problems, distinct card_source_public values)``.
    """
    problems: List[str] = []
    for frag in fragments:
        if not (frag.get("card_line") or "").strip():
            problems.append(f"{frag['doc_id']}: empty card_line")
        if not is_human_source(frag.get("card_source_public")):
            problems.append(f"{frag['doc_id']}: card_source_public not a human source: {frag.get('card_source_public')!r}")
    return problems, sorted({str(frag.get("card_source_public")) for frag in fragments})


def validate_references(output: Dict[str, Any]) -> List[str]:
    """Check that every theme, genre, cluster, ref and day id resolves (check e).

    :param output: Public JSON.
    :return: Problems found.
    """
    problems: List[str] = []
    theme_ids = {theme["id"] for theme in output["themes"]}
    genre_ids = {genre["id"] for genre in output["genres"]}
    cluster_ids = {cluster["id"] for cluster in output["clusters"]}
    fragment_ids = {frag["doc_id"] for frag in output["fragments"]}
    day_ids = {day["id"] for day in output["festival"]["days"]}
    refs = output["refs"]
    for theme in output["themes"]:
        if theme["genre"] not in genre_ids:
            problems.append(f"theme {theme['id']}: genre {theme['genre']!r} missing")
        problems.extend(f"theme {theme['id']}: ref {ref!r} missing" for ref in theme["refs"] if ref not in refs)
        problems.extend(f"theme {theme['id']}: fragment {doc!r} missing"
                        for doc in theme["fragment_ids"] if doc not in fragment_ids)
        if not theme["fragment_ids"]:
            problems.append(f"theme {theme['id']}: no fragments")
    for cluster in output["clusters"]:
        missing = [doc for doc in cluster["members"] if doc not in fragment_ids]
        problems.extend(f"cluster {cluster['id']}: member {doc!r} missing" for doc in missing)
        if len(cluster["members"]) < 2:
            problems.append(f"cluster {cluster['id']}: fewer than 2 members")
    for frag in output["fragments"]:
        if frag["theme"] not in theme_ids:
            problems.append(f"{frag['doc_id']}: theme {frag['theme']!r} missing")
        if frag["genre"] not in genre_ids:
            problems.append(f"{frag['doc_id']}: genre {frag['genre']!r} missing")
        problems.extend(f"{frag['doc_id']}: cluster {cid!r} missing" for cid in frag["clusters"] if cid not in cluster_ids)
        day = frag.get("festival_day")
        if day:
            for day_id in [day["day"]] + list(day.get("span") or []):
                if day_id not in day_ids:
                    problems.append(f"{frag['doc_id']}: festival day {day_id!r} missing")
        if not (frag.get("image") or {}).get("url"):
            problems.append(f"{frag['doc_id']}: no image url")
    if len(fragment_ids) != len(output["fragments"]):
        problems.append("duplicate doc_ids in fragments")
    return problems


def validate_hedges(output: Dict[str, Any], seed_themes: Dict[str, Dict[str, Any]]) -> List[str]:
    """Check that themes keep their intro, hedges and contested positions verbatim (check f).

    :param output: Public JSON.
    :param seed_themes: Seed ``themes`` dict.
    :return: Problems found.
    """
    problems: List[str] = []
    for theme in output["themes"]:
        seed_theme = seed_themes[theme["id"]]
        if theme["intro"] != seed_theme["intro"]:
            problems.append(f"theme {theme['id']}: intro differs from the seed")
        if theme["hedges"] != list(seed_theme.get("hedges") or []):
            problems.append(f"theme {theme['id']}: hedges differ from the seed")
        if theme["contested"] != list(seed_theme.get("contested") or []):
            problems.append(f"theme {theme['id']}: contested positions differ from the seed")
        if seed_theme.get("contested") and not theme["hedges"]:
            problems.append(f"theme {theme['id']}: contested theme without hedges")
    return problems


def validate_size(payload: bytes, limit: int = MAX_BYTES) -> List[str]:
    """Check the serialised output size (check g).

    :param payload: Encoded JSON.
    :param limit: Maximum size in bytes.
    :return: Problems found.
    """
    return [] if len(payload) <= limit else [f"output is {len(payload)} bytes, over the {limit}-byte limit"]


def validate_images(results: Dict[str, Optional[int]]) -> List[str]:
    """Turn image HTTP results into problems (check d).

    :param results: url -> HTTP status (None when the request failed).
    :return: One problem per URL that did not return 200.
    """
    return [f"image {url} returned {status}" for url, status in results.items() if status != 200]


def run_validations(output: Dict[str, Any], seed: Dict[str, Any], payload: bytes,
                    image_results: Optional[Dict[str, Optional[int]]]) -> Dict[str, Any]:
    """Run every validator and collect the results.

    :param output: Public JSON.
    :param seed: Parsed seed JSON.
    :param payload: Encoded output.
    :param image_results: url -> status from :func:`check_images`, or None when not checked.
    :return: ``{"errors": [...], "checks": {name: [problems]}, "card_sources": [...]}``.
    """
    card_problems, card_sources = validate_card_sources(output["fragments"])
    checks = {
        "a_forbidden_terms": find_forbidden_terms(output),
        "a_internal_keys": find_internal_keys(output),
        "b_machine_reads": validate_machine_reads(output["fragments"]),
        "c_card_sources": card_problems,
        "d_images": validate_images(image_results) if image_results is not None else [],
        "e_references": validate_references(output),
        "f_hedges": validate_hedges(output, seed["themes"]),
        "g_size": validate_size(payload),
    }
    errors = [problem for problems in checks.values() for problem in problems]
    return {"errors": errors, "checks": checks, "card_sources": card_sources,
            "images_checked": image_results is not None}


# --------------------------------------------------------------------------- network and thumbnails


def make_session() -> Any:
    """Create an HTTP session with a browser-like User-Agent.

    :return: ``requests.Session``.
    """
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def image_status(session: Any, url: str) -> Optional[int]:
    """Return the HTTP status of an image: HEAD first, then a streamed GET that reads no body.

    :param session: ``requests.Session``.
    :param url: Image URL.
    :return: Status code, or None when the request itself failed.
    """
    import requests

    try:
        response = session.head(url, allow_redirects=True, timeout=30)
        if response.status_code == 200:
            return 200
        with session.get(url, stream=True, timeout=30) as streamed:
            return streamed.status_code
    except requests.RequestException:
        return None


def check_images(urls: List[str], status_fn: Callable[[str], Optional[int]],
                 pause_s: float = 0.1, sleep: Callable[[float], None] = time.sleep) -> Dict[str, Optional[int]]:
    """Check each distinct image URL once.

    :param urls: Image URLs (duplicates are checked once).
    :param status_fn: Function returning the HTTP status of a URL.
    :param pause_s: Pause between requests.
    :param sleep: Sleep function (injectable for tests).
    :return: url -> status.
    """
    results: Dict[str, Optional[int]] = {}
    for url in urls:
        if url in results:
            continue
        results[url] = status_fn(url)
        sleep(pause_s)
    return results


def make_thumbnail(image_bytes: bytes, width: int = THUMB_WIDTH, quality: int = THUMB_QUALITY) -> bytes:
    """Resize an image to a JPEG thumbnail of the given width (never upscaled).

    :param image_bytes: Source image file content.
    :param width: Target width in pixels.
    :param quality: JPEG quality.
    :return: JPEG bytes.
    """
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    if image.width > width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buffer.getvalue()


def write_thumbnails(fragments: List[Dict[str, Any]], out_dir: Path, session: Any) -> Tuple[Dict[str, str], List[str]]:
    """Download each card image once and write ``<out_dir>/<doc_id>.jpg`` (opt-in ``--thumbs``).

    :param fragments: Included seed rows.
    :param out_dir: Thumbnail directory (e.g. ``src/frontend/public/sukkot/thumbs``).
    :param session: ``requests.Session``.
    :return: ``(doc_id -> public thumb URL, failures)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs: Dict[str, str] = {}
    failures: List[str] = []
    for frag in fragments:
        url = frag["images"]["ok_url"]
        response = session.get(url, timeout=60)
        if response.status_code != 200:
            failures.append(f"{frag['doc_id']}: {url} returned {response.status_code}")
            continue
        (out_dir / f"{frag['doc_id']}.jpg").write_bytes(make_thumbnail(response.content))
        thumbs[frag["doc_id"]] = f"{THUMB_URL_PREFIX}{frag['doc_id']}.jpg"
        time.sleep(REQUEST_PAUSE_S)
    return thumbs, failures


# --------------------------------------------------------------------------- CLI


def encode_output(output: Dict[str, Any]) -> bytes:
    """Serialise the public JSON the way it is written to disk.

    :param output: Public JSON.
    :return: UTF-8 bytes.
    """
    return (json.dumps(output, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line options.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :return: Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED, help="Seed JSON.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Public JSON to write.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Build report (not shipped).")
    parser.add_argument("--check-images", action="store_true", help="HEAD/GET every card image; fail on non-200.")
    parser.add_argument("--refresh-reads", action="store_true",
                        help="Re-fetch every card's machine read from the public API (paced, read-only).")
    parser.add_argument("--thumbs", type=Path, default=None,
                        help="Opt-in: download images and write 320 px JPEG thumbnails to this directory.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Build, validate and write the page JSON and the build report.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :return: Process exit code (0 on success, 1 when any validation fails).
    """
    args = parse_args(argv)
    seed = json.loads(args.seed.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    thumbs: Dict[str, str] = {}
    thumb_failures: List[str] = []
    session = make_session() if (args.check_images or args.thumbs or args.refresh_reads) else None
    # select_fragments returns the seed's own row dicts, so refreshing them here
    # is seen by build_output's selection below.
    reads_report = refresh_reads(select_fragments(seed)[0], session) if args.refresh_reads else None
    if args.thumbs:
        included, _excluded = select_fragments(seed)
        thumbs, thumb_failures = write_thumbnails(included, args.thumbs, session)
    output, report = build_output(seed, generated_at, thumbs)
    image_results = None
    if args.check_images:
        urls = [frag["image"]["url"] for frag in output["fragments"]]
        image_results = check_images(urls, lambda url: image_status(session, url))
    payload = encode_output(output)
    validation = run_validations(output, seed, payload, image_results)
    report.update({
        "generated_at": generated_at, "seed": str(args.seed), "out": str(args.out),
        "reads_refresh": reads_report,
        "output_bytes": len(payload), "validation": validation,
        "image_check": image_results, "thumbnails": {"written": len(thumbs), "failures": thumb_failures},
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if validation["errors"]:
        print(f"BUILD FAILED: {len(validation['errors'])} validation error(s); {args.out} not written.")
        for error in validation["errors"]:
            print(f"  - {error}")
        print(f"Report: {args.report}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    counts = report["counts"]
    print(f"Wrote {args.out} ({len(payload)} bytes): {counts['included']} fragments, "
          f"{len(output['themes'])} themes, {len(output['clusters'])} clusters, "
          f"{counts['machine_read_links']} machine-read links; images checked: {image_results is not None}.")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
