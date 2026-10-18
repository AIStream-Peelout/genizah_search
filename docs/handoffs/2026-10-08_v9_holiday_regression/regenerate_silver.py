#!/usr/bin/env python3
"""Rebuild the holiday-regression silver relevance sets and score a ranked run against the seed eval.

Read-only and stdlib-only. It never writes to Elasticsearch: the only request it sends is ``POST <index>/_search``.

Two subcommands:

``build``
    Dump ``doc_id / description / alt_titles / document_type / semantic_eligible`` of one index (``search_after``
    paging, sorted on ``doc_id``; about 78k records / 50 MB for a v9-sized index, so run it once per new index) or read
    an existing dump (``--from-dump``: a JSON list of those rows), apply the relevance rules below and write
    ``{set: {strict, broad, desc_strict, ineligible_strict}}``. With ``--seed`` it also prints the set sizes next to the
    seed's sizes for the same index name. The rules are a verbatim copy of the diagnosis script (provenance only: the
    diagnosis scratchpad, ``holiday_eval/silver.py`` and its 2026-10-08 dumps, is deleted); run on those dumps they
    reproduced the seed's v8 and v9 lists exactly. v8 and v9 lists are already in the seed, so only the new index needs
    ``build``.

``score``
    Score a run file against ``holiday_eval_seed.json``. Relevance = the union of every index's list in the seed for
    that set and rule (v8 and v9 today), plus any ``--silver`` file built for a new index: a record counts when any
    index's catalogue text names the topic. The seed's v8 top-10 lists are re-scored on that same union, so the
    "drop vs v8" metrics and the paired comparison are not biased by the wider relevance set. Prints P@10 per rule next
    to the seed's v8 / v9 / fusion numbers, the group means, the alt_titles-only share, distinct text groups (when the
    run carries hashes), a paired comparison with v8 (mean difference, standard error, bootstrap 95% CI,
    wins/losses/ties) and the gate. ``NOT MEASURED`` counts as a failure in the overall gate verdict.

Run file format (JSON object keyed by the exact ``queries[].query`` strings of the seed, including the Hebrew ones
and both "simchat torah" and "Simchat Torah"; each value ranked best first, at least 10 entries)::

    {"Pesach": ["Cambridge_CUL_T_S_NS_271_77", ...], ...}
    {"Pesach": [{"doc_id": "...", "semantic_text_hash": "..."}, ...], ...}   # hashes enable B8 / D8

Credentials come from the environment and are never printed: ``ES_URL`` (default ``http://localhost:9200``),
``ES_USER``, ``ES_PASSWORD`` (both optional).

Examples::

    python3 regenerate_silver.py build --index genizah_merged_v10 --out silver_v10.json --seed holiday_eval_seed.json
    python3 regenerate_silver.py score --seed holiday_eval_seed.json --run run_v10.json --silver silver_v10.json \
        --gate beta
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Dict, List, Optional, Pattern, Sequence, Set, Tuple

DUMP_FIELDS = ["doc_id", "description", "alt_titles", "document_type", "semantic_eligible"]
VARIANTS = ("strict", "broad", "desc_strict")
BARE_NAME_FORMS = ("english", "transliterated", "hebrew", "ashkenazi", "typo")

# ---------------------------------------------------------------------------------------------------------------
# relevance rules (verbatim from the diagnosis' holiday_eval/silver.py)
# ---------------------------------------------------------------------------------------------------------------
HB = r"א-ת"
PFX = r"(?:[ובלהשמכ])?"  # optional one-letter Hebrew prefix


def fold(text: Optional[str]) -> str:
    """Fold text the way the rules expect: NFKD, combining marks dropped, ayin/alef marks -> ``'``, lowercased.

    :param text: Raw text (``None`` is treated as empty).
    :returns: Folded text.
    :rtype: str
    """
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[ʿʾ’‘ʼ`´]", "'", t)
    return t.lower()


def heb(word: str) -> str:
    """Hebrew term with an optional one-letter prefix and Hebrew-letter boundaries.

    :param word: Hebrew phrase (regex allowed).
    :returns: Regex string.
    :rtype: str
    """
    return rf"(?<![{HB}]){PFX}{word}(?![{HB}])"


# A festival mention preceded (within 45 chars, no ';' or '|') by one of these is a tractate / law-code / commentary
# title ("Talmud Bavli: Rosh ha-Shanah", "Hilkhot ha-Rif: Sukkah"): counted by ``broad`` only.
TRACTATE_CTX = re.compile(
    r"(talmud|mishna|tosefta|yerushalmi|bavli|hilkhot ha-rif|\brif\b|mishneh torah|rambam|"
    r"mishnaic|commentar|halakhot gedolot|"
    r"תלמוד|משנה|תוספתא|ירושלמי|הלכות|הל'|מסכת|רי\"ף|רמב\"ם)[^;|]{0,45}$"
)
# Patronymics ("b. Pesah", "Pesah b."): never a festival mention.
NAME_CTX_BEFORE = re.compile(r"(\bb\.|\bben|\bbt\.|\bbint|\br\.|בן|בר)\s*$")
NAME_CTX_AFTER = re.compile(r"^\s*(b\.|ben\b|bar\b|בן\b)")
# Hebrew שבועות meaning "oaths" (ארבע שבועות, שבועת ...): not Shavuot.
OATH_CTX = re.compile(r"(ארבע|בארבע|שבועת)\s*$")

# holiday -> (festival regex, extra regex counted by ``broad`` only: same-name tractates)
HOLIDAYS: Dict[str, Tuple[str, Optional[str]]] = {
    "pesach": (
        r"\bpesa(?:c)?h\b|\bpessah\b|\bpassover\b|\bhaggad(?:ah|a|ot)\b|" + heb("פסח"),
        r"\bpesahim\b|" + heb("פסחים"),
    ),
    "rosh_hashanah": (
        r"\brosh\s*-?\s*ha[-\s]?shana?h?\b|\bjewish new year\b|" + heb(r"ראש\s+השנה"),
        None,
    ),
    "yom_kippur": (
        r"\byom\s+(?:ha-?)?kipp?urim?\b|\bkippur\b|\bday of atonement\b|"
        + heb(r"יום\s+ה?כי?פו?רי?ם?") + r"|יוה\"כ|יו\"כ|יוה״כ|יו״כ",
        None,
    ),
    "sukkot": (
        r"\bsukk?o(?:t|th|s)\b|\bsucc?o(?:t|th)\b|\btabernacles\b|\bfeast of booths\b|"
        r"\bhosh'?an(?:ot|oth|a rabba|ah rabbah|a rabbah)\b|"
        + heb("סוכות") + "|" + heb("סכות") + "|" + heb("הושענות") + "|" + heb(r"הושענ[אה]\s+רבה"),
        r"\bsukk?ah?\b|" + heb("סוכה"),
    ),
    "simchat_torah": (
        r"\bsim(?:c)?h(?:a)?t\s+torah?\b|\bsimchas\s+torah?\b|" + heb(r"שמחת\s+תורה"),
        None,
    ),
    "shavuot": (
        r"\bshavu'?oth?\b|\bshabu'?oth?\b|\bshavuos\b|\bpentecost\b|\bfeast of weeks\b|" + heb("שבועות"),
        None,
    ),
    "purim": (r"\bpurim\b|" + heb("פורים"), None),
    "hanukkah": (
        r"\b(?:hanukkah|hanukka|hanukah|hanuka|hannukah|hannuka|hannukka|chanukah|chanukkah|channukah|hanouka)\b|"
        + heb("חנוכה"),
        None,
    ),
    "tisha_bav": (
        r"\btish'?ah?\s*(?:be|b)\s*[-']?\s*'?av\b|\bninth of av\b|\b9th of av\b|\bfast of av\b|"
        + heb(r"תשעה\s*-?\s*באב"),
        None,
    ),
}

# control topic -> ([regexes that must ALL match the folded text], document_type that also counts as relevant)
CONTROL: Dict[str, Tuple[List[str], Optional[str]]] = {
    "ketubah": ([r"\bketubb?ah?\b|\bketubb?ot\b|\bmarriage (?:contract|deed)\b|" + heb("כתובה")], "ketubah"),
    "india_merchant": ([r"\bindia\b|\bindian\b|" + heb("הודו"),
                        r"letter|merchant|trader|trade|business|commerc|shipment|account|" + heb("מכתב")], None),
    "maimonides_responsum": ([r"maimonides|\brambam\b|moses b\.? maimon|moshe b\.? maimon|רמב\"ם|רמב״ם|משה בן מימון",
                              r"respons|teshuv|fatw|question|query|שאלה|שו\"ת|תשובה|תשובות"], None),
    "karaite_marriage": ([r"kara'?it|qara'?it|" + heb("קראי") + "|" + heb("קראים"),
                          r"marriage|ketub|betroth|nuptial|wedding|engagement|נישואין|כתובה|אירוסין|שידוכין"], None),
    "medical": ([r"medical|medicine|medicament|prescription|remed(?:y|ies)|recipe|pharmac|\bdrugs?\b|"
                 r"רפואה|רפואות|רפואי|מרשם"], "medical"),
    "saadia_bible_commentary": ([r"saadi?a|saadya|sa'adi?a|sa'adya|se'adyah?|סעדיה|רס\"ג|רס״ג",
                                 r"commentar|tafsir|translation|exegesis|פירוש|פרוש|תפסיר|תרגום"], None),
    "divorce": ([r"divorc|bill of repudiation|repudiat|גט פיטורין|גירושין|גרושין|גט אשה"], None),
    "court_fustat": ([r"\bcourt\b|bet din|beit din|bet-din|בית דין|legal deed|deposition|testimony",
                      r"fustat|fostat|old cairo|\bmisr\b|פסטאט|פוסטאט"], None),
    "amulet": ([r"amulet|talisman|" + heb("קמיע") + "|" + heb("קמע") + "|" + heb("קמיעות") + r"|\bmagical\b"], None),
}


def festival_hits(text: str, rx: Pattern, key: str) -> Tuple[bool, bool]:
    """Classify the festival mentions in one folded text.

    :param text: Folded text (description, or description and alt_titles joined by ``" || "``).
    :param rx: Compiled festival regex.
    :param key: Holiday key (Shavuot has the extra oath rule).
    :returns: (has a festival mention outside tractate context, has a mention inside tractate context).
    :rtype: Tuple[bool, bool]
    """
    strict = tract = False
    for m in rx.finditer(text):
        before = text[max(0, m.start() - 60):m.start()]
        after = text[m.end():m.end() + 8]
        if NAME_CTX_BEFORE.search(before) or NAME_CTX_AFTER.search(after):
            continue
        if key == "shavuot" and re.match(rf"{PFX}שבועות", m.group(0)):
            if OATH_CTX.search(before) or TRACTATE_CTX.search(before):
                continue
        if TRACTATE_CTX.search(before):
            tract = True
        else:
            strict = True
    return strict, tract


def build_sets(rows: Sequence[dict]) -> Dict[str, Dict[str, List[str]]]:
    """Apply every rule to one index dump.

    ``desc_strict`` = the topic is named in ``description`` (outside tractate context); ``strict`` = named in
    ``description`` or ``alt_titles`` (outside tractate context); ``broad`` = ``strict`` plus tractate-context
    mentions plus the same-name tractates (Pesahim, Sukkah). Controls: ``strict`` = ``broad`` = all regexes match
    description + alt_titles (ketubah: not in tractate context) or ``document_type`` equals the listed type.

    :param rows: Dump rows with ``doc_id`` (or ``_id``), ``description``, ``alt_titles``, ``document_type`` and,
        where the index has it, ``semantic_eligible``.
    :returns: ``{set: {strict, broad, desc_strict, ineligible_strict}}`` with sorted doc id lists;
        ``ineligible_strict`` lists strict records with ``semantic_eligible`` false (empty when the field is absent).
    :rtype: Dict[str, Dict[str, List[str]]]
    """
    prepared = []
    for r in rows:
        did = r.get("doc_id") or r["_id"]
        desc = fold(r.get("description") or "")
        alts = fold(" || ".join(r.get("alt_titles") or []))
        prepared.append((did, desc, desc + " || " + alts, r.get("document_type"), r.get("semantic_eligible")))
    out: Dict[str, Dict[str, List[str]]] = {}
    for key, (frx, trx) in HOLIDAYS.items():
        rx = re.compile(frx)
        trx_c = re.compile(trx) if trx else None
        sets: Dict[str, List[str]] = {v: [] for v in VARIANTS}
        for did, desc, full, _dt, _el in prepared:
            s_desc, _ = festival_hits(desc, rx, key)
            s_full, t_full = festival_hits(full, rx, key)
            extra = bool(trx_c and trx_c.search(full))
            if s_desc:
                sets["desc_strict"].append(did)
            if s_full:
                sets["strict"].append(did)
            if s_full or t_full or extra:
                sets["broad"].append(did)
        out[key] = sets
    for key, (rxs, dtype) in CONTROL.items():
        cr = [re.compile(x) for x in rxs]
        sets = {v: [] for v in VARIANTS}
        for did, desc, full, dt, _el in prepared:
            hit_full = all(c.search(full) for c in cr)
            if key == "ketubah" and hit_full:
                hit_full = any(not TRACTATE_CTX.search(full[max(0, m.start() - 60):m.start()])
                               for m in cr[0].finditer(full))
            if all(c.search(desc) for c in cr):
                sets["desc_strict"].append(did)
            if hit_full or (dtype and dt == dtype):
                sets["strict"].append(did)
                sets["broad"].append(did)
        out[key] = sets
    ineligible = {did for did, *_rest, el in prepared if el is False}
    for sets in out.values():
        for v in VARIANTS:
            sets[v] = sorted(set(sets[v]))
        sets["ineligible_strict"] = sorted(d for d in sets["strict"] if d in ineligible)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Elasticsearch (read-only)
# ---------------------------------------------------------------------------------------------------------------
def es_search(base_url: str, index: str, body: dict, timeout: int = 120) -> dict:
    """POST one ``_search`` request (the only call this script makes).

    :param base_url: Elasticsearch URL, e.g. ``http://localhost:9200``.
    :param index: Index name.
    :param body: Search body.
    :param timeout: Socket timeout in seconds.
    :returns: Parsed response.
    :rtype: dict
    """
    headers = {"Content-Type": "application/json"}
    user, password = os.environ.get("ES_USER"), os.environ.get("ES_PASSWORD")
    if user:
        headers["Authorization"] = "Basic " + base64.b64encode(f"{user}:{password or ''}".encode()).decode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/{index}/_search", data=json.dumps(body).encode(),
                                 method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def dump_index(base_url: str, index: str, page_size: int = 5000) -> List[dict]:
    """Fetch the rule fields of every record of an index with ``search_after`` (no scroll, no PIT).

    :param base_url: Elasticsearch URL.
    :param index: Index name.
    :param page_size: Hits per request.
    :returns: Source dicts, each with ``_id`` added.
    :rtype: List[dict]
    """
    rows: List[dict] = []
    after = None
    while True:
        body = {"size": page_size, "_source": DUMP_FIELDS, "sort": [{"doc_id": "asc"}, {"_doc": "asc"}],
                "query": {"match_all": {}}, "track_total_hits": False}
        if after:
            body["search_after"] = after
        hits = es_search(base_url, index, body)["hits"]["hits"]
        if not hits:
            return rows
        for h in hits:
            src = dict(h["_source"])
            src["_id"] = h["_id"]
            rows.append(src)
        after = hits[-1]["sort"]


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------
def relevant_sets(seed: dict, extra_silvers: Sequence[dict]) -> Dict[str, Dict[str, Set[str]]]:
    """Union of every index's list per set and rule (seed indexes plus freshly built ``--silver`` files).

    :param seed: Parsed seed eval.
    :param extra_silvers: Parsed outputs of ``build``.
    :returns: ``{set: {rule: ids}}`` plus ``{set: {"alt_only": strict - desc_strict}}``.
    :rtype: Dict[str, Dict[str, Set[str]]]
    """
    rel: Dict[str, Dict[str, Set[str]]] = {}
    for key, per_index in seed["silver"].items():
        if not isinstance(per_index, dict):
            continue  # "note"
        rel[key] = {v: set() for v in VARIANTS}
        for lists in per_index.values():
            for v in VARIANTS:
                rel[key][v].update(lists.get(v) or [])
    for silver in extra_silvers:
        for key, lists in silver["sets"].items():
            for v in VARIANTS:
                rel.setdefault(key, {vv: set() for vv in VARIANTS})[v].update(lists.get(v) or [])
    for key in rel:
        rel[key]["alt_only"] = rel[key]["strict"] - rel[key]["desc_strict"]
    return rel


def read_run(path: str) -> Dict[str, List[dict]]:
    """Read a run file; plain doc id lists become ``{"doc_id": id}`` dicts.

    :param path: JSON file mapping query string to a ranked list.
    :returns: ``{query: [{"doc_id", optional "semantic_text_hash"}]}``.
    :rtype: Dict[str, List[dict]]
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {q: [x if isinstance(x, dict) else {"doc_id": x} for x in hits] for q, hits in raw.items()}


def p_at(ids: Sequence[str], rel: Set[str], k: int = 10) -> float:
    """Precision at k (missing ranks count as non-relevant).

    :param ids: Ranked doc ids.
    :param rel: Relevant ids.
    :param k: Cutoff.
    :returns: P@k.
    :rtype: float
    """
    return sum(1 for d in ids[:k] if d in rel) / k


def paired_vs_v8(rows: Sequence[dict], variant: str = "strict", n_resamples: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of the run with v8 over queries (P@10 differences, run minus v8).

    :param rows: Per-query rows of :func:`score` (each with ``p10`` and ``v8_p10``).
    :param variant: Relevance rule.
    :param n_resamples: Bootstrap resamples over queries.
    :param seed: RNG seed (fixed, so the printout is reproducible).
    :returns: ``{n, mean_diff, se, ci95, wins, losses, ties}``.
    :rtype: dict
    """
    diffs = [r["p10"][variant] - r["v8_p10"][variant] for r in rows]
    n = len(diffs)
    rng = random.Random(seed)
    boot = sorted(mean(rng.choice(diffs) for _ in range(n)) for _ in range(n_resamples))
    return {"n": n, "mean_diff": round(mean(diffs), 3), "se": round(stdev(diffs) / math.sqrt(n), 3),
            "ci95": [round(boot[int(0.025 * n_resamples)], 3), round(boot[int(0.975 * n_resamples) - 1], 3)],
            "wins": sum(1 for d in diffs if d > 1e-9), "losses": sum(1 for d in diffs if d < -1e-9),
            "ties": sum(1 for d in diffs if abs(d) <= 1e-9)}


def score(seed: dict, run: Dict[str, List[dict]], rel: Dict[str, Dict[str, Set[str]]]) -> dict:
    """Score a run on every seed query and compute the gate metrics.

    The seed's v8 top-10 lists are re-scored on the same relevance union (``v8_p10``), so the "drop vs v8" metrics and
    the paired comparison use one relevance set for both sides.

    :param seed: Parsed seed eval.
    :param run: Output of :func:`read_run`.
    :param rel: Output of :func:`relevant_sets`.
    :returns: ``{"per_query": [...], "metrics": {...}, "paired_vs_v8": {...}}``.
    :rtype: dict
    """
    per_query = []
    for q in seed["queries"]:
        hits = run.get(q["query"])
        row = {"query": q["query"], "set": q["set"], "group": q["group"], "form": q["form"],
               "missing": hits is None or len(hits) < 10}
        ids = [h["doc_id"] for h in hits or []]
        row["p10"] = {v: p_at(ids, rel[q["set"]][v]) for v in VARIANTS}
        row["v8_p10"] = {v: p_at(q["top10"]["v8"], rel[q["set"]][v]) for v in VARIANTS}
        strict_hits = [d for d in ids[:10] if d in rel[q["set"]]["strict"]]
        row["strict_hits10"] = len(strict_hits)
        row["alt_only_hits10"] = sum(1 for d in strict_hits if d in rel[q["set"]]["alt_only"])
        hashes = [h.get("semantic_text_hash") for h in (hits or [])[:10]]
        row["distinct_text_groups10"] = len(set(hashes)) if hashes and all(hashes) else None
        row["seed_p10"] = q["p10"]
        per_query.append(row)
    hol = [r for r in per_query if r["group"] == "holiday"]
    ctl = [r for r in per_query if r["group"] == "control"]
    m: Dict[str, Optional[float]] = {}
    for v in VARIANTS:
        m[f"holiday.{v}.P@10"] = round(mean(r["p10"][v] for r in hol), 3)
        m[f"control.{v}.P@10"] = round(mean(r["p10"][v] for r in ctl), 3)
    drops = []
    for key in sorted({r["set"] for r in hol}):
        rs = [r for r in hol if r["set"] == key]
        drops.append(mean(r["v8_p10"]["strict"] for r in rs) - mean(r["p10"]["strict"] for r in rs))
    m["holiday.max_set_drop_vs_v8.strict.P@10"] = round(max(drops), 3)
    bare = [r for r in hol if r["form"] in BARE_NAME_FORMS]
    m["holiday.bare_name_max_drop_vs_v8.strict.P@10"] = round(
        max(r["v8_p10"]["strict"] - r["p10"]["strict"] for r in bare), 3)
    n_strict = sum(r["strict_hits10"] for r in hol)
    m["holiday.alt_only_share@10"] = round(sum(r["alt_only_hits10"] for r in hol) / n_strict, 3) if n_strict else None
    groups = [r["distinct_text_groups10"] for r in hol]
    m["holiday.distinct_text_groups@10"] = round(mean(groups), 2) if all(g is not None for g in groups) else None
    m["queries_missing"] = sum(1 for r in per_query if r["missing"])
    paired = {f"holiday.{v}.P@10": paired_vs_v8(hol, v) for v in ("strict", "desc_strict")}
    return {"per_query": per_query, "metrics": m, "paired_vs_v8": paired}


def check_gate(metrics: Dict[str, Optional[float]], gate: List[dict]) -> List[Tuple[str, str, str]]:
    """Compare metrics with one gate of the seed's ``acceptance`` section.

    :param metrics: ``score(...)["metrics"]``.
    :param gate: List of ``{"id", "metric", "op", "value", ...}``.
    :returns: ``[(id, verdict, detail)]`` with verdict ``PASS`` / ``FAIL`` / ``NOT MEASURED``.
    :rtype: List[Tuple[str, str, str]]
    """
    ops = {">=": lambda a, b: a >= b - 1e-9, "<=": lambda a, b: a <= b + 1e-9, "==": lambda a, b: abs(a - b) < 1e-9}
    out = []
    for c in gate:
        val = metrics.get(c["metric"])
        if val is None:
            out.append((c["id"], "NOT MEASURED", f"{c['metric']} (needs data this run does not carry)"))
            continue
        ok = ops[c["op"]](val, c["value"])
        out.append((c["id"], "PASS" if ok else "FAIL", f"{c['metric']} = {val} (need {c['op']} {c['value']})"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> None:
    """``build`` subcommand: dump (or load) an index and write its silver sets.

    :param args: Parsed CLI arguments.
    """
    if args.from_dump:
        with open(args.from_dump, encoding="utf-8") as fh:
            rows = json.load(fh)
        source = f"dump:{os.path.basename(args.from_dump)}"
    else:
        rows = dump_index(os.environ.get("ES_URL", "http://localhost:9200"), args.index)
        source = "elasticsearch"
    sets = build_sets(rows)
    out = {"index": args.index, "source": source, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_records": len(rows), "n_semantic_ineligible": sum(1 for r in rows if r.get("semantic_eligible") is False),
           "sets": sets}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    seed_lists = {}
    if args.seed:
        with open(args.seed, encoding="utf-8") as fh:
            seed_lists = {k: v.get(args.index) for k, v in json.load(fh)["silver"].items() if isinstance(v, dict)}
    print(f"{args.index}: {len(rows)} records -> {args.out}")
    print(f"{'set':24s} {'desc_strict':>11s} {'strict':>7s} {'broad':>7s} {'inelig':>7s}   seed (same index)")
    for key, s in sets.items():
        ref = seed_lists.get(key)
        cmp = ""
        if ref:
            same = all(sorted(ref.get(v) or []) == s[v] for v in VARIANTS)
            cmp = (f"{len(ref['desc_strict']):6d}/{len(ref['strict']):6d}/{len(ref['broad']):6d}  "
                   f"{'identical' if same else 'DIFFERS'}")
        print(f"{key:24s} {len(s['desc_strict']):11d} {len(s['strict']):7d} {len(s['broad']):7d} "
              f"{len(s['ineligible_strict']):7d}   {cmp}")


def cmd_score(args: argparse.Namespace) -> None:
    """``score`` subcommand: score a run file and print the gate.

    :param args: Parsed CLI arguments.
    """
    with open(args.seed, encoding="utf-8") as fh:
        seed = json.load(fh)
    extra = []
    for path in args.silver or []:
        with open(path, encoding="utf-8") as fh:
            extra.append(json.load(fh))
    res = score(seed, read_run(args.run), relevant_sets(seed, extra))
    print(f"{'query':32s} {'form':14s}  strict run | v8  v9  fus   broad run | v8  v9  fus   desc run | v8  v9  fus")
    for r in res["per_query"]:
        s = r["seed_p10"]
        cells = "   ".join(f"{r['p10'][v]:.1f} | {s[v]['v8']:.1f} {s[v]['v9']:.1f} {s[v]['fusion']:.1f}" for v in VARIANTS)
        flag = "  MISSING" if r["missing"] else ""
        print(f"{r['query'][:32]:32s} {r['form']:14s}  {cells}{flag}")
    print("\nmetrics:")
    for k, v in res["metrics"].items():
        print(f"  {k:48s} {v}")
    print("\npaired vs v8 over the 44 holiday queries (run minus v8, v8 re-scored on the same relevance union):")
    for k, p in res["paired_vs_v8"].items():
        print(f"  {k:28s} diff {p['mean_diff']:+.3f}  SE {p['se']:.3f}  bootstrap 95% CI "
              f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  W/L/T {p['wins']}/{p['losses']}/{p['ties']}")
    for name in ([args.gate] if args.gate else []):
        print(f"\ngate '{name}':")
        verdicts = check_gate(res["metrics"], seed["acceptance"][name]["criteria"])
        for cid, verdict, detail in verdicts:
            print(f"  {cid:4s} {verdict:13s} {detail}")
        failed = [cid for cid, verdict, _ in verdicts if verdict != "PASS"]
        print(f"  overall: {'PASS' if not failed else 'FAIL (' + ', '.join(failed) + ')'}"
              f"  [NOT MEASURED counts as FAIL]")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse arguments and run a subcommand.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="rebuild silver sets for one index")
    b.add_argument("--index", required=True, help="index name (also the key used to compare with the seed)")
    b.add_argument("--out", required=True, help="output JSON path")
    b.add_argument("--from-dump", help="read a JSON list of dump rows instead of querying Elasticsearch")
    b.add_argument("--seed", help="holiday_eval_seed.json, to print the seed's sizes for the same index")
    b.set_defaults(func=cmd_build)
    s = sub.add_parser("score", help="score a run against the seed eval")
    s.add_argument("--seed", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--silver", action="append", help="output of build for the run's index (repeatable)")
    s.add_argument("--gate", choices=("beta", "default"), help="print this acceptance gate")
    s.add_argument("--json-out", help="write per-query results and metrics here")
    s.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
