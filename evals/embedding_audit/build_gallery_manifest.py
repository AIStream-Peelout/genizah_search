"""Build the full-gallery manifest: one row per imaged fragment of the merged catalogue (labels, URLs, flags).

The probe benchmark (``image_manifest_v1.json``, 5,176 images) was built around known joins; real discovery needs
every imaged fragment. This script streams the re-merged catalogue (HDA ``merged_shelfmarks.jsonl``) once and writes
``<AUDIT_ROOT>/gallery_v1/``:

* ``manifest.jsonl`` — sorted by ``canonical_id``. Per row: ``canonical_id``, ``shelfmark``, ``institution``,
  ``collection`` (``institution|collection``, the v1 manifest string), ``holding`` (``holding_key`` of
  ``discover_joins_scribes.py``: the id's leading institution tokens), ``image_url`` (the one to embed: the merge's
  preferred-source first URL, except that a benchmark fragment keeps its benchmark URL when the record still lists it,
  so benchmark and gallery vectors stay comparable), ``image_url_2`` (kept for a later verso pass), ``n_images``,
  ``in_benchmark``, ``benchmark_url_status``, ``join_group`` (connected component of the known joins: FJP clean join
  lists + PGP shared documents of 2-6 fragments, recomputed for the whole catalogue exactly as
  ``build_image_manifest.py`` does), ``link_group`` (component of every known link — also the FJP prefix-bug suspects,
  PGP documents of any size, loosely matched FJP join lists, FJP list members found by :class:`Resolver` and FJP
  items carried by several records — pairs inside one are never proposed as discoveries), ``dup_group`` (one fragment
  catalogued as several records: overlapping image lists or the same holding + shelfmark tokens),
  ``container_record`` / ``image_overlap_ids`` (box-level records whose image list overlaps >= 3 other records; the
  non-container records sharing an image file), ``image_live`` / ``image_url_dead`` (from ``url_check.json``: the
  embedded URL answered a HEAD request, or was dead and a live alternative of the record replaced it; false = no live
  image, the Colab run skips the row),
  PGP fields (``pgpids``, ``pgp_types``, ``pgp_tags``, ``pgp_years``, ``pgp_scribe`` / ``pgp_scribes`` from certain
  ``Scribe`` relations, ``documentary``), KTIV fields (``ktiv_sys_num``, ``material``, ``script_type``,
  ``script_style``, ``script_region``), ``fjp_shelfmarks``, ``shelf_norm`` (shelfmark tokens, for same-shelfmark
  variants), and ``tower_ms_split`` / ``tower_training_ms`` (the fragment's KTIV manuscript supplied lines to the
  fine-tuned tower's ``train`` / ``val`` split, or was an I2 ``test`` manuscript — see ``hf_image/dataset_v1``;
  ``tower_training_ms_twin``: a duplicate record of a training-manuscript fragment).
* ``known_joins.json`` — the pair lists by source (gallery-internal, both sides imaged) and the join groups.
* ``manifest_stats.json`` (also copied to ``results/gallery_v1_manifest_stats.json``) — coverage and how many known
  join pairs / groups fall inside the gallery, same-holding vs cross-holding.

Run (JSON only, no model; seconds): ``nice -n 10 <HDA .venv>/bin/python build_gallery_manifest.py``; add
``--check-urls`` to redo the HEAD sweep of the embedded image URLs first (network only, ~5 min; writes
``url_check.json``, which later builds reuse).
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

HDA = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
sys.path.insert(0, str(HDA))
from src.datasets.document_models.genizah_document import _merged_image_urls  # noqa: E402

from build_image_manifest import first, shelf_key  # noqa: E402
from discover_joins_scribes import DOCUMENTARY_TYPES, UnionFind, holding_key, pgp_years  # noqa: E402
from embed_utils import AUDIT_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
MERGED = HDA / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
FJP_ALL = HDA / "src/datasets/raw_data/cairo_genizah/fjp_all/merged_princeton_friedberger_all_documents_final.json"
PGP_RELATIONS = Path.home() / "Documents/GitHub/ktiv-scraper/princeton/pgp_person_document_relations.csv"
TRAIN_LINES = AUDIT_ROOT / "hf_image" / "dataset_v1" / "lines.jsonl"
BENCH_LABELS = AUDIT_ROOT / "hf_image" / "dataset_v1" / "fragment_labels.json"
OUT = AUDIT_ROOT / "gallery_v1"
PGP_JOIN_MAX = 6  # build_image_manifest: larger shared-document groups are composite dossiers, not physical joins


def shelf_norm(shelfmark: str) -> str:
    """Shelfmark tokens for spotting two records of one shelfmark (letters and numbers split, leading zeros dropped).

    ``"T-S 13J1.1"`` and ``"T-S 13 J 1.1"`` both give ``"t_s_13_j_1_1"``; ``"T-S 1.11"`` and ``"T-S 11.1"`` differ.

    :param shelfmark: Display shelfmark.
    :returns: Normalised key.
    :rtype: str
    """
    toks = re.findall(r"[a-z]+|\d+", (shelfmark or "").lower())
    return "_".join(str(int(t)) if t.isdigit() else t for t in toks)


def shelf_tokens(shelfmark: str) -> Tuple[str, ...]:
    """Shelfmark tokens as in :func:`shelf_norm` (letters / numbers split, leading zeros dropped), as a tuple.

    :param shelfmark: Any shelfmark string.
    :returns: Tokens.
    :rtype: Tuple[str, ...]
    """
    return tuple(str(int(t)) if t.isdigit() else t for t in re.findall(r"[a-z]+|\d+", (shelfmark or "").lower()))


class Resolver:
    """FJP shelfmark -> canonical id, beyond the exact FJP-source lookup of ``build_image_manifest``.

    The exact resolver only knows the shelfmarks of records that carry an FJP source, so an FJP join list that names
    a fragment catalogued only by KTIV / PGP / a library ("Cambridge, CUL: T-S AS 77.234" vs the KTIV-only record
    ``Cambridge_CUL_T_S_AS_77_234``) leaves that member unresolved and the join unknown. Fallbacks, each accepted only
    when it names exactly one record of the whole catalogue: the shelf key of a canonical id, then the shelfmark
    proper (the part after the colon) as a token suffix of a display shelfmark (at least two tokens, a letter and a
    number).
    """

    def __init__(self, shelf_to_cid: Dict[str, str], cid_keys: Dict[str, Set[str]],
                 display: Dict[str, str]) -> None:
        """Index the catalogue.

        :param shelf_to_cid: FJP shelf key -> canonical id (exact resolver).
        :param cid_keys: ``shelf_key(canonical_id)`` -> canonical ids (all records).
        :param display: canonical id -> display shelfmark (all records).
        """
        self.shelf_to_cid, self.cid_keys = shelf_to_cid, cid_keys
        self.suffix: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
        for cid, sm in display.items():
            t = shelf_tokens(sm)
            for k in range(len(t)):
                self.suffix[t[k:]].add(cid)

    def resolve(self, shelfmark: str) -> Tuple[Optional[str], str]:
        """Resolve one FJP shelfmark.

        :param shelfmark: e.g. ``"Cambridge, CUL:  T-S F1(2).47"``.
        :returns: (canonical id or None, method: ``fjp_exact`` | ``cid_exact`` | ``display_suffix`` | ``none``).
        :rtype: Tuple[Optional[str], str]
        """
        key = shelf_key(shelfmark)
        if key in self.shelf_to_cid:
            return self.shelf_to_cid[key], "fjp_exact"
        hits = self.cid_keys.get(key, set())
        if len(hits) == 1:
            return next(iter(hits)), "cid_exact"
        t = shelf_tokens(shelfmark.split(":", 1)[1] if ":" in shelfmark else shelfmark)
        if len(t) >= 2 and any(x.isdigit() for x in t) and any(x.isalpha() for x in t):
            hits = self.suffix.get(t, set())
            if len(hits) == 1:
                return next(iter(hits)), "display_suffix"
        return None, "none"


def read_catalogue() -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]],
                              Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    """Stream the merged catalogue once.

    :returns: (rows for imaged records keyed by cid, FJP shelf key -> cid over ALL records, PGP id -> cids from
        ``pgpids`` (all records), PGP id -> cids from ``pgpids`` + document ids (all records), KTIV sys -> cids,
        ``shelf_key(cid)`` -> cids (all records), cid -> display shelfmark (all records)).
    :rtype: tuple
    """
    rows: Dict[str, dict] = {}
    shelf_to_cid: Dict[str, str] = {}
    pgp_to_cids: Dict[str, Set[str]] = defaultdict(set)
    pgp_any: Dict[str, Set[str]] = defaultdict(set)
    sys_to_cids: Dict[str, Set[str]] = defaultdict(set)
    cid_keys: Dict[str, Set[str]] = defaultdict(set)
    display: Dict[str, str] = {}
    with open(MERGED, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid = rec["canonical_id"]
            cid_keys[shelf_key(cid)].add(cid)
            display[cid] = rec.get("shelfmark_display") or cid
            src = rec.get("sources") or {}
            for fjp in src.get("fjp") or []:
                if fjp.get("shelf_mark"):
                    shelf_to_cid.setdefault(shelf_key(fjp["shelf_mark"]), cid)
            for pid in rec.get("pgpids") or []:
                pgp_to_cids[str(pid)].add(cid)
            docs = (src.get("pgp") or {}).get("documents") or []
            all_pgp = {str(p) for p in rec.get("pgpids") or []} | {str(d.get("pgpid")) for d in docs if d.get("pgpid")}
            for pid in all_pgp:
                pgp_any[pid].add(cid)
            ktiv = src.get("ktiv") or {}
            sys_num = str(ktiv.get("sys_num") or "")
            if sys_num:
                sys_to_cids[sys_num].add(cid)
            urls = [u for u in _merged_image_urls(rec.get("images") or {}) if u and not u.endswith("/")]
            if not urls:
                continue
            attrs = {}
            for entry in ktiv.get("scholarly_entries") or []:  # build_image_manifest's KTIV attribute rule
                sub = entry.get("subsections") or {}
                wc = sub.get("wrinting_characteristics") or {}
                for field, key in (("script_style", "script_style"), ("region_of_script", "script_region"),
                                   ("script_type", "script_type")):
                    val = first(wc.get(field))
                    if val and key not in attrs:
                        attrs[key] = val
                mat = first((sub.get("physical_description") or {}).get("material_type"))
                if mat and "material" not in attrs:
                    attrs["material"] = mat
            collection = f"{rec.get('institution')}|{rec.get('collection')}"
            types = sorted({d.get("type") or "" for d in docs} - {""})
            rows[cid] = {
                "canonical_id": cid,
                "shelfmark": rec.get("shelfmark_display") or cid,
                "institution": rec.get("institution"),
                "collection": collection,
                "holding": holding_key(cid, collection),
                "sources_present": rec.get("sources_present") or [],
                "urls": urls,
                "pgpids": sorted(all_pgp),
                "pgp_types": types,
                "pgp_tags": sorted({t.strip() for d in docs for t in (d.get("tags") or "").split(",") if t.strip()}),
                "pgp_years": pgp_years(docs),
                "documentary": bool(set(types) & DOCUMENTARY_TYPES),
                "ktiv_sys_num": sys_num or None,
                **{k: attrs.get(k) for k in ("material", "script_type", "script_style", "script_region")},
                "fjp_shelfmarks": sorted({f.get("shelf_mark") for f in src.get("fjp") or [] if f.get("shelf_mark")}),
                "shelf_norm": shelf_norm(rec.get("shelfmark_display") or ""),
            }
    return rows, shelf_to_cid, pgp_to_cids, pgp_any, sys_to_cids, cid_keys, display


def fjp_pairs(shelf_to_cid: Dict[str, str]) -> Tuple[Set[tuple], Set[tuple], Dict[str, Set[str]], Set[str], Counter,
                                                      List[Tuple[str, List[str], bool]]]:
    """FJP join pairs over the whole catalogue, with ``build_image_manifest``'s prefix-bug suspect rule.

    :param shelf_to_cid: FJP shelf key -> canonical id.
    :returns: (all pairs, clean pairs, raw join lists keyed by FJP shelf key, prefix-bug suspect keys, counters,
        the raw join items as (owner shelfmark, joined shelfmarks, owner is a prefix-bug suspect)).
    :rtype: tuple
    """
    fjp = json.loads(FJP_ALL.read_text())
    items = list(fjp.values() if isinstance(fjp, dict) else fjp)
    del fjp
    lists: Dict[str, frozenset] = {}
    for item in items:
        joined = ((item.get("joins_data") or {}).get("joinedManuscripts")) or []
        if joined:
            lists[shelf_key(item.get("shelf_mark") or "")] = frozenset(shelf_key(j.get("shelfmark") or "") for j in joined)
    by_list = defaultdict(list)
    for key, js in lists.items():
        by_list[js].append(key)
    suspects = {x for group in by_list.values() if len(group) > 1 for x in group for y in group
                if x != y and y.startswith(x) and x not in lists.get(y, ())}
    all_pairs: Set[tuple] = set()
    clean: Set[tuple] = set()
    unresolved = Counter()
    join_items: List[Tuple[str, List[str], bool]] = []
    for item in items:
        joined = ((item.get("joins_data") or {}).get("joinedManuscripts")) or []
        if not joined:
            continue
        suspect = shelf_key(item.get("shelf_mark") or "") in suspects
        join_items.append((item.get("shelf_mark") or "", [j.get("shelfmark") or "" for j in joined], suspect))
        cids = []
        for sm in [item.get("shelf_mark") or ""] + [j.get("shelfmark") or "" for j in joined]:
            cid = shelf_to_cid.get(shelf_key(sm))
            if cid:
                cids.append(cid)
            else:
                unresolved["fjp_shelfmarks_unresolved"] += 1
        cids = sorted(set(cids))
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                all_pairs.add((cids[i], cids[j]))
                if not suspect:
                    clean.add((cids[i], cids[j]))
    unresolved["fjp_items_with_join_list"] = len(lists)
    unresolved["fjp_prefix_bug_suspects"] = len(suspects)
    return all_pairs, clean, {k: set(v) for k, v in lists.items()}, suspects, unresolved, join_items


def fjp_resolved_extra(join_items: List[Tuple[str, List[str], bool]], resolver: Resolver, rows: Dict[str, dict],
                       known: Set[tuple]) -> Tuple[Dict[tuple, Tuple[str, bool]], Counter]:
    """Gallery pairs of FJP join lists in which at least one member resolves only through :class:`Resolver` fallbacks.

    Every item is used, also one whose own shelfmark is unresolved or unimaged (its joined members are still one
    join). A pair counts as a clean positive only when both members resolve exactly (FJP source or canonical id) and
    the item is not a prefix-bug suspect; a display-suffix resolution makes it a link only.

    :param join_items: Output of :func:`fjp_pairs` (owner shelfmark, joined shelfmarks, suspect).
    :param resolver: Shelfmark resolver.
    :param rows: Gallery rows (imaged records).
    :param known: Pairs already known (skipped).
    :returns: ({(cid, cid): (matched key ``<method>:<shelf key>``, not a clean positive)}, resolution counts).
    :rtype: Tuple[Dict[tuple, Tuple[str, bool]], Counter]
    """
    out: Dict[tuple, Tuple[str, bool]] = {}
    methods: Counter = Counter()
    for owner, joined, suspect in join_items:
        res = []
        for sm in [owner] + joined:
            cid, how = resolver.resolve(sm)
            methods[how] += 1
            if cid in rows:
                res.append((cid, how, sm))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                (a, ha, sa), (b, hb, sb) = res[i], res[j]
                if a == b or (ha == "fjp_exact" and hb == "fjp_exact"):
                    continue  # exact pairs are fjp_pairs' business
                pair = tuple(sorted((a, b)))
                if pair in known:
                    continue
                how, sm = (ha, sa) if ha != "fjp_exact" else (hb, sb)
                not_clean = suspect or "display_suffix" in (ha, hb)
                prev = out.get(pair)
                if prev is None or (prev[1] and not not_clean):  # a clean resolution wins over a link-only one
                    out[pair] = (f"{how}:{shelf_key(sm)}", not_clean)
    return out, methods


def shared_fjp_item_pairs(rows: Dict[str, dict]) -> Set[tuple]:
    """Imaged records that carry the same FJP item (same FJP shelf key).

    FGP catalogues many joins as ONE item whose shelfmark lists every part ("Cambridge CUL: T-S 16.326 Manchester:
    P 49"); the merge attaches that item to each part's record, and its ``joinedManuscripts`` list is often empty, so
    ``build_image_manifest`` never sees the join. The same key on two records can also be one shelfmark catalogued
    twice. Either way the pair is not a discovery: these pairs are links (never proposed, not calibration negatives).

    :param rows: Gallery rows.
    :returns: Sorted id pairs.
    :rtype: Set[tuple]
    """
    by_key: Dict[str, Set[str]] = defaultdict(set)
    for cid, r in rows.items():
        for sm in r["fjp_shelfmarks"]:
            if shelf_key(sm):
                by_key[shelf_key(sm)].add(cid)
    return group_pairs(by_key.values(), 2, None)


def image_overlap(rows: Dict[str, dict], urls: Dict[str, List[str]], container_min: int
                  ) -> Tuple[Dict[str, List[str]], Set[str], Counter]:
    """Records that list the same image file.

    Box-level records ("Cambridge CUL: T-S AS 60", 542 images) carry the photographs of their items, and FGP join
    items carry one composite photograph on every part's record. A record whose image list overlaps that of
    ``container_min`` or more other records is a *container*: its first image shows some other record's fragment, so
    it is never proposed. Below that, two records sharing a file are one fragment (or one composite) catalogued twice.

    :param rows: Gallery rows.
    :param urls: cid -> all image URLs of the record.
    :param container_min: Overlapping records from which a record counts as a container.
    :returns: ({cid: overlapping non-container cids} for non-container records, container cids, overlap-degree counts).
    :rtype: Tuple[Dict[str, List[str]], Set[str], Counter]
    """
    by_url: Dict[str, Set[str]] = defaultdict(set)
    for cid, us in urls.items():
        for u in set(us):
            by_url[u].add(cid)
    nb: Dict[str, Set[str]] = defaultdict(set)
    for cs in by_url.values():
        if len(cs) > 1:
            for c in cs:
                nb[c] |= cs - {c}
    containers = {c for c, v in nb.items() if len(v) >= container_min}
    overlap = {c: sorted(v - containers) for c, v in nb.items() if c not in containers and v - containers}
    return overlap, containers, Counter(min(len(v), 10) for v in nb.values())


def check_urls(rows: Dict[str, dict], urls: Dict[str, List[str]], out: Path, threads: int = 32,
               max_alternatives: int = 6) -> None:
    """HEAD-check the URL each record would embed (and, when it is dead, up to ``max_alternatives`` more).

    Network only (no image download). About 5 % of first images return 404 on the public bucket, clustered by box
    (one 2,000-fragment shard of the manifest order had 29 % dead), so the manifest marks them and embeds a live
    alternative when the record has one.

    :param rows: Gallery rows.
    :param urls: cid -> image URLs in embedding order.
    :param out: JSON written: {cid: {url, status, alt}}.
    :param threads: Concurrent requests.
    :param max_alternatives: Further URLs tried for a dead first one.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import requests

    local = threading.local()

    def head(u: str) -> object:
        """Status code of a HEAD request (exception name after 3 attempts).

        :param u: URL.
        :returns: Status code or error name.
        """
        if getattr(local, "s", None) is None:
            local.s = requests.Session()
        err = "error"
        for _ in range(3):
            try:
                return local.s.head(u, timeout=20).status_code
            except requests.RequestException as exc:  # transient network error: retried, then reported
                err = type(exc).__name__
        return err

    def one(cid: str) -> Tuple[str, dict]:
        """Check one record.

        :param cid: Canonical id.
        :returns: (cid, {url, status, alt}).
        :rtype: Tuple[str, dict]
        """
        us = urls[cid]
        st = head(us[0])
        alt = None
        if st == 404:
            alt = next((u for u in us[1:1 + max_alternatives] if head(u) == 200), None)
        return cid, {"url": us[0], "status": st, "alt": alt}

    with ThreadPoolExecutor(threads) as pool:
        res = dict(pool.map(one, sorted(rows)))
    out.write_text(json.dumps(res))
    print(f"url check: {Counter(v['status'] for v in res.values())}, live alternatives "
          f"{sum(1 for v in res.values() if v['alt'])} -> {out}", flush=True)


def group_pairs(groups: Iterable[Set[str]], lo: int = 2, hi: Optional[int] = None) -> Set[tuple]:
    """All pairs inside groups whose size is within [lo, hi].

    :param groups: Sets of canonical ids.
    :param lo: Minimum group size.
    :param hi: Maximum group size (None = unbounded).
    :returns: Sorted id pairs.
    :rtype: Set[tuple]
    """
    out = set()
    for g in groups:
        c = sorted(g)
        if lo <= len(c) and (hi is None or len(c) <= hi):
            out |= {(c[i], c[j]) for i in range(len(c)) for j in range(i + 1, len(c))}
    return out


def loose_fjp_pairs(rows: Dict[str, dict], lists: Dict[str, Set[str]], suspects: Set[str],
                    known: Set[tuple]) -> Dict[tuple, Tuple[str, bool]]:
    """Gallery pairs that an FJP join list names under a shelfmark the exact resolver missed.

    Port of ``discover_joins_scribes.loose_fjp_pairs`` for the full gallery, stricter because the gallery is ten
    times larger: a joined key matches a fragment's key exactly, or by its longest underscore-aligned suffix of at
    least three tokens that contains a letter token and is at least 8 characters long (a bare ``"123_45"`` would hit
    unrelated collections).

    :param rows: Gallery rows.
    :param lists: FJP shelf key -> joined shelf keys.
    :param suspects: FJP shelf keys flagged by the prefix-bug rule (their lists may belong to a longer shelfmark).
    :param known: Pairs already known exactly (skipped).
    :returns: {(cid, cid): (matched key, whether the list came from a prefix-bug suspect item)}.
    :rtype: Dict[tuple, Tuple[str, bool]]
    """
    exact: Dict[str, Set[str]] = defaultdict(set)
    for cid, r in rows.items():
        keys = {shelf_key(s) for s in r["fjp_shelfmarks"]} | {shelf_key(r["shelfmark"]), shelf_key(cid)}
        for k in keys:
            if len(k) >= 4:
                exact[k].add(cid)
    out: Dict[tuple, Tuple[str, bool]] = {}
    for cid, r in rows.items():
        joined = [(jk, shelf_key(s) in suspects) for s in r["fjp_shelfmarks"] for jk in lists.get(shelf_key(s), ())]
        for jk, suspect in sorted(joined, key=lambda t: t[1]):  # a clean list's match wins over a suspect one
            parts = jk.split("_")
            for start in range(len(parts)):
                suffix = "_".join(parts[start:])
                tail = parts[start:]
                ok = start == 0 or (len(tail) >= 3 and len(suffix) >= 8 and any(t.isalpha() for t in tail))
                if suffix in exact and ok:
                    for other in exact[suffix]:
                        if other != cid:
                            pair = tuple(sorted((cid, other)))
                            if pair not in known:
                                out.setdefault(pair, (suffix, suspect))
                    break
    return out


def components(ids: List[str], pairs: Iterable[tuple], prefix: str) -> Dict[str, str]:
    """Connected components of id pairs, named ``<prefix>NNNNN`` in order of each group's smallest id.

    :param ids: All gallery ids (sorted).
    :param pairs: Id pairs.
    :param prefix: Group-id prefix.
    :returns: cid -> group id, for fragments in a group of >= 2.
    :rtype: Dict[str, str]
    """
    pos = {c: i for i, c in enumerate(ids)}
    uf = UnionFind(len(ids))
    for a, b in pairs:
        uf.union(pos[a], pos[b])
    members: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(ids)):
        members[uf.find(i)].append(i)
    groups = sorted((m for m in members.values() if len(m) > 1), key=min)
    return {ids[i]: f"{prefix}{k:05d}" for k, m in enumerate(groups) for i in m}


def tower_splits(sys_to_cids: Dict[str, Set[str]]) -> Dict[str, str]:
    """Which fragments belong to a KTIV manuscript of the tower fine-tune's line dataset, by split.

    :param sys_to_cids: KTIV sys number -> canonical ids.
    :returns: cid -> ``train`` | ``val`` | ``test`` (train wins over val over test).
    :rtype: Dict[str, str]
    """
    split_of: Dict[str, str] = {}
    rank = {"train": 0, "val": 1, "test": 2}
    with open(TRAIN_LINES, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["ms"] not in split_of or rank[r["split"]] < rank[split_of[r["ms"]]]:
                split_of[r["ms"]] = r["split"]
    out: Dict[str, str] = {}
    for ms, split in split_of.items():
        for cid in sys_to_cids.get(ms, ()):
            if cid not in out or rank[split] < rank[out[cid]]:
                out[cid] = split
    return out


def pair_stats(pairs: Iterable[tuple], hold: Dict[str, str]) -> dict:
    """Pair count split into same- and cross-holding.

    :param pairs: Id pairs.
    :param hold: cid -> holding.
    :returns: {pairs, same_holding, cross_holding}.
    :rtype: dict
    """
    pairs = list(pairs)
    cross = sum(hold[a] != hold[b] for a, b in pairs)
    return {"pairs": len(pairs), "same_holding": len(pairs) - cross, "cross_holding": cross}


def group_stats(groups: Dict[str, str], hold: Dict[str, str]) -> dict:
    """Group count, size distribution and how many span several holdings.

    :param groups: cid -> group id.
    :param hold: cid -> holding.
    :returns: Stats dict.
    :rtype: dict
    """
    members: Dict[str, List[str]] = defaultdict(list)
    for c, g in groups.items():
        members[g].append(c)
    sizes = Counter(len(m) for m in members.values())
    return {"groups": len(members), "fragments": len(groups),
            "cross_holding_groups": sum(len({hold[c] for c in m}) > 1 for m in members.values()),
            "size_distribution": {str(k): v for k, v in sorted(sizes.items())}}


def duplicate_groups(rows: Dict[str, dict], overlap: Dict[str, List[str]], containers: Set[str]) -> Dict[str, str]:
    """Components of exact duplicate-record relations (one fragment catalogued as several records).

    Relations: the image lists overlap (non-container records only, see :func:`image_overlap`), or the same holding
    and the same shelfmark tokens. Containers are left out so that a box record cannot merge all of its items.

    :param rows: Gallery rows.
    :param overlap: Output of :func:`image_overlap`.
    :param containers: Container records.
    :returns: cid -> ``dgNNNNN`` for records in a group of >= 2.
    :rtype: Dict[str, str]
    """
    pairs: Set[tuple] = set()
    for c, others in overlap.items():
        pairs |= {tuple(sorted((c, o))) for o in others}
    by_shelf: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for c, r in rows.items():
        if c not in containers and r["shelf_norm"] and any(ch.isdigit() for ch in r["shelf_norm"]):
            by_shelf[(r["holding"], r["shelf_norm"])].add(c)
    pairs |= group_pairs(by_shelf.values(), 2, None)
    return components(sorted(rows), pairs, "dg")


def main() -> None:
    """Read everything, label, and write the manifest, join lists and stats."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-urls", action="store_true",
                        help="HEAD-check every embedded image URL first (network, no download; ~5 min) -> url_check.json")
    parser.add_argument("--url-check", type=Path, default=OUT / "url_check.json",
                        help="URL liveness file to apply (rows whose URL was checked dead get a live alternative or "
                             "image_live = false); ignored when missing")
    parser.add_argument("--container-min", type=int, default=3)
    args = parser.parse_args()
    rows, shelf_to_cid, pgp_to_cids, pgp_any, sys_to_cids, cid_keys, display = read_catalogue()
    ids = sorted(rows)
    hold = {c: rows[c]["holding"] for c in ids}
    print(f"imaged records: {len(ids)}", flush=True)

    bench_ids = json.loads(BENCH_LABELS.read_text())["ids"]
    bench_set = set(bench_ids)
    bench_url = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())["images"]
    url_status: Counter = Counter()
    bstatus: Dict[str, str] = {}
    all_urls: Dict[str, List[str]] = {}
    for cid in ids:
        urls = rows[cid].pop("urls")
        if cid in bench_set and cid in bench_url:
            b = bench_url[cid]
            bstatus[cid] = "first" if urls[0] == b else "listed_not_first" if b in urls else "not_listed"
            if bstatus[cid] == "listed_not_first":  # embed the benchmarked photograph, keep the merge's first as url_2
                urls = [b] + [u for u in urls if u != b]
            url_status[bstatus[cid]] += 1
        all_urls[cid] = urls
    OUT.mkdir(parents=True, exist_ok=True)
    if args.check_urls:
        check_urls(rows, all_urls, args.url_check)
    check = json.loads(args.url_check.read_text()) if args.url_check.exists() else {}
    live: Dict[str, Optional[bool]] = {}
    dead_url: Dict[str, str] = {}
    for cid in ids:
        c = check.get(cid)
        if not c or c["url"] != all_urls[cid][0]:
            live[cid] = None  # not checked (or the record changed since the check)
        elif c["status"] == 200:
            live[cid] = True
        elif c.get("alt") and c["alt"] in all_urls[cid]:
            dead_url[cid] = all_urls[cid][0]
            all_urls[cid] = [c["alt"]] + [u for u in all_urls[cid] if u != c["alt"]]
            live[cid] = True
        else:
            live[cid] = False

    fjp_all, fjp_clean, lists, suspects, fjp_counts, join_items = fjp_pairs(shelf_to_cid)
    pgp_join = group_pairs(pgp_to_cids.values(), 2, PGP_JOIN_MAX)
    pgp_link = group_pairs(pgp_any.values(), 2, None)
    inside = lambda pairs: {p for p in pairs if p[0] in rows and p[1] in rows}  # noqa: E731
    g_fjp_all, g_fjp_clean, g_pgp, g_pgp_link = inside(fjp_all), inside(fjp_clean), inside(pgp_join), inside(pgp_link)
    known = g_fjp_clean | g_pgp
    loose = loose_fjp_pairs(rows, lists, suspects, g_fjp_all | g_pgp_link)
    extra, resolve_counts = fjp_resolved_extra(join_items, Resolver(shelf_to_cid, cid_keys, display), rows,
                                               g_fjp_all | g_pgp_link)
    for pair, val in extra.items():  # the resolver's matches complete the loose list (same positive rule)
        if pair not in loose or (loose[pair][1] and not val[1]):
            loose[pair] = val
    shared_item = shared_fjp_item_pairs(rows) - g_fjp_all - g_pgp_link - set(loose)
    links = g_fjp_all | g_pgp_link | set(loose) | shared_item
    join_group = components(ids, known, "jg")
    link_group = components(ids, links, "lg")
    overlap, containers, overlap_degree = image_overlap(rows, all_urls, args.container_min)
    dup_group = duplicate_groups(rows, overlap, containers)

    # PGP scribes (certain "Scribe" relations, through the record's own pgpids, first slug wins: build_image_manifest)
    scribes: Dict[str, List[str]] = defaultdict(list)
    with open(PGP_RELATIONS, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["relation"].strip() == "Scribe":
                for cid in pgp_to_cids.get(r["pgpid"].strip(), ()):
                    if cid in rows and r["person_slug"] not in scribes[cid]:
                        scribes[cid].append(r["person_slug"])

    split = tower_splits(sys_to_cids)
    dup_members: Dict[str, List[str]] = defaultdict(list)
    for c, g in dup_group.items():
        dup_members[g].append(c)
    twin_train = {c for m in dup_members.values() if any(split.get(x) == "train" for x in m)
                  for c in m if split.get(c) != "train"}
    hosts, n_img = Counter(), Counter()
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for cid in ids:
            r = rows[cid]
            urls = all_urls[cid]
            status = bstatus.get(cid)
            row = {"canonical_id": cid, "shelfmark": r["shelfmark"], "institution": r["institution"],
                   "collection": r["collection"], "holding": r["holding"], "image_url": urls[0],
                   "image_url_2": urls[1] if len(urls) > 1 else None, "n_images": len(urls),
                   "image_live": live[cid], "image_url_dead": dead_url.get(cid),
                   "in_benchmark": status is not None, "benchmark_url_status": status,
                   "benchmark_url": bench_url.get(cid) if status == "not_listed" else None,
                   "join_group": join_group.get(cid), "link_group": link_group.get(cid),
                   "dup_group": dup_group.get(cid), "container_record": cid in containers,
                   "image_overlap_ids": overlap.get(cid, []),
                   "pgp_scribe": scribes[cid][0] if scribes.get(cid) else None, "pgp_scribes": scribes.get(cid, []),
                   "tower_ms_split": split.get(cid), "tower_training_ms": split.get(cid) == "train",
                   "tower_training_ms_twin": cid in twin_train}
            row.update({k: v for k, v in r.items() if k not in row})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            hosts[re.sub(r"^https?://([^/]+)/.*$", r"\1", urls[0])] += 1
            n_img[min(len(urls), 5)] += 1

    def as_list(pairs: Iterable[tuple]) -> List[list]:
        """Sorted JSON-friendly pair list.

        :param pairs: Id pairs.
        :returns: Sorted list of [a, b].
        :rtype: List[list]
        """
        return sorted([list(p) for p in pairs])
    group_members: Dict[str, List[str]] = defaultdict(list)
    for c, g in join_group.items():
        group_members[g].append(c)
    (OUT / "known_joins.json").write_text(json.dumps({
        "pairs": {"fjp_clean": as_list(g_fjp_clean), "fjp_suspect": as_list(g_fjp_all - g_fjp_clean),
                  "pgp": as_list(g_pgp), "pgp_large_or_document_only": as_list(g_pgp_link - g_pgp),
                  "fjp_loose": [[a, b, k, sus] for (a, b), (k, sus) in sorted(loose.items())],
                  "fjp_shared_item": as_list(shared_item)},
        "join_groups": dict(sorted(group_members.items())),
        "definitions": {
            "known_join": "fjp_clean | pgp (build_image_manifest rules: FJP join lists minus prefix-bug suspects; "
                          f"fragments sharing a PGP pgpid in groups of 2-{PGP_JOIN_MAX})",
            "link": "every known link: fjp_clean | fjp_suspect | pgp | pgp_large_or_document_only | fjp_loose | "
                    "fjp_shared_item",
            "fjp_loose": "[a, b, matched key, not a clean positive]; joins the exact shelfmark resolver missed: "
                         "suffix matches of imaged records' own lists (key = shelf key) and Resolver fallbacks over "
                         "every FJP join item (key = 'cid_exact:...' or 'display_suffix:...'). The flag is true for "
                         "lists of prefix-bug suspect items and display-suffix matches (links only); "
                         "discover_joins_v2 uses the others as positives",
            "fjp_shared_item": "records carrying the same FJP item (FGP's one-item-per-join cataloguing, or one "
                               "shelfmark catalogued twice): links only",
        }}, ensure_ascii=False))

    # statistics
    in_bench = lambda pairs: [p for p in pairs if p[0] in bench_set and p[1] in bench_set]  # noqa: E731
    is_live = lambda c: live[c] is not False  # noqa: E731
    in_live = lambda pairs: [p for p in pairs if is_live(p[0]) and is_live(p[1])]  # noqa: E731
    train_frag = {c for c, s in split.items() if s == "train" and c in rows}
    scribe_counts = Counter(s[0] for c, s in scribes.items() if s)
    documentary = [c for c in ids if rows[c]["documentary"]]
    clean_loose = [p for p, (_, sus) in loose.items() if not sus]
    stats = {
        "built_from": {"merged_catalogue": str(MERGED), "merged_mtime": MERGED.stat().st_mtime, "fjp": str(FJP_ALL),
                       "pgp_relations": str(PGP_RELATIONS), "url_check": str(args.url_check) if check else None},
        "imaged_fragments": len(ids),
        "image_url_check": {"checked": sum(v is not None for v in live.values()),
                            "live": sum(v is True for v in live.values()) - len(dead_url),
                            "dead_first_url_live_alternative_used": len(dead_url),
                            "no_live_image": sum(v is False for v in live.values()),
                            "no_live_image_in_benchmark": sum(1 for c in ids if live[c] is False and c in bench_set),
                            "no_live_image_in_known_join_groups": sum(1 for c in ids if live[c] is False and c in join_group)},
        "image_hosts": dict(hosts.most_common()),
        "n_images_per_fragment_capped_at_5": {str(k): v for k, v in sorted(n_img.items())},
        "holdings": len(set(hold.values())),
        "top_holdings": Counter(hold.values()).most_common(25),
        "benchmark": {"ids": len(bench_ids), "in_gallery": len(bench_set & set(ids)),
                      "missing_from_gallery": sorted(bench_set - set(ids))[:50],
                      "url_status": dict(url_status)},
        "fjp": dict(fjp_counts, resolver_methods_over_all_join_items=dict(resolve_counts)),
        "known_joins_inside_gallery": {
            "fjp_clean": pair_stats(g_fjp_clean, hold), "pgp_2_to_6": pair_stats(g_pgp, hold),
            "known_join_pairs_total": pair_stats(known, hold),
            "fjp_suspect_extra": pair_stats(g_fjp_all - g_fjp_clean, hold),
            "pgp_large_or_document_only_extra": pair_stats(g_pgp_link - g_pgp, hold),
            "fjp_loose_extra": pair_stats(loose, hold),
            "fjp_loose_extra_from_clean_lists": pair_stats(clean_loose, hold),
            "fjp_loose_extra_from_resolver": pair_stats([p for p in loose if ":" in loose[p][0]], hold),
            "fjp_shared_item_extra": pair_stats(shared_item, hold),
            "all_links": pair_stats(links, hold),
            "join_groups": group_stats(join_group, hold), "link_groups": group_stats(link_group, hold),
            "fragments_with_known_partner": len(join_group),
            "discovery_positives_both_live": pair_stats(in_live(known | set(clean_loose)), hold),
        },
        "duplicates": {"container_records": len(containers), "container_min_overlapping_records": args.container_min,
                       "records_with_image_overlap": len(overlap),
                       "overlap_degree_capped_at_10": {str(k): v for k, v in sorted(overlap_degree.items())},
                       "dup_groups": group_stats(dup_group, hold)},
        "known_join_pairs_with_a_side_not_imaged": {
            "fjp_clean": len(fjp_clean) - len(g_fjp_clean), "pgp_2_to_6": len(pgp_join) - len(g_pgp)},
        "known_joins_inside_benchmark_for_reference": pair_stats(in_bench(known), hold),
        "scribes": {"labelled_fragments": sum(1 for s in scribes.values() if s),
                    "with_several_scribes": sum(1 for s in scribes.values() if len(s) > 1),
                    "scribes": len(scribe_counts), "scribes_ge5": sum(1 for n in scribe_counts.values() if n >= 5),
                    "fragments_of_scribes_ge5": sum(n for n in scribe_counts.values() if n >= 5),
                    "documentary_fragments": len(documentary),
                    "documentary_unlabelled": sum(1 for c in documentary if not scribes.get(c))},
        "attributes_present": {k: sum(1 for c in ids if rows[c][k]) for k in ("material", "script_type", "script_style", "script_region")},
        "pgp_dated_fragments": sum(1 for c in ids if rows[c]["pgp_years"]),
        "tower_line_dataset": {"fragments_by_split": dict(Counter(s for c, s in split.items() if c in rows)),
                               "training_ms_fragments": len(train_frag),
                               "duplicate_records_of_training_ms_fragments": len(twin_train),
                               "known_join_pairs_touching_training_ms": sum(1 for a, b in known if a in train_frag or b in train_frag),
                               "benchmark_fragments_in_training_ms": len(train_frag & bench_set)},
    }
    (OUT / "manifest_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "gallery_v1_manifest_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("top_holdings", "built_from")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
