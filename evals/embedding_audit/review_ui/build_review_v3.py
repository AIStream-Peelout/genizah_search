"""Assemble the v3 review page: held-out subjects, their test queries, PGP tag curation, join and scribe candidates.

Writes ``review_ui/v3/index.html`` (the page template ``review_v3_template.html`` with the review data inlined) and
``review_ui/v3/thumbs/*.jpg`` (backdrop-masked fragment thumbnails for the join and scribe candidates). Isaac's verdicts
are stored by the page in the artifact's ``db`` collection ``review_v3`` (one doc per item id) and read back with
ArtifactData.
"""

import argparse
import csv
import io
import json
import re
import time
import urllib.request
from pathlib import Path

from PIL import Image

from embed_utils import AUDIT_ROOT
from image_features import IMG_DIR, fragment_mask

HERE = Path(__file__).parent
EVAL = AUDIT_ROOT / "v3" / "eval"
DISC = AUDIT_ROOT / "results" / "discovery_v1"
DISC2 = AUDIT_ROOT / "results" / "discovery_v2"
GALLERY_MANIFEST = AUDIT_ROOT / "gallery_v1" / "manifest.jsonl"
CURATION = HERE.parent / "results" / "pgp_tag_curation.csv"


def thumb(cid: str, out_dir: Path, max_side: int = 520) -> str:
    """Write a backdrop-masked, downscaled JPEG of one fragment and return its published path.

    :param cid: Canonical fragment id (image file stem).
    :param out_dir: Thumbnail directory.
    :param max_side: Longest side in px.
    :returns: Path relative to the page (``thumbs/<cid>.jpg``).
    :rtype: str
    """
    out = out_dir / f"{cid.replace('/', '_')}.jpg"
    if not out.exists():
        img = Image.open(IMG_DIR / f"{cid.replace('/', '_')}.jpg").convert("RGB")
        masked = fragment_mask(img)[0]
        masked.thumbnail((max_side, max_side))
        masked.save(out, quality=78)
    return f"thumbs/{out.name}"


def subjects_and_queries() -> tuple:
    """Held-out subjects with their review facts, and every test query scored against them.

    :returns: (subjects, queries).
    :rtype: tuple
    """
    held = json.loads((EVAL / "held_out_subjects.json").read_text())["subjects"]
    allq = json.loads((EVAL / "subject_queries.json").read_text())["subjects"]
    subjects, queries = [], []
    for s in held:
        subjects.append({"id": s["id"], "kind": s["kind"], "reason": s["reason"], "names_en": s["names_en"],
                         "names_he": s["names_he"], "n_eligible": s.get("n_eligible"),
                         "n_strong": s.get("n_strong_eligible"), "n_gold": s.get("n_gold", 0),
                         "n_linked": len(s.get("linked_subjects", []))})
        for i, q in enumerate(allq[s["id"]]["queries"]):
            queries.append({"id": f"q:{s['id']}:{i}", "subject": s["id"], "text": q["text"], "source": q["source"],
                            "lang": q.get("lang") or ("he" if any("֐" <= ch <= "׿" for ch in q["text"]) else "en")})
    return subjects, queries


def pgp_tags(min_records: int) -> list:
    """PGP tag curation rows with at least ``min_records`` records (the ones that become subjects).

    :param min_records: Record threshold.
    :returns: Rows sorted by record count.
    :rtype: list
    """
    rows = list(csv.DictReader(open(CURATION, encoding="utf-8-sig")))
    out = []
    for r in rows:
        n = int(r["n_records"])
        if n < min_records:
            continue
        decision, _, target = r["decision"].partition("->")
        out.append({"id": f"t:{r['raw_tag']}", "tag": r["raw_tag"], "n": n, "decision": decision,
                    "target": target, "category": r["category"]})
    return sorted(out, key=lambda r: -r["n"])


def joins(n_cross: int, n_same: int, thumbs: Path) -> list:
    """Top join candidates, cross-collection first, with thumbnails.

    :param n_cross: Cross-collection pairs to include.
    :param n_same: Same-collection pairs to include.
    :param thumbs: Thumbnail directory.
    :returns: Candidate dicts.
    :rtype: list
    """
    rows = list(csv.DictReader(open(DISC / "join_candidates.csv", encoding="utf-8-sig")))
    cross = [r for r in rows if r["cross_collection"] == "1" and r["near_identical_image"] != "1"][:n_cross]
    same = [r for r in rows if r["cross_collection"] != "1" and r["near_identical_image"] != "1"][:n_same]
    out = []
    for r in cross + same:
        out.append({"id": f"j:{r['pair']}", "pair": r["pair"], "cross": r["cross_collection"] == "1",
                    "a": {"id": r["id_a"], "shelfmark": r["shelfmark_a"], "holding": r["holding_a"],
                          "img": thumb(r["id_a"], thumbs), "material": r["material_a"], "script": r["script_type_a"],
                          "style": r["script_style_a"], "types": r["pgp_types_a"]},
                    "b": {"id": r["id_b"], "shelfmark": r["shelfmark_b"], "holding": r["holding_b"],
                          "img": thumb(r["id_b"], thumbs), "material": r["material_b"], "script": r["script_type_b"],
                          "style": r["script_style_b"], "types": r["pgp_types_b"]},
                    "cosine": float(r["cosine"]), "band": r["band_labelled"], "p_est": float(r["p_discovery_est"] or 0),
                    "same_box": r["same_box_or_volume"] == "1"})
    return out


def scribes(n: int, thumbs: Path) -> list:
    """Top scribe-attribution candidates with three exemplars of the proposed scribe.

    :param n: Candidates to include.
    :param thumbs: Thumbnail directory.
    :returns: Candidate dicts.
    :rtype: list
    """
    rows = list(csv.DictReader(open(DISC / "scribe_candidates.csv", encoding="utf-8-sig")))
    rows.sort(key=lambda r: -float(r["score_margin"]))
    out = []
    for r in rows[:n]:
        ex = [e for e in (r["exemplar_1"], r["exemplar_2"], r["exemplar_3"]) if e]
        out.append({"id": f"s:{r['cand']}", "cand": r["cand"], "frag": r["id"], "shelfmark": r["shelfmark"],
                    "holding": r["holding"], "img": thumb(r["id"], thumbs), "scribe": r["proposed_scribe"],
                    "window": r["scribe_window"], "dates": r["pgp_dates"], "date_check": r["date_check"],
                    "margin": float(r["score_margin"]), "types": r["pgp_types"], "tags": r["pgp_tags"],
                    "exemplars": [{"id": e, "img": thumb(e, thumbs)} for e in ex]})
    return out


def thumb_url(cid: str, url: str, out_dir: Path, max_side: int = 520) -> str:
    """Like :func:`thumb` for a gallery fragment whose image lives on the public bucket (downloaded once, then masked).

    :param cid: Canonical fragment id.
    :param url: Image URL.
    :param out_dir: Thumbnail directory.
    :param max_side: Longest side in px.
    :returns: Path relative to the page.
    :rtype: str
    """
    out = out_dir / f"{cid.replace('/', '_')}.jpg"
    if not out.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    img = Image.open(io.BytesIO(r.read())).convert("RGB")
                break
            except OSError:
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"could not fetch {url}")
        img.thumbnail((1600, 1600))  # the gallery's max-1600 view, as embedded
        masked = fragment_mask(img)[0]
        masked.thumbnail((max_side, max_side))
        masked.save(out, quality=78)
    return f"thumbs/{out.name}"


def gallery_urls() -> dict:
    """canonical_id -> first image URL from the gallery manifest.

    :returns: Mapping.
    :rtype: dict
    """
    out = {}
    for line in open(GALLERY_MANIFEST, encoding="utf-8"):
        r = json.loads(line)
        urls = r.get("image_urls") or [r.get("image_url")]
        if urls and urls[0]:
            out[r["canonical_id"]] = urls[0]
    return out


def record_kind(cid: str, n_images: dict) -> str:
    """Classify a gallery record as a single fragment or a catalogue artefact the join search should not propose.

    Box- or series-level records (e.g. ``Paris_AIU_III_C`` with 183 images, ``Item_page``) carry one of their items' photos,
    so their "matches" are not joins. The manifest's container flag misses some of them (found 2026-10-08).

    :param cid: Canonical id.
    :param n_images: canonical_id -> number of images in the merged record.
    :returns: ``fragment``, ``multi_leaf`` (11-20 images: plausible quire, flagged) or ``artefact`` (excluded).
    :rtype: str
    """
    tail = cid.split("_", 2)[-1] if cid.count("_") >= 2 else cid
    n = n_images.get(cid, 0)
    if n > 20 or not re.search(r"\d", tail) or re.match(r"^Paris_AIU_[IVX]+_[A-E]$", cid):
        return "artefact"
    return "multi_leaf" if n > 10 else "fragment"


def gallery_n_images() -> dict:
    """canonical_id -> number of images in the merged record (from the gallery manifest).

    :returns: Mapping.
    :rtype: dict
    """
    return {json.loads(line)["canonical_id"]: json.loads(line).get("n_images") or 0
            for line in open(GALLERY_MANIFEST, encoding="utf-8")}


def joins_v2(n_cross: int, n_same: int, thumbs: Path) -> list:
    """Top gallery-scale join candidates (discovery v2), cross-holding first, best calibrated odds then similarity.

    :param n_cross: Cross-holding pairs.
    :param n_same: Same-holding pairs.
    :param thumbs: Thumbnail directory.
    :returns: Candidate dicts in the page's join layout.
    :rtype: list
    """
    rows = list(csv.DictReader(open(DISC2 / "join_candidates.csv", encoding="utf-8-sig")))
    nimg = gallery_n_images()
    rows = [r for r in rows if "artefact" not in (record_kind(r["id_a"], nimg), record_kind(r["id_b"], nimg))]
    rows.sort(key=lambda r: (-float(r["p_discovery"] or 0), -float(r["cosine_fused"] or 0)))
    cross = [r for r in rows if r["cross_holding"] == "1"][:n_cross]
    same = [r for r in rows if r["cross_holding"] != "1"][:n_same]
    out = []
    for r in cross + same:
        side = lambda x: {"id": r[f"id_{x}"], "shelfmark": r[f"shelfmark_{x}"], "holding": r[f"holding_{x}"],  # noqa: E731
                          "img": thumb_url(r[f"id_{x}"], r[f"image_{x}"], thumbs), "material": r[f"material_{x}"],
                          "script": r[f"script_type_{x}"], "style": r[f"script_style_{x}"], "types": r[f"pgp_types_{x}"]}
        out.append({"id": f"j2:{r['pair']}", "pair": r["pair"], "cross": r["cross_holding"] == "1", "a": side("a"),
                    "b": side("b"), "cosine": float(r["cosine_fused"]), "band": r["band_labelled"],
                    "p_est": float(r["p_discovery"] or 0), "same_box": r["same_box_or_volume"] == "1",
                    "tower_ms": r["tower_training_ms"] == "1",
                    "multi_leaf": "multi_leaf" in (record_kind(r["id_a"], nimg), record_kind(r["id_b"], nimg))})
    return out


def scribes_v2(n: int, per_scribe: int, thumbs: Path) -> list:
    """Top scribe attributions (discovery v2) with three exemplars each, at most ``per_scribe`` per scribe.

    :param n: Candidates to include.
    :param per_scribe: Cap per proposed scribe (keeps one prolific scribe from filling the page).
    :param thumbs: Thumbnail directory.
    :returns: Candidate dicts in the page's scribe layout.
    :rtype: list
    """
    urls = gallery_urls()
    rows = list(csv.DictReader(open(DISC2 / "scribe_candidates.csv", encoding="utf-8-sig")))
    nimg = gallery_n_images()
    rows = [r for r in rows if record_kind(r["id"], nimg) != "artefact"]
    rows.sort(key=lambda r: -float(r["score_margin"]))
    per, out = {}, []
    for r in rows:
        if len(out) >= n or per.get(r["proposed_scribe"], 0) >= per_scribe:
            continue
        ex = [e for e in (r["exemplar_1"], r["exemplar_2"], r["exemplar_3"]) if e and e in urls]
        per[r["proposed_scribe"]] = per.get(r["proposed_scribe"], 0) + 1
        out.append({"id": f"s2:{r['cand']}", "cand": r["cand"], "frag": r["id"], "shelfmark": r["shelfmark"],
                    "holding": r["holding"], "img": thumb_url(r["id"], r["image"], thumbs), "scribe": r["proposed_scribe"],
                    "window": r["scribe_window"], "dates": r["pgp_dates"], "date_check": r["date_check"],
                    "margin": float(r["score_margin"]), "types": r["pgp_types"], "tags": r["pgp_tags"],
                    "exemplars": [{"id": e, "img": thumb_url(e, urls[e], thumbs)} for e in ex]})
    return out


def main() -> None:
    """Build data + thumbnails and write the page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-tag-records", type=int, default=20)
    parser.add_argument("--joins-cross", type=int, default=15)
    parser.add_argument("--joins-same", type=int, default=15)
    parser.add_argument("--scribes", type=int, default=20)
    parser.add_argument("--discovery", choices=["v1", "v2"], default="v2",
                        help="v2 = gallery-scale candidates from the fine-tuned tower (55k fragments)")
    parser.add_argument("--scribes-per-scribe", type=int, default=3)
    args = parser.parse_args()
    out = HERE / "v3"
    thumbs = out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    subjects, queries = subjects_and_queries()
    data = {"subjects": subjects, "queries": queries, "tags": pgp_tags(args.min_tag_records),
            "joins": (joins_v2(args.joins_cross, args.joins_same, thumbs) if args.discovery == "v2"
                      else joins(args.joins_cross, args.joins_same, thumbs)),
            "scribes": (scribes_v2(args.scribes, args.scribes_per_scribe, thumbs) if args.discovery == "v2"
                        else scribes(args.scribes, thumbs))}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (HERE / "review_v3_template.html").read_text().replace("/*__REVIEW_DATA__*/null", blob)
    (out / "index.html").write_text(page)
    print(json.dumps({k: len(v) for k, v in data.items()} | {"thumbs": len(list(thumbs.glob("*.jpg")))}))


if __name__ == "__main__":
    main()
