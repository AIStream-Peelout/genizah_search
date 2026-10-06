"""Assemble the v3 review page: held-out subjects, their test queries, PGP tag curation, join and scribe candidates.

Writes ``review_ui/v3/index.html`` (the page template ``review_v3_template.html`` with the review data inlined) and
``review_ui/v3/thumbs/*.jpg`` (backdrop-masked fragment thumbnails for the join and scribe candidates). Isaac's verdicts
are stored by the page in the artifact's ``db`` collection ``review_v3`` (one doc per item id) and read back with
ArtifactData.
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from embed_utils import AUDIT_ROOT
from image_features import IMG_DIR, fragment_mask

HERE = Path(__file__).parent
EVAL = AUDIT_ROOT / "v3" / "eval"
DISC = AUDIT_ROOT / "results" / "discovery_v1"
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


def main() -> None:
    """Build data + thumbnails and write the page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-tag-records", type=int, default=20)
    parser.add_argument("--joins-cross", type=int, default=15)
    parser.add_argument("--joins-same", type=int, default=15)
    parser.add_argument("--scribes", type=int, default=20)
    args = parser.parse_args()
    out = HERE / "v3"
    thumbs = out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    subjects, queries = subjects_and_queries()
    data = {"subjects": subjects, "queries": queries, "tags": pgp_tags(args.min_tag_records),
            "joins": joins(args.joins_cross, args.joins_same, thumbs), "scribes": scribes(args.scribes, thumbs)}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (HERE / "review_v3_template.html").read_text().replace("/*__REVIEW_DATA__*/null", blob)
    (out / "index.html").write_text(page)
    print(json.dumps({k: len(v) for k, v in data.items()} | {"thumbs": len(list(thumbs.glob("*.jpg")))}))


if __name__ == "__main__":
    main()
