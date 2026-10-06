"""Download the image-probe subset from the project's public GCS bucket to the NAS.

Selects fragments for the image probes from ``image_manifest_v1.json``:
every fragment in a resolved join pair, scribe-labelled fragments (scribes with
>= 5 fragments, capped per scribe), script-style/region-labelled fragments
(capped per class), and random distractors stratified by collection. Each
image is downscaled to ``--max-side`` (JPEG q90) so the NAS copy stays small.
Already-downloaded files are skipped (resumable).
"""

import argparse
import io
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import requests
from PIL import Image

from embed_utils import AUDIT_ROOT

Image.MAX_IMAGE_PIXELS = 400_000_000


def select(manifest: dict, per_scribe: int, per_class: int, n_distractors: int, seed: int = 0) -> Dict[str, List[str]]:
    """Choose the fragment ids for each probe.

    :param manifest: Parsed manifest.
    :param per_scribe: Max fragments per scribe.
    :param per_class: Max fragments per script class.
    :param n_distractors: Random distractor count.
    :param seed: RNG seed.
    :returns: Mapping probe-name -> fragment ids.
    :rtype: Dict[str, List[str]]
    """
    rng = random.Random(seed)
    joins = sorted({c for pair in manifest["joins_fjp"] + manifest["joins_pgp"] for c in pair})
    by_scribe = defaultdict(list)
    for cid, s in sorted(manifest["scribe"].items()):
        by_scribe[s].append(cid)
    scribes = []
    for s, cids in by_scribe.items():
        if len(cids) >= 5:
            rng.shuffle(cids)
            scribes += cids[:per_scribe]
    by_class = defaultdict(list)
    for cid, a in sorted(manifest["attrs"].items()):
        if a.get("script_style") in ("Square", "Semi-Cursive", "Cursive", "Naskhi", "Rabbinical"):
            by_class["style:" + a["script_style"]].append(cid)
        if a.get("script_region") in ("Oriental", "Spanish", "Yemenite", "Italian", "Ashkenazi", "North African", "Syrian"):
            by_class["region:" + a["script_region"]].append(cid)
    script = []
    for k, cids in by_class.items():
        rng.shuffle(cids)
        script += cids[:per_class]
    # distractors: stratified to the join set's collection mix (controls for imaging-condition confounds)
    coll_mix = Counter(manifest["collection"][c] for c in joins)
    pool = defaultdict(list)
    chosen = set(joins) | set(scribes) | set(script)
    for cid in manifest["images"]:
        if cid not in chosen:
            pool[manifest["collection"][cid]].append(cid)
    distractors = []
    total = sum(coll_mix.values())
    for coll, n in coll_mix.items():
        want = round(n_distractors * n / total)
        cands = pool.get(coll, [])
        rng.shuffle(cands)
        distractors += cands[:want]
    return {"joins": joins, "scribes": scribes, "script": sorted(set(script)), "distractors": distractors}


def fetch(cid: str, url: str, out_dir: Path, max_side: int) -> str:
    """Download and downscale one image.

    :param cid: Canonical fragment id (file stem).
    :param url: Image URL.
    :param out_dir: Destination directory.
    :param max_side: Longest side after downscaling.
    :returns: Status string.
    :rtype: str
    """
    dest = out_dir / f"{cid.replace('/', '_')}.jpg"
    if dest.exists():
        return "cached"
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        return f"http{resp.status_code}"
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img.thumbnail((max_side, max_side))
    tmp = dest.with_suffix(".part")
    img.save(tmp, "JPEG", quality=90)
    tmp.rename(dest)
    return "ok"


def main() -> None:
    """Select, then download in parallel."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--per-scribe", type=int, default=60)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--distractors", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())
    sel = select(manifest, args.per_scribe, args.per_class, args.distractors)
    (AUDIT_ROOT / "image_selection_v1.json").write_text(json.dumps(sel))
    ids = sorted({c for v in sel.values() for c in v})
    print({k: len(v) for k, v in sel.items()}, "unique", len(ids), flush=True)
    out_dir = AUDIT_ROOT / "images" / f"max{args.max_side}"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Counter()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, c, manifest["images"][c], out_dir, args.max_side): c for c in ids}
        for i, fut in enumerate(futures):
            try:
                status[fut.result()] += 1
            except Exception as exc:  # network/decoding errors are counted, not fatal
                status[type(exc).__name__] += 1
            if (i + 1) % 500 == 0:
                print(i + 1, dict(status), flush=True)
    print("done", dict(status), flush=True)


if __name__ == "__main__":
    main()
