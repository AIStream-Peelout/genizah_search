"""Contact sheet of nearest neighbours for a few join queries (what does the encoder match on?).

For ``--n`` random fragments that have a known join partner, shows the query, its known
partner(s), and the top-``--k`` neighbours under one feature set, with the partner's rank.
Writes a JPEG to ``results/`` for the report.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from embed_utils import AUDIT_ROOT
from eval_images import JOIN_KEY, load_feats
from image_features import IMG_DIR, fragment_crop

HERE = Path(__file__).parent


def thumb(cid: str, size: int = 180) -> Image.Image:
    """Fragment-crop thumbnail.

    :param cid: Fragment id.
    :param size: Max side.
    :returns: Thumbnail.
    :rtype: Image.Image
    """
    im = fragment_crop(Image.open(IMG_DIR / f"{cid.replace('/', '_')}.jpg").convert("RGB"))
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size + 14), "white")
    canvas.paste(im, ((size - im.width) // 2, 0))
    return canvas


def main() -> None:
    """Render the sheet."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default="dinov2-base__crop")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())
    ids, X = load_feats(args.features)
    pos = {c: i for i, c in enumerate(ids)}
    partners = defaultdict(set)
    for a, b in manifest[JOIN_KEY] + manifest["joins_pgp"]:
        if a in pos and b in pos:
            partners[a].add(b)
            partners[b].add(a)
    rng = random.Random(args.seed)
    queries = rng.sample(sorted(partners), args.n)
    size = 180
    sheet = Image.new("RGB", ((args.k + 2) * (size + 6), args.n * (size + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, q in enumerate(queries):
        s = X @ X[pos[q]]
        s[pos[q]] = -9
        order = list(np.argsort(-s))
        partner = min(partners[q], key=lambda c: order.index(pos[c]))
        prank = order.index(pos[partner]) + 1
        row = [(q, "query"), (partner, f"partner r{prank}")] + [
            (ids[j], "JOIN" if ids[j] in partners[q] else f"{s[j]:.2f}") for j in order[: args.k]]
        for c, (cid, label) in enumerate(row):
            x, y = c * (size + 6), r * (size + 20)
            sheet.paste(thumb(cid, size), (x, y))
            draw.text((x + 2, y + size + 2), f"{label} {manifest['collection'][cid].split('|')[0][:14]}", fill="black")
    out = HERE / "results" / f"neighbours_{args.features}.jpg"
    sheet.save(out, quality=85)
    print(out)


if __name__ == "__main__":
    main()
