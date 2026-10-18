"""Classical palaeographic/physical descriptor for a fragment crop (CPU, no model).

A baseline against the learned encoders, in the spirit of pre-deep-learning writer
identification and of the Genizah join work (Wolf et al., IJCV 2011), which combined
handwriting statistics with physical measurements. Blocks (each sqrt-normalised, then the
whole vector L2-normalised):

* ``orient``  — 36-bin histogram of gradient orientation on ink edges (slant, stroke direction);
* ``hinge``   — 12x12 co-occurrence of edge orientations at pixel offsets (3,0)/(0,3)/(3,3)
  (a cheap stand-in for the "hinge" writer feature: curvature/joins of strokes);
* ``width``   — histogram of stroke half-widths from the distance transform of the ink mask;
* ``layout``  — line pitch / ink-height ratio from the horizontal projection profile and ink density;
* ``colour``  — mean/std of ink and of substrate in Lab (ink fading and paper/vellum tone are physical
  cues that joined leaves share).
"""

from typing import Dict

import numpy as np
from PIL import Image


def _hist(values: np.ndarray, bins: int, rng: tuple, weights=None) -> np.ndarray:
    """Normalised histogram.

    :param values: Values.
    :param bins: Bin count.
    :param rng: (min, max).
    :param weights: Optional weights.
    :returns: Histogram summing to 1 (or zeros).
    :rtype: np.ndarray
    """
    h, _ = np.histogram(values, bins=bins, range=rng, weights=weights)
    s = h.sum()
    return h / s if s else h.astype(float)


def describe(img: Image.Image) -> np.ndarray:
    """Compute the handcrafted descriptor of one cropped fragment image.

    :param img: RGB fragment crop.
    :returns: L2-normalised float32 vector.
    :rtype: np.ndarray
    """
    from scipy import ndimage
    from skimage.color import rgb2lab
    from skimage.filters import threshold_otsu

    img = img.copy()
    img.thumbnail((1200, 1200))
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(2)
    try:
        ink = gray < threshold_otsu(gray)
    except ValueError:
        ink = np.zeros_like(gray, dtype=bool)
    if ink.mean() > 0.5:  # inverted / dark substrate: treat the minority class as ink
        ink = ~ink
    gy, gx = np.gradient(ndimage.gaussian_filter(gray, 1.0))
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # orientation, not direction
    edge = mag > np.percentile(mag, 90)
    blocks: Dict[str, np.ndarray] = {}
    blocks["orient"] = _hist(ang[edge], 36, (0, np.pi), weights=mag[edge])
    q = np.minimum((ang / np.pi * 12).astype(int), 11)
    co = np.zeros((12, 12))
    for dy, dx in ((0, 3), (3, 0), (3, 3)):
        a = edge[: edge.shape[0] - dy, : edge.shape[1] - dx] & edge[dy:, dx:]
        np.add.at(co, (q[: q.shape[0] - dy, : q.shape[1] - dx][a], q[dy:, dx:][a]), 1)
    blocks["hinge"] = (co / co.sum()).ravel() if co.sum() else co.ravel()
    dist = ndimage.distance_transform_edt(ink)
    blocks["width"] = _hist(dist[ink], 20, (0, 10))
    prof = ink.mean(1) - ink.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac = ac / ac[0] if ac[0] else ac
    peak = int(np.argmax(ac[5:200]) + 5) if len(ac) > 200 else 0
    rows_ink = np.where(ink.mean(1) > 0.02)[0]
    blocks["layout"] = np.array([peak / max(1, gray.shape[0]) * 10, ink.mean() * 10,
                                 len(rows_ink) / max(1, gray.shape[0])], dtype=float)
    lab = rgb2lab(rgb)
    sub = ~ink & (gray < 0.97)
    def stats(mask):
        if mask.sum() < 50:
            return np.zeros(6)
        v = lab[mask]
        return np.concatenate([v.mean(0) / 100, v.std(0) / 50])
    blocks["colour"] = np.concatenate([stats(ink), stats(sub)])
    vec = np.concatenate([np.sqrt(np.abs(b)) * np.sign(b) / max(1e-9, np.linalg.norm(np.sqrt(np.abs(b))))
                          for b in blocks.values()])
    return (vec / max(1e-9, np.linalg.norm(vec))).astype(np.float32)
