"""Check ``colab/gallery_preprocess.py`` (NumPy/PIL port) against the local scikit-image/SciPy pipeline.

Runs directly (no torch, no model): ``nice -n 10 <HDA .venv>/bin/python test_fragment_mask_port.py``.

1. **Real images** — ``--n`` (30) probe images from ``images/max1600`` spread over the sorted file list (so every
   holding's photographic setup is represented): the port's ``fragment_mask`` must return the identical masked crop
   and mask, and the derived views must be identical to ``image_features.load_view`` (``masked`` square, the 16
   ``mpatches`` ink patches) and to ``package_image_dataset.fragment_crop`` (the Qwen input before JPEG encoding).
2. **Synthetic edge cases** — no warm region, a warm region below 3 % of the thumbnail, two equal warm blobs (label
   tie order), a warm ring with a hole, warm pixels touching the border, a flat grey image: same checks for
   ``fragment_mask`` (these exercise the ``fragment_crop`` fallback and its ``remove_small_objects`` step).
3. **Primitive fuzz** — random masks/images: connected-component labels (8 and 4 connectivity) vs
   ``skimage.measure.label``, ``fill_holes`` vs ``scipy.ndimage.binary_fill_holes``, ``binary_closing`` vs skimage
   with ``disk(4)``/``disk(5)``, ``remove_small_objects`` vs skimage, ``rgb2hsv`` vs skimage, ``otsu_threshold`` vs
   ``skimage.filters.threshold_otsu``.

Every comparison is exact (``np.array_equal``). Exits non-zero on the first mismatch; writes
``results/fragment_mask_port_test.json``.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "colab"))
import gallery_preprocess as gp  # noqa: E402
import image_features as imf  # noqa: E402
from package_image_dataset import fragment_crop as package_fragment_crop  # noqa: E402

# probe images whose colour segmentation fails (found by scanning 800 probe images with the port: ~0.4 %), so the
# real-image check also covers the fragment_crop fallback
FALLBACK_EXAMPLES = ("Cambridge_CUL_T_S_AS_152_19", "New_York_JTS_ENA_NS_85_1312_2", "New_York_JTS_ENA_NS_85_663")


def same_image(a: Image.Image, b: Image.Image) -> bool:
    """Pixel-exact equality of two PIL images (size, mode and data).

    :param a: Image.
    :param b: Image.
    :returns: True when identical.
    :rtype: bool
    """
    return a.size == b.size and a.mode == b.mode and np.array_equal(np.asarray(a), np.asarray(b))


def same_mask(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    """Equality of two optional masks.

    :param a: Bool array or None.
    :param b: Bool array or None.
    :returns: True when both are None or both arrays are identical.
    :rtype: bool
    """
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and np.array_equal(a, b)


def pick_images(n: int) -> List[Path]:
    """Evenly spaced probe images over the sorted file list, plus the known fallback examples.

    :param n: How many evenly spaced images.
    :returns: Paths.
    :rtype: List[Path]
    """
    files = sorted(imf.IMG_DIR.glob("*.jpg"))
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    extra = [imf.IMG_DIR / f"{c}.jpg" for c in FALLBACK_EXAMPLES if (imf.IMG_DIR / f"{c}.jpg").exists()]
    return [files[i] for i in sorted(set(idx))] + [p for p in extra if p not in {files[i] for i in idx}]


def check_real(paths: List[Path]) -> List[dict]:
    """Compare masks and every derived view on real probe images.

    :param paths: Image files in ``images/max1600``.
    :returns: Per-image result rows.
    :rtype: List[dict]
    """
    rows = []
    for p in paths:
        cid = p.stem  # load_view maps cid -> file by replacing "/" with "_"; probe ids contain no "/"
        img = Image.open(p).convert("RGB")
        t0 = time.time()
        ref_masked, ref_mask = imf.fragment_mask(img)
        t_ref = time.time() - t0
        t0 = time.time()
        views = gp.views_from_max1600(img, cid)
        t_port = time.time() - t0
        assert same_image(ref_masked, views["masked"]), f"masked crop differs: {cid}"
        assert same_mask(ref_mask, views["mask"]), f"mask differs: {cid}"
        ref_canvas = imf.load_view(cid, "masked", 16, 320)[0]
        assert same_image(ref_canvas, views["canvas"]), f"masked square differs: {cid}"
        ref_patches = imf.load_view(cid, "mpatches", gp.PATCH_K, gp.PATCH_SIZE)
        assert len(ref_patches) == len(views["patches"]), f"patch count differs: {cid}"
        assert all(same_image(a, b) for a, b in zip(ref_patches, views["patches"])), f"ink patch differs: {cid}"
        assert same_image(package_fragment_crop(cid), views["qwen_resized"]), f"Qwen input differs: {cid}"
        rows.append({"id": cid, "size": list(img.size), "mask_ok": ref_mask is not None, "crop": list(ref_masked.size),
                     "patches": len(ref_patches), "t_ref_s": round(t_ref, 3), "t_port_s": round(t_port, 3)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def synthetic_images() -> Dict[str, Image.Image]:
    """Edge-case images for the segmentation paths.

    :returns: name -> RGB image.
    :rtype: Dict[str, Image.Image]
    """
    rng = np.random.default_rng(7)
    paper, blue, black = (205, 180, 140), (60, 90, 170), (12, 12, 12)

    def canvas(bg: Tuple[int, int, int], h: int = 900, w: int = 700, noise: float = 6) -> np.ndarray:
        """Noisy uniform background.

        :param bg: RGB colour.
        :param h: Height.
        :param w: Width.
        :param noise: Gaussian noise sigma.
        :returns: uint8 array.
        """
        arr = np.clip(np.array(bg, np.float32) + rng.normal(0, noise, (h, w, 3)), 0, 255)
        return arr.astype(np.uint8)

    out = {}
    a = canvas(blue)
    a[150:750, 120:560] = canvas(paper, 600, 440)
    a[300:420, 250:380] = canvas(blue, 120, 130)  # hole inside the fragment
    a[400:520, 200:260] = 20  # ink-dark blob inside
    out["warm_with_hole"] = Image.fromarray(a)
    b = canvas(black)
    b[100:400, 80:300] = canvas(paper, 300, 220)
    b[500:800, 380:600] = canvas(paper, 300, 220)  # equal-size twin: label order decides
    out["two_equal_blobs"] = Image.fromarray(b)
    c = canvas(blue)
    c[440:470, 340:370] = canvas(paper, 30, 30)  # < 3 % of the thumbnail -> fragment_crop fallback
    c[100:800, 100:200] = canvas((230, 230, 230), 700, 100)  # neutral grey card for the fallback to find
    out["tiny_warm"] = Image.fromarray(c)
    d = canvas((120, 120, 120), noise=3)
    d[200:700, 150:550] = canvas((200, 200, 200), 500, 400)
    out["no_warm_grey_card"] = Image.fromarray(d)
    e = canvas(blue)
    e[:, 400:] = canvas(paper, 900, 300)  # touches three borders
    out["warm_touching_border"] = Image.fromarray(e)
    out["flat_grey"] = Image.fromarray(np.full((600, 800, 3), 128, np.uint8))
    f = canvas(black)
    yy, xx = np.mgrid[:900, :700]
    ring = ((yy - 450) ** 2 + (xx - 350) ** 2 < 300 ** 2) & ((yy - 450) ** 2 + (xx - 350) ** 2 > 120 ** 2)
    f[ring] = canvas(paper)[ring]
    out["warm_ring"] = Image.fromarray(f)
    g = canvas((30, 30, 30))
    for _ in range(400):  # speckle of small warm and grey dots: many components
        y, x = rng.integers(0, 880), rng.integers(0, 680)
        g[y:y + rng.integers(2, 18), x:x + rng.integers(2, 18)] = paper if rng.random() < 0.5 else (200, 200, 200)
    out["speckle"] = Image.fromarray(g)
    return out


def check_synthetic() -> List[dict]:
    """Compare ``fragment_mask`` / ``fragment_crop`` on the edge cases.

    :returns: Per-image rows.
    :rtype: List[dict]
    """
    rows = []
    for name, img in synthetic_images().items():
        ref_masked, ref_mask = imf.fragment_mask(img)
        masked, mask = gp.fragment_mask(img)
        assert same_image(ref_masked, masked), f"synthetic masked crop differs: {name}"
        assert same_mask(ref_mask, mask), f"synthetic mask differs: {name}"
        assert same_image(imf.fragment_crop(img), gp.fragment_crop(img)), f"synthetic fragment_crop differs: {name}"
        rows.append({"case": name, "mask_ok": ref_mask is not None, "crop": list(ref_masked.size)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def port_label_image(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Label image from the port's run components, numbered like skimage (1.., label order).

    :param mask: Bool mask.
    :param connectivity: 8 or 4.
    :returns: int32 label image.
    :rtype: np.ndarray
    """
    cc = gp.components(mask, connectivity)
    out = np.zeros(mask.shape, np.int32)
    for k in range(len(cc["roots"])):
        sel = cc["inv"] == k
        out[gp.paint_runs(mask.shape, cc["rows"][sel], cc["starts"][sel], cc["ends"][sel])] = k + 1
    return out


def check_primitives(trials: int) -> Dict[str, int]:
    """Random-input comparison of every ported primitive.

    :param trials: Random masks per density.
    :returns: Counts of passed comparisons per primitive.
    :rtype: Dict[str, int]
    """
    from scipy import ndimage
    from skimage import measure, morphology
    from skimage.color import rgb2hsv
    from skimage.filters import threshold_otsu

    rng = np.random.default_rng(0)
    passed = {"label8": 0, "label4": 0, "fill_holes": 0, "closing_disk4": 0, "closing_disk5": 0,
              "remove_small_objects": 0, "rgb2hsv": 0, "otsu": 0}
    for density in (0.05, 0.3, 0.55, 0.8):
        for _ in range(trials):
            h, w = int(rng.integers(1, 90)), int(rng.integers(1, 90))
            m = rng.random((h, w)) < density
            if rng.random() < 0.5:  # blobby masks as well as salt-and-pepper ones
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w * 3, h * 3), Image.BILINEAR)) > 127
            for conn, key in ((2, "label8"), (1, "label4")):
                ref = measure.label(m, connectivity=conn)
                assert np.array_equal(ref, port_label_image(m, 8 if conn == 2 else 4)), f"{key} differs"
                passed[key] += 1
            assert np.array_equal(ndimage.binary_fill_holes(m), gp.fill_holes(m)), "fill_holes differs"
            passed["fill_holes"] += 1
            for r in (4, 5):
                assert np.array_equal(morphology.binary_closing(m, morphology.disk(r)), gp.binary_closing(m, r)), \
                    f"binary_closing disk({r}) differs"
                passed[f"closing_disk{r}"] += 1
            ms = int(rng.integers(1, 40))
            assert np.array_equal(morphology.remove_small_objects(m, ms), gp.remove_small_objects(m, ms)), \
                "remove_small_objects differs"
            passed["remove_small_objects"] += 1
    for _ in range(trials):
        rgb = rng.integers(0, 256, (int(rng.integers(1, 60)), int(rng.integers(1, 60)), 3), dtype=np.uint8)
        if rng.random() < 0.3:  # many exact ties between channels (grey and two-channel maxima)
            rgb[..., 1] = rgb[..., 0]
        if rng.random() < 0.2:
            rgb[..., 2] = rgb[..., 0]
        assert np.array_equal(rgb2hsv(rgb), gp.rgb2hsv(rgb)), "rgb2hsv differs"
        passed["rgb2hsv"] += 1
        g = (rng.random((int(rng.integers(2, 80)), int(rng.integers(2, 80)))) * 255).astype(np.float32)
        if rng.random() < 0.3:
            g = np.round(g / 64) * 64  # few distinct values
        if rng.random() < 0.05:
            g[:] = 17.0
        ref_t, port_t = threshold_otsu(g), gp.otsu_threshold(g)
        assert ref_t == port_t and np.asarray(ref_t).dtype == np.asarray(port_t).dtype, "otsu differs"
        passed["otsu"] += 1
    return passed


def main() -> None:
    """Run all checks and write the result file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=30, help="real probe images")
    parser.add_argument("--trials", type=int, default=60, help="random inputs per primitive and density")
    args = parser.parse_args()
    t0 = time.time()
    prim = check_primitives(args.trials)
    print(json.dumps({"primitives_passed": prim}), flush=True)
    synth = check_synthetic()
    real = check_real(pick_images(args.n))
    res = {"status": "PASS", "real_images": len(real), "real_mask_ok": sum(r["mask_ok"] for r in real),
           "real_fallback_crop": sum(not r["mask_ok"] for r in real), "synthetic_cases": synth,
           "primitives_passed": prim,
           "mean_seconds_reference_mask": round(float(np.mean([r["t_ref_s"] for r in real])), 3),
           "mean_seconds_port_views": round(float(np.mean([r["t_port_s"] for r in real])), 3),
           "real": real, "seconds": round(time.time() - t0, 1)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "fragment_mask_port_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("real", "synthetic_cases")}), flush=True)


if __name__ == "__main__":
    main()
