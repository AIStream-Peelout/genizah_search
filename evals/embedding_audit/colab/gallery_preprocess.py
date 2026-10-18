"""NumPy/PIL port of the audit's fragment preprocessing, for the full-gallery embedding run (Colab; importable locally).

Everything here reproduces, bit for bit on the same Pillow/NumPy build, what the local benchmark pipeline did to the
5,176 probe images, without scikit-image or SciPy (so a different skimage/scipy on Colab cannot change a mask):

* :func:`decode_max1600` — ``fetch_images.fetch``: full decode, ``thumbnail((1600, 1600))``, JPEG q90. The local
  features were computed from those JPEG files, so the copy is re-encoded and decoded again in memory.
* :func:`fragment_mask` / :func:`fragment_crop` — ``image_features.fragment_mask`` / ``fragment_crop``: substrate-colour
  segmentation (skimage ``rgb2hsv``, ``binary_closing(disk(4))``, ``measure.label`` + largest ``regionprops`` area,
  ``ndimage.binary_fill_holes``) and the backdrop-distance fallback (``binary_closing(disk(5))``,
  ``remove_small_objects(200)``). Connected components are labelled on row runs (union of overlapping runs in
  adjacent rows, 8- or 4-connectivity), numbered in raster order of their first pixel like skimage, so the
  "largest region, first on ties" choice is the same.
* :func:`ink_patches` — ``image_features.ink_patches`` with a port of skimage ``threshold_otsu`` (float path).
* :func:`masked_square` — the ``masked`` view of ``image_features.load_view`` (grey-padded square).
* :func:`qwen_fragment_image` — ``package_image_dataset.fragment_crop`` + its JPEG q92 shard encoding: the masked
  square resized to the Qwen3-VL processor's 512x512-pixel-budget size, exactly what the tuned tower was benchmarked on.
* :func:`preprocess_fragment` — all of the above for one downloaded image, returning uint8 arrays ready for the GPU
  (DINOv2 ``masked`` at 448, the ``mpatches`` ink patches at 224, the Qwen input image) plus provenance.

``test_fragment_mask_port.py`` (one level up) checks the port against the originals on local images.
"""

import hashlib
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 400_000_000  # as fetch_images.py (large KTIV/FGP scans)

MAX_SIDE = 1600            # images/max1600
MAX_SIDE_QUALITY = 90      # fetch_images.fetch JPEG quality
QWEN_JPEG_QUALITY = 92     # package_image_dataset.py --quality (fragment shards the tuned tower was evaluated on)
GRID = 32                  # Qwen3-VL patch 16 x spatial merge 2
FRAG_MIN_PIXELS, FRAG_MAX_PIXELS = 128 * 128, 512 * 512
DINO_RES = 448             # image_features.Encoder: DINOv2 input for whole views
PATCH_RES = 224            # image_features.main: patches are embedded at 224
PATCH_K, PATCH_SIZE = 16, 320  # image_features.main defaults (--k, --patch) used for dinov2-base__mpatches
GREY = 128


# ------------------------------------------------------------------------------------------------- decoding
def decode_max1600(content: bytes, max_side: int = MAX_SIDE, quality: int = MAX_SIDE_QUALITY) -> Image.Image:
    """Decode a downloaded image the way ``fetch_images.fetch`` stored it, then read it back as the probes did.

    :param content: Raw bytes of the full-size image.
    :param max_side: Longest side after ``thumbnail``.
    :param quality: JPEG quality of the stored copy.
    :returns: RGB image identical to ``Image.open(<max1600 file>).convert("RGB")``.
    :rtype: Image.Image
    """
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode to JPEG in memory and decode again (reproduces a stored-JPEG step of the local pipeline).

    :param img: RGB image.
    :param quality: JPEG quality.
    :returns: Decoded RGB image.
    :rtype: Image.Image
    """
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


# ------------------------------------------------------------------------------------------------- colour
def border_color(arr: np.ndarray) -> np.ndarray:
    """Median colour of the outer 3 % frame (``image_features.border_color``).

    :param arr: HxWx3 uint8 image.
    :returns: RGB colour (float64).
    :rtype: np.ndarray
    """
    h, w = arr.shape[:2]
    b = max(4, int(0.03 * min(h, w)))
    frame = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                            arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    return np.median(frame, axis=0)


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Port of ``skimage.color.rgb2hsv`` (0.25) for uint8 input: same float64 arithmetic, same tie order.

    :param rgb: HxWx3 uint8 array.
    :returns: HxWx3 float64 array (h, s, v in [0, 1]).
    :rtype: np.ndarray
    """
    arr = np.multiply(rgb, 1.0 / 255, dtype=np.float64)  # skimage img_as_float64 on uint8
    out = np.empty_like(arr)
    out_v = arr.max(-1)
    delta = np.ptp(arr, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_s = delta / out_v
        out_s[delta == 0.0] = 0.0
        idx = arr[..., 0] == out_v
        out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
        idx = arr[..., 1] == out_v
        out[idx, 0] = 2.0 + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
        idx = arr[..., 2] == out_v
        out[idx, 0] = 4.0 + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
        out_h = (out[..., 0] / 6.0) % 1.0
    out_h[delta == 0.0] = 0.0
    out[..., 0] = out_h
    out[..., 1] = out_s
    out[..., 2] = out_v
    out[np.isnan(out)] = 0
    return out


# ------------------------------------------------------------------------------------------------- morphology
def disk_offsets(radius: int) -> List[Tuple[int, int]]:
    """Offsets of ``skimage.morphology.disk(radius)`` (``x**2 + y**2 <= radius**2``).

    :param radius: Disk radius.
    :returns: (dy, dx) offsets.
    :rtype: List[Tuple[int, int]]
    """
    r = np.arange(-radius, radius + 1)
    return [(int(dy), int(dx)) for dy in r for dx in r if dy * dy + dx * dx <= radius * radius]


def _shift_reduce(mask: np.ndarray, radius: int, erode: bool) -> np.ndarray:
    """Dilation (OR, outside = False) or erosion (AND, outside = True) with a disk footprint.

    The disk is symmetric, so reflecting the footprint (as SciPy does for dilation) changes nothing. Border values
    follow skimage's ``mode="ignore"``: dilation sees False and erosion sees True outside the image.

    :param mask: 2-D bool array.
    :param radius: Disk radius.
    :param erode: Erosion if True, else dilation.
    :returns: Result (bool).
    :rtype: np.ndarray
    """
    h, w = mask.shape
    pad = np.full((h + 2 * radius, w + 2 * radius), erode, dtype=bool)
    pad[radius:radius + h, radius:radius + w] = mask
    out = np.full((h, w), erode, dtype=bool)
    for dy, dx in disk_offsets(radius):
        view = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
        if erode:
            out &= view
        else:
            out |= view
    return out


def binary_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """``skimage.morphology.binary_closing(mask, disk(radius))``: dilation then erosion.

    :param mask: 2-D bool array.
    :param radius: Disk radius.
    :returns: Closed mask.
    :rtype: np.ndarray
    """
    return _shift_reduce(_shift_reduce(mask, radius, erode=False), radius, erode=True)


# ------------------------------------------------------------------------------------------------- components
def mask_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal runs of True pixels in raster order.

    :param mask: 2-D bool array.
    :returns: (row, start, end) int64 arrays; ``end`` is exclusive.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    return rows.astype(np.int64), starts.astype(np.int64), ends.astype(np.int64)


def label_runs(rows: np.ndarray, starts: np.ndarray, ends: np.ndarray, width: int, connectivity: int) -> np.ndarray:
    """Connected components of runs: each run gets the index of the first (raster-order) run of its component.

    Two runs in adjacent rows are connected when their column intervals overlap (4-connectivity) or overlap after
    widening by one pixel (8-connectivity). Components are found by minimum-label hooking with pointer jumping
    (vectorised union-find), so the label of a component is its smallest run index — the run holding its first pixel
    in raster order, which is the order skimage numbers labels in.

    :param rows: Run rows (raster order).
    :param starts: Run starts.
    :param ends: Run ends (exclusive).
    :param width: Image width.
    :param connectivity: 8 or 4.
    :returns: Root run index per run.
    :rtype: np.ndarray
    """
    n = len(rows)
    lab = np.arange(n)
    if n < 2:
        return lab
    stride = width + 2
    gs, ge = rows * stride + starts, rows * stride + ends  # both strictly increasing in raster order
    prev = rows - 1
    lo_key, hi_key = prev * stride + starts, prev * stride + ends
    if connectivity == 8:  # a in row r touches b in row r+1 iff e_a >= s_b and s_a <= e_b
        first = np.searchsorted(ge, lo_key, side="left")
        last = np.searchsorted(gs, hi_key, side="right")
    elif connectivity == 4:  # iff e_a > s_b and s_a < e_b
        first = np.searchsorted(ge, lo_key, side="right")
        last = np.searchsorted(gs, hi_key, side="left")
    else:
        raise ValueError(connectivity)
    counts = np.maximum(last - first, 0)
    total = int(counts.sum())
    if total == 0:
        return lab
    u = np.repeat(np.arange(n), counts)
    v = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(first, counts)
    while True:
        lu, lv = lab[u], lab[v]
        if np.array_equal(lu, lv):
            return lab
        m = np.minimum(lu, lv)
        np.minimum.at(lab, lu, m)
        np.minimum.at(lab, lv, m)
        while True:  # pointer jumping to full compression
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt


def paint_runs(shape: Tuple[int, int], rows: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Rasterise runs into a bool mask.

    :param shape: (h, w).
    :param rows: Run rows.
    :param starts: Run starts.
    :param ends: Run ends (exclusive).
    :returns: Bool mask.
    :rtype: np.ndarray
    """
    h, w = shape
    acc = np.zeros((h, w + 1), dtype=np.int32)
    np.add.at(acc, (rows, starts), 1)
    np.add.at(acc, (rows, ends), -1)
    return np.cumsum(acc, axis=1)[:, :w] > 0


def components(mask: np.ndarray, connectivity: int) -> dict:
    """Connected components of a mask (runs, root per run, roots in label order, pixel area per root).

    :param mask: 2-D bool array.
    :param connectivity: 8 (skimage ``measure.label`` default) or 4 (SciPy / ``remove_small_objects`` default).
    :returns: {rows, starts, ends, root, roots (ascending = skimage label order), areas (per root)}.
    :rtype: dict
    """
    rows, starts, ends = mask_runs(mask)
    root = label_runs(rows, starts, ends, mask.shape[1], connectivity)
    roots, inv = np.unique(root, return_inverse=True)
    areas = np.bincount(inv, weights=(ends - starts).astype(np.float64), minlength=len(roots)).astype(np.int64)
    return {"rows": rows, "starts": starts, "ends": ends, "root": root, "inv": inv, "roots": roots, "areas": areas}


def largest_component(mask: np.ndarray, connectivity: int = 8) -> Optional[dict]:
    """Largest component (first in label order on ties, like ``max(regionprops(...), key=area)``).

    :param mask: 2-D bool array.
    :param connectivity: Pixel connectivity.
    :returns: None when the mask is empty, else {area, bbox (min_row, min_col, max_row + 1, max_col + 1), rows,
        starts, ends} of the component's runs.
    :rtype: Optional[dict]
    """
    cc = components(mask, connectivity)
    if len(cc["roots"]) == 0:
        return None
    k = int(np.argmax(cc["areas"]))  # first maximum = smallest label
    sel = cc["inv"] == k
    rows, starts, ends = cc["rows"][sel], cc["starts"][sel], cc["ends"][sel]
    bbox = (int(rows.min()), int(starts.min()), int(rows.max()) + 1, int(ends.max()))
    return {"area": int(cc["areas"][k]), "bbox": bbox, "rows": rows, "starts": starts, "ends": ends}


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """``scipy.ndimage.binary_fill_holes`` (default cross structure): background not 4-connected to the border is filled.

    :param mask: 2-D bool array.
    :returns: Filled mask.
    :rtype: np.ndarray
    """
    h, w = mask.shape
    cc = components(~mask, 4)
    if len(cc["roots"]) == 0:
        return mask.copy()
    touches = (cc["rows"] == 0) | (cc["rows"] == h - 1) | (cc["starts"] == 0) | (cc["ends"] == w)
    outside = np.zeros(len(cc["roots"]), dtype=bool)
    outside[cc["inv"][touches]] = True
    keep = outside[cc["inv"]]
    return ~paint_runs(mask.shape, cc["rows"][keep], cc["starts"][keep], cc["ends"][keep])


def remove_small_objects(mask: np.ndarray, min_size: int, connectivity: int = 4) -> np.ndarray:
    """``skimage.morphology.remove_small_objects`` on a bool mask (0.25 semantics: drop components < ``min_size``).

    :param mask: 2-D bool array.
    :param min_size: Smallest component kept (pixels).
    :param connectivity: 4 (skimage default ``connectivity=1``) or 8.
    :returns: Cleaned mask.
    :rtype: np.ndarray
    """
    cc = components(mask, connectivity)
    if len(cc["roots"]) == 0:
        return mask.copy()
    keep = (cc["areas"] >= min_size)[cc["inv"]]
    return paint_runs(mask.shape, cc["rows"][keep], cc["starts"][keep], cc["ends"][keep])


# ------------------------------------------------------------------------------------------------- segmentation
def fragment_crop(img: Image.Image) -> Image.Image:
    """Port of ``image_features.fragment_crop``: crop to the largest non-background component.

    :param img: RGB image.
    :returns: Cropped image (the input if segmentation fails).
    :rtype: Image.Image
    """
    small = img.copy()
    small.thumbnail((512, 512))
    arr = np.asarray(small).astype(np.float32)
    dist = np.linalg.norm(arr - border_color(arr.astype(np.uint8)), axis=2)
    mask = dist > 40
    mask = binary_closing(mask, 5)
    mask = remove_small_objects(mask, 200)
    region = largest_component(mask, 8)
    if region is None:
        return img
    y0, x0, y1, x1 = region["bbox"]
    if (y1 - y0) * (x1 - x0) < 0.05 * mask.size:
        return img
    sx, sy = img.width / small.width, img.height / small.height
    return img.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))


def fragment_mask(img: Image.Image) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """Port of ``image_features.fragment_mask``: keep the largest warm (paper/vellum) component, grey out the rest.

    :param img: RGB image.
    :returns: (masked image cropped to the fragment, bool mask at crop size); the mask is None when the colour
        segmentation fails and the backdrop-distance crop is used instead.
    :rtype: Tuple[Image.Image, Optional[np.ndarray]]
    """
    small = img.copy()
    small.thumbnail((512, 512))
    hsv = rgb2hsv(np.asarray(small))
    hue = hsv[..., 0] * 360
    warm = (hue >= 8) & (hue <= 65) & (hsv[..., 1] >= 0.10) & (hsv[..., 2] >= 0.18)
    warm = binary_closing(warm, 4)
    region = largest_component(warm, 8)
    if region is None:
        return fragment_crop(img), None
    if region["area"] < 0.03 * warm.size:
        return fragment_crop(img), None
    mask_small = fill_holes(paint_runs(warm.shape, region["rows"], region["starts"], region["ends"]))
    mask = np.asarray(Image.fromarray(mask_small.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) > 0
    arr = np.asarray(img).copy()
    arr[~mask] = GREY
    y0, x0, y1, x1 = region["bbox"]
    sx, sy = img.width / small.width, img.height / small.height
    box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
    return Image.fromarray(arr).crop(box), mask[box[1]:box[3], box[0]:box[2]]


def masked_square(masked: Image.Image) -> Image.Image:
    """The ``masked`` view of ``image_features.load_view``: the masked crop centred on a grey square.

    :param masked: Output of :func:`fragment_mask`.
    :returns: Square RGB image.
    :rtype: Image.Image
    """
    side = max(masked.size)
    canvas = Image.new("RGB", (side, side), (GREY, GREY, GREY))
    canvas.paste(masked, ((side - masked.width) // 2, (side - masked.height) // 2))
    return canvas


# ------------------------------------------------------------------------------------------------- ink patches
def otsu_threshold(image: np.ndarray, nbins: int = 256) -> np.floating:
    """Port of ``skimage.filters.threshold_otsu`` (0.25) for a float image (``np.histogram`` path, float32 counts).

    :param image: 2-D float array.
    :param nbins: Histogram bins.
    :returns: Threshold (same dtype/value as skimage's).
    """
    first_pixel = image.reshape(-1)[0]
    if np.all(image == first_pixel):
        return first_pixel
    hist, bin_edges = np.histogram(image.reshape(-1).flatten(), bins=nbins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    counts = hist.astype("float32", copy=False)
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return bin_centers[np.argmax(variance12)]


def ink_patches(img: Image.Image, k: int, size: int, rng: np.random.Generator) -> List[Image.Image]:
    """Port of ``image_features.ink_patches``: up to k square patches with 4-35 % ink (Otsu) pixels.

    :param img: Cropped fragment image.
    :param k: Patches wanted.
    :param size: Patch side in pixels.
    :param rng: Random generator (seeded per fragment by :func:`patch_seed`).
    :returns: Patches (at least one; a resize of the whole crop when nothing qualifies).
    :rtype: List[Image.Image]
    """
    gray = np.asarray(img.convert("L")).astype(np.float32)
    if gray.shape[0] < size or gray.shape[1] < size:
        return [img.resize((size, size))]
    ink = gray < otsu_threshold(gray)
    out, tries = [], 0
    while len(out) < k and tries < k * 20:
        tries += 1
        y = int(rng.integers(0, gray.shape[0] - size + 1))
        x = int(rng.integers(0, gray.shape[1] - size + 1))
        frac = ink[y:y + size, x:x + size].mean()
        if 0.04 <= frac <= 0.35:
            out.append(img.crop((x, y, x + size, y + size)))
    return out or [img.resize((size, size))]


def patch_seed(cid: str) -> int:
    """Per-fragment patch-sampling seed of ``image_features.load_view``.

    :param cid: Canonical fragment id.
    :returns: Seed.
    :rtype: int
    """
    return int.from_bytes(cid.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 32)


# ------------------------------------------------------------------------------------------------- Qwen input
def smart_resize(height: int, width: int, factor: int = GRID, min_pixels: int = FRAG_MIN_PIXELS,
                 max_pixels: int = FRAG_MAX_PIXELS) -> Tuple[int, int]:
    """The Qwen2-VL/Qwen3-VL processor's target size (``package_image_dataset.smart_resize``).

    :param height: Input height.
    :param width: Input width.
    :param factor: Patch x merge (32).
    :param min_pixels: Pixel budget floor.
    :param max_pixels: Pixel budget ceiling.
    :returns: (height, width), multiples of ``factor``.
    :rtype: Tuple[int, int]
    """
    import math

    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_resized(canvas: Image.Image) -> Image.Image:
    """``package_image_dataset.fragment_crop`` before JPEG encoding: masked square at the 512x512-budget size.

    :param canvas: Output of :func:`masked_square`.
    :returns: RGB image whose sides are multiples of 32.
    :rtype: Image.Image
    """
    h, w = smart_resize(canvas.height, canvas.width)
    return canvas if (w, h) == canvas.size else canvas.resize((w, h), Image.BICUBIC)


def qwen_fragment_image(canvas: Image.Image, quality: int = QWEN_JPEG_QUALITY) -> Image.Image:
    """The tuned tower's benchmark input: :func:`qwen_resized` stored as JPEG q92 and read back.

    :param canvas: Output of :func:`masked_square`.
    :param quality: Shard JPEG quality.
    :returns: RGB image.
    :rtype: Image.Image
    """
    return jpeg_roundtrip(qwen_resized(canvas), quality)


# ------------------------------------------------------------------------------------------------- provenance
def dhash(img: Image.Image) -> int:
    """64-bit difference hash of an image (near-identical photographs have Hamming distance ~0).

    :param img: RGB image.
    :returns: Unsigned 64-bit hash as a Python int.
    :rtype: int
    """
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def preprocess_fragment(content: bytes, cid: str, k: int = PATCH_K, patch: int = PATCH_SIZE) -> Dict[str, object]:
    """Every model input for one fragment, from the raw downloaded bytes.

    :param content: Full-size image bytes (GCS).
    :param cid: Canonical id (seeds the ink-patch sampling).
    :param k: Ink patches wanted.
    :param patch: Ink patch side in pixels (before resizing to 224).
    :returns: {``dino_masked`` (448x448x3 uint8), ``dino_patches`` (n x 224 x 224 x 3 uint8), ``qwen`` (HxWx3 uint8,
        sides multiples of 32), ``sha1``, ``max1600_wh``, ``crop_wh``, ``mask_ok``, ``mask_frac``, ``n_patches``,
        ``dhash``}.
    :rtype: Dict[str, object]
    """
    img = decode_max1600(content)
    masked, mask = fragment_mask(img)
    canvas = masked_square(masked)
    dino = np.asarray(canvas.convert("RGB").resize((DINO_RES, DINO_RES), Image.BICUBIC))
    patches = ink_patches(masked, k, patch, np.random.default_rng(patch_seed(cid)))
    parr = np.stack([np.asarray(p.convert("RGB").resize((PATCH_RES, PATCH_RES), Image.BICUBIC)) for p in patches])
    qwen = np.asarray(qwen_fragment_image(canvas))
    return {"dino_masked": dino, "dino_patches": parr, "qwen": qwen, "sha1": hashlib.sha1(content).hexdigest(),
            "max1600_wh": img.size, "crop_wh": masked.size, "mask_ok": mask is not None,
            "mask_frac": float(mask.mean()) if mask is not None else -1.0, "n_patches": len(patches),
            "dhash": dhash(masked)}


def views_from_max1600(img: Image.Image, cid: str, k: int = PATCH_K, patch: int = PATCH_SIZE) -> Dict[str, object]:
    """Same views as :func:`preprocess_fragment` starting from an already-decoded max1600 image (local tests).

    :param img: RGB image as read from ``images/max1600``.
    :param cid: Canonical id.
    :param k: Ink patches wanted.
    :param patch: Ink patch side.
    :returns: {masked, mask, canvas, patches, qwen_resized} as PIL images / arrays.
    :rtype: Dict[str, object]
    """
    masked, mask = fragment_mask(img)
    canvas = masked_square(masked)
    return {"masked": masked, "mask": mask, "canvas": canvas,
            "patches": ink_patches(masked, k, patch, np.random.default_rng(patch_seed(cid))),
            "qwen_resized": qwen_resized(canvas)}
