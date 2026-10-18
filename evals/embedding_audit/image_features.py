"""Extract image features for the image-similarity probes (one encoder x one view per run).

Views
-----
* ``global``  — the whole photograph (fragment + background + library label/ruler/border),
  padded to a square with its border colour. This is what a naive page embedding sees.
* ``crop``    — the fragment's bounding box only (largest non-background component),
  removing most imaging-setup cues (backgrounds, colour charts, rulers, labels).
* ``patches`` — K ink-bearing square patches sampled inside the crop at near-native
  resolution, embedded separately and mean-aggregated: a handwriting-texture descriptor.

Encoders: DINOv2 (self-supervised ViT), SigLIP2 (image-text contrastive), CLIP, and the
stock Qwen3-VL-8B vision tower (the VLM's own image features, mean-pooled after the merger —
the closest proxy for the old ColNomic/ColQwen mean-pooled page vectors).

Writes ``cache/image_feats/<encoder>__<view>/part_XXXX.npz`` (ids + feats); resumable.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from embed_utils import AUDIT_ROOT, device

# AUDIT_IMG_DIR: a local mirror of the same files (the NAS share can be slow over Wi-Fi); default = the NAS copy
IMG_DIR = Path(os.environ.get("AUDIT_IMG_DIR", str(AUDIT_ROOT / "images" / "max1600")))
QWEN_VL_SNAPSHOT = Path("/Volumes/home/studio_offload/hf_home_merge/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
HF_IDS = {
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    "clip-b32": "openai/clip-vit-base-patch32",
}
MEAN_STD = {
    "dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "siglip2": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "clip": ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
}


def border_color(arr: np.ndarray) -> np.ndarray:
    """Median colour of the outer 3% frame (the photographic background).

    :param arr: HxWx3 uint8 image.
    :returns: RGB background colour.
    :rtype: np.ndarray
    """
    h, w = arr.shape[:2]
    b = max(4, int(0.03 * min(h, w)))
    frame = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                            arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    return np.median(frame, axis=0)


def fragment_crop(img: Image.Image) -> Image.Image:
    """Crop to the largest non-background connected component (the fragment).

    :param img: RGB image.
    :returns: Cropped image (the input if segmentation fails).
    :rtype: Image.Image
    """
    from skimage import measure, morphology

    small = img.copy()
    small.thumbnail((512, 512))
    arr = np.asarray(small).astype(np.float32)
    dist = np.linalg.norm(arr - border_color(arr.astype(np.uint8)), axis=2)
    mask = dist > 40
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.remove_small_objects(mask, 200)
    labels = measure.label(mask)
    if labels.max() == 0:
        return img
    region = max(measure.regionprops(labels), key=lambda r: r.area)
    y0, x0, y1, x1 = region.bbox
    if (y1 - y0) * (x1 - x0) < 0.05 * mask.size:
        return img
    sx, sy = img.width / small.width, img.height / small.height
    return img.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))


def fragment_mask(img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """Segment the fragment by substrate colour and blank everything else.

    Paper and vellum are warm (hue ~10-60 deg, some saturation); the photographic backdrops seen in
    the corpus are grey/white, blue (CUL "T-S AS" boxes) or black (Bodleian). The largest
    warm component (holes filled) is kept; everything else is set to neutral grey so backdrop
    colour, colour charts, rulers and mylar stitching cannot drive similarity.

    :param img: RGB image.
    :returns: (masked image cropped to the fragment, boolean mask at crop size).
    :rtype: Tuple[Image.Image, np.ndarray]
    """
    from scipy import ndimage
    from skimage import measure, morphology
    from skimage.color import rgb2hsv

    small = img.copy()
    small.thumbnail((512, 512))
    hsv = rgb2hsv(np.asarray(small))
    hue = hsv[..., 0] * 360
    warm = (hue >= 8) & (hue <= 65) & (hsv[..., 1] >= 0.10) & (hsv[..., 2] >= 0.18)
    warm = morphology.binary_closing(warm, morphology.disk(4))
    labels = measure.label(warm)
    if labels.max() == 0:
        return fragment_crop(img), None
    region = max(measure.regionprops(labels), key=lambda r: r.area)
    if region.area < 0.03 * warm.size:
        return fragment_crop(img), None
    mask_small = ndimage.binary_fill_holes(labels == region.label)
    mask = np.asarray(Image.fromarray(mask_small.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) > 0
    arr = np.asarray(img).copy()
    arr[~mask] = 128
    y0, x0, y1, x1 = region.bbox
    sx, sy = img.width / small.width, img.height / small.height
    box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
    return Image.fromarray(arr).crop(box), mask[box[1]:box[3], box[0]:box[2]]


def pad_square(img: Image.Image) -> Image.Image:
    """Pad to square with the image's border colour.

    :param img: RGB image.
    :returns: Square image.
    :rtype: Image.Image
    """
    side = max(img.size)
    bg = tuple(int(c) for c in border_color(np.asarray(img)))
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


def ink_patches(img: Image.Image, k: int, size: int, rng: np.random.Generator) -> List[Image.Image]:
    """Sample up to k square patches containing a plausible amount of ink.

    :param img: Cropped fragment image (near-native resolution).
    :param k: Patches wanted.
    :param size: Patch side in pixels.
    :param rng: Random generator (seeded per image for reproducibility).
    :returns: List of patches (may be fewer than k on tiny fragments).
    :rtype: List[Image.Image]
    """
    from skimage.filters import threshold_otsu

    gray = np.asarray(img.convert("L")).astype(np.float32)
    if gray.shape[0] < size or gray.shape[1] < size:
        return [img.resize((size, size))]
    ink = gray < threshold_otsu(gray)
    out, tries = [], 0
    while len(out) < k and tries < k * 20:
        tries += 1
        y = int(rng.integers(0, gray.shape[0] - size + 1))
        x = int(rng.integers(0, gray.shape[1] - size + 1))
        frac = ink[y:y + size, x:x + size].mean()
        if 0.04 <= frac <= 0.35:
            out.append(img.crop((x, y, x + size, y + size)))
    return out or [img.resize((size, size))]


def to_tensor(imgs: List[Image.Image], res: int, family: str):
    """Resize + normalise a list of images into a batch tensor.

    :param imgs: Images.
    :param res: Square input resolution.
    :param family: Normalisation family key.
    :returns: Float tensor (B,3,res,res).
    """
    import torch

    mean, std = MEAN_STD[family]
    arr = np.stack([np.asarray(i.convert("RGB").resize((res, res), Image.BICUBIC), dtype=np.float32) / 255.0 for i in imgs])
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class Encoder:
    """Uniform wrapper: ``embed(list[PIL]) -> (B, D)`` normalised numpy."""

    def __init__(self, name: str) -> None:
        """Load the encoder.

        :param name: Encoder key.
        """
        import torch

        self.name, self.dev = name, device()
        if name == "handcrafted":
            self.model, self.family, self.res = None, "handcrafted", None
            return
        if name.startswith("dinov2"):
            from transformers import AutoModel
            self.model, self.family, self.res = AutoModel.from_pretrained(HF_IDS[name]), "dinov2", 448
        elif name.startswith("siglip2"):
            from transformers import SiglipVisionModel
            self.model, self.family, self.res = SiglipVisionModel.from_pretrained(HF_IDS[name]), "siglip2", 384
        elif name.startswith("clip"):
            from transformers import CLIPVisionModelWithProjection
            self.model, self.family, self.res = CLIPVisionModelWithProjection.from_pretrained(HF_IDS[name]), "clip", 224
        elif name.startswith("qwen3vl-vit"):
            # "qwen3vl-vit512-<lmstudio model dir>" = a fine-tuned Hebrew VLM's tower (e.g. heb-v21b-step1200);
            # an "@ds8" / "@ds16" / "@ds24" suffix reads that deepstack layer instead of the final merger output
            # (on single lines the merger output encodes WHAT is written, early layers HOW: line_probe I2)
            base, _, readout = name.partition("@")
            self.readout = readout or "merger"
            ckpt = base.split("vit512-", 1)[1] if "vit512-" in base else None
            self.model, self.processor = self._load_qwen_vit(ckpt)
            self.family, self.res = "qwen", None
            # "qwen3vl-vit512" = 512x512-pixel budget (CPU-feasible); plain = 1024x1024
            self.max_pixels = (512 * 512) if "512" in name else (1024 * 1024)
        else:
            raise ValueError(name)
        self.model = self.model.to(self.dev).eval()
        if self.family == "qwen" and self.dev == "cpu":
            self.model = self.model.float()  # bf16 matmuls are slow on CPU
        torch.set_grad_enabled(False)

    @staticmethod
    def _load_qwen_vit(ckpt: Optional[str] = None):
        """Build the Qwen3-VL-8B vision tower: stock weights, or a fine-tuned Hebrew VLM's tower.

        Fine-tuned towers are read from the LM Studio MLX export of the sibling project's merged checkpoints
        (``~/.lmstudio/models/isaacmg/qwen3-vl-8b-<ckpt>``, else the NAS copy
        ``/Volumes/home/studio_offload/v19b_merge/qwen3-vl-8b-<ckpt>-mlx``), where the vision tower is stored unquantised in bf16
        under ``vision_tower.*``; the only layout difference is the channels-last patch-embedding conv.

        :param ckpt: e.g. ``heb-v21b-step1200``; None = stock Qwen3-VL-8B-Instruct.
        :returns: (vision model, image processor).
        """
        import torch
        from safetensors import safe_open
        from transformers import AutoImageProcessor
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        cfg = Qwen3VLConfig.from_pretrained(QWEN_VL_SNAPSHOT)
        model = Qwen3VLVisionModel._from_config(cfg.vision_config, torch_dtype=torch.bfloat16)
        index = json.loads((QWEN_VL_SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.visual."
        state = {}
        for shard in sorted({f for k, f in index.items() if k.startswith(prefix)}):
            with safe_open(str(QWEN_VL_SNAPSHOT / shard), framework="pt") as fh:
                for key in fh.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = fh.get_tensor(key).to(torch.bfloat16)
        if ckpt:
            mdir = Path.home() / ".lmstudio" / "models" / "isaacmg" / f"qwen3-vl-8b-{ckpt}"
            if not mdir.exists():  # checkpoints the sibling removed from LM Studio keep their MLX export on the NAS;
                # ~/audit_local/towers holds a local vision-tower-only copy (fast loads while the share is slow)
                local = Path.home() / "audit_local" / "towers" / f"qwen3-vl-8b-{ckpt}-mlx"
                mdir = local if (local / "model.safetensors.index.json").exists() else \
                    Path("/Volumes/home/studio_offload/v19b_merge") / f"qwen3-vl-8b-{ckpt}-mlx"
            wmap = json.loads((mdir / "model.safetensors.index.json").read_text())["weight_map"]
            state = {}
            for shard in sorted({f for k, f in wmap.items() if k.startswith("vision_tower.")}):
                with safe_open(str(mdir / shard), framework="pt") as fh:
                    for key in fh.keys():
                        if key.startswith("vision_tower."):
                            t = fh.get_tensor(key).to(torch.bfloat16)
                            if key.endswith("patch_embed.proj.weight") and t.dim() == 5 and t.shape[-1] == 3:
                                t = t.permute(0, 4, 1, 2, 3).contiguous()  # MLX (O,T,H,W,I) -> torch (O,I,T,H,W)
                            state[key[len("vision_tower."):]] = t
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and not [m for m in missing if "rotary" not in m], (missing[:5], unexpected[:5])
        proc = AutoImageProcessor.from_pretrained(QWEN_VL_SNAPSHOT)
        return model, proc

    def embed(self, imgs: List[Image.Image], res: Optional[int] = None) -> np.ndarray:
        """Embed images into L2-normalised vectors.

        :param imgs: PIL images.
        :param res: Optional input resolution override (patches use the model's native 224).
        :returns: (B, D) array.
        :rtype: np.ndarray
        """
        import torch

        if self.family == "handcrafted":
            from handcrafted_features import describe
            return np.stack([describe(im) for im in imgs])
        if self.family == "qwen":
            vecs = []
            for im in imgs:  # variable token counts: one image per forward
                inp = self.processor(images=[im], return_tensors="pt", min_pixels=128 * 128, max_pixels=self.max_pixels)
                dtype = torch.bfloat16 if self.dev != "cpu" else torch.float32
                out = self.model(inp["pixel_values"].to(self.dev, dtype), grid_thw=inp["image_grid_thw"].to(self.dev))
                hidden = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
                if self.readout != "merger":
                    hidden = out[1][list(self.model.deepstack_visual_indexes).index(int(self.readout[2:]))]
                vecs.append(hidden.float().mean(0).cpu().numpy())
            v = np.stack(vecs)
        else:
            x = to_tensor(imgs, res or self.res, self.family).to(self.dev)
            out = self.model(pixel_values=x)
            if self.family == "dinov2":
                # CLS concatenated with mean patch token (standard DINOv2 retrieval descriptor)
                v = torch.cat([out.last_hidden_state[:, 0], out.last_hidden_state[:, 1:].mean(1)], dim=1)
            elif self.family == "clip":
                v = out.image_embeds
            else:
                v = out.pooler_output
            v = v.float().cpu().numpy()
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_view(cid: str, view: str, k: int, patch: int) -> List[Image.Image]:
    """Load one fragment image and produce the requested view.

    :param cid: Fragment id.
    :param view: global|crop|patches.
    :param k: Patches per image.
    :param patch: Patch size.
    :returns: List of images (1 for global/crop, k for patches).
    :rtype: List[Image.Image]
    """
    img = Image.open(IMG_DIR / f"{cid.replace('/', '_')}.jpg").convert("RGB")
    if view == "global":
        return [pad_square(img)]
    crop = fragment_crop(img)
    if view == "crop":
        return [pad_square(crop)]
    if view == "rawcrop":  # unpadded crop (handcrafted descriptor measures the substrate itself)
        return [crop]
    if view == "masked":  # backdrop blanked to grey, then cropped + padded with grey
        masked, _ = fragment_mask(img)
        side = max(masked.size)
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(masked, ((side - masked.width) // 2, (side - masked.height) // 2))
        return [canvas]
    if view == "mpatches":  # ink patches sampled only inside the fragment mask
        masked, _ = fragment_mask(img)
        seed = int.from_bytes(cid.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 32)
        return ink_patches(masked, k, patch, np.random.default_rng(seed))
    seed = int.from_bytes(cid.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 32)
    return ink_patches(crop, k, patch, np.random.default_rng(seed))


def main() -> None:
    """Extract features for the selected ids."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--view", choices=["global", "crop", "patches", "rawcrop", "masked", "mpatches"], required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--patch", type=int, default=320)
    parser.add_argument("--part-size", type=int, default=500)
    args = parser.parse_args()

    sel = json.loads((AUDIT_ROOT / "image_selection_v1.json").read_text())
    ids = sorted({c for v in sel.values() for c in v if (IMG_DIR / f"{c.replace('/', '_')}.jpg").exists()})
    out_dir = AUDIT_ROOT / "cache" / "image_feats" / f"{args.encoder}__{args.view}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "SKIP").exists():  # deprioritised by hand (e.g. too slow while the GPU is shared)
        print(f"{out_dir.name} marked SKIP", flush=True)
        return
    # One writer per feature set: a CPU-side queue and the GPU queue may both reach it.
    lock = out_dir / "LOCK"
    if lock.exists():
        try:
            os.kill(int(lock.read_text().strip()), 0)
            print(f"{out_dir.name} is being extracted by pid {lock.read_text().strip()}; skipping", flush=True)
            return
        except (ProcessLookupError, ValueError):
            pass
    lock.write_text(str(os.getpid()))
    enc = None
    for p in range(0, len(ids), args.part_size):
        path = out_dir / f"part_{p // args.part_size:04d}.npz"
        if path.exists():
            continue
        enc = enc or Encoder(args.encoder)
        chunk, feats = ids[p:p + args.part_size], []
        for cid in chunk:
            views = load_view(cid, args.view, args.k, args.patch)
            if args.view in ("patches", "mpatches"):
                v = enc.embed(views, res=224 if enc.family == "dinov2" else None).mean(0)
                feats.append(v / np.linalg.norm(v))
            else:
                feats.append(enc.embed(views)[0])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(chunk), feats=np.stack(feats).astype(np.float32))
        tmp.rename(path)
        print(f"{args.encoder}/{args.view} part {p // args.part_size} done ({p + len(chunk)}/{len(ids)})", flush=True)
    lock.unlink(missing_ok=True)
    print("done", out_dir, flush=True)


if __name__ == "__main__":
    main()
