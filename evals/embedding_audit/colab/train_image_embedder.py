"""Fine-tune the Qwen3-VL-8B vision tower as a handwriting / fragment-similarity encoder (Colab A100; importable locally).

Backbone: the vision tower of our fine-tuned Hebrew VLM (``heb-v22b-step1200``, the audit's best fragment-level scribe
encoder), exported by ``export_tower.py`` as a standalone ``Qwen3VLVisionModel`` (bf16, 0.58B params).

Pipeline (each stage is a function so the notebook runs them cell by cell):
1. ``load_data``  — the private dataset repo: KTIV line crops + masked fragment crops (tar shards), labels, local
                    DINOv2 features (see ``package_image_dataset.py``).
2. ``load_tower`` — the exported tower + its (stock Qwen3-VL) image processor.
3. ``evaluate``   — base tower, the audit's pooling (mean of merged tokens, 4096-d) for every readout (final merger
                    and the deepstack taps at layers 8/16/24, one forward): I2 = held-out manuscripts, line queries,
                    positives = lines of the same manuscript on another page (P@1, mAP, AUC vs same/other-library
                    negatives); I1 = masked fragments: joins R@10 (all / cross-collection / FJP / PGP), scribe P@1,
                    script kNN, collection confound; each also fused with DINOv2.
4. ``train``      — LoRA + projection head, supervised contrastive loss over P x K manuscript batches, GradCache.
5. ``evaluate``   — tuned: the tower's own pooled vector (``pooled``) and the head output (``head``).
6. ``save_tuned`` / ``push`` — LoRA adapter, head, fp32-merged tower and the reports -> a private HF model repo.

Design choices
--------------
* **LoRA, not "unfreeze the last N blocks".** r=16 on every block's attention (``qkv``, ``proj``) and MLP
  (``linear_fc1/2``) linears plus the mergers (final + deepstack; ~8-10M trainable params). (a) The data are small (~900 training
  manuscripts, ~20k lines) and a low-rank update is a strong regulariser against forgetting what the HTR fine-tune
  taught the tower (which is why v22b beats the stock tower on scribes). (b) Hand/scribe style is a stroke-level
  property computed in early and middle blocks (the audit reads the deepstack taps at layers 8/16/24 for that
  reason); unfreezing only the top blocks cannot change it. (c) Trainable state stays tiny, so fp32 master weights +
  Adam cost nothing while the frozen base stays bf16. (d) The update merges back into a plain tower — a drop-in
  replacement for ``image_features`` (and in principle the VLM).
* **Readout + head.** ``readout`` picks which merged-token means form the pooled vector: ``merger`` (final, the
  audit's fragment pooling), ``ds8``/``ds16``/``ds24`` (deepstack taps), or a ``+`` combination (each part
  L2-normalised, concatenated, /sqrt(k)). Local I2 on the 600 test manuscripts (queue v9, h96) says the final merger
  is nearly blind to the hand at line scale (P@1 0.067) while the layer-8 tap is the best single line encoder in the
  audit (P@1 0.371 vs DINOv2 0.344); at fragment scale the merger is the best scribe encoder (P@1 0.457). Default
  ``ds8+merger`` keeps both; ``ds8`` alone truncates the tower after block 8 (~3x cheaper steps). Head: LayerNorm ->
  1024 -> GELU -> 256. The loss is applied to the head output and, with weight ``pooled_weight``, to the normalised
  pooled vector itself, so the merged tower is useful without the head (fusion with DINOv2, swapping into probes).
* **Loss.** Supervised contrastive (temperature 0.07). Positives = items of the same manuscript on a DIFFERENT page;
  same-manuscript same-page pairs are always ignored (neither positive nor negative), so one photograph's lighting,
  resolution and ruling can never be the cue.
* **Sampler.** P manuscripts x K items, items spread round-robin over the manuscript's pages. Only manuscripts with
  >= 2 training pages are drawn (a single-page manuscript could only supply same-page positives; 189 of the 898
  dataset_v1 training manuscripts are single-page). Manuscripts are drawn in blocks of ``library_block`` from the
  same holding library, so in-batch negatives share imaging conditions (counteracts the collection confound).
* **Schedule.** Linear warmup, cosine decay to 0 at the planned horizon. With ``max_minutes`` the horizon shrinks
  to the number of steps that fit in the time cap (measured after a few steps and at every validation), so a capped
  run still ends annealed instead of being cut off at a high learning rate.
* **Items.** Single lines (height 96 = the I2 protocol; augmented with a random window, letter-scale jitter,
  brightness/contrast and 20 % grayscale) or, with probability ``stack_prob``, a *line stack*: 4-8 lines of one page
  (all of them when the page has only 3)
  stacked at 28-48 px line height, ~10-14 px letters — the scale of a whole fragment at the 512x512 pixel budget —
  so the line-trained encoder also sees fragment-like inputs (I1 is evaluated on whole fragments).
* **Memory.** Frozen base in bf16, trainable params fp32 under bf16 autocast, gradient checkpointing, and GradCache:
  the batch is embedded without grad, the loss gives gradients w.r.t. the embeddings, then chunks of at most
  ``chunk_patches`` patches are re-run with grad. The batch size is therefore independent of GPU memory (A100 40 GB:
  ``chunk_patches=65536`` is comfortable).
"""

import json
import math
import os
import random
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

LINE_HEIGHT = 96
MAX_WIDTH = 2048
GRID = 32  # patch 16 x spatial merge 2: tower inputs must be multiples of this
FRAG_MIN_PIXELS, FRAG_MAX_PIXELS = 128 * 128, 512 * 512
STACK_MAX_WIDTH = 768
LORA_TARGETS = (r"(.*\.)?(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
                r"|(merger|deepstack_merger_list\.\d+)\.linear_fc[12])")
DEFAULT_READOUT = "ds8+merger"
JOIN_KEY = "joins_fjp_clean"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ----------------------------------------------------------------------------------------------------------- data
def load_data(repo_id: str, token: Optional[str] = None, local_dir: str = "/content/genizah_image_data",
              revision: Optional[str] = None, extract_dir: Optional[str] = None) -> Dict[str, object]:
    """Download the dataset repo (or use a local folder), extract the image shards once, and parse the manifests.

    :param repo_id: Private HF dataset repo id, or a local packaged folder (smoke tests).
    :param token: HF token (Colab secret ``HF_TOKEN``).
    :param local_dir: Download directory for the repo.
    :param revision: Dataset tag/commit to pin.
    :param extract_dir: Where the tar shards are unpacked (default ``<dataset>/_extracted``).
    :returns: Dict with ``lines`` (rows with ``file``), ``fragments``, ``labels``, ``dinov2``, ``local_tower``, ``path``.
    :rtype: Dict[str, object]
    """
    if Path(repo_id).is_dir():
        path = Path(repo_id)
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", token=token, local_dir=local_dir,
                                      revision=revision))
    files = Path(extract_dir) if extract_dir else path / "_extracted"
    files.mkdir(parents=True, exist_ok=True)
    for shard in sorted(path.glob("*.tar")):
        done = files / f".{shard.name}.done"
        if not done.exists():
            with tarfile.open(shard) as tar:
                tar.extractall(files, filter="data")
            done.touch()
    read = lambda name: [json.loads(l) for l in open(path / name, encoding="utf-8")]  # noqa: E731
    lines, fragments = read("lines.jsonl"), read("fragments.jsonl")
    for row in lines + fragments:
        row["file"] = str(files / row["path"])
    npz = lambda name: dict(np.load(path / name)) if (path / name).exists() else None  # noqa: E731
    return {"lines": lines, "fragments": fragments, "labels": json.loads((path / "fragment_labels.json").read_text()),
            "dinov2": npz("fragment_dinov2.npz"), "local_tower": npz("fragment_tower_local.npz"), "path": path}


def border_color(arr: np.ndarray) -> np.ndarray:
    """Median colour of the outer 3 % frame (same as ``image_features.border_color``).

    :param arr: HxWx3 uint8 image.
    :returns: RGB colour.
    :rtype: np.ndarray
    """
    h, w = arr.shape[:2]
    b = max(4, int(0.03 * min(h, w)))
    frame = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                            arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    return np.median(frame, axis=0)


def prepare_line(img: Image.Image, height: int = LINE_HEIGHT, max_width: int = MAX_WIDTH) -> Image.Image:
    """Fixed height (aspect kept), centre-crop to ``max_width``, pad the width to the 32-px grid with the border colour.

    Port of ``line_probe.tower_line_image`` working on an image; on the packaged crops (already at ``height``,
    already cropped) the resize and crop are no-ops, so test lines go through exactly the local I2 pipeline.

    :param img: RGB line crop.
    :param height: Target height (multiple of 32).
    :param max_width: Width cap (multiple of 32).
    :returns: RGB image with both sides multiples of 32.
    :rtype: Image.Image
    """
    width = max(GRID, round(img.width * height / img.height))
    img = img.resize((width, height), Image.BICUBIC)
    if width > max_width:
        x0 = (width - max_width) // 2
        img, width = img.crop((x0, 0, x0 + max_width, height)), max_width
    target = -(-width // GRID) * GRID
    if target == width:
        return img
    canvas = Image.new("RGB", (target, height), tuple(int(c) for c in border_color(np.asarray(img))))
    canvas.paste(img, ((target - width) // 2, 0))
    return canvas


def photometric(img: Image.Image, rng: random.Random) -> Image.Image:
    """Brightness/contrast jitter and 20 % grayscale (ink colour and photo white balance are not hand features).

    :param img: RGB image.
    :param rng: Item RNG.
    :returns: Augmented image.
    :rtype: Image.Image
    """
    from PIL import ImageEnhance, ImageOps

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.25))
    if rng.random() < 0.2:
        img = ImageOps.grayscale(img).convert("RGB")
    return img


def augment_line(img: Image.Image, rng: random.Random, height: int = LINE_HEIGHT) -> Image.Image:
    """Training view of one line: random window (50-100 % of the width), letter scale 80-100 %, photometric.

    :param img: Stored crop (``height`` px tall, unpadded).
    :param rng: Item RNG.
    :param height: Line height.
    :returns: Augmented RGB image, still ``height`` px tall (call :func:`prepare_line` next).
    :rtype: Image.Image
    """
    w = img.width
    if w > 4 * height and rng.random() < 0.8:
        win = rng.randint(max(4 * height, w // 2), w)
        x0 = rng.randint(0, w - win)
        img = img.crop((x0, 0, x0 + win, img.height))
    scale = rng.uniform(0.8, 1.0)
    if scale < 0.98:
        nh = round(height * scale)
        small = img.resize((max(GRID, round(img.width * nh / img.height)), nh), Image.BICUBIC)
        img = Image.new("RGB", (small.width, height), tuple(int(c) for c in border_color(np.asarray(small))))
        img.paste(small, (0, rng.randint(0, height - nh)))
    return photometric(img, rng)


def stack_lines(imgs: List[Image.Image], line_h: int, rng: random.Random, max_width: int = STACK_MAX_WIDTH) -> Image.Image:
    """Stack lines of one page into a small "page region" at fragment scale (right-aligned, Hebrew).

    :param imgs: Line crops (same page).
    :param line_h: Line height in the stack (28-48 px: letters ~10-14 px, as in a fragment at 512x512 pixels).
    :param rng: Item RNG (window position of over-wide lines).
    :param max_width: Stack width cap.
    :returns: RGB image with both sides multiples of 32.
    :rtype: Image.Image
    """
    rows = [im.resize((max(GRID, round(im.width * line_h / im.height)), line_h), Image.BICUBIC) for im in imgs]
    width = max(GRID, min(max_width, int(np.median([r.width for r in rows]))) // GRID * GRID)
    height = -(-(line_h * len(rows)) // GRID) * GRID
    bg = tuple(int(c) for c in np.median([border_color(np.asarray(r)) for r in rows], axis=0))
    canvas = Image.new("RGB", (width, height), bg)
    y = (height - line_h * len(rows)) // 2
    for r in rows:
        if r.width > width:
            x0 = rng.randint(0, r.width - width)
            r = r.crop((x0, 0, x0 + width, line_h))
        canvas.paste(r, (width - r.width, y))
        y += line_h
    return canvas


class LineItems:
    """Map-style dataset over line rows. Keys: an int (eval: the plain line) or a spec ``(kind, line_ids, seed)``
    from :class:`PKSampler` (train: ``kind`` = ``line`` | ``stack``, augmented)."""

    def __init__(self, rows: List[dict], processor, augment: bool = False, height: int = LINE_HEIGHT,
                 max_width: int = MAX_WIDTH) -> None:
        """Index the rows.

        :param rows: Line rows (``file``, ``ms``, ``page``).
        :param processor: Qwen3-VL image processor.
        :param augment: Apply training augmentation.
        :param height: Line height.
        :param max_width: Width cap.
        """
        self.rows, self.processor, self.augment = rows, processor, augment
        self.height, self.max_width = height, max_width
        self.ms_id = {m: k for k, m in enumerate(sorted({r["ms"] for r in rows}))}
        self.page_id = {p: k for k, p in enumerate(sorted({r["page"] for r in rows}))}

    def __len__(self) -> int:
        """Number of rows.

        :returns: Count.
        :rtype: int
        """
        return len(self.rows)

    def image(self, key) -> Tuple[Image.Image, int]:
        """Build the input image for a key.

        :param key: Row index or sampler spec.
        :returns: (image with sides multiple of 32, first row index).
        :rtype: Tuple[Image.Image, int]
        """
        kind, idx, seed = ("line", (key,), 0) if isinstance(key, int) else key
        rng = random.Random(seed)
        imgs = [Image.open(self.rows[i]["file"]).convert("RGB") for i in idx]
        if kind == "stack":
            return photometric(stack_lines(imgs, rng.randint(28, 48), rng), rng), idx[0]
        img = augment_line(imgs[0], rng, self.height) if self.augment else imgs[0]
        return prepare_line(img, self.height, self.max_width), idx[0]

    def __getitem__(self, key) -> dict:
        """Processor output + labels for one item.

        :param key: Row index or sampler spec.
        :returns: {pixel_values, grid_thw, ms, page}.
        :rtype: dict
        """
        img, i = self.image(key)
        inp = self.processor(images=[img], return_tensors="pt", do_resize=False)
        return {"pixel_values": inp["pixel_values"], "grid_thw": inp["image_grid_thw"],
                "ms": self.ms_id[self.rows[i]["ms"]], "page": self.page_id[self.rows[i]["page"]]}


class FragmentItems:
    """Map-style dataset over the packaged masked fragment crops (processor at the 512x512-pixel budget)."""

    def __init__(self, rows: List[dict], processor) -> None:
        """Store rows.

        :param rows: Fragment rows (``file``).
        :param processor: Qwen3-VL image processor.
        """
        self.rows, self.processor = rows, processor

    def __len__(self) -> int:
        """Number of fragments.

        :returns: Count.
        :rtype: int
        """
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        """Processor output for one fragment (same call as ``image_features.Encoder.embed``).

        :param i: Row index.
        :returns: {pixel_values, grid_thw, ms, page} (labels unused, -1).
        :rtype: dict
        """
        img = Image.open(self.rows[i]["file"]).convert("RGB")
        inp = self.processor(images=[img], return_tensors="pt", min_pixels=FRAG_MIN_PIXELS, max_pixels=FRAG_MAX_PIXELS)
        return {"pixel_values": inp["pixel_values"], "grid_thw": inp["image_grid_thw"], "ms": -1, "page": -1}


def collate(items: List[dict]) -> dict:
    """Pack variable-size images into one tower call (the tower takes concatenated patches + per-image grids).

    :param items: Dataset items.
    :returns: {pixel_values (N, 1536), grid_thw (B, 3), ms (B,), page (B,)}.
    :rtype: dict
    """
    import torch

    return {"pixel_values": torch.cat([it["pixel_values"] for it in items]),
            "grid_thw": torch.cat([it["grid_thw"] for it in items]),
            "ms": torch.tensor([it["ms"] for it in items]), "page": torch.tensor([it["page"] for it in items])}


class PKSampler:
    """Batches of P manuscripts x K items (lines or line stacks), spread over pages, library-blocked.

    Only manuscripts with at least two pages are drawn: the loss never uses same-page pairs as positives, so a
    single-page manuscript would contribute no anchor (and earlier versions turned its same-page pairs into positives,
    i.e. taught the photograph instead of the hand)."""

    def __init__(self, rows: List[dict], P: int, K: int, steps: int, seed: int = 0, stack_prob: float = 0.3,
                 library_block: int = 8) -> None:
        """Index train rows by manuscript/page/library.

        :param rows: Train line rows.
        :param P: Manuscripts per batch.
        :param K: Items per manuscript (spread round-robin over its pages, so K >= 2 gives other-page positives).
        :param steps: Batches to yield.
        :param seed: RNG seed.
        :param stack_prob: Probability that an item is a line stack (page has >= 3 lines).
        :param library_block: Manuscripts drawn together from one holding library (0 = uniform).
        """
        assert K >= 2, "K >= 2 items per manuscript are needed for other-page positives"
        self.P, self.K, self.steps, self.seed = P, K, steps, seed
        self.stack_prob, self.library_block = stack_prob, library_block
        self.pages: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        lib = {}
        for i, r in enumerate(rows):
            self.pages[r["ms"]][r["page"]].append(i)
            lib[r["ms"]] = r.get("library") or ""
        self.ms = sorted(m for m, pg in self.pages.items() if len(pg) >= 2)
        self.n_single_page = len(self.pages) - len(self.ms)
        assert len(self.ms) >= P, f"only {len(self.ms)} multi-page manuscripts for P={P}"
        self.by_lib: Dict[str, List[str]] = defaultdict(list)
        for m in self.ms:
            self.by_lib[lib[m]].append(m)

    def __len__(self) -> int:
        """Number of batches.

        :returns: ``steps``.
        :rtype: int
        """
        return self.steps

    def pick_manuscripts(self, rng: random.Random) -> List[str]:
        """Draw P distinct manuscripts, in same-library blocks when ``library_block`` > 0.

        :param rng: Sampler RNG.
        :returns: Manuscript ids.
        :rtype: List[str]
        """
        if not self.library_block:
            return rng.sample(self.ms, self.P)
        left = {lib: list(ms) for lib, ms in sorted(self.by_lib.items())}
        chosen: List[str] = []
        while len(chosen) < self.P:  # libraries weighted by their remaining manuscripts (exhausted ones drop out)
            libs = [lib for lib, ms in left.items() if ms]
            lib = rng.choices(libs, [len(left[l]) for l in libs])[0]
            take = rng.sample(left[lib], min(self.library_block, len(left[lib]), self.P - len(chosen)))
            chosen += take
            left[lib] = [m for m in left[lib] if m not in set(take)]
        return chosen

    def __iter__(self):
        """Yield batches of item specs ``(kind, line_ids, seed)``.

        :returns: Iterator of lists.
        """
        rng = random.Random(self.seed)
        for _ in range(self.steps):
            batch = []
            for m in self.pick_manuscripts(rng):
                pages = sorted(self.pages[m])
                rng.shuffle(pages)
                unused = {p: rng.sample(self.pages[m][p], len(self.pages[m][p])) for p in pages}
                for k in range(self.K):
                    page = pages[k % len(pages)]
                    lines = self.pages[m][page]
                    if self.stack_prob and len(lines) >= 3 and rng.random() < self.stack_prob:
                        batch.append(("stack", tuple(sorted(rng.sample(lines, min(len(lines), rng.randint(4, 8))))),
                                      rng.getrandbits(32)))
                    else:
                        pool = unused[page] or lines
                        line = pool.pop() if unused[page] else rng.choice(pool)
                        batch.append(("line", (line,), rng.getrandbits(32)))
            yield batch


# ---------------------------------------------------------------------------------------------------------- model
def load_tower(tower_id: str, token: Optional[str] = None, revision: Optional[str] = None,
               device: Optional[str] = None, dtype=None, attn_implementation: str = "sdpa"):
    """Load the exported tower and its image processor.

    :param tower_id: HF repo id or local folder written by ``export_tower.py``.
    :param token: HF token.
    :param revision: Repo revision.
    :param device: Target device (default cuda if available).
    :param dtype: Weight dtype (default bf16 on GPU, fp32 on CPU).
    :param attn_implementation: ``sdpa`` (default) or ``flash_attention_2`` if installed.
    :returns: (tower, processor).
    """
    import torch
    from transformers import AutoImageProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device != "cpu" else torch.float32)
    tower = Qwen3VLVisionModel.from_pretrained(tower_id, dtype=dtype, token=token, revision=revision,
                                               attn_implementation=attn_implementation)
    processor = AutoImageProcessor.from_pretrained(tower_id, token=token, revision=revision)
    return tower.to(device).eval(), processor


def build_embedder(tower, readout: str = DEFAULT_READOUT, head_dim: int = 256, head_hidden: int = 1024):
    """Wrap a tower with readout pooling + projection head.

    :param tower: ``Qwen3VLVisionModel`` (LoRA is added later by :func:`add_lora`).
    :param readout: ``merger``, ``ds<layer>`` (deepstack tap, e.g. ``ds8``) or a ``+`` combination.
    :param head_dim: Output dimension of the head.
    :param head_hidden: Hidden width of the head.
    :returns: ``TowerEmbedder`` module (``taps``, ``combine``, ``forward`` -> (pooled, head)).
    """
    import torch
    import torch.nn.functional as F

    class TowerEmbedder(torch.nn.Module):
        """Tower -> per-readout means of merged tokens -> pooled (normalised parts, concatenated) -> head."""

        def __init__(self) -> None:
            """Build the head on the tower's device."""
            super().__init__()
            cfg = tower.config
            self.tower, self.readout = tower, readout
            self.parts = readout.split("+")
            self.tap_layers = {f"ds{i}": k for k, i in enumerate(cfg.deepstack_visual_indexes)}
            bad = [p for p in self.parts if p != "merger" and p not in self.tap_layers]
            assert not bad, f"unknown readout {bad}; choose from merger, {sorted(self.tap_layers)}"
            self.merge_unit = cfg.spatial_merge_size ** 2
            self.depth = cfg.depth
            self.full_blocks = None  # set by truncate()
            dim = cfg.out_hidden_size * len(self.parts)
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, head_hidden), torch.nn.GELU(),
                                            torch.nn.Linear(head_hidden, head_dim))
            self.head.to(next(tower.parameters()).device)
            self.head_config = {"readout": readout, "in_dim": dim, "hidden": head_hidden, "out_dim": head_dim}

        def base_tower(self):
            """The underlying ``Qwen3VLVisionModel`` (inside the peft wrapper once LoRA is added).

            :returns: Vision model.
            """
            return self.tower.get_base_model() if hasattr(self.tower, "get_base_model") else self.tower

        def available_taps(self) -> List[str]:
            """Readouts this (possibly truncated) tower can produce.

            :returns: Names.
            :rtype: List[str]
            """
            n = len(self.base_tower().blocks)
            names = ["merger"] if n == self.depth else []
            return names + [name for name, k in self.tap_layers.items() if int(name[2:]) < n]

        def truncate(self) -> int:
            """Drop the blocks after the deepest tap the readout needs (only when ``merger`` is not used).

            :returns: Blocks kept.
            :rtype: int
            """
            if "merger" in self.parts:
                return self.depth
            base = self.base_tower()
            keep = max(int(p[2:]) for p in self.parts) + 1
            self.full_blocks = list(base.blocks)
            base.blocks = torch.nn.ModuleList(self.full_blocks[:keep])
            return keep

        def untruncate(self) -> None:
            """Restore the full block list (LoRA-wrapped blocks are the same objects)."""
            if self.full_blocks is not None:
                self.base_tower().blocks = torch.nn.ModuleList(self.full_blocks)
                self.full_blocks = None

        def taps(self, pixel_values, grid_thw) -> Dict[str, "torch.Tensor"]:
            """One tower call; mean of merged tokens per image for every available readout (fp32, unnormalised).

            :param pixel_values: Packed patches (N, 1536).
            :param grid_thw: Per-image grids (B, 3).
            :returns: {readout: (B, out_hidden)}.
            """
            hidden, deepstack = self.tower(pixel_values, grid_thw=grid_thw)
            counts = (grid_thw.prod(-1) // self.merge_unit).to(hidden.device)
            seg = torch.repeat_interleave(torch.arange(len(counts), device=hidden.device), counts)
            out = {}
            for name in self.available_taps():
                h = hidden if name == "merger" else deepstack[self.tap_layers[name]]
                sums = torch.zeros(len(counts), h.shape[-1], device=h.device, dtype=torch.float32).index_add_(0, seg, h.float())
                out[name] = sums / counts.float()[:, None]
            return out

        def combine(self, taps: Dict[str, "torch.Tensor"]):
            """Pooled vector of the configured readout: normalised parts concatenated / sqrt(k) (unit norm).

            :param taps: Output of :meth:`taps`.
            :returns: (B, out_hidden * k) tensor.
            """
            return torch.cat([F.normalize(taps[p], dim=-1) for p in self.parts], dim=-1) / math.sqrt(len(self.parts))

        def forward(self, pixel_values, grid_thw):
            """Normalised pooled and head vectors.

            :param pixel_values: Packed patches.
            :param grid_thw: Per-image grids.
            :returns: (pooled, head) tensors, both L2-normalised.
            """
            p = self.combine(self.taps(pixel_values, grid_thw))
            return F.normalize(p, dim=-1), F.normalize(self.head(p).float(), dim=-1)

    return TowerEmbedder()


def add_lora(model, r: int = 16, alpha: int = 32, gradient_checkpointing: bool = True) -> int:
    """Freeze the tower, add LoRA to the vision blocks' attention/MLP linears and the merger, keep the head trainable.

    LoRA dropout is 0 so GradCache's two forward passes are identical; adapters stay fp32 on a bf16 base
    (peft ``autocast_adapter_dtype``), and the head is fp32.

    :param model: Output of :func:`build_embedder`.
    :param r: LoRA rank.
    :param alpha: LoRA alpha.
    :param gradient_checkpointing: Recompute block activations in backward (non-reentrant).
    :returns: Number of trainable parameters.
    :rtype: int

    If the readout uses only deepstack taps, the tower is first truncated after the deepest one (see
    ``TowerEmbedder.truncate``); :func:`save_tuned` restores the full tower before merging.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    kept = model.truncate()
    tower = model.tower
    for p in tower.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")
    model.tower = get_peft_model(tower, cfg)
    if gradient_checkpointing:  # after wrapping: peft's input-grad hook needs text embeddings the tower lacks, and
        # non-reentrant checkpointing does not need inputs that require grad
        tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head.float()
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    print(f"LoRA on {kept}/{model.depth} blocks (readout {model.readout})", flush=True)
    return sum(p.numel() for p in trainable)


# -------------------------------------------------------------------------------------------------------- metrics
def pair_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(pos > neg) via ranks (same as ``eval_images.pair_auc``).

    :param pos: Positive-pair similarities.
    :param neg: Negative-pair similarities.
    :returns: AUC.
    :rtype: float
    """
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def i2_metrics(X: np.ndarray, ms: Sequence[str], page: Sequence[str], lib: Optional[Sequence[str]] = None,
               block: int = 512) -> dict:
    """I2 line retrieval (vectorised port of ``line_probe.main``'s scoring; identical numbers).

    Every line queries all lines on OTHER pages; relevant = same manuscript. P@1, mAP, and the AUC of
    same-manuscript pairs vs different-manuscript pairs from the same / another holding library.

    :param X: (n, D) L2-normalised features.
    :param ms: Manuscript per line.
    :param page: Page per line.
    :param lib: Holding library per line (None = skip the AUCs).
    :param block: Query rows per block (memory).
    :returns: Metrics dict.
    :rtype: dict
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(ms)
    ms_a, page_a = np.asarray(ms), np.asarray(page)
    lib_a = np.asarray([l or "" for l in lib]) if lib is not None else None
    p1, aps, chance = [], [], []
    pos_s, neg_same, neg_other = [], [], []
    ranks = np.arange(1, n + 1)
    for s in range(0, n, block):
        with np.errstate(all="ignore"):  # spurious macOS Accelerate FP flags; values are finite (checked in embed)
            S = X[s:s + block] @ X.T
        valid = page_a[s:s + block, None] != page_a[None, :]
        rel = (ms_a[s:s + block, None] == ms_a[None, :]) & valid
        order = np.argsort(-np.where(valid, S, -np.inf), axis=1, kind="stable")
        relo = np.take_along_axis(rel, order, axis=1)
        p1.append(relo[:, 0])
        hits = np.cumsum(relo, axis=1)
        aps.append(((hits / ranks) * relo).sum(1) / np.maximum(rel.sum(1), 1))
        chance.append(rel.sum(1) / valid.sum(1))
        if lib_a is not None:
            same_lib = lib_a[s:s + block, None] == lib_a[None, :]
            neg = valid & ~rel
            pos_s.append(S[rel])
            neg_same.append(S[neg & same_lib])
            neg_other.append(S[neg & ~same_lib])
    out = {"n_lines": n, "n_ms": len(set(ms)), "P@1_other_page_same_ms": round(float(np.concatenate(p1).mean()), 4),
           "mAP": round(float(np.concatenate(aps).mean()), 4), "chance_P@1": round(float(np.concatenate(chance).mean()), 5)}
    if lib_a is not None:
        pos, ns, no = np.concatenate(pos_s), np.concatenate(neg_same), np.concatenate(neg_other)
        out["auc_vs_same_library"] = round(pair_auc(pos, ns), 4) if len(ns) else None
        out["auc_vs_other_library"] = round(pair_auc(pos, no), 4) if len(no) else None
    return out


def balanced_knn(sims: np.ndarray, labels: List[str], k: int = 10) -> float:
    """Leave-one-out kNN balanced accuracy (same as ``eval_images.balanced_knn``).

    :param sims: Square similarity matrix.
    :param labels: Label per item.
    :param k: Neighbours.
    :returns: Balanced accuracy.
    :rtype: float
    """
    s = sims.copy()
    np.fill_diagonal(s, -9)
    nn = np.argpartition(-s, k, axis=1)[:, :k]
    pred = [Counter(labels[j] for j in row).most_common(1)[0][0] for row in nn]
    per = defaultdict(list)
    for p, y in zip(pred, labels):
        per[y].append(p == y)
    return float(np.mean([np.mean(v) for v in per.values()]))


def fragment_metrics(X: np.ndarray, labels: dict) -> dict:
    """I1 fragment metrics (port of ``eval_images.evaluate``; identical numbers on identical features).

    :param X: (n, D) L2-normalised features in ``labels["ids"]`` order.
    :param labels: ``fragment_labels.json``.
    :returns: Results dict (joins, scribes, script, confound_collection).
    :rtype: dict
    """
    ids = labels["ids"]
    pos = {c: i for i, c in enumerate(ids)}
    coll = [labels["collection"][c] for c in ids]
    with np.errstate(all="ignore"):  # numpy 2 + macOS Accelerate raise spurious FP flags in float32 matmul
        S = X @ X.T
    rng = np.random.default_rng(0)
    res: dict = {"n_images": len(ids)}
    partners: Dict[int, set] = defaultdict(set)
    source: Dict[int, set] = defaultdict(set)
    for key in (JOIN_KEY, "joins_pgp"):
        for a, b in labels[key]:
            if a in pos and b in pos:
                partners[pos[a]].add(pos[b])
                partners[pos[b]].add(pos[a])
                source[pos[a]].add(key)
                source[pos[b]].add(key)
    ranks, same_c, cross_c, by_src = [], [], [], defaultdict(list)
    for q, ps in partners.items():
        s = S[q].copy()
        s[q] = -9
        order = np.argsort(-s)
        r = int(min(np.where(np.isin(order, list(ps)))[0])) + 1
        ranks.append(r)
        (same_c if any(coll[p] == coll[q] for p in ps) else cross_c).append(r)
        for src in source[q]:
            by_src[src].append(r)

    def rstats(rs):
        rs = np.array(rs)
        return {"n": int(len(rs)), "R@1": round(float((rs <= 1).mean()), 4), "R@10": round(float((rs <= 10).mean()), 4),
                "R@100": round(float((rs <= 100).mean()), 4), "MRR": round(float((1 / rs).mean()), 4),
                "median_rank": int(np.median(rs))} if len(rs) else {"n": 0}
    pos_pairs = [(q, p) for q, ps in partners.items() for p in ps if q < p]
    join_sims = np.array([S[a, b] for a, b in pos_pairs])
    by_coll = defaultdict(list)
    for i, c in enumerate(coll):
        by_coll[c].append(i)
    neg = []
    for a, b in pos_pairs:
        cands = by_coll[coll[a]]
        for _ in range(3):
            j = int(rng.choice(cands))
            if j != a and j not in partners[a]:
                neg.append(S[a, j])
    prior_r10 = float(np.mean([min(1.0, 10 * len(ps) / max(1, len(by_coll[coll[q]]) - 1)) for q, ps in partners.items()])) if partners else 0.0
    res["joins"] = {"all": rstats(ranks), "same_collection": rstats(same_c), "cross_collection": rstats(cross_c),
                    "fjp_literary": rstats(by_src[JOIN_KEY]), "pgp_documentary": rstats(by_src["joins_pgp"]),
                    "auc_vs_same_collection_nonjoin": round(pair_auc(join_sims, np.array(neg)), 4) if neg else None,
                    "mean_sim_join": round(float(join_sims.mean()), 4) if len(join_sims) else None,
                    "mean_sim_samecoll_nonjoin": round(float(np.mean(neg)), 4) if neg else None,
                    "collection_prior_R@10_upper": round(prior_r10, 4)}
    same_doc = set()
    for a, b in labels["joins_pgp"]:
        same_doc.add((a, b))
        same_doc.add((b, a))
    sc = [(pos[c], s) for c, s in labels["scribe"].items() if c in pos]
    if sc:
        idx = [i for i, _ in sc]
        lab = [s for _, s in sc]
        sub = S[np.ix_(idx, idx)]
        p1, aps, sp, dp = [], [], [], []
        for qi in range(len(idx)):
            valid = [j for j in range(len(idx)) if j != qi and (ids[idx[qi]], ids[idx[j]]) not in same_doc]
            rel = np.array([lab[j] == lab[qi] for j in valid])
            if rel.sum() == 0:
                continue
            order = np.argsort(-sub[qi, valid])
            relo = rel[order]
            p1.append(relo[0])
            hits = np.cumsum(relo)
            aps.append(float(((hits / np.arange(1, len(relo) + 1)) * relo).sum() / relo.sum()))
            for j, r in zip(valid, rel):
                (sp if r else dp).append(sub[qi, j])
        if p1:
            chance_p1 = float(np.mean([(Counter(lab)[l] - 1) / (len(lab) - 1) for l in lab]))
            res["scribes"] = {"n_fragments": len(idx), "n_scribes": len(set(lab)), "P@1": round(float(np.mean(p1)), 4),
                              "mAP": round(float(np.mean(aps)), 4), "chance_P@1": round(chance_p1, 4),
                              "auc_same_vs_diff_scribe": round(pair_auc(np.array(sp), np.array(dp)), 4)}
    res["script"] = {}
    for key, classes in (("script_style", {"Square", "Semi-Cursive", "Cursive", "Naskhi", "Rabbinical"}),
                         ("script_region", {"Oriental", "Spanish", "Yemenite", "Italian", "Ashkenazi", "North African", "Syrian"})):
        items = [(pos[c], a[key]) for c, a in labels["attrs"].items() if c in pos and a.get(key) in classes]
        cnt = Counter(l for _, l in items)
        items = [(i, l) for i, l in items if cnt[l] >= 8]
        if len(items) > 20:
            idx = [i for i, _ in items]
            res["script"][key] = {"n": len(items), "knn10_balanced_acc": round(balanced_knn(S[np.ix_(idx, idx)], [l for _, l in items]), 4),
                                  "chance": round(1 / len({l for _, l in items}), 4)}
    top = [c for c, n in Counter(coll).most_common(12) if n >= 30]
    idx = [i for i, c in enumerate(coll) if c in top]
    if len(top) >= 2:
        res["confound_collection"] = {"n": len(idx), "n_collections": len(top),
                                      "knn10_balanced_acc": round(balanced_knn(S[np.ix_(idx, idx)], [coll[i] for i in idx]), 4),
                                      "chance": round(1 / len(top), 4)}
    return res


def fuse(parts: List[np.ndarray]) -> np.ndarray:
    """Equal-weight late fusion of L2-normalised blocks: concatenate and renormalise (= mean of the cosines).

    Same arithmetic as ``eval_images.load_feats`` (bit-identical fused features), so the blocks must already be
    unit-norm (every packaged feature and every :func:`embed` output is).

    :param parts: L2-normalised feature arrays with the same row order.
    :returns: Fused, normalised features.
    :rtype: np.ndarray
    """
    for p in parts:
        assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-3), "fuse expects L2-normalised blocks"
    X = np.concatenate(parts, axis=1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------------ embedding
def _autocast(device, bf16: bool):
    """bf16 autocast context on CUDA (no-op elsewhere).

    :param device: torch device.
    :param bf16: Enable.
    :returns: Context manager.
    """
    import torch

    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")


def embed(model, dataset, batch_size: int = 32, num_workers: int = 4, bf16: bool = True,
          kinds: Sequence[str] = ("pooled", "head")) -> Dict[str, np.ndarray]:
    """Embed a dataset without grad; every requested kind comes from the same tower call.

    :param model: Embedder.
    :param dataset: ``LineItems`` (no augmentation) or ``FragmentItems``.
    :param batch_size: Images per tower call.
    :param num_workers: DataLoader workers.
    :param bf16: Autocast on CUDA.
    :param kinds: ``pooled`` (configured readout), ``head``, and/or single readouts (``merger``, ``ds8`` ...).
    :returns: {kind: (n, D) float32 L2-normalised}; readouts the (truncated) tower cannot produce are omitted.
    :rtype: Dict[str, np.ndarray]
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    dtype = next(model.tower.parameters()).dtype
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    was_training = model.training
    model.eval()
    taps_wanted = [k for k in kinds if k in model.available_taps()]
    out: Dict[str, list] = defaultdict(list)
    with torch.no_grad(), _autocast(device, bf16):
        for b in loader:
            taps = model.taps(b["pixel_values"].to(device, dtype), b["grid_thw"].to(device))
            if "pooled" in kinds or "head" in kinds:
                p = model.combine(taps)
                out["pooled"].append(F.normalize(p, dim=-1).float().cpu().numpy())
                out["head"].append(F.normalize(model.head(p).float(), dim=-1).cpu().numpy())
            for k in taps_wanted:
                out[k].append(F.normalize(taps[k], dim=-1).cpu().numpy())
    model.train(was_training)
    res = {k: np.concatenate(v) for k, v in out.items() if k in kinds}
    assert all(np.isfinite(v).all() for v in res.values()), "non-finite embeddings"
    return res


def evaluate(model, data: dict, processor, label: str, kinds: Sequence[str] = ("pooled",),
             splits: Sequence[str] = ("test",), fragments: bool = True, batch_lines: int = 64,
             batch_frags: int = 16, num_workers: int = 4, bf16: bool = True) -> dict:
    """The audit's held-out image metrics for one model state.

    :param model: Embedder (base = before :func:`add_lora`; ``head`` is untrained there).
    :param data: Output of :func:`load_data`.
    :param processor: Image processor.
    :param label: Report name.
    :param kinds: ``pooled`` (configured readout), ``head``, and/or single readouts (``merger``, ``ds8``, ``ds16``,
        ``ds24``) — all from one tower call per batch.
    :param splits: Line splits to score with the I2 protocol (``test`` = the audit's 600 held-out manuscripts).
    :param fragments: Also run I1 on the masked fragments (+ DINOv2 fusion when the features are packaged).
    :param batch_lines: Lines per tower call.
    :param batch_frags: Fragments per tower call.
    :param num_workers: DataLoader workers.
    :param bf16: Autocast on CUDA.
    :returns: Report dict (keys ``i2_<split>_<kind>``, ``i1_<kind>``, ``i1_<kind>+dinov2``, ``i1_dinov2_only``).
    :rtype: dict
    """
    t0 = time.time()
    out: dict = {"model": label, "readout": model.readout}
    for split in splits:
        rows = [r for r in data["lines"] if r["split"] == split]
        vecs = embed(model, LineItems(rows, processor), batch_lines, num_workers, bf16, kinds)
        for kind in kinds:
            if kind in vecs:
                out[f"i2_{split}_{kind}"] = i2_metrics(vecs[kind], [r["ms"] for r in rows], [r["page"] for r in rows],
                                                       [r.get("library", "") for r in rows])
    if fragments:
        labels = data["labels"]
        rows = {r["id"]: r for r in data["fragments"]}
        frows = [rows[c] for c in labels["ids"]]
        vecs = embed(model, FragmentItems(frows, processor), batch_frags, num_workers, bf16, kinds)
        dino, loc = data.get("dinov2"), data.get("local_tower")
        if loc is not None and "merger" in vecs and loc["feats"].shape == vecs["merger"].shape:
            # the local base cache is the merger readout from the full-resolution masked view (base: expect >~0.99)
            L = loc["feats"].astype(np.float32)
            L /= np.linalg.norm(L, axis=1, keepdims=True)
            out["fragments_merger_cos_vs_local_base_cache"] = round(float((L * vecs["merger"]).sum(1).mean()), 4)
        for kind in kinds:
            if kind not in vecs:
                continue
            out[f"i1_{kind}"] = fragment_metrics(vecs[kind], labels)
            if dino is not None:
                out[f"i1_{kind}+dinov2"] = fragment_metrics(fuse([dino["masked"], dino["mpatches"], vecs[kind]]), labels)
        if dino is not None:
            out["i1_dinov2_only"] = fragment_metrics(fuse([dino["masked"], dino["mpatches"]]), labels)
    out["eval_seconds"] = round(time.time() - t0, 1)
    return out


def summary_row(report: dict) -> dict:
    """Headline numbers of one report (for side-by-side printing).

    :param report: Output of :func:`evaluate`.
    :returns: Flat dict.
    :rtype: dict
    """
    row = {"model": report["model"]}
    for key, val in report.items():
        if key.startswith("i2_"):
            row[f"{key}:P@1"] = val["P@1_other_page_same_ms"]
            row[f"{key}:mAP"] = val["mAP"]
        elif key.startswith("i1_"):
            j = val["joins"]
            row[f"{key}:joinsR@10"] = j["all"].get("R@10")
            row[f"{key}:crossR@10"] = j["cross_collection"].get("R@10")
            row[f"{key}:fjpR@10"] = j["fjp_literary"].get("R@10")
            row[f"{key}:pgpR@10"] = j["pgp_documentary"].get("R@10")
            row[f"{key}:scribeP@1"] = val.get("scribes", {}).get("P@1")
            row[f"{key}:collection_knn"] = val.get("confound_collection", {}).get("knn10_balanced_acc")
    return row


# ------------------------------------------------------------------------------------------------------- training
def supcon_loss(z, ms, page, temperature: float):
    """Supervised contrastive loss; positives = same manuscript on another page; same-page pairs always ignored.

    Rows without an other-page positive are not anchors (they still serve as negatives for the other rows).

    :param z: (B, D) normalised embeddings (fp32).
    :param ms: (B,) manuscript labels.
    :param page: (B,) page labels (page ids are unique across manuscripts).
    :param temperature: Softmax temperature.
    :returns: Scalar loss.
    """
    import torch

    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same_ms = ms[:, None] == ms[None, :]
    same_page = page[:, None] == page[None, :]
    pos = same_ms & ~same_page
    ignore = eye | (same_ms & same_page)
    logits = (z @ z.T / temperature).masked_fill(ignore, float("-inf"))
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per = -torch.where(pos, logp, torch.zeros_like(logp)).sum(1) / pos.sum(1).clamp(min=1)
    keep = pos.any(1)
    return per[keep].mean()


def patch_chunks(grid_thw, max_patches: int) -> List[Tuple[int, int, int, int]]:
    """Split a packed batch into consecutive chunks of at most ``max_patches`` patches (>= 1 image each).

    :param grid_thw: (B, 3) grids.
    :param max_patches: Patch budget per chunk.
    :returns: List of (item_start, item_end, patch_start, patch_end).
    :rtype: List[Tuple[int, int, int, int]]
    """
    counts = grid_thw.prod(-1).tolist()
    chunks, i0, p0, used = [], 0, 0, 0
    for i, c in enumerate(counts):
        if used and used + c > max_patches:
            chunks.append((i0, i, p0, p0 + used))
            i0, p0, used = i, p0 + used, 0
        used += c
    chunks.append((i0, len(counts), p0, p0 + used))
    return chunks


def gradcache_step(model, batch: dict, temperature: float, pooled_weight: float, chunk_patches: int,
                   bf16: bool) -> float:
    """One GradCache forward/backward: no-grad embedding of the whole batch, loss, then chunked re-forward with grad.

    Gradients accumulate in ``.grad``; the caller steps the optimizer.

    :param model: LoRA embedder.
    :param batch: Collated batch.
    :param temperature: SupCon temperature.
    :param pooled_weight: Weight of the loss on the pooled vector.
    :param chunk_patches: Patch budget per grad chunk.
    :param bf16: Autocast on CUDA.
    :returns: Loss value.
    :rtype: float
    """
    import torch

    device = next(model.parameters()).device
    dtype = next(model.tower.parameters()).dtype
    pv, grid = batch["pixel_values"].to(device, dtype), batch["grid_thw"].to(device)
    ms, page = batch["ms"].to(device), batch["page"].to(device)
    chunks = patch_chunks(grid, chunk_patches)
    model.eval()
    with torch.no_grad(), _autocast(device, bf16):
        reps = [model(pv[a:b], grid[i:j]) for i, j, a, b in chunks]
    P = torch.cat([r[0] for r in reps]).float().requires_grad_(bool(pooled_weight))
    Z = torch.cat([r[1] for r in reps]).float().requires_grad_()
    loss = supcon_loss(Z, ms, page, temperature)
    if pooled_weight:
        loss = loss + pooled_weight * supcon_loss(P, ms, page, temperature)
    loss.backward()
    model.train()
    for i, j, a, b in chunks:
        with _autocast(device, bf16):
            p, z = model(pv[a:b], grid[i:j])
        outs, grads = [z.float()], [Z.grad[i:j]]
        if pooled_weight:  # with pooled_weight=0 the pooled vector has no loss and P.grad is None
            outs.append(p.float())
            grads.append(P.grad[i:j])
        torch.autograd.backward(outs, grads)
    return float(loss.item())


def trainable_state(model) -> Dict[str, object]:
    """CPU copy of the trainable tensors (LoRA + head) for best-checkpoint keeping.

    :param model: LoRA embedder.
    :returns: {name: tensor}.
    :rtype: Dict[str, object]
    """
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def train(model, data: dict, processor, steps: int = 1000, P: int = 48, K: int = 4, lr: float = 2e-4,
          head_lr: float = 1e-3, weight_decay: float = 0.01, warmup: int = 50, temperature: float = 0.07,
          pooled_weight: float = 0.5, stack_prob: float = 0.3, library_block: int = 8, chunk_patches: int = 65536,
          lora_r: int = 16, lora_alpha: int = 32, eval_every: int = 200, num_workers: int = 8, bf16: bool = True,
          seed: int = 13, log_every: int = 10, max_minutes: Optional[float] = None, select_on: str = "mean",
          calib_after: int = 5) -> List[dict]:
    """LoRA + head fine-tune with GradCache SupCon; keeps the best state by validation I2 mAP.

    The deliverables are the merged tower (``pooled``) and the head, so by default the best state is chosen on the
    mean of their validation mAPs (``select_on``), not on the head alone.

    :param model: Base embedder from :func:`build_embedder` (LoRA is added here).
    :param data: Output of :func:`load_data`.
    :param processor: Image processor.
    :param steps: Optimizer steps.
    :param P: Manuscripts per batch.
    :param K: Items per manuscript.
    :param lr: LoRA learning rate.
    :param head_lr: Head learning rate.
    :param weight_decay: AdamW weight decay.
    :param warmup: Linear warmup steps (then cosine to 0).
    :param temperature: SupCon temperature.
    :param pooled_weight: Weight of the SupCon loss on the pooled vector.
    :param stack_prob: Probability of a line-stack item.
    :param library_block: Same-library manuscript block size in the sampler (0 = uniform).
    :param chunk_patches: GradCache chunk budget in patches (memory knob).
    :param lora_r: LoRA rank.
    :param lora_alpha: LoRA alpha.
    :param eval_every: Validation interval in steps (0 = only at the end).
    :param num_workers: DataLoader workers.
    :param bf16: Autocast on CUDA.
    :param seed: Seed.
    :param log_every: Print interval.
    :param max_minutes: Optional wall-clock cap. The cosine horizon shrinks to the steps that fit (re-measured
        after ``calib_after`` + 10 steps and at every validation) so the run ends annealed; the cap itself stays a
        hard stop. The best validation state is kept either way.
    :param select_on: Validation mAP that picks the best state: ``head``, ``pooled`` or ``mean`` of the two.
    :param calib_after: Steps excluded from the step-time measurement (worker start-up, allocator warm-up).
    :returns: Training history (losses + validation rows).
    :rtype: List[dict]
    """
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    assert select_on in ("head", "pooled", "mean"), select_on
    n_train = add_lora(model, lora_r, lora_alpha)
    rows = [r for r in data["lines"] if r["split"] == "train"]
    sampler = PKSampler(rows, P, K, steps, seed, stack_prob, library_block)
    loader = DataLoader(LineItems(rows, processor, augment=True), batch_sampler=sampler, num_workers=num_workers,
                        collate_fn=collate, persistent_workers=num_workers > 0)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": lora, "lr": lr}, {"params": head, "lr": head_lr}], weight_decay=weight_decay)
    horizon = [steps]  # cosine horizon; shrinks to what fits in max_minutes

    def lr_factor(s: int) -> float:
        """Linear warmup, then cosine to 0 at the current horizon.

        :param s: Scheduler step.
        :returns: LR multiplier.
        :rtype: float
        """
        return min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, horizon[0]))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    print(f"trainable params {n_train:,} (LoRA {sum(p.numel() for p in lora):,}, head {sum(p.numel() for p in head):,}); "
          f"{len(rows)} train lines / {len(sampler.ms)} multi-page manuscripts ({sampler.n_single_page} single-page "
          f"manuscripts not drawn); batch {P}x{K}", flush=True)
    history, best, best_score, best_step, t0 = [], None, -1.0, None, time.time()
    t_cal: List[float] = []  # wall clock at the end of step calib_after (start of the rate measurement)
    has_val = any(r["split"] == "val" for r in data["lines"])

    def fit_horizon(done: int) -> None:
        """Shrink the cosine horizon to the steps that still fit in ``max_minutes`` at the rate measured so far.

        :param done: Steps completed.
        """
        if not max_minutes or not t_cal or done <= calib_after:
            return
        per = (time.time() - t_cal[0]) / (done - calib_after)  # includes validations run so far
        left = max_minutes * 60 - (time.time() - t0)
        fit = done + int(0.95 * left / per)
        if fit < horizon[0]:
            horizon[0] = max(done + 1, fit)
            print(f"time cap: cosine horizon -> {horizon[0]} steps ({per:.2f} s/step measured)", flush=True)

    def validate(step: int) -> None:
        """Validation I2 on the val manuscripts; keep the best trainable state.

        :param step: Steps completed.
        """
        nonlocal best, best_score, best_step
        rep = evaluate(model, data, processor, f"step{step}", kinds=("pooled", "head"), splits=("val",),
                       fragments=False, num_workers=min(num_workers, 4), bf16=bf16)
        maps = {"head": rep["i2_val_head"]["mAP"], "pooled": rep["i2_val_pooled"]["mAP"]}
        score = maps[select_on] if select_on in maps else (maps["head"] + maps["pooled"]) / 2
        row = {"step": step, "select_score": round(score, 4),
               **{k: v for k, v in summary_row(rep).items() if k != "model"}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score, best_step, best = score, step, trainable_state(model)

    model.train()
    step = -1
    for step, batch in enumerate(loader):
        ts = time.time()
        loss = gradcache_step(model, batch, temperature, pooled_weight, chunk_patches, bf16)
        torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step + 1 == calib_after:
            t_cal.append(time.time())
        if step % log_every == 0 or step + 1 == horizon[0]:
            row = {"step": step, "loss": round(loss, 4), "lr": sched.get_last_lr()[0], "s_per_step": round(time.time() - ts, 2),
                   "patches": int(batch["grid_thw"].prod(-1).sum()), "items": int(len(batch["ms"])),
                   "horizon": horizon[0], "elapsed_min": round((time.time() - t0) / 60, 1)}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step + 1 == calib_after + 10:
            fit_horizon(step + 1)
        if has_val and eval_every and (step + 1) % eval_every == 0 and step + 1 < horizon[0]:
            validate(step + 1)
            fit_horizon(step + 1)
        if step + 1 >= horizon[0]:
            break
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            print(f"stopping at step {step + 1}: {max_minutes} min cap", flush=True)
            break
    if has_val:
        validate(step + 1)
        if best is not None:
            model.load_state_dict(best, strict=False)
            print(f"restored best validation state: step {best_step}, val {select_on} mAP {best_score:.4f}", flush=True)
            history.append({"restored_step": best_step, "select_on": select_on, "select_score": round(best_score, 4)})
    model.eval()
    return history


# ------------------------------------------------------------------------------------------------- save / reload
def save_tuned(model, processor, out_dir: str, reports: List[dict], history: List[dict], train_args: dict,
               merge: bool = True) -> Path:
    """Write the LoRA adapter, head, fp32-merged tower, reports and a model card.

    Merging happens in fp32: LoRA deltas are small, and adding them to bf16 weights would round many of them away.
    The merged tower replaces the LoRA tower in ``model`` (the adapter is saved first).

    :param model: Trained embedder.
    :param processor: Image processor (saved with the merged tower).
    :param out_dir: Output folder.
    :param reports: Eval reports (base, tuned).
    :param history: Training history.
    :param train_args: Hyper-parameters used.
    :param merge: Also write ``tower_merged/``.
    :returns: Output path.
    :rtype: Path
    """
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.tower.save_pretrained(out / "lora_adapter")
    torch.save({k: v.float().cpu() for k, v in model.head.state_dict().items()}, out / "head.pt")
    (out / "head_config.json").write_text(json.dumps(model.head_config, indent=1))
    (out / "genizah_image_eval_report.json").write_text(json.dumps(reports, indent=1))
    (out / "train_history.json").write_text(json.dumps({"args": train_args, "history": history}, indent=1))
    if merge:
        model.untruncate()
        merged = model.tower.to(torch.float32).merge_and_unload()
        merged.save_pretrained(out / "tower_merged", safe_serialization=True)
        processor.save_pretrained(out / "tower_merged")
        model.tower = merged
    rows = [summary_row(r) for r in reports]
    (out / "README.md").write_text(
        "---\nlicense: other\n---\n# Genizah handwriting/fragment encoder (Qwen3-VL-8B vision tower + LoRA)\n\n"
        "Base tower: our Hebrew VLM's (`heb-v22b-step1200`), exported with `export_tower.py`. Fine-tuned with "
        "`train_image_embedder.py` (supervised contrastive, same-manuscript KTIV line crops, P x K batches).\n\n"
        "* `tower_merged/` — fp32 `Qwen3VLVisionModel` with the LoRA merged. `pooled` vector for readout "
        f"`{model.head_config['readout']}`: per readout part (`merger` = final merged tokens, `dsN` = deepstack tap "
        "after block N) the mean over the image's merged tokens, each part L2-normalised, concatenated, renormalised "
        "(`load_tuned` + `embed` reproduce it).\n"
        "* `lora_adapter/` + `head.pt` / `head_config.json` — the trainable parts; `head` vector = head(pooled).\n"
        "* `genizah_image_eval_report.json` — I2 (held-out manuscripts) and I1 (fragments) for base vs tuned.\n\n"
        "Headline numbers:\n\n```\n" + "\n".join(json.dumps(r) for r in rows) + "\n```\n")
    return out


def load_tuned(out_dir: str, device: Optional[str] = None):
    """Reload a saved fine-tune (merged tower + head) for inference.

    :param out_dir: Folder written by :func:`save_tuned`.
    :param device: Target device.
    :returns: (embedder, processor).
    """
    import torch

    cfg = json.loads((Path(out_dir) / "head_config.json").read_text())
    tower, processor = load_tower(str(Path(out_dir) / "tower_merged"), device=device, dtype=torch.float32)
    model = build_embedder(tower, cfg["readout"], cfg["out_dim"], cfg["hidden"])  # full depth (taps unaffected)
    model.head.load_state_dict(torch.load(Path(out_dir) / "head.pt", map_location="cpu"))
    return model.eval(), processor


def push(out_dir: str, repo_id: str, token: str, commit_message: str = "genizah image encoder fine-tune") -> str:
    """Upload the saved fine-tune to a private HF model repo; return the commit sha.

    :param out_dir: Folder written by :func:`save_tuned`.
    :param repo_id: Target model repo.
    :param token: HF token with write access.
    :param commit_message: Commit message.
    :returns: Commit sha.
    :rtype: str
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=True, exist_ok=True)
    return api.upload_folder(folder_path=out_dir, repo_id=repo_id, commit_message=commit_message).oid


# ------------------------------------------------------------------------------------------------------ smoke test
def gradcache_check(model, batch: dict, temperature: float, pooled_weight: float, chunk_patches: int) -> float:
    """Max relative difference between GradCache gradients and plain full-batch gradients (should be ~1e-6 in fp32).

    :param model: LoRA embedder (fp32, CPU).
    :param batch: Collated batch.
    :param temperature: SupCon temperature.
    :param pooled_weight: Pooled-loss weight.
    :param chunk_patches: Small budget so several chunks are used.
    :returns: Max relative gradient difference.
    :rtype: float
    """
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    gradcache_step(model, batch, temperature, pooled_weight, chunk_patches, bf16=False)
    params = [p for p in params if p.grad is not None]  # LoRA on readouts the loss does not use gets no gradient
    g1 = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    model.train()
    p, z = model(batch["pixel_values"], batch["grid_thw"])
    loss = supcon_loss(z, batch["ms"], batch["page"], temperature)
    if pooled_weight:
        loss = loss + pooled_weight * supcon_loss(p, batch["ms"], batch["page"], temperature)
    loss.backward()
    g2 = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    return max(float((a - b).abs().max() / (b.abs().max() + 1e-12)) for a, b in zip(g1, g2))


def main() -> None:
    """CPU smoke test: tiny tower, a few dozen lines, 2 steps, the full eval path, save + merge + reload."""
    import argparse

    import torch

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data", required=True, help="packaged dataset folder (package_image_dataset.py --smoke)")
    parser.add_argument("--tower", required=True, help="tower folder (export_tower.py --tiny for smoke tests)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--readout", default="ds0+merger", help="tiny tower taps are ds0/ds1; 'ds0' tests truncation")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--P", type=int, default=4)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--chunk-patches", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-minutes", type=float, default=None, help="exercise the time-cap horizon")
    parser.add_argument("--pooled-weight", type=float, default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("AUDIT_THREADS", "2")))
    t0 = time.time()
    data = load_data(args.data, extract_dir=str(Path(args.out) / "extracted"))
    print({"lines": Counter(r["split"] for r in data["lines"]), "fragments": len(data["fragments"])}, flush=True)
    tower, processor = load_tower(args.tower, device="cpu", dtype=torch.float32)
    model = build_embedder(tower, readout=args.readout, head_hidden=128, head_dim=64)
    taps = model.available_taps()
    base = evaluate(model, data, processor, "base", kinds=["pooled"] + taps, splits=("test", "val"), num_workers=0,
                    bf16=False)
    print(json.dumps(summary_row(base)), flush=True)
    hist = train(model, data, processor, steps=args.steps, P=args.P, K=args.K, chunk_patches=args.chunk_patches,
                 eval_every=args.eval_every, num_workers=0, bf16=False, stack_prob=0.5, library_block=2, warmup=1,
                 log_every=1, max_minutes=args.max_minutes, pooled_weight=args.pooled_weight)
    print("taps after LoRA:", model.available_taps(), flush=True)
    train_rows = [r for r in data["lines"] if r["split"] == "train"]
    items = LineItems(train_rows, processor, augment=True)
    for seed in range(50):  # a check batch that contains line stacks as well as single lines
        specs = next(iter(PKSampler(train_rows, args.P, args.K, 1, seed=seed, stack_prob=0.5, library_block=2)))
        kinds = Counter(s[0] for s in specs)
        if kinds["stack"] and kinds["line"]:
            break
    batch = collate([items[s] for s in specs])
    for kind in ("line", "stack"):  # one augmented example of each, for eyeballing
        items.image(next(s for s in specs if s[0] == kind))[0].save(Path(args.out) / f"sample_{kind}.jpg")
    diff = gradcache_check(model, batch, 0.07, 0.5, args.chunk_patches)
    diff0 = gradcache_check(model, batch, 0.07, 0.0, args.chunk_patches)  # head-only loss (pooled_weight=0)
    print(f"gradcache vs full-batch max rel grad diff: {diff:.2e}, head-only {diff0:.2e} (items {dict(kinds)}, "
          f"{len(patch_chunks(batch['grid_thw'], args.chunk_patches))} chunks)", flush=True)
    tuned = evaluate(model, data, processor, "tuned", kinds=["pooled", "head"] + taps, splits=("test",), num_workers=0,
                     bf16=False)
    print(json.dumps(summary_row(tuned)), flush=True)
    rows = [r for r in data["lines"] if r["split"] == "test"][:6]
    before = embed(model, LineItems(rows, processor), 6, 0, False)
    save_tuned(model, processor, args.out, [base, tuned], hist, vars(args))
    after_inplace = embed(model, LineItems(rows, processor), 6, 0, False)
    reloaded, _ = load_tuned(args.out, device="cpu")
    after_reload = embed(reloaded, LineItems(rows, processor), 6, 0, False)
    cos = lambda a, b: float((a * b).sum(1).min())  # noqa: E731
    print(json.dumps({"readout": args.readout, "taps_after_save": model.available_taps(),
                      "merge_cos_pooled_min": round(cos(before["pooled"], after_inplace["pooled"]), 6),
                      "reload_cos_pooled_min": round(cos(before["pooled"], after_reload["pooled"]), 6),
                      "reload_cos_head_min": round(cos(before["head"], after_reload["head"]), 6),
                      "gradcache_rel_diff": diff, "gradcache_rel_diff_head_only": diff0,
                      "train_steps_run": max([h["step"] for h in hist if "loss" in h], default=-1) + 1,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
