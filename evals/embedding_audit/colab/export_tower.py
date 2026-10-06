"""Export a fine-tuned Hebrew VLM's vision tower as a standalone HF ``Qwen3VLVisionModel`` folder (runs locally).

The sibling project's merged Qwen3-VL-8B checkpoints are exported for LM Studio as MLX folders, where the vision
tower is stored unquantised in bf16 under ``vision_tower.*``. The only layout difference from transformers is the
patch-embedding Conv3d weight, which MLX keeps channels-last ``(O, T, H, W, I)`` and torch wants ``(O, I, T, H, W)``
(same remap as :func:`image_features.Encoder._load_qwen_vit`). This script copies those tensors into
``<AUDIT_ROOT>/hf_image/tower_<ckpt>/``:

* ``model-0000N-of-0000M.safetensors`` + ``model.safetensors.index.json`` — the 351 tower tensors (bf16), written in
  ~400 MB shards so the export never holds the whole 1.2 GB tower in memory;
* ``config.json`` — the ``Qwen3VLVisionConfig`` (architecture identical to stock Qwen3-VL-8B-Instruct; checked);
* ``preprocessor_config.json`` — the STOCK Qwen3-VL image processor (the one the audit's encoders use; the sibling's
  export carries its high-resolution VLM inference pixel budget instead, which the embedder never wants);
* ``export_info.json`` — source dir, parameter count, bytes, and the load-back verification.

Load it with ``Qwen3VLVisionModel.from_pretrained(dir, dtype=torch.bfloat16)`` + ``AutoImageProcessor.from_pretrained(dir)``.

``--verify`` loads the exported folder back (bf16, CPU) and compares the mean-pooled merger vector of one masked
fragment image with the vector :mod:`image_features`' own loader cached for it
(``cache/image_feats/qwen3vl-vit512-<ckpt>__masked``, computed on CPU in fp32); ``--fp32`` repeats the forward in fp32.
``--tiny <dir>`` writes a random-init tiny tower (depth 2, hidden 64) for CPU smoke tests of the trainer.

Run under the guard (``guard.py --small``): export peaks at ~1 GB, verify at ~1.5 GB (bf16) / ~2.7 GB (fp32).
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

AUDIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AUDIT))
from embed_utils import AUDIT_ROOT  # noqa: E402
from image_features import QWEN_VL_SNAPSHOT  # noqa: E402

DEFAULT_CKPT_DIR = Path("/Volumes/home/studio_offload/v19b_merge/qwen3-vl-8b-heb-v22b-step1200-mlx")
VISION_PREFIX = "vision_tower."
SHARD_BYTES = 400 * 1024 ** 2
ARCH_KEYS = ("depth", "hidden_size", "hidden_act", "intermediate_size", "num_heads", "in_channels", "patch_size",
             "spatial_merge_size", "temporal_patch_size", "out_hidden_size", "num_position_embeddings",
             "deepstack_visual_indexes")


def ckpt_name(ckpt_dir: Path) -> str:
    """Short checkpoint name used in output and cache paths.

    :param ckpt_dir: e.g. ``.../qwen3-vl-8b-heb-v22b-step1200-mlx`` or ``~/.lmstudio/models/isaacmg/qwen3-vl-8b-heb-v21b-step1200``.
    :returns: e.g. ``heb-v22b-step1200``.
    :rtype: str
    """
    name = ckpt_dir.name
    name = name[len("qwen3-vl-8b-"):] if name.startswith("qwen3-vl-8b-") else name
    return name[: -len("-mlx")] if name.endswith("-mlx") else name


def stock_vision_config():
    """The stock Qwen3-VL-8B-Instruct vision config (what :mod:`image_features` builds the tower from).

    :returns: ``Qwen3VLVisionConfig``.
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig.from_pretrained(QWEN_VL_SNAPSHOT).vision_config


def check_architecture(ckpt_dir: Path, cfg) -> None:
    """Assert the checkpoint's own vision config matches the stock architecture.

    :param ckpt_dir: MLX export directory.
    :param cfg: Stock ``Qwen3VLVisionConfig``.
    """
    theirs = json.loads((ckpt_dir / "config.json").read_text())["vision_config"]
    diff = {k: (theirs.get(k), getattr(cfg, k)) for k in ARCH_KEYS if theirs.get(k) != getattr(cfg, k)}
    assert not diff, f"checkpoint vision config differs from stock: {diff}"


def expected_shapes(cfg) -> Dict[str, tuple]:
    """Parameter names and shapes of a ``Qwen3VLVisionModel`` built from ``cfg`` (meta device, no memory).

    :param cfg: ``Qwen3VLVisionConfig``.
    :returns: {state-dict key: shape}; non-persistent buffers (rotary ``inv_freq``) are not included.
    :rtype: Dict[str, tuple]
    """
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    with torch.device("meta"):
        model = Qwen3VLVisionModel._from_config(cfg)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def mlx_tower_index(ckpt_dir: Path) -> Dict[str, str]:
    """Map every ``vision_tower.*`` tensor of an MLX export to its shard file.

    :param ckpt_dir: MLX export directory.
    :returns: {mlx key: shard filename}.
    :rtype: Dict[str, str]
    """
    wmap = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())["weight_map"]
    return {k: f for k, f in wmap.items() if k.startswith(VISION_PREFIX)}


def to_torch_layout(key: str, tensor):
    """Apply the MLX -> torch layout remap (patch-embedding conv only) and cast to bf16.

    :param key: Torch-side key (prefix stripped).
    :param tensor: Tensor as stored in the MLX shard.
    :returns: Contiguous bf16 tensor in torch layout.
    """
    import torch

    t = tensor.to(torch.bfloat16)
    if key == "patch_embed.proj.weight" and t.dim() == 5 and t.shape[-1] == 3:
        t = t.permute(0, 4, 1, 2, 3)  # MLX (O,T,H,W,I) -> torch (O,I,T,H,W)
    return t.contiguous()


def plan_shards(ckpt_dir: Path, index: Dict[str, str], shard_bytes: int) -> List[List[str]]:
    """Group tower keys into output shards of at most ``shard_bytes`` (bf16 sizes), in name order.

    :param ckpt_dir: MLX export directory.
    :param index: Output of :func:`mlx_tower_index`.
    :param shard_bytes: Shard size cap.
    :returns: List of key lists (MLX keys).
    :rtype: List[List[str]]
    """
    from safetensors import safe_open

    sizes = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(ckpt_dir / shard), framework="pt") as fh:
            for key in fh.keys():
                if key in index:
                    n = 1
                    for d in fh.get_slice(key).get_shape():
                        n *= d
                    sizes[key] = 2 * n
    shards, current, used = [], [], 0
    for key in sorted(index):
        if current and used + sizes[key] > shard_bytes:
            shards.append(current)
            current, used = [], 0
        current.append(key)
        used += sizes[key]
    if current:
        shards.append(current)
    return shards


def export_tower(ckpt_dir: Path, out_dir: Path, shard_bytes: int = SHARD_BYTES) -> dict:
    """Write the HF tower folder (sharded safetensors + config + stock processor).

    :param ckpt_dir: MLX export directory of the fine-tuned VLM.
    :param out_dir: Output folder.
    :param shard_bytes: Output shard size cap.
    :returns: Export info dict (also written to ``export_info.json``).
    :rtype: dict
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoImageProcessor

    cfg = stock_vision_config()
    check_architecture(ckpt_dir, cfg)
    want = expected_shapes(cfg)
    index = mlx_tower_index(ckpt_dir)
    have = {k[len(VISION_PREFIX):] for k in index}
    assert have == set(want), (sorted(set(want) - have)[:5], sorted(have - set(want))[:5])
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_shards(ckpt_dir, index, shard_bytes)
    weight_map, total, n_params = {}, 0, 0
    for i, keys in enumerate(plan):
        name = f"model-{i + 1:05d}-of-{len(plan):05d}.safetensors"
        tensors = {}
        for src in sorted({index[k] for k in keys}):
            with safe_open(str(ckpt_dir / src), framework="pt") as fh:
                for key in keys:
                    if index[key] == src:
                        dst = key[len(VISION_PREFIX):]
                        tensors[dst] = to_torch_layout(dst, fh.get_tensor(key))
                        assert tuple(tensors[dst].shape) == want[dst], (dst, tensors[dst].shape, want[dst])
        for dst, t in tensors.items():
            weight_map[dst] = name
            total += t.numel() * t.element_size()
            n_params += t.numel()
        save_file(tensors, str(out_dir / name), metadata={"format": "pt"})
        del tensors
        print(f"wrote {name} ({len(keys)} tensors)", flush=True)
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=1))
    cfg.architectures = ["Qwen3VLVisionModel"]
    cfg.dtype = torch.bfloat16
    cfg.save_pretrained(out_dir)
    AutoImageProcessor.from_pretrained(QWEN_VL_SNAPSHOT).save_pretrained(out_dir)
    digest = hashlib.sha256()
    for name in sorted(set(weight_map.values())):
        with open(out_dir / name, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                digest.update(block)
    info = {"source": str(ckpt_dir), "ckpt": ckpt_name(ckpt_dir), "n_tensors": len(weight_map), "n_params": n_params,
            "bytes": total, "dtype": "bfloat16", "shards": len(plan), "sha256_shards": digest.hexdigest(),
            "processor": f"stock {QWEN_VL_SNAPSHOT.parent.parent.name}@{QWEN_VL_SNAPSHOT.name}",
            "remap": "vision_tower.<k> -> <k>; patch_embed.proj.weight permute(0,4,1,2,3) (MLX channels-last -> torch)",
            "exported_at": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "export_info.json").write_text(json.dumps(info, indent=1))
    write_readme(out_dir, info)
    return info


def write_readme(out_dir: Path, info: dict) -> None:
    """Model card for the private tower repo.

    :param out_dir: Exported tower folder.
    :param info: Export info (``export_info.json``).
    """
    (out_dir / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-VL-8B-Instruct\n---\n"
        f"# Qwen3-VL-8B vision tower — Hebrew VLM fine-tune `{info['ckpt']}`\n\n"
        "The vision tower of our Hebrew manuscript VLM (sibling project `historical-document-analysis`), extracted from "
        "its LM Studio MLX export (tower stored unquantised in bf16; only the patch-embedding conv is re-laid out "
        "channels-first) as a standalone `Qwen3VLVisionModel` for the Genizah image-encoder fine-tune.\n\n"
        "```python\nfrom transformers import AutoImageProcessor\n"
        "from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel\n"
        "tower = Qwen3VLVisionModel.from_pretrained(repo, dtype=torch.bfloat16)\n"
        "processor = AutoImageProcessor.from_pretrained(repo)  # stock Qwen3-VL processor\n"
        "hidden, deepstack = tower(pixel_values, grid_thw=grid_thw)  # mean of `hidden` = the audit's image vector\n```\n\n"
        f"{info['n_params']:,} parameters, {info['bytes'] / 1e9:.2f} GB bf16. Source: `{Path(info['source']).name}`. "
        "Verified against the audit's own loader (`image_features.py`) on a masked fragment: see `export_info.json`.\n")


def make_tiny_tower(out_dir: Path, seed: int = 0) -> dict:
    """Write a random-init tiny ``Qwen3VLVisionModel`` (+ stock processor) for CPU smoke tests.

    Same patch/merge geometry as the real tower (patch 16, merge 2, temporal 2), so every data path is exercised.

    :param out_dir: Output folder.
    :param seed: Init seed.
    :returns: Small info dict.
    :rtype: dict
    """
    import torch
    from transformers import AutoImageProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(seed)
    cfg = Qwen3VLVisionConfig(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=96,
                              deepstack_visual_indexes=[0, 1], num_position_embeddings=2304)
    cfg.architectures = ["Qwen3VLVisionModel"]
    model = Qwen3VLVisionModel._from_config(cfg)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoImageProcessor.from_pretrained(QWEN_VL_SNAPSHOT).save_pretrained(out_dir)
    return {"dir": str(out_dir), "n_params": sum(p.numel() for p in model.parameters())}


def cached_reference(ckpt: str, cid: Optional[str] = None) -> tuple:
    """One fragment id and its cached masked-view vector from :mod:`image_features` (CPU fp32 run).

    :param ckpt: Checkpoint name (``heb-v22b-step1200``).
    :param cid: Fragment id; default = first id of the first cache part.
    :returns: (fragment id, cached vector).
    :rtype: tuple
    """
    import numpy as np

    parts = sorted((AUDIT_ROOT / "cache" / "image_feats" / f"qwen3vl-vit512-{ckpt}__masked").glob("part_*.npz"))
    assert parts, f"no cached qwen3vl-vit512-{ckpt}__masked features to compare against"
    for p in parts:
        z = np.load(p)
        ids = list(z["ids"])
        if cid is None or cid in ids:
            k = 0 if cid is None else ids.index(cid)
            return ids[k], z["feats"][k]
    raise KeyError(cid)


def pooled_vector(model, processor, img, dtype) -> "np.ndarray":
    """Mean-pooled merger vector exactly as :meth:`image_features.Encoder.embed` computes it (512x512 budget).

    :param model: ``Qwen3VLVisionModel``.
    :param processor: Qwen3-VL image processor.
    :param img: PIL image (the masked view).
    :param dtype: Compute dtype.
    :returns: L2-normalised vector.
    :rtype: np.ndarray
    """
    import numpy as np
    import torch

    inp = processor(images=[img], return_tensors="pt", min_pixels=128 * 128, max_pixels=512 * 512)
    with torch.no_grad():
        out = model(inp["pixel_values"].to(dtype), grid_thw=inp["image_grid_thw"])
    v = out[0].float().mean(0).numpy()
    return v / np.linalg.norm(v)


def verify(out_dir: Path, ckpt: str, fp32: bool, cid: Optional[str] = None) -> dict:
    """Load the exported tower back (bf16, CPU) and compare one image's vector with image_features' cached one.

    :param out_dir: Exported tower folder.
    :param ckpt: Checkpoint name (selects the feature cache).
    :param fp32: Also run the forward in fp32 (same compute as the cached CPU run; ~2.7 GB footprint).
    :param cid: Fragment id to test (default: first cached id).
    :returns: Verification dict (also merged into ``export_info.json``).
    :rtype: dict
    """
    import numpy as np
    import torch
    from transformers import AutoImageProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    from image_features import load_view

    torch.set_grad_enabled(False)
    model, info = Qwen3VLVisionModel.from_pretrained(out_dir, dtype=torch.bfloat16, output_loading_info=True)
    model.eval()
    problems = {k: v for k, v in info.items() if v}
    processor = AutoImageProcessor.from_pretrained(out_dir)
    cid, ref = cached_reference(ckpt, cid)
    img = load_view(cid, "masked", 16, 320)[0]
    res = {"fragment": cid, "loading_problems": problems, "n_params": sum(p.numel() for p in model.parameters()),
           "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
    v = pooled_vector(model, processor, img, torch.bfloat16)
    res["cos_bf16_vs_cache"] = round(float(np.dot(v.astype(np.float64), ref.astype(np.float64))), 6)
    if fp32:
        model = model.float()
        v = pooled_vector(model, processor, img, torch.float32)
        res["cos_fp32_vs_cache"] = round(float(np.dot(v.astype(np.float64), ref.astype(np.float64))), 6)
    res["pass"] = not problems and min(c for k, c in res.items() if k.startswith("cos_")) > 0.999
    res["verified_at"] = datetime.now().isoformat(timespec="seconds")
    path = out_dir / "export_info.json"
    if path.exists():
        exp = json.loads(path.read_text())
        exp.setdefault("verify", []).append(res)
        path.write_text(json.dumps(exp, indent=1))
    return res


def main() -> None:
    """Export, verify, or write a tiny smoke-test tower."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR, help="MLX export of the fine-tuned VLM")
    parser.add_argument("--out", type=Path, default=None, help="default <AUDIT_ROOT>/hf_image/tower_<ckpt>")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--fp32", action="store_true", help="verify: also compare an fp32 forward")
    parser.add_argument("--fragment", default=None, help="verify: fragment id (default first cached)")
    parser.add_argument("--tiny", type=Path, default=None, help="write a tiny random tower here and exit")
    args = parser.parse_args()
    if args.tiny:
        print(json.dumps(make_tiny_tower(args.tiny)))
        return
    ckpt = ckpt_name(args.ckpt_dir)
    out = args.out or AUDIT_ROOT / "hf_image" / f"tower_{ckpt}"
    if args.export:
        print(json.dumps(export_tower(args.ckpt_dir, out), indent=1), flush=True)
    if args.verify:
        print(json.dumps(verify(out, ckpt, args.fp32, args.fragment), indent=1), flush=True)


if __name__ == "__main__":
    main()
