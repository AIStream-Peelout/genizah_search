"""Embed every imaged Genizah fragment for join / scribe discovery (Colab A100; images streamed from GCS).

Inputs come from the private dataset repo ``isaacmg/genizah-image-gallery`` (tag ``inputs-v1``, uploaded from
``<AUDIT_ROOT>/hf_gallery`` by ``package_gallery.py``): the gallery manifest (55,406 records, embedded image URL each;
rows whose images were all HEAD-checked dead, ``image_live`` false, are skipped),
the benchmark ids/URLs/labels, the published tuned-tower I1 numbers, and 200 locally computed parity vectors.

Per fragment (exactly the local benchmark pipeline, see ``gallery_preprocess.py``): download the full image (thread
pool, retries with backoff), ``thumbnail`` to max side 1600 + JPEG q90 (``images/max1600``), ``fragment_mask``, then

* tuned v22b tower (``isaacmg/genizah-image-tower-v22b-lora`` @ ``TOWER_REVISION``, ``train_image_embedder.load_tuned``
  + ``embed``): the masked square resized to the 512x512-pixel budget and stored/read as JPEG q92 (the benchmark's
  fragment shards), mean of merged tokens for ``merger`` and the ``ds8`` deepstack tap, the configured ``pooled``
  readout (``ds8+merger``: both parts L2-normalised, concatenated / sqrt 2) and the ``head`` vector;
* DINOv2-base ``masked`` (CLS ‖ mean patch token at 448 on the grey-padded masked square) and ``mpatches`` (mean of
  the L2-normalised vectors of up to 16 ink patches of 320 px inside the mask, each embedded at 224).

Shards of ``SHARD_SIZE`` fragments (manifest order) are written as fp16 ``.npz`` and uploaded to ``shards/`` of the
same repo together with ``failures/part_XXXX.jsonl``; shards already in the repo are skipped, so a disconnected run
is resumed by re-running the cell. ``pooled`` is not stored: it is rebuilt exactly from ``ds8`` and ``merger``
(:func:`pooled_from_parts`; the parity cell checks the rebuild against the embedder's own pooled output).

Gates before the full run: :func:`run_parity` (200 benchmark fragments: same max1600 size, crop, mask path and patch
count as the local pipeline; DINOv2 ``masked``/``mpatches`` cosine > 0.99 to the local cache) and
:func:`run_benchmark` (the 5,176 benchmark fragments streamed through this pipeline must reproduce the published
tuned ``merger`` joins R@10 = 0.592 within 0.01).
"""

import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gallery_preprocess as gp  # noqa: E402

GALLERY_REPO = "isaacmg/genizah-image-gallery"
INPUT_REVISION = "inputs-v1"
TOWER_REPO = "isaacmg/genizah-image-tower-v22b-lora"
TOWER_REVISION = "c45c8de439585c21e03f434c3b7f9c91a5a623dd"
DINO_ID = "facebook/dinov2-base"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"  # the snapshot the local caches were computed with
SHARD_SIZE = 2000
CHUNK = 16
TOWER_KINDS = ("pooled", "head", "merger", "ds8")
STORED = ("merger", "ds8", "head", "dino_masked", "dino_mpatches")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
PARITY_COS = 0.99
GEOMETRY_MIN_SHARE = 0.97  # crops / mask paths / patch counts identical to the local pipeline (JPEG/libjpeg builds may
                           # move a mask edge by a pixel on a few images; max1600 sizes must match exactly)
BENCH_TOLERANCE = 0.01
MAX_SHARD_FAILURE_SHARE = 0.2  # more failures than this in one shard = an outage: stop instead of uploading a hollow shard
PILLOW_VERSION = "12.0.0"  # the Studio's Pillow (resampling + bundled libjpeg-turbo of the local features)


# ------------------------------------------------------------------------------------------------------ inputs
def load_inputs(repo_id: str = GALLERY_REPO, token: Optional[str] = None, revision: str = INPUT_REVISION,
                local_dir: str = "/content/gallery_inputs") -> dict:
    """Download (or read) the packaged inputs.

    :param repo_id: Dataset repo id, or a local ``hf_gallery`` folder.
    :param token: HF token.
    :param revision: Tag/commit of the inputs.
    :param local_dir: Download folder.
    :returns: {rows, bench_ids, bench_urls, labels, reference, parity, path}.
    :rtype: dict
    """
    if Path(repo_id).is_dir():
        path = Path(repo_id)
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id, repo_type="dataset", revision=revision, token=token, local_dir=local_dir,
                                      allow_patterns=["inputs/*", "code/*", "README.md"]))
    inp = path / "inputs"
    rows = [json.loads(l) for l in open(inp / "manifest.jsonl", encoding="utf-8")]
    bench = json.loads((inp / "benchmark.json").read_text())
    return {"rows": rows, "bench_ids": bench["ids"], "bench_urls": bench["urls"],
            "labels": json.loads((inp / "fragment_labels.json").read_text()),
            "reference": json.loads((inp / "benchmark_reference.json").read_text()),
            "parity": {k: v for k, v in np.load(inp / "parity_local.npz").items()}, "path": path}


def gallery_rows(rows: Sequence[dict]) -> List[dict]:
    """Manifest rows to embed: every row except those whose images were all checked dead (``image_live`` false).

    :param rows: Manifest rows.
    :returns: Rows in manifest order.
    :rtype: List[dict]
    """
    return [r for r in rows if r.get("image_live") is not False]


def shard_plan(rows: Sequence[dict], shard_size: int = SHARD_SIZE) -> List[List[Tuple[str, str]]]:
    """Fixed shards of the manifest order (dead-image rows left out): shard k = rows [k * size, (k + 1) * size).

    :param rows: Manifest rows.
    :param shard_size: Fragments per shard.
    :returns: List of [(canonical_id, image_url), ...] per shard.
    :rtype: List[List[Tuple[str, str]]]
    """
    items = [(r["canonical_id"], r["image_url"]) for r in gallery_rows(rows)]
    return [items[k:k + shard_size] for k in range(0, len(items), shard_size)]


def is_permanent(error: str) -> bool:
    """Whether a logged failure is a permanent HTTP error (404 / 403 ...: retrying cannot help).

    :func:`_download` logs ``"FetchError: http 404"`` for a status :func:`fetch` does not retry, and
    ``"FetchError: gave up after N attempts: ..."`` once the retries for a transient one ran out.

    :param error: Failure row's ``error``.
    :returns: True for a non-retried 4xx status.
    :rtype: bool
    """
    m = re.fullmatch(r"FetchError: http (\d{3})", error.strip())
    return bool(m) and 400 <= int(m.group(1)) < 500 and int(m.group(1)) not in RETRY_STATUS


# ------------------------------------------------------------------------------------------------------ download
class FetchError(RuntimeError):
    """An image could not be downloaded."""


_LOCAL = threading.local()
_PROC: Dict[str, object] = {}


def _session() -> "requests.Session":
    """One HTTP session per thread (requests sessions are not thread-safe).

    :returns: requests.Session.
    """
    import requests

    if getattr(_LOCAL, "session", None) is None or getattr(_LOCAL, "pid", None) != os.getpid():
        _LOCAL.session, _LOCAL.pid = requests.Session(), os.getpid()
    return _LOCAL.session


def fetch(url: str, retries: int = 6, timeout: Tuple[float, float] = (10.0, 120.0)) -> bytes:
    """GET one image with exponential backoff on transient errors (connection, timeout, 408/429/5xx).

    :param url: Image URL.
    :param retries: Attempts.
    :param timeout: (connect, read) seconds.
    :returns: Response body.
    :rtype: bytes
    """
    import requests

    delay, err = 1.0, ""
    for _ in range(retries):
        try:
            resp = _session().get(url, timeout=timeout)
        except requests.RequestException as exc:  # transient network failure: retried
            err = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200 and resp.content:
                return resp.content
            err = f"http {resp.status_code}"
            if resp.status_code not in RETRY_STATUS:
                raise FetchError(err)
        time.sleep(delay * (1 + random.random()))
        delay = min(delay * 2, 30.0)
    raise FetchError(f"gave up after {retries} attempts: {err}")


def _pool(threads: int) -> ThreadPoolExecutor:
    """Download thread pool of this (worker) process, created after the DataLoader fork.

    :param threads: Pool size.
    :returns: Executor.
    :rtype: ThreadPoolExecutor
    """
    if _PROC.get("pid") != os.getpid():
        _PROC.update(pid=os.getpid(), pool=ThreadPoolExecutor(threads))
    return _PROC["pool"]


def _download(job: Tuple[str, str]) -> Tuple[Optional[bytes], str]:
    """Thread job: (bytes, "") or (None, error).

    :param job: (cid, url).
    :returns: (content, error).
    :rtype: Tuple[Optional[bytes], str]
    """
    try:
        return fetch(job[1]), ""
    except Exception as exc:  # every download failure becomes a logged per-fragment failure, never a crash
        return None, f"{type(exc).__name__}: {exc}"


class GalleryChunks:
    """Map-style dataset: item k = one chunk of up to ``CHUNK`` fragments, downloaded concurrently and preprocessed.

    Runs in DataLoader worker processes; returns tensors already packed for one tower call (pixel values concatenated
    over the chunk, like ``train_image_embedder.collate``) plus uint8 DINOv2 inputs, provenance and failures.
    """

    def __init__(self, chunks: List[Tuple[int, List[Tuple[str, str]]]], processor: object, threads: int = 8) -> None:
        """Store the chunk specs.

        :param chunks: [(shard id, [(cid, url), ...]), ...].
        :param processor: Qwen3-VL image processor of the tuned tower.
        :param threads: Concurrent downloads per worker.
        """
        self.chunks, self.processor, self.threads = chunks, processor, threads

    def __len__(self) -> int:
        """Number of chunks.

        :returns: Count.
        :rtype: int
        """
        return len(self.chunks)

    def __getitem__(self, k: int) -> dict:
        """Download + preprocess one chunk.

        :param k: Chunk index.
        :returns: Packed chunk (see class doc).
        :rtype: dict
        """
        import torch
        from PIL import Image

        shard, items = self.chunks[k]
        t0 = time.time()
        got = list(_pool(self.threads).map(_download, items))
        t1 = time.time()
        ok, failed = [], []
        for (cid, url), (content, err) in zip(items, got):
            if content is None:
                failed.append({"id": cid, "url": url, "stage": "download", "error": err[:300], "shard": shard})
                continue
            try:
                v = gp.preprocess_fragment(content, cid)
                inp = self.processor(images=[Image.fromarray(v["qwen"])], return_tensors="pt",
                                     min_pixels=gp.FRAG_MIN_PIXELS, max_pixels=gp.FRAG_MAX_PIXELS)
            except Exception as exc:  # unreadable/corrupt image: logged per fragment, the run continues
                failed.append({"id": cid, "url": url, "stage": "preprocess", "error": f"{type(exc).__name__}: {exc}"[:300],
                               "shard": shard})
                continue
            v.update(cid=cid, url=url, pixel_values=inp["pixel_values"], grid_thw=inp["image_grid_thw"])
            ok.append(v)
        out = {"shard": shard, "chunk": k, "ids": [v["cid"] for v in ok], "urls": [v["url"] for v in ok],
               "failed": failed, "t_download": t1 - t0, "t_preprocess": time.time() - t1, "n": len(ok)}
        if ok:
            out.update(
                pixel_values=torch.cat([v["pixel_values"] for v in ok]), grid_thw=torch.cat([v["grid_thw"] for v in ok]),
                dino_masked=torch.from_numpy(np.stack([v["dino_masked"] for v in ok])),
                dino_patches=torch.from_numpy(np.concatenate([v["dino_patches"] for v in ok])),
                patch_counts=[int(v["n_patches"]) for v in ok],
                meta=[{k2: v[k2] for k2 in ("sha1", "max1600_wh", "crop_wh", "mask_ok", "mask_frac", "n_patches", "dhash")}
                      for v in ok])
        return out


def _identity(x: dict) -> dict:
    """DataLoader collate for ``batch_size=None`` items that are already batches.

    :param x: Item.
    :returns: The item unchanged.
    """
    return x


def iterate_chunks(chunks: List[Tuple[int, List[Tuple[str, str]]]], processor: object, workers: int, threads: int,
                   prefetch: int = 2) -> Iterator[dict]:
    """Yield preprocessed chunks in order from a multi-process DataLoader.

    :param chunks: Chunk specs.
    :param processor: Qwen image processor.
    :param workers: DataLoader worker processes.
    :param threads: Download threads per worker.
    :param prefetch: Chunks prefetched per worker.
    :returns: Iterator of packed chunks.
    """
    from torch.utils.data import DataLoader

    ds = GalleryChunks(chunks, processor, threads)
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, collate_fn=_identity,
                        prefetch_factor=prefetch if workers else None, persistent_workers=False)
    yield from loader


def make_chunks(shards: Dict[int, List[Tuple[str, str]]], chunk: int = CHUNK) -> List[Tuple[int, List[Tuple[str, str]]]]:
    """Split each shard into chunks (chunks never straddle shards).

    :param shards: shard id -> items.
    :param chunk: Fragments per chunk.
    :returns: [(shard id, items), ...] in shard order.
    :rtype: List[Tuple[int, List[Tuple[str, str]]]]
    """
    return [(s, shards[s][k:k + chunk]) for s in sorted(shards) for k in range(0, len(shards[s]), chunk)]


# ------------------------------------------------------------------------------------------------------ models
class GalleryEmbedder:
    """Tuned tower (via ``train_image_embedder``) + DINOv2-base, both on the GPU."""

    def __init__(self, tower_dir: str, device: str = "cuda", bf16: bool = True, dino_tf32: bool = True) -> None:
        """Load both encoders.

        :param tower_dir: Folder with ``tower_merged/``, ``head.pt``, ``head_config.json`` (the tuned model repo).
        :param device: Torch device.
        :param bf16: bf16 autocast for the tower (``train_image_embedder.embed`` default, as in the published eval).
        :param dino_tf32: Allow TF32 matmuls/convolutions (DINOv2 runs in fp32 weights; the local cache was CPU fp32).
        """
        import torch
        from transformers import AutoModel

        import train_image_embedder as tie

        self.tie, self.device, self.bf16 = tie, torch.device(device), bf16
        torch.backends.cuda.matmul.allow_tf32 = dino_tf32
        torch.backends.cudnn.allow_tf32 = dino_tf32
        self.model, self.processor = tie.load_tuned(tower_dir, device=device)
        assert self.model.readout == "ds8+merger", self.model.readout
        self.dino = AutoModel.from_pretrained(DINO_ID, revision=DINO_REVISION).to(self.device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        torch.set_grad_enabled(False)

    def tower(self, pixel_values: "torch.Tensor", grid_thw: "torch.Tensor", bf16: Optional[bool] = None) -> Dict[str, np.ndarray]:
        """Tower vectors for one packed chunk through ``train_image_embedder.embed`` (one batch, no workers).

        :param pixel_values: Packed patches (N, 1536).
        :param grid_thw: Grids (B, 3).
        :param bf16: Override the autocast setting (precision diagnostics).
        :returns: {pooled, head, merger, ds8} float32, L2-normalised.
        :rtype: Dict[str, np.ndarray]
        """
        counts = grid_thw.prod(-1).tolist()
        parts = pixel_values.split(counts)
        items = [{"pixel_values": p, "grid_thw": grid_thw[i:i + 1], "ms": -1, "page": -1} for i, p in enumerate(parts)]
        return self.tie.embed(self.model, items, batch_size=len(items), num_workers=0,
                              bf16=self.bf16 if bf16 is None else bf16, kinds=TOWER_KINDS)

    def _dino(self, u8: "torch.Tensor", batch: int) -> np.ndarray:
        """DINOv2 CLS ‖ mean-patch vectors, L2-normalised (``image_features.Encoder.embed`` on the GPU).

        :param u8: (B, H, W, 3) uint8 tensor (already resized: 448 or 224).
        :param batch: Images per forward.
        :returns: (B, 1536) float32.
        :rtype: np.ndarray
        """
        import torch
        import torch.nn.functional as F

        out = []
        for s in range(0, len(u8), batch):
            x = u8[s:s + batch].to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            h = self.dino(pixel_values=x).last_hidden_state
            out.append(F.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float(), dim=1).cpu().numpy())
        return np.concatenate(out)

    def embed_chunk(self, c: dict) -> Dict[str, np.ndarray]:
        """All stored vectors for one preprocessed chunk.

        :param c: Output of :class:`GalleryChunks`.
        :returns: {merger, ds8, pooled, head, dino_masked, dino_mpatches} float32, rows = ``c["ids"]``.
        :rtype: Dict[str, np.ndarray]
        """
        out = self.tower(c["pixel_values"], c["grid_thw"])
        out["dino_masked"] = self._dino(c["dino_masked"], 32)
        per_patch = self._dino(c["dino_patches"], 256)
        bounds = np.cumsum([0] + c["patch_counts"])
        mp = np.stack([per_patch[bounds[i]:bounds[i + 1]].mean(0) for i in range(len(c["patch_counts"]))])
        out["dino_mpatches"] = mp / np.linalg.norm(mp, axis=1, keepdims=True)
        return out


def pooled_from_parts(ds8: np.ndarray, merger: np.ndarray) -> np.ndarray:
    """Rebuild the tuned ``pooled`` readout (``ds8+merger``) from its stored parts.

    :param ds8: (n, 4096) ds8 tap vectors (L2-normalised; fp16 storage is fine).
    :param merger: (n, 4096) merger vectors.
    :returns: (n, 8192) float32 unit vectors: [ds8, merger] / sqrt 2, renormalised.
    :rtype: np.ndarray
    """
    parts = [p.astype(np.float32) / np.linalg.norm(p.astype(np.float32), axis=1, keepdims=True) for p in (ds8, merger)]
    X = np.concatenate(parts, axis=1) / math.sqrt(2)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------------ running
class Progress:
    """Throughput / ETA printer."""

    def __init__(self, total: int, label: str, every: float = 60.0) -> None:
        """Start the clock.

        :param total: Fragments to process.
        :param label: Prefix.
        :param every: Seconds between prints.
        """
        self.total, self.label, self.every = total, label, every
        self.t0 = self.last = time.time()
        self.done = self.failed = 0
        self.recent: List[Tuple[float, int]] = [(self.t0, 0)]
        self.t_wait = self.t_gpu = 0.0

    def update(self, n_ok: int, n_failed: int, t_wait: float, t_gpu: float, force: bool = False) -> None:
        """Account one chunk and print when due.

        :param n_ok: Fragments embedded.
        :param n_failed: Fragments failed.
        :param t_wait: Seconds waiting for the DataLoader.
        :param t_gpu: Seconds in the encoders.
        :param force: Print now.
        """
        now = time.time()
        self.done += n_ok + n_failed
        self.failed += n_failed
        self.t_wait += t_wait
        self.t_gpu += t_gpu
        self.recent = [(t, d) for t, d in self.recent if now - t < 300] + [(now, self.done)]
        if force or now - self.last >= self.every:
            self.last = now
            print(self.line(), flush=True)

    def rate(self) -> float:
        """Fragments per second over the last ~5 minutes (overall before that).

        :returns: Rate.
        :rtype: float
        """
        (t_a, d_a), (t_b, d_b) = self.recent[0], self.recent[-1]
        return (d_b - d_a) / max(1e-9, t_b - t_a) if t_b > t_a else 0.0

    def line(self) -> str:
        """Status line with ETA.

        :returns: Text.
        :rtype: str
        """
        el = time.time() - self.t0
        r = self.rate()
        left = (self.total - self.done) / r if r > 0 else float("nan")
        busy = self.t_wait + self.t_gpu
        return (f"[{self.label}] {self.done:,}/{self.total:,} fragments ({self.failed} failed) | "
                f"{self.done / max(el, 1e-9):.1f}/s overall, {r:.1f}/s recent | elapsed {el / 60:.1f} min, "
                f"ETA {left / 60:.1f} min | GPU {self.t_gpu / max(busy, 1e-9):.0%} / waiting for data "
                f"{self.t_wait / max(busy, 1e-9):.0%}")


def embed_items(emb: GalleryEmbedder, items: List[Tuple[str, str]], workers: int, threads: int, label: str,
                every: float = 30.0) -> Tuple[List[str], Dict[str, np.ndarray], List[dict], List[dict], Progress]:
    """Embed a list of (cid, url) in memory (parity and benchmark runs).

    :param emb: Embedder.
    :param items: Fragments.
    :param workers: DataLoader workers.
    :param threads: Download threads per worker.
    :param label: Progress label.
    :param every: Progress print interval (s).
    :returns: (ids, {kind: (n, d) float32}, meta rows, failures, progress).
    :rtype: tuple
    """
    chunks = make_chunks({0: items})
    prog = Progress(len(items), label, every)
    ids, meta, failed, vecs = [], [], [], defaultdict(list)
    t = time.time()
    for c in iterate_chunks(chunks, emb.processor, workers, threads):
        t_wait = time.time() - t
        t = time.time()
        failed += c["failed"]
        if c["n"]:
            out = emb.embed_chunk(c)
            for k, v in out.items():
                vecs[k].append(v)
            ids += c["ids"]
            meta += c["meta"]
        t_gpu = time.time() - t
        prog.update(c["n"], len(c["failed"]), t_wait, t_gpu)
        t = time.time()
    prog.update(0, 0, 0.0, 0.0, force=True)
    return ids, {k: np.concatenate(v) for k, v in vecs.items()}, meta, failed, prog


def run_parity(emb: GalleryEmbedder, inputs: dict, workers: int, threads: int, min_cos: float = PARITY_COS,
               precision_check: bool = True) -> dict:
    """Gate 1: 200 benchmark fragments through this pipeline vs the locally computed features and crop geometry.

    Checks (all must hold): the same max1600 size for every fragment; the same masked-crop size, mask path and
    ink-patch count for at least ``GEOMETRY_MIN_SHARE`` of them (identical preprocessing; a different JPEG/Pillow build
    can move a mask edge on a few images); DINOv2 ``masked`` and ``mpatches`` cosine > ``min_cos`` to the local cache
    for every fragment; ``pooled`` == :func:`pooled_from_parts` (ds8, merger). Also prints how much the tower output
    moves between bf16 autocast and fp32 weights (information only) and a gallery ETA from the measured throughput.

    :param emb: Embedder.
    :param inputs: Output of :func:`load_inputs`.
    :param workers: DataLoader workers.
    :param threads: Download threads per worker.
    :param min_cos: Cosine gate.
    :param precision_check: Also run the tower without autocast on these fragments.
    :returns: Report dict.
    :rtype: dict
    """
    import PIL

    par = inputs["parity"]
    pids = [str(c) for c in par["ids"]]
    items = [(c, str(u)) for c, u in zip(pids, par["urls"])]
    ids, vec, meta, failed, prog = embed_items(emb, items, workers, threads, "parity", every=15)
    assert not failed, f"parity fragments failed to download/preprocess: {failed[:3]}"
    pos = {c: i for i, c in enumerate(pids)}
    order = [pos[c] for c in ids]
    rep: dict = {"n": len(ids), "pillow": PIL.__version__, "pillow_expected": PILLOW_VERSION}
    geo = {"max1600_wh": np.array([m["max1600_wh"] for m in meta]), "crop_wh": np.array([m["crop_wh"] for m in meta]),
           "mask_ok": np.array([m["mask_ok"] for m in meta]), "n_patches": np.array([m["n_patches"] for m in meta])}
    for key, val in geo.items():
        ref = par[key][order]
        same = (val.reshape(len(ids), -1) == ref.reshape(len(ids), -1)).all(1)
        rep[f"same_{key}"] = int(same.sum())
        if not same.all():
            bad = [ids[i] for i in np.nonzero(~same)[0][:5]]
            print(f"!! {key} differs for {int((~same).sum())} fragments, e.g. {bad}", flush=True)
    for key in ("dino_masked", "dino_mpatches"):
        ref = par[key][order].astype(np.float32)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        cos = (ref * vec[key]).sum(1)
        rep[f"{key}_cos"] = {"min": round(float(cos.min()), 5), "p01": round(float(np.quantile(cos, 0.01)), 5),
                             "mean": round(float(cos.mean()), 6), "below_gate": int((cos <= min_cos).sum())}
    rebuilt = pooled_from_parts(vec["ds8"].astype(np.float16), vec["merger"].astype(np.float16))
    rep["pooled_rebuild_min_cos"] = round(float((rebuilt * vec["pooled"]).sum(1).min()), 6)
    rep["throughput_fragments_per_s"] = round(prog.done / max(1e-9, time.time() - prog.t0), 2)
    n_gallery = len(gallery_rows(inputs["rows"]))
    rep["gallery_eta_hours_at_this_rate"] = round(n_gallery / max(1e-9, rep["throughput_fragments_per_s"]) / 3600, 2)
    if precision_check:  # bf16 autocast (published eval) vs the fp32 merged weights without autocast (TF32 if enabled)
        sub = make_chunks({0: items[:48]})
        cs = []
        for c in iterate_chunks(sub, emb.processor, min(workers, 4), threads):
            a = emb.tower(c["pixel_values"], c["grid_thw"], bf16=True)["merger"]
            b = emb.tower(c["pixel_values"], c["grid_thw"], bf16=False)["merger"]
            cs.append((a * b).sum(1))
        cs = np.concatenate(cs)
        rep["tower_merger_cos_bf16_autocast_vs_fp32_weights"] = {"min": round(float(cs.min()), 5), "mean": round(float(cs.mean()), 6)}
    print(json.dumps(rep, indent=1), flush=True)
    ok = (rep["dino_masked_cos"]["below_gate"] == 0 and rep["dino_mpatches_cos"]["below_gate"] == 0
          and rep["same_max1600_wh"] == len(ids) and rep["pooled_rebuild_min_cos"] > 0.999
          and all(rep[f"same_{k}"] >= GEOMETRY_MIN_SHARE * len(ids) for k in ("crop_wh", "mask_ok", "n_patches")))
    assert ok, "PARITY FAILED: this pipeline does not reproduce the local features; do not start the gallery run"
    print(f"PARITY OK: {len(ids)} fragments; gallery of {n_gallery:,} at {rep['throughput_fragments_per_s']}/s "
          f"-> about {rep['gallery_eta_hours_at_this_rate']} h (the parity set is too small for a steady-state rate)",
          flush=True)
    return rep


def run_benchmark(emb: GalleryEmbedder, inputs: dict, workers: int, threads: int, out_dir: str,
                  tolerance: float = BENCH_TOLERANCE) -> dict:
    """Gate 2: the 5,176 benchmark fragments streamed through this pipeline must reproduce the published I1 numbers.

    Uses the benchmark's own image URLs (the photographs the local features came from) and
    ``train_image_embedder.fragment_metrics``; the gate is ``merger`` joins R@10 within ``tolerance`` of the
    published value (0.592). Fragments that fail to download are dropped from the metric (reported).

    :param emb: Embedder.
    :param inputs: Output of :func:`load_inputs`.
    :param workers: DataLoader workers.
    :param threads: Download threads per worker.
    :param out_dir: Where ``benchmark_features.npz`` / ``benchmark_report.json`` are written.
    :param tolerance: Allowed |R@10 - published|.
    :returns: Report dict.
    :rtype: dict
    """
    tie = emb.tie
    labels = inputs["labels"]
    items = [(c, inputs["bench_urls"][c]) for c in labels["ids"]]
    ids, vec, meta, failed, prog = embed_items(emb, items, workers, threads, "benchmark", every=60)
    keep = dict(labels, ids=ids)  # fragment_metrics reads labels["ids"] and filters every label by membership
    fused = {"merger+dinov2": tie.fuse([vec["dino_masked"], vec["dino_mpatches"], vec["merger"]]),
             "pooled+dinov2": tie.fuse([vec["dino_masked"], vec["dino_mpatches"], vec["pooled"]]),
             "dinov2_only": tie.fuse([vec["dino_masked"], vec["dino_mpatches"]])}
    rep: dict = {"n_benchmark": len(items), "n_embedded": len(ids), "failed": failed[:50], "n_failed": len(failed),
                 "seconds": round(time.time() - prog.t0, 1), "metrics": {}}
    for name, X in [(k, vec[k]) for k in ("merger", "pooled", "head")] + list(fused.items()):
        m = tie.fragment_metrics(X, keep)
        ref = inputs["reference"].get(f"i1_{name}")
        row = {"joins_R@10": m["joins"]["all"]["R@10"], "cross_R@10": m["joins"]["cross_collection"].get("R@10"),
               "median_rank": m["joins"]["all"].get("median_rank"), "scribe_P@1": m.get("scribes", {}).get("P@1")}
        if ref:
            row["published_joins_R@10"] = ref["joins"]["all"]["R@10"]
            row["published_scribe_P@1"] = ref.get("scribes", {}).get("P@1")
            row["delta_joins_R@10"] = round(row["joins_R@10"] - row["published_joins_R@10"], 4)
        rep["metrics"][name] = row
        print(f"{name:16s} {json.dumps(row)}", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out_dir) / "benchmark_features.npz", ids=np.array(ids), urls=np.array([inputs["bench_urls"][c] for c in ids]),
             **{k: vec[k].astype(np.float16) for k in STORED})
    (Path(out_dir) / "benchmark_report.json").write_text(json.dumps(rep, indent=1))
    d = rep["metrics"]["merger"]["delta_joins_R@10"]
    assert abs(d) <= tolerance, (f"BENCHMARK GATE FAILED: merger joins R@10 {rep['metrics']['merger']['joins_R@10']} vs "
                                 f"published {rep['metrics']['merger']['published_joins_R@10']} (|delta| {abs(d)} > "
                                 f"{tolerance}); do not start the gallery run")
    print(f"BENCHMARK OK: merger joins R@10 {rep['metrics']['merger']['joins_R@10']} (published "
          f"{rep['metrics']['merger']['published_joins_R@10']}), {len(ids)}/{len(items)} fragments, "
          f"{rep['seconds'] / 60:.1f} min", flush=True)
    return rep


def write_shard(path: Path, shard: int, ids: List[str], urls: List[str], vecs: Dict[str, List[np.ndarray]],
                meta: List[dict], run_info: dict) -> None:
    """Write one fp16 shard (tmp + rename).

    :param path: Output ``.npz`` path.
    :param shard: Shard id.
    :param ids: Fragment ids (rows).
    :param urls: Image URL per row.
    :param vecs: kind -> list of float32 blocks.
    :param meta: Provenance per row.
    :param run_info: Settings recorded in the shard.
    """
    arrays = {k: np.concatenate(vecs[k]).astype(np.float16) for k in STORED} if ids else \
        {k: np.zeros((0, 1), np.float16) for k in STORED}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(ids, dtype=str), urls=np.array(urls, dtype=str),
             sha1=np.array([m["sha1"] for m in meta], dtype=str),
             max1600_wh=np.array([m["max1600_wh"] for m in meta], dtype=np.int32).reshape(-1, 2),
             crop_wh=np.array([m["crop_wh"] for m in meta], dtype=np.int32).reshape(-1, 2),
             mask_ok=np.array([m["mask_ok"] for m in meta], dtype=bool),
             mask_frac=np.array([m["mask_frac"] for m in meta], dtype=np.float32),
             n_patches=np.array([m["n_patches"] for m in meta], dtype=np.int16),
             dhash=np.array([m["dhash"] for m in meta], dtype=np.uint64),
             run_info=np.array(json.dumps(dict(run_info, shard=shard))), **arrays)
    tmp.rename(path)


def _upload(api: "HfApi", repo_id: str, files: List[Tuple[Path, str]], message: str, retries: int = 5) -> str:
    """One commit with several files, retried.

    :param api: HfApi.
    :param repo_id: Dataset repo.
    :param files: [(local path, path in repo)].
    :param message: Commit message.
    :param retries: Attempts.
    :returns: Commit sha.
    :rtype: str
    """
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in files]
    for attempt in range(retries):
        try:
            return api.create_commit(repo_id, operations=ops, commit_message=message, repo_type="dataset").oid
        except Exception as exc:  # hub hiccups (429/5xx/timeouts): back off and retry, never lose a finished shard
            print(f"upload {message!r} failed ({type(exc).__name__}: {exc}); retry {attempt + 1}/{retries}", flush=True)
            time.sleep(10 * 2 ** attempt)
    raise RuntimeError(f"upload failed: {message}")


def done_shards(api: "HfApi", repo_id: str, prefix: str = "shards/part_") -> set:
    """Shard names already in the repo.

    :param api: HfApi.
    :param repo_id: Dataset repo.
    :param prefix: Shard path prefix.
    :returns: Set of file names like ``part_0003.npz``.
    :rtype: set
    """
    return {f.split("/")[-1] for f in api.list_repo_files(repo_id, repo_type="dataset") if f.startswith(prefix)}


def run_info(emb: GalleryEmbedder, extra: Optional[dict] = None) -> dict:
    """Software/settings record stored in every shard.

    :param emb: Embedder.
    :param extra: Additional fields.
    :returns: Dict.
    :rtype: dict
    """
    import PIL
    import torch
    import transformers

    return {"tower_repo": TOWER_REPO, "tower_revision": TOWER_REVISION, "dino": DINO_ID, "dino_revision": DINO_REVISION,
            "tower_bf16_autocast": emb.bf16, "dino_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "stored": list(STORED), "pooled": "pooled_from_parts(ds8, merger)",
            "torch": torch.__version__, "transformers": transformers.__version__, "pillow": PIL.__version__,
            "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}


def run_shards(emb: GalleryEmbedder, shards: Dict[str, List[Tuple[str, str]]], repo_id: str, token: str,
               out_dir: str, workers: int, threads: int, label: str = "gallery",
               max_failure_share: Optional[float] = MAX_SHARD_FAILURE_SHARE) -> dict:
    """Embed shards, write each when complete, upload it (background thread) with its failure log; skip done ones.

    :param emb: Embedder.
    :param shards: shard name (``part_0000``) -> items.
    :param repo_id: Dataset repo (created if missing).
    :param token: HF token with write access.
    :param out_dir: Local shard folder.
    :param workers: DataLoader workers.
    :param threads: Download threads per worker.
    :param label: Progress label.
    :param max_failure_share: Stop (without uploading the shard) when more of a shard fails; None = never.
    :returns: Summary {shards_done, skipped, fragments, failed}.
    :rtype: dict
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    have = done_shards(api, repo_id)
    names = sorted(shards)
    todo = [n for n in names if f"{n}.npz" not in have]
    print(f"{len(names) - len(todo)} of {len(names)} shards already in {repo_id}; {len(todo)} to do "
          f"({sum(len(shards[n]) for n in todo):,} fragments)", flush=True)
    if not todo:
        return {"shards_done": 0, "skipped": len(names), "fragments": 0, "failed": 0}
    out = Path(out_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(parents=True, exist_ok=True)
    info = run_info(emb, {"repo": repo_id})
    idx = {n: k for k, n in enumerate(todo)}
    chunks = make_chunks({idx[n]: shards[n] for n in todo})
    n_chunks = Counter(s for s, _ in chunks)
    prog = Progress(sum(len(shards[n]) for n in todo), label, every=120)
    buf = defaultdict(lambda: {"ids": [], "urls": [], "vecs": defaultdict(list), "meta": [], "failed": [], "chunks": 0})
    uploader, futures = ThreadPoolExecutor(1), []
    totals = Counter()
    t = time.time()
    for c in iterate_chunks(chunks, emb.processor, workers, threads):
        t_wait = time.time() - t
        t = time.time()
        b = buf[c["shard"]]
        b["chunks"] += 1
        b["failed"] += c["failed"]
        if c["n"]:
            vecs = emb.embed_chunk(c)
            for k in STORED:
                b["vecs"][k].append(vecs[k])
            b["ids"] += c["ids"]
            b["urls"] += c["urls"]
            b["meta"] += c["meta"]
        prog.update(c["n"], len(c["failed"]), t_wait, time.time() - t)
        if b["chunks"] == n_chunks[c["shard"]]:  # shard complete: write + queue the upload
            name = todo[c["shard"]]
            n_items = len(b["ids"]) + len(b["failed"])
            # only transient failures signal an outage: permanent 404s cluster by box (one manifest shard had 29 %
            # dead images before the URL check) and would otherwise stop every re-run at the same shard
            transient = [f for f in b["failed"] if f["stage"] == "download" and not is_permanent(f["error"])]
            if max_failure_share is not None and len(transient) > max_failure_share * n_items:  # an outage: stop
                for f in futures:
                    f.result()
                errors = Counter(f["error"][:60] for f in transient).most_common(3)
                raise RuntimeError(f"{name}: {len(transient)}/{n_items} transient download failures (e.g. {errors}); "
                                   "not uploaded. Check the network / bucket and re-run the cell: finished shards are "
                                   "skipped.")
            path, fpath = out / "shards" / f"{name}.npz", out / "failures" / f"{name}.jsonl"
            write_shard(path, c["shard"], b["ids"], b["urls"], b["vecs"], b["meta"], dict(info, name=name))
            with open(fpath, "w", encoding="utf-8") as fh:
                for f in b["failed"]:
                    fh.write(json.dumps(dict(f, shard=name)) + "\n")
            totals.update(fragments=len(b["ids"]), failed=len(b["failed"]), shards_done=1)
            futures.append(uploader.submit(_upload, api, repo_id, [(path, f"shards/{name}.npz"),
                                                                   (fpath, f"failures/{name}.jsonl")],
                                           f"{name}: {len(b['ids'])} fragments, {len(b['failed'])} failed"))
            print(f"{name} written ({len(b['ids'])} ok, {len(b['failed'])} failed); upload queued | {prog.line()}",
                  flush=True)
            del buf[c["shard"]]
        t = time.time()
    for f in futures:
        f.result()  # surface upload errors
    uploader.shutdown()
    prog.update(0, 0, 0.0, 0.0, force=True)
    return dict(totals, skipped=len(names) - len(todo))


def run_gallery(emb: GalleryEmbedder, inputs: dict, repo_id: str, token: str, out_dir: str, workers: int,
                threads: int, limit_shards: Optional[int] = None) -> dict:
    """The full gallery: every manifest row, in fixed shards of ``SHARD_SIZE``.

    :param emb: Embedder.
    :param inputs: Output of :func:`load_inputs`.
    :param repo_id: Dataset repo.
    :param token: HF token (write).
    :param out_dir: Local output folder.
    :param workers: DataLoader workers.
    :param threads: Download threads per worker.
    :param limit_shards: Only the first N pending shards (a timed trial).
    :returns: Summary.
    :rtype: dict
    """
    plan = {f"part_{k:04d}": items for k, items in enumerate(shard_plan(inputs["rows"]))}
    if limit_shards:
        from huggingface_hub import HfApi

        have = done_shards(HfApi(token=token), repo_id)
        plan = dict([(n, v) for n, v in plan.items() if f"{n}.npz" not in have][:limit_shards])
    return run_shards(emb, plan, repo_id, token, out_dir, workers, threads)


def collect_failures(repo_id: str, token: str, local_dir: str = "/content/gallery_failures") -> Tuple[List[dict], set]:
    """All failure rows in the repo, and the ids that a retry shard has since embedded.

    :param repo_id: Dataset repo.
    :param token: HF token.
    :param local_dir: Download folder.
    :returns: (failure rows of the main shards, ids embedded by ``part_r*`` retry shards).
    :rtype: Tuple[List[dict], set]
    """
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id, repo_type="dataset", token=token, local_dir=local_dir,
                                  allow_patterns=["failures/*", "shards/part_r*"]))
    rows = [json.loads(l) for f in sorted((path / "failures").glob("part_[0-9]*.jsonl"))
            for l in open(f, encoding="utf-8") if l.strip()]
    fixed = {str(i) for f in sorted((path / "shards").glob("part_r*.npz")) for i in np.load(f)["ids"]} \
        if (path / "shards").exists() else set()
    return rows, fixed


def run_retry(emb: GalleryEmbedder, inputs: dict, repo_id: str, token: str, out_dir: str, workers: int,
              threads: int) -> dict:
    """Retry the logged failures that no retry shard has fixed yet (not permanent HTTP 4xx) into ``part_rNNN``.

    :param emb: Embedder.
    :param inputs: Output of :func:`load_inputs` (unused; kept for a uniform cell signature).
    :param repo_id: Dataset repo.
    :param token: HF token (write).
    :param out_dir: Local output folder.
    :param workers: DataLoader workers.
    :param threads: Download threads per worker.
    :returns: Summary, with the fragments still missing listed.
    :rtype: dict
    """
    from huggingface_hub import HfApi

    rows, fixed = collect_failures(repo_id, token)
    open_rows = [r for r in rows if r["id"] not in fixed]
    permanent = [r for r in open_rows if is_permanent(r["error"])]
    todo = sorted({(r["id"], r["url"]) for r in open_rows if not is_permanent(r["error"])})
    print(f"{len(rows)} logged failures, {len(fixed)} fixed by earlier retries, {len(permanent)} permanent "
          f"(HTTP 4xx); retrying {len(todo)}", flush=True)
    n_prev = len({f for f in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
                  if f.startswith("shards/part_r")})
    plan = {f"part_r{n_prev + k:03d}": todo[i:i + SHARD_SIZE] for k, i in enumerate(range(0, len(todo), SHARD_SIZE))}
    summary = run_shards(emb, plan, repo_id, token, out_dir, workers, threads, label="retry",
                         max_failure_share=None) if plan else {}
    return dict(summary, permanent=[{k: r[k] for k in ("id", "url", "error")} for r in permanent])


def gate_passed_in_repo(repo_id: str, token: str, tolerance: float = BENCH_TOLERANCE) -> bool:
    """Whether a benchmark gate run of this pipeline already passed (``benchmark/benchmark_report.json`` in the repo).

    Lets a resumed session go straight to the gallery cell without re-running the 5,176-fragment gate.

    :param repo_id: Dataset repo.
    :param token: HF token.
    :param tolerance: Allowed |R@10 - published|.
    :returns: True when the stored report passes.
    :rtype: bool
    """
    from huggingface_hub import HfApi, hf_hub_download

    name = "benchmark/benchmark_report.json"
    if name not in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"):
        return False
    rep = json.loads(Path(hf_hub_download(repo_id, name, repo_type="dataset", token=token)).read_text())
    d = rep["metrics"]["merger"].get("delta_joins_R@10")
    return d is not None and abs(d) <= tolerance


def upload_benchmark(repo_id: str, token: str, out_dir: str) -> str:
    """Upload the gate's features and report to ``benchmark/``.

    :param repo_id: Dataset repo.
    :param token: HF token (write).
    :param out_dir: Folder :func:`run_benchmark` wrote to.
    :returns: Commit sha.
    :rtype: str
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    out = Path(out_dir)
    return _upload(api, repo_id, [(out / "benchmark_features.npz", "benchmark/benchmark_features.npz"),
                                  (out / "benchmark_report.json", "benchmark/benchmark_report.json")],
                   "benchmark gate: 5,176 fragments through the gallery pipeline")


def verify_repo(repo_id: str, token: str, n_manifest: int) -> dict:
    """Count shards / fragments in the repo and write ``gallery_index.json`` (shard -> ids) there.

    :param repo_id: Dataset repo.
    :param token: HF token (write).
    :param n_manifest: Manifest size.
    :returns: Summary.
    :rtype: dict
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = sorted(f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.startswith("shards/") and f.endswith(".npz"))
    index, seen = {}, set()
    for f in files:
        z = np.load(hf_hub_download(repo_id, f, repo_type="dataset", token=token))
        ids = [str(i) for i in z["ids"]]
        index[f] = len(ids)
        seen |= set(ids)
    summary = {"shards": len(files), "fragments_in_shards": sum(index.values()), "unique_fragments": len(seen),
               "manifest": n_manifest, "coverage": round(len(seen) / max(1, n_manifest), 5), "per_shard": index}
    tmp = Path("/content/gallery_index.json")
    tmp.write_text(json.dumps(summary, indent=1))
    _upload(api, repo_id, [(tmp, "gallery_index.json")], f"gallery index: {len(seen)} fragments")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_shard"}), flush=True)
    return summary
