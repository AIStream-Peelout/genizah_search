#!/usr/bin/env python3
"""Memory/disk guard for embedding-audit probes on the shared Mac Studio.

The Studio also runs LM Studio (production chat models) and the sibling
``historical-document-analysis`` checkpoint-eval chain, whose PEFT merge +
MLX convert phases spike to ~20-25 GB RSS. This wrapper launches one probe
command, polls box health every few seconds, and kills the probe (it must be
resumable from its own on-disk cache) whenever the box is under pressure or a
sibling heavy phase starts, then relaunches once things are calm again.

Usage::

    guard.py --name text_probe -- /path/to/python probe.py --args

It never touches any process other than the child it launched.
"""

import argparse
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

STATUS_FILE = Path.home() / "box_jobs" / "genizah_search_embedding_audit.status"
HDA_REPO = Path.home() / "Documents" / "GitHub" / "historical-document-analysis"
HEAVY_PROC_PATTERN = r"merge_ckpt_generic|mlx_vlm.*convert"  # merge/convert spikes only; the benchmark phase is compatible
# sibling checkpoint benchmarks (GPU-bound): staging (hard_eval_ckpt.sh), the harness scripts (incl. run_pgp131_v19b.py,
# hence digits) and the v23 chain's LM Studio benchmarks, run as modules (python -m ...helper_eval_scripts.run_...)
EVAL_PROC_PATTERN = (r"hard_eval_ckpt\.sh|eval_harness/[a-z0-9_]+\.py"
                     r"|helper_eval_scripts\.run_(arabic_benchmark|vqa_parse_eval)")
SMALL_MAX_GB = 3.0  # --small smoke tests: hard footprint cap; they run beside a queue step, outside the mutex
SMALL_JOB = False

# Launch gates (stricter) and kill gates (looser) so we don't flap.
START_MIN_FREE_PCT = 28
START_MAX_SWAP_USED_GB = 13.0  # macOS swap stays allocated after pressure passes
START_MIN_DISK_GB = 30.0
KILL_MIN_FREE_PCT = 25
KILL_MAX_SWAP_USED_GB = 15.0  # box crises were at 16-18.5 GB (HDA memory_budget_report)
KILL_MIN_DISK_GB = 25.0
CHILD_MAX_RSS_GB = 8.0
POLL_SECONDS = 5


def free_memory_pct() -> int:
    """Return macOS system-wide free memory percentage.

    :returns: Free memory percent as reported by ``memory_pressure``.
    :rtype: int
    """
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    match = re.search(r"free percentage:\s*(\d+)%", out)
    return int(match.group(1)) if match else 0


def swap_used_gb() -> float:
    """Return swap currently in use, in GB.

    :returns: Used swap in GB (dynamic swap total is not a useful ceiling).
    :rtype: float
    """
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) / 1024 if match else 99.0


def disk_free_gb(path: str = "/") -> float:
    """Return free space on the internal disk in GB.

    :param path: Mount point to check.
    :returns: Free GB.
    :rtype: float
    """
    return shutil.disk_usage(path).free / 1e9


def sibling_heavy_phase() -> Optional[str]:
    """Detect a sibling merge/convert phase.

    :returns: A short reason string if a heavy phase is active, else None.
    :rtype: Optional[str]
    """
    found = subprocess.run(["pgrep", "-fl", HEAVY_PROC_PATTERN], capture_output=True, text=True).stdout.strip()
    # ignore other guards' own pgrep probes, whose command line contains the pattern
    hits = [l for l in found.splitlines() if " pgrep " not in f" {l.split(' ', 1)[-1]}" and "guard.py" not in l
            and "shell-snapshots" not in l]  # a Claude session's shell command that merely mentions the pattern
    if hits:
        return f"sibling process: {hits[0][:120]}"
    now = time.time()
    for log in glob.glob(str(HDA_REPO / "logs" / "hard_eval_*_full.log")):  # any version (v22b, v23a …)
        created = os.stat(log).st_birthtime
        if now - created < 15 * 60:
            return f"fresh sibling eval log {Path(log).name}"
    return None


def sibling_eval_running() -> Optional[str]:
    """Detect a sibling v22b checkpoint eval (its benchmarks are GPU-bound in LM Studio/MLX).

    An eval is in progress when the newest ``hard_eval_v22b_<step>_full.log`` is under 3 h old
    and the chain log has not yet recorded that step's exit. GPU probes yield during it
    (agreed with the sibling session: sharing the GPU slowed its benchmark by 1/3-1/2).

    :returns: Reason string if an eval is running, else None.
    :rtype: Optional[str]
    """
    # any version's checkpoint eval (v23a+ chains run the same harness): the process itself is the signal
    found = subprocess.run(["pgrep", "-fl", EVAL_PROC_PATTERN], capture_output=True, text=True).stdout.strip()
    hits = [l for l in found.splitlines() if " pgrep " not in f" {l.split(' ', 1)[-1]}" and "guard.py" not in l
            and "shell-snapshots" not in l]  # a Claude session's shell command that merely mentions the pattern
    if hits:
        return f"sibling checkpoint eval process (GPU): {hits[0][:100]}"
    chain = HDA_REPO / "logs" / "v22b_eval_chain.log"
    text = chain.read_text(errors="ignore") if chain.exists() else ""
    # Final LM Studio window after the last checkpoint: the Arabic benchmark (~70 min), ended by
    # "=== V22B EVAL CHAIN DONE" (wording confirmed by the sibling session).
    arabic = [m for m in re.finditer(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\n]*arabic benchmark", text, re.I | re.M)]
    if arabic and "V22B EVAL CHAIN DONE" not in text[arabic[-1].end():]:
        started = time.mktime(time.strptime(arabic[-1].group(1), "%Y-%m-%d %H:%M:%S"))
        if time.time() - started < 3 * 3600:
            return "sibling final Arabic benchmark in progress (GPU)"
    logs = glob.glob(str(HDA_REPO / "logs" / "hard_eval_v22b_*_full.log"))
    if not logs:
        return None
    newest = max(logs, key=lambda p: os.stat(p).st_birthtime)
    if time.time() - os.stat(newest).st_birthtime > 3 * 3600:
        return None
    step = re.search(r"hard_eval_v22b_(\d+)_full", newest).group(1)
    if re.search(rf"hard eval[^\n]*step {step}[^\n]*exit", text, re.I):
        return None
    return f"sibling eval step {step} in progress (GPU)"


def child_rss_gb(pid: int) -> float:
    """Return a process's physical footprint (includes MPS/GPU allocations).

    :param pid: Process id.
    :returns: Footprint in GB, or 0 if unavailable.
    :rtype: float
    """
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    match = re.search(r"Footprint:\s*([\d.]+)\s*([KMG])B", out)
    if not match:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return int(rss) / 1048576 if rss else 0.0
    value, unit = float(match.group(1)), match.group(2)
    return value / {"K": 1048576, "M": 1024, "G": 1}[unit]


GPU_JOB = True  # set False via --cpu-only for jobs that never touch MPS


def box_state() -> Tuple[int, float, float, Optional[str]]:
    """Collect free %, swap used, disk free and sibling-heavy reason.

    :returns: Tuple of (free_pct, swap_used_gb, disk_free_gb, heavy_reason).
    :rtype: Tuple[int, float, float, Optional[str]]
    """
    heavy = sibling_heavy_phase() or (sibling_eval_running() if GPU_JOB else None)
    return free_memory_pct(), swap_used_gb(), disk_free_gb(), heavy


MUTEX = Path.home() / "box_jobs" / "genizah_search_embedding_audit.mutex"


GPU_WAITING = Path.home() / "box_jobs" / "genizah_search_embedding_audit.gpu_waiting"


def gpu_job_waiting() -> bool:
    """True if a GPU job is queued for the mutex while the GPU is free of sibling evals.

    GPU jobs can only run between the sibling's eval windows; CPU jobs can run any time. So
    while the GPU is usable, a waiting GPU job takes priority and CPU jobs step aside.

    :returns: Whether CPU jobs should yield.
    :rtype: bool
    """
    try:
        os.kill(int(GPU_WAITING.read_text().split()[0]), 0)
    except (ProcessLookupError, ValueError, IndexError, FileNotFoundError):
        return False
    return sibling_eval_running() is None


def acquire_mutex(name: str) -> None:
    """Block until this guard holds the audit-wide mutex (one probe process at a time, all queues).

    The mutex file holds the owning guard's pid; a stale file (dead pid) is taken over. ``--small`` jobs
    (smoke tests capped at 3 GB, CPU) skip the mutex so they never wait hours behind a queue step.

    :param name: Probe name for the status file.
    """
    if SMALL_JOB:
        return
    while True:
        if GPU_JOB:
            GPU_WAITING.write_text(f"{os.getpid()} {name}")
        elif gpu_job_waiting():
            write_status(name, "queued (yielding to a GPU job while the GPU is free)", None, "")
            time.sleep(30)
            continue
        try:
            fd = os.open(MUTEX, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {name}".encode())
            os.close(fd)
            if GPU_JOB:
                GPU_WAITING.unlink(missing_ok=True)
            return
        except FileExistsError:
            try:
                owner = int(MUTEX.read_text().split()[0])
                os.kill(owner, 0)
            except (ProcessLookupError, ValueError, IndexError, FileNotFoundError):
                MUTEX.unlink(missing_ok=True)
                continue
            if owner == os.getpid():
                return
            write_status(name, "queued (another audit probe holds the mutex)", None, f"owner pid {owner}")
            time.sleep(30)


def release_mutex() -> None:
    """Release the audit-wide mutex if this guard owns it."""
    if SMALL_JOB:
        return
    try:
        if int(MUTEX.read_text().split()[0]) == os.getpid():
            MUTEX.unlink()
    except (FileNotFoundError, ValueError, IndexError):
        pass


def write_status(name: str, state: str, pid: Optional[int], detail: str) -> None:
    """Write the shared status file other sessions read before heavy launches.

    :param name: Probe name.
    :param state: Short state label.
    :param pid: Child pid if running.
    :param detail: Free-text detail.
    """
    if SMALL_JOB:  # smoke tests must not overwrite the running queue step's status
        return
    STATUS_FILE.parent.mkdir(exist_ok=True)
    STATUS_FILE.write_text(
        "job: genizah_search embedding audit (Claude session 'Semantic retrieval model fine-tuning analysis')\n"
        f"updated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"probe: {name}\nstate: {state}\npid: {pid}\ndetail: {detail}\n"
        "footprint cap: <= 10 GB per probe (killed above); no LM Studio use; caches on NAS\n"
        f"kills itself when free RAM < {KILL_MIN_FREE_PCT}%, swap used > {KILL_MAX_SWAP_USED_GB} GB, "
        f"disk < {KILL_MIN_DISK_GB} GB, or a sibling merge/convert is running\n"
    )


def log(msg: str) -> None:
    """Print a timestamped guard message.

    :param msg: Message text.
    """
    print(f"[guard {datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def wait_until_calm(name: str) -> None:
    """Block until launch gates pass on two consecutive checks.

    :param name: Probe name for the status file.
    """
    calm = 0
    while calm < 2:
        free, swap, disk, heavy = box_state()
        ok = free >= START_MIN_FREE_PCT and swap <= START_MAX_SWAP_USED_GB and disk >= START_MIN_DISK_GB and not heavy
        calm = calm + 1 if ok else 0
        if not ok:
            detail = f"free={free}% swap={swap:.1f}GB disk={disk:.0f}GB heavy={heavy}"
            write_status(name, "waiting", None, detail)
            log(f"waiting: {detail}")
            time.sleep(60)
        else:
            time.sleep(POLL_SECONDS)


def run_guarded(name: str, cmd: List[str], max_restarts: int, max_gb: float = CHILD_MAX_RSS_GB) -> int:
    """Run ``cmd`` under the guard, relaunching after pressure kills.

    :param name: Probe name.
    :param cmd: Command line to execute.
    :param max_restarts: Maximum relaunches after guard kills.
    :param max_gb: Child footprint cap in GB (hard ceiling 10).
    :returns: The child's final exit code (or 75 if restarts were exhausted).
    :rtype: int
    """
    env = dict(os.environ)
    # ~8 GB hard MPS cap on this box; the low watermark must sit below the high one.
    max_gb = min(max_gb, 10.0)
    # MPS allocations get a hard cap ~3 GB below the footprint cap (CPU-side tensors, weights
    # staging, Python) so the allocator errors out instead of the process outgrowing the cap.
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", f"{(max_gb - 3) / 100:.3f}")
    env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", f"{(max_gb - 3) * 0.75 / 100:.3f}")
    env.setdefault("HF_HOME", "/Volumes/home/studio_offload/genizah_search_embedding_audit/hf")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for attempt in range(max_restarts + 1):
        # Wait for calm WITHOUT holding the mutex (a GPU job waiting out a sibling eval must not
        # block CPU-only jobs), then take the mutex and re-check before launching.
        while True:
            wait_until_calm(name)
            acquire_mutex(name)
            free, swap, disk, heavy = box_state()
            if free >= START_MIN_FREE_PCT and swap <= START_MAX_SWAP_USED_GB and disk >= START_MIN_DISK_GB and not heavy:
                break
            release_mutex()
        child = subprocess.Popen(cmd, env=env)
        log(f"launched pid={child.pid} attempt={attempt}: {' '.join(cmd)[:200]}")
        write_status(name, "running", child.pid, f"attempt {attempt}")
        reason = None
        while child.poll() is None:
            time.sleep(POLL_SECONDS)
            free, swap, disk, heavy = box_state()
            rss = child_rss_gb(child.pid)
            if free < KILL_MIN_FREE_PCT:
                reason = f"free RAM {free}%"
            elif swap > KILL_MAX_SWAP_USED_GB:
                reason = f"swap used {swap:.1f}GB"
            elif disk < KILL_MIN_DISK_GB:
                reason = f"disk free {disk:.0f}GB"
            elif heavy:
                reason = heavy
            elif rss > max_gb:
                reason = f"child footprint {rss:.1f}GB"
            elif not GPU_JOB and not SMALL_JOB and gpu_job_waiting():
                reason = "yielding to a queued GPU job (GPU free between sibling evals)"
            if reason:
                log(f"killing pid={child.pid}: {reason}")
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
        release_mutex()
        if reason is None:
            write_status(name, f"finished rc={child.returncode}", None, "idle")
            log(f"finished rc={child.returncode}")
            return child.returncode
        write_status(name, "killed-by-guard", None, reason)
        if reason.startswith("child footprint"):
            log("child exceeded its own cap; not relaunching")
            return 70
    return 75


def main() -> None:
    """Parse arguments and run the guarded command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-restarts", type=int, default=30)
    parser.add_argument("--max-gb", type=float, default=CHILD_MAX_RSS_GB)
    parser.add_argument("--cpu-only", action="store_true", help="job never uses the GPU; ignore sibling GPU evals")
    parser.add_argument("--small", action="store_true",
                        help="smoke test: CPU only, footprint capped at 3 GB, 2 threads, no mutex")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    global GPU_JOB, SMALL_JOB
    SMALL_JOB = args.small
    GPU_JOB = not (args.cpu_only or args.small)
    if SMALL_JOB:
        args.max_gb = min(args.max_gb, SMALL_MAX_GB)
        os.environ.update(AUDIT_DEVICE="cpu", AUDIT_THREADS="2", OMP_NUM_THREADS="2")
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    try:
        sys.exit(run_guarded(args.name, cmd, args.max_restarts, args.max_gb))
    finally:
        release_mutex()


if __name__ == "__main__":
    main()
