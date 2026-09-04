"""Compare synthesis models on the agentic RAG evals: quality, speed, and repair effort.

For each candidate synthesis model, every dataset is run through a private
backend instance (started here, on its own port, with an ephemeral evaluation
API key that is never written to disk) using the model-override evaluation
endpoint. Router and verification models stay at their configured defaults, so
differences isolate the synthesis model.

Per case the runner records deterministic checks, LLM-judge scores, request
latency, per-stage timings, verification cycles / repair attempts, and per-stage
token usage. This script then aggregates everything into ``summary.json`` and a
human-readable ``summary.md`` in the output directory.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/run_synthesis_model_comparison.py \\
        --models qwen/qwen3.6-35b-a3b qwen/qwen3.6-27b \\
        --datasets evals/agentic_rag_v1.json evals/agentic_rag_multiturn_v1.json \\
        --judge-model qwen/qwen3-4b-2507
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import httpx

from scripts.run_agentic_rag_eval import lm_studio_busy_models, wait_for_lm_studio_idle


LMS_CLI = os.path.expanduser("~/.lmstudio/bin/lms")


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :returns: Parsed namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="LM Studio model ids to compare, in run order.")
    parser.add_argument("--summarize-only", metavar="OUTPUT_DIR",
                        help="Rebuild summary.md/summary.json from the JSONL files of a finished run "
                             "(e.g. after scripts/rejudge_eval_results.py) and exit.")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["evals/agentic_rag_v1.json", "evals/agentic_rag_multiturn_v1.json"],
    )
    parser.add_argument("--judge-model", default="qwen/qwen3-4b-2507")
    parser.add_argument("--judge-provider", choices=["lmstudio", "anthropic"], default="lmstudio",
                        help="'anthropic' judges via the Claude API (no local judge load/idle handling).")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--lm-studio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--port", type=int, default=8010, help="Port for the private backend instance.")
    parser.add_argument("--context-length", type=int, default=32768,
                        help="Minimum context length for candidate synthesis models loaded by this script.")
    parser.add_argument("--judge-context-length", type=int, default=32768,
                        help="Minimum context length for the judge model (its input is the whole case plus "
                             "evidence). The judge is also the router; it is (re)loaded pinned and never unloaded.")
    parser.add_argument("--model-ttl", type=int, default=7200,
                        help="Idle TTL (seconds) for candidate models this script loads; 0 pins them (no TTL).")
    parser.add_argument("--baseline-results", nargs="*", default=[], metavar="JSONL",
                        help="Result JSONL files from an earlier run to include in the summary as baselines "
                             "(their dataset_id and synthesis_model are read from the rows), so a new candidate "
                             "can be compared without re-running the baseline model.")
    parser.add_argument("--keep-models-loaded", action="store_true",
                        help="Do not unload models this script loaded when finished.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Restrict to case ids (smoke tests).")
    parser.add_argument("--case-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=900.0,
                        help="LM_STUDIO_REQUEST_TIMEOUT for the private backend. Thinking-heavy dense "
                             "models (~23 tok/s, up to ~6k reasoning tokens) need well over the 300 s default.")
    parser.add_argument("--no-wait-for-idle", action="store_true",
                        help="Do not wait for LM Studio to be idle before starting, loading models, and each case. "
                             "By default the run waits whenever another job (e.g. a checkpoint audit) is generating.")
    parser.add_argument("--idle-quiet-seconds", type=float, default=30.0)
    parser.add_argument("--idle-max-wait-seconds", type=float, default=12 * 3600)
    parser.add_argument("--output-dir")
    return parser.parse_args()


class Progress:
    """Timestamped progress log written to stdout and a file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """Append one timestamped line.

        :param message: Text to log.
        """
        line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def wait_for_health(base_url: str, timeout_seconds: float = 240.0) -> None:
    """Block until the backend answers ``/health``.

    :param base_url: Backend base URL.
    :param timeout_seconds: Give up after this long.
    :raises RuntimeError: If the backend does not come up in time.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise RuntimeError(f"Backend at {base_url} did not become healthy within {timeout_seconds:.0f}s")


def start_backend(
    project_root: Path,
    port: int,
    eval_api_key: str,
    log_path: Path,
    request_timeout: float = 900.0,
) -> subprocess.Popen:
    """Start the private backend instance used for the comparison.

    :param project_root: Repository root.
    :param port: Port to serve on.
    :param eval_api_key: Ephemeral key enabling the evaluation endpoint.
    :param log_path: File receiving the backend's stdout/stderr.
    :param request_timeout: Per-request LM Studio timeout in seconds; must cover
        the slowest candidate's full thinking + answer generation.
    :returns: The backend process.
    :rtype: subprocess.Popen
    """
    env = dict(os.environ)
    env["EVAL_API_KEY"] = eval_api_key
    env["DEV_BACKEND_PORT"] = str(port)
    env["PYTHONPATH"] = str(project_root)
    env["LM_STUDIO_REQUEST_TIMEOUT"] = str(request_timeout)
    log_handle = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(project_root / "scripts" / "dev_backend_local.py")],
        cwd=project_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )


def loaded_models(lm_studio_url: str) -> Dict[str, Dict[str, Any]]:
    """Return LM Studio's model table keyed by id.

    :param lm_studio_url: LM Studio base URL.
    :returns: Model entries by id.
    :rtype: Dict[str, Dict[str, Any]]
    """
    response = httpx.get(f"{lm_studio_url}/api/v0/models", timeout=10.0)
    response.raise_for_status()
    return {entry["id"]: entry for entry in response.json().get("data", []) if entry.get("id")}


def ensure_model_loaded(
    model: str,
    lm_studio_url: str,
    context_length: int,
    ttl: Optional[int],
    progress: Progress,
) -> Dict[str, Any]:
    """Make sure a model is resident with at least the given context length.

    JIT loading through the API uses LM Studio's default context length, which
    is too small for the synthesis prompt and for the judge input; loading here
    keeps every candidate on the same footing. A model that is resident with a
    smaller context is reloaded.

    :param model: LM Studio model id.
    :param lm_studio_url: LM Studio base URL.
    :param context_length: Minimum context length required.
    :param ttl: Idle TTL for the load, or ``None`` to pin the model (no TTL).
    :param progress: Progress logger.
    :returns: ``{"loaded_by_script": bool, "reloaded": bool, "load_seconds": float | None, "state": ...}``.
    :rtype: Dict[str, Any]
    :raises RuntimeError: If the model is unknown or fails to load.
    """
    table = loaded_models(lm_studio_url)
    if model not in table:
        raise RuntimeError(f"Model {model!r} is not downloaded in LM Studio")
    entry = table[model]
    reloaded = False
    if entry.get("state") == "loaded":
        current_context = int(entry.get("loaded_context_length") or 0)
        if current_context >= context_length:
            progress.log(f"{model}: already loaded (ctx {current_context})")
            return {"loaded_by_script": False, "reloaded": False, "load_seconds": None, "state": entry}
        progress.log(f"{model}: loaded with ctx {current_context} < {context_length}; reloading")
        unload_model(model, progress)
        reloaded = True
    if not Path(LMS_CLI).exists():
        raise RuntimeError(f"{model!r} is not loaded and the lms CLI was not found at {LMS_CLI}")
    command = [LMS_CLI, "load", model, "-c", str(context_length), "-y"]
    if ttl is not None:
        command.extend(["--ttl", str(ttl)])
    progress.log(f"{model}: loading with context {context_length} ({'ttl ' + str(ttl) + 's' if ttl else 'pinned'})…")
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=1800)
    load_seconds = round(time.monotonic() - started, 1)
    if result.returncode != 0:
        raise RuntimeError(f"lms load {model} failed: {result.stderr.strip() or result.stdout.strip()}")
    entry = loaded_models(lm_studio_url).get(model, {})
    progress.log(f"{model}: loaded in {load_seconds}s (ctx {entry.get('loaded_context_length')})")
    return {"loaded_by_script": True, "reloaded": reloaded, "load_seconds": load_seconds, "state": entry}


def unload_model(model: str, progress: Progress) -> None:
    """Unload a model this script loaded.

    :param model: LM Studio model id.
    :param progress: Progress logger.
    """
    result = subprocess.run([LMS_CLI, "unload", model], capture_output=True, text=True, check=False, timeout=300)
    progress.log(f"{model}: unload {'ok' if result.returncode == 0 else 'failed: ' + result.stderr.strip()}")


def slug(value: str) -> str:
    """Make a filesystem-safe slug.

    :param value: Arbitrary string (model id, dataset id).
    :returns: Slug.
    :rtype: str
    """
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def run_eval(
    project_root: Path,
    dataset: Path,
    model: str,
    args: argparse.Namespace,
    eval_api_key: str,
    output_path: Path,
    progress: Progress,
) -> int:
    """Run the eval runner for one dataset × model, streaming its output to the log.

    :param project_root: Repository root.
    :param dataset: Dataset path.
    :param model: Synthesis model id.
    :param args: Parsed options.
    :param eval_api_key: Key for the evaluation endpoint.
    :param output_path: JSONL output path for this run.
    :param progress: Progress logger.
    :returns: Runner exit code.
    :rtype: int
    """
    command = [
        sys.executable, str(project_root / "scripts" / "run_agentic_rag_eval.py"),
        "--dataset", str(dataset),
        "--api-base-url", f"http://127.0.0.1:{args.port}",
        "--synthesis-model", model,
        "--output", str(output_path),
        "--timeout", str(args.case_timeout),
        "--judge-base-url", args.judge_base_url,
    ]
    if args.no_judge:
        command.append("--no-judge")
    else:
        command.extend(["--judge-model", args.judge_model, "--judge-provider", args.judge_provider])
    if not args.no_wait_for_idle:
        command.extend([
            "--wait-for-idle",
            "--idle-quiet-seconds", str(args.idle_quiet_seconds),
            "--idle-max-wait-seconds", str(args.idle_max_wait_seconds),
        ])
    for case_id in args.case_ids or []:
        command.extend(["--case", case_id])
    env = dict(os.environ)
    env["EVAL_API_KEY"] = eval_api_key
    env["PYTHONPATH"] = str(project_root)
    process = subprocess.Popen(
        command, cwd=project_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line and "SecurityWarning" not in line and "_transport = transport_class" not in line:
            progress.log(f"  [{slug(model)}/{dataset.stem}] {line}")
    return process.wait()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON Lines file.

    :param path: File path.
    :returns: Parsed records.
    :rtype: List[Dict[str, Any]]
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-null values, rounded, or ``None``.

    :param values: Numbers or ``None``.
    :returns: Rounded mean.
    :rtype: Optional[float]
    """
    numbers = [float(v) for v in values if v is not None]
    return round(statistics.fmean(numbers), 3) if numbers else None


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    """Median of the non-null values, rounded, or ``None``.

    :param values: Numbers or ``None``.
    :returns: Rounded median.
    :rtype: Optional[float]
    """
    numbers = [float(v) for v in values if v is not None]
    return round(statistics.median(numbers), 3) if numbers else None


def aggregate_run(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one dataset × model run.

    :param records: JSONL records from the eval runner.
    :returns: Aggregate statistics.
    :rtype: Dict[str, Any]
    """
    ok = [r for r in records if not r.get("error")]
    metrics = [r.get("metrics") or {} for r in ok]
    judged = [r["judge"] for r in ok if r.get("judge")]
    dimensions: Dict[str, List[float]] = {}
    for judge in judged:
        for name, score in (judge.get("scores") or {}).items():
            dimensions.setdefault(name, []).append(float(score))
    critical = sum(len(judge.get("critical_failures") or []) for judge in judged)
    return {
        "cases": len(records),
        "errors": len(records) - len(ok),
        "judge_errors": sum(1 for r in ok if r.get("judge_error")),
        "deterministic_pass_rate": mean(1.0 if (r.get("deterministic") or {}).get("overall_pass") else 0.0 for r in ok),
        "judged_cases": len(judged),
        "judge_pass_rate": mean(1.0 if j.get("computed_overall_pass") else 0.0 for j in judged),
        "judge_score_mean": mean(j.get("score_mean") for j in judged),
        "judge_dimension_means": {name: mean(scores) for name, scores in sorted(dimensions.items())},
        "critical_failures_total": critical,
        "elapsed_seconds_mean": mean(m.get("elapsed_seconds") for m in metrics),
        "elapsed_seconds_median": median(m.get("elapsed_seconds") for m in metrics),
        "elapsed_seconds_max": max((m.get("elapsed_seconds") or 0.0) for m in metrics) if metrics else None,
        "synthesis_seconds_mean": mean(m.get("synthesis_seconds") for m in metrics),
        "synthesis_completion_tokens_mean": mean(m.get("synthesis_completion_tokens") for m in metrics),
        "synthesis_reasoning_tokens_mean": mean(m.get("synthesis_reasoning_tokens") for m in metrics),
        "synthesis_tokens_per_second_mean": mean(m.get("synthesis_tokens_per_second") for m in metrics),
        "verification_seconds_mean": mean(m.get("verification_seconds") for m in metrics),
        "verification_cycles_mean": mean(m.get("verification_cycles") for m in metrics),
        "verification_cycles_max": max((m.get("verification_cycles") or 0) for m in metrics) if metrics else None,
        "repair_attempts_mean": mean(m.get("repair_attempts") for m in metrics),
        "repair_attempts_total": sum((m.get("repair_attempts") or 0) for m in metrics),
        "first_pass_verification_rate": mean(
            1.0 if (m.get("verification_cycles") or 0) == 1 else 0.0 for m in metrics
        ),
        "flagged_claims_mean": mean(m.get("flagged_claims") for m in metrics),
        "verified_claims_mean": mean(m.get("verified_claims") for m in metrics),
        "answer_chars_mean": mean(m.get("answer_chars") for m in metrics),
        "response_success_rate": mean(1.0 if m.get("success") else 0.0 for m in metrics),
    }


def format_number(value: Any) -> str:
    """Format a number for the markdown table.

    :param value: Number or ``None``.
    :returns: Display string.
    :rtype: str
    """
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_summary(output_dir: Path, runs: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Write ``summary.json`` and ``summary.md``.

    :param output_dir: Directory holding the run JSONL files.
    :param runs: One entry per dataset × model with ``path``, ``dataset``, ``model``.
    :param meta: Run metadata (models, load info, timings).
    """
    per_run: List[Dict[str, Any]] = []
    per_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run in runs:
        records = read_jsonl(Path(run["path"]))
        per_run.append({**run, "aggregate": aggregate_run(records)})
        for record in records:
            key = f"{run['dataset_id']}::{record['case_id']}"
            metrics = record.get("metrics") or {}
            judge = record.get("judge") or {}
            per_case.setdefault(key, {})[run["model"]] = {
                "error": (record.get("error") or {}).get("message"),
                "deterministic_pass": (record.get("deterministic") or {}).get("overall_pass"),
                "judge_score_mean": judge.get("score_mean"),
                "judge_pass": judge.get("computed_overall_pass"),
                "critical_failures": judge.get("critical_failures"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "synthesis_seconds": metrics.get("synthesis_seconds"),
                "synthesis_tokens_per_second": metrics.get("synthesis_tokens_per_second"),
                "verification_cycles": metrics.get("verification_cycles"),
                "repair_attempts": metrics.get("repair_attempts"),
                "flagged_claims": metrics.get("flagged_claims"),
                "answer_chars": metrics.get("answer_chars"),
            }
    summary = {"meta": meta, "runs": per_run, "cases": per_case}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = ["# Synthesis model comparison", ""]
    lines.append(f"Generated {meta.get('finished_at')} · models: {', '.join(meta['models'])} · "
                 f"judge: {meta.get('judge_model') or 'none'} · run order: model-major")
    lines.append("")
    for info in meta.get("model_load", []):
        lines.append(f"- `{info['model']}`: {'loaded by script in ' + str(info['load_seconds']) + 's' if info.get('loaded_by_script') else 'already resident'}"
                     f" (ctx {((info.get('state') or {}).get('loaded_context_length'))})")
    lines.append("")
    columns = [
        ("cases", "Cases"), ("errors", "Errors"), ("deterministic_pass_rate", "Det. pass"),
        ("judge_pass_rate", "Judge pass"), ("judge_score_mean", "Judge mean"),
        ("critical_failures_total", "Critical"), ("elapsed_seconds_mean", "Latency mean s"),
        ("elapsed_seconds_median", "Latency median s"), ("synthesis_seconds_mean", "Synth s"),
        ("synthesis_tokens_per_second_mean", "Synth tok/s"), ("synthesis_completion_tokens_mean", "Synth tokens"),
        ("synthesis_reasoning_tokens_mean", "Synth reasoning tokens"),
        ("verification_cycles_mean", "Verify cycles"), ("first_pass_verification_rate", "1st-pass verify"),
        ("repair_attempts_total", "Repairs"), ("flagged_claims_mean", "Flagged"), ("answer_chars_mean", "Answer chars"),
    ]
    lines.append("## Per dataset × model")
    lines.append("")
    lines.append("| Dataset | Model | " + " | ".join(label for _, label in columns) + " |")
    lines.append("|" + "---|" * (len(columns) + 2))
    for run in per_run:
        agg = run["aggregate"]
        label = f"`{run['model']}`" + (" (baseline, earlier run)" if run.get("baseline") else "")
        lines.append(f"| {run['dataset_id']} | {label} | "
                     + " | ".join(format_number(agg.get(key)) for key, _ in columns) + " |")
    lines.append("")
    dims = sorted({d for run in per_run for d in run["aggregate"]["judge_dimension_means"]})
    if dims:
        lines.append("## Judge dimensions (mean 0–4)")
        lines.append("")
        lines.append("| Dataset | Model | " + " | ".join(dims) + " |")
        lines.append("|" + "---|" * (len(dims) + 2))
        for run in per_run:
            means = run["aggregate"]["judge_dimension_means"]
            lines.append(f"| {run['dataset_id']} | `{run['model']}` | "
                         + " | ".join(format_number(means.get(d)) for d in dims) + " |")
        lines.append("")
    lines.append("## Per case")
    lines.append("")
    models = meta["models"]
    lines.append("| Case | " + " | ".join(f"{m} (det/judge/latency s/verify cycles/repairs)" for m in models) + " |")
    lines.append("|" + "---|" * (len(models) + 1))
    for key in sorted(per_case):
        cells = []
        for model in models:
            entry = per_case[key].get(model)
            if not entry:
                cells.append("—")
            elif entry.get("error"):
                cells.append(f"ERROR: {str(entry['error'])[:60]}")
            else:
                cells.append(
                    f"{'✓' if entry['deterministic_pass'] else '✗'} / "
                    f"{format_number(entry['judge_score_mean'])} / "
                    f"{format_number(entry['elapsed_seconds'])} / "
                    f"{format_number(entry['verification_cycles'])} / {format_number(entry['repair_attempts'])}"
                )
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Legend: det = deterministic routing/retrieval/resolution checks; judge = mean judge score (0–4); "
                 "verify cycles = verify_claims node executions (1 = passed first time); repairs = repair_answer executions.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_existing(output_dir: Path) -> None:
    """Rebuild the summary of a finished run from its ``summary.json`` and JSONL files.

    :param output_dir: Directory of an earlier comparison run.
    :raises RuntimeError: If no ``summary.json`` is present to describe the runs.
    """
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"{summary_path} not found; cannot rebuild the summary")
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = [{k: run[k] for k in ("dataset_id", "dataset", "model", "path", "exit_code", "seconds", "baseline") if k in run}
            for run in previous.get("runs", [])]
    meta = dict(previous.get("meta") or {})
    meta["resummarized_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(output_dir, runs, meta)
    print(f"Rebuilt {output_dir / 'summary.md'}")


def gate_on_idle(args: argparse.Namespace, progress: Progress, reason: str) -> None:
    """Wait for LM Studio to be idle unless the gate is disabled.

    :param args: Parsed options.
    :param progress: Progress logger.
    :param reason: What is about to happen (for the log).
    """
    if args.no_wait_for_idle:
        return
    busy = lm_studio_busy_models()
    if busy:
        progress.log(f"Before {reason}: LM Studio busy ({'; '.join(busy)}); waiting for idle")
    wait_for_lm_studio_idle(
        quiet_seconds=args.idle_quiet_seconds,
        max_wait_seconds=args.idle_max_wait_seconds,
        log=progress.log,
    )


def main() -> None:
    """Run the comparison end to end."""
    args = parse_args()
    if args.summarize_only:
        summarize_existing(Path(args.summarize_only))
        return
    if not args.models:
        raise SystemExit("--models is required unless --summarize-only is given")
    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "evals" / "results" / f"model_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress(output_dir / "progress.log")
    progress.log(f"Output directory: {output_dir}")
    progress.log(f"Models: {args.models}; datasets: {args.datasets}; judge: {'none' if args.no_judge else args.judge_model}")

    gate_on_idle(args, progress, "starting the comparison")
    eval_api_key = secrets.token_hex(24)
    backend_log = output_dir / "backend.log"
    base_url = f"http://127.0.0.1:{args.port}"
    progress.log(f"Starting private backend on {base_url} (log: {backend_log})")
    backend = start_backend(project_root, args.port, eval_api_key, backend_log, args.request_timeout)
    meta: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "datasets": args.datasets,
        "judge_model": None if args.no_judge else args.judge_model,
        "context_length_for_loads": args.context_length,
        "model_load": [],
        "run_seconds": {},
    }
    runs: List[Dict[str, Any]] = []
    loaded_here: List[str] = []
    baseline_models: List[str] = []
    for baseline_path in args.baseline_results:
        baseline_rows = read_jsonl(Path(baseline_path))
        if not baseline_rows:
            progress.log(f"Baseline {baseline_path} is empty; skipping")
            continue
        baseline_model = str(baseline_rows[0].get("synthesis_model") or "")
        baseline_dataset = str(baseline_rows[0].get("dataset_id") or Path(baseline_path).stem)
        if baseline_model not in baseline_models:
            baseline_models.append(baseline_model)
        runs.append({"dataset_id": baseline_dataset, "dataset": None, "model": baseline_model,
                     "path": str(Path(baseline_path).resolve()), "exit_code": None, "seconds": None,
                     "baseline": True})
        progress.log(f"Baseline included: {baseline_dataset} × {baseline_model} ({Path(baseline_path).name})")
    meta["models"] = baseline_models + [m for m in args.models if m not in baseline_models]
    meta["baseline_models"] = baseline_models
    candidate_ttl: Optional[int] = args.model_ttl if args.model_ttl > 0 else None
    try:
        wait_for_health(base_url)
        progress.log("Backend healthy")
        if not args.no_judge and args.judge_provider == "lmstudio":
            judge_load = ensure_model_loaded(
                args.judge_model, args.lm_studio_url, args.judge_context_length, None, progress,
            )
            meta["judge_model_load"] = {"model": args.judge_model, **judge_load}
        for model in args.models:
            gate_on_idle(args, progress, f"loading/running {model}")
            load_info = ensure_model_loaded(model, args.lm_studio_url, args.context_length, candidate_ttl, progress)
            meta["model_load"].append({"model": model, **load_info})
            if load_info["loaded_by_script"]:
                loaded_here.append(model)
            for dataset_arg in args.datasets:
                dataset = Path(dataset_arg)
                dataset_id = json.loads(dataset.read_text(encoding="utf-8")).get("dataset_id", dataset.stem)
                output_path = output_dir / f"{slug(dataset_id)}__{slug(model)}.jsonl"
                progress.log(f"=== {dataset_id} × {model} → {output_path.name}")
                started = time.monotonic()
                code = run_eval(project_root, dataset, model, args, eval_api_key, output_path, progress)
                seconds = round(time.monotonic() - started, 1)
                meta["run_seconds"][f"{dataset_id}::{model}"] = seconds
                progress.log(f"=== done {dataset_id} × {model}: exit {code}, {seconds}s")
                runs.append({"dataset_id": dataset_id, "dataset": str(dataset), "model": model,
                             "path": str(output_path), "exit_code": code, "seconds": seconds})
                write_summary(output_dir, runs, {**meta, "finished_at": datetime.now(timezone.utc).isoformat()})
    finally:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        if runs:
            write_summary(output_dir, runs, meta)
        progress.log("Stopping private backend")
        backend.terminate()
        try:
            backend.wait(timeout=30)
        except subprocess.TimeoutExpired:
            backend.kill()
        if not args.keep_models_loaded:
            for model in loaded_here:
                unload_model(model, progress)
        progress.log(f"Finished. Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
