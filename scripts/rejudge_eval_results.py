"""Re-run the LLM judge over saved evaluation results without re-running RAG.

Each result row stores the full RAG response, so a judge that failed (context
overflow, malformed JSON) or a changed judge prompt/model can be re-applied
in place. By default only rows with a missing or failed judge are re-judged;
``--all`` re-judges every successful row.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/rejudge_eval_results.py \\
        --results evals/results/model_comparison_X/*.jsonl \\
        --judge-model qwen/qwen3-4b-2507
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from scripts.run_agentic_rag_eval import evaluate_deterministically, judge_case, load_json


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :returns: Parsed namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="Result JSONL files to update in place.")
    parser.add_argument("--datasets-dir", default="evals", help="Directory scanned for dataset JSON files.")
    parser.add_argument("--judge-prompt", default="evals/judge_prompt.md")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-provider", choices=["lmstudio", "anthropic"], default="lmstudio",
                        help="'anthropic' judges via the Claude API (reads ANTHROPIC_API_KEY).")
    parser.add_argument("--all", action="store_true", help="Re-judge every row, not only failed/missing judges.")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def load_datasets(datasets_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Index every dataset file in a directory by its ``dataset_id``.

    :param datasets_dir: Directory containing dataset JSON files.
    :returns: Datasets keyed by id.
    :rtype: Dict[str, Dict[str, Any]]
    """
    datasets: Dict[str, Dict[str, Any]] = {}
    for path in sorted(datasets_dir.glob("*.json")):
        try:
            dataset = load_json(path)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(dataset.get("cases"), list) and dataset.get("dataset_id"):
            datasets[str(dataset["dataset_id"])] = dataset
    return datasets


def needs_judging(row: Dict[str, Any], rejudge_all: bool) -> bool:
    """Decide whether a row should be (re-)judged.

    :param row: Result row.
    :param rejudge_all: Re-judge even rows with a valid judge result.
    :returns: Whether to judge the row.
    :rtype: bool
    """
    if row.get("error") or not row.get("response"):
        return False
    return rejudge_all or not row.get("judge") or bool(row.get("judge_error"))


async def rejudge_file(
    path: Path,
    datasets: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    judge_instructions: str,
) -> Dict[str, int]:
    """Re-judge the qualifying rows of one JSONL file in place.

    :param path: Result file.
    :param datasets: Datasets keyed by id.
    :param args: Parsed options.
    :param judge_instructions: Judge system prompt.
    :returns: Counts of rows judged and rows that still failed.
    :rtype: Dict[str, int]
    """
    rows: List[Dict[str, Any]] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    judged = failed = 0
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for row in rows:
            if not needs_judging(row, args.all):
                continue
            dataset = datasets.get(str(row.get("dataset_id")))
            if dataset is None:
                print(f"{path.name}: no dataset with id {row.get('dataset_id')!r}; skipping {row.get('case_id')}")
                continue
            case = next((c for c in dataset["cases"] if c.get("id") == row.get("case_id")), None)
            if case is None:
                print(f"{path.name}: case {row.get('case_id')!r} not in dataset; skipping")
                continue
            deterministic = row.get("deterministic") or evaluate_deterministically(case, row["response"])
            last_error: Optional[str] = None
            for attempt in range(2):
                try:
                    row["judge"] = await judge_case(
                        client, args.judge_base_url, args.judge_model, judge_instructions,
                        dataset, case, row["response"], deterministic,
                        judge_provider=args.judge_provider,
                    )
                    row["judge_error"] = None
                    row["rejudged_at"] = datetime.now(timezone.utc).isoformat()
                    last_error = None
                    break
                except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError) as error:
                    last_error = f"{type(error).__name__}: {error}"
            judged += 1
            if last_error:
                failed += 1
                row["judge_error"] = last_error
                print(f"{path.name}: {row['case_id']}: judge still failing: {last_error}")
            else:
                print(f"{path.name}: {row['case_id']}: judge mean {row['judge'].get('score_mean')} "
                      f"pass={row['judge'].get('computed_overall_pass')}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)
    return {"judged": judged, "failed": failed}


async def run(args: argparse.Namespace) -> None:
    """Re-judge every requested file.

    :param args: Parsed options.
    """
    datasets = load_datasets(Path(args.datasets_dir))
    judge_instructions = Path(args.judge_prompt).read_text(encoding="utf-8")
    for result_path in args.results:
        counts = await rejudge_file(Path(result_path), datasets, args, judge_instructions)
        print(f"{result_path}: re-judged {counts['judged']} rows, {counts['failed']} still failing")


def main() -> None:
    """Command-line entry point."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
