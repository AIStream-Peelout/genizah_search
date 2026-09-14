"""Test whether a judge model can be trusted, using human-established gold verdicts.

Loads ``evals/judge_acceptance_v1.json`` — real answers whose correct grade a
human established (including failures the local 4B judge rubber-stamped) — runs
a candidate judge over them, and checks each judge result against the expected
score bands. A judge that does not fail the broken items, or that gives them
perfect scores, is reported as UNTRUSTWORTHY.

Run it against any judge before trusting that judge's comparison numbers::

    PYTHONPATH=. .venv/bin/python scripts/judge_acceptance.py \\
        --judge-provider anthropic --judge-model claude-opus-5
    PYTHONPATH=. .venv/bin/python scripts/judge_acceptance.py \\
        --judge-provider lmstudio --judge-model qwen/qwen3-4b-2507
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from scripts.run_agentic_rag_eval import evaluate_deterministically, judge_case, load_json


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :returns: Parsed namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="evals/judge_acceptance_v1.json")
    parser.add_argument("--judge-prompt", default="evals/judge_prompt.md")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-provider", choices=["lmstudio", "anthropic"], default="anthropic")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", help="Optional JSON report path.")
    return parser.parse_args()


def check_expectations(scores: Dict[str, Any], computed_pass: bool, expect: Dict[str, Any]) -> List[str]:
    """Return the list of violated expectations for one gold case.

    :param scores: The judge's per-dimension scores.
    :param computed_pass: The independently computed overall pass.
    :param expect: The gold case's ``expect`` block.
    :returns: Human-readable violation strings; empty when the judge agreed.
    :rtype: List[str]
    """
    violations: List[str] = []
    if "overall_pass" in expect and computed_pass != expect["overall_pass"]:
        violations.append(f"overall_pass={computed_pass}, expected {expect['overall_pass']}")
    for dim, ceiling in (expect.get("dimension_max") or {}).items():
        if scores.get(dim, 99) > ceiling:
            violations.append(f"{dim}={scores.get(dim)} exceeds max {ceiling}")
    for dim, floor in (expect.get("dimension_min") or {}).items():
        if scores.get(dim, -1) < floor:
            violations.append(f"{dim}={scores.get(dim)} below min {floor}")
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    if "mean_max" in expect and mean > expect["mean_max"]:
        violations.append(f"mean={mean:.2f} exceeds max {expect['mean_max']}")
    if "mean_min" in expect and mean < expect["mean_min"]:
        violations.append(f"mean={mean:.2f} below min {expect['mean_min']}")
    return violations


async def evaluate_gold_case(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    gold: Dict[str, Any],
    gold_case: Dict[str, Any],
    judge_instructions: str,
) -> Dict[str, Any]:
    """Judge one gold case and compare against its expected bands.

    :param client: Shared HTTP client.
    :param args: Parsed options.
    :param gold: The gold dataset (for judge config and rubric).
    :param gold_case: One gold case with ``case``, ``response``, ``expect``.
    :param judge_instructions: Judge system prompt.
    :returns: Per-case result with the judge scores and any violations.
    :rtype: Dict[str, Any]
    """
    case = gold_case["case"]
    response = gold_case["response"]
    deterministic = evaluate_deterministically(case, response)
    dataset = {"global_answer_rubric": gold["global_answer_rubric"], "judge": gold["judge"]}
    try:
        judged = await judge_case(
            client, args.judge_base_url, args.judge_model, judge_instructions,
            dataset, case, response, deterministic, judge_provider=args.judge_provider,
        )
    except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError) as error:
        return {"id": gold_case["id"], "error": f"{type(error).__name__}: {error}",
                "trustworthy": False, "violations": ["judge call failed"]}
    scores = judged.get("scores") or {}
    violations = check_expectations(scores, judged.get("computed_overall_pass", False), gold_case["expect"])
    return {
        "id": gold_case["id"],
        "human_verdict": gold_case["human_verdict"],
        "judge_scores": scores,
        "judge_mean": judged.get("score_mean"),
        "judge_computed_pass": judged.get("computed_overall_pass"),
        "judge_summary": judged.get("summary"),
        "violations": violations,
        "trustworthy": not violations,
    }


async def run(args: argparse.Namespace) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run the acceptance test against every gold case.

    :param args: Parsed options.
    :returns: Overall trust verdict and per-case results.
    :rtype: Tuple[bool, List[Dict[str, Any]]]
    """
    gold = load_json(Path(args.gold))
    judge_instructions = Path(args.judge_prompt).read_text(encoding="utf-8")
    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for gold_case in gold["gold_cases"]:
            results.append(await evaluate_gold_case(client, args, gold, gold_case, judge_instructions))
    trustworthy = all(r["trustworthy"] for r in results)
    return trustworthy, results


def print_report(args: argparse.Namespace, trustworthy: bool, results: List[Dict[str, Any]]) -> None:
    """Print a per-case agreement report and the overall verdict.

    :param args: Parsed options.
    :param trustworthy: Overall verdict.
    :param results: Per-case results.
    """
    print(f"\nJudge acceptance: {args.judge_provider}:{args.judge_model}\n" + "=" * 60)
    for r in results:
        mark = "✓ TRUST" if r["trustworthy"] else "✗ FAIL "
        print(f"{mark}  {r['id']}  (human: {r.get('human_verdict')})")
        if r.get("error"):
            print(f"         error: {r['error']}")
            continue
        print(f"         scores: {r['judge_scores']}  mean={r['judge_mean']}  pass={r['judge_computed_pass']}")
        for violation in r["violations"]:
            print(f"         VIOLATION: {violation}")
    print("=" * 60)
    passed = sum(1 for r in results if r["trustworthy"])
    verdict = "TRUSTWORTHY" if trustworthy else "NOT TRUSTWORTHY — do not use this judge for decision-grade scoring"
    print(f"{passed}/{len(results)} gold cases agreed → {verdict}\n")


def main() -> None:
    """Run the acceptance test and exit nonzero when the judge is untrustworthy."""
    args = parse_args()
    trustworthy, results = asyncio.run(run(args))
    print_report(args, trustworthy, results)
    if args.output:
        Path(args.output).write_text(
            json.dumps({"judge_provider": args.judge_provider, "judge_model": args.judge_model,
                        "trustworthy": trustworthy, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    raise SystemExit(0 if trustworthy else 1)


if __name__ == "__main__":
    main()
