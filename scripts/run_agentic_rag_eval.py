"""Run deterministic and LLM-judged evaluations for the agentic RAG service."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import httpx


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :returns: Parsed command-line namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/agentic_rag_v1.json")
    parser.add_argument("--judge-prompt", default="evals/judge_prompt.md")
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--judge-base-url", default="http://localhost:1234")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--synthesis-model",
        help="Downloaded LM Studio model for the private evaluation endpoint.",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--output")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk.

    :param path: JSON file to read.
    :returns: Parsed JSON object.
    :rtype: Dict[str, Any]
    """
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def select_cases(
    cases: Iterable[Dict[str, Any]],
    case_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Select requested evaluation cases while preserving dataset order.

    :param cases: Dataset cases.
    :param case_ids: Optional repeated case identifiers.
    :returns: Selected cases.
    :rtype: List[Dict[str, Any]]
    """
    selected = list(cases)
    if not case_ids:
        return selected
    requested = set(case_ids)
    available = {str(case.get("id")) for case in selected}
    missing = requested - available
    if missing:
        raise ValueError(f"Unknown case ids: {', '.join(sorted(missing))}")
    return [case for case in selected if case.get("id") in requested]


def normalize_text(value: Any) -> str:
    """Normalize a value for conservative case-insensitive matching.

    :param value: Value to normalize.
    :returns: Normalized text.
    :rtype: str
    """
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def bibliography_target_matches(
    target: Dict[str, Any],
    result: Dict[str, Any],
) -> bool:
    """Return whether a bibliography result matches an oracle target.

    :param target: Retrieval target specification.
    :param result: Bibliography response result.
    :returns: Whether all supplied target constraints match.
    :rtype: bool
    """
    title = normalize_text(result.get("title"))
    authors = normalize_text(result.get("authors") or result.get("author"))
    if target.get("title_contains") and normalize_text(target["title_contains"]) not in title:
        return False
    if target.get("author_contains") and normalize_text(target["author_contains"]) not in authors:
        return False
    pages = target.get("pages_any") or []
    if pages and result.get("extracted_page_number") not in pages:
        return False
    return True


def graph_target_matches(target: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """Return whether a graph neighborhood matches an oracle target.

    :param target: Graph target specification.
    :param result: Graph result dictionary.
    :returns: Whether the canonical scholar name matches.
    :rtype: bool
    """
    expected = normalize_text(target.get("scholar_name"))
    actual = normalize_text((result.get("scholar") or {}).get("name"))
    return bool(expected and expected == actual)


def target_matches(target: Dict[str, Any], response: Dict[str, Any]) -> bool:
    """Match one target against the appropriate response evidence collection.

    :param target: Target specification.
    :param response: Agentic RAG response.
    :returns: Whether any response item matches.
    :rtype: bool
    """
    kind = target.get("kind")
    if kind == "bibliography":
        return any(
            bibliography_target_matches(target, result)
            for result in response.get("bibliography_results", [])
        )
    if kind == "graph":
        return any(
            graph_target_matches(target, result)
            for result in response.get("graph_results", [])
        )
    if kind == "primary":
        expected_doc_id = normalize_text(target.get("doc_id"))
        expected_shelf_mark = normalize_text(target.get("shelf_mark"))
        for result in response.get("primary_source_results", []):
            if expected_doc_id and expected_doc_id == normalize_text(result.get("doc_id")):
                return True
            if expected_shelf_mark and expected_shelf_mark == normalize_text(result.get("shelf_mark")):
                return True
    return False


def evaluate_deterministically(
    case: Dict[str, Any],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate routing and retrieval constraints without an LLM.

    :param case: Evaluation case.
    :param response: Agentic RAG response.
    :returns: Deterministic check results.
    :rtype: Dict[str, Any]
    """
    actions = (
        (response.get("query_plan") or {}).get("actions")
        if isinstance(response.get("query_plan"), dict)
        else []
    ) or []
    action_types = [str(action.get("search_type")) for action in actions]
    routing = case.get("routing", {})
    required_any = routing.get("must_include_any") or []
    forbidden_primary = routing.get("must_not_use_as_primary") or []
    retrieval_targets = case.get("retrieval", {}).get("must_find_any") or []
    matched_targets = [
        target for target in retrieval_targets if target_matches(target, response)
    ]
    checks = {
        "action_types": action_types,
        "routing_required_any_pass": (
            not required_any or any(action in action_types for action in required_any)
        ),
        "routing_primary_not_forbidden_pass": (
            not action_types or action_types[0] not in forbidden_primary
        ),
        "retrieval_target_applicable": bool(retrieval_targets),
        "retrieval_must_find_any_pass": (
            not retrieval_targets or bool(matched_targets)
        ),
        "matched_targets": matched_targets,
    }
    checks["overall_pass"] = all([
        checks["routing_required_any_pass"],
        checks["routing_primary_not_forbidden_pass"],
        checks["retrieval_must_find_any_pass"],
    ])
    return checks


def bounded_evidence(response: Dict[str, Any]) -> Dict[str, Any]:
    """Create a bounded evidence view for the LLM judge.

    :param response: Full RAG response.
    :returns: Evidence metadata and excerpts suitable for a judge prompt.
    :rtype: Dict[str, Any]
    """
    bibliography = []
    for result in response.get("bibliography_results", [])[:12]:
        bibliography.append({
            "doc_id": result.get("doc_id"),
            "title": result.get("title"),
            "authors": result.get("authors") or result.get("author"),
            "page": result.get("extracted_page_number"),
            "description": str(result.get("description") or "")[:500],
            "source_text": str(result.get("full_text") or "")[:1600],
        })
    primary = []
    for result in response.get("primary_source_results", [])[:12]:
        primary.append({
            "doc_id": result.get("doc_id"),
            "shelf_mark": result.get("shelf_mark"),
            "title": result.get("title"),
            "description": str(result.get("description") or "")[:600],
        })
    return {
        "bibliography": bibliography,
        "primary": primary,
        "graph": response.get("graph_results", [])[:5],
    }


def parse_json_response(content: str) -> Dict[str, Any]:
    """Parse a possibly fenced JSON response.

    :param content: Model response content.
    :returns: Parsed JSON object.
    :rtype: Dict[str, Any]
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def validate_judge_result(
    result: Dict[str, Any],
    dimensions: List[str],
    deterministic: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate judge scores and compute the pass rule deterministically.

    :param result: Parsed judge response.
    :param dimensions: Required scoring dimension names.
    :param deterministic: Routing and retrieval check results.
    :returns: Judge result annotated with independently computed pass metadata.
    :rtype: Dict[str, Any]
    :raises ValueError: If required judge fields are missing or invalid.
    """
    scores = result.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Judge response is missing the scores object")
    missing_dimensions = [name for name in dimensions if name not in scores]
    if missing_dimensions:
        raise ValueError(
            "Judge response is missing score dimensions: "
            + ", ".join(missing_dimensions)
        )
    for name in dimensions:
        score = scores[name]
        if type(score) is not int or not 0 <= score <= 4:
            raise ValueError(f"Judge score {name!r} must be an integer from 0 through 4")

    critical_failures = result.get("critical_failures")
    if not isinstance(critical_failures, list):
        raise ValueError("Judge response critical_failures must be a list")
    if type(result.get("overall_pass")) is not bool:
        raise ValueError("Judge response overall_pass must be a boolean")

    relevant_scores = [scores[name] for name in dimensions]
    score_mean = sum(relevant_scores) / len(relevant_scores)
    computed_pass = all([
        deterministic.get("overall_pass", False),
        not critical_failures,
        all(score >= 2 for score in relevant_scores),
        score_mean >= 3.0,
    ])
    annotated = dict(result)
    annotated["score_mean"] = round(score_mean, 3)
    annotated["computed_overall_pass"] = computed_pass
    annotated["reported_pass_matches_rule"] = result["overall_pass"] == computed_pass
    return annotated


async def run_chat_case(
    client: httpx.AsyncClient,
    api_base_url: str,
    case: Dict[str, Any],
    synthesis_model: Optional[str],
) -> Dict[str, Any]:
    """Execute one evaluation query against the backend.

    :param client: Shared HTTP client.
    :param api_base_url: Backend base URL.
    :param case: Evaluation case.
    :param synthesis_model: Optional private evaluation-model override.
    :returns: Parsed RAG response.
    :rtype: Dict[str, Any]
    """
    body: Dict[str, Any] = {
        "message": case["question"],
        "conversation_history": None,
    }
    if synthesis_model:
        eval_api_key = os.getenv("EVAL_API_KEY", "").strip()
        if not eval_api_key:
            raise ValueError(
                "EVAL_API_KEY is required when --synthesis-model is used"
            )
        body["model"] = synthesis_model
        endpoint = "/internal/eval/chat"
        headers = {"X-Eval-API-Key": eval_api_key}
    else:
        endpoint = "/chat"
        headers = {}
        chat_api_key = os.getenv("CHAT_API_KEY", "").strip()
        if chat_api_key:
            headers["X-API-Key"] = chat_api_key
    response = await client.post(
        f"{api_base_url.rstrip('/')}{endpoint}",
        json=body,
        headers=headers,
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Chat endpoint returned a non-object response")
    return value


async def judge_case(
    client: httpx.AsyncClient,
    judge_base_url: str,
    judge_model: str,
    judge_instructions: str,
    dataset: Dict[str, Any],
    case: Dict[str, Any],
    response: Dict[str, Any],
    deterministic: Dict[str, Any],
) -> Dict[str, Any]:
    """Judge one RAG response using the configured chat-completion model.

    :param client: Shared HTTP client.
    :param judge_base_url: OpenAI-compatible judge server base URL.
    :param judge_model: Judge model identifier.
    :param judge_instructions: Static judge prompt.
    :param dataset: Full dataset metadata and global rubric.
    :param case: Evaluation case.
    :param response: Agentic RAG response.
    :param deterministic: Deterministic check results.
    :returns: Parsed structured judge result.
    :rtype: Dict[str, Any]
    """
    judge_input = {
        "global_answer_rubric": dataset["global_answer_rubric"],
        "judge_configuration": dataset["judge"],
        "case": case,
        "candidate_answer": response.get("answer"),
        "query_plan": response.get("query_plan"),
        "retrieved_evidence": bounded_evidence(response),
        "deterministic_checks": deterministic,
    }
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": judge_instructions},
            {
                "role": "user",
                "content": json.dumps(judge_input, ensure_ascii=False),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    response_message = await client.post(
        f"{judge_base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
    )
    response_message.raise_for_status()
    content = response_message.json()["choices"][0]["message"]["content"]
    parsed = parse_json_response(content)
    return validate_judge_result(
        parsed,
        dataset["judge"]["dimensions"],
        deterministic,
    )


async def run(args: argparse.Namespace) -> Path:
    """Run selected evaluation cases and write JSONL results.

    :param args: Parsed command-line arguments.
    :returns: Output path.
    :rtype: Path
    """
    dataset_path = Path(args.dataset)
    dataset = load_json(dataset_path)
    cases = select_cases(dataset.get("cases", []), args.case_ids)
    judge_instructions = ""
    if not args.no_judge:
        judge_instructions = Path(args.judge_prompt).read_text(encoding="utf-8")
    if not args.no_judge and not args.judge_model:
        raise ValueError("--judge-model is required unless --no-judge is used")

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("evals/results") / f"{dataset['dataset_id']}_{timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        with output_path.open("w", encoding="utf-8") as output:
            for case in cases:
                try:
                    rag_response = await run_chat_case(
                        client,
                        args.api_base_url,
                        case,
                        args.synthesis_model,
                    )
                    deterministic = evaluate_deterministically(case, rag_response)
                    judge_result = None
                    if not args.no_judge:
                        judge_result = await judge_case(
                            client,
                            args.judge_base_url,
                            args.judge_model,
                            judge_instructions,
                            dataset,
                            case,
                            rag_response,
                            deterministic,
                        )
                    record = build_result_record(
                        dataset,
                        case,
                        rag_response,
                        deterministic,
                        judge_result,
                    )
                    status = "PASS" if deterministic["overall_pass"] else "FAIL"
                except (
                    httpx.HTTPError,
                    IndexError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    record = build_error_record(dataset, case, error)
                    status = "ERROR"
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{case['id']}: deterministic={status}")
    return output_path


def build_result_record(
    dataset: Dict[str, Any],
    case: Dict[str, Any],
    rag_response: Dict[str, Any],
    deterministic: Dict[str, Any],
    judge_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one successful JSONL evaluation record.

    :param dataset: Full evaluation dataset.
    :param case: Evaluation case.
    :param rag_response: Raw agentic RAG response.
    :param deterministic: Deterministic check results.
    :param judge_result: Optional structured LLM-judge result.
    :returns: Serializable evaluation record.
    :rtype: Dict[str, Any]
    """
    return {
        "dataset_id": dataset["dataset_id"],
        "case_id": case["id"],
        "question": case["question"],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "deterministic": deterministic,
        "judge": judge_result,
        "response": rag_response,
        "error": None,
    }


def build_error_record(
    dataset: Dict[str, Any],
    case: Dict[str, Any],
    error: Exception,
) -> Dict[str, Any]:
    """Build one JSONL record for a case that could not be evaluated.

    :param dataset: Full evaluation dataset.
    :param case: Evaluation case.
    :param error: Case execution or parsing error.
    :returns: Serializable error record.
    :rtype: Dict[str, Any]
    """
    return {
        "dataset_id": dataset["dataset_id"],
        "case_id": case["id"],
        "question": case["question"],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "deterministic": None,
        "judge": None,
        "response": None,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def main() -> None:
    """Run the command-line evaluation program."""
    output_path = asyncio.run(run(parse_args()))
    print(f"Wrote evaluation results to {output_path}")


if __name__ == "__main__":
    main()
