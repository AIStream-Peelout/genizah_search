"""Run deterministic and LLM-judged evaluations for the agentic RAG service."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import httpx


LMS_CLI = os.path.expanduser("~/.lmstudio/bin/lms")


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
        "--judge-provider", choices=["lmstudio", "anthropic"], default="lmstudio",
        help="'lmstudio' posts to the OpenAI-compatible --judge-base-url; 'anthropic' calls the "
             "Claude API with the anthropic SDK (reads ANTHROPIC_API_KEY; e.g. --judge-model claude-opus-5).",
    )
    parser.add_argument(
        "--synthesis-model",
        help="Downloaded LM Studio model for the private evaluation endpoint.",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--output")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--wait-for-idle", action="store_true",
        help="Before each case, wait until no LM Studio model is generating/queued "
             "(e.g. another job such as a checkpoint audit), so the eval never competes "
             "for the local inference server or its memory.",
    )
    parser.add_argument("--idle-quiet-seconds", type=float, default=30.0,
                        help="How long LM Studio must stay idle before a case starts.")
    parser.add_argument("--idle-max-wait-seconds", type=float, default=12 * 3600,
                        help="Give up waiting for idle after this long.")
    return parser.parse_args()


def busy_models_from_entries(entries: List[Dict[str, Any]]) -> List[str]:
    """Describe the resident LM Studio models that are not idle.

    :param entries: Parsed ``lms ps --json`` output.
    :returns: Human-readable descriptions of busy models (empty when idle).
    :rtype: List[str]
    """
    busy: List[str] = []
    for entry in entries:
        status = str(entry.get("status") or "idle").lower()
        queued = int(entry.get("queued") or 0)
        if status != "idle" or queued > 0:
            busy.append(f"{entry.get('identifier') or entry.get('modelKey')} ({status}, queued {queued})")
    return busy


def lm_studio_busy_models(lms_cli: str = LMS_CLI) -> List[str]:
    """Ask the ``lms`` CLI which resident models are busy right now.

    :param lms_cli: Path to the ``lms`` executable.
    :returns: Busy-model descriptions; empty when LM Studio is idle.
    :rtype: List[str]
    :raises RuntimeError: If the CLI is unavailable or fails.
    """
    import subprocess

    result = subprocess.run([lms_cli, "ps", "--json"], capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"lms ps failed: {result.stderr.strip() or result.stdout.strip()}")
    entries = json.loads(result.stdout or "[]")
    return busy_models_from_entries(entries if isinstance(entries, list) else [])


def wait_for_lm_studio_idle(
    quiet_seconds: float = 30.0,
    max_wait_seconds: float = 12 * 3600,
    poll_seconds: float = 15.0,
    check: Any = lm_studio_busy_models,
    log: Any = print,
    sleep: Any = time.sleep,
    clock: Any = time.monotonic,
) -> float:
    """Block until LM Studio has been idle for ``quiet_seconds``.

    The local inference server is shared with production chat and with batch
    jobs (checkpoint audits); starting a model-heavy eval while another job is
    generating doubles memory pressure and invalidates timings. Busy status is
    re-checked every ``poll_seconds``; the wait is logged once per minute.

    :param quiet_seconds: Required consecutive idle time before returning.
    :param max_wait_seconds: Give up after this long.
    :param poll_seconds: Polling interval.
    :param check: Callable returning busy-model descriptions (injectable for tests).
    :param log: Logging callable.
    :param sleep: Sleep callable (injectable for tests).
    :param clock: Monotonic clock callable (injectable for tests).
    :returns: Seconds spent waiting.
    :rtype: float
    :raises TimeoutError: If LM Studio did not become idle in time.
    """
    started = clock()
    idle_since: Optional[float] = None
    last_log = -60.0
    while True:
        now = clock()
        busy = check()
        if busy:
            idle_since = None
            if now - last_log >= 60.0:
                log(f"LM Studio busy ({'; '.join(busy)}); waiting {now - started:.0f}s so far")
                last_log = now
        else:
            idle_since = idle_since if idle_since is not None else now
            if now - idle_since >= quiet_seconds:
                waited = now - started
                if waited > 0:
                    log(f"LM Studio idle for {quiet_seconds:.0f}s; proceeding after {waited:.0f}s")
                return waited
        if now - started > max_wait_seconds:
            raise TimeoutError(f"LM Studio stayed busy for {max_wait_seconds:.0f}s: {'; '.join(busy)}")
        sleep(poll_seconds)


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


HEBREW_CHAR_REGEX = re.compile(r"[\u0590-\u05FF]")


def text_hebrew_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Hebrew.

    :param text: Text to measure.
    :returns: Ratio in [0, 1]; 0.0 for text with no alphabetic characters.
    :rtype: float
    """
    alphabetic = [ch for ch in text if ch.isalpha()]
    if not alphabetic:
        return 0.0
    hebrew = sum(1 for ch in alphabetic if HEBREW_CHAR_REGEX.match(ch))
    return hebrew / len(alphabetic)


def evaluate_language(case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    """Check that the answer is written in the language the case expects.

    Cases may declare ``language.expected_answer_language`` (``"he"`` or
    ``"en"``). Only the prose (text before the first appended ``---`` section)
    is measured: works-cited lists legitimately mix scripts.

    :param case: Evaluation case.
    :param response: Agentic RAG response.
    :returns: Language check results; ``applicable`` is False without a block.
    :rtype: Dict[str, Any]
    """
    block = case.get("language") or {}
    expected = block.get("expected_answer_language")
    if not expected:
        return {"applicable": False, "pass": True}
    prose = str(response.get("answer") or "").split("\n---")[0]
    ratio = text_hebrew_ratio(prose)
    if expected == "he":
        passed = ratio >= float(block.get("min_hebrew_ratio", 0.25))
    else:
        passed = ratio <= float(block.get("max_hebrew_ratio", 0.25))
    return {
        "applicable": True,
        "expected_answer_language": expected,
        "hebrew_ratio": round(ratio, 3),
        "pass": passed,
    }


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


def evaluate_resolution(case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    """Check how a multi-turn case's follow-up was interpreted.

    Cases may declare a ``resolution`` block:

    - ``must_contain_any``: the response's ``resolved_query`` must contain at
      least one of these substrings (case-insensitive) — the follow-up was
      contextualized with the right subject.
    - ``must_not_be_rewritten``: the resolved query must equal the question —
      a topic change was not polluted with earlier context.

    :param case: Evaluation case.
    :param response: Agentic RAG response.
    :returns: Resolution check results; ``applicable`` is False when the case
        declares no resolution expectations.
    :rtype: Dict[str, Any]
    """
    expectations = case.get("resolution") or {}
    resolved = normalize_text(response.get("resolved_query") or case.get("question"))
    question = normalize_text(case.get("question"))
    must_contain_any = [normalize_text(term) for term in expectations.get("must_contain_any") or []]
    contains_pass = not must_contain_any or any(term in resolved for term in must_contain_any)
    unchanged_pass = not expectations.get("must_not_be_rewritten") or resolved == question
    return {
        "applicable": bool(expectations),
        "resolved_query": response.get("resolved_query"),
        "must_contain_any_pass": contains_pass,
        "must_not_be_rewritten_pass": unchanged_pass,
        "pass": contains_pass and unchanged_pass,
    }


def evaluate_deterministically(
    case: Dict[str, Any],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate routing, retrieval, and follow-up resolution without an LLM.

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
    resolution = evaluate_resolution(case, response)
    language = evaluate_language(case, response)
    prose = str(response.get("answer") or "").split("\n---")[0].strip()
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
        "resolution": resolution,
        "resolution_pass": resolution["pass"],
        "language": language,
        "language_pass": language["pass"],
        # An answer whose prose is (near-)empty is broken regardless of what
        # the appendices contain — observed when a thinking model exhausts its
        # output budget before writing the answer.
        "answer_prose_chars": len(prose),
        "answer_has_prose_pass": len(prose) >= 40,
    }
    checks["overall_pass"] = all([
        checks["routing_required_any_pass"],
        checks["routing_primary_not_forbidden_pass"],
        checks["retrieval_must_find_any_pass"],
        checks["resolution_pass"],
        checks["language_pass"],
        checks["answer_has_prose_pass"],
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
        "graph": [bounded_graph_result(result) for result in response.get("graph_results", [])[:5]],
    }


def bounded_graph_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a scholar graph neighborhood to a judge-sized view.

    A prolific scholar's raw neighborhood (100+ works, each with fragment
    samples) serializes to hundreds of thousands of characters and overflows
    the judge's context. Counts are kept exact; lists are truncated.

    :param result: Graph result dictionary from the RAG response.
    :returns: Compact dictionary with counts and truncated samples.
    :rtype: Dict[str, Any]
    """
    if not isinstance(result, dict):
        return {"value": str(result)[:500]}
    works = result.get("works") or []
    samples = result.get("studied_fragment_samples") or []
    relationships = result.get("relationships") or []
    return {
        "scholar": result.get("scholar"),
        "resolution": result.get("resolution"),
        "work_count": len(works),
        "works_sample": [
            {
                "title": str(work.get("title") or "")[:200],
                "year": work.get("year"),
                "referenced_fragment_count": work.get("referenced_fragment_count"),
            }
            for work in works[:15]
            if isinstance(work, dict)
        ],
        "studied_fragment_count": result.get("studied_fragment_count"),
        "studied_fragment_samples": [
            (sample.get("shelfmark") if isinstance(sample, dict) else str(sample))
            for sample in samples[:10]
        ],
        "relationship_count": len(relationships),
        "relationships_sample": [str(rel)[:200] for rel in relationships[:10]],
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
    judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate judge scores and compute the pass rule deterministically.

    The scoring scale and pass thresholds come from the dataset's ``judge``
    block (``scale: {min, max}`` and ``pass_thresholds: {min_dimension,
    min_mean}``). Without a config the historical defaults apply: integers
    0-4, every dimension >= 2, mean >= 3.0.

    :param result: Parsed judge response.
    :param dimensions: Required scoring dimension names.
    :param deterministic: Routing and retrieval check results.
    :param judge_config: The dataset's ``judge`` block, if available.
    :returns: Judge result annotated with independently computed pass metadata.
    :rtype: Dict[str, Any]
    :raises ValueError: If required judge fields are missing or invalid.
    """
    scale = (judge_config or {}).get("scale") or {}
    score_min = int(scale.get("min", 0))
    score_max = int(scale.get("max", 4))
    thresholds = (judge_config or {}).get("pass_thresholds") or {}
    min_dimension = thresholds.get("min_dimension", 2 if score_max == 4 else score_max / 2)
    min_mean = thresholds.get("min_mean", 3.0 if score_max == 4 else 0.75 * score_max)

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
        if type(score) is not int or not score_min <= score <= score_max:
            raise ValueError(
                f"Judge score {name!r} must be an integer from {score_min} through {score_max}"
            )

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
        all(score >= min_dimension for score in relevant_scores),
        score_mean >= min_mean,
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

    Multi-turn cases carry a fixed ``conversation_history`` (``role`` /
    ``content`` turns, the same shape the chat UI sends) so the follow-up in
    ``question`` is evaluated against a known prior exchange; single-turn
    cases send no history.

    :param client: Shared HTTP client.
    :param api_base_url: Backend base URL.
    :param case: Evaluation case.
    :param synthesis_model: Optional private evaluation-model override.
    :returns: Parsed RAG response.
    :rtype: Dict[str, Any]
    """
    body: Dict[str, Any] = {
        "message": case["question"],
        "conversation_history": case.get("conversation_history") or None,
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


def anthropic_judge_text(
    judge_model: str,
    judge_instructions: str,
    user_content: str,
    timeout: float,
) -> str:
    """Run one judge call against the Claude API and return the reply text.

    Credentials resolve from the environment (``ANTHROPIC_API_KEY``). Sampling
    parameters are intentionally not set: current Claude models reject
    ``temperature`` and run adaptive thinking by default.

    :param judge_model: Claude model id (e.g. ``claude-opus-5``).
    :param judge_instructions: Judge system prompt.
    :param user_content: Serialized judge input JSON.
    :param timeout: Request timeout in seconds.
    :returns: Concatenated text blocks of the response.
    :rtype: str
    :raises RuntimeError: With actionable guidance when no credential is configured.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The anthropic SDK is not installed. Run: .venv/bin/pip install anthropic"
        ) from exc

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "No Anthropic credential found. Export a key before judging with the Claude API:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "(the key is read from the environment and never written to disk by these scripts)."
        )

    client = anthropic.Anthropic(timeout=timeout)
    message = client.messages.create(
        model=judge_model,
        max_tokens=8192,
        system=judge_instructions,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


async def judge_case(
    client: httpx.AsyncClient,
    judge_base_url: str,
    judge_model: str,
    judge_instructions: str,
    dataset: Dict[str, Any],
    case: Dict[str, Any],
    response: Dict[str, Any],
    deterministic: Dict[str, Any],
    judge_provider: str = "lmstudio",
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
    user_content = json.dumps(judge_input, ensure_ascii=False)
    if judge_provider == "anthropic":
        content = await asyncio.to_thread(
            anthropic_judge_text,
            judge_model,
            judge_instructions,
            user_content,
            float(client.timeout.read or 600.0) if client.timeout else 600.0,
        )
    else:
        payload = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": judge_instructions},
                {"role": "user", "content": user_content},
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
        judge_config=dataset.get("judge"),
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
                if args.wait_for_idle:
                    # Never start a case while another job (audit, other chat)
                    # is using LM Studio: it contends for memory and skews timings.
                    await asyncio.to_thread(
                        wait_for_lm_studio_idle,
                        args.idle_quiet_seconds,
                        args.idle_max_wait_seconds,
                        15.0,
                        lm_studio_busy_models,
                        lambda message: print(f"{case['id']}: {message}", flush=True),
                    )
                try:
                    started = time.monotonic()
                    rag_response = await run_chat_case(
                        client,
                        args.api_base_url,
                        case,
                        args.synthesis_model,
                    )
                    elapsed_seconds = time.monotonic() - started
                except (
                    httpx.HTTPError,
                    IndexError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    record = build_error_record(dataset, case, error)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{case['id']}: deterministic=ERROR ({type(error).__name__}: {error})")
                    continue

                if rag_response.get("error_type") == "MODEL_UNAVAILABLE":
                    # Capacity outage, not an answer: judging the apology
                    # message would poison the aggregates with zeros.
                    record = build_error_record(
                        dataset, case,
                        RuntimeError(f"MODEL_UNAVAILABLE after {elapsed_seconds:.0f}s: "
                                     f"{str(rag_response.get('answer'))[:160]}"),
                    )
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{case['id']}: deterministic=ERROR (MODEL_UNAVAILABLE after {elapsed_seconds:.0f}s)")
                    continue

                deterministic = evaluate_deterministically(case, rag_response)
                judge_result = None
                judge_error = None
                if not args.no_judge:
                    # A judge hiccup must not discard a multi-minute RAG run:
                    # retry once, then record the failure beside the response.
                    for judge_attempt in range(2):
                        try:
                            judge_result = await judge_case(
                                client,
                                args.judge_base_url,
                                args.judge_model,
                                judge_instructions,
                                dataset,
                                case,
                                rag_response,
                                deterministic,
                                judge_provider=args.judge_provider,
                            )
                            judge_error = None
                            break
                        except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError) as error:
                            judge_error = f"{type(error).__name__}: {error}"
                            print(f"{case['id']}: judge attempt {judge_attempt + 1} failed: {judge_error}")
                record = build_result_record(
                    dataset,
                    case,
                    rag_response,
                    deterministic,
                    judge_result,
                    elapsed_seconds=elapsed_seconds,
                    synthesis_model=args.synthesis_model,
                    judge_error=judge_error,
                )
                status = "PASS" if deterministic["overall_pass"] else "FAIL"
                status += f" {elapsed_seconds:.0f}s"
                if judge_result:
                    status += f" judge_mean={judge_result.get('score_mean')}"
                if case.get("conversation_history"):
                    status += f" resolved_query={rag_response.get('resolved_query')!r}"
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{case['id']}: deterministic={status}")
    return output_path


def summarize_metrics(
    response: Dict[str, Any],
    elapsed_seconds: Optional[float],
    synthesis_model: Optional[str],
) -> Dict[str, Any]:
    """Extract the per-case efficiency metrics used to compare synthesis models.

    :param response: Agentic RAG response (its ``metrics`` block is read).
    :param elapsed_seconds: Wall time of the chat request as seen by the runner.
    :param synthesis_model: Model override requested, if any.
    :returns: Flat metrics record.
    :rtype: Dict[str, Any]
    """
    metrics = response.get("metrics") or {}
    by_stage = metrics.get("llm_by_stage") or {}
    synthesis = by_stage.get("synthesize_answer") or {}
    repair = by_stage.get("repair_answer") or {}
    verify = by_stage.get("verify_claims") or {}
    generation_seconds = synthesis.get("seconds") or 0.0
    completion_tokens = synthesis.get("completion_tokens") or 0
    return {
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "pipeline_seconds": metrics.get("total_seconds"),
        "synthesis_model_requested": synthesis_model,
        "synthesis_model_used": metrics.get("synthesis_model"),
        "stage_timings": metrics.get("stage_timings") or {},
        "verification_cycles": metrics.get("verification_cycles"),
        "repair_attempts": metrics.get("repair_attempts"),
        "synthesis_seconds": generation_seconds,
        "synthesis_prompt_tokens": synthesis.get("prompt_tokens"),
        "synthesis_completion_tokens": completion_tokens,
        "synthesis_reasoning_tokens": synthesis.get("reasoning_tokens"),
        "synthesis_tokens_per_second": (
            round(completion_tokens / generation_seconds, 2) if generation_seconds else None
        ),
        "repair_seconds": repair.get("seconds"),
        "repair_completion_tokens": repair.get("completion_tokens"),
        "verification_seconds": verify.get("seconds"),
        "verification_completion_tokens": verify.get("completion_tokens"),
        "llm_calls_total": len(metrics.get("llm_calls") or []),
        "answer_chars": len(str(response.get("answer") or "")),
        "verified_claims": len(response.get("verified_claims") or []),
        "flagged_claims": len(response.get("flagged_claims") or []),
        "success": response.get("success"),
        "error_type": response.get("error_type"),
    }


def build_result_record(
    dataset: Dict[str, Any],
    case: Dict[str, Any],
    rag_response: Dict[str, Any],
    deterministic: Dict[str, Any],
    judge_result: Optional[Dict[str, Any]],
    elapsed_seconds: Optional[float] = None,
    synthesis_model: Optional[str] = None,
    judge_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one successful JSONL evaluation record.

    :param dataset: Full evaluation dataset.
    :param case: Evaluation case.
    :param rag_response: Raw agentic RAG response.
    :param deterministic: Deterministic check results.
    :param judge_result: Optional structured LLM-judge result.
    :param elapsed_seconds: Wall time of the chat request.
    :param synthesis_model: Synthesis model override used for the run, if any.
    :param judge_error: Error text when judging failed after retry.
    :returns: Serializable evaluation record.
    :rtype: Dict[str, Any]
    """
    return {
        "dataset_id": dataset["dataset_id"],
        "case_id": case["id"],
        "question": case["question"],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "synthesis_model": synthesis_model,
        "metrics": summarize_metrics(rag_response, elapsed_seconds, synthesis_model),
        "deterministic": deterministic,
        "judge": judge_result,
        "judge_error": judge_error,
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
