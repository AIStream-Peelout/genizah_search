"""Tests for judge spec v2 (0-10 scale), judge providers, language checks, and annotation."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.build_annotation_page import build_page, case_payload
from scripts.run_agentic_rag_eval import (
    evaluate_deterministically,
    evaluate_language,
    judge_case,
    load_json,
    text_hebrew_ratio,
    validate_judge_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HEBREW_ANSWER = (
    "החוקרים מתארים את הכתובות מן הגניזה כעדות מרכזית לדיני הנישואין. "
    "פרידמן מנתח את הנוסח הארץ-ישראלי (Friedman, Jewish Marriage in Palestine)."
    "\n\n---\n**Works cited:**\n\n- Mordechai Akiva Friedman, Jewish Marriage in Palestine"
)
ENGLISH_ANSWER = (
    "Scholars describe the Genizah ketubbot as central evidence for marriage law "
    "(Friedman, p. 7).\n\n---\n**Works cited:**\n\n- Friedman, Jewish Marriage in Palestine"
)


def v2_judge_config() -> Dict[str, Any]:
    """Return a judge config matching the v2 dataset blocks.

    :returns: Judge configuration.
    :rtype: Dict[str, Any]
    """
    return {"scale": {"min": 0, "max": 10}, "pass_thresholds": {"min_dimension": 5, "min_mean": 7.5}}


# ---------------------------------------------------------------------------
# Scale-aware validation
# ---------------------------------------------------------------------------

def test_validate_judge_result_uses_v2_scale_and_thresholds() -> None:
    """0-10 scores validate against the dataset's scale and pass thresholds."""
    dimensions = ["question_answered", "answer_flow"]
    result = {
        "scores": {"question_answered": 9, "answer_flow": 7},
        "critical_failures": [],
        "overall_pass": True,
    }

    annotated = validate_judge_result(result, dimensions, {"overall_pass": True}, v2_judge_config())

    assert annotated["score_mean"] == 8.0
    assert annotated["computed_overall_pass"]

    failing = validate_judge_result(
        {"scores": {"question_answered": 9, "answer_flow": 4}, "critical_failures": [], "overall_pass": True},
        dimensions, {"overall_pass": True}, v2_judge_config(),
    )
    assert not failing["computed_overall_pass"]  # dimension below 5

    with pytest.raises(ValueError, match="0 through 10"):
        validate_judge_result(
            {"scores": {"question_answered": 11, "answer_flow": 7}, "critical_failures": [], "overall_pass": True},
            dimensions, {"overall_pass": True}, v2_judge_config(),
        )


def test_validate_judge_result_defaults_stay_backward_compatible() -> None:
    """Without a judge config the historical 0-4 rule applies unchanged."""
    result = {"scores": {"a": 4}, "critical_failures": [], "overall_pass": True}

    annotated = validate_judge_result(result, ["a"], {"overall_pass": True})

    assert annotated["computed_overall_pass"]
    with pytest.raises(ValueError, match="0 through 4"):
        validate_judge_result(
            {"scores": {"a": 7}, "critical_failures": [], "overall_pass": True},
            ["a"], {"overall_pass": True},
        )


# ---------------------------------------------------------------------------
# Judge providers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_case_anthropic_provider_uses_sdk_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider='anthropic' routes through the SDK helper and validates with the dataset config."""
    import httpx

    import scripts.run_agentic_rag_eval as runner

    captured: Dict[str, Any] = {}

    def fake_anthropic(judge_model: str, instructions: str, user_content: str, timeout: float) -> str:
        captured["model"] = judge_model
        captured["input"] = json.loads(user_content)
        return json.dumps({
            "scores": {"question_answered": 9, "answer_flow": 8},
            "critical_failures": [],
            "rubric_findings": [],
            "unsupported_claims": [],
            "overall_pass": True,
            "summary": "ok",
        })

    monkeypatch.setattr(runner, "anthropic_judge_text", fake_anthropic)
    dataset = {
        "global_answer_rubric": {},
        "judge": {**v2_judge_config(), "dimensions": ["question_answered", "answer_flow"]},
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        result = await judge_case(
            client, "http://unused", "claude-opus-5", "instructions",
            dataset, {"id": "c", "question": "q"}, {"answer": "a"}, {"overall_pass": True},
            judge_provider="anthropic",
        )

    assert captured["model"] == "claude-opus-5"
    assert captured["input"]["candidate_answer"] == "a"
    assert result["score_mean"] == 8.5
    assert result["computed_overall_pass"]


# ---------------------------------------------------------------------------
# Language checks
# ---------------------------------------------------------------------------

def test_text_hebrew_ratio() -> None:
    """Hebrew ratio counts alphabetic characters only."""
    assert text_hebrew_ratio("שלום עולם") == 1.0
    assert text_hebrew_ratio("hello world") == 0.0
    assert text_hebrew_ratio("123 --- !!") == 0.0
    assert 0.4 < text_hebrew_ratio("שלום hello") < 0.6


def test_language_check_passes_hebrew_and_fails_english_prose() -> None:
    """A Hebrew-expected case passes on Hebrew prose and fails on English prose."""
    case = {"question": "מה?", "routing": {}, "retrieval": {}, "language": {"expected_answer_language": "he"}}

    good = evaluate_deterministically(case, {"answer": HEBREW_ANSWER})
    bad = evaluate_deterministically(case, {"answer": ENGLISH_ANSWER})

    assert good["language_pass"] and good["language"]["hebrew_ratio"] > 0.5
    assert not bad["language_pass"] and not bad["overall_pass"]


def test_language_check_english_control_and_absent_block() -> None:
    """'en' expectation fails on Hebrew prose; no language block is always applicable-false pass."""
    en_case = {"question": "q", "routing": {}, "retrieval": {}, "language": {"expected_answer_language": "en"}}

    assert evaluate_deterministically(en_case, {"answer": ENGLISH_ANSWER})["language_pass"]
    assert not evaluate_deterministically(en_case, {"answer": HEBREW_ANSWER})["language_pass"]
    no_block = evaluate_language({"id": "x"}, {"answer": HEBREW_ANSWER})
    assert no_block == {"applicable": False, "pass": True}


def test_language_check_measures_prose_not_appendices() -> None:
    """English works-cited sections do not fail a Hebrew answer."""
    case = {"question": "מה?", "routing": {}, "retrieval": {}, "language": {"expected_answer_language": "he"}}
    answer = "תשובה עברית קצרה." + "\n\n---\n**Works cited:**\n\n- " + ("English Title, " * 40)

    assert evaluate_deterministically(case, {"answer": answer})["language_pass"]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def test_all_datasets_carry_judge_v2_blocks() -> None:
    """Every dataset now declares the 0-10 scale and the seven v2 dimensions."""
    expected_dimensions = {
        "question_answered", "answer_flow", "primary_source_linking",
        "groundedness_and_accuracy", "citation_and_quote_quality",
        "retrieval_evidence_coverage", "restraint_and_limitations",
    }
    for name in ["agentic_rag_v1.json", "agentic_rag_multiturn_v1.json", "agentic_rag_crosslingual_v1.json"]:
        dataset = load_json(PROJECT_ROOT / "evals" / name)
        judge = dataset["judge"]
        assert judge["version"] == 2, name
        assert judge["scale"] == {"min": 0, "max": 10}, name
        assert set(judge["dimensions"]) == expected_dimensions, name
        assert judge["pass_thresholds"]["min_dimension"] == 5, name
        assert set(judge["dimension_definitions"]) == expected_dimensions, name


def test_crosslingual_dataset_is_well_formed() -> None:
    """Cross-lingual cases declare language expectations and retrieval targets."""
    dataset = load_json(PROJECT_ROOT / "evals" / "agentic_rag_crosslingual_v1.json")
    ids = [case["id"] for case in dataset["cases"]]

    assert len(ids) == len(set(ids)) >= 4
    languages = set()
    for case in dataset["cases"]:
        assert case["language"]["expected_answer_language"] in {"he", "en"}, case["id"]
        languages.add(case["language"]["expected_answer_language"])
        assert case["retrieval"]["must_find_any"], case["id"]
        assert case["routing"]["must_include_any"], case["id"]
        assert case["answer_rubric"]["must_include"], case["id"]
    assert languages == {"he", "en"}  # includes the English control
    assert any("conversation_history" in case for case in dataset["cases"])  # cross-lingual follow-up


# ---------------------------------------------------------------------------
# Annotation page
# ---------------------------------------------------------------------------

def test_annotation_page_groups_models_per_case() -> None:
    """The page embeds every model's answer for a case and the grading dimensions."""
    rows = [
        {"dataset_id": "d", "case_id": "c1", "question": "Q1", "synthesis_model": "model-a",
         "response": {"answer": "Answer A <script>", "metrics": {}},
         "judge": {"score_mean": 8.0, "scores": {}, "critical_failures": []},
         "deterministic": {"overall_pass": True}, "metrics": {"elapsed_seconds": 10}},
        {"dataset_id": "d", "case_id": "c1", "question": "Q1", "synthesis_model": "model-b",
         "response": {"answer": "Answer B", "metrics": {}},
         "judge": None, "deterministic": {"overall_pass": False}, "metrics": {}},
    ]

    cases = case_payload(rows)
    assert len(cases) == 1
    assert [a["model"] for a in cases[0]["answers"]] == ["model-a", "model-b"]

    page = build_page(rows, "Test page", ["question_answered"], "test_store")
    assert "Answer A" in page and "Answer B" in page
    assert "question_answered" in page
    assert "<script>Answer" not in page  # embedded JSON is escaped against tag breakout
    assert "genizah_annotations::test_store" in page
    assert "preferred_model" in page


# ---------------------------------------------------------------------------
# Empty-synthesis guards (observed: thinking model spent 8191/8191 tokens
# reasoning, emitted no answer, and every downstream layer passed it)
# ---------------------------------------------------------------------------

def test_empty_prose_answer_fails_deterministically() -> None:
    """An appendix-only answer (no prose) fails the answer_has_prose check."""
    case = {"question": "q", "routing": {}, "retrieval": {}}
    appendix_only = "\n---\n**Related catalog entries:**\n\n- **[X 1](doc:x1)**\n---\n**Works cited:**\n\n- A, B"

    broken = evaluate_deterministically(case, {"answer": appendix_only})
    fine = evaluate_deterministically(case, {"answer": "A real answer with actual prose content here." + appendix_only})

    assert not broken["answer_has_prose_pass"]
    assert broken["answer_prose_chars"] == 0
    assert not broken["overall_pass"]
    assert fine["answer_has_prose_pass"]


@pytest.mark.asyncio
async def test_synthesis_retries_once_then_fails_honestly_on_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty synthesis output triggers one retry, then an honest failure state."""
    from unittest.mock import AsyncMock
    from src.backend.lms_agentic_search import SYNTHESIS_MAX_TOKENS, AgenticRAGService

    service = AgenticRAGService()
    llm = AsyncMock(return_value="   \n")
    monkeypatch.setattr(service, "_call_llm", llm)
    state: Dict[str, Any] = {
        "user_query": "q", "conversation_history": [], "resolved_query": None,
        "is_followup": False, "bibliography_results": [], "graph_results": [],
        "synthesis_model_override": None, "excluded_claims": [], "retry_count": 0,
        "processing_steps": [], "error": None, "error_type": None,
    }

    result = await service._synthesize_answer_node(state)

    assert llm.await_count == 2  # one retry
    assert llm.call_args.kwargs["max_tokens"] == SYNTHESIS_MAX_TOKENS
    assert result["error_type"] == "NO_ANSWER_GENERATED"
    assert "could not compose an answer" in result["draft_answer"]


@pytest.mark.asyncio
async def test_verifier_refuses_to_verify_an_empty_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verify node never runs the model against an empty draft."""
    from unittest.mock import AsyncMock
    from src.backend.lms_agentic_search import AgenticRAGService

    service = AgenticRAGService()
    verifier = AsyncMock()
    monkeypatch.setattr(service, "_call_verifier", verifier)
    state: Dict[str, Any] = {
        "draft_answer": "  \n", "error_type": None, "error": None,
        "bibliography_results": [], "graph_results": [], "processing_steps": [],
        "verified_claims": [], "verification_summary": {}, "retry_count": 0,
    }

    result = await service._verify_claims_node(state)

    verifier.assert_not_awaited()
    assert result["error_type"] == "NO_ANSWER_GENERATED"


@pytest.mark.asyncio
async def test_finalize_returns_honest_failure_without_appendices() -> None:
    """NO_ANSWER_GENERATED responses carry no catalog/works-cited sections."""
    from src.backend.lms_agentic_search import AgenticRAGService

    service = AgenticRAGService()
    state: Dict[str, Any] = {
        "error_type": "NO_ANSWER_GENERATED",
        "draft_answer": "The assistant could not compose an answer to this question just now. Please try again in a moment.",
        "primary_source_results": [{"doc_id": "x", "shelf_mark": "T-S 1.1", "title": "t", "description": "d"}],
        "shelf_mark_lookup": {}, "shelf_marks_in_bibliography": set(),
        "bibliography_results": [], "processing_steps": [], "soft_flagged_claims": [],
        "work_manuscripts": {},
    }

    result = await service._finalize_response_node(state)

    assert "Related catalog entries" not in result["final_answer"]
    assert "Works cited" not in result["final_answer"]
    assert "could not compose an answer" in result["final_answer"]


# ---------------------------------------------------------------------------
# Judge acceptance harness (encodes the human-caught failures as gold)
# ---------------------------------------------------------------------------

def test_check_expectations_flags_band_violations() -> None:
    """The acceptance checker reports every violated expected band."""
    from scripts.judge_acceptance import check_expectations

    # Empty answer the judge wrongly scored high on question_answered.
    violations = check_expectations(
        {"question_answered": 8, "answer_flow": 9}, computed_pass=True,
        expect={"overall_pass": False, "dimension_max": {"question_answered": 2}, "mean_max": 3.5},
    )
    assert any("question_answered=8 exceeds max 2" in v for v in violations)
    assert any("overall_pass=True, expected False" in v for v in violations)
    assert any("mean" in v for v in violations)

    # A judge that grades the same empty answer correctly has no violations.
    assert check_expectations(
        {"question_answered": 1, "answer_flow": 1}, computed_pass=False,
        expect={"overall_pass": False, "dimension_max": {"question_answered": 2}, "mean_max": 3.5},
    ) == []


def test_check_expectations_enforces_minimums_for_good_answers() -> None:
    """A judge that fails a genuinely good answer is flagged too."""
    from scripts.judge_acceptance import check_expectations

    violations = check_expectations(
        {"groundedness_and_accuracy": 3, "question_answered": 4}, computed_pass=False,
        expect={"overall_pass": True, "dimension_min": {"groundedness_and_accuracy": 6}, "mean_min": 6.5},
    )
    assert any("groundedness_and_accuracy=3 below min 6" in v for v in violations)
    assert any("mean" in v and "below min" in v for v in violations)


@pytest.mark.asyncio
async def test_judge_acceptance_run_detects_a_rubber_stamp_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stub judge that scores everything perfect is reported NOT trustworthy."""
    import scripts.run_agentic_rag_eval as runner
    from scripts.judge_acceptance import run as acceptance_run

    def rubber_stamp(model: str, instructions: str, user_content: str, timeout: float) -> str:
        return json.dumps({
            "scores": {d: 10 for d in [
                "question_answered", "answer_flow", "primary_source_linking",
                "groundedness_and_accuracy", "citation_and_quote_quality",
                "retrieval_evidence_coverage", "restraint_and_limitations"]},
            "critical_failures": [], "rubric_findings": [], "unsupported_claims": [],
            "overall_pass": True, "summary": "everything is great",
        })

    monkeypatch.setattr(runner, "anthropic_judge_text", rubber_stamp)
    args = type("Args", (), {
        "gold": str(PROJECT_ROOT / "evals" / "judge_acceptance_v1.json"),
        "judge_prompt": str(PROJECT_ROOT / "evals" / "judge_prompt.md"),
        "judge_base_url": "http://unused", "judge_model": "stub",
        "judge_provider": "anthropic", "timeout": 5.0,
    })()

    trustworthy, results = await acceptance_run(args)

    assert not trustworthy  # a rubber-stamp judge must be caught
    empty_case = next(r for r in results if r["id"] == "empty_appendix_only_must_fail")
    assert not empty_case["trustworthy"]
    assert empty_case["violations"]


@pytest.mark.asyncio
async def test_judge_acceptance_passes_a_discerning_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stub judge that grades each gold case within its bands is trustworthy."""
    import scripts.run_agentic_rag_eval as runner
    from scripts.judge_acceptance import run as acceptance_run

    def discerning(model: str, instructions: str, user_content: str, timeout: float) -> str:
        payload = json.loads(user_content)
        case_id = payload["case"]["id"]
        dims = ["question_answered", "answer_flow", "primary_source_linking",
                "groundedness_and_accuracy", "citation_and_quote_quality",
                "retrieval_evidence_coverage", "restraint_and_limitations"]
        if case_id == "empty_appendix_only_must_fail":
            scores = {d: 1 for d in dims}
            crit = ["no_synthesis"]
        elif case_id == "goitein_name_not_resolved_not_perfect":
            scores = {d: 8 for d in dims}
            scores["question_answered"] = 7
            scores["answer_flow"] = 6
            crit = []
        else:
            scores = {d: 8 for d in dims}
            crit = []
        return json.dumps({"scores": scores, "critical_failures": crit, "rubric_findings": [],
                           "unsupported_claims": [], "overall_pass": not crit, "summary": "graded"})

    monkeypatch.setattr(runner, "anthropic_judge_text", discerning)
    args = type("Args", (), {
        "gold": str(PROJECT_ROOT / "evals" / "judge_acceptance_v1.json"),
        "judge_prompt": str(PROJECT_ROOT / "evals" / "judge_prompt.md"),
        "judge_base_url": "http://unused", "judge_model": "stub",
        "judge_provider": "anthropic", "timeout": 5.0,
    })()

    trustworthy, results = await acceptance_run(args)

    assert trustworthy, [r["violations"] for r in results if r["violations"]]


def test_anthropic_judge_text_gives_actionable_error_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing credentials raise a clear instruction, not a raw SDK error."""
    from scripts.run_agentic_rag_eval import anthropic_judge_text

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_judge_text("claude-opus-5", "sys", "{}", 5.0)
