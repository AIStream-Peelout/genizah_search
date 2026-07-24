"""Structural and matching tests for the agentic RAG evaluation dataset."""

from pathlib import Path
from typing import Any, Dict

from scripts.run_agentic_rag_eval import (
    bibliography_target_matches,
    evaluate_deterministically,
    graph_target_matches,
    load_json,
    validate_judge_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "evals" / "agentic_rag_v1.json"
EXPECTED_CASE_IDS = {
    "yom_kippur_piyyut_literature",
    "goitein_identity_and_contributions",
    "bible_codices_overview",
    "ketubbot_overview",
    "purim_fragments",
    "passover_seder_evolution",
    "estara_arrant_profile",
    "shavuot_dairy_custom",
    "tisha_bav_fragments",
    "tisha_bav_kinnot",
    "tu_bav_history",
}


def dataset_by_case_id() -> Dict[str, Dict[str, Any]]:
    """Load the evaluation cases keyed by stable identifier.

    :returns: Evaluation cases keyed by case identifier.
    :rtype: Dict[str, Dict[str, Any]]
    """
    dataset = load_json(DATASET_PATH)
    return {case["id"]: case for case in dataset["cases"]}


def test_dataset_contains_complete_unique_case_set() -> None:
    """Keep the agreed Phase 1 question set complete and uniquely identified."""
    dataset = load_json(DATASET_PATH)
    case_ids = [case["id"] for case in dataset["cases"]]

    assert len(case_ids) == len(set(case_ids))
    assert set(case_ids) == EXPECTED_CASE_IDS


def test_every_case_has_actionable_routing_retrieval_and_answer_rubrics() -> None:
    """Require every case to specify each independently scored failure surface."""
    for case in dataset_by_case_id().values():
        assert case["question"].strip()
        assert case["query_variants"]
        assert case["expected_intent"].strip()
        assert case["corpus_coverage"] in {"strong", "partial", "not_found"}
        assert case["routing"]["must_include_any"]
        assert "must_not_use_as_primary" in case["routing"]
        assert "must_find_any" in case["retrieval"]
        assert case["retrieval"]["known_distractors"]
        assert case["answer_rubric"]["must_include"]
        assert case["answer_rubric"]["must_not_include"]
        assert case["answer_rubric"]["limitation_behavior"].strip()


def test_coverage_level_controls_required_retrieval_targets() -> None:
    """Reserve empty retrieval oracles for intentionally unsupported questions."""
    for case in dataset_by_case_id().values():
        targets = case["retrieval"]["must_find_any"]
        if case["corpus_coverage"] == "not_found":
            assert targets == []
        elif case["corpus_coverage"] == "strong":
            assert targets


def test_tu_bav_case_explicitly_guards_against_tisha_bav_confusion() -> None:
    """Make the near-name holiday collision an explicit critical expectation."""
    case = dataset_by_case_id()["tu_bav_history"]
    rubric_text = " ".join(
        case["answer_rubric"]["must_include"]
        + case["answer_rubric"]["must_not_include"]
        + case["retrieval"]["known_distractors"]
    ).casefold()

    assert "fifteenth of av" in rubric_text
    assert "tisha b'av" in rubric_text
    assert "ninth of av" in rubric_text


def test_bibliography_target_matching_requires_all_supplied_constraints() -> None:
    """Match title, author, and audited page rather than a loose keyword hit."""
    target = {
        "kind": "bibliography",
        "title_contains": "A Unique Kol-nidré Piyyut",
        "author_contains": "Meir Wallenstein",
        "pages_any": [488, 489],
    }
    result = {
        "title": "A Unique Kol-nidré Piyyut from the Cairo Genizah",
        "authors": ["Meir Wallenstein"],
        "extracted_page_number": 489,
    }

    assert bibliography_target_matches(target, result)
    assert not bibliography_target_matches(
        target,
        {**result, "extracted_page_number": 43},
    )


def test_graph_target_matching_uses_canonical_scholar_name() -> None:
    """Require the resolved canonical scholar rather than an abbreviation."""
    target = {"kind": "graph", "scholar_name": "Shelomo Dov Goitein"}

    assert graph_target_matches(
        target,
        {"scholar": {"name": "Shelomo Dov Goitein"}},
    )
    assert not graph_target_matches(
        target,
        {"scholar": {"name": "S.D. Goitein"}},
    )


def test_deterministic_evaluation_scores_routing_and_retrieval_separately() -> None:
    """Fail an otherwise correct route when its known evidence target is absent."""
    case = dataset_by_case_id()["yom_kippur_piyyut_literature"]
    response = {
        "query_plan": {
            "actions": [
                {
                    "search_type": "bibliography_hybrid",
                    "query": "Yom Kippur piyyut Cairo Genizah",
                }
            ]
        },
        "bibliography_results": [],
    }

    result = evaluate_deterministically(case, response)

    assert result["routing_required_any_pass"]
    assert not result["retrieval_must_find_any_pass"]
    assert not result["overall_pass"]


def test_judge_pass_rule_is_computed_independently() -> None:
    """Do not trust a judge model's self-reported pass-rule arithmetic."""
    dimensions = [
        "intent_and_relevance",
        "retrieval_evidence_coverage",
        "groundedness_and_accuracy",
        "citation_and_quote_quality",
        "synthesis_and_coherence",
        "restraint_and_limitations",
    ]
    judge_result = {
        "scores": {dimension: 3 for dimension in dimensions},
        "critical_failures": [],
        "overall_pass": False,
    }

    annotated = validate_judge_result(
        judge_result,
        dimensions,
        {"overall_pass": True},
    )

    assert annotated["score_mean"] == 3.0
    assert annotated["computed_overall_pass"]
    assert not annotated["reported_pass_matches_rule"]


def test_judge_pass_rule_honors_deterministic_retrieval_failure() -> None:
    """Prevent fluent synthesis from passing when required retrieval failed."""
    dimensions = ["groundedness_and_accuracy"]
    judge_result = {
        "scores": {"groundedness_and_accuracy": 4},
        "critical_failures": [],
        "overall_pass": True,
    }

    annotated = validate_judge_result(
        judge_result,
        dimensions,
        {"overall_pass": False},
    )

    assert not annotated["computed_overall_pass"]
    assert not annotated["reported_pass_matches_rule"]
