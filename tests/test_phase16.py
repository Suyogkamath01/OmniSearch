from __future__ import annotations

from pathlib import Path

import pytest

from omnisearch.phase16 import (
    _caption_features,
    _text_heuristic_categories,
    classify_rank_movement,
    classify_transition,
    default_failure_definitions,
    failure_triggers,
    first_relevant_rank,
    rank_severity,
    score_margin,
    success_at_k,
    validate_failure_definitions,
    validate_phase16_artifacts,
)


def _ranking(*, candidates: list[str], scores: list[float], relevant: list[str]) -> dict:
    return {
        "query_id": "q1",
        "candidate_ids": candidates,
        "scores": scores,
        "relevant_ids": relevant,
        "candidate_count": 100,
    }


def test_failure_definitions_are_fixed_and_valid() -> None:
    definitions = default_failure_definitions()
    validate_failure_definitions(definitions)
    assert definitions["rank_prefix"] == 10
    assert definitions["label_policy"]["no_forced_semantic_label"] is True


def test_first_relevant_rank_and_severity_are_deterministic() -> None:
    observed = first_relevant_rank(_ranking(candidates=["wrong", "target"], scores=[0.8, 0.7], relevant=["target"]))
    missing = first_relevant_rank(_ranking(candidates=["wrong"], scores=[0.8], relevant=["target"]))
    assert observed == {"rank": 2, "observed": True, "lower_bound": 2, "relevant_in_prefix": True}
    assert missing["observed"] is False
    assert missing["lower_bound"] == 2
    assert rank_severity(observed) == "low"
    assert rank_severity({"rank": None, "observed": False, "lower_bound": 11}) == "high"


def test_failure_triggers_and_success_at_k() -> None:
    info = {"rank": 6, "observed": True, "lower_bound": 6}
    assert failure_triggers(info) == {"top1_failure": True, "top5_failure": True, "severe_failure": False}
    assert success_at_k(info, 5) is False
    assert success_at_k(info, 10) is True


@pytest.mark.parametrize(
    ("left", "right", "category"),
    [
        ({"rank": 8, "observed": True}, {"rank": 2, "observed": True}, "major_improvement"),
        ({"rank": 2, "observed": True}, {"rank": 3, "observed": True}, "minor_regression"),
        ({"rank": 2, "observed": True}, {"rank": 2, "observed": True}, "unchanged"),
        ({"rank": None, "observed": False}, {"rank": 3, "observed": True}, "major_improvement"),
        ({"rank": None, "observed": False}, {"rank": None, "observed": False}, "censored"),
    ],
)
def test_rank_movement_thresholds(left: dict, right: dict, category: str) -> None:
    assert classify_rank_movement(left, right)["movement_category"] == category


def test_transition_categories_are_exhaustive() -> None:
    assert classify_transition(True, True) == "both_succeed"
    assert classify_transition(True, False) == "only_left_succeeds"
    assert classify_transition(False, True) == "only_right_succeeds"
    assert classify_transition(False, False) == "both_fail"


def test_score_margin_distinguishes_close_and_missing_target() -> None:
    close = score_margin(_ranking(candidates=["wrong", "target"], scores=[0.8, 0.79], relevant=["target"]))
    missing = score_margin(_ranking(candidates=["wrong"], scores=[0.8], relevant=["target"]))
    assert close["category"] == "near_miss"
    assert close["label_source"] == "mechanical"
    assert missing["category"] == "target_not_in_retained_prefix"


def test_heuristic_taxonomy_is_marked_as_heuristic() -> None:
    features = _caption_features("two red dogs running beside a car", {"two": 1, "red": 1, "dogs": 1, "running": 1, "beside": 1, "a": 1, "car": 1})
    categories = _text_heuristic_categories(features)
    assert categories
    assert all(row["label_source"] == "heuristic" for row in categories)
    assert any(row["category"] == "counting_quantity" for row in categories)


def test_malformed_artifact_directory_fails_validation(tmp_path: Path) -> None:
    result = validate_phase16_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False


def test_invalid_definitions_are_rejected() -> None:
    definitions = default_failure_definitions()
    definitions["label_policy"]["no_forced_semantic_label"] = False
    with pytest.raises(ValueError, match="force unsupported"):
        validate_failure_definitions(definitions)
