from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnisearch.phase17 import (
    apply_logistic_calibrator,
    area_under_risk_coverage,
    confidence_proxies,
    correctness_targets,
    expected_calibration_error,
    fit_logistic_calibrator,
    reliability_bins,
    selective_metrics,
    validate_phase17_artifacts,
)


def test_confidence_proxies_and_margin_are_deterministic() -> None:
    first = confidence_proxies([0.9, 0.4, 0.1])
    second = confidence_proxies([0.9, 0.4, 0.1])
    assert first == second
    assert first["top1_score"] == 0.9
    assert first["top1_top2_margin"] == pytest.approx(0.5)
    assert 0.0 < first["softmax_top1_mass"] < 1.0
    assert 0.0 <= first["entropy_confidence"] <= 1.0


def test_confidence_proxy_rejects_malformed_scores() -> None:
    with pytest.raises(ValueError):
        confidence_proxies([0.9])
    with pytest.raises(ValueError):
        confidence_proxies([0.9, float("nan")])
    with pytest.raises(ValueError):
        confidence_proxies([0.9, 0.4], temperature=0.0)


def test_correctness_targets_encode_rank_and_severity() -> None:
    success = correctness_targets(["a", "b", "c"], ["a"])
    assert success["top1_correct"] is True
    assert success["top5_correct"] is True
    assert success["rank_severity"] == "success"

    low = correctness_targets(["b", "a", "c"], ["a"])
    assert low["top1_correct"] is False
    assert low["top5_correct"] is True
    assert low["rank_severity"] == "low"

    severe = correctness_targets(["a", "b", "c"], ["missing"])
    assert severe["rank_observed"] is False
    assert severe["rank_severity"] == "severe"

    with pytest.raises(ValueError):
        correctness_targets(["a", "a"], ["a"])


def test_logistic_calibration_is_validation_only_and_bounded() -> None:
    parameters = fit_logistic_calibrator(
        [0.1, 0.2, 0.8, 0.9],
        [False, False, True, True],
    )
    assert parameters["fit_split"] == "validation"
    assert parameters["test_labels_used_for_fit"] is False
    assert 0.0 < apply_logistic_calibrator(parameters, 0.85) < 1.0
    with pytest.raises(ValueError):
        apply_logistic_calibrator({**parameters, "fit_split": "test"}, 0.85)


def test_reliability_bins_and_ece_use_explicit_empty_bin_handling() -> None:
    bins = reliability_bins([0.05, 0.15, 0.95], [True, False, True], bins=10)
    assert [row["bin"] for row in bins] == [0, 1, 9]
    assert expected_calibration_error(bins, 3) >= 0.0
    assert all("empirical_accuracy" in row for row in bins)


def test_selective_metrics_coverage_accuracy_and_risk() -> None:
    result = selective_metrics([0.9, 0.8, 0.2, 0.1], [True, False, True, False], 0.8)
    assert result["accepted_queries"] == 2
    assert result["coverage"] == pytest.approx(0.5)
    assert result["selective_top1_accuracy"] == pytest.approx(0.5)
    assert result["risk"] == pytest.approx(0.5)


def test_auc_and_aurc_are_defined_for_binary_targets() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [True, True, False, False]
    curve = area_under_risk_coverage(scores, labels)
    assert curve["aurc"] == pytest.approx(0.2083333333)
    assert len(curve["points"]) == 4


def test_artifact_validation_rejects_missing_or_malformed_directory(tmp_path: Path) -> None:
    result = validate_phase17_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False

    (tmp_path / "confidence_definition.json").write_text(json.dumps({"raw_proxies": []}))
    malformed = validate_phase17_artifacts(tmp_path)
    assert malformed["passed"] is False
