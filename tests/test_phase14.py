from __future__ import annotations

import json

import pytest

from omnisearch.phase14 import (
    MANDATORY_ABLATION_IDS,
    classify_component_value,
    component_delta,
    default_ablation_plan,
    effect_label,
    validate_ablation_plan,
    validate_comparison_compatibility,
    validate_phase14_artifacts,
)


def test_default_plan_covers_mandatory_ablations_and_one_component_changes() -> None:
    plan = default_ablation_plan()
    result = validate_ablation_plan(plan)
    selected = {item["id"] for item in plan["selected_ablations"]}
    assert set(MANDATORY_ABLATION_IDS) <= selected
    assert result["all_selected_change_one_component"] is True
    assert result["new_true_ablations"] == ["hard_negative_ratio_25_vs_existing_0_50"]


def test_plan_rejects_two_changed_components_and_forbidden_work() -> None:
    plan = default_ablation_plan()
    plan["selected_ablations"][0]["changed_components"] = ["fine_tuning", "reranking"]
    with pytest.raises(ValueError, match="exactly one component"):
        validate_ablation_plan(plan)

    plan = default_ablation_plan()
    plan["selected_ablations"][0]["changed_components"] = ["fine_tuning"]
    plan["selected_ablations"][0]["description"] = "start CIRCO execution"
    with pytest.raises(ValueError, match="forbidden"):
        validate_ablation_plan(plan)


def test_comparison_compatibility_and_delta_calculation() -> None:
    metadata = {
        "manifest_sha256": "same",
        "protocol_version": "retrieval_eval_v1",
        "split": "test",
        "direction": "text_to_image",
        "candidate_unit": "image_group",
    }
    assert validate_comparison_compatibility(metadata, dict(metadata))["status"] == "compatible"
    with pytest.raises(ValueError, match="incompatible"):
        validate_comparison_compatibility(metadata, {**metadata, "split": "validation"})
    assert component_delta({"r5": 0.9, "mrr": 0.8}, {"r5": 0.7, "mrr": 0.75}) == {
        "mrr": pytest.approx(0.05),
        "r5": pytest.approx(0.2),
    }


def test_component_classification_does_not_upgrade_mixed_evidence() -> None:
    assert effect_label([0.1, 0.2, 0.3]) == "positive"
    assert effect_label([-0.1, -0.2, -0.3]) == "negative"
    assert effect_label([0.1, -0.1, 0.0]) == "mixed / seed-sensitive"
    assert classify_component_value([0.1, 0.2, 0.3]) == "KEEP"
    assert classify_component_value([0.1, -0.1, 0.0]) == "OPTIONAL"
    assert classify_component_value([-0.1, -0.2, -0.3]) == "REMOVE / NOT RECOMMENDED"
    assert classify_component_value([0.1, 0.2, 0.3], efficiency_tradeoff=True) == "OPTIONAL"


def test_artifact_validation_rejects_malformed_report(tmp_path) -> None:
    (tmp_path / "phase14_report.json").write_text(
        json.dumps({"phase": 14, "status": "PARTIAL", "quality_gate": {}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        validate_phase14_artifacts(tmp_path)
