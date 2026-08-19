from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnisearch.phase19 import (
    MIN_GROUP_SIZE,
    caption_group_features,
    compute_disparity,
    image_aspect_group,
    minimum_group_status,
    validate_finding_category,
    validate_no_protected_attributes,
    validate_phase19_artifacts,
)


def test_caption_groups_are_reproducible_and_dataset_derived() -> None:
    counts = {"dogs": 60, "ball": 6, "red": 100}
    first = caption_group_features("Dogs and a red ball", counts)
    second = caption_group_features("Dogs and a red ball", counts)
    assert first == second
    assert first["caption_length_group"] == "short"
    assert first["concept_rarity_group"] == "medium"
    assert first["complexity_group"] == "multi_object_or_structured"
    assert first["feature_label_source"] == "mechanical_or_lexical_heuristic"


def test_image_aspect_and_minimum_group_size_contract() -> None:
    assert image_aspect_group(100, 100) == "near_square"
    assert image_aspect_group(200, 100) == "wide_or_tall_extreme"
    assert image_aspect_group(100, 200) == "wide_or_tall_extreme"
    assert minimum_group_status(MIN_GROUP_SIZE)["eligible_for_strong_interpretation"] is True
    assert minimum_group_status(MIN_GROUP_SIZE - 1)["interpretation"] == "descriptive_only_small_group"


def test_disparity_is_absolute_and_not_a_fairness_claim() -> None:
    result = compute_disparity(
        {"group_label": "common", "top1_rate": 0.8},
        {"group_label": "rare", "top1_rate": 0.6},
    )
    assert result["absolute_gap"] == pytest.approx(0.2)
    assert result["signed_comparison_minus_reference"] == pytest.approx(-0.2)
    assert result["fairness_claim_made"] is False


def test_evidence_categories_and_protected_labels_are_guarded() -> None:
    assert validate_finding_category("MEASURED RISK") is True
    assert validate_finding_category("NOT EVALUATED") is True
    with pytest.raises(ValueError):
        validate_finding_category("FAIR")
    assert validate_no_protected_attributes({"caption_length": {"labels": ["short", "long"]}}) is True
    with pytest.raises(ValueError):
        validate_no_protected_attributes({"group_label": "gender"})
    with pytest.raises(ValueError):
        validate_no_protected_attributes({"race": {"labels": ["not supplied"]}})


def test_phase19_artifact_validation_rejects_missing_directory(tmp_path: Path) -> None:
    result = validate_phase19_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False

    (tmp_path / "group_definitions.json").write_text(json.dumps({"families": {}}))
    malformed = validate_phase19_artifacts(tmp_path)
    assert malformed["passed"] is False
