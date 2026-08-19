from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from omnisearch.manifest import CaptionRecord, ImageRecord
from omnisearch.phase15 import (
    IMAGE_FAMILIES,
    SEVERITIES,
    TEXT_FAMILIES,
    _shift_definition,
    apply_image_corruption,
    apply_text_corruption,
    default_corruption_config,
    metric_degradation,
    rank_stability,
    validate_corruption_config,
)


def test_protocol_declares_all_families_and_severities() -> None:
    config = default_corruption_config()
    validate_corruption_config(config)
    assert tuple(config["severity_levels"]) == SEVERITIES
    assert tuple(config["text_corruptions"]) == TEXT_FAMILIES
    assert tuple(config["image_corruptions"]) == IMAGE_FAMILIES
    assert config["application"]["new_dataset_written"] is False


@pytest.mark.parametrize("family", TEXT_FAMILIES)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_text_corruptions_are_deterministic_and_nonempty(family: str, severity: str) -> None:
    original = "A brown dog runs beside a red bicycle."
    first = apply_text_corruption(original, family, severity, seed=123)
    second = apply_text_corruption(original, family, severity, seed=123)
    assert first == second
    assert original == "A brown dog runs beside a red bicycle."
    assert first.strip()


@pytest.mark.parametrize("family", IMAGE_FAMILIES)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_image_corruptions_return_same_sized_rgb_copy(family: str, severity: str) -> None:
    original = Image.new("RGB", (32, 24), (100, 120, 140))
    before = original.tobytes()
    first = apply_image_corruption(original, family, severity, seed=123)
    second = apply_image_corruption(original, family, severity, seed=123)
    assert first.mode == "RGB"
    assert first.size == original.size
    assert first.tobytes() == second.tobytes()
    assert original.tobytes() == before


def test_metric_degradation_is_zero_safe() -> None:
    assert metric_degradation(0.0, 0.2) == {
        "absolute_delta": 0.2,
        "relative_degradation": None,
        "retention": None,
    }
    values = metric_degradation(0.8, 0.6)
    assert values["absolute_delta"] == pytest.approx(-0.2)
    assert values["relative_degradation"] == pytest.approx(0.25)
    assert values["retention"] == pytest.approx(0.75)


def test_rank_stability_reports_overlap_and_relevant_displacement() -> None:
    values = rank_stability(["a", "b", "c", "d", "e"], ["b", "a", "c", "d", "e"], {"a"}, 5)
    assert values["top1_preserved"] == 0
    assert values["top_k_overlap"] == pytest.approx(1.0)
    assert values["relevant_rank_displacement"] == 1.0
    assert values["rank_displacement_observed"] == 1


def test_shift_selection_is_metadata_only_and_balanced(tmp_path: Path) -> None:
    records = []
    shapes = [(20, 40), (40, 20), (30, 45), (45, 30), (40, 40), (42, 42)]
    for index, shape in enumerate(shapes):
        filename = f"{index}.jpg"
        Image.new("RGB", shape, (index, index, index)).save(tmp_path / filename)
        records.append(
            ImageRecord(
                image_id=f"img-{index}",
                filename=filename,
                captions=(CaptionRecord(caption_id=f"cap-{index}", text="a caption"),),
                split="test",
            )
        )
    definition = _shift_definition(records, tmp_path)
    assert definition["selection_precedes_metrics"] is True
    assert len(definition["shifted_groups"]) == 2
    assert len(definition["control_groups"]) == 2
    assert {row["image_id"] for row in definition["shifted_groups"]}.isdisjoint(
        {row["image_id"] for row in definition["control_groups"]}
    )
