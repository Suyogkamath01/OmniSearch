from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnisearch.phase18 import (
    delete_token,
    faithfulness_comparison,
    grid_regions,
    occlude_region,
    rank_target,
    score_rank_delta,
    token_importance_rows,
    tokenize_for_occlusion,
    validate_phase18_artifacts,
)


def test_tokenization_and_deletion_are_deterministic() -> None:
    text = "A red dog, running."
    assert tokenize_for_occlusion(text) == ["A", "red", "dog", ",", "running", "."]
    assert delete_token(text, 1) == "A dog, running."
    assert delete_token(text, 3) == "A red dog running."
    assert delete_token(text, 0) == "red dog, running."
    with pytest.raises(IndexError):
        delete_token(text, 99)


def test_rank_target_and_score_rank_delta() -> None:
    baseline = rank_target(["a", "b", "c"], [0.9, 0.4, 0.2], ["a"])
    perturbed = rank_target(["a", "b", "c"], [0.1, 0.8, 0.2], ["a"])
    assert baseline["target_rank"] == 1
    assert perturbed["target_rank"] == 3
    delta = score_rank_delta(baseline, perturbed)
    assert delta["score_delta"] == pytest.approx(0.8)
    assert delta["rank_delta"] == 2


def test_token_importance_and_faithfulness_comparison() -> None:
    baseline = rank_target(["a", "b"], [0.8, 0.2], ["a"])
    variants = [
        {"token_index": 0, "token": "dog", "text_without_token": "runs", "summary": rank_target(["a", "b"], [0.2, 0.7], ["a"])},
        {"token_index": 1, "token": "runs", "text_without_token": "dog", "summary": rank_target(["a", "b"], [0.75, 0.25], ["a"])},
    ]
    rows = token_importance_rows("dog runs", baseline, variants)
    assert rows[0]["importance_score_delta"] == pytest.approx(0.6)
    comparison = faithfulness_comparison(rows, "token")
    assert comparison["score_order_supports_faithfulness"] is True
    assert comparison["most_sensitive"]["token"] == "dog"


def test_grid_regions_and_occlusion() -> None:
    regions = grid_regions(9, 6, 3)
    assert len(regions) == 9
    assert regions[0]["pixel_box"] == [0, 0, 3, 2]
    assert regions[-1]["pixel_box"] == [6, 4, 9, 6]

    pytest.importorskip("PIL")
    from PIL import Image

    image = Image.new("RGB", (9, 6), (255, 0, 0))
    for x in range(3):
        for y in range(2):
            image.putpixel((x, y), (x * 40, y * 40, 20))
    changed = occlude_region(image, regions[0])
    assert changed.size == image.size
    assert changed.getpixel((0, 0)) != image.getpixel((0, 0))


def test_malformed_token_importance_is_rejected() -> None:
    baseline = rank_target(["a", "b"], [0.8, 0.2], ["a"])
    with pytest.raises(TypeError):
        token_importance_rows("x", baseline, [{"token_index": 0, "token": "x", "text_without_token": "[MASK]"}])


def test_phase18_artifact_validation_rejects_missing_directory(tmp_path: Path) -> None:
    result = validate_phase18_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False

    (tmp_path / "explanation_records.json").write_text(json.dumps({"rows": []}))
    malformed = validate_phase18_artifacts(tmp_path)
    assert malformed["passed"] is False
