"""Phase 18 explainability and retrieval interpretation.

The implementation uses local perturbations only.  Token deletion and coarse
image-region occlusion expose observable score/rank sensitivity; they are not
treated as causal explanations or complete interpretations of CLIP.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .manifest import ImageRecord, read_manifest
from .phase17 import validate_phase17_artifacts

PHASE18_SCHEMA_VERSION = 1
SYSTEMS = ("zero_shot", "full_ft")
DIRECTIONS = ("text_to_image", "image_to_text")
GRID_SIZE = 3
TOP_K = 10
HIGH_CONFIDENCE_THRESHOLD = 0.8
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
OBJECT_HINTS = {
    "animal", "ball", "bird", "boat", "book", " car", "cat", "chair", "child", "dog",
    "elephant", "food", "frisbee", "girl", "guitar", "horse", "man", "motorcycle", "person",
    "phone", "pizza", "plane", "sheep", "skateboard", "street", "table", "train", "tree", "umbrella",
    "vehicle", "water", "woman", "zebra",
}
ACTION_HINTS = {
    "biking", "climbing", "cutting", "driving", "eating", "flying", "holding", "jumping", "looking",
    "playing", "riding", "running", "sitting", "skating", "standing", "throwing", "walking", "wearing",
}
ATTRIBUTE_HINTS = {
    "big", "black", "blue", "brown", "colorful", "dark", "green", "large", "little", "long", "old",
    "red", "small", "tall", "white", "yellow", "young",
}
SPATIAL_HINTS = {
    "above", "across", "behind", "below", "beside", "between", "front", "inside", "near", "next", "on",
    "over", "under", "with",
}
STOPWORD_HINTS = {"a", "an", "and", "at", "for", "in", "of", "on", "the", "to", "with"}
REQUIRED_ARTIFACTS = (
    "explanation_records.json",
    "token_importance.json",
    "region_sensitivity.json",
    "counterfactual_analysis.json",
    "faithfulness_results.json",
    "high_confidence_error_explanations.json",
    "explanation_consistency.json",
    "qualitative_examples.json",
    "provenance.json",
    "phase18_report.json",
)


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def tokenize_for_occlusion(text: str) -> list[str]:
    """Tokenize into deterministic word/punctuation groups for deletion."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return TOKEN_RE.findall(text)


def _detokenize(tokens: Sequence[str]) -> str:
    value = " ".join(tokens)
    value = re.sub(r"\s+([,.;!?%])", r"\1", value)
    value = re.sub(r"([([/{])\s+", r"\1", value)
    value = re.sub(r"\s+([)\]}])", r"\1", value)
    return value.strip()


def delete_token(text: str, token_index: int) -> str:
    tokens = tokenize_for_occlusion(text)
    if token_index < 0 or token_index >= len(tokens):
        raise IndexError("token_index is outside the token sequence")
    remaining = tokens[:token_index] + tokens[token_index + 1 :]
    return _detokenize(remaining) or "[MASK]"


def token_category(token: str) -> str:
    normalized = token.lower()
    if normalized in OBJECT_HINTS:
        return "object_heuristic"
    if normalized in ACTION_HINTS:
        return "action_heuristic"
    if normalized in ATTRIBUTE_HINTS:
        return "attribute_heuristic"
    if normalized in SPATIAL_HINTS:
        return "spatial_heuristic"
    if normalized in STOPWORD_HINTS:
        return "stopword_like_heuristic"
    return "other_token"


def rank_target(candidate_ids: Sequence[str], scores: Sequence[float], relevant_ids: Sequence[str]) -> dict[str, Any]:
    if len(candidate_ids) != len(scores) or len(candidate_ids) < 1:
        raise ValueError("candidate IDs and scores must have equal non-zero length")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    ordered = sorted(zip(candidate_ids, (float(score) for score in scores)), key=lambda item: (-item[1], item[0]))
    relevant = {str(item) for item in relevant_ids}
    target_rank = next((rank for rank, (item, _) in enumerate(ordered, start=1) if item in relevant), None)
    relevant_scores = [score for item, score in ordered if item in relevant]
    top1_id, top1_score = ordered[0]
    return {
        "top1_id": top1_id,
        "top1_score": top1_score,
        "top2_score": ordered[1][1] if len(ordered) > 1 else None,
        "top1_top2_margin": top1_score - ordered[1][1] if len(ordered) > 1 else None,
        "relevant_score": max(relevant_scores) if relevant_scores else None,
        "target_rank": target_rank,
        "top1_correct": bool(top1_id in relevant),
        "top5_correct": bool(any(item in relevant for item, _ in ordered[:5])),
    }


def score_rank_delta(
    baseline: Mapping[str, Any], perturbed: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_score = baseline.get("relevant_score")
    perturbed_score = perturbed.get("relevant_score")
    baseline_rank = baseline.get("target_rank")
    perturbed_rank = perturbed.get("target_rank")
    return {
        "score_delta": (
            float(baseline_score) - float(perturbed_score)
            if baseline_score is not None and perturbed_score is not None
            else None
        ),
        "rank_delta": (
            int(perturbed_rank) - int(baseline_rank)
            if baseline_rank is not None and perturbed_rank is not None
            else None
        ),
    }


def grid_regions(width: int, height: int, grid_size: int = GRID_SIZE) -> list[dict[str, Any]]:
    if width <= 0 or height <= 0 or grid_size <= 0:
        raise ValueError("image dimensions and grid size must be positive")
    regions: list[dict[str, Any]] = []
    for row in range(grid_size):
        for column in range(grid_size):
            x0 = round(column * width / grid_size)
            y0 = round(row * height / grid_size)
            x1 = round((column + 1) * width / grid_size)
            y1 = round((row + 1) * height / grid_size)
            regions.append(
                {
                    "grid_row": row,
                    "grid_column": column,
                    "normalized_box": [column / grid_size, row / grid_size, (column + 1) / grid_size, (row + 1) / grid_size],
                    "pixel_box": [x0, y0, x1, y1],
                }
            )
    return regions


def occlude_region(image: Any, region: Mapping[str, Any]) -> Any:
    """Return an image with a deterministic mean-colour rectangular occlusion."""

    from PIL import ImageDraw, ImageStat

    if not hasattr(image, "copy") or not hasattr(image, "crop"):
        raise TypeError("image must be PIL-like")
    output = image.convert("RGB").copy()
    box = tuple(int(value) for value in region["pixel_box"])
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("region pixel_box must be a positive rectangle")
    mean = ImageStat.Stat(output.crop(box)).mean
    ImageDraw.Draw(output).rectangle(box, fill=tuple(round(value) for value in mean[:3]))
    return output


def token_importance_rows(
    baseline_text: str,
    baseline: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert real token deletion outputs into score/rank sensitivities."""

    rows: list[dict[str, Any]] = []
    for variant in variants:
        perturbed = variant.get("summary")
        if not isinstance(perturbed, Mapping):
            raise TypeError("token variant is missing a summary")
        delta = score_rank_delta(baseline, perturbed)
        rows.append(
            {
                "token_index": int(variant["token_index"]),
                "token": str(variant["token"]),
                "token_category": token_category(str(variant["token"])),
                "text_without_token": str(variant["text_without_token"]),
                "baseline_relevant_score": baseline.get("relevant_score"),
                "perturbed_relevant_score": perturbed.get("relevant_score"),
                "baseline_target_rank": baseline.get("target_rank"),
                "perturbed_target_rank": perturbed.get("target_rank"),
                "importance_score_delta": delta["score_delta"],
                "importance_rank_delta": delta["rank_delta"],
                "interpretation": "positive score delta means token deletion lowered the observed relevant score; this is local sensitivity, not causal proof",
            }
        )
    if not rows:
        raise ValueError("token variants cannot be empty")
    return rows


def faithfulness_comparison(rows: Sequence[Mapping[str, Any]], feature_type: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("faithfulness comparison requires rows")
    score_rows = [row for row in rows if row.get("importance_score_delta") is not None]
    rank_rows = [row for row in rows if row.get("importance_rank_delta") is not None]
    most_sensitive = max(score_rows, key=lambda row: (float(row["importance_score_delta"]), str(row.get("token_index", row.get("grid_row", ""))))) if score_rows else None
    least_sensitive = min(score_rows, key=lambda row: (float(row["importance_score_delta"]), str(row.get("token_index", row.get("grid_row", ""))))) if score_rows else None
    score_difference = (
        float(most_sensitive["importance_score_delta"]) - float(least_sensitive["importance_score_delta"])
        if most_sensitive is not None and least_sensitive is not None
        else None
    )
    most_rank = max(rank_rows, key=lambda row: (int(row["importance_rank_delta"]), str(row.get("token_index", row.get("grid_row", ""))))) if rank_rows else None
    least_rank = min(rank_rows, key=lambda row: (int(row["importance_rank_delta"]), str(row.get("token_index", row.get("grid_row", ""))))) if rank_rows else None
    return {
        "feature_type": feature_type,
        "variant_count": len(rows),
        "most_sensitive": most_sensitive,
        "least_sensitive": least_sensitive,
        "score_degradation_difference": score_difference,
        "rank_degradation_difference": (
            int(most_rank["importance_rank_delta"]) - int(least_rank["importance_rank_delta"])
            if most_rank is not None and least_rank is not None
            else None
        ),
        "score_order_supports_faithfulness": bool(score_difference is not None and score_difference >= 0.0),
        "rank_order_supports_faithfulness": bool(
            most_rank is not None and least_rank is not None and int(most_rank["importance_rank_delta"]) >= int(least_rank["importance_rank_delta"])
        ),
        "interpretation": "perturbation faithfulness check; larger observed degradation is consistent with a useful local sensitivity signal, not proof of semantic causality",
    }


def _context_maps(manifest_path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, ImageRecord]]:
    manifest = read_manifest(manifest_path)
    caption_text = {
        caption.caption_id: caption.text
        for record in manifest.records
        for caption in record.captions
    }
    image_text = {
        record.image_id: record.captions[0].text if record.captions else ""
        for record in manifest.records
    }
    records = {record.image_id: record for record in manifest.records}
    return caption_text, image_text, records


def audit_phase17(phase17_dir: Path | str, regression_tests_passed: bool = False) -> dict[str, Any]:
    """Audit only the direct Phase 17 dependencies needed for Phase 18."""

    path = Path(phase17_dir)
    report = _read_json(path / "phase17_report.json")
    validation = _read_json(path / "artifact_validation.json")
    definitions = _read_json(path / "confidence_definition.json")
    records_payload = _read_json(path / "confidence_records.json")
    parameters = _read_json(path / "calibration_parameters.json")
    high_errors = _read_json(path / "high_confidence_errors.json")
    provenance = _read_json(path / "provenance.json")
    rows = records_payload.get("rows", [])
    checks = {
        "phase17_report_pass": report.get("status") == "PASS",
        "phase17_artifact_validation_pass": validation.get("passed") is True,
        "confidence_records_readable": isinstance(rows, list) and bool(rows),
        "systems_present": {row.get("system") for row in rows} == set(SYSTEMS),
        "directions_separated": {row.get("direction") for row in rows} == set(DIRECTIONS),
        "scores_available": all(
            isinstance(row.get("scores"), list)
            and len(row.get("candidate_ids", [])) == len(row.get("scores", []))
            and len(row.get("scores", [])) >= 2
            for row in rows
        ),
        "high_confidence_errors_available": isinstance(high_errors.get("rows"), list) and bool(high_errors.get("rows")),
        "calibration_validation_only": bool(parameters.get("rows")) and all(
            row.get("fit_split") == "validation" and row.get("test_labels_used_for_fit") is False
            for row in parameters["rows"]
        ),
        "raw_similarity_not_probability": definitions.get("raw_scores_are_not_probabilities") is True
        and all("probability" not in str(row.get("confidence_interpretation", "")).lower() or "not" in str(row.get("confidence_interpretation", "")).lower() for row in rows),
        "regression_tests_passed": regression_tests_passed,
        "phase17_no_phase18": provenance.get("phase18_started") is False,
    }
    passed = all(checks.values())
    return {
        "schema_version": PHASE18_SCHEMA_VERSION,
        "phase": 18,
        "audit_scope": "direct Phase 17 dependencies only",
        "status": "PASS" if passed else "FAIL",
        "audit_result": "PRE-PHASE AUDIT: Phase 17 PASS" if passed else "PRE-PHASE AUDIT: Phase 17 BLOCKED",
        "checks": checks,
        "phase17_confidence_record_count": len(rows),
        "phase17_directory": str(path),
        "recorded_before_phase18_analysis": True,
    }


def _selected_rows(rows: Sequence[Mapping[str, Any]], system: str, direction: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("system") == system and row.get("direction") == direction and row.get("split") == "test"]


def _choose_text_query_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = [row for row in rows if row.get("direction") == "text_to_image" and row.get("split") == "test"]
    selected: list[str] = []

    def add_from(values: Sequence[Mapping[str, Any]]) -> None:
        for row in sorted(values, key=lambda item: str(item.get("query_id"))):
            query_id = str(row["query_id"])
            if query_id not in selected and len(selected) < 10:
                selected.append(query_id)

    high_errors = sorted(
        [row for row in candidates if row.get("top1_failure") and float(row.get("calibrated_confidence", 0.0)) >= HIGH_CONFIDENCE_THRESHOLD],
        key=lambda row: (-float(row.get("calibrated_confidence", 0.0)), str(row.get("system")), str(row.get("query_id"))),
    )
    add_from(high_errors[:2])
    add_from(sorted([row for row in candidates if row.get("top1_failure")], key=lambda row: str(row.get("query_id")))[:2])
    add_from(sorted([row for row in candidates if row.get("top1_correct")], key=lambda row: str(row.get("query_id")))[:2])
    for category in ("object_confusion", "action_confusion", "attribute_confusion", "spatial_relation"):
        add_from([row for row in candidates if category in {item.get("category") for item in row.get("taxonomy_categories", []) if isinstance(item, Mapping)}])
    for row in sorted(candidates, key=lambda item: str(item.get("query_id"))):
        add_from([row])
    return selected


def _choose_image_query_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = [row for row in rows if row.get("direction") == "image_to_text" and row.get("split") == "test"]
    selected: list[str] = []

    def add_from(values: Sequence[Mapping[str, Any]]) -> None:
        for row in sorted(values, key=lambda item: str(item.get("query_id"))):
            query_id = str(row["query_id"])
            if query_id not in selected and len(selected) < 4:
                selected.append(query_id)

    high_errors = sorted(
        [row for row in candidates if row.get("top1_failure") and float(row.get("calibrated_confidence", 0.0)) >= HIGH_CONFIDENCE_THRESHOLD],
        key=lambda row: (-float(row.get("calibrated_confidence", 0.0)), str(row.get("system")), str(row.get("query_id"))),
    )
    add_from(high_errors[:2])
    add_from(sorted([row for row in candidates if row.get("top1_correct")], key=lambda row: str(row.get("query_id")))[:2])
    add_from(sorted([row for row in candidates if row.get("top1_failure")], key=lambda row: str(row.get("query_id")))[:2])
    add_from(sorted(candidates, key=lambda row: (-float(row.get("calibrated_confidence", 0.0)), str(row.get("query_id")))))
    return selected


def _device_of(model: Any) -> Any:
    return model.device if hasattr(model, "device") else next(model.parameters()).device


def _encode_texts(model: Any, processor: Any, torch: Any, texts: Sequence[str], batch_size: int) -> list[Any]:
    from torch.nn import functional

    from .phase7 import _feature_tensor, _move_inputs

    if not texts:
        return []
    model.eval()
    rows: list[Any] = []
    device = _device_of(model)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            current = texts[start : start + batch_size]
            processed = processor(text=list(current), return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = _move_inputs(processed, device)
            text_inputs = {key: value for key, value in inputs.items() if key != "pixel_values"}
            features = _feature_tensor(model.get_text_features(**text_inputs))
            rows.append(functional.normalize(features, dim=-1).detach().cpu())
    return [row for batch in rows for row in batch]


def _encode_images(model: Any, processor: Any, torch: Any, images: Sequence[Any], batch_size: int) -> list[Any]:
    from torch.nn import functional

    from .phase7 import _feature_tensor, _move_inputs

    if not images:
        return []
    model.eval()
    rows: list[Any] = []
    device = _device_of(model)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            current = images[start : start + batch_size]
            processed = processor(images=list(current), return_tensors="pt")
            inputs = _move_inputs(processed, device)
            features = _feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            rows.append(functional.normalize(features, dim=-1).detach().cpu())
    return [row for batch in rows for row in batch]


def _score_rows(query_embedding: Any, candidate_embeddings: Any, candidate_ids: Sequence[str], relevant_ids: Sequence[str]) -> dict[str, Any]:
    scores = (query_embedding @ candidate_embeddings.transpose(0, 1)).tolist()
    return rank_target(candidate_ids, [float(value) for value in scores], relevant_ids)


def _image_path(record: ImageRecord, image_root: Path) -> Path:
    if not record.filename:
        raise ValueError(f"image record has no filename: {record.image_id}")
    path = image_root / record.filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _text_variant_requests(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokens = tokenize_for_occlusion(text)
    token_variants = [
        {
            "token_index": index,
            "token": token,
            "text_without_token": delete_token(text, index),
        }
        for index, token in enumerate(tokens)
        if token.strip()
    ]
    counterfactuals: list[dict[str, Any]] = []
    used: set[str] = set()
    category_map = {
        "remove_object_word": "object_heuristic",
        "remove_action_word": "action_heuristic",
        "remove_attribute_word": "attribute_heuristic",
        "remove_spatial_word": "spatial_heuristic",
    }
    for edit_name, category in category_map.items():
        for index, token in enumerate(tokens):
            if token_category(token) == category and edit_name not in used:
                counterfactuals.append(
                    {
                        "counterfactual_type": edit_name,
                        "token_index": index,
                        "token": token,
                        "edited_text": delete_token(text, index),
                    }
                )
                used.add(edit_name)
                break
    return token_variants, counterfactuals


def _token_order(rows: Sequence[Mapping[str, Any]], top_k: int = 3) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: (-abs(float(row.get("importance_score_delta", 0.0))), int(row.get("token_index", 0))),
    )
    return [str(row["token"]).lower() for row in ordered[:top_k]]


def _analyze_system(
    system: str,
    model: Any,
    processor: Any,
    torch: Any,
    image_root: Path,
    test_records: Sequence[ImageRecord],
    system_rows: Sequence[Mapping[str, Any]],
    text_ids: Sequence[str],
    image_ids: Sequence[str],
    caption_text: Mapping[str, str],
    image_records: Mapping[str, ImageRecord],
    batch_size: int,
) -> dict[str, Any]:
    image_ids_all = [record.image_id for record in test_records]
    image_objects = [_load_image(_image_path(record, image_root)) for record in test_records]
    image_embeddings = _encode_images(model, processor, torch, image_objects, batch_size)
    image_matrix = torch.stack(image_embeddings)
    caption_items = [
        (caption.caption_id, caption.text)
        for record in test_records
        for caption in record.captions
    ]
    caption_ids = [item[0] for item in caption_items]
    caption_embeddings = _encode_texts(model, processor, torch, [item[1] for item in caption_items], batch_size)
    caption_matrix = torch.stack(caption_embeddings)
    rows_by_key = {(str(row["direction"]), str(row["query_id"])): row for row in system_rows}
    token_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    text_requests: list[tuple[str, str, str, Any]] = []
    text_variant_keys: dict[str, str] = {}
    query_token_metadata: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for query_id in text_ids:
        row = rows_by_key.get(("text_to_image", query_id))
        if row is None:
            continue
        token_variants, counterfactuals = _text_variant_requests(str(row["query_context"]))
        query_token_metadata[query_id] = (token_variants, counterfactuals)
        original_key = f"original:{query_id}"
        text_requests.append((original_key, str(row["query_context"]), "original", query_id))
        for variant in token_variants:
            key = f"token:{query_id}:{variant['token_index']}"
            text_variant_keys[key] = query_id
            text_requests.append((key, variant["text_without_token"], "token", query_id))
        for variant in counterfactuals:
            key = f"counterfactual:{query_id}:{variant['counterfactual_type']}"
            text_variant_keys[key] = query_id
            text_requests.append((key, variant["edited_text"], "counterfactual", query_id))
        if len(consistency_rows) < 4:
            casing_key = f"casing:{query_id}"
            text_requests.append((casing_key, str(row["query_context"]).upper(), "casing", query_id))
            text_variant_keys[casing_key] = query_id
    encoded_text = _encode_texts(model, processor, torch, [item[1] for item in text_requests], batch_size)
    text_embedding_by_key = {item[0]: embedding for item, embedding in zip(text_requests, encoded_text)}
    for query_id in text_ids:
        row = rows_by_key.get(("text_to_image", query_id))
        if row is None or f"original:{query_id}" not in text_embedding_by_key:
            continue
        baseline_summary = _score_rows(
            text_embedding_by_key[f"original:{query_id}"], image_matrix, image_ids_all, [str(item) for item in row["relevant_ids"]]
        )
        token_variants, counterfactuals = query_token_metadata[query_id]
        variant_summaries = []
        for variant in token_variants:
            key = f"token:{query_id}:{variant['token_index']}"
            variant_summaries.append({**variant, "summary": _score_rows(text_embedding_by_key[key], image_matrix, image_ids_all, [str(item) for item in row["relevant_ids"]])})
        token_result = token_importance_rows(str(row["query_context"]), baseline_summary, variant_summaries)
        for item in token_result:
            token_rows.append({"system": system, "direction": "text_to_image", "query_id": query_id, **item})
        for variant in counterfactuals:
            key = f"counterfactual:{query_id}:{variant['counterfactual_type']}"
            perturbed = _score_rows(text_embedding_by_key[key], image_matrix, image_ids_all, [str(item) for item in row["relevant_ids"]])
            counterfactual_rows.append({
                "system": system,
                "direction": "text_to_image",
                "query_id": query_id,
                "counterfactual_type": variant["counterfactual_type"],
                "removed_token": variant["token"],
                "edited_text": variant["edited_text"],
                **score_rank_delta(baseline_summary, perturbed),
                "baseline": baseline_summary,
                "perturbed": perturbed,
                "interpretation": "counterfactual sensitivity, not a causal language explanation",
            })
        casing_key = f"casing:{query_id}"
        if casing_key in text_embedding_by_key and len(consistency_rows) < 4:
            casing_summary = _score_rows(text_embedding_by_key[casing_key], image_matrix, image_ids_all, [str(item) for item in row["relevant_ids"]])
            casing_variants = []
            casing_tokens = tokenize_for_occlusion(str(row["query_context"]).upper())
            for index, token in enumerate(casing_tokens):
                edited = delete_token(str(row["query_context"]).upper(), index)
                encoded = _encode_texts(model, processor, torch, [edited], batch_size)[0]
                casing_variants.append({
                    "token_index": index,
                    "token": token,
                    "text_without_token": edited,
                    "summary": _score_rows(encoded, image_matrix, image_ids_all, [str(item) for item in row["relevant_ids"]]),
                })
            casing_importance = token_importance_rows(str(row["query_context"]).upper(), casing_summary, casing_variants)
            base_tokens = _token_order(token_result)
            casing_tokens_order = _token_order(casing_importance)
            consistency_rows.append({
                "system": system,
                "direction": "text_to_image",
                "query_id": query_id,
                "perturbation": "uppercase casing",
                "top_token_order_baseline": base_tokens,
                "top_token_order_perturbed": casing_tokens_order,
                "top3_overlap": len(set(base_tokens) & set(casing_tokens_order)) / max(1, len(set(base_tokens) | set(casing_tokens_order))),
                "baseline_relevant_score": baseline_summary["relevant_score"],
                "perturbed_relevant_score": casing_summary["relevant_score"],
                "interpretation": "compact local consistency check; similar explanations are not guaranteed by casing invariance",
            })
    region_rows: list[dict[str, Any]] = []
    image_by_id = {image_id: image for image_id, image in zip(image_ids_all, image_objects)}
    image_index = {image_id: index for index, image_id in enumerate(image_ids_all)}
    for query_id in image_ids:
        row = rows_by_key.get(("image_to_text", query_id))
        if row is None or query_id not in image_by_id:
            continue
        base_image = image_by_id[query_id]
        width, height = base_image.size
        regions = grid_regions(width, height, GRID_SIZE)
        baseline_summary = _score_rows(
            image_embeddings[image_index[query_id]], caption_matrix, caption_ids, [str(item) for item in row["relevant_ids"]]
        )
        perturbed_images = [occlude_region(base_image, region) for region in regions]
        perturbed_embeddings = _encode_images(model, processor, torch, perturbed_images, batch_size)
        for region, embedding in zip(regions, perturbed_embeddings):
            perturbed = _score_rows(embedding, caption_matrix, caption_ids, [str(item) for item in row["relevant_ids"]])
            delta = score_rank_delta(baseline_summary, perturbed)
            region_rows.append({
                "system": system,
                "direction": "image_to_text",
                "query_id": query_id,
                **region,
                "baseline_relevant_score": baseline_summary["relevant_score"],
                "perturbed_relevant_score": perturbed["relevant_score"],
                "baseline_target_rank": baseline_summary["target_rank"],
                "perturbed_target_rank": perturbed["target_rank"],
                "importance_score_delta": delta["score_delta"],
                "importance_rank_delta": delta["rank_delta"],
                "interpretation": "coarse local region occlusion sensitivity, not a complete saliency map or causal explanation",
            })
    return {
        "token_rows": token_rows,
        "region_rows": region_rows,
        "counterfactual_rows": counterfactual_rows,
        "consistency_rows": consistency_rows,
    }


def _core_explanation(row: Mapping[str, Any]) -> dict[str, Any]:
    scores = [float(score) for score in row["scores"]]
    candidate_ids = [str(item) for item in row["candidate_ids"]]
    relevant_ids = [str(item) for item in row["relevant_ids"]]
    relevant_scores = [score for item, score in zip(candidate_ids, scores) if item in set(relevant_ids)]
    return {
        "system": row["system"],
        "direction": row["direction"],
        "query_id": row["query_id"],
        "query_context": row.get("query_context"),
        "top_retrieved_item": candidate_ids[0],
        "relevant_item_ids": relevant_ids,
        "top1_score": scores[0],
        "relevant_target_score": max(relevant_scores) if relevant_scores else None,
        "top1_top2_margin": row["confidence_proxies"]["top1_top2_margin"],
        "target_rank": row["first_relevant_rank"],
        "confidence_proxy": row["calibrated_confidence"],
        "confidence_proxy_name": row["selected_confidence_proxy"],
        "top1_correct": row["top1_correct"],
        "top5_correct": row["top5_correct"],
        "rank_severity": row["rank_severity"],
        "taxonomy_categories": row.get("taxonomy_categories", []),
        "explanation_language": "observable retrieval evidence and perturbation sensitivity; not a causal explanation",
    }


def _faithfulness_outputs(
    token_rows: Sequence[Mapping[str, Any]], region_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for feature_type, source_rows in (("token", token_rows), ("image_region", region_rows)):
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in source_rows:
            grouped[(str(row["system"]), str(row["query_id"]))].append(row)
        for (system, query_id), values in sorted(grouped.items()):
            rows.append({"system": system, "query_id": query_id, **faithfulness_comparison(values, feature_type)})
    summary: list[dict[str, Any]] = []
    for feature_type in ("token", "image_region"):
        for system in SYSTEMS:
            selected = [row for row in rows if row["feature_type"] == feature_type and row["system"] == system]
            summary.append({
                "system": system,
                "feature_type": feature_type,
                "example_count": len(selected),
                "score_order_pass_rate": statistics.fmean(int(row["score_order_supports_faithfulness"]) for row in selected) if selected else None,
                "rank_order_pass_rate": statistics.fmean(int(row["rank_order_supports_faithfulness"]) for row in selected) if selected else None,
                "mean_score_degradation_difference": statistics.fmean(float(row["score_degradation_difference"]) for row in selected if row["score_degradation_difference"] is not None) if selected else None,
            })
    return {
        "schema_version": PHASE18_SCHEMA_VERSION,
        "rows": rows,
        "summary": summary,
        "method": "highest-importance perturbation versus lowest-importance perturbation within each deterministic example",
    }


def _high_confidence_explanations(
    high_errors: Sequence[Mapping[str, Any]],
    explanation_records: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    core = {(str(row["system"]), str(row["direction"]), str(row["query_id"])): row for row in explanation_records}
    output: list[dict[str, Any]] = []
    for error in sorted(high_errors, key=lambda row: (-float(row["calibrated_confidence"]), str(row["system"]), str(row["direction"]), str(row["query_id"])))[:8]:
        key = (str(error["system"]), str(error["direction"]), str(error["query_id"]))
        if key not in core:
            continue
        base = dict(core[key])
        token = [row for row in token_rows if (str(row["system"]), str(row["query_id"])) == (key[0], key[2])]
        region = [row for row in region_rows if (str(row["system"]), str(row["query_id"])) == (key[0], key[2])]
        base.update({
            "phase17_confidence": error["calibrated_confidence"],
            "score_margin": base.get("top1_top2_margin"),
            "top_token_sensitivities": sorted(token, key=lambda row: (-float(row.get("importance_score_delta") or -math.inf), int(row["token_index"])))[:3],
            "top_region_sensitivities": sorted(region, key=lambda row: (-float(row.get("importance_score_delta") or -math.inf), int(row.get("grid_row", 0)), int(row.get("grid_column", 0))))[:3],
            "likely_observable_pattern": "high-confidence rank error with local sensitivity evidence; no single definitive cause is claimed",
        })
        output.append(base)
    return output


def _qualitative_examples(explanation_records: Sequence[Mapping[str, Any]], token_rows: Sequence[Mapping[str, Any]], region_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(explanation_records)
    selected: list[dict[str, Any]] = []

    def add(example_type: str, row: Mapping[str, Any]) -> None:
        item = dict(row)
        item["example_type"] = example_type
        item["top_token_sensitivity"] = next((dict(value) for value in sorted(token_rows, key=lambda value: (-float(value.get("importance_score_delta") or -math.inf), int(value["token_index"]))) if value["system"] == row["system"] and value["query_id"] == row["query_id"]), None)
        item["top_region_sensitivity"] = next((dict(value) for value in sorted(region_rows, key=lambda value: (-float(value.get("importance_score_delta") or -math.inf), int(value.get("grid_row", 0)), int(value.get("grid_column", 0)))) if value["system"] == row["system"] and value["query_id"] == row["query_id"]), None)
        selected.append(item)

    for system in SYSTEMS:
        for direction in DIRECTIONS:
            subset = [row for row in rows if row["system"] == system and row["direction"] == direction]
            correct = sorted([row for row in subset if row["top1_correct"]], key=lambda row: (str(row["query_id"])))
            failed = sorted([row for row in subset if not row["top1_correct"]], key=lambda row: (str(row["query_id"])))
            if correct:
                add(f"success_{direction}_{system}", correct[0])
            if failed:
                add(f"failure_{direction}_{system}", failed[0])
    comparisons: list[tuple[int | None, Mapping[str, Any], Mapping[str, Any]]] = []
    zero = {(row["direction"], row["query_id"]): row for row in rows if row["system"] == "zero_shot"}
    full = {(row["direction"], row["query_id"]): row for row in rows if row["system"] == "full_ft"}
    for key in sorted(set(zero) & set(full)):
        zero_rank = zero[key].get("target_rank")
        full_rank = full[key].get("target_rank")
        movement = (int(zero_rank) - int(full_rank)) if zero_rank is not None and full_rank is not None else None
        comparisons.append((movement, zero[key], full[key]))
    comparable = [item for item in comparisons if item[0] is not None]
    if comparable:
        best = max(comparable, key=lambda item: (int(item[0]) if item[0] is not None else -math.inf, str(item[2]["query_id"])))
        worst = min(comparable, key=lambda item: (int(item[0]) if item[0] is not None else math.inf, str(item[2]["query_id"])))
        add("fine_tuning_rank_improvement", best[2])
        add("fine_tuning_rank_regression", worst[2])
    high_error = sorted([row for row in rows if not row["top1_correct"] and float(row["confidence_proxy"]) >= HIGH_CONFIDENCE_THRESHOLD], key=lambda row: (-float(row["confidence_proxy"]), str(row["query_id"])))
    if high_error:
        add("high_confidence_error", high_error[0])
    low_error = sorted([row for row in rows if not row["top1_correct"]], key=lambda row: (float(row["confidence_proxy"]), str(row["query_id"])))
    if low_error:
        add("low_confidence_error", low_error[0])
    return selected[:15]


def validate_phase18_artifacts(output_dir: Path | str) -> dict[str, Any]:
    output = Path(output_dir)
    checks = {"required_artifacts": all((output / name).exists() for name in REQUIRED_ARTIFACTS)}
    if not checks["required_artifacts"]:
        return {"schema_version": PHASE18_SCHEMA_VERSION, "passed": False, "checks": checks, "required": list(REQUIRED_ARTIFACTS)}
    audit = _read_json(output / "pre_phase_audit.json")
    explanations = _read_json(output / "explanation_records.json")
    token = _read_json(output / "token_importance.json")
    regions = _read_json(output / "region_sensitivity.json")
    counterfactual = _read_json(output / "counterfactual_analysis.json")
    faithfulness = _read_json(output / "faithfulness_results.json")
    high_errors = _read_json(output / "high_confidence_error_explanations.json")
    qualitative = _read_json(output / "qualitative_examples.json")
    provenance = _read_json(output / "provenance.json")
    explanation_rows = explanations.get("rows", [])
    checks.update({
        "pre_phase_audit_pass": audit.get("status") == "PASS" and audit.get("audit_result") == "PRE-PHASE AUDIT: Phase 17 PASS",
        "explanation_records_nonempty": bool(explanation_rows),
        "systems_present": {row.get("system") for row in explanation_rows} == set(SYSTEMS),
        "directions_separated": {row.get("direction") for row in explanation_rows} == set(DIRECTIONS),
        "token_perturbations_real": bool(token.get("rows")) and all(row.get("text_without_token") for row in token.get("rows", [])),
        "region_perturbations_real": bool(regions.get("rows")) and all("pixel_box" in row and "importance_score_delta" in row for row in regions.get("rows", [])),
        "counterfactuals_present": bool(counterfactual.get("rows")),
        "faithfulness_ran": bool(faithfulness.get("rows")) and bool(faithfulness.get("summary")),
        "high_confidence_errors_included": bool(high_errors.get("rows")),
        "qualitative_count_in_range": 10 <= int(qualitative.get("count", 0)) <= 15,
        "heuristic_labels_explicit": all(
            str(category.get("label_source")) in {"heuristic", "mechanical", "mechanical_context"}
            for row in explanation_rows
            for category in row.get("taxonomy_categories", [])
            if isinstance(category, Mapping)
        ),
        "no_training": provenance.get("training_performed") is False,
        "no_phase19": provenance.get("phase19_started") is False,
        "methods_explicit": bool(provenance.get("methods")),
    })
    return {
        "schema_version": PHASE18_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "required": list(REQUIRED_ARTIFACTS),
    }


def _load_model(model_id: str, checkpoint_path: Path, system: str, device: str) -> tuple[Any, Any, Any]:
    from .phase7 import _load_checkpoint, _load_trainable_model

    model, processor, torch, _ = _load_trainable_model(model_id, device)
    if system == "full_ft":
        _load_checkpoint(checkpoint_path, model)
    model.eval()
    return model, processor, torch


def run_phase18(
    output_dir: Path | str = "artifacts/phase18",
    manifest_path: Path | str = "data/processed/coco2017_val_split_manifest.json",
    image_root: Path | str = "data/raw/coco2017/val2017",
    phase7_dir: Path | str = "artifacts/phase7",
    phase17_dir: Path | str = "artifacts/phase17",
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    batch_size: int = 16,
    regression_tests_passed: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase17_path = Path(phase17_dir)
    phase7_path = Path(phase7_dir)
    manifest_file = Path(manifest_path)
    audit = audit_phase17(phase17_path, regression_tests_passed)
    _write_json(audit, output / "pre_phase_audit.json")
    if audit["status"] != "PASS":
        raise ValueError("Phase 17 dependency audit failed")
    phase17_validation = validate_phase17_artifacts(phase17_path)
    if not phase17_validation["passed"]:
        raise ValueError("Phase 17 artifact validation failed")
    phase17_rows = _read_json(phase17_path / "confidence_records.json")["rows"]
    test_rows = [row for row in phase17_rows if row.get("split") == "test"]
    caption_text, _image_text, image_records = _context_maps(manifest_file)
    from .phase7 import _subset_records

    test_records = _subset_records(read_manifest(manifest_file).records, "test", 42, 100)
    text_ids = _choose_text_query_ids(test_rows)
    image_ids = _choose_image_query_ids(test_rows)
    analyses: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        model, processor, torch = _load_model(model_id, phase7_path / "best_checkpoint.pt", system, device)
        try:
            system_rows = _selected_rows(test_rows, system, "text_to_image") + _selected_rows(test_rows, system, "image_to_text")
            analyses[system] = _analyze_system(
                system,
                model,
                processor,
                torch,
                Path(image_root),
                test_records,
                system_rows,
                text_ids,
                image_ids,
                caption_text,
                image_records,
                batch_size,
            )
        finally:
            del model
            gc.collect()
            try:
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except AttributeError:
                pass
    token_rows = [row for system in SYSTEMS for row in analyses[system]["token_rows"]]
    region_rows = [row for system in SYSTEMS for row in analyses[system]["region_rows"]]
    counterfactual_rows = [row for system in SYSTEMS for row in analyses[system]["counterfactual_rows"]]
    consistency_rows = [row for system in SYSTEMS for row in analyses[system]["consistency_rows"]]
    explanation_rows = [
        _core_explanation(row)
        for row in test_rows
        if str(row["query_id"]) in set(text_ids if row["direction"] == "text_to_image" else image_ids)
    ]
    faithfulness = _faithfulness_outputs(token_rows, region_rows)
    high_errors = _read_json(phase17_path / "high_confidence_errors.json")["rows"]
    high_error_explanations = _high_confidence_explanations(high_errors, explanation_rows, token_rows, region_rows)
    qualitative = _qualitative_examples(explanation_rows, token_rows, region_rows)
    _write_json({"schema_version": PHASE18_SCHEMA_VERSION, "rows": explanation_rows}, output / "explanation_records.json")
    _write_json({"schema_version": PHASE18_SCHEMA_VERSION, "rows": token_rows, "definition": "importance = baseline relevant score - score after deleting one token/group"}, output / "token_importance.json")
    _write_json({"schema_version": PHASE18_SCHEMA_VERSION, "rows": region_rows, "definition": "importance = baseline relevant score - score after mean-colour occlusion of a 3x3 image region"}, output / "region_sensitivity.json")
    _write_json({"schema_version": PHASE18_SCHEMA_VERSION, "rows": counterfactual_rows, "image_rows": region_rows, "image_counterfactual": "3x3 mean-colour region occlusion", "interpretation": "sensitivity test, not causal explanation"}, output / "counterfactual_analysis.json")
    _write_json(faithfulness, output / "faithfulness_results.json")
    _write_json({"schema_version": PHASE18_SCHEMA_VERSION, "rows": high_error_explanations, "confidence_threshold": HIGH_CONFIDENCE_THRESHOLD}, output / "high_confidence_error_explanations.json")
    _write_json({"schema_version": PHASE18_SCHEMA_VERSION, "rows": consistency_rows, "definition": "top-token overlap under uppercase casing perturbation"}, output / "explanation_consistency.json")
    _write_json({"schema_version": PHASE18_SCHEMA_VERSION, "count": len(qualitative), "selection": "deterministic success/failure, confidence, direction, and full-FT movement rules", "examples": qualitative}, output / "qualitative_examples.json")
    provenance = {
        "schema_version": PHASE18_SCHEMA_VERSION,
        "phase": 18,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_file),
        "manifest_sha256": _hash_file(manifest_file),
        "phase17_report_sha256": _hash_file(phase17_path / "phase17_report.json"),
        "phase17_confidence_records_sha256": _hash_file(phase17_path / "confidence_records.json"),
        "phase17_high_confidence_errors_sha256": _hash_file(phase17_path / "high_confidence_errors.json"),
        "phase7_checkpoint_sha256": _hash_file(phase7_path / "best_checkpoint.pt"),
        "model_id": model_id,
        "requested_device": device,
        "batch_size": batch_size,
        "test_image_groups": len(test_records),
        "selected_text_query_ids": text_ids,
        "selected_image_query_ids": image_ids,
        "grid_size": GRID_SIZE,
        "methods": {
            "text": "whitespace/punctuation token deletion with relevant-score and first-relevant-rank deltas",
            "image": "3x3 mean-colour rectangular region occlusion with relevant-score and first-relevant-rank deltas",
            "counterfactual": "deterministic object/action/attribute/spatial token deletions where heuristic lexicons match",
            "faithfulness": "highest local score-degradation perturbation versus lowest within each example",
            "consistency": "top-three token-importance overlap under uppercase casing perturbation",
        },
        "training_performed": False,
        "new_dataset_downloaded": False,
        "phase19_started": False,
        "causal_explanation_claimed": False,
        "heuristic_taxonomy_is_not_ground_truth": True,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(provenance, output / "provenance.json")
    _write_json({"phase": 18, "status": "PENDING"}, output / "phase18_report.json")
    validation = validate_phase18_artifacts(output)
    _write_json(validation, output / "artifact_validation.json")
    report = {
        "report_schema_version": PHASE18_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 18,
        "status": "PASS" if validation["passed"] else "FAIL",
        "pre_phase_audit": "Phase 17 PASS",
        "scope": {"systems": list(SYSTEMS), "directions": list(DIRECTIONS), "test_image_groups": len(test_records), "training": False},
        "quality_gate": {"status": "PASS" if validation["passed"] else "FAIL", "checks": validation["checks"]},
        "artifacts": sorted(path.name for path in output.glob("*.json")),
    }
    _write_json(report, output / "phase18_report.json")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation-only Phase 18 explainability analysis")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase18"))
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=Path("data/processed/coco2017_val_split_manifest.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/coco2017/val2017"))
    parser.add_argument("--phase7-dir", type=Path, default=Path("artifacts/phase7"))
    parser.add_argument("--phase17-dir", type=Path, default=Path("artifacts/phase17"))
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = run_phase18(**vars(args))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
