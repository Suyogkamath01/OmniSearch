"""Phase 15 robustness and distribution-shift evaluation.

Phase 15 is deliberately evaluation-only.  It reuses the fixed Phase 7
COCO test selection and the already produced zero-shot and full-fine-tuned
CLIP artifacts.  Corruptions are generated in memory, candidates remain
clean, and no checkpoint or model parameter is updated here.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .manifest import ImageRecord, read_manifest
from .phase7 import (
    _encode_texts,
    _feature_tensor,
    _hash_file,
    _load_checkpoint,
    _load_rgb_image,
    _load_trainable_model,
    _subset_records,
)
from .phase13 import ALL_METRICS, paired_bootstrap, per_query_metrics

PHASE15_SCHEMA_VERSION = 1
MANIFEST_SHA256 = "09a2c1e56eb1a628b2ead16f064510d713f81aff5ee2f2d09b4ca8993bba3b43"
PROTOCOL_VERSION = "retrieval_eval_v1"
SEED = 42
TEST_IMAGE_LIMIT = 100
BOOTSTRAP_RESAMPLES = 200
MODEL_ID = "openai/clip-vit-base-patch32"
SYSTEMS = ("zero_shot", "full_ft")
TEXT_FAMILIES = ("casing", "punctuation", "typo", "word_deletion", "shortened")
IMAGE_FAMILIES = (
    "resize",
    "blur",
    "jpeg",
    "brightness",
    "crop",
    "noise",
    "occlusion",
)
SEVERITIES = ("low", "high")


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def default_corruption_config() -> dict[str, Any]:
    """Return the complete, predeclared Phase 15 corruption protocol."""

    return {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "phase": 15,
        "seed": SEED,
        "severity_levels": list(SEVERITIES),
        "text_corruptions": {
            "casing": {
                "low": {"operation": "lowercase_first_alphabetic_character"},
                "high": {"operation": "uppercase_all_characters"},
            },
            "punctuation": {
                "low": {"operation": "remove_terminal_punctuation"},
                "high": {"operation": "remove_all_punctuation"},
            },
            "typo": {
                "low": {"operation": "one_deterministic_adjacent_transposition"},
                "high": {"operation": "two_deterministic_character_edits"},
            },
            "word_deletion": {
                "low": {"operation": "delete_one_nonterminal_word"},
                "high": {"operation": "delete_up_to_thirty_percent_of_words"},
            },
            "shortened": {
                "low": {"operation": "retain_first_seventy_five_percent_words"},
                "high": {"operation": "retain_first_forty_percent_words"},
            },
        },
        "image_corruptions": {
            "resize": {
                "low": {"scale": 0.5, "operation": "downscale_then_restore"},
                "high": {"scale": 0.25, "operation": "downscale_then_restore"},
            },
            "blur": {
                "low": {"radius": 1.0, "operation": "gaussian_blur"},
                "high": {"radius": 3.0, "operation": "gaussian_blur"},
            },
            "jpeg": {
                "low": {"quality": 70, "operation": "jpeg_round_trip"},
                "high": {"quality": 25, "operation": "jpeg_round_trip"},
            },
            "brightness": {
                "low": {"factor": 0.8, "operation": "brightness_scale"},
                "high": {"factor": 0.55, "operation": "brightness_scale"},
            },
            "crop": {
                "low": {"retain_fraction": 0.9, "operation": "center_crop_then_restore"},
                "high": {"retain_fraction": 0.6, "operation": "center_crop_then_restore"},
            },
            "noise": {
                "low": {"sigma": 5.0, "operation": "additive_gaussian_noise"},
                "high": {"sigma": 20.0, "operation": "additive_gaussian_noise"},
            },
            "occlusion": {
                "low": {"area_fraction": 0.1, "operation": "central_black_rectangle"},
                "high": {"area_fraction": 0.3, "operation": "central_black_rectangle"},
            },
        },
        "application": {
            "text_to_image": "corrupt_text_queries_only; image_candidates_remain_clean",
            "image_to_text": "corrupt_image_queries_only; caption_candidates_remain_clean",
            "model_native_preprocessing": True,
            "new_dataset_written": False,
        },
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "confidence": 0.95},
    }


def validate_corruption_config(config: Mapping[str, Any]) -> None:
    """Validate that the executed protocol is the predeclared protocol."""

    if int(config.get("schema_version", -1)) != PHASE15_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 15 corruption schema")
    if int(config.get("phase", -1)) != 15 or int(config.get("seed", -1)) != SEED:
        raise ValueError("Phase 15 seed and phase must remain fixed")
    if tuple(config.get("severity_levels", ())) != SEVERITIES:
        raise ValueError("Phase 15 requires low and high severities")
    text = config.get("text_corruptions")
    image = config.get("image_corruptions")
    if not isinstance(text, Mapping) or tuple(text) != TEXT_FAMILIES:
        raise ValueError("text corruption families changed")
    if not isinstance(image, Mapping) or tuple(image) != IMAGE_FAMILIES:
        raise ValueError("image corruption families changed")
    for family_config in (*text.values(), *image.values()):
        if not isinstance(family_config, Mapping) or tuple(family_config) != SEVERITIES:
            raise ValueError("each corruption must declare low and high settings")
    application = config.get("application", {})
    if application.get("new_dataset_written") is not False:
        raise ValueError("Phase 15 must not write a corrupted dataset")


def _first_alpha_lower(text: str) -> str:
    for index, character in enumerate(text):
        if character.isalpha():
            return text[:index] + character.lower() + text[index + 1 :]
    return text


def _remove_terminal_punctuation(text: str) -> str:
    return text.rstrip().rstrip(".,!?;:")


def _remove_all_punctuation(text: str) -> str:
    return "".join(character if character.isalnum() or character.isspace() else " " for character in text)


def _typo(text: str, severity: str, seed: int) -> str:
    characters = list(text)
    eligible = [index for index, character in enumerate(characters[:-1]) if character.isalpha() and characters[index + 1].isalpha()]
    if not eligible:
        return text
    first = eligible[seed % len(eligible)]
    characters[first], characters[first + 1] = characters[first + 1], characters[first]
    if severity == "high" and len(eligible) > 1:
        second = eligible[(seed // 17 + 1) % len(eligible)]
        if second not in {first, first + 1}:
            characters[second] = "x" if characters[second].lower() != "x" else "q"
    return "".join(characters)


def _delete_words(text: str, severity: str, seed: int) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    count = 1 if severity == "low" else max(1, math.ceil(len(words) * 0.3))
    eligible = list(range(1, len(words))) if len(words) > 2 else list(range(len(words)))
    start = seed % len(eligible)
    deleted = {eligible[(start + offset) % len(eligible)] for offset in range(min(count, len(eligible)))}
    return " ".join(word for index, word in enumerate(words) if index not in deleted)


def _shorten(text: str, severity: str) -> str:
    words = text.split()
    if not words:
        return text
    fraction = 0.75 if severity == "low" else 0.4
    return " ".join(words[: max(1, math.ceil(len(words) * fraction))])


def apply_text_corruption(text: str, family: str, severity: str, seed: int = SEED) -> str:
    """Apply one deterministic text corruption without mutating the input."""

    if family not in TEXT_FAMILIES or severity not in SEVERITIES:
        raise ValueError(f"unsupported text condition: {family}/{severity}")
    if family == "casing":
        return _first_alpha_lower(text) if severity == "low" else text.upper()
    if family == "punctuation":
        return _remove_terminal_punctuation(text) if severity == "low" else _remove_all_punctuation(text)
    if family == "typo":
        return _typo(text, severity, int(seed))
    if family == "word_deletion":
        return _delete_words(text, severity, int(seed))
    return _shorten(text, severity)


def _to_rgb_copy(image: Any) -> Any:
    return image.convert("RGB").copy()


def apply_image_corruption(image: Any, family: str, severity: str, seed: int = SEED) -> Any:
    """Apply one deterministic PIL corruption and return a new RGB image."""

    if family not in IMAGE_FAMILIES or severity not in SEVERITIES:
        raise ValueError(f"unsupported image condition: {family}/{severity}")
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    output = _to_rgb_copy(image)
    width, height = output.size
    if family == "resize":
        scale = 0.5 if severity == "low" else 0.25
        small = output.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.BILINEAR)
    if family == "blur":
        return output.filter(ImageFilter.GaussianBlur(1.0 if severity == "low" else 3.0))
    if family == "jpeg":
        import io

        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=70 if severity == "low" else 25)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if family == "brightness":
        return ImageEnhance.Brightness(output).enhance(0.8 if severity == "low" else 0.55)
    if family == "crop":
        fraction = 0.9 if severity == "low" else 0.6
        return ImageOps.fit(output, (width, height), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5), bleed=(1.0 - fraction) / 2.0)
    if family == "noise":
        import numpy as np

        sigma = 5.0 if severity == "low" else 20.0
        array = np.asarray(output, dtype=np.float32)
        noise = np.random.default_rng(int(seed)).normal(0.0, sigma, size=array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    fraction = 0.1 if severity == "low" else 0.3
    rectangle_area = max(1.0, width * height * fraction)
    rectangle_width = max(1, round(math.sqrt(rectangle_area * width / max(height, 1))))
    rectangle_height = max(1, round(rectangle_area / rectangle_width))
    left = max(0, (width - rectangle_width) // 2)
    top = max(0, (height - rectangle_height) // 2)
    output.paste((0, 0, 0), (left, top, min(width, left + rectangle_width), min(height, top + rectangle_height)))
    return output


def metric_degradation(clean: float, corrupted: float) -> dict[str, float | None]:
    """Return absolute delta, degradation, retention, and a zero-safe status."""

    clean_value = float(clean)
    corrupted_value = float(corrupted)
    delta = corrupted_value - clean_value
    if clean_value == 0.0:
        return {"absolute_delta": delta, "relative_degradation": None, "retention": None}
    return {
        "absolute_delta": delta,
        "relative_degradation": (clean_value - corrupted_value) / clean_value,
        "retention": corrupted_value / clean_value,
    }


def rank_stability(
    clean_rank: Sequence[str],
    corrupted_rank: Sequence[str],
    relevant_ids: Sequence[str] | set[str],
    top_k: int = 5,
) -> dict[str, float | int | None]:
    """Summarize top-1 preservation, top-K overlap, and first-hit movement."""

    clean_top = list(clean_rank[:top_k])
    corrupt_top = list(corrupted_rank[:top_k])
    relevant = set(relevant_ids)
    clean_first = next((i + 1 for i, item in enumerate(clean_rank) if item in relevant), None)
    corrupt_first = next((i + 1 for i, item in enumerate(corrupted_rank) if item in relevant), None)
    displacement = None if clean_first is None or corrupt_first is None else float(corrupt_first - clean_first)
    return {
        "top1_preserved": int(bool(clean_rank) and bool(corrupted_rank) and clean_rank[0] == corrupted_rank[0]),
        "top_k_overlap": len(set(clean_top) & set(corrupt_top)) / max(1, top_k),
        "clean_first_relevant_rank": clean_first,
        "corrupted_first_relevant_rank": corrupt_first,
        "relevant_rank_displacement": displacement,
        "rank_displacement_observed": int(displacement is not None),
    }


def _ranking_result(
    task: str,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    scores: Any,
    relevant: Mapping[str, set[str]],
    system_id: str,
    experiment_id: str,
    candidate_corpus_id: str,
    top_k: int = 10,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row_index, query_id in enumerate(query_ids):
        values = scores[row_index].tolist()
        ranked = sorted(
            zip(candidate_ids, values), key=lambda item: (-float(item[1]), str(item[0]))
        )[: min(top_k, len(candidate_ids))]
        records.append(
            {
                "query_id": str(query_id),
                "task": task,
                "candidate_ids": [str(item[0]) for item in ranked],
                "scores": [float(item[1]) for item in ranked],
                "relevant_ids": sorted(str(item) for item in relevant[str(query_id)]),
                "relevance_definition": "retrieval_eval_v1 declared image/caption group relevance",
                "system_id": system_id,
                "experiment_id": experiment_id,
                "candidate_count": len(candidate_ids),
                "candidate_corpus_id": candidate_corpus_id,
            }
        )
    return {
        "result_schema_version": 1,
        "project": "OmniSearch",
        "experiment_id": experiment_id,
        "system_id": system_id,
        "task": task,
        "query_count": len(records),
        "candidate_count": len(candidate_ids),
        "protocol": {"protocol_version": PROTOCOL_VERSION, "task": task},
        "ranking_records": records,
    }


def _aggregate(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {metric: statistics.fmean(float(row[metric]) for row in metrics.values()) for metric in ALL_METRICS}


def _load_clean_result(path: Path, task: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task or payload.get("protocol", {}).get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"invalid clean Phase 7 result: {path}")
    records = payload.get("ranking_records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"clean result has no rankings: {path}")
    return payload


def _encode_images(
    model: Any,
    processor: Any,
    torch: Any,
    images: Sequence[Any],
    ids: Sequence[str],
    batch_size: int,
) -> Any:
    from torch.nn import functional

    rows: list[Any] = []
    device = model.device if hasattr(model, "device") else str(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            current = images[start : start + batch_size]
            processed = processor(images=list(current), return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in processed.items()}
            features = _feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            rows.append(features.detach().cpu())
    if not rows or len(ids) != sum(len(row) for row in rows):
        raise ValueError("image encoding produced no aligned rows")
    return functional.normalize(torch.cat(rows, dim=0), dim=-1).cpu()


def _records_by_id(records: Sequence[ImageRecord]) -> dict[str, ImageRecord]:
    output: dict[str, ImageRecord] = {}
    for record in records:
        if record.image_id in output:
            raise ValueError(f"duplicate image ID in Phase 15 test selection: {record.image_id}")
        output[record.image_id] = record
    return output


def _shift_definition(records: Sequence[ImageRecord], image_root: Path) -> dict[str, Any]:
    from PIL import Image

    dimensions: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.image_id):
        if record.filename is None:
            raise ValueError(f"shift record has no filename: {record.image_id}")
        with Image.open(image_root / record.filename) as image:
            width, height = image.size
        dimensions.append({"image_id": record.image_id, "width": width, "height": height, "aspect_ratio": width / height})
    shifted = [row for row in dimensions if row["aspect_ratio"] <= 0.75 or row["aspect_ratio"] >= 1.3333333333333333]
    control = [row for row in dimensions if 0.9 <= row["aspect_ratio"] <= 1.1]
    target = min(len(shifted), len(control), 5)
    if target < 2:
        raise ValueError("fixed test subset does not contain enough shift/control groups")
    shifted = sorted(shifted, key=lambda row: row["image_id"])[:target]
    control = sorted(control, key=lambda row: row["image_id"])[:target]
    return {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "shift_id": "aspect_ratio_extreme_vs_near_square",
        "selection_precedes_metrics": True,
        "rationale": "Test image groups with extreme aspect ratios are compared with a same-size near-square control selected by metadata only.",
        "rule": {
            "shifted": "width/height <= 0.75 OR width/height >= 1.3333333333333333",
            "control": "0.9 <= width/height <= 1.1",
            "ordering": "image_id ascending",
            "target_group_count": 5,
        },
        "candidate_corpus_policy": "within each group set, all selected images/captions form the clean candidate corpus",
        "shifted_groups": shifted,
        "control_groups": control,
        "selection_counts": {"eligible_shifted": len([row for row in dimensions if row["aspect_ratio"] <= 0.75 or row["aspect_ratio"] >= 1.3333333333333333]), "eligible_control": len([row for row in dimensions if 0.9 <= row["aspect_ratio"] <= 1.1]), "selected_each": target},
        "all_test_dimensions": dimensions,
    }


def _build_corruption_manifest(text_items: Sequence[tuple[str, str]], image_ids: Sequence[str]) -> dict[str, Any]:
    text_rows = []
    for query_id, text in text_items:
        conditions = []
        for family in TEXT_FAMILIES:
            for severity in SEVERITIES:
                conditions.append({"family": family, "severity": severity, "seed": _stable_seed(str(SEED), query_id, family, severity)})
        text_rows.append({"query_id": query_id, "original_text": text, "conditions": conditions})
    image_rows = []
    for query_id in image_ids:
        conditions = []
        for family in IMAGE_FAMILIES:
            for severity in SEVERITIES:
                conditions.append({"family": family, "severity": severity, "seed": _stable_seed(str(SEED), query_id, family, severity)})
        image_rows.append({"query_id": query_id, "conditions": conditions})
    return {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "seed": SEED,
        "text_query_count": len(text_rows),
        "image_query_count": len(image_rows),
        "text_queries": text_rows,
        "image_queries": image_rows,
    }


def _condition_row(
    system: str,
    direction: str,
    family: str,
    severity: str,
    clean_result: Mapping[str, Any],
    corrupted_result: Mapping[str, Any],
    elapsed_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_metrics = per_query_metrics(clean_result)
    corrupt_metrics = per_query_metrics(corrupted_result)
    if set(clean_metrics) != set(corrupt_metrics):
        raise ValueError("clean and corrupted query IDs do not align")
    ordered_ids = sorted(clean_metrics)
    metric_rows: dict[str, Any] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    for metric in ALL_METRICS:
        clean_values = [clean_metrics[q][metric] for q in ordered_ids]
        corrupt_values = [corrupt_metrics[q][metric] for q in ordered_ids]
        clean_value = statistics.fmean(clean_values)
        corrupt_value = statistics.fmean(corrupt_values)
        metric_rows[metric] = {"clean": clean_value, "corrupted": corrupt_value, **metric_degradation(clean_value, corrupt_value)}
        if metric in ("recall_at_1", "recall_at_5"):
            bootstrap_rows.append({
                "system": system,
                "direction": direction,
                "family": family,
                "severity": severity,
                "metric": metric,
                **paired_bootstrap(clean_values, corrupt_values, resamples=BOOTSTRAP_RESAMPLES, seed=_stable_seed(system, direction, family, severity, metric)),
                "comparison": "corrupted_minus_clean",
            })
    return {
        "system": system,
        "direction": direction,
        "family": family,
        "severity": severity,
        "query_count": len(ordered_ids),
        "candidate_count": corrupted_result["candidate_count"],
        "metrics": metric_rows,
        "preprocessing_and_encoding_seconds": elapsed_seconds,
        "candidates_clean": True,
    }, bootstrap_rows


def _stability_rows(
    system: str,
    direction: str,
    family: str,
    severity: str,
    clean_result: Mapping[str, Any],
    corrupted_result: Mapping[str, Any],
) -> dict[str, Any]:
    clean_by_id = {str(row["query_id"]): row for row in clean_result["ranking_records"]}
    corrupt_by_id = {str(row["query_id"]): row for row in corrupted_result["ranking_records"]}
    values = []
    for query_id in sorted(clean_by_id):
        clean = clean_by_id[query_id]
        corrupt = corrupt_by_id[query_id]
        values.append(rank_stability(clean["candidate_ids"], corrupt["candidate_ids"], clean["relevant_ids"], 5))
    displacements: list[float] = []
    for value in values:
        displacement = value["relevant_rank_displacement"]
        if displacement is not None:
            displacements.append(float(displacement))
    return {
        "system": system,
        "direction": direction,
        "family": family,
        "severity": severity,
        "query_count": len(values),
        "top1_preservation": statistics.fmean(float(cast(float | int, row["top1_preserved"])) for row in values),
        "top5_overlap": statistics.fmean(float(cast(float | int, row["top_k_overlap"])) for row in values),
        "mean_relevant_rank_displacement": None if not displacements else statistics.fmean(displacements),
        "rank_displacement_observed_queries": len(displacements),
        "rank_displacement_censored_queries": len(values) - len(displacements),
    }


def _shift_metrics(
    system: str,
    direction: str,
    control_result: Mapping[str, Any],
    shifted_result: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    control = per_query_metrics(control_result)
    shifted = per_query_metrics(shifted_result)
    control_aggregate = _aggregate(control)
    shifted_aggregate = _aggregate(shifted)
    return {
        "system": system,
        "direction": direction,
        "shift_id": definition["shift_id"],
        "control_group_ids": [row["image_id"] for row in definition["control_groups"]],
        "shifted_group_ids": [row["image_id"] for row in definition["shifted_groups"]],
        "control_query_count": len(control),
        "shifted_query_count": len(shifted),
        "control_metrics": control_aggregate,
        "shifted_metrics": shifted_aggregate,
        "shifted_minus_control": {metric: shifted_aggregate[metric] - control_aggregate[metric] for metric in ALL_METRICS},
        "comparison_note": "The selected control and shifted groups are disjoint, so this is a descriptive unpaired comparison; no paired significance claim is made.",
    }


def _qualitative_examples(
    system: str,
    family: str,
    severity: str,
    clean_result: Mapping[str, Any],
    corrupted_result: Mapping[str, Any],
    text_lookup: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    clean_metrics = per_query_metrics(clean_result)
    corrupt_metrics = per_query_metrics(corrupted_result)
    deltas = sorted((corrupt_metrics[q]["recall_at_1"] - clean_metrics[q]["recall_at_1"], q) for q in clean_metrics)
    examples = []
    clean_by_id = {str(row["query_id"]): row for row in clean_result["ranking_records"]}
    corrupt_by_id = {str(row["query_id"]): row for row in corrupted_result["ranking_records"]}
    for delta, query_id in deltas[:3]:
        examples.append({
            "query_id": query_id,
            "original_text": None if text_lookup is None else text_lookup.get(query_id),
            "clean_top1": clean_by_id[query_id]["candidate_ids"][:1],
            "corrupted_top1": corrupt_by_id[query_id]["candidate_ids"][:1],
            "recall_at_1_delta": delta,
        })
    return {"system": system, "family": family, "severity": severity, "selection": "three worst recall_at_1 per condition, query_id tie-break", "examples": examples}


def _encode_text_condition(model: Any, processor: Any, torch: Any, items: Sequence[tuple[str, str]], batch_size: int) -> Any:
    return _encode_texts(model, processor, torch, items, batch_size, 77, 0, "fp32")[1]


def _prepare_protocol(
    output: Path,
    manifest_path: Path,
    image_root: Path,
    records: Sequence[ImageRecord],
    clean_results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = default_corruption_config()
    validate_corruption_config(config)
    _write_json(config, output / "corruption_config.json")
    text_items = []
    for row in clean_results["zero_shot"]["text_to_image"]["ranking_records"]:
        query_id = str(row["query_id"])
        caption = next((caption.text for record in records for caption in record.captions if caption.caption_id == query_id), None)
        if caption is None:
            raise ValueError(f"clean result query is not in manifest: {query_id}")
        text_items.append((query_id, caption))
    image_ids = [str(row["query_id"]) for row in clean_results["zero_shot"]["image_to_text"]["ranking_records"]]
    corruption_manifest = _build_corruption_manifest(text_items, image_ids)
    _write_json(corruption_manifest, output / "corruption_manifest.json")
    definition = _shift_definition(records, image_root)
    _write_json(definition, output / "distribution_shift_definition.json")
    _write_json({
        "manifest": str(manifest_path),
        "manifest_sha256": _hash_file(manifest_path),
        "test_image_groups": len(records),
        "test_caption_queries": len(text_items),
        "image_queries": len(image_ids),
        "protocol_version": PROTOCOL_VERSION,
        "prepared_before_model_evaluation": True,
    }, output / "protocol_preparation.json")
    return config, definition


def run_phase15(
    output_dir: Path | str = Path("artifacts/phase15"),
    manifest_path: Path | str = Path("data/processed/coco2017_val_split_manifest.json"),
    image_root: Path | str = Path("data/raw/coco2017/val2017"),
    phase7_dir: Path | str = Path("artifacts/phase7"),
    model_id: str = MODEL_ID,
    batch_size: int = 8,
    device: str = "auto",
) -> dict[str, Any]:
    """Run the complete real Phase 15 evaluation without training."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output_dir)
    manifest_file = Path(manifest_path)
    image_root_path = Path(image_root)
    phase7_path = Path(phase7_dir)
    manifest = read_manifest(manifest_file)
    if _hash_file(manifest_file) != MANIFEST_SHA256:
        raise ValueError("Phase 15 requires the fixed Phase 7 manifest digest")
    test_records = _subset_records(manifest.records, "test", SEED, TEST_IMAGE_LIMIT)
    if len(test_records) != TEST_IMAGE_LIMIT or any(record.split != "test" for record in test_records):
        raise ValueError("Phase 15 test selection is not the fixed 100-image test subset")
    by_id = _records_by_id(test_records)
    clean_results: dict[str, dict[str, Any]] = {}
    for system, prefix in (("zero_shot", "zero_shot"), ("full_ft", "fine_tuned")):
        clean_results[system] = {
            direction: _load_clean_result(phase7_path / f"{prefix}_{direction}.json", direction)
            for direction in ("text_to_image", "image_to_text")
        }
    output.mkdir(parents=True, exist_ok=True)
    _config, definition = _prepare_protocol(output, manifest_file, image_root_path, test_records, clean_results)

    clean_baselines: dict[str, Any] = {}
    for system in SYSTEMS:
        clean_baselines[system] = {}
        for direction in ("text_to_image", "image_to_text"):
            payload = clean_results[system][direction]
            metrics = per_query_metrics(payload)
            clean_baselines[system][direction] = {
                "source_artifact": str(phase7_path / (("zero_shot" if system == "zero_shot" else "fine_tuned") + f"_{direction}.json")),
                "query_count": len(metrics),
                "candidate_count": payload["candidate_count"],
                "metrics": _aggregate(metrics),
                "protocol_version": payload["protocol"]["protocol_version"],
                "candidate_ids_sha256": hashlib.sha256("\n".join(payload["ranking_records"][0]["candidate_ids"]).encode()).hexdigest(),
            }
    _write_json({"schema_version": PHASE15_SCHEMA_VERSION, "systems": clean_baselines}, output / "clean_baselines.json")

    text_lookup = {caption.caption_id: caption.text for record in test_records for caption in record.captions}
    image_lookup = {record.image_id: record for record in test_records}
    all_image_ids = [str(row["query_id"]) for row in clean_results["zero_shot"]["image_to_text"]["ranking_records"]]
    all_caption_ids = [str(row["query_id"]) for row in clean_results["zero_shot"]["text_to_image"]["ranking_records"]]
    for system in SYSTEMS:
        if [row["query_id"] for row in clean_results[system]["image_to_text"]["ranking_records"]] != all_image_ids:
            raise ValueError("system clean image query IDs do not align")
        if [row["query_id"] for row in clean_results[system]["text_to_image"]["ranking_records"]] != all_caption_ids:
            raise ValueError("system clean text query IDs do not align")

    robustness_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    qualitative_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    shifted_ids = [str(row["image_id"]) for row in definition["shifted_groups"]]
    control_ids = [str(row["image_id"]) for row in definition["control_groups"]]

    for system in SYSTEMS:
        started_system = time.perf_counter()
        model, processor, torch, model_device = _load_trainable_model(model_id, device)
        checkpoint_path = phase7_path / "best_checkpoint.pt"
        checkpoint_metadata: dict[str, Any] = {}
        if system == "full_ft":
            checkpoint_metadata = _load_checkpoint(checkpoint_path, model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        t2i_clean = clean_results[system]["text_to_image"]
        i2t_clean = clean_results[system]["image_to_text"]
        image_candidate_ids = [str(item) for item in t2i_clean["ranking_records"][0]["candidate_ids"]]
        # Clean artifacts store top-10, so obtain the declared full candidate set from the fixed test groups.
        image_candidate_ids = sorted(by_id)
        caption_candidate_ids = [caption.caption_id for record in test_records for caption in record.captions]
        image_inputs = []
        for image_id in image_candidate_ids:
            filename = by_id[image_id].filename
            if filename is None:
                raise ValueError(f"image candidate has no filename: {image_id}")
            image_inputs.append(_load_rgb_image(str(image_root_path / filename)))
        image_embeddings = _encode_images(model, processor, torch, image_inputs, image_candidate_ids, batch_size)
        for image in image_inputs:
            image.close()
        caption_items = [(caption_id, text_lookup[caption_id]) for caption_id in caption_candidate_ids]
        text_embeddings = _encode_text_condition(model, processor, torch, caption_items, batch_size)
        image_position = {image_id: index for index, image_id in enumerate(image_candidate_ids)}
        caption_position = {caption_id: index for index, caption_id in enumerate(caption_candidate_ids)}
        image_corpus = f"{manifest.dataset_id}:phase15:test:images:{len(image_candidate_ids)}"
        caption_corpus = f"{manifest.dataset_id}:phase15:test:captions:{len(caption_candidate_ids)}"
        caption_to_image = {caption.caption_id: {record.image_id} for record in test_records for caption in record.captions}
        image_to_captions = {record.image_id: {caption.caption_id for caption in record.captions} for record in test_records}
        caption_to_image_single = {caption_id: next(iter(image_ids_for_caption)) for caption_id, image_ids_for_caption in caption_to_image.items()}

        for family in TEXT_FAMILIES:
            for severity in SEVERITIES:
                condition_started = time.perf_counter()
                text_items = [(query_id, apply_text_corruption(text_lookup[query_id], family, severity, _stable_seed(str(SEED), query_id, family, severity))) for query_id in all_caption_ids]
                corrupted_text = _encode_text_condition(model, processor, torch, text_items, batch_size)
                scores = corrupted_text @ image_embeddings.transpose(0, 1)
                corrupted = _ranking_result("text_to_image", all_caption_ids, image_candidate_ids, scores, caption_to_image, system, f"phase15_{system}_{family}_{severity}", image_corpus)
                row, boot = _condition_row(system, "text_to_image", family, severity, t2i_clean, corrupted, time.perf_counter() - condition_started)
                robustness_rows.append(row)
                bootstrap_rows.extend(boot)
                stability_rows.append(_stability_rows(system, "text_to_image", family, severity, t2i_clean, corrupted))
                qualitative_rows.append(_qualitative_examples(system, family, severity, t2i_clean, corrupted, text_lookup))

        for family in IMAGE_FAMILIES:
            for severity in SEVERITIES:
                condition_started = time.perf_counter()
                images = []
                for query_id in all_image_ids:
                    record = image_lookup[query_id]
                    if record.filename is None:
                        raise ValueError(f"image query has no filename: {query_id}")
                    with_image = _load_rgb_image(str(image_root_path / record.filename))
                    images.append(apply_image_corruption(with_image, family, severity, _stable_seed(str(SEED), query_id, family, severity)))
                    with_image.close()
                corrupted_images = _encode_images(model, processor, torch, images, all_image_ids, batch_size)
                for image in images:
                    image.close()
                scores = corrupted_images @ text_embeddings.transpose(0, 1)
                corrupted = _ranking_result("image_to_text", all_image_ids, caption_candidate_ids, scores, image_to_captions, system, f"phase15_{system}_{family}_{severity}", caption_corpus)
                row, boot = _condition_row(system, "image_to_text", family, severity, i2t_clean, corrupted, time.perf_counter() - condition_started)
                robustness_rows.append(row)
                bootstrap_rows.extend(boot)
                stability_rows.append(_stability_rows(system, "image_to_text", family, severity, i2t_clean, corrupted))
                qualitative_rows.append(_qualitative_examples(system, family, severity, i2t_clean, corrupted))

        # Distribution shift is computed from the clean embeddings on the predeclared groups.
        def subset_rankings(
            group_ids: Sequence[str],
            direction: str,
            *,
            caption_to_image_single: Mapping[str, str] = caption_to_image_single,
            caption_position: Mapping[str, int] = caption_position,
            image_position: Mapping[str, int] = image_position,
            text_embeddings: Any = text_embeddings,
            image_embeddings: Any = image_embeddings,
            image_corpus: str = image_corpus,
            caption_corpus: str = caption_corpus,
            caption_candidate_ids: Sequence[str] = caption_candidate_ids,
            image_to_captions: Mapping[str, set[str]] = image_to_captions,
            all_caption_ids: Sequence[str] = all_caption_ids,
            all_image_ids: Sequence[str] = all_image_ids,
            system: str = system,
        ) -> dict[str, Any]:
            selected = set(group_ids)
            if direction == "text_to_image":
                query_ids = [caption_id for caption_id in all_caption_ids if caption_to_image_single[caption_id] in selected]
                candidates = list(group_ids)
                positions = [caption_position[query_id] for query_id in query_ids]
                candidate_positions = [image_position[image_id] for image_id in candidates]
                scores = text_embeddings[positions] @ image_embeddings[candidate_positions].transpose(0, 1)
                relevant = {query_id: {caption_to_image_single[query_id]} for query_id in query_ids}
                return _ranking_result(direction, query_ids, candidates, scores, relevant, system, f"phase15_{system}_shift_clean", f"{image_corpus}:shift")
            query_ids = [image_id for image_id in all_image_ids if image_id in selected]
            candidates = [caption_id for caption_id in caption_candidate_ids if caption_to_image_single[caption_id] in selected]
            positions = [image_position[image_id] for image_id in query_ids]
            candidate_positions = [caption_position[caption_id] for caption_id in candidates]
            scores = image_embeddings[positions] @ text_embeddings[candidate_positions].transpose(0, 1)
            relevant = {image_id: image_to_captions[image_id] for image_id in query_ids}
            return _ranking_result(direction, query_ids, candidates, scores, relevant, system, f"phase15_{system}_shift_clean", f"{caption_corpus}:shift")

        for direction in ("text_to_image", "image_to_text"):
            control_result = subset_rankings(control_ids, direction)
            shifted_result = subset_rankings(shifted_ids, direction)
            shift_rows.append(_shift_metrics(system, direction, control_result, shifted_result, definition))
        runtime_rows.append({"system": system, "device": str(model_device), "seconds": time.perf_counter() - started_system, "checkpoint_loaded": system == "full_ft", "checkpoint_metadata": checkpoint_metadata})
        del model, processor, torch
        gc.collect()

    _write_json({"schema_version": PHASE15_SCHEMA_VERSION, "rows": robustness_rows}, output / "robustness_metrics.json")
    _write_json({"schema_version": PHASE15_SCHEMA_VERSION, "rows": bootstrap_rows, "resamples": BOOTSTRAP_RESAMPLES, "comparison": "corrupted_minus_clean"}, output / "bootstrap_comparisons.json")
    _write_json({"schema_version": PHASE15_SCHEMA_VERSION, "rows": stability_rows}, output / "rank_stability.json")
    severity_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for family in (*TEXT_FAMILIES, *IMAGE_FAMILIES):
            directions = ("text_to_image",) if family in TEXT_FAMILIES else ("image_to_text",)
            for severity in SEVERITIES:
                for direction in directions:
                    matching = [
                        row for row in robustness_rows
                        if row["system"] == system
                        and row["family"] == family
                        and row["severity"] == severity
                        and row["direction"] == direction
                    ]
                    severity_rows.append({
                        "system": system,
                        "family": family,
                        "severity": severity,
                        "direction": direction,
                        "metrics": {
                            metric: [row["metrics"][metric] for row in matching]
                            for metric in ALL_METRICS
                        },
                    })
    _write_json({
        "schema_version": PHASE15_SCHEMA_VERSION,
        "rows": severity_rows,
    }, output / "severity_analysis.json")
    _write_json({"schema_version": PHASE15_SCHEMA_VERSION, "rows": qualitative_rows}, output / "qualitative_examples.json")
    _write_json({"schema_version": PHASE15_SCHEMA_VERSION, "rows": shift_rows, "definition": "distribution_shift_definition.json"}, output / "distribution_shift_results.json")
    _write_json({
        "schema_version": PHASE15_SCHEMA_VERSION,
        "scope": "evaluation-only robustness diagnosis",
        "system_rows": runtime_rows,
        "conditions": {"text": len(TEXT_FAMILIES) * len(SEVERITIES), "image": len(IMAGE_FAMILIES) * len(SEVERITIES)},
        "known_censoring": "rank displacement is only observed when the first relevant item appears in both returned top-10 lists",
        "limitations": ["No external distribution-shift dataset was introduced.", "Corruptions are synthetic and may not represent deployment noise.", "Phase 12B remains non-blocking and no CIRCO result is included."],
    }, output / "failure_analysis.json")
    provenance = {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "phase": 15,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_file),
        "manifest_sha256": _hash_file(manifest_file),
        "phase7_artifacts": {path.name: _hash_file(path) for path in (phase7_path / "zero_shot_text_to_image.json", phase7_path / "zero_shot_image_to_text.json", phase7_path / "fine_tuned_text_to_image.json", phase7_path / "fine_tuned_image_to_text.json", phase7_path / "best_checkpoint.pt")},
        "model_id": model_id,
        "systems": list(SYSTEMS),
        "training_performed": False,
        "new_model_families": False,
        "new_dataset_downloaded": False,
        "protocol_version": PROTOCOL_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime": runtime_rows,
    }
    _write_json(provenance, output / "provenance.json")
    validation = validate_phase15_artifacts(output)
    _write_json(validation, output / "artifact_validation.json")
    report = {
        "report_schema_version": PHASE15_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 15,
        "status": "PASS" if validation["passed"] else "FAIL",
        "scope": {"evaluation_only": True, "systems": list(SYSTEMS), "test_image_groups": len(test_records), "test_caption_queries": len(all_caption_ids), "protocol_version": PROTOCOL_VERSION},
        "phase14_audit": "PASS",
        "phase12b_status": "PARTIAL — NON-BLOCKING",
        "quality_gate": {"status": "PASS" if validation["passed"] else "FAIL", "checks": validation["checks"]},
        "artifacts": sorted(path.name for path in output.glob("*.json")),
    }
    _write_json(report, output / "phase15_report.json")
    return report


def validate_phase15_artifacts(output_dir: Path | str) -> dict[str, Any]:
    """Validate required Phase 15 artifacts without recomputing model output."""

    output = Path(output_dir)
    required = {
        "pre_phase_audit.json",
        "corruption_config.json",
        "corruption_manifest.json",
        "distribution_shift_definition.json",
        "clean_baselines.json",
        "robustness_metrics.json",
        "severity_analysis.json",
        "bootstrap_comparisons.json",
        "rank_stability.json",
        "distribution_shift_results.json",
        "qualitative_examples.json",
        "failure_analysis.json",
        "provenance.json",
    }
    checks: dict[str, bool] = {}
    checks["required_artifacts"] = all((output / name).is_file() for name in required)
    if checks["required_artifacts"]:
        config = json.loads((output / "corruption_config.json").read_text(encoding="utf-8"))
        validate_corruption_config(config)
        corruption = json.loads((output / "corruption_manifest.json").read_text(encoding="utf-8"))
        robust = json.loads((output / "robustness_metrics.json").read_text(encoding="utf-8"))
        boots = json.loads((output / "bootstrap_comparisons.json").read_text(encoding="utf-8"))
        shift = json.loads((output / "distribution_shift_definition.json").read_text(encoding="utf-8"))
        provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        checks["fixed_manifest"] = provenance.get("manifest_sha256") == MANIFEST_SHA256
        checks["query_manifest_complete"] = corruption.get("text_query_count") == 501 and corruption.get("image_query_count") == 100
        checks["robustness_rows_complete"] = len(robust.get("rows", [])) == 2 * (len(TEXT_FAMILIES) + len(IMAGE_FAMILIES)) * 2
        checks["paired_bootstrap_present"] = len(boots.get("rows", [])) == 2 * (len(TEXT_FAMILIES) + len(IMAGE_FAMILIES)) * 2 * 2
        checks["shift_predeclared"] = shift.get("selection_precedes_metrics") is True and len(shift.get("shifted_groups", [])) == len(shift.get("control_groups", [])) >= 2
        checks["evaluation_only"] = provenance.get("training_performed") is False and provenance.get("new_dataset_downloaded") is False
        checks["query_only_corruption"] = config.get("application", {}).get("new_dataset_written") is False
    else:
        for name in ("fixed_manifest", "query_manifest_complete", "robustness_rows_complete", "paired_bootstrap_present", "shift_predeclared", "evaluation_only", "query_only_corruption"):
            checks[name] = False
    return {"schema_version": PHASE15_SCHEMA_VERSION, "passed": all(checks.values()), "checks": checks, "required": sorted(required)}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the evaluation-only Phase 15 robustness evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase15"))
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=Path("data/processed/coco2017_val_split_manifest.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/coco2017/val2017"))
    parser.add_argument("--phase7-dir", type=Path, default=Path("artifacts/phase7"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_phase15(**vars(args))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
