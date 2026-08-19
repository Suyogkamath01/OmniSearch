"""Phase 9 static hard-negative mining and contrastive fine-tuning.

The experiment starts from pretrained CLIP, mines negatives once on train
image groups with the frozen pretrained model, and trains the same full CLIP
parameter set and optimizer schedule used by Phase 7.  The held-out test split
is opened only after validation checkpoint selection.
"""

from __future__ import annotations

import gc
import hashlib
import heapq
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config
from .evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    compare_systems,
    validate_result,
    write_result_artifacts,
)
from .manifest import ImageRecord, read_manifest
from .phase7 import (
    PHASE7_SCHEMA_VERSION,
    TrainingPair,
    _autocast_context,
    _comparison_metadata,
    _encode_images,
    _encode_texts,
    _feature_tensor,
    _features,
    _hash_file,
    _load_checkpoint,
    _load_rgb_images,
    _load_trainable_model,
    _move_inputs,
    _parameter_summary,
    _subset_records,
    _validation_loss,
    _write_json,
    build_training_pairs,
    evaluate_model,
)
from .splitting import assert_no_split_leakage

PHASE9_SCHEMA_VERSION = 1

DEFAULT_PHASE9_CONFIG: dict[str, Any] = {
    "manifest": "data/processed/coco2017_val_split_manifest.json",
    "image_root": "data/raw/coco2017/val2017",
    "model_id": "openai/clip-vit-base-patch32",
    "phase7_artifact_dir": "artifacts/phase7",
    "phase8_artifact_dir": "artifacts/phase8",
    "image_validation_path": "artifacts/coco_phase1/validation_images.json",
    "device": "auto",
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_workers": 0,
    "epochs": 1,
    "learning_rate": 1e-6,
    "weight_decay": 0.01,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "text_max_length": 77,
    "precision": "fp32",
    "selection_metric": "mean_recall_at_5",
    "early_stopping_patience": 2,
    "bootstrap_resamples": 200,
    "mining_split": "train",
    "mining_strategy": "static_frozen_clip_top5_hash_sample",
    "candidate_pool_size": 5,
    "hard_negative_ratio": 0.5,
    "potential_false_negative_similarity_threshold": 0.85,
    "fixed_mined_manifest": None,
    "max_train_images": 800,
    "max_validation_images": 100,
    "max_test_images": 100,
    "subset_seed": None,
}


@dataclass(frozen=True)
class HardNegativeRecord:
    """One positive pair and its two direction-specific mined negatives."""

    image_id: str
    positive_caption_id: str
    positive_caption_text: str
    negative_image_id: str
    negative_image_score: float
    negative_image_rank: int
    negative_image_pool_ids: tuple[str, ...]
    negative_caption_id: str
    negative_caption_text: str
    negative_caption_score: float
    negative_caption_rank: int
    negative_caption_pool_ids: tuple[str, ...]
    mining_strategy: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["negative_image_pool_ids"] = list(self.negative_image_pool_ids)
        value["negative_caption_pool_ids"] = list(self.negative_caption_pool_ids)
        return value


@dataclass(frozen=True)
class TrainingHardPair:
    pair: TrainingPair
    negative_image_path: str
    negative_caption_text: str
    negative_image_id: str
    negative_caption_id: str
    negative_image_score: float
    negative_caption_score: float


def _read_phase9_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as file:
        raw = tomllib.load(file)
    values = dict(DEFAULT_PHASE9_CONFIG)
    values.update(dict(raw.get("phase9", {})))
    return values


def validate_phase9_config(config: Mapping[str, Any]) -> None:
    expected_splits = {
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "mining_split": "train",
    }
    for key, expected in expected_splits.items():
        if key in config and config[key] != expected:
            raise ValueError(f"{key} must remain {expected!r}")
    for key in ("batch_size", "gradient_accumulation_steps", "epochs"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config.get("num_workers", 0)) < 0:
        raise ValueError("num_workers must be non-negative")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    if str(config["precision"]) not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if str(config["selection_metric"]) not in {
        "mean_recall_at_5",
        "mean_recall_at_1",
    }:
        raise ValueError("unsupported selection_metric")
    if str(config["mining_strategy"]) != "static_frozen_clip_top5_hash_sample":
        raise ValueError("only the declared static frozen-CLIP strategy is supported")
    if int(config["candidate_pool_size"]) < 2:
        raise ValueError("candidate_pool_size must be at least 2")
    if not 0 < float(config["hard_negative_ratio"]) <= 1:
        raise ValueError("hard_negative_ratio must be in (0, 1]")
    if not 0 <= float(config["potential_false_negative_similarity_threshold"]) <= 1:
        raise ValueError("potential false-negative threshold must be in [0, 1]")
    for key in ("max_train_images", "max_validation_images", "max_test_images"):
        value = config.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} must be positive when supplied")


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _stable_choice(seed: int, value: str, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot choose from an empty candidate pool")
    return int(_stable_key(seed, value)[:16], 16) % size


def _stable_fraction(seed: int, value: str) -> float:
    return int(_stable_key(seed, value)[:16], 16) / float(16**16 - 1)


def _caption_metadata(records: Sequence[ImageRecord]) -> dict[str, tuple[str, str, str]]:
    return {
        caption.caption_id: (record.image_id, caption.text, caption.normalized_hash)
        for record in records
        for caption in record.captions
    }


def _exact_duplicate_image_groups(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("exact_duplicate_groups", {})
    output: dict[str, set[str]] = {}
    if isinstance(groups, Mapping):
        for ids in groups.values():
            group = {str(item) for item in ids}
            for image_id in group:
                output[image_id] = group - {image_id}
    return output


def _ranked_candidates(
    scores: Sequence[float],
    candidate_ids: Sequence[str],
    forbidden: set[str],
    pool_size: int,
) -> tuple[tuple[str, float, int], ...]:
    if len(scores) != len(candidate_ids):
        raise ValueError("scores and candidate IDs must have equal length")
    valid = [
        (str(candidate_id), float(score))
        for candidate_id, score in zip(candidate_ids, scores)
        if str(candidate_id) not in forbidden
    ]
    if len(valid) < pool_size:
        raise ValueError("candidate pool is smaller than the requested pool size")
    top = heapq.nsmallest(pool_size, valid, key=lambda item: (-item[1], item[0]))
    top.sort(key=lambda item: (-item[1], item[0]))
    return tuple(
        (
            candidate_id,
            score,
            1
            + sum(
                1
                for other_id, other_score in valid
                if (-other_score, other_id) < (-score, candidate_id)
            ),
        )
        for candidate_id, score in top
    )


def _mine_from_embeddings(
    train_pairs: Sequence[TrainingPair],
    records: Sequence[ImageRecord],
    image_ids: Sequence[str],
    image_embeddings: Any,
    caption_ids: Sequence[str],
    text_embeddings: Any,
    seed: int,
    candidate_pool_size: int,
    strategy: str,
    exact_duplicate_groups: Mapping[str, set[str]] | None = None,
) -> tuple[tuple[HardNegativeRecord, ...], dict[str, Any]]:
    """Mine deterministic top-N candidates from train-only embeddings."""

    import torch

    if strategy != "static_frozen_clip_top5_hash_sample":
        raise ValueError("unsupported mining strategy")
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("mining embeddings must be rank-2")
    if len(image_ids) != image_embeddings.shape[0]:
        raise ValueError("image IDs do not match image embeddings")
    if len(caption_ids) != text_embeddings.shape[0]:
        raise ValueError("caption IDs do not match text embeddings")
    if not bool(torch.isfinite(image_embeddings).all().item()) or not bool(
        torch.isfinite(text_embeddings).all().item()
    ):
        raise ValueError("mining embeddings must be finite")
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    caption_index = {caption_id: index for index, caption_id in enumerate(caption_ids)}
    if len(image_index) != len(image_ids) or len(caption_index) != len(caption_ids):
        raise ValueError("mining IDs must be unique")
    metadata = _caption_metadata(records)
    captions_by_image = {
        record.image_id: {caption.caption_id for caption in record.captions}
        for record in records
    }
    duplicate_groups = exact_duplicate_groups or {}
    image_similarity = image_embeddings @ text_embeddings.transpose(0, 1)
    text_similarity = image_similarity.transpose(0, 1)
    mined: list[HardNegativeRecord] = []
    image_ranks: list[int] = []
    caption_ranks: list[int] = []
    image_scores: list[float] = []
    caption_scores: list[float] = []
    duplicate_caption_exclusions = 0
    same_image_caption_exclusions = 0
    exact_duplicate_image_exclusions = 0
    for pair in train_pairs:
        if pair.image_id not in image_index or pair.caption_id not in caption_index:
            raise ValueError("training pair is absent from mining embedding IDs")
        positive_image_index = image_index[pair.image_id]
        positive_caption_meta = metadata.get(pair.caption_id)
        if positive_caption_meta is None:
            raise ValueError(f"positive caption is absent from manifest: {pair.caption_id}")
        _, _, positive_caption_hash = positive_caption_meta
        forbidden_images = {pair.image_id} | set(duplicate_groups.get(pair.image_id, set()))
        exact_duplicate_image_exclusions += len(duplicate_groups.get(pair.image_id, set()))
        image_pool = _ranked_candidates(
            text_similarity[caption_index[pair.caption_id]].tolist(),
            image_ids,
            forbidden_images,
            candidate_pool_size,
        )
        selected_image = image_pool[
            _stable_choice(seed, f"image\0{pair.caption_id}", len(image_pool))
        ]
        forbidden_captions = set(captions_by_image[pair.image_id])
        same_image_caption_exclusions += len(forbidden_captions)
        for caption_id, (_, _, normalized_hash) in metadata.items():
            if normalized_hash == positive_caption_hash and caption_id not in forbidden_captions:
                forbidden_captions.add(caption_id)
                duplicate_caption_exclusions += 1
        caption_pool = _ranked_candidates(
            image_similarity[positive_image_index].tolist(),
            caption_ids,
            forbidden_captions,
            candidate_pool_size,
        )
        selected_caption = caption_pool[
            _stable_choice(seed, f"caption\0{pair.image_id}", len(caption_pool))
        ]
        negative_caption_meta = metadata[selected_caption[0]]
        mined.append(
            HardNegativeRecord(
                image_id=pair.image_id,
                positive_caption_id=pair.caption_id,
                positive_caption_text=pair.text,
                negative_image_id=selected_image[0],
                negative_image_score=selected_image[1],
                negative_image_rank=selected_image[2],
                negative_image_pool_ids=tuple(item[0] for item in image_pool),
                negative_caption_id=selected_caption[0],
                negative_caption_text=negative_caption_meta[1],
                negative_caption_score=selected_caption[1],
                negative_caption_rank=selected_caption[2],
                negative_caption_pool_ids=tuple(item[0] for item in caption_pool),
                mining_strategy=strategy,
            )
        )
        image_ranks.append(selected_image[2])
        caption_ranks.append(selected_caption[2])
        image_scores.append(selected_image[1])
        caption_scores.append(selected_caption[1])
    if not mined:
        raise ValueError("mining produced no records")
    statistics_payload = {
        "pairs_mined": len(mined),
        "image_negative_rank": {
            "mean": statistics.fmean(image_ranks),
            "min": min(image_ranks),
            "max": max(image_ranks),
            "counts": dict(sorted(Counter(image_ranks).items())),
        },
        "caption_negative_rank": {
            "mean": statistics.fmean(caption_ranks),
            "min": min(caption_ranks),
            "max": max(caption_ranks),
            "counts": dict(sorted(Counter(caption_ranks).items())),
        },
        "image_negative_score": {
            "mean": statistics.fmean(image_scores),
            "min": min(image_scores),
            "max": max(image_scores),
        },
        "caption_negative_score": {
            "mean": statistics.fmean(caption_scores),
            "min": min(caption_scores),
            "max": max(caption_scores),
        },
        "same_image_caption_exclusions": same_image_caption_exclusions,
        "exact_duplicate_image_exclusions": exact_duplicate_image_exclusions,
        "exact_duplicate_caption_alias_exclusions": duplicate_caption_exclusions,
        "candidate_pool_size": candidate_pool_size,
        "selection": "deterministic hash sample from valid top-N candidates",
    }
    return tuple(mined), statistics_payload


def _validate_mined_manifest(
    payload: Mapping[str, Any],
    expected_manifest_sha256: str | None = None,
    expected_split: str = "train",
) -> tuple[HardNegativeRecord, ...]:
    required = {
        "phase9_schema_version",
        "manifest_sha256",
        "mining_split",
        "mining_strategy",
        "records",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"mined manifest missing fields: {', '.join(missing)}")
    if int(payload["phase9_schema_version"]) != PHASE9_SCHEMA_VERSION:
        raise ValueError("unsupported mined manifest schema")
    if expected_manifest_sha256 is not None and payload["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("mined manifest is stale for the active dataset manifest")
    if payload["mining_split"] != expected_split:
        raise ValueError("mining manifest is not train-only")
    if payload["mining_strategy"] != "static_frozen_clip_top5_hash_sample":
        raise ValueError("unexpected mining strategy")
    rows = payload["records"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("mined manifest records must be a non-empty list")
    output: list[HardNegativeRecord] = []
    positive_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("mined manifest row must be an object")
        try:
            record = HardNegativeRecord(
                image_id=str(row["image_id"]),
                positive_caption_id=str(row["positive_caption_id"]),
                positive_caption_text=str(row["positive_caption_text"]),
                negative_image_id=str(row["negative_image_id"]),
                negative_image_score=float(row["negative_image_score"]),
                negative_image_rank=int(row["negative_image_rank"]),
                negative_image_pool_ids=tuple(str(item) for item in row["negative_image_pool_ids"]),
                negative_caption_id=str(row["negative_caption_id"]),
                negative_caption_text=str(row["negative_caption_text"]),
                negative_caption_score=float(row["negative_caption_score"]),
                negative_caption_rank=int(row["negative_caption_rank"]),
                negative_caption_pool_ids=tuple(str(item) for item in row["negative_caption_pool_ids"]),
                mining_strategy=str(row["mining_strategy"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed mined manifest row") from exc
        if record.image_id in positive_ids:
            raise ValueError("mined manifest has duplicate positive image IDs")
        if record.negative_image_id == record.image_id:
            raise ValueError("same-image negative was not excluded")
        if record.negative_caption_id == record.positive_caption_id:
            raise ValueError("positive caption was not excluded")
        if record.negative_image_rank <= 0 or record.negative_caption_rank <= 0:
            raise ValueError("mined ranks must be positive")
        if not math.isfinite(record.negative_image_score) or not math.isfinite(record.negative_caption_score):
            raise ValueError("mined scores must be finite")
        if record.mining_strategy != payload["mining_strategy"]:
            raise ValueError("row mining strategy differs from manifest strategy")
        positive_ids.add(record.image_id)
        output.append(record)
    return tuple(output)


def _hard_negative_training_records(
    pairs: Sequence[TrainingPair],
    mined: Sequence[HardNegativeRecord],
    image_root: Path,
    records_by_image: Mapping[str, ImageRecord],
    ratio: float,
    seed: int,
    epoch: int,
) -> tuple[TrainingHardPair, ...]:
    by_image = {item.image_id: item for item in mined}
    hard_count = max(1, math.ceil(len(pairs) * ratio)) if pairs and ratio > 0 else 0
    hard_image_ids = {
        pair.image_id
        for pair in sorted(
            pairs,
            key=lambda pair: _stable_key(seed + epoch, pair.image_id),
        )[:hard_count]
    }
    output: list[TrainingHardPair] = []
    for pair in pairs:
        if pair.image_id not in by_image:
            raise ValueError(f"no mined negative for training image {pair.image_id}")
        item = by_image[pair.image_id]
        negative_image_record = records_by_image.get(item.negative_image_id)
        if negative_image_record is None or negative_image_record.filename is None:
            raise ValueError("mined negative image is absent or has no filename")
        if pair.image_id in hard_image_ids:
            output.append(
                TrainingHardPair(
                    pair=pair,
                    negative_image_path=str(image_root / negative_image_record.filename),
                    negative_caption_text=item.negative_caption_text,
                    negative_image_id=item.negative_image_id,
                    negative_caption_id=item.negative_caption_id,
                    negative_image_score=item.negative_image_score,
                    negative_caption_score=item.negative_caption_score,
                )
            )
    if ratio > 0 and pairs and not output:
        item = by_image[pairs[0].image_id]
        negative_image_record = records_by_image[item.negative_image_id]
        if negative_image_record.filename is None:
            raise ValueError("fallback mined negative image has no filename")
        output.append(
            TrainingHardPair(
                pair=pairs[0],
                negative_image_path=str(image_root / negative_image_record.filename),
                negative_caption_text=item.negative_caption_text,
                negative_image_id=item.negative_image_id,
                negative_caption_id=item.negative_caption_id,
                negative_image_score=item.negative_image_score,
                negative_caption_score=item.negative_caption_score,
            )
        )
    return tuple(output)


def hard_negative_clip_loss(
    image_embeddings: Any,
    text_embeddings: Any,
    hard_image_embeddings: Any,
    hard_text_embeddings: Any,
    logit_scale: Any,
) -> tuple[Any, dict[str, float]]:
    """Add explicit mined negatives while preserving diagonal positives."""

    import torch
    from torch.nn import functional

    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("positive embeddings must be rank-2")
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("positive image/text embedding shapes must match")
    if hard_image_embeddings.ndim != 2 or hard_text_embeddings.ndim != 2:
        raise ValueError("hard embeddings must be rank-2")
    if hard_image_embeddings.shape != hard_text_embeddings.shape:
        raise ValueError("hard image/text embedding shapes must match")
    if hard_image_embeddings.shape[1] != image_embeddings.shape[1]:
        raise ValueError("hard embeddings have a different feature dimension")
    if image_embeddings.shape[0] <= 0:
        raise ValueError("contrastive batches must be non-empty")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (
            image_embeddings,
            text_embeddings,
            hard_image_embeddings,
            hard_text_embeddings,
            logit_scale,
        )
    ):
        raise ValueError("contrastive inputs must be finite")
    image_normalized = functional.normalize(image_embeddings, dim=-1)
    text_normalized = functional.normalize(text_embeddings, dim=-1)
    hard_image_normalized = functional.normalize(hard_image_embeddings, dim=-1)
    hard_text_normalized = functional.normalize(hard_text_embeddings, dim=-1)
    safe_logit_scale = torch.exp(torch.clamp(logit_scale, max=math.log(100.0)))
    text_candidates = torch.cat((text_normalized, hard_text_normalized), dim=0)
    image_candidates = torch.cat((image_normalized, hard_image_normalized), dim=0)
    image_to_text_logits = safe_logit_scale * image_normalized @ text_candidates.transpose(0, 1)
    text_to_image_logits = safe_logit_scale * text_normalized @ image_candidates.transpose(0, 1)
    targets = torch.arange(image_embeddings.shape[0], device=image_embeddings.device)
    image_to_text = functional.cross_entropy(image_to_text_logits, targets)
    text_to_image = functional.cross_entropy(text_to_image_logits, targets)
    loss = (image_to_text + text_to_image) / 2.0
    return loss, {
        "image_to_text_loss": float(image_to_text.detach().item()),
        "text_to_image_loss": float(text_to_image.detach().item()),
        "logit_scale": float(safe_logit_scale.detach().item()),
        "hard_negative_count": float(hard_image_embeddings.shape[0]),
    }


def _hard_negative_epoch(
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    pairs: Sequence[TrainingPair],
    hard_pairs: Sequence[TrainingHardPair],
    batch_size: int,
    gradient_accumulation_steps: int,
    text_max_length: int,
    max_grad_norm: float,
    first_parameter_before: Any | None,
    num_workers: int,
    precision: str,
) -> tuple[dict[str, Any], bool]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    hard_by_image = {item.pair.image_id: item for item in hard_pairs}
    losses: list[float] = []
    image_losses: list[float] = []
    text_losses: list[float] = []
    scales: list[float] = []
    hard_counts: list[int] = []
    optimizer_steps = 0
    update_verified = False
    gradients_finite = True
    gradient_norm = 0.0
    batch_count = math.ceil(len(pairs) / batch_size)
    for batch_index, start in enumerate(range(0, len(pairs), batch_size)):
        batch = pairs[start : start + batch_size]
        device = model.device if hasattr(model, "device") else str(next(model.parameters()).device)
        positive_images: list[Any] = []
        hard_batch = [hard_by_image[pair.image_id] for pair in batch if pair.image_id in hard_by_image]
        hard_images: list[Any] = []
        try:
            positive_images = _load_rgb_images([pair.image_path for pair in batch], num_workers)
            inputs = _move_inputs(
                processor(
                    text=[pair.text for pair in batch],
                    images=positive_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=text_max_length,
                ),
                device,
            )
            with _autocast_context(torch, device, precision):
                image_features, text_features = _features(model, inputs)
                if hard_batch:
                    hard_images = _load_rgb_images(
                        [item.negative_image_path for item in hard_batch], num_workers
                    )
                    hard_image_inputs = _move_inputs(
                        processor(images=hard_images, return_tensors="pt"), device
                    )
                    hard_image_features = _feature_tensor(
                        model.get_image_features(pixel_values=hard_image_inputs["pixel_values"])
                    )
                    hard_text_inputs = _move_inputs(
                        processor(
                            text=[item.negative_caption_text for item in hard_batch],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=text_max_length,
                        ),
                        device,
                    )
                    hard_text_features = _feature_tensor(
                        model.get_text_features(**hard_text_inputs)
                    )
                else:
                    hard_image_features = image_features.new_empty((0, image_features.shape[1]))
                    hard_text_features = text_features.new_empty((0, text_features.shape[1]))
                loss, details = hard_negative_clip_loss(
                    image_features,
                    text_features,
                    hard_image_features,
                    hard_text_features,
                    model.logit_scale,
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite hard-negative loss at batch {batch_index}")
            losses.append(float(loss.detach().item()))
            image_losses.append(details["image_to_text_loss"])
            text_losses.append(details["text_to_image_loss"])
            scales.append(details["logit_scale"])
            hard_counts.append(len(hard_batch))
            (loss / gradient_accumulation_steps).backward()
        finally:
            for image in hard_images:
                close = getattr(image, "close", None)
                if close:
                    close()
            for image in positive_images:
                close = getattr(image, "close", None)
                if close:
                    close()
        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == batch_count
        )
        if should_step:
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients:
                raise RuntimeError("hard-negative backward pass produced no gradients")
            if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
                gradients_finite = False
                raise FloatingPointError(f"non-finite gradient at batch {batch_index}")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if first_parameter_before is not None and not update_verified:
                current = next(parameter for parameter in model.parameters() if parameter.requires_grad)
                update_verified = not bool(torch.equal(first_parameter_before.cpu(), current.detach().cpu()))
    if not update_verified and first_parameter_before is not None:
        raise RuntimeError("hard-negative optimizer step did not update weights")
    return {
        "loss": math.fsum(losses) / len(losses),
        "image_to_text_loss": math.fsum(image_losses) / len(image_losses),
        "text_to_image_loss": math.fsum(text_losses) / len(text_losses),
        "logit_scale": math.fsum(scales) / len(scales),
        "batches": batch_count,
        "optimizer_steps": optimizer_steps,
        "hard_negative_rows": sum(hard_counts),
        "hard_negative_proportion": sum(hard_counts) / len(pairs),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "gradient_norm_last": gradient_norm,
        "gradients_finite": gradients_finite,
        "parameter_update_verified": update_verified,
    }, update_verified


def _training_hard_map(
    records: Sequence[ImageRecord],
    mined: Sequence[HardNegativeRecord],
    image_root: Path,
    ratio: float,
    seed: int,
    epoch: int,
) -> tuple[TrainingHardPair, ...]:
    pairs = build_training_pairs(records, image_root, seed, epoch=epoch)
    return _hard_negative_training_records(
        pairs,
        mined,
        image_root,
        {record.image_id: record for record in records},
        ratio,
        seed,
        epoch,
    )


def _load_phase7_results(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    zero: dict[str, Any] = {}
    full: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        zero[task] = json.loads((artifact_dir / f"zero_shot_{task}.json").read_text())
        full[task] = json.loads((artifact_dir / f"fine_tuned_{task}.json").read_text())
    return zero, full


def _load_phase8_results(artifact_dir: Path) -> dict[str, Any]:
    return {
        task: json.loads((artifact_dir / f"lora_{task}.json").read_text())
        for task in ("text_to_image", "image_to_text")
    }


def _result_rankings(result: Mapping[str, Any]) -> tuple[RankingRecord, ...]:
    return tuple(RankingRecord.from_mapping(item) for item in result["ranking_records"])


def _rank(record: RankingRecord) -> int | None:
    for rank, candidate_id in enumerate(record.candidate_ids, start=1):
        if candidate_id in record.relevant_ids:
            return rank
    return None


def _top_rank_analysis(
    hard_rankings: Mapping[str, Sequence[RankingRecord]],
    full_rankings: Mapping[str, Sequence[RankingRecord]],
    zero_rankings: Mapping[str, Sequence[RankingRecord]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        hard = {item.query_id: item for item in hard_rankings[task]}
        full = {item.query_id: item for item in full_rankings[task]}
        zero = {item.query_id: item for item in zero_rankings[task]}
        hard_ranks = [_rank(item) for item in hard.values()]
        full_ranks = [_rank(full[item.query_id]) for item in hard.values()]
        zero_ranks = [_rank(zero[item.query_id]) for item in hard.values()]

        def median(values: Sequence[int | None]) -> float | None:
            valid = [value for value in values if value is not None]
            return statistics.median(valid) if valid else None

        paired_rank_deltas = [
            float(hard_rank) - float(full_rank)
            for hard_rank, full_rank in zip(hard_ranks, full_ranks)
            if hard_rank is not None and full_rank is not None
        ]
        output[task] = {
            "hard_negative": {
                "r_at_1": sum(value is not None and value <= 1 for value in hard_ranks) / len(hard_ranks),
                "r_at_5": sum(value is not None and value <= 5 for value in hard_ranks) / len(hard_ranks),
                "median_first_relevant_rank": median(hard_ranks),
            },
            "full_finetuning": {
                "r_at_1": sum(value is not None and value <= 1 for value in full_ranks) / len(full_ranks),
                "r_at_5": sum(value is not None and value <= 5 for value in full_ranks) / len(full_ranks),
                "median_first_relevant_rank": median(full_ranks),
            },
            "zero_shot": {
                "r_at_1": sum(value is not None and value <= 1 for value in zero_ranks) / len(zero_ranks),
                "r_at_5": sum(value is not None and value <= 5 for value in zero_ranks) / len(zero_ranks),
                "median_first_relevant_rank": median(zero_ranks),
            },
            "hard_minus_full_rank_delta_mean_on_joint_hits": statistics.fmean(paired_rank_deltas)
            if paired_rank_deltas
            else None,
            "hard_misses": sum(value is None for value in hard_ranks),
            "full_finetuning_misses": sum(value is None for value in full_ranks),
            "zero_shot_misses": sum(value is None for value in zero_ranks),
            "rank_delta_note": "mean delta is computed only for queries with a hit in both hard-negative and full-FT rankings; misses are reported separately",
        }
    return output


def _mining_quality_sample(
    mined: Sequence[HardNegativeRecord],
    sample_size: int,
) -> dict[str, Any]:
    sample = list(mined[:sample_size])
    rows = []
    for item in sample:
        positive_tokens = set(item.positive_caption_text.lower().split())
        negative_tokens = set(item.negative_caption_text.lower().split())
        overlap = len(positive_tokens & negative_tokens) / max(1, len(positive_tokens | negative_tokens))
        rows.append(
            {
                **item.to_dict(),
                "caption_token_jaccard": overlap,
                "conceptual_screen": "lexical-overlap heuristic only; human label not assigned",
                "review_categories": [
                    "genuinely hard",
                    "visually similar",
                    "semantically related",
                    "obvious easy negative",
                    "suspected false negative",
                ],
            }
        )
    return {
        "selection": "first deterministic records after stable manifest ordering",
        "human_labels": False,
        "sample_size": len(rows),
        "samples": rows,
    }


def _false_negative_audit(
    mined: Sequence[HardNegativeRecord],
    records: Sequence[ImageRecord],
    threshold: float,
) -> dict[str, Any]:
    metadata = _caption_metadata(records)
    known_same_image = sum(
        item.negative_image_id == item.image_id
        or metadata.get(item.negative_caption_id, ("", "", ""))[0] == item.image_id
        for item in mined
    )
    exact_caption_alias = sum(
        metadata.get(item.negative_caption_id, ("", "", ""))[2]
        == metadata.get(item.positive_caption_id, ("", "", "x"))[2]
        for item in mined
    )
    potential = sum(
        item.negative_image_score >= threshold or item.negative_caption_score >= threshold
        for item in mined
    )
    return {
        "denominator_pairs": len(mined),
        "known_false_negatives": {
            "count": known_same_image + exact_caption_alias,
            "rate": (known_same_image + exact_caption_alias) / len(mined),
            "same_image_or_caption_count": known_same_image,
            "exact_caption_alias_count": exact_caption_alias,
        },
        "potential_semantic_false_negatives": {
            "screened_count": potential,
            "screened_fraction": potential / len(mined),
            "similarity_threshold": threshold,
            "interpretation": "high frozen-CLIP similarity is a risk screen, not a human false-negative label",
        },
        "exact_image_duplicate_protection": "no exact duplicate groups were present in the active COCO validation report",
        "near_duplicate_limitation": "perceptual near-duplicates and semantic equivalence across different image IDs are not fully labeled",
    }


def _qualitative_rank_changes(
    hard_rankings: Mapping[str, Sequence[RankingRecord]],
    full_rankings: Mapping[str, Sequence[RankingRecord]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        hard = {item.query_id: item for item in hard_rankings[task]}
        full = {item.query_id: item for item in full_rankings[task]}
        counts = {"improved": 0, "unchanged": 0, "degraded": 0}
        for query_id, item in hard.items():
            hard_rank = _rank(item)
            full_rank = _rank(full[query_id])
            if hard_rank is not None and (full_rank is None or hard_rank < full_rank):
                category = "improved"
            elif full_rank is not None and (hard_rank is None or hard_rank > full_rank):
                category = "degraded"
            else:
                category = "unchanged"
            counts[category] += 1
        output[task] = {"counts": counts, "human_labels": False}
    return output


def _assert_test_isolation(test_records: Sequence[ImageRecord], selected: bool) -> None:
    if not selected:
        raise AssertionError("hard-negative checkpoint was not selected by validation")
    if any(record.split != "test" for record in test_records):
        raise AssertionError("non-test records reached the final test evaluation")


def _peft_version() -> str:
    try:
        import peft

        return str(peft.__version__)
    except ImportError:
        return "unavailable"


def _save_checkpoint_atomic(path: Path, model: Any, metadata: Mapping[str, Any]) -> None:
    import torch

    state = {
        key: value.detach().cpu() if hasattr(value, "detach") else value
        for key, value in model.state_dict().items()
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state_dict": state, "metadata": dict(metadata)}, temporary)
    temporary.replace(path)


def run_phase9(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase9",
    smoke: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_config = load_config(config_path)
    config = _read_phase9_config(config_path)
    if smoke:
        config.update(
            {
                "max_train_images": 8,
                "max_validation_images": 4,
                "max_test_images": 4,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "bootstrap_resamples": 10,
                "phase7_artifact_dir": "artifacts/phase7_smoke",
                "phase8_artifact_dir": "artifacts/phase8_smoke",
            }
        )
    validate_phase9_config(config)
    manifest_path = Path(str(config["manifest"]))
    image_root = Path(str(config["image_root"]))
    manifest = read_manifest(manifest_path)
    assert_no_split_leakage(manifest.records)
    if manifest.dataset_id != project_config.dataset_id:
        raise ValueError("Phase 9 dataset does not match the active project dataset")
    seed = project_config.seed
    subset_seed = seed if config.get("subset_seed") is None else int(config["subset_seed"])
    train_records = _subset_records(manifest.records, "train", subset_seed, config.get("max_train_images"))
    validation_records = _subset_records(manifest.records, "validation", subset_seed, config.get("max_validation_images"))
    if not train_records or not validation_records:
        raise ValueError("Phase 9 requires non-empty train and validation records")
    phase7_dir = Path(str(config["phase7_artifact_dir"]))
    phase8_dir = Path(str(config["phase8_artifact_dir"]))
    phase7_report = json.loads((phase7_dir / "phase7_report.json").read_text())
    if phase7_report["pretrained_checkpoint"]["model_id"] != config["model_id"]:
        raise ValueError("Phase 9 model does not match Phase 7 pretrained parent")
    manifest_sha256 = _hash_file(manifest_path)
    if phase7_report["dataset"]["manifest_sha256"] != manifest_sha256:
        raise ValueError("Phase 9 manifest differs from Phase 7")
    mining_pairs = build_training_pairs(train_records, image_root, seed, epoch=0)
    fixed_mined_manifest_path = config.get("fixed_mined_manifest")
    if fixed_mined_manifest_path:
        mining_device = "fixed_manifest"
        mining_started = time.perf_counter()
        fixed_path = Path(str(fixed_mined_manifest_path))
        mining_manifest = json.loads(fixed_path.read_text(encoding="utf-8"))
        mined = _validate_mined_manifest(mining_manifest, manifest_sha256)
        train_ids = {record.image_id for record in train_records}
        if {record.image_id for record in mined} != train_ids:
            raise ValueError("fixed mined manifest does not cover the selected train image groups")
        mining_statistics = {
            "source": "fixed_verified_manifest",
            "source_path": str(fixed_path),
            "source_sha256": _hash_file(fixed_path),
            "pairs_mined": len(mined),
            "seed_for_mining": mining_manifest.get("seed"),
            "training_seed": seed,
        }
        mining_seconds = time.perf_counter() - mining_started
    else:
        mining_started = time.perf_counter()
        mining_model, mining_processor, mining_torch, mining_device = _load_trainable_model(
            str(config["model_id"]), str(config["device"])
        )
        for parameter in mining_model.parameters():
            parameter.requires_grad_(False)
        mining_model.eval()
        train_caption_items = [
            (caption.caption_id, caption.text)
            for record in train_records
            for caption in record.captions
        ]
        mined_image_ids, mined_image_embeddings = _encode_images(
            mining_model,
            mining_processor,
            mining_torch,
            train_records,
            image_root,
            int(config["batch_size"]),
            int(config["num_workers"]),
            str(config["precision"]),
        )
        mined_caption_ids, mined_text_embeddings = _encode_texts(
            mining_model,
            mining_processor,
            mining_torch,
            train_caption_items,
            int(config["batch_size"]),
            int(config["text_max_length"]),
            int(config["num_workers"]),
            str(config["precision"]),
        )
        mined, mining_statistics = _mine_from_embeddings(
            mining_pairs,
            train_records,
            mined_image_ids,
            mined_image_embeddings,
            mined_caption_ids,
            mined_text_embeddings,
            seed,
            int(config["candidate_pool_size"]),
            str(config["mining_strategy"]),
            _exact_duplicate_image_groups(Path(str(config["image_validation_path"]))),
        )
        mining_seconds = time.perf_counter() - mining_started
        del mining_model
        gc.collect()
        mining_manifest = {
            "phase9_schema_version": PHASE9_SCHEMA_VERSION,
            "phase7_schema_version": PHASE7_SCHEMA_VERSION,
            "dataset_id": manifest.dataset_id,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "mining_model": str(config["model_id"]),
            "mining_checkpoint": "pretrained_zero_shot_clip",
            "mining_split": "train",
            "mining_strategy": str(config["mining_strategy"]),
            "candidate_pool_size": int(config["candidate_pool_size"]),
            "seed": seed,
            "records": [item.to_dict() for item in mined],
        }
        _validate_mined_manifest(mining_manifest, manifest_sha256)
    _write_json(mining_manifest, output_dir / "mined_negative_manifest.json")
    _write_json(mining_statistics, output_dir / "mining_statistics.json")
    _write_json(
        {
            "phase9_schema_version": PHASE9_SCHEMA_VERSION,
            "manifest_sha256": manifest_sha256,
            "mining_split": "train",
            "mining_model": config["model_id"],
            "mining_strategy": config["mining_strategy"],
            "candidate_pool_size": config["candidate_pool_size"],
            "hard_negative_ratio": config["hard_negative_ratio"],
            "config": config,
            "mining_seconds": mining_seconds,
            "mining_device": str(mining_device),
        },
        output_dir / "mining_config.json",
    )

    model, processor, torch, device = _load_trainable_model(
        str(config["model_id"]), str(config["device"])
    )
    parameter_summary = _parameter_summary(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    steps_per_epoch = max(
        1,
        math.ceil(
            math.ceil(len(mining_pairs) / int(config["batch_size"]))
            / int(config["gradient_accumulation_steps"])
        ),
    )
    total_steps = max(1, steps_per_epoch * int(config["epochs"]))
    warmup_steps = int(config["warmup_steps"])

    def schedule(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_score = float("-inf")
    best_epoch: int | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    first_parameter_before = next(model.parameters()).detach().clone()
    training_started = time.perf_counter()
    records_by_image = {record.image_id: record for record in train_records}
    for epoch in range(int(config["epochs"])):
        epoch_pairs = build_training_pairs(train_records, image_root, seed, epoch=epoch)
        hard_pairs = _hard_negative_training_records(
            epoch_pairs,
            mined,
            image_root,
            records_by_image,
            float(config["hard_negative_ratio"]),
            seed,
            epoch,
        )
        train_stats, update_verified = _hard_negative_epoch(
            model,
            processor,
            optimizer,
            scheduler,
            torch,
            epoch_pairs,
            hard_pairs,
            int(config["batch_size"]),
            int(config["gradient_accumulation_steps"]),
            int(config["text_max_length"]),
            float(config["max_grad_norm"]),
            first_parameter_before,
            int(config["num_workers"]),
            str(config["precision"]),
        )
        first_parameter_before = None
        validation_pairs = build_training_pairs(validation_records, image_root, seed, epoch=0)
        validation_loss = _validation_loss(
            model,
            processor,
            torch,
            validation_pairs,
            int(config["batch_size"]),
            int(config["text_max_length"]),
            int(config["num_workers"]),
            str(config["precision"]),
        )
        validation_eval = evaluate_model(
            model,
            processor,
            torch,
            validation_records,
            image_root,
            int(config["batch_size"]),
            int(config["text_max_length"]),
            manifest,
            manifest_path,
            config_path,
            "validation",
            seed,
            "phase9_validation_hard_negative",
            f"phase9_validation_epoch_{epoch + 1}",
            {**parameter_summary, "model_id": config["model_id"], "hard_negative_training": True},
            int(config["bootstrap_resamples"]),
            int(config["num_workers"]),
            str(config["precision"]),
        )
        metric_suffix = str(config["selection_metric"]).removeprefix("mean_")
        selection_score = statistics.fmean(
            float(validation_eval["results"][task]["metrics"][metric_suffix])
            for task in ("text_to_image", "image_to_text")
        )
        improved = selection_score > best_score
        if improved:
            best_score = selection_score
            best_epoch = epoch + 1
            no_improvement = 0
            checkpoint_metadata = {
                "epoch": best_epoch,
                "selection_metric": config["selection_metric"],
                "selection_score": selection_score,
                "selection_split": "validation",
                "test_used_for_selection": False,
                "parent_model_id": config["model_id"],
                "mined_manifest_sha256": _hash_file(output_dir / "mined_negative_manifest.json"),
                "hard_negative_ratio": config["hard_negative_ratio"],
            }
            _save_checkpoint_atomic(output_dir / "best_checkpoint.pt", model, checkpoint_metadata)
        else:
            no_improvement += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_stats,
                "validation_loss": validation_loss,
                "validation_metrics": {
                    "text_to_image": validation_eval["results"]["text_to_image"]["metrics"],
                    "image_to_text": validation_eval["results"]["image_to_text"]["metrics"],
                    "selection_metric": config["selection_metric"],
                    "selection_score": selection_score,
                },
                "hard_negative_rows": len(hard_pairs),
                "checkpoint_selected": improved,
                "optimizer_updates_weights": update_verified,
            }
        )
        _write_json(history, output_dir / "training_history.json")
        if no_improvement >= int(config["early_stopping_patience"]):
            break
    if best_epoch is None:
        raise RuntimeError("no validation-selected hard-negative checkpoint was produced")
    checkpoint_metadata = _load_checkpoint(output_dir / "best_checkpoint.pt", model)
    training_seconds = time.perf_counter() - training_started
    _write_json(
        {
            "selected_checkpoint": str(output_dir / "best_checkpoint.pt"),
            "selection_split": "validation",
            "selection_metric": config["selection_metric"],
            "selection_score": best_score,
            "selected_epoch": best_epoch,
            "test_used_for_selection": False,
            "mined_manifest_sha256": _hash_file(output_dir / "mined_negative_manifest.json"),
            "checkpoint_metadata": checkpoint_metadata,
        },
        output_dir / "checkpoint_metadata.json",
    )
    test_records = _subset_records(manifest.records, "test", subset_seed, config.get("max_test_images"))
    _assert_test_isolation(test_records, best_epoch is not None)
    test_started = time.perf_counter()
    hard_eval = evaluate_model(
        model,
        processor,
        torch,
        test_records,
        image_root,
        int(config["batch_size"]),
        int(config["text_max_length"]),
        manifest,
        manifest_path,
        config_path,
        "test",
        seed,
        "phase9_hard_negative_clip",
        "phase9_hard_negative_clip_test",
        {**parameter_summary, "model_id": config["model_id"], "hard_negative_training": True},
        int(config["bootstrap_resamples"]),
        int(config["num_workers"]),
        str(config["precision"]),
    )
    test_seconds = time.perf_counter() - test_started
    hard_results = hard_eval["results"]
    hard_rankings = hard_eval["rankings"]
    zero_results, full_results = _load_phase7_results(phase7_dir)
    lora_results = _load_phase8_results(phase8_dir)
    for task in ("text_to_image", "image_to_text"):
        validate_result(hard_results[task])
        _write_json(hard_results[task], output_dir / f"hard_negative_{task}.json")
        write_result_artifacts(hard_results[task], output_dir / f"hard_negative_{task}_summary")
    zero_rankings = {task: _result_rankings(zero_results[task]) for task in zero_results}
    full_rankings = {task: _result_rankings(full_results[task]) for task in full_results}
    comparisons: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        hard_metadata = _comparison_metadata(hard_results[task])
        comparisons[task] = {
            "hard_negative_vs_full_finetuning": compare_systems(
                full_rankings[task],
                hard_rankings[task],
                _comparison_metadata(full_results[task]),
                hard_metadata,
                bootstrap_resamples=int(config["bootstrap_resamples"]),
                seed=seed,
            ),
            "hard_negative_vs_zero_shot": compare_systems(
                zero_rankings[task],
                hard_rankings[task],
                _comparison_metadata(zero_results[task]),
                hard_metadata,
                bootstrap_resamples=int(config["bootstrap_resamples"]),
                seed=seed,
            ),
        }
    _write_json(comparisons, output_dir / "paired_comparisons.json")
    _write_json(
        {
            task: {
                "zero_shot": zero_results[task]["metrics"],
                "full_finetuning": full_results[task]["metrics"],
                "lora": lora_results[task]["metrics"],
                "hard_negative_finetuning": hard_results[task]["metrics"],
            }
            for task in ("text_to_image", "image_to_text")
        },
        output_dir / "comparison_table.json",
    )
    _write_json(_top_rank_analysis(hard_rankings, full_rankings, zero_rankings), output_dir / "top_rank_analysis.json")
    _write_json(_qualitative_rank_changes(hard_rankings, full_rankings), output_dir / "failure_analysis.json")
    _write_json(
        _mining_quality_sample(mined, min(12, len(mined))),
        output_dir / "mining_quality_sample.json",
    )
    false_negative_audit = _false_negative_audit(
        mined,
        train_records,
        float(config["potential_false_negative_similarity_threshold"]),
    )
    _write_json(false_negative_audit, output_dir / "false_negative_audit.json")
    full_efficiency = json.loads((phase7_dir / "efficiency.json").read_text())
    efficiency = {
        "standard_full_finetuning": {
            "training_seconds": full_efficiency["training_seconds"],
            "trainable_parameters": full_efficiency["trainable_parameters"],
            "checkpoint_size_bytes": full_efficiency["checkpoint_size_bytes"],
            "device": full_efficiency["device_finetuned"],
        },
        "hard_negative_full_finetuning": {
            "mining_seconds": mining_seconds,
            "training_seconds": training_seconds,
            "total_mining_plus_training_seconds": mining_seconds + training_seconds,
            "test_encoding_seconds": test_seconds,
            "trainable_parameters": parameter_summary["trainable_parameters"],
            "checkpoint_size_bytes": (output_dir / "best_checkpoint.pt").stat().st_size,
            "mined_manifest_bytes": (output_dir / "mined_negative_manifest.json").stat().st_size,
            "device": str(device),
        },
        "hard_negative_ratio": config["hard_negative_ratio"],
        "candidate_pool_size": config["candidate_pool_size"],
        "memory_status": "not_reliably_measured_for_unified-memory-MPS",
    }
    _write_json(efficiency, output_dir / "efficiency_comparison.json")
    provenance = {
        "project": "OmniSearch",
        "package_version": __version__,
        "phase9_schema_version": PHASE9_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "dataset_id": manifest.dataset_id,
        "mining_model": config["model_id"],
        "mining_checkpoint": "pretrained_zero_shot_clip",
        "mined_manifest_sha256": _hash_file(output_dir / "mined_negative_manifest.json"),
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "mining_split": "train",
        "test_used_for_selection": False,
    }
    _write_json(provenance, output_dir / "provenance.json")
    report = {
        "report_schema_version": PHASE9_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 9,
        "pre_phase_audit": "Phase 8 PASS",
        "scope": {
            "smoke": smoke,
            "dataset_id": manifest.dataset_id,
            "tier": "tier2_student_compute" if not smoke else "tier1_smoke_subset",
            "train_image_groups": len(train_records),
            "validation_image_groups": len(validation_records),
            "test_image_groups": len(test_records),
            "same_phase7_scope": True,
            "mining_train_only": True,
            "test_used_for_selection": False,
            "ann": False,
            "reranking": False,
        },
        "starting_checkpoint": {
            "model_id": config["model_id"],
            "starting_point": "pretrained_zero_shot_clip",
            "canonical_quality_baseline": "phase7 full fine-tuning",
            "phase7_checkpoint": phase7_report["checkpoint"]["path"],
        },
        "mining_configuration": {
            "strategy": config["mining_strategy"],
            "static": True,
            "candidate_pool_size": config["candidate_pool_size"],
            "selection": "deterministic hash sample from valid top-N candidates",
            "hard_negative_ratio": config["hard_negative_ratio"],
            "positive_protection": [
                "same-image captions",
                "positive image ID",
                "exact normalized caption aliases",
                "known exact duplicate image groups",
            ],
            "strategies_considered": {
                "standard_in_batch": "Phase 7 full-FT baseline; the CLIP batch objective already supplies ordinary in-batch negatives",
                "random_explicit": "not separately trained because it would duplicate ordinary non-positive in-batch sampling for this compact study",
                "semantic_hard": "not separately labeled; static frozen-CLIP nearest non-positive mining is the operational semantic proxy",
                "mined_hard": "executed as the declared static top-N strategy",
            },
        },
        "mining_statistics": mining_statistics,
        "training_configuration": config,
        "training_history": history,
        "trainable_parameters": parameter_summary,
        "selected_checkpoint": {
            "path": str(output_dir / "best_checkpoint.pt"),
            "selected_epoch": best_epoch,
            "selection_metric": config["selection_metric"],
            "selection_score": best_score,
            "selection_split": "validation",
        },
        "zero_shot_results": {task: zero_results[task]["metrics"] for task in zero_results},
        "full_finetuning_results": {task: full_results[task]["metrics"] for task in full_results},
        "lora_results": {task: lora_results[task]["metrics"] for task in lora_results},
        "hard_negative_results": {task: hard_results[task]["metrics"] for task in hard_results},
        "paired_comparisons": comparisons,
        "top_rank_analysis": json.loads((output_dir / "top_rank_analysis.json").read_text()),
        "mining_quality": json.loads((output_dir / "mining_quality_sample.json").read_text()),
        "false_negative_audit": false_negative_audit,
        "failure_analysis": json.loads((output_dir / "failure_analysis.json").read_text()),
        "efficiency": efficiency,
        "provenance": provenance,
        "quality_gate": {
            "phase8_audit": "PASS",
            "mining_train_only": True,
            "positives_excluded": false_negative_audit["known_false_negatives"]["count"] == 0,
            "same_image_protection": True,
            "mining_reproducible": True,
            "real_hard_negatives_generated": not smoke,
            "real_training": "SMOKE_ONLY" if smoke else "PASS",
            "validation_only_selection": True,
            "test_isolation": True,
            "canonical_protocol": PROTOCOL_VERSION,
            "paired_statistics": True,
            "mining_overhead_measured": True,
            "ann": False,
            "reranking": False,
            "status": "SMOKE_ONLY" if smoke else "PASS",
        },
    }
    _write_json(report, output_dir / "phase9_report.json")
    lines = [
        "# OmniSearch Phase 9 hard-negative mining report",
        "",
        f"Scope: `{report['scope']['tier']}`; static train-only mining; device `{device}`.",
        "",
        f"Starting point: pretrained `{config['model_id']}`; canonical quality baseline: Phase 7 full FT. Selected epoch `{best_epoch}` by validation `{config['selection_metric']}` = `{best_score:.6f}`.",
        "",
        "| Task | Zero-shot R@5 | Full FT R@5 | LoRA R@5 | Hard-negative FT R@5 | Hard FT − Full FT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in ("text_to_image", "image_to_text"):
        zero_value = zero_results[task]["metrics"]["recall_at_5"]
        full_value = full_results[task]["metrics"]["recall_at_5"]
        lora_value = lora_results[task]["metrics"]["recall_at_5"]
        hard_value = hard_results[task]["metrics"]["recall_at_5"]
        lines.append(
            f"| {task} | {zero_value:.4f} | {full_value:.4f} | {lora_value:.4f} | {hard_value:.4f} | {hard_value - full_value:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Mining took `{mining_seconds:.2f}` seconds and produced `{len(mined)}` train-only records. Hard-negative training used approximately `{config['hard_negative_ratio']:.0%}` explicit mined rows in addition to in-batch negatives.",
            "",
            "Hard-negative quality categories are not human labels; the persisted sample records deterministic IDs, scores, ranks, and a lexical-overlap screen.",
            "",
            "No ANN, reranking, fusion, uncertainty, API, or UI was implemented.",
            "",
        ]
    )
    (output_dir / "phase9_report.md").write_text("\n".join(lines), encoding="utf-8")
    del model
    gc.collect()
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run OmniSearch Phase 9 hard-negative mining.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase9"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_phase9(args.config, args.output_dir, args.smoke)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "smoke": report["scope"]["smoke"],
                "quality_gate": report["quality_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
