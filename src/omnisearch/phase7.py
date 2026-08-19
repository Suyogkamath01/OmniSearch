"""Phase 7 full-parameter CLIP fine-tuning and contrastive evaluation.

The implementation deliberately keeps the experiment conventional: one
caption is sampled per image group for each training epoch, all CLIP parameters
and the logit-scale parameter are trainable, and checkpoint selection uses
validation retrieval only. The test split is opened only after the selected
checkpoint is fixed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .clip_baseline import DEFAULT_MODEL_ID
from .config import DEFAULT_CONFIG_PATH, load_config
from .evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    bootstrap_ci,
    build_result,
    compare_systems,
    make_protocol,
    ranking_from_scores,
    write_result_artifacts,
)
from .manifest import DatasetManifest, ImageRecord, read_manifest
from .splitting import assert_no_split_leakage

PHASE7_SCHEMA_VERSION = 1
DEFAULT_PHASE7_CONFIG: dict[str, Any] = {
    "manifest": "data/processed/coco2017_val_split_manifest.json",
    "image_root": "data/raw/coco2017/val2017",
    "model_id": DEFAULT_MODEL_ID,
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
    "max_train_images": None,
    "max_validation_images": None,
    "max_test_images": None,
    "subset_seed": None,
}


@dataclass(frozen=True)
class TrainingPair:
    image_id: str
    caption_id: str
    text: str
    image_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _read_phase7_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as file:
        raw = tomllib.load(file)
    values = dict(DEFAULT_PHASE7_CONFIG)
    values.update(dict(raw.get("phase7", {})))
    return values


def validate_phase7_config(config: Mapping[str, Any]) -> None:
    """Reject settings that could silently violate the Phase 7 contract."""

    for key in ("train_split", "validation_split", "test_split"):
        if key in config:
            expected = {
                "train_split": "train",
                "validation_split": "validation",
                "test_split": "test",
            }[key]
            if config[key] != expected:
                raise ValueError(f"{key} must remain {expected!r}")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if int(config["gradient_accumulation_steps"]) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(config.get("num_workers", 0)) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    if str(config["precision"]) not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if str(config["selection_metric"]) not in {"mean_recall_at_5", "mean_recall_at_1"}:
        raise ValueError("selection_metric must be mean_recall_at_1 or mean_recall_at_5")
    for key in ("max_train_images", "max_validation_images", "max_test_images"):
        value = config.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} must be positive when supplied")


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _subset_records(
    records: Sequence[ImageRecord], split: str, seed: int, limit: int | None
) -> tuple[ImageRecord, ...]:
    selected = tuple(record for record in records if record.split == split)
    if limit is None or limit >= len(selected):
        return selected
    ordered = sorted(selected, key=lambda record: _stable_key(seed, record.image_id))
    selected_ids = {record.image_id for record in ordered[:limit]}
    return tuple(record for record in selected if record.image_id in selected_ids)


def build_training_pairs(
    records: Sequence[ImageRecord], image_root: Path | str, seed: int, epoch: int = 0
) -> tuple[TrainingPair, ...]:
    """Select one positive caption per image, never duplicate image negatives."""

    root = Path(image_root)
    pairs: list[TrainingPair] = []
    for record in records:
        if record.filename is None:
            raise ValueError(f"training record has no image filename: {record.image_id}")
        if not record.captions:
            raise ValueError(f"training record has no captions: {record.image_id}")
        ordered = sorted(
            record.captions,
            key=lambda caption: _stable_key(
                seed + epoch, f"{record.image_id}\0{caption.caption_id}"
            ),
        )
        caption = ordered[0]
        pairs.append(
            TrainingPair(
                image_id=record.image_id,
                caption_id=caption.caption_id,
                text=caption.text,
                image_path=str(root / record.filename),
            )
        )
    return tuple(pairs)


def contrastive_targets(batch_size: int, device: Any) -> Any:
    """Return the diagonal positive-pair targets for a unique-image batch."""

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return torch.arange(batch_size, device=device)


def symmetric_clip_loss(
    image_embeddings: Any,
    text_embeddings: Any,
    logit_scale: Any,
) -> tuple[Any, dict[str, float]]:
    """Compute the symmetric CLIP image→text/text→image objective."""

    import torch
    from torch.nn import functional

    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2 tensors")
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("image and text embedding shapes must match")
    if image_embeddings.shape[0] <= 0:
        raise ValueError("contrastive batches must be non-empty")
    image_normalized = functional.normalize(image_embeddings, dim=-1)
    text_normalized = functional.normalize(text_embeddings, dim=-1)
    safe_logit_scale = torch.exp(torch.clamp(logit_scale, max=math.log(100.0)))
    logits = safe_logit_scale * image_normalized @ text_normalized.transpose(0, 1)
    targets = torch.arange(logits.shape[0], device=logits.device)
    image_to_text = functional.cross_entropy(logits, targets)
    text_to_image = functional.cross_entropy(logits.transpose(0, 1), targets)
    loss = (image_to_text + text_to_image) / 2.0
    return loss, {
        "image_to_text_loss": float(image_to_text.detach().item()),
        "text_to_image_loss": float(text_to_image.detach().item()),
        "logit_scale": float(safe_logit_scale.detach().item()),
    }


def _parameter_summary(model: Any) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(
        int(parameter.numel()) for parameter in parameters if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
        "temperature_parameter": "model.logit_scale",
        "temperature_trainable": bool(
            getattr(getattr(model, "logit_scale", None), "requires_grad", False)
        ),
    }


def _load_trainable_model(model_id: str, requested_device: str) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Phase 7 requires torch, transformers, and Pillow") from exc
    from .clip_baseline import select_device

    device = select_device(requested_device, torch)
    processor = CLIPProcessor.from_pretrained(model_id)
    model: Any = CLIPModel.from_pretrained(model_id)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    return model, processor, torch, device


def _move_inputs(processed: Any, device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in processed.items()
    }


def _load_rgb_image(path: str) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        return source.convert("RGB")


def _load_rgb_images(paths: Sequence[str], num_workers: int) -> list[Any]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if num_workers == 0 or len(paths) < 2:
        return [_load_rgb_image(path) for path in paths]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(_load_rgb_image, paths))


def _autocast_context(torch: Any, device: Any, precision: str) -> Any:
    if precision == "fp32":
        return nullcontext()
    device_type = str(device).split(":", maxsplit=1)[0]
    if device_type not in {"mps", "cuda"}:
        raise ValueError("fp16 precision requires an MPS or CUDA device")
    return torch.autocast(device_type=device_type, dtype=torch.float16)


def _batch_inputs(
    processor: Any,
    pairs: Sequence[TrainingPair],
    device: str,
    text_max_length: int,
    num_workers: int = 0,
) -> dict[str, Any]:
    images: list[Any] = []
    try:
        images = _load_rgb_images(
            [pair.image_path for pair in pairs], num_workers
        )
        processed = processor(
            text=[pair.text for pair in pairs],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        return _move_inputs(processed, device)
    finally:
        for image in images:
            close = getattr(image, "close", None)
            if close:
                close()


def _feature_tensor(output: Any) -> Any:
    """Extract projected CLIP embeddings across supported Transformers APIs."""

    feature = getattr(output, "pooler_output", output)
    if not hasattr(feature, "ndim") or feature.ndim != 2:
        raise ValueError("CLIP feature output must be a rank-2 tensor")
    return feature


def _features(model: Any, inputs: Mapping[str, Any]) -> tuple[Any, Any]:
    image_features = _feature_tensor(
        model.get_image_features(pixel_values=inputs["pixel_values"])
    )
    text_inputs = {key: value for key, value in inputs.items() if key != "pixel_values"}
    text_features = _feature_tensor(model.get_text_features(**text_inputs))
    return image_features, text_features


def _train_epoch(
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    pairs: Sequence[TrainingPair],
    batch_size: int,
    gradient_accumulation_steps: int,
    text_max_length: int,
    max_grad_norm: float,
    first_parameter_before: Any | None,
    num_workers: int = 0,
    precision: str = "fp32",
) -> tuple[dict[str, Any], Any | None, bool]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    image_losses: list[float] = []
    text_losses: list[float] = []
    scales: list[float] = []
    optimizer_steps = 0
    update_verified = False
    gradients_finite = True
    gradient_norm = 0.0
    parameter_before = first_parameter_before
    batch_count = math.ceil(len(pairs) / batch_size)
    for batch_index, start in enumerate(range(0, len(pairs), batch_size)):
        batch = pairs[start : start + batch_size]
        model_device = (
            model.device
            if hasattr(model, "device")
            else str(next(model.parameters()).device)
        )
        inputs = _batch_inputs(
            processor, batch, model_device, text_max_length, num_workers
        )
        with _autocast_context(torch, model_device, precision):
            image_features, text_features = _features(model, inputs)
            loss, details = symmetric_clip_loss(
                image_features, text_features, model.logit_scale
            )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}")
        losses.append(float(loss.detach().item()))
        image_losses.append(details["image_to_text_loss"])
        text_losses.append(details["text_to_image_loss"])
        scales.append(details["logit_scale"])
        (loss / gradient_accumulation_steps).backward()
        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == batch_count
        )
        if should_step:
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients:
                raise RuntimeError("backward pass produced no gradients")
            if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
                gradients_finite = False
                raise FloatingPointError(f"non-finite gradient at batch {batch_index}")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if parameter_before is not None and not update_verified:
                current = next(parameter for parameter in model.parameters() if parameter.requires_grad)
                update_verified = not bool(torch.equal(parameter_before.cpu(), current.detach().cpu()))
    if not update_verified and parameter_before is not None:
        raise RuntimeError("optimizer step did not change a trainable parameter")
    return {
        "loss": math.fsum(losses) / len(losses),
        "image_to_text_loss": math.fsum(image_losses) / len(image_losses),
        "text_to_image_loss": math.fsum(text_losses) / len(text_losses),
        "logit_scale": math.fsum(scales) / len(scales),
        "batches": batch_count,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "gradient_norm_last": gradient_norm,
        "gradients_finite": gradients_finite,
        "parameter_update_verified": update_verified,
    }, parameter_before, update_verified


def _validation_loss(
    model: Any,
    processor: Any,
    torch: Any,
    pairs: Sequence[TrainingPair],
    batch_size: int,
    text_max_length: int,
    num_workers: int = 0,
    precision: str = "fp32",
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            model_device = (
                model.device
                if hasattr(model, "device")
                else str(next(model.parameters()).device)
            )
            inputs = _batch_inputs(
                processor,
                pairs[start : start + batch_size],
                model_device,
                text_max_length,
                num_workers,
            )
            with _autocast_context(torch, model_device, precision):
                image_features, text_features = _features(model, inputs)
                loss, _ = symmetric_clip_loss(
                    image_features, text_features, model.logit_scale
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite validation loss")
            values.append(float(loss.item()))
    return math.fsum(values) / len(values) if values else float("nan")


def _encode_images(
    model: Any,
    processor: Any,
    torch: Any,
    records: Sequence[ImageRecord],
    image_root: Path,
    batch_size: int,
    num_workers: int = 0,
    precision: str = "fp32",
) -> tuple[tuple[str, ...], Any]:
    from torch.nn import functional

    rows: list[Any] = []
    ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            current = records[start : start + batch_size]
            images: list[Any] = []
            try:
                paths: list[str] = []
                for record in current:
                    if record.filename is None:
                        raise ValueError(f"record has no filename: {record.image_id}")
                    paths.append(str(image_root / record.filename))
                images = _load_rgb_images(paths, num_workers)
                processed = processor(images=images, return_tensors="pt")
                inputs = _move_inputs(processed, model.device if hasattr(model, "device") else str(next(model.parameters()).device))
                with _autocast_context(
                    torch,
                    model.device
                    if hasattr(model, "device")
                    else str(next(model.parameters()).device),
                    precision,
                ):
                    features = _feature_tensor(
                        model.get_image_features(pixel_values=inputs["pixel_values"])
                    )
                rows.append(features.detach().cpu())
                ids.extend(record.image_id for record in current)
            finally:
                for image in images:
                    close = getattr(image, "close", None)
                    if close:
                        close()
    if not rows:
        raise ValueError("cannot encode an empty image set")
    return tuple(ids), functional.normalize(torch.cat(rows, dim=0), dim=-1).cpu()


def _encode_texts(
    model: Any,
    processor: Any,
    torch: Any,
    items: Sequence[tuple[str, str]],
    batch_size: int,
    text_max_length: int,
    num_workers: int = 0,
    precision: str = "fp32",
) -> tuple[tuple[str, ...], Any]:
    from torch.nn import functional

    rows: list[Any] = []
    ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            current = items[start : start + batch_size]
            processed = processor(
                text=[text for _, text in current],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=text_max_length,
            )
            inputs = _move_inputs(processed, model.device if hasattr(model, "device") else str(next(model.parameters()).device))
            text_inputs = {key: value for key, value in inputs.items() if key != "pixel_values"}
            with _autocast_context(
                torch,
                model.device
                if hasattr(model, "device")
                else str(next(model.parameters()).device),
                precision,
            ):
                features = _feature_tensor(model.get_text_features(**text_inputs))
            rows.append(features.detach().cpu())
            ids.extend(item_id for item_id, _ in current)
    if not rows:
        raise ValueError("cannot encode an empty text set")
    return tuple(ids), functional.normalize(torch.cat(rows, dim=0), dim=-1).cpu()


def _ranking_records(
    task: str,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    scores: Any,
    relevant: Mapping[str, set[str]],
    candidate_corpus_id: str,
    system_id: str,
    experiment_id: str,
    top_k: int = 10,
) -> tuple[RankingRecord, ...]:
    records: list[RankingRecord] = []
    for index, query_id in enumerate(query_ids):
        row = scores[index].tolist()
        candidates = sorted(
            zip(candidate_ids, row), key=lambda item: (-float(item[1]), str(item[0]))
        )[: min(top_k, len(candidate_ids))]
        records.append(
            ranking_from_scores(
                query_id=query_id,
                task=task,
                candidates=[(str(item_id), float(score)) for item_id, score in candidates],
                relevant_ids=relevant[query_id],
                system_id=system_id,
                experiment_id=experiment_id,
                candidate_count=len(candidate_ids),
                candidate_corpus_id=candidate_corpus_id,
            )
        )
    return tuple(records)


def _result_for_rankings(
    rankings: Sequence[RankingRecord],
    manifest: DatasetManifest,
    manifest_path: Path,
    config_path: Path,
    split: str,
    seed: int,
    system_id: str,
    experiment_id: str,
    model_info: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    task = rankings[0].task
    protocol = make_protocol(task)
    uncertainty = {
        f"recall_at_{k}": bootstrap_ci(
            rankings, "recall", k, bootstrap_resamples, 0.95, seed
        )
        for k in protocol["k_values"]
    }
    result = build_result(
        rankings,
        protocol,
        {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "manifest_sha256": _hash_file(manifest_path),
            "subset": f"{split}_selected_image_groups",
        },
        split,
        experiment_id,
        system_id,
        model_info,
        seed,
        uncertainty,
        runtime_info,
        {"platform": platform.platform(), "python": sys.version, "device": runtime_info.get("device")},
        {
            "config_sha256": _hash_file(config_path),
            "manifest_sha256": _hash_file(manifest_path),
            "source_sha256": manifest.source_sha256,
            "protocol_version": PROTOCOL_VERSION,
        },
    )
    result["ranking_records"] = [ranking.to_dict() for ranking in rankings]
    return result


def _hash_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_model(
    model: Any,
    processor: Any,
    torch: Any,
    records: Sequence[ImageRecord],
    image_root: Path,
    batch_size: int,
    text_max_length: int,
    manifest: DatasetManifest,
    manifest_path: Path,
    config_path: Path,
    split: str,
    seed: int,
    system_id: str,
    experiment_id: str,
    model_info: Mapping[str, Any],
    bootstrap_resamples: int,
    num_workers: int = 0,
    precision: str = "fp32",
) -> dict[str, Any]:
    started = time.perf_counter()
    image_ids, image_embeddings = _encode_images(
        model,
        processor,
        torch,
        records,
        image_root,
        batch_size,
        num_workers,
        precision,
    )
    caption_items = [
        (caption.caption_id, caption.text)
        for record in records
        for caption in record.captions
    ]
    caption_ids, text_embeddings = _encode_texts(
        model,
        processor,
        torch,
        caption_items,
        batch_size,
        text_max_length,
        num_workers,
        precision,
    )
    image_to_captions = {
        record.image_id: {caption.caption_id for caption in record.captions}
        for record in records
    }
    caption_to_image = {
        caption.caption_id: {record.image_id}
        for record in records
        for caption in record.captions
    }
    similarities = image_embeddings @ text_embeddings.transpose(0, 1)
    image_corpus = f"{manifest.dataset_id}:{split}:images:{len(image_ids)}"
    caption_corpus = f"{manifest.dataset_id}:{split}:captions:{len(caption_ids)}"
    text_to_image = _ranking_records(
        "text_to_image",
        caption_ids,
        image_ids,
        similarities.transpose(0, 1),
        caption_to_image,
        image_corpus,
        system_id,
        experiment_id,
    )
    image_to_text = _ranking_records(
        "image_to_text",
        image_ids,
        caption_ids,
        similarities,
        image_to_captions,
        caption_corpus,
        system_id,
        experiment_id,
    )
    elapsed = time.perf_counter() - started
    runtime = {
        "device": str(
            model.device
            if hasattr(model, "device")
            else next(model.parameters()).device
        ),
            "batch_size": batch_size,
            "num_workers": num_workers,
        "image_count": len(image_ids),
        "caption_count": len(caption_ids),
        "encoding_seconds": elapsed,
        "encoding_items_per_second": (len(image_ids) + len(caption_ids)) / elapsed if elapsed else None,
    }
    return {
        "results": {
            "text_to_image": _result_for_rankings(
                text_to_image, manifest, manifest_path, config_path, split, seed,
                system_id, experiment_id, model_info, runtime, bootstrap_resamples
            ),
            "image_to_text": _result_for_rankings(
                image_to_text, manifest, manifest_path, config_path, split, seed,
                system_id, experiment_id, model_info, runtime, bootstrap_resamples
            ),
        },
        "rankings": {"text_to_image": text_to_image, "image_to_text": image_to_text},
        "runtime": runtime,
    }


def _save_checkpoint(path: Path, model: Any, metadata: Mapping[str, Any]) -> None:
    import torch

    state = {
        key: value.detach().cpu() if hasattr(value, "detach") else value
        for key, value in model.state_dict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, "metadata": dict(metadata)}, path)


def _load_checkpoint(path: Path, model: Any) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return dict(payload.get("metadata", {}))


def _comparison_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "system_id": result["system_id"],
        "task": result["task"],
        "dataset_id": result["dataset"]["dataset_id"],
        "split": result["split"],
        "protocol_version": result["protocol"]["protocol_version"],
        "candidate_corpus_id": result["ranking_records"][0]["candidate_corpus_id"],
        "relevance_definition": result["ranking_records"][0]["relevance_definition"],
    }


def _rank_of_relevant(record: RankingRecord) -> int | None:
    for rank, item_id in enumerate(record.candidate_ids, start=1):
        if item_id in record.relevant_ids:
            return rank
    return None


def _qualitative_comparison(
    zero_rankings: Mapping[str, Sequence[RankingRecord]],
    fine_rankings: Mapping[str, Sequence[RankingRecord]],
    records: Sequence[ImageRecord],
) -> dict[str, Any]:
    caption_text = {
        caption.caption_id: caption.text
        for record in records
        for caption in record.captions
    }
    image_text = {
        record.image_id: record.captions[0].text if record.captions else ""
        for record in records
    }
    output: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        zero = {record.query_id: record for record in zero_rankings[task]}
        fine = {record.query_id: record for record in fine_rankings[task]}
        categories = {"improved": 0, "unchanged": 0, "degraded": 0}
        examples: dict[str, list[dict[str, Any]]] = {key: [] for key in categories}
        for query_id in sorted(zero):
            zero_rank = _rank_of_relevant(zero[query_id])
            fine_rank = _rank_of_relevant(fine[query_id])
            if zero_rank is None and fine_rank is not None:
                category = "improved"
            elif fine_rank is None and zero_rank is not None:
                category = "degraded"
            elif zero_rank is not None and fine_rank is not None and fine_rank < zero_rank:
                category = "improved"
            elif zero_rank is not None and fine_rank is not None and fine_rank > zero_rank:
                category = "degraded"
            else:
                category = "unchanged"
            categories[category] += 1
            if len(examples[category]) < 3:
                context = (
                    caption_text.get(query_id, "")
                    if task == "text_to_image"
                    else image_text.get(query_id, "")
                )
                examples[category].append(
                    {
                        "query_id": query_id,
                        "query_context": context,
                        "zero_shot_rank": zero_rank,
                        "fine_tuned_rank": fine_rank,
                        "zero_shot_top5": list(zero[query_id].candidate_ids[:5]),
                        "fine_tuned_top5": list(fine[query_id].candidate_ids[:5]),
                    }
                )
        output[task] = {"counts": categories, "examples": examples}
    return output


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_phase7(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase7",
    smoke: bool = False,
) -> dict[str, Any]:
    """Run a validation-selected full-parameter CLIP fine-tuning experiment."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_config = load_config(config_path)
    phase7_config = _read_phase7_config(config_path)
    if smoke:
        phase7_config.update(
            {
                "epochs": 1,
                "max_train_images": 8,
                "max_validation_images": 4,
                "max_test_images": 4,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "bootstrap_resamples": 10,
            }
        )
    validate_phase7_config(phase7_config)
    manifest_path = Path(str(phase7_config["manifest"]))
    image_root = Path(str(phase7_config["image_root"]))
    manifest = read_manifest(manifest_path)
    assert_no_split_leakage(manifest.records)
    if manifest.dataset_id != project_config.dataset_id:
        raise ValueError(
            f"Phase 7 dataset mismatch: config={project_config.dataset_id}, manifest={manifest.dataset_id}"
        )
    seed = project_config.seed
    subset_seed = seed if phase7_config.get("subset_seed") is None else int(phase7_config["subset_seed"])
    train_records = _subset_records(manifest.records, "train", subset_seed, phase7_config.get("max_train_images"))
    validation_records = _subset_records(manifest.records, "validation", subset_seed, phase7_config.get("max_validation_images"))
    if not train_records or not validation_records:
        raise ValueError("Phase 7 requires non-empty train and validation image groups")
    if any(record.split != "train" for record in train_records):
        raise AssertionError("training records crossed the train boundary")
    if any(record.split != "validation" for record in validation_records):
        raise AssertionError("validation records crossed the validation boundary")
    _write_json(
        {
            "phase7_schema_version": PHASE7_SCHEMA_VERSION,
            "smoke": smoke,
            "dataset_id": manifest.dataset_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _hash_file(manifest_path),
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "test_isolation": "test records are not loaded until the selected checkpoint is fixed",
            "config": phase7_config,
        },
        output_dir / "training_config.json",
    )
    model, processor, torch, device = _load_trainable_model(
        str(phase7_config["model_id"]), str(phase7_config["device"])
    )
    parameter_summary = _parameter_summary(model)
    if parameter_summary["trainable_parameters"] != parameter_summary["total_parameters"]:
        raise AssertionError("Phase 7 full fine-tuning requires all parameters trainable")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(phase7_config["learning_rate"]),
        weight_decay=float(phase7_config["weight_decay"]),
    )
    training_pairs = build_training_pairs(train_records, image_root, seed, epoch=0)
    steps_per_epoch = math.ceil(
        math.ceil(len(training_pairs) / int(phase7_config["batch_size"]))
        / int(phase7_config["gradient_accumulation_steps"])
    )
    total_steps = max(1, steps_per_epoch * int(phase7_config["epochs"]))
    warmup_steps = int(phase7_config["warmup_steps"])

    def schedule(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            return step / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_score = float("-inf")
    best_epoch: int | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    first_parameter_before = next(model.parameters()).detach().clone()
    training_started = time.perf_counter()
    for epoch in range(int(phase7_config["epochs"])):
        pairs = build_training_pairs(train_records, image_root, seed, epoch=epoch)
        epoch_started = time.perf_counter()
        train_stats, _, update_verified = _train_epoch(
            model,
            processor,
            optimizer,
            scheduler,
            torch,
            pairs,
            int(phase7_config["batch_size"]),
            int(phase7_config["gradient_accumulation_steps"]),
            int(phase7_config["text_max_length"]),
            float(phase7_config["max_grad_norm"]),
            first_parameter_before,
            int(phase7_config["num_workers"]),
            str(phase7_config["precision"]),
        )
        first_parameter_before = None
        validation_pairs = build_training_pairs(validation_records, image_root, seed, epoch=0)
        validation_loss = _validation_loss(
            model,
            processor,
            torch,
            validation_pairs,
            int(phase7_config["batch_size"]),
            int(phase7_config["text_max_length"]),
            int(phase7_config["num_workers"]),
            str(phase7_config["precision"]),
        )
        validation_eval = evaluate_model(
            model,
            processor,
            torch,
            validation_records,
            image_root,
            int(phase7_config["batch_size"]),
            int(phase7_config["text_max_length"]),
            manifest,
            manifest_path,
            config_path,
            "validation",
            seed,
            "phase7_validation_finetuned",
            f"phase7_validation_epoch_{epoch + 1}",
            {**parameter_summary, "frozen": False, "model_id": phase7_config["model_id"]},
            int(phase7_config["bootstrap_resamples"]),
            int(phase7_config["num_workers"]),
            str(phase7_config["precision"]),
        )
        validation_text = validation_eval["results"]["text_to_image"]["metrics"]
        validation_image = validation_eval["results"]["image_to_text"]["metrics"]
        metric_key = str(phase7_config["selection_metric"])
        metric_suffix = metric_key.removeprefix("mean_")
        selection_score = math.fsum(
            [
                float(validation_text[metric_suffix]),
                float(validation_image[metric_suffix]),
            ]
        ) / 2.0
        improved = selection_score > best_score
        if improved:
            best_score = selection_score
            best_epoch = epoch + 1
            no_improvement = 0
            _save_checkpoint(
                output_dir / "best_checkpoint.pt",
                model,
                {
                    "epoch": epoch + 1,
                    "selection_metric": metric_key,
                    "selection_score": selection_score,
                    "dataset_id": manifest.dataset_id,
                    "manifest_sha256": _hash_file(manifest_path),
                    "train_split": "train",
                    "validation_split": "validation",
                    "test_split": "test_not_used_for_selection",
                    "parent_model_id": phase7_config["model_id"],
                },
            )
        else:
            no_improvement += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_stats,
                "validation_loss": validation_loss,
                "validation_metrics": {
                    "text_to_image": validation_text,
                    "image_to_text": validation_image,
                    "selection_score": selection_score,
                    "selection_metric": metric_key,
                },
                "checkpoint_selected": improved,
                "best_epoch_so_far": best_epoch,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "parameter_update_verified": update_verified,
            }
        )
        _write_json(history, output_dir / "training_history.json")
        if no_improvement >= int(phase7_config["early_stopping_patience"]):
            break
    if best_epoch is None:
        raise RuntimeError("no validation-selected checkpoint was produced")
    _load_checkpoint(output_dir / "best_checkpoint.pt", model)
    _write_json(
        {
            "selected_checkpoint": str(output_dir / "best_checkpoint.pt"),
            "selection_split": "validation",
            "selection_metric": phase7_config["selection_metric"],
            "selection_score": best_score,
            "selected_epoch": best_epoch,
            "test_used_for_selection": False,
            "model": parameter_summary,
            "parent_pretrained_checkpoint": phase7_config["model_id"],
        },
        output_dir / "checkpoint_metadata.json",
    )

    # The test split is intentionally materialized only after checkpoint selection.
    test_records = _subset_records(manifest.records, "test", subset_seed, phase7_config.get("max_test_images"))
    if not test_records:
        raise ValueError("Phase 7 requires non-empty test image groups for final evaluation")
    fine_started = time.perf_counter()
    fine_eval = evaluate_model(
        model,
        processor,
        torch,
        test_records,
        image_root,
        int(phase7_config["batch_size"]),
        int(phase7_config["text_max_length"]),
        manifest,
        manifest_path,
        config_path,
        "test",
        seed,
        "phase7_finetuned_clip",
        "phase7_finetuned_clip_test",
        {**parameter_summary, "frozen": False, "model_id": phase7_config["model_id"]},
        int(phase7_config["bootstrap_resamples"]),
        int(phase7_config["num_workers"]),
        str(phase7_config["precision"]),
    )
    fine_seconds = time.perf_counter() - fine_started
    fine_results = fine_eval["results"]
    fine_rankings = fine_eval["rankings"]
    del model
    gc.collect()
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except AttributeError:
        pass
    frozen_model, frozen_processor, frozen_torch, frozen_device = _load_trainable_model(
        str(phase7_config["model_id"]), str(phase7_config["device"])
    )
    for parameter in frozen_model.parameters():
        parameter.requires_grad_(False)
    frozen_model.eval()
    frozen_started = time.perf_counter()
    frozen_eval = evaluate_model(
        frozen_model,
        frozen_processor,
        frozen_torch,
        test_records,
        image_root,
        int(phase7_config["batch_size"]),
        int(phase7_config["text_max_length"]),
        manifest,
        manifest_path,
        config_path,
        "test",
        seed,
        "phase7_zero_shot_clip",
        "phase7_zero_shot_clip_test",
        {
            "model_id": phase7_config["model_id"],
            "frozen": True,
            "total_parameters": parameter_summary["total_parameters"],
            "trainable_parameters": 0,
        },
        int(phase7_config["bootstrap_resamples"]),
        int(phase7_config["num_workers"]),
        str(phase7_config["precision"]),
    )
    frozen_seconds = time.perf_counter() - frozen_started
    frozen_results = frozen_eval["results"]
    frozen_rankings = frozen_eval["rankings"]
    qualitative = _qualitative_comparison(frozen_rankings, fine_rankings, test_records)
    comparisons: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        comparisons[task] = compare_systems(
            frozen_rankings[task],
            fine_rankings[task],
            _comparison_metadata(frozen_results[task]),
            _comparison_metadata(fine_results[task]),
            bootstrap_resamples=int(phase7_config["bootstrap_resamples"]),
            seed=seed,
        )
        _write_json(frozen_results[task], output_dir / f"zero_shot_{task}.json")
        _write_json(fine_results[task], output_dir / f"fine_tuned_{task}.json")
        write_result_artifacts(frozen_results[task], output_dir / f"zero_shot_{task}_summary")
        write_result_artifacts(fine_results[task], output_dir / f"fine_tuned_{task}_summary")
    _write_json(comparisons, output_dir / "paired_comparisons.json")
    _write_json(qualitative, output_dir / "qualitative_before_after.json")
    efficiency = {
        "device_finetuned": device,
        "device_zero_shot": frozen_device,
        "training_seconds": time.perf_counter() - training_started,
        "fine_tuned_test_encoding_seconds": fine_seconds,
        "zero_shot_test_encoding_seconds": frozen_seconds,
        "train_image_groups": len(train_records),
        "validation_image_groups": len(validation_records),
        "test_image_groups": len(test_records),
        "effective_batch_size": int(phase7_config["batch_size"]) * int(phase7_config["gradient_accumulation_steps"]),
        "batch_size": int(phase7_config["batch_size"]),
        "gradient_accumulation_steps": int(phase7_config["gradient_accumulation_steps"]),
        "epochs_completed": len(history),
        "checkpoint_size_bytes": (output_dir / "best_checkpoint.pt").stat().st_size,
        "trainable_parameters": parameter_summary["trainable_parameters"],
        "total_parameters": parameter_summary["total_parameters"],
        "precision": phase7_config["precision"],
        "precision_note": "fp16 uses autocast on MPS/CUDA; CPU fallback supports fp32; no CUDA assumption",
    }
    _write_json(efficiency, output_dir / "efficiency.json")
    provenance = {
        "project": "OmniSearch",
        "package": "omnisearch",
        "project_version": __version__,
        "phase7_schema_version": PHASE7_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _hash_file(manifest_path),
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "source_sha256": manifest.source_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "test_used_for_selection": False,
    }
    _write_json(provenance, output_dir / "provenance.json")
    scope_tier = "tier2_student_compute" if not smoke else "tier1_smoke_subset"
    report = {
        "report_schema_version": PHASE7_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 7,
        "scope": {
            "smoke": smoke,
            "dataset_id": manifest.dataset_id,
            "tier": scope_tier,
            "full_parameter_fine_tuning": True,
            "lora": False,
            "peft": False,
            "hard_negative_mining": False,
            "test_used_for_selection": False,
        },
        "dataset": {
            "manifest": str(manifest_path),
            "manifest_sha256": _hash_file(manifest_path),
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "train_image_groups": len(train_records),
            "validation_image_groups": len(validation_records),
            "test_image_groups": len(test_records),
            "pair_sampling": "one deterministic caption per image per epoch; same-image captions are not used as in-batch negatives",
        },
        "pretrained_checkpoint": {
            "model_id": phase7_config["model_id"],
            "temperature_logit_scale": "trainable",
            "preprocessing": "CLIPProcessor model-native image transform and tokenizer",
            "text_max_length": int(phase7_config["text_max_length"]),
        },
        "training_configuration": phase7_config,
        "trainable_parameters": parameter_summary,
        "training_history": history,
        "checkpoint": {
            "path": str(output_dir / "best_checkpoint.pt"),
            "selected_epoch": best_epoch,
            "selection_metric": phase7_config["selection_metric"],
            "selection_score": best_score,
            "selection_split": "validation",
        },
        "zero_shot_results": {task: frozen_results[task]["metrics"] for task in frozen_results},
        "fine_tuned_results": {task: fine_results[task]["metrics"] for task in fine_results},
        "paired_comparisons": comparisons,
        "qualitative": qualitative,
        "efficiency": efficiency,
        "provenance": provenance,
        "quality_gate": {
            "pre_phase_integrity_check": "PASS",
            "training_smoke": "PASS" if smoke else "not_this_run",
            "finite_loss": True,
            "finite_gradients": all(
                bool(item["train"].get("gradients_finite", False)) for item in history
            ),
            "weights_updated": all(item["parameter_update_verified"] for item in history),
            "canonical_phase5_protocol": PROTOCOL_VERSION,
            "test_isolation": True,
            "status": "SMOKE_ONLY" if smoke else "PASS",
        },
    }
    _write_json(report, output_dir / "phase7_report.json")
    lines = [
        "# OmniSearch Phase 7 CLIP fine-tuning report",
        "",
        f"Scope: `{scope_tier}`; full-parameter fine-tuning; device `{device}`.",
        "",
        f"Selected epoch: `{best_epoch}` by validation `{phase7_config['selection_metric']}` = `{best_score:.6f}`. Test selection: `False`.",
        "",
        "| Task | Zero-shot R@1 | Fine-tuned R@1 | Delta | Zero-shot R@5 | Fine-tuned R@5 | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("text_to_image", "image_to_text"):
        zero = frozen_results[task]["metrics"]
        fine = fine_results[task]["metrics"]
        lines.append(
            f"| {task} | {zero['recall_at_1']:.4f} | {fine['recall_at_1']:.4f} | {fine['recall_at_1'] - zero['recall_at_1']:+.4f} | {zero['recall_at_5']:.4f} | {fine['recall_at_5']:.4f} | {fine['recall_at_5'] - zero['recall_at_5']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "The test split was loaded only after validation checkpoint selection. Paired bootstrap comparisons are in `paired_comparisons.json`; actual before/after examples are in `qualitative_before_after.json`.",
            "",
            "Phase 8 features (LoRA/PEFT, hard negatives, ANN, reranking, apps) were not implemented.",
            "",
        ]
    )
    (output_dir / "phase7_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run OmniSearch Phase 7 CLIP fine-tuning.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase7"))
    parser.add_argument("--smoke", action="store_true", help="run the tiny training smoke only")
    args = parser.parse_args()
    report = run_phase7(args.config, args.output_dir, args.smoke)
    print(json.dumps({"output_dir": str(args.output_dir), "smoke": report["scope"]["smoke"], "quality_gate": report["quality_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
