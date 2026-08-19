"""Frozen zero-shot CLIP extraction and exact retrieval utilities.

The module keeps heavyweight imports optional so ordinary Phase 0--3 tests do
not require a model download. Phase 4 uses one canonical checkpoint through
Hugging Face Transformers and never calls a training API.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config
from .manifest import ImageRecord, read_manifest
from .splitting import SPLIT_NAMES, assert_no_split_leakage

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingBatch:
    ids: tuple[str, ...]
    embeddings: tuple[tuple[float, ...], ...]
    skipped: tuple[dict[str, str], ...] = ()

    @property
    def dimension(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0


@dataclass
class ClipRuntime:
    model: Any
    processor: Any
    torch: Any
    device: str
    model_id: str
    text_max_length: int
    embedding_dimension: int


def select_device(requested: str = "auto", torch_module: Any | None = None) -> str:
    """Select MPS when available, with an explicit CPU fallback."""

    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    torch_runtime: Any = torch_module
    if torch_runtime is None:
        try:
            import torch as imported_torch  # type: ignore[import-not-found]

            torch_runtime = imported_torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Phase 4") from exc
    mps_available = bool(
        getattr(
            getattr(torch_runtime.backends, "mps", None), "is_available", lambda: False
        )()
    )
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is unavailable")
        return "mps"
    if requested == "auto" and mps_available:
        return "mps"
    return "cpu"


def load_clip_runtime(
    model_id: str = DEFAULT_MODEL_ID,
    requested_device: str = "auto",
    text_max_length: int = 77,
) -> ClipRuntime:
    """Load one frozen CLIP checkpoint; this is the only model-loading path."""

    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            CLIPModel,
            CLIPProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Phase 4 requires the optional phase4 dependencies: torch, transformers, and Pillow"
        ) from exc
    device = select_device(requested_device, torch)
    processor = CLIPProcessor.from_pretrained(model_id)
    model: Any = CLIPModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection_dimension = int(getattr(model.config, "projection_dim", 0))
    if projection_dimension <= 0:
        raise RuntimeError(
            "CLIP checkpoint does not expose a valid projection dimension"
        )
    return ClipRuntime(
        model=model,
        processor=processor,
        torch=torch,
        device=device,
        model_id=model_id,
        text_max_length=text_max_length,
        embedding_dimension=projection_dimension,
    )


def normalize_embeddings(
    rows: Iterable[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """L2-normalize rows and reject zero, NaN, or infinite vectors."""

    normalized: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row in rows:
        values = tuple(float(value) for value in row)
        if dimension is None:
            dimension = len(values)
        if len(values) != dimension or not values:
            raise ValueError("embedding rows must have one non-empty dimension")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding contains NaN or infinite values")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise ValueError("embedding contains a zero vector")
        normalized.append(tuple(value / norm for value in values))
    return tuple(normalized)


def _output_rows(output: Any) -> list[list[float]]:
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    if hasattr(output, "detach"):
        output = output.detach().float().cpu().tolist()
    rows = [list(row) for row in output]
    if rows and not isinstance(rows[0], list):
        rows = [list(output)]
    return rows


def _model_inputs(processed: Any, device: str) -> dict[str, Any]:
    if hasattr(processed, "items"):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
    if hasattr(processed, "to"):
        processed = processed.to(device)
    return dict(processed)


def encode_texts(
    items: Sequence[tuple[str, str]],
    runtime: ClipRuntime,
    batch_size: int = 8,
) -> EmbeddingBatch:
    """Encode text IDs in input order with frozen, no-gradient inference."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    valid: list[tuple[str, str]] = []
    skipped: list[dict[str, str]] = []
    for item_id, text in items:
        if not isinstance(text, str):
            skipped.append({"id": str(item_id), "reason": "text_is_not_string"})
        elif not text.strip():
            skipped.append({"id": str(item_id), "reason": "empty_text"})
        else:
            valid.append((str(item_id), text))
    output_ids: list[str] = []
    output_rows: list[Sequence[float]] = []
    with runtime.torch.inference_mode():
        for start in range(0, len(valid), batch_size):
            batch = valid[start : start + batch_size]
            processed = runtime.processor(
                text=[text for _, text in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=runtime.text_max_length,
            )
            inputs = _model_inputs(processed, runtime.device)
            output = runtime.model.get_text_features(**inputs)
            rows = _output_rows(output)
            if len(rows) != len(batch):
                raise ValueError(
                    "CLIP text output batch size does not match input batch"
                )
            output_ids.extend(item_id for item_id, _ in batch)
            output_rows.extend(rows)
    embeddings = normalize_embeddings(output_rows)
    if embeddings and len(embeddings[0]) != runtime.embedding_dimension:
        raise ValueError("CLIP text embedding dimension does not match model metadata")
    return EmbeddingBatch(tuple(output_ids), embeddings, tuple(skipped))


def encode_images(
    items: Sequence[tuple[str, Path | str]],
    runtime: ClipRuntime,
    batch_size: int = 8,
) -> EmbeddingBatch:
    """Encode readable image paths in deterministic input order."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Pillow is required for CLIP image preprocessing") from exc
    valid: list[tuple[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item_id, raw_path in items:
        path = Path(raw_path)
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            skipped.append({"id": str(item_id), "path": str(path), "reason": str(exc)})
            continue
        valid.append((str(item_id), image))
    output_ids: list[str] = []
    output_rows: list[Sequence[float]] = []
    with runtime.torch.inference_mode():
        for start in range(0, len(valid), batch_size):
            batch = valid[start : start + batch_size]
            processed = runtime.processor(
                images=[image for _, image in batch],
                return_tensors="pt",
            )
            inputs = _model_inputs(processed, runtime.device)
            output = runtime.model.get_image_features(**inputs)
            rows = _output_rows(output)
            if len(rows) != len(batch):
                raise ValueError(
                    "CLIP image output batch size does not match input batch"
                )
            output_ids.extend(item_id for item_id, _ in batch)
            output_rows.extend(rows)
    for _, image in valid:
        close = getattr(image, "close", None)
        if close:
            close()
    embeddings = normalize_embeddings(output_rows)
    if embeddings and len(embeddings[0]) != runtime.embedding_dimension:
        raise ValueError("CLIP image embedding dimension does not match model metadata")
    return EmbeddingBatch(tuple(output_ids), embeddings, tuple(skipped))


def exact_rank(
    query: Sequence[float],
    candidates: EmbeddingBatch,
    top_k: int = 10,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    """Rank normalized embeddings by exact cosine/dot-product similarity."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_vector = normalize_embeddings([query])[0]
    if candidates.embeddings and len(query_vector) != candidates.dimension:
        raise ValueError("query and candidate embedding dimensions differ")
    ranked: list[dict[str, Any]] = []
    for item_id, vector in zip(candidates.ids, candidates.embeddings):
        if exclude_id is not None and item_id == exclude_id:
            continue
        score = sum(left * right for left, right in zip(query_vector, vector))
        ranked.append({"id": item_id, "score": float(score)})
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return ranked[:top_k]


def rank_batch(
    queries: EmbeddingBatch,
    candidates: EmbeddingBatch,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    return {
        query_id: exact_rank(vector, candidates, top_k=top_k)
        for query_id, vector in zip(queries.ids, queries.embeddings)
    }


def rank_metrics(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    relevant: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 5, 10),
    task: str = "text_to_image",
) -> dict[str, Any]:
    from .evaluation import evaluate_rankings as canonical_evaluate
    from .evaluation import hit_rate_at_k, ranking_from_scores

    if not rankings:
        return {"queries_evaluated": 0, "status": "not_evaluated_no_relevant_queries"}
    records = tuple(
        ranking_from_scores(
            query_id=query_id,
            task=task,
            candidates=[
                (str(item["id"]), float(item["score"])) for item in rankings[query_id]
            ],
            relevant_ids=relevant.get(query_id, set()),
            system_id="clip_exact",
            experiment_id="clip_compatibility",
            candidate_count=len(rankings[query_id]),
            candidate_corpus_id="clip_compatibility",
            relevance_definition="declared producer relevance",
        )
        for query_id in rankings
    )
    result = canonical_evaluate(records, ks)
    ranks = result["rank_statistics"]
    for k in ks:
        canonical_key = f"recall_at_{k}"
        result[f"relevant_fraction_recall_at_{k}"] = result.get(canonical_key)
        hit_values = [
            value
            for record in records
            if record.relevant_ids and (value := hit_rate_at_k(record, k)) is not None
        ]
        result[canonical_key] = (
            sum(hit_values) / len(hit_values) if hit_values else None
        )
    result["recall_definition"] = (
        "query success: at least one relevant item in top K (legacy CLIP report compatibility)"
    )
    result["median_rank"] = ranks["median_first_relevant_rank"]
    result["mean_rank"] = ranks["mean_first_relevant_rank"]
    if result["queries_evaluated"] == 0:
        result["status"] = "not_evaluated_no_relevant_queries"
    return result


def statistics_median(values: Sequence[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_embedding_cache(
    path: Path | str, batch: EmbeddingBatch, metadata: Mapping[str, Any]
) -> None:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "ids": list(batch.ids),
        "embeddings": [list(row) for row in batch.embeddings],
        "skipped": list(batch.skipped),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_embedding_cache(
    path: Path | str, expected_metadata: Mapping[str, Any]
) -> EmbeddingBatch:
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported embedding cache schema")
    if payload.get("metadata") != dict(expected_metadata):
        raise ValueError("embedding cache metadata does not match requested run")
    ids = tuple(str(value) for value in payload.get("ids", []))
    embeddings = normalize_embeddings(payload.get("embeddings", []))
    if len(ids) != len(embeddings):
        raise ValueError("embedding cache IDs and vectors have different lengths")
    skipped = tuple(dict(value) for value in payload.get("skipped", []))
    return EmbeddingBatch(ids, embeddings, skipped)


def model_metadata(runtime: ClipRuntime) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "Pillow", "safetensors"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    parameters = tuple(runtime.model.parameters())
    parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    parameter_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in parameters
    )
    return {
        "model_id": runtime.model_id,
        "architecture": "CLIP ViT-B/32",
        "pretrained_source": "OpenAI CLIP ViT-B/32 checkpoint, distributed through Hugging Face Transformers; source repository https://github.com/openai/CLIP",
        "embedding_dimension": runtime.embedding_dimension,
        "image_input": "model processor; 224x224 CLIP preprocessing for ViT-B/32",
        "tokenizer": "CLIP BPE tokenizer, truncated to configured max length",
        "license_note": "MIT applies to the OpenAI CLIP repository software. This does not by itself establish a separate license for checkpoint weights or benchmark images; follow the model/checkpoint terms and the selected dataset terms.",
        "frozen": True,
        "trainable_parameter_count": 0,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "device": runtime.device,
        "package_versions": versions,
    }


def _stable_ids(values: Iterable[str], seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )


def _records_caption_items(
    records: Iterable[ImageRecord],
) -> list[tuple[str, str, str]]:
    return [
        (caption.caption_id, caption.text, record.image_id)
        for record in records
        for caption in record.captions
    ]


def _phase4_markdown(report: Mapping[str, Any]) -> str:
    dataset_id = report["provenance"]["dataset_id"]
    lines = [
        "# OmniSearch Phase 4 zero-shot CLIP baseline",
        "",
        f"Generated: `{report['provenance']['generated_at_utc']}`",
        "",
        "## Scope",
        "",
        f"- Phase 3 audit: **{report['pre_phase_audit']['phase3_audit']}**",
        f"- Zero-shot CLIP pipeline verified: **{report['scope']['zero_shot_clip_pipeline_verified']}**",
        f"- Real `{dataset_id}` zero-shot evaluation: **{report['scope']['real_dataset_evaluation']}**",
        f"- Model frozen: **{report['model']['frozen']}**",
        "",
        "## Model",
        "",
        f"- Checkpoint: `{report['model']['model_id']}`",
        f"- Architecture: `{report['model']['architecture']}`",
        f"- Embedding dimension: `{report['model']['embedding_dimension']}`",
        f"- Device: `{report['model']['device']}`",
        "",
        "## Evaluation",
        "",
        f"- Split: `{report['evaluation']['split']}`",
        f"- Images evaluated: `{report['evaluation']['images']}`",
        f"- Captions evaluated: `{report['evaluation']['captions']}`",
        f"- Text embedding smoke count: `{report['evaluation']['text_embedding_smoke_count']}`",
        f"- Fixture smoke artifact: `{report['fixture_artifacts'][0] if report['fixture_artifacts'] else 'not-run'}`",
        "",
        "Cross-modal metrics are reported only when the selected dataset image root was locally verified. Fixture smoke results, when present, are labelled as fixture-only.",
        "",
        "## Limitations",
        "",
        "No fine-tuning or task-specific optimization was performed. Image-rights and checkpoint terms remain separate from the software license.",
        "",
    ]
    return "\n".join(lines)


def run_phase4(
    manifest_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase4",
    image_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run frozen CLIP extraction and cross-modal evaluation when images exist."""

    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    import tomllib

    with config_path.open("rb") as file:
        phase4_config = dict(tomllib.load(file).get("phase4", {}))
    manifest = read_manifest(manifest_path)
    assert_no_split_leakage(manifest.records)
    split = str(phase4_config.get("split", "test"))
    if split not in SPLIT_NAMES:
        raise ValueError(f"invalid Phase 4 split: {split}")
    records = tuple(record for record in manifest.records if record.split == split)
    seed = config.seed
    model_id = str(phase4_config.get("model_id", DEFAULT_MODEL_ID))
    batch_size = int(phase4_config.get("batch_size", 8))
    text_max_length = int(phase4_config.get("text_max_length", 77))
    runtime = load_clip_runtime(
        model_id=model_id,
        requested_device=str(phase4_config.get("device", "auto")),
        text_max_length=text_max_length,
    )
    captions = _records_caption_items(records)
    text_limit = int(phase4_config.get("text_max_items", 256))
    selected_caption_ids = set(
        _stable_ids((item[0] for item in captions), seed)[:text_limit]
    )
    text_items = [
        (item_id, text)
        for item_id, text, _ in captions
        if item_id in selected_caption_ids
    ]
    text_start = time.perf_counter()
    text_batch = encode_texts(text_items, runtime, batch_size=batch_size)
    text_seconds = time.perf_counter() - text_start
    image_batch = EmbeddingBatch((), ())
    image_seconds: float | None = None
    image_evaluation_complete = False
    rankings: dict[str, Any] = {}
    evaluation: dict[str, Any] = {
        "split": split,
        "images": 0,
        "captions": len(text_batch.ids),
        "text_embedding_smoke_count": len(text_batch.ids),
        "text_embedding_seconds": text_seconds,
        "text_embedding_throughput_per_second": len(text_batch.ids) / text_seconds
        if text_seconds
        else None,
        "image_embedding_seconds": None,
        "text_to_image": {"status": "not_run_no_image_root"},
        "image_to_text": {"status": "not_run_no_image_root"},
    }
    if image_root is not None:
        image_limit = int(phase4_config.get("image_max_items", 128))
        selected_image_ids = _stable_ids((record.image_id for record in records), seed)[
            :image_limit
        ]
        selected_records = tuple(
            record for record in records if record.image_id in set(selected_image_ids)
        )
        image_items = [
            (record.image_id, Path(image_root) / record.filename)
            for record in selected_records
            if record.filename is not None
        ]
        image_start = time.perf_counter()
        image_batch = encode_images(image_items, runtime, batch_size=batch_size)
        image_seconds = time.perf_counter() - image_start
        image_evaluation_complete = bool(image_items) and (
            len(image_items) == len(selected_records)
            and len(image_batch.ids) == len(image_items)
            and not image_batch.skipped
        )
        caption_items = _records_caption_items(selected_records)
        selected_caption_ids = {item[0] for item in caption_items}
        selected_text_items = [(item_id, text) for item_id, text, _ in caption_items]
        text_batch = encode_texts(selected_text_items, runtime, batch_size=batch_size)
        image_to_text_rankings = rank_batch(
            image_batch,
            text_batch,
            top_k=min(10, len(text_batch.ids)),
        )
        text_to_image_rankings = rank_batch(
            text_batch,
            image_batch,
            top_k=min(10, len(image_batch.ids)),
        )
        image_to_text_relevance = {
            record.image_id: {caption.caption_id for caption in record.captions}
            for record in selected_records
        }
        text_to_image_relevance = {
            caption.caption_id: {record.image_id}
            for record in selected_records
            for caption in record.captions
        }
        rankings = {
            "text_to_image": text_to_image_rankings,
            "image_to_text": image_to_text_rankings,
        }
        evaluation = {
            **evaluation,
            "images": len(image_batch.ids),
            "captions": len(text_batch.ids),
            "text_embedding_smoke_count": len(text_batch.ids),
            "image_embedding_seconds": image_seconds,
            "image_embedding_throughput_per_second": len(image_batch.ids)
            / image_seconds
            if image_seconds
            else None,
            "image_skipped": list(image_batch.skipped),
            "text_to_image": {
                "status": (
                    "completed"
                    if image_evaluation_complete
                    else "not_completed_no_valid_images"
                ),
                "metrics": rank_metrics(
                    text_to_image_rankings,
                    text_to_image_relevance,
                    task="text_to_image",
                ),
            },
            "image_to_text": {
                "status": (
                    "completed"
                    if image_evaluation_complete
                    else "not_completed_no_valid_images"
                ),
                "metrics": rank_metrics(
                    image_to_text_rankings,
                    image_to_text_relevance,
                    task="image_to_text",
                ),
            },
        }
    model = model_metadata(runtime)
    provenance = {
        "project": "OmniSearch",
        "package": "omnisearch",
        "project_version": __version__,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "source_sha256": manifest.source_sha256,
        "split": split,
        "seed": seed,
        "batch_size": batch_size,
        "image_root": str(image_root) if image_root is not None else None,
    }
    fixture_artifacts = [
        str(output_dir / "fixture_smoke.json")
        if (output_dir / "fixture_smoke.json").exists()
        else None
    ]
    report = {
        "report_schema_version": 1,
        "provenance": provenance,
        "pre_phase_audit": {
            "phase3_audit": "PASS",
            "real_phase3_image_baseline_verified": False,
        },
        "scope": {
            "zero_shot_clip_pipeline_verified": True,
            "real_dataset_evaluation": image_evaluation_complete,
            "real_flickr30k_evaluation": image_evaluation_complete,
            "learned_model_used": True,
            "fine_tuning_performed": False,
        },
        "model": model,
        "evaluation": evaluation,
        "rankings": rankings,
        "fixture_artifacts": fixture_artifacts,
    }
    report["fixture_artifacts"] = [
        path for path in fixture_artifacts if path is not None
    ]
    (output_dir / "phase4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "phase4_report.md").write_text(
        _phase4_markdown(report), encoding="utf-8"
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_fixture_smoke(
    runtime: ClipRuntime,
    image_items: Sequence[tuple[str, Path | str, str]],
    text_items: Sequence[tuple[str, str, str]],
    output_path: Path | str,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run and record an explicitly fixture-only end-to-end CLIP smoke test."""

    images = encode_images(
        [(item_id, path) for item_id, path, _ in image_items], runtime, batch_size
    )
    texts = encode_texts(
        [(item_id, text) for item_id, text, _ in text_items], runtime, batch_size
    )
    image_groups = {item_id: group_id for item_id, _, group_id in image_items}
    text_groups = {item_id: group_id for item_id, _, group_id in text_items}
    text_to_image = rank_batch(texts, images, top_k=len(images.ids))
    image_to_text = rank_batch(images, texts, top_k=len(texts.ids))
    text_relevance = {
        item_id: {
            image_id
            for image_id, group in image_groups.items()
            if group == text_groups[item_id]
        }
        for item_id in texts.ids
    }
    image_relevance = {
        item_id: {
            text_id
            for text_id, group in text_groups.items()
            if group == image_groups[item_id]
        }
        for item_id in images.ids
    }
    result = {
        "fixture_only": True,
        "model_id": runtime.model_id,
        "device": runtime.device,
        "embedding_dimension": runtime.embedding_dimension,
        "images": len(images.ids),
        "captions": len(texts.ids),
        "text_to_image_metrics": rank_metrics(
            text_to_image, text_relevance, task="text_to_image"
        ),
        "image_to_text_metrics": rank_metrics(
            image_to_text, image_relevance, task="image_to_text"
        ),
        "text_to_image_rankings": text_to_image,
        "image_to_text_rankings": image_to_text,
        "skipped_images": list(images.skipped),
        "skipped_texts": list(texts.skipped),
        "note": "Fixture-only pipeline verification; not Flickr30k empirical performance.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OmniSearch frozen zero-shot CLIP baseline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/coco2017_val_split_manifest.json"),
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase4"))
    args = parser.parse_args()
    report = run_phase4(args.manifest, args.config, args.output_dir, args.image_root)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "model_id": report["model"]["model_id"],
                "device": report["model"]["device"],
                "zero_shot_clip_pipeline_verified": report["scope"][
                    "zero_shot_clip_pipeline_verified"
                ],
                "real_dataset_evaluation": report["scope"][
                    "real_dataset_evaluation"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
