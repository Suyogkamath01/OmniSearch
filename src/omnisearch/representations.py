"""Frozen unimodal representation extraction for Phase 6.

The module deliberately contains no optimizer, loss, gradient, adapter, or
index code. It provides optional-Transformers loaders, deterministic batched
encoding, finite-value validation, exact same-space similarity, and
provenance-checked caches.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 1
REPRESENTATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RepresentationSpec:
    key: str
    modality: str
    model_id: str
    architecture: str
    pooling: str
    normalization: str
    max_length: int | None = None
    preprocessing: str = "model-native processor"
    role: str = "unimodal frozen representation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TEXT_SPECS: dict[str, RepresentationSpec] = {
    "minilm_mean": RepresentationSpec(
        key="minilm_mean",
        modality="text",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        architecture="MiniLM-L6 Transformer sentence encoder",
        pooling="attention_mask_mean",
        normalization="l2",
        max_length=128,
        role="sentence-level pretrained semantic embedding",
    ),
    "distilbert_mean": RepresentationSpec(
        key="distilbert_mean",
        modality="text",
        model_id="distilbert/distilbert-base-uncased",
        architecture="DistilBERT encoder",
        pooling="attention_mask_mean",
        normalization="l2",
        max_length=128,
        role="general pretrained encoder control; not sentence-similarity fine-tuned",
    ),
}


VISION_SPECS: dict[str, RepresentationSpec] = {
    "resnet18_native": RepresentationSpec(
        key="resnet18_native",
        modality="image",
        model_id="microsoft/resnet-18",
        architecture="ResNet-18 CNN",
        pooling="native_global_pool",
        normalization="l2",
        preprocessing="Pillow/Torch standard ImageNet resize-shorter-256 center-crop-224 normalize",
        role="pretrained convolutional visual representation",
    ),
    "vit_base_native": RepresentationSpec(
        key="vit_base_native",
        modality="image",
        model_id="google/vit-base-patch16-224-in21k",
        architecture="Vision Transformer Base/16",
        pooling="native_pooler",
        normalization="l2",
        preprocessing="AutoImageProcessor model-native resize/normalize",
        role="pretrained transformer visual representation",
    ),
}


@dataclass(frozen=True)
class RepresentationBatch:
    ids: tuple[str, ...]
    embeddings: tuple[tuple[float, ...], ...]
    skipped: tuple[dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.embeddings):
            raise ValueError("representation IDs and embeddings have different lengths")
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("representation IDs must be unique")
        dimensions = {len(row) for row in self.embeddings}
        if len(dimensions) > 1:
            raise ValueError("representation embeddings have inconsistent dimensions")
        for row in self.embeddings:
            if not row or not all(_finite(value) for value in row):
                raise ValueError(
                    "representation embeddings must be finite and non-empty"
                )

    @property
    def dimension(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0


@dataclass
class RepresentationRuntime:
    model: Any
    processor: Any
    torch: Any
    device: str
    spec: RepresentationSpec
    embedding_dimension: int
    model_revision: str | None
    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def select_device(requested: str = "auto", torch_module: Any | None = None) -> str:
    """Select MPS when available and otherwise use CPU."""

    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    torch_runtime: Any = torch_module
    if torch_runtime is None:
        try:
            import torch as imported_torch  # type: ignore[import-not-found]

            torch_runtime = imported_torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Phase 6") from exc
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


def _transformers() -> Any:
    try:
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Phase 6 requires the optional phase4 Transformers stack"
        ) from exc
    return transformers


def _freeze_model(model: Any) -> None:
    model.eval()
    model.requires_grad_(False)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError("frozen representation model has trainable parameters")


def _parameter_metadata(model: Any) -> tuple[int, int, int]:
    parameters = tuple(model.parameters())
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(
        int(parameter.numel()) for parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in parameters
    )
    return total, trainable, parameter_bytes


def _revision(model: Any) -> str | None:
    config = getattr(model, "config", None)
    return getattr(config, "_commit_hash", None) or getattr(
        config, "_name_or_path", None
    )


class _ResNetProcessor:
    """Torch/Pillow-only ImageNet preprocessing for ResNet-18.

    Transformers 5 can expose the ResNet model without torchvision, but its
    automatic image processor currently requires torchvision. This explicit
    processor keeps the dependency optional and records the standard
    ImageNet resize/crop/normalization contract.
    """

    def __init__(self, size: int = 224, resize_shorter: int = 256) -> None:
        self.size = size
        self.resize_shorter = resize_shorter

    def __call__(
        self, *, images: Sequence[Any], return_tensors: str = "pt"
    ) -> dict[str, Any]:
        if return_tensors != "pt":
            raise ValueError(
                "Phase 6 image processors currently return PyTorch tensors only"
            )
        import torch  # type: ignore[import-not-found]

        tensors = []
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        for image in images:
            image = image.convert("RGB")
            width, height = image.size
            scale = self.resize_shorter / min(width, height)
            resized = image.resize(
                (
                    max(self.size, round(width * scale)),
                    max(self.size, round(height * scale)),
                )
            )
            left = (resized.width - self.size) // 2
            top = (resized.height - self.size) // 2
            cropped = resized.crop((left, top, left + self.size, top + self.size))
            raw = torch.tensor(list(cropped.tobytes()), dtype=torch.uint8)
            tensor = raw.view(self.size, self.size, 3).permute(2, 0, 1).float() / 255.0
            tensors.append((tensor - mean) / std)
        return {"pixel_values": torch.stack(tensors)}


def _hidden_dimension(model: Any) -> int:
    config = getattr(model, "config", None)
    for key in ("hidden_size", "projection_dim"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[-1])
    raise ValueError("could not infer representation dimension from model config")


def load_text_runtime(
    spec: RepresentationSpec,
    requested_device: str = "auto",
    cache_dir: Path | str | None = None,
) -> RepresentationRuntime:
    if spec.modality != "text":
        raise ValueError("load_text_runtime requires a text RepresentationSpec")
    transformers = _transformers()
    import torch  # type: ignore[import-not-found]

    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model_id, **kwargs)
    model = transformers.AutoModel.from_pretrained(spec.model_id, **kwargs)
    _freeze_model(model)
    device = select_device(requested_device, torch)
    model.to(device)
    total, trainable, parameter_bytes = _parameter_metadata(model)
    return RepresentationRuntime(
        model=model,
        processor=tokenizer,
        torch=torch,
        device=device,
        spec=spec,
        embedding_dimension=_hidden_dimension(model),
        model_revision=_revision(model),
        parameter_count=total,
        trainable_parameter_count=trainable,
        parameter_bytes=parameter_bytes,
    )


def load_vision_runtime(
    spec: RepresentationSpec,
    requested_device: str = "auto",
    cache_dir: Path | str | None = None,
) -> RepresentationRuntime:
    if spec.modality != "image":
        raise ValueError("load_vision_runtime requires an image RepresentationSpec")
    transformers = _transformers()
    import torch  # type: ignore[import-not-found]

    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    if spec.key.startswith("resnet"):
        processor = _ResNetProcessor()
    elif spec.key.startswith("vit"):
        processor_class = (
            getattr(transformers, "ViTImageProcessorPil", None)
            or transformers.ViTImageProcessor
        )
        processor = processor_class.from_pretrained(spec.model_id, **kwargs)
    else:
        processor = transformers.AutoImageProcessor.from_pretrained(
            spec.model_id, **kwargs
        )
    model = transformers.AutoModel.from_pretrained(spec.model_id, **kwargs)
    _freeze_model(model)
    device = select_device(requested_device, torch)
    model.to(device)
    total, trainable, parameter_bytes = _parameter_metadata(model)
    return RepresentationRuntime(
        model=model,
        processor=processor,
        torch=torch,
        device=device,
        spec=spec,
        embedding_dimension=_hidden_dimension(model),
        model_revision=_revision(model),
        parameter_count=total,
        trainable_parameter_count=trainable,
        parameter_bytes=parameter_bytes,
    )


def normalize_rows(
    rows: Sequence[Sequence[float]], mode: str = "l2"
) -> tuple[tuple[float, ...], ...]:
    if mode not in {"none", "l2"}:
        raise ValueError("normalization must be none or l2")
    normalized: list[tuple[float, ...]] = []
    for row in rows:
        values = tuple(float(value) for value in row)
        if not values or not all(_finite(value) for value in values):
            raise ValueError("embedding rows must be finite and non-empty")
        if mode == "none":
            normalized.append(values)
            continue
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("embedding rows must have finite non-zero L2 norm")
        normalized.append(tuple(value / norm for value in values))
    return tuple(normalized)


def _move_inputs(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _tensor_rows(value: Any) -> list[list[float]]:
    if value is None or not hasattr(value, "detach"):
        raise ValueError("model output did not contain a tensor")
    rows = value.detach().float().cpu().tolist()
    if rows and isinstance(rows[0], (int, float)):
        rows = [rows]
    return rows


def _text_hidden(outputs: Any) -> Any:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, Mapping):
        hidden = outputs.get("last_hidden_state")
    if hidden is None:
        raise ValueError("text model output has no last_hidden_state")
    return hidden


def _text_pool(hidden: Any, attention_mask: Any, pooling: str) -> Any:
    if hidden.ndim != 3:
        raise ValueError(
            "text last_hidden_state must have shape [batch, tokens, dimension]"
        )
    if pooling == "attention_mask_mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        total = (hidden * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return total / denominator
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "native":
        return hidden[:, 0]
    raise ValueError(f"unsupported text pooling: {pooling}")


def encode_texts(
    items: Sequence[tuple[str, str]],
    runtime: RepresentationRuntime,
    batch_size: int = 8,
) -> RepresentationBatch:
    if runtime.spec.modality != "text":
        raise ValueError("encode_texts requires a text runtime")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    valid = [(str(item_id), str(text)) for item_id, text in items if str(text).strip()]
    skipped = tuple(
        {"id": str(item_id), "reason": "empty_text"}
        for item_id, text in items
        if not str(text).strip()
    )
    ids: list[str] = []
    rows: list[tuple[float, ...]] = []
    with runtime.torch.inference_mode():
        for start in range(0, len(valid), batch_size):
            batch = valid[start : start + batch_size]
            encoded = runtime.processor(
                [text for _, text in batch],
                padding=True,
                truncation=True,
                max_length=runtime.spec.max_length,
                return_tensors="pt",
            )
            encoded = _move_inputs(encoded, runtime.device)
            outputs = runtime.model(**encoded)
            pooled = _text_pool(
                _text_hidden(outputs), encoded["attention_mask"], runtime.spec.pooling
            )
            batch_rows = normalize_rows(
                _tensor_rows(pooled), runtime.spec.normalization
            )
            if len(batch_rows) != len(batch):
                raise ValueError(
                    "text model output batch size does not match input batch"
                )
            ids.extend(item_id for item_id, _ in batch)
            rows.extend(batch_rows)
    if rows and len(rows[0]) != runtime.embedding_dimension:
        raise ValueError("text embedding dimension does not match model metadata")
    return RepresentationBatch(tuple(ids), tuple(rows), skipped)


def _vision_output(outputs: Any, pooling: str) -> Any:
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None and pooling in {"native_pooler", "native_global_pool"}:
        return pooler.flatten(start_dim=1)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, Mapping):
        hidden = outputs.get("last_hidden_state")
    if hidden is None:
        raise ValueError("vision model output has no usable representation tensor")
    if hidden.ndim == 4:
        return hidden.mean(dim=(2, 3))
    if hidden.ndim == 3:
        return (
            hidden[:, 0] if pooling in {"cls", "native_pooler"} else hidden.mean(dim=1)
        )
    if hidden.ndim == 2:
        return hidden
    raise ValueError("vision output has unsupported tensor rank")


def encode_images(
    items: Sequence[tuple[str, Path | str]],
    runtime: RepresentationRuntime,
    batch_size: int = 8,
) -> RepresentationBatch:
    if runtime.spec.modality != "image":
        raise ValueError("encode_images requires an image runtime")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Pillow is required for Phase 6 image extraction") from exc
    valid: list[tuple[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item_id, raw_path in sorted(
        ((str(item_id), Path(path)) for item_id, path in items),
        key=lambda item: item[0],
    ):
        try:
            with Image.open(raw_path) as source:
                valid.append((item_id, source.convert("RGB")))
        except (OSError, ValueError) as exc:
            skipped.append({"id": item_id, "path": str(raw_path), "reason": str(exc)})
    ids: list[str] = []
    rows: list[tuple[float, ...]] = []
    try:
        with runtime.torch.inference_mode():
            for start in range(0, len(valid), batch_size):
                batch = valid[start : start + batch_size]
                encoded = runtime.processor(
                    images=[image for _, image in batch],
                    return_tensors="pt",
                )
                encoded = _move_inputs(encoded, runtime.device)
                outputs = runtime.model(**encoded)
                pooled = _vision_output(outputs, runtime.spec.pooling)
                batch_rows = normalize_rows(
                    _tensor_rows(pooled), runtime.spec.normalization
                )
                if len(batch_rows) != len(batch):
                    raise ValueError(
                        "vision model output batch size does not match input batch"
                    )
                ids.extend(item_id for item_id, _ in batch)
                rows.extend(batch_rows)
    finally:
        for _, image in valid:
            close = getattr(image, "close", None)
            if close:
                close()
    if rows and len(rows[0]) != runtime.embedding_dimension:
        raise ValueError("vision embedding dimension does not match model metadata")
    return RepresentationBatch(tuple(ids), tuple(rows), tuple(skipped))


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(item) for item in ids).encode("utf-8")
    ).hexdigest()


def write_embedding_cache(
    path: Path | str,
    batch: RepresentationBatch,
    metadata: Mapping[str, Any],
) -> None:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "representation_schema_version": REPRESENTATION_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "ids_sha256": ids_sha256(batch.ids),
        "embedding_dimension": batch.dimension,
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
) -> RepresentationBatch:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported representation cache schema")
    if payload.get("representation_schema_version") != REPRESENTATION_SCHEMA_VERSION:
        raise ValueError("unsupported representation schema")
    if payload.get("metadata") != dict(expected_metadata):
        raise ValueError("representation cache metadata does not match requested run")
    ids = tuple(str(value) for value in payload.get("ids", []))
    if payload.get("ids_sha256") != ids_sha256(ids):
        raise ValueError("representation cache IDs fingerprint does not match IDs")
    rows = normalize_rows(payload.get("embeddings", []), "none")
    if payload.get("embedding_dimension", 0) != (len(rows[0]) if rows else 0):
        raise ValueError("representation cache dimension is inconsistent")
    skipped = tuple(dict(value) for value in payload.get("skipped", []))
    return RepresentationBatch(ids, rows, skipped)


def runtime_metadata(runtime: RepresentationRuntime) -> dict[str, Any]:
    return {
        **runtime.spec.to_dict(),
        "model_revision": runtime.model_revision,
        "embedding_dimension": runtime.embedding_dimension,
        "device": runtime.device,
        "frozen": runtime.trainable_parameter_count == 0,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "parameter_count": runtime.parameter_count,
        "parameter_bytes": runtime.parameter_bytes,
        "processor_class": type(runtime.processor).__name__,
        "model_class": type(runtime.model).__name__,
    }


def rank_embedding_batches(
    queries: RepresentationBatch,
    candidates: RepresentationBatch,
    space_id: str,
    candidate_space_id: str,
    top_k: int = 10,
    exclude_self: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Rank within one declared representation space only.

    Requiring matching space IDs prevents accidental cosine comparisons between
    unrelated text and image encoders. CLIP's shared space is handled by its
    existing Phase 4 path, not by this generic unimodal function.
    """

    if space_id != candidate_space_id:
        raise ValueError(
            "cross-space similarity is forbidden for unimodal representations"
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    # Torch is already required by the Phase 6 model runtimes.  A matrix
    # product keeps the real COCO text comparison tractable while the
    # fallback preserves the lightweight, dependency-free unit-test path.
    torch_runtime: Any | None
    try:
        import torch as imported_torch  # type: ignore[import-not-found]

        torch_runtime = imported_torch
    except ImportError:
        torch_runtime = None
    if torch_runtime is not None:
        query_tensor = torch_runtime.tensor(
            queries.embeddings, dtype=torch_runtime.float32
        )
        candidate_tensor = torch_runtime.tensor(
            candidates.embeddings, dtype=torch_runtime.float32
        )
        query_norms = torch_runtime.linalg.vector_norm(query_tensor, dim=1)
        candidate_norms = torch_runtime.linalg.vector_norm(candidate_tensor, dim=1)
        if not bool(torch_runtime.isfinite(query_norms).all()) or bool(
            (query_norms == 0).any()
        ):
            raise ValueError("query embedding has invalid norm")
        if not bool(torch_runtime.isfinite(candidate_norms).all()) or bool(
            (candidate_norms == 0).any()
        ):
            raise ValueError("candidate embedding has invalid norm")
        score_matrix = (
            (query_tensor @ candidate_tensor.T)
            / query_norms[:, None]
            / candidate_norms[None, :]
        )
        torch_result: dict[str, list[dict[str, Any]]] = {}
        candidate_ids = tuple(candidates.ids)
        for row_index, query_id in enumerate(queries.ids):
            torch_ranked = [
                (candidate_id, float(score))
                for candidate_id, score in zip(
                    candidate_ids, score_matrix[row_index].tolist()
                )
                if not (exclude_self and candidate_id == query_id)
            ]
            torch_ranked.sort(key=lambda item: (-item[1], item[0]))
            torch_result[query_id] = [
                {"id": item_id, "score": score}
                for item_id, score in torch_ranked[:top_k]
            ]
        return torch_result

    candidate_vectors = dict(zip(candidates.ids, candidates.embeddings))
    result: dict[str, list[dict[str, Any]]] = {}
    for query_id, query in zip(queries.ids, queries.embeddings):
        query_norm = math.sqrt(sum(value * value for value in query))
        if query_norm == 0 or not math.isfinite(query_norm):
            raise ValueError("query embedding has invalid norm")
        ranked: list[tuple[str, float]] = []
        for candidate_id, vector in candidate_vectors.items():
            if exclude_self and candidate_id == query_id:
                continue
            candidate_norm = math.sqrt(sum(value * value for value in vector))
            if candidate_norm == 0 or not math.isfinite(candidate_norm):
                raise ValueError("candidate embedding has invalid norm")
            score = sum(left * right for left, right in zip(query, vector)) / (
                query_norm * candidate_norm
            )
            ranked.append((candidate_id, float(score)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        result[query_id] = [
            {"id": item_id, "score": score} for item_id, score in ranked[:top_k]
        ]
    return result
