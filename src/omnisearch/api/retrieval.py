"""Reusable retrieval service behind the Phase 21 HTTP routes."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import DatasetManifest, read_manifest
from ..phase7 import _feature_tensor, _load_checkpoint, _load_trainable_model
from ..phase10 import BuiltIndex, load_persisted_index
from .config import ServiceConfig
from .errors import (
    MalformedImageError,
    ResourceUnavailableError,
    RetrievalExecutionError,
    StartupValidationError,
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StartupValidationError(f"could not read required JSON artifact: {path.name}") from exc


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StartupValidationError(f"could not hash required artifact: {path.name}") from exc
    return digest.hexdigest()


def _hash_ids(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _as_float_list(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return [float(value) for value in array]


@dataclass(frozen=True)
class ValidatedResources:
    manifest: DatasetManifest
    image_ids: tuple[str, ...]
    caption_ids: tuple[str, ...]
    image_vectors: np.ndarray
    caption_vectors: np.ndarray
    image_index: Any
    caption_index: Any
    image_metadata: dict[str, dict[str, Any]]
    caption_metadata: dict[str, dict[str, Any]]
    dimension: int
    checkpoint_sha256: str
    manifest_sha256: str
    model_id: str


class RetrievalService:
    """Load validated resources once and expose thread-safe read retrieval."""

    system_identifier = "phase7_full_ft_clip_phase10_cached_faiss_flat"
    retrieval_backend = "FAISS Flat exact inner-product search"

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._resources: ValidatedResources | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._model_lock = threading.Lock()
        self.startup_report: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self._resources is not None and self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def dimension(self) -> int | None:
        return self._resources.dimension if self._resources is not None else None

    def validate_startup(self) -> dict[str, Any]:
        """Validate provenance, dimensions, IDs, and index compatibility."""

        self.config.validate_settings()
        checks: dict[str, bool] = {}
        required_files = {
            "phase20_report": self.config.phase20_report_path,
            "phase20_recommendations": self.config.phase20_recommendations_path,
            "phase20_provenance": self.config.phase20_provenance_path,
            "checkpoint": self.config.checkpoint_path,
            "manifest": self.config.manifest_path,
            "image_index": self.config.image_index_path,
            "image_index_metadata": self.config.image_index_metadata_path,
            "caption_index": self.config.caption_index_path,
            "caption_index_metadata": self.config.caption_index_metadata_path,
            "cache_metadata": self.config.cache_dir / "metadata.json",
            "image_vectors": self.config.cache_dir / "images.npy",
            "caption_vectors": self.config.cache_dir / "captions.npy",
            "image_ids": self.config.cache_dir / "image_ids.json",
            "caption_ids": self.config.cache_dir / "caption_ids.json",
        }
        checks["required_artifacts_exist"] = all(path.is_file() for path in required_files.values())
        checks["image_root_exists"] = self.config.image_root.is_dir()
        if not checks["required_artifacts_exist"] or not checks["image_root_exists"]:
            missing = [name for name, path in required_files.items() if not path.is_file()]
            if not self.config.image_root.is_dir():
                missing.append("image_root")
            raise StartupValidationError(f"missing Phase 20 service resources: {', '.join(missing)}")

        phase20_report = _read_json(self.config.phase20_report_path)
        recommendations = _read_json(self.config.phase20_recommendations_path)
        provenance = _read_json(self.config.phase20_provenance_path)
        cache_metadata = _read_json(self.config.cache_dir / "metadata.json")
        image_index_metadata = _read_json(self.config.image_index_metadata_path)
        caption_index_metadata = _read_json(self.config.caption_index_metadata_path)
        checkpoint_metadata_path = self.config.checkpoint_path.with_name("checkpoint_metadata.json")
        checkpoint_metadata = _read_json(checkpoint_metadata_path) if checkpoint_metadata_path.is_file() else {}
        checks.update(
            {
                "phase20_quality_gate_pass": phase20_report.get("status") == "PASS" and phase20_report.get("quality_gate", {}).get("status") == "PASS",
                "recommended_default_available": any(
                    row.get("name") == "quality" and row.get("status") == "RECOMMENDED_DEFAULT" and row.get("components") == ["Phase 7 full-FT CLIP", "cached embeddings", "FAISS Flat exact search"]
                    for row in recommendations.get("configurations", [])
                ),
                "reranker_disabled": not any("reranker" in str(row).lower() and row.get("status") == "RECOMMENDED_DEFAULT" for row in recommendations.get("configurations", [])),
                "unsupported_optimization_not_enabled": provenance.get("float16_production_cache_written") is False and provenance.get("phase21_started") is False,
                "training_and_download_absent": provenance.get("training_performed") is False and provenance.get("new_dataset_downloaded") is False,
                "image_index_is_faiss_flat": image_index_metadata.get("index_type") == "faiss_flat" and image_index_metadata.get("candidate_unit") == "image_group",
                "caption_index_is_faiss_flat": caption_index_metadata.get("index_type") == "faiss_flat" and caption_index_metadata.get("candidate_unit") == "caption",
            }
        )
        if not all(checks.values()):
            failed = [name for name, value in checks.items() if not value]
            raise StartupValidationError(f"Phase 20 dependency validation failed: {', '.join(failed)}")

        try:
            manifest = read_manifest(self.config.manifest_path)
            manifest_sha256 = _hash_file(self.config.manifest_path)
            checkpoint_sha256 = _hash_file(self.config.checkpoint_path)
            image_ids = tuple(str(value) for value in _read_json(self.config.cache_dir / "image_ids.json"))
            caption_ids = tuple(str(value) for value in _read_json(self.config.cache_dir / "caption_ids.json"))
            image_vectors = np.load(self.config.cache_dir / "images.npy", allow_pickle=False)
            caption_vectors = np.load(self.config.cache_dir / "captions.npy", allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise StartupValidationError("Phase 20 cache or manifest could not be loaded") from exc

        if image_vectors.dtype != np.float32 or caption_vectors.dtype != np.float32:
            raise StartupValidationError("canonical service embeddings must be float32")
        if image_vectors.ndim != 2 or caption_vectors.ndim != 2 or image_vectors.shape[1] != caption_vectors.shape[1]:
            raise StartupValidationError("image and caption embedding dimensions are incompatible")
        dimension = int(image_vectors.shape[1])
        checks.update(
            {
                "manifest_hash_matches_cache": cache_metadata.get("embedding_source", {}).get("manifest_sha256") == manifest_sha256,
                "manifest_hash_matches_indexes": image_index_metadata.get("dataset_manifest_sha256") == manifest_sha256 and caption_index_metadata.get("dataset_manifest_sha256") == manifest_sha256,
                "checkpoint_hash_matches_cache": cache_metadata.get("embedding_source", {}).get("checkpoint_sha256") == checkpoint_sha256,
                "checkpoint_hash_matches_indexes": image_index_metadata.get("embedding_source", {}).get("checkpoint_sha256") == checkpoint_sha256 and caption_index_metadata.get("embedding_source", {}).get("checkpoint_sha256") == checkpoint_sha256,
                "model_identity_matches": cache_metadata.get("embedding_source", {}).get("model_id") == self.config.model_id and checkpoint_metadata.get("parent_pretrained_checkpoint", self.config.model_id) == self.config.model_id,
                "embedding_dimension_matches": dimension == int(image_index_metadata.get("embedding_dimension", -1)) == int(caption_index_metadata.get("embedding_dimension", -2)) == 512,
                "cache_counts_match": len(image_ids) == image_vectors.shape[0] == int(cache_metadata.get("image_count", -1)) and len(caption_ids) == caption_vectors.shape[0] == int(cache_metadata.get("caption_count", -1)),
                "index_counts_match": int(image_index_metadata.get("candidate_count", -1)) == len(image_ids) and int(caption_index_metadata.get("candidate_count", -1)) == len(caption_ids),
                "cache_ids_unique": len(set(image_ids)) == len(image_ids) and len(set(caption_ids)) == len(caption_ids),
                "index_ids_match_cache": image_index_metadata.get("candidate_ids_sha256") == _hash_ids(image_ids) and caption_index_metadata.get("candidate_ids_sha256") == _hash_ids(caption_ids),
            }
        )
        if not all(checks.values()):
            failed = [name for name, value in checks.items() if not value]
            raise StartupValidationError(f"Phase 20 provenance or shape validation failed: {', '.join(failed)}")

        image_index = self._load_index(self.config.image_index_path, self.config.image_index_metadata_path, image_ids, "image_group", manifest_sha256, dimension)
        caption_index = self._load_index(self.config.caption_index_path, self.config.caption_index_metadata_path, caption_ids, "caption", manifest_sha256, dimension)
        image_by_id = {record.image_id: record for record in manifest.records}
        caption_by_id = {
            caption.caption_id: (record, caption)
            for record in manifest.records
            for caption in record.captions
        }
        if not set(image_ids).issubset(image_by_id) or not set(caption_ids).issubset(caption_by_id):
            raise StartupValidationError("cache IDs are not covered by the validated manifest")
        image_metadata = {
            image_id: {"image_id": image_id, "filename": image_by_id[image_id].filename, "caption_count": len(image_by_id[image_id].captions)}
            for image_id in image_ids
        }
        caption_metadata = {
            caption_id: {"caption_id": caption_id, "text": caption.text, "image_id": record.image_id}
            for caption_id in caption_ids
            for record, caption in [caption_by_id[caption_id]]
        }
        self._resources = ValidatedResources(
            manifest=manifest,
            image_ids=image_ids,
            caption_ids=caption_ids,
            image_vectors=image_vectors,
            caption_vectors=caption_vectors,
            image_index=image_index,
            caption_index=caption_index,
            image_metadata=image_metadata,
            caption_metadata=caption_metadata,
            dimension=dimension,
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
            model_id=self.config.model_id,
        )
        report = {
            "schema_version": 1,
            "phase": 21,
            "passed": True,
            "checks": checks,
            "model_id": self.config.model_id,
            "embedding_dimension": dimension,
            "image_count": len(image_ids),
            "caption_count": len(caption_ids),
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
            "backend": self.retrieval_backend,
            "reranker_enabled": False,
            "approximate_index_default": False,
        }
        self.startup_report = report
        return report

    def _load_index(self, path: Path, metadata_path: Path, ids: Sequence[str], candidate_unit: str, manifest_sha256: str, dimension: int) -> Any:
        metadata = _read_json(metadata_path)
        built = BuiltIndex(index=None, metadata=metadata, index_path=path, metadata_path=metadata_path)
        expected = {
            "dataset_manifest_sha256": manifest_sha256,
            "tier": "tier3",
            "candidate_unit": candidate_unit,
            "embedding_dimension": dimension,
            "candidate_count": len(ids),
        }
        try:
            return load_persisted_index(built, ids, expected)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise StartupValidationError(f"could not load compatible FAISS Flat index: {path.name}") from exc

    def load(self) -> dict[str, Any]:
        """Validate and load model/index resources once per process."""

        if self.ready:
            return self.startup_report or {}
        self.validate_startup()
        try:
            model, processor, torch, device = _load_trainable_model(self.config.model_id, self.config.device)
            _load_checkpoint(self.config.checkpoint_path, model)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            projection_dimension = int(getattr(model.config, "projection_dim", 0))
            if projection_dimension != self.dimension:
                raise StartupValidationError("loaded model projection dimension does not match persisted indexes")
            self._model = model
            self._processor = processor
            self._torch = torch
            self._device = str(device)
        except StartupValidationError:
            raise
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise StartupValidationError("validated artifacts could not load into the CLIP runtime") from exc
        return self.startup_report or {}

    def close(self) -> None:
        """Release process-owned model, index, and processor references safely."""

        with self._model_lock:
            self._resources = None
            self._model = None
            self._processor = None
            self._torch = None
            self._device = None
            self.startup_report = None

    def _require_ready(self) -> ValidatedResources:
        if not self.ready or self._model is None or self._processor is None or self._torch is None:
            raise ResourceUnavailableError("retrieval resources are not ready")
        if self._resources is None:
            raise ResourceUnavailableError("retrieval indexes are not ready")
        return self._resources

    def info(self) -> dict[str, Any]:
        return self.config.public_info(self.dimension, self.device)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ready else "degraded",
            "project": "OmniSearch",
            "model_loaded": self._model is not None,
            "indexes_loaded": self._resources is not None,
            "device": self.device,
            "api_version": self.config.api_version,
        }

    def readiness(self) -> tuple[bool, str | None]:
        if self.ready:
            return True, None
        return False, "validated model and index resources are not loaded"

    def _encode_text(self, query: str) -> tuple[np.ndarray, float]:
        assert self._model is not None and self._processor is not None and self._torch is not None
        started = time.perf_counter()
        with self._model_lock, self._torch.no_grad():
            processed = self._processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.text_max_length,
            )
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in processed.items()
            }
            text_inputs = {key: value for key, value in inputs.items() if key != "pixel_values"}
            features = _feature_tensor(self._model.get_text_features(**text_inputs))
            features = self._torch.nn.functional.normalize(features, dim=-1)
            vector = features.detach().cpu().numpy().astype(np.float32, copy=False)
        return vector, (time.perf_counter() - started) * 1000.0

    def _encode_image(self, image: Any) -> tuple[np.ndarray, float]:
        assert self._model is not None and self._processor is not None and self._torch is not None
        started = time.perf_counter()
        with self._model_lock, self._torch.no_grad():
            processed = self._processor(images=[image], return_tensors="pt")
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in processed.items()
            }
            features = _feature_tensor(self._model.get_image_features(pixel_values=inputs["pixel_values"]))
            features = self._torch.nn.functional.normalize(features, dim=-1)
            vector = features.detach().cpu().numpy().astype(np.float32, copy=False)
        return vector, (time.perf_counter() - started) * 1000.0

    def _search_results(self, index: Any, vector: np.ndarray, top_k: int, metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        ids, scores = index.search(vector, top_k)
        result_ids = ids[0].tolist() if hasattr(ids[0], "tolist") else list(ids[0])
        result_scores = scores[0].tolist() if hasattr(scores[0], "tolist") else list(scores[0])
        return [
            {"id": str(item_id), "rank": rank, "score": float(score), "metadata": dict(metadata[str(item_id)])}
            for rank, (item_id, score) in enumerate(zip(result_ids, result_scores), start=1)
        ]

    def _validate_top_k(self, top_k: int) -> None:
        if not 1 <= int(top_k) <= self.config.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.config.max_top_k}")

    def search_text_to_image(self, query: str, top_k: int, request_id: str) -> dict[str, Any]:
        resources = self._require_ready()
        self._validate_top_k(top_k)
        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        normalized_query = query.strip()
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        try:
            vector, encoding_ms = self._encode_text(normalized_query)
            search_started = time.perf_counter()
            results = self._search_results(resources.image_index, vector, top_k, resources.image_metadata)
            search_ms = (time.perf_counter() - search_started) * 1000.0
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RetrievalExecutionError("text retrieval failed") from exc
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return self._response("text-to-image", normalized_query, results, preprocessing_ms, encoding_ms, search_ms, total_ms, request_id)

    def decode_image(self, payload: bytes) -> Any:
        if not payload or len(payload) > self.config.max_upload_bytes:
            raise MalformedImageError("image payload exceeds the configured upload limit")
        try:
            from PIL import Image, UnidentifiedImageError

            with Image.open(io.BytesIO(payload)) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP"}:
                    raise MalformedImageError("unsupported image format")
                if probe.width <= 0 or probe.height <= 0 or probe.width * probe.height > self.config.max_image_pixels:
                    raise MalformedImageError("image dimensions are not supported")
                probe.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                return image.convert("RGB")
        except MalformedImageError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise MalformedImageError("uploaded file is not a valid supported image") from exc

    def search_image_to_text(self, payload: bytes, top_k: int, request_id: str) -> dict[str, Any]:
        resources = self._require_ready()
        self._validate_top_k(top_k)
        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        image = self.decode_image(payload)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        try:
            vector, encoding_ms = self._encode_image(image)
            search_started = time.perf_counter()
            results = self._search_results(resources.caption_index, vector, top_k, resources.caption_metadata)
            search_ms = (time.perf_counter() - search_started) * 1000.0
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RetrievalExecutionError("image retrieval failed") from exc
        finally:
            image.close()
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return self._response("image-to-text", None, results, preprocessing_ms, encoding_ms, search_ms, total_ms, request_id)

    def _response(self, query_type: str, query: str | None, results: list[dict[str, Any]], preprocessing_ms: float, encoding_ms: float, search_ms: float, total_ms: float, request_id: str) -> dict[str, Any]:
        return {
            "query_type": query_type,
            "query": query,
            "results": results,
            "model_system": self.system_identifier,
            "retrieval_backend": self.retrieval_backend,
            "latency_ms": {
                "preprocessing_ms": max(0.0, float(preprocessing_ms)),
                "query_encoding_ms": max(0.0, float(encoding_ms)),
                "search_ms": max(0.0, float(search_ms)),
                "total_server_ms": max(0.0, float(total_ms)),
            },
            "request_id": request_id,
        }
