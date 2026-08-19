"""Configuration for the Phase 21 retrieval API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _path_from_env(name: str, default: Path, root: Path) -> Path:
    value = Path(os.environ.get(name, str(default)))
    return value if value.is_absolute() else root / value


@dataclass(frozen=True)
class ServiceConfig:
    """Validated service settings; clients cannot override these paths."""

    root: Path
    phase20_report_path: Path
    phase20_recommendations_path: Path
    phase20_provenance_path: Path
    checkpoint_path: Path
    manifest_path: Path
    image_root: Path
    cache_dir: Path
    image_index_path: Path
    image_index_metadata_path: Path
    caption_index_path: Path
    caption_index_metadata_path: Path
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    batch_size: int = 8
    text_max_length: int = 77
    default_top_k: int = 5
    max_top_k: int = 50
    max_query_chars: int = 1_000
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    api_version: str = "v1"
    protocol_version: str = "retrieval_eval_v1"
    latency_enabled: bool = True
    log_query_content: bool = False

    @classmethod
    def from_env(cls, root: Path | None = None) -> ServiceConfig:
        base = (root or Path(os.environ.get("OMNISEARCH_ROOT", Path.cwd()))).resolve()
        config = cls(
            root=base,
            phase20_report_path=_path_from_env("OMNISEARCH_PHASE20_REPORT", Path("artifacts/phase20/phase20_report.json"), base),
            phase20_recommendations_path=_path_from_env("OMNISEARCH_PHASE20_RECOMMENDATIONS", Path("artifacts/phase20/recommended_configurations.json"), base),
            phase20_provenance_path=_path_from_env("OMNISEARCH_PHASE20_PROVENANCE", Path("artifacts/phase20/provenance.json"), base),
            checkpoint_path=_path_from_env("OMNISEARCH_CHECKPOINT", Path("artifacts/phase7/best_checkpoint.pt"), base),
            manifest_path=_path_from_env("OMNISEARCH_MANIFEST", Path("data/processed/coco2017_val_split_manifest.json"), base),
            image_root=_path_from_env("OMNISEARCH_IMAGE_ROOT", Path("data/raw/coco2017/val2017"), base),
            cache_dir=_path_from_env("OMNISEARCH_CACHE_DIR", Path("artifacts/phase10/embedding_cache"), base),
            image_index_path=_path_from_env("OMNISEARCH_IMAGE_INDEX", Path("artifacts/phase10/indexes/tier3/text_to_image/faiss_flat.faiss"), base),
            image_index_metadata_path=_path_from_env("OMNISEARCH_IMAGE_INDEX_METADATA", Path("artifacts/phase10/indexes/tier3/text_to_image/faiss_flat.metadata.json"), base),
            caption_index_path=_path_from_env("OMNISEARCH_CAPTION_INDEX", Path("artifacts/phase10/indexes/tier3/image_to_text/faiss_flat.faiss"), base),
            caption_index_metadata_path=_path_from_env("OMNISEARCH_CAPTION_INDEX_METADATA", Path("artifacts/phase10/indexes/tier3/image_to_text/faiss_flat.metadata.json"), base),
            model_id=os.environ.get("OMNISEARCH_MODEL_ID", "openai/clip-vit-base-patch32"),
            device=os.environ.get("OMNISEARCH_DEVICE", "auto"),
            batch_size=_env_int("OMNISEARCH_BATCH_SIZE", 8),
            text_max_length=_env_int("OMNISEARCH_TEXT_MAX_LENGTH", 77),
            default_top_k=_env_int("OMNISEARCH_DEFAULT_TOP_K", 5),
            max_top_k=_env_int("OMNISEARCH_MAX_TOP_K", 50),
            max_query_chars=_env_int("OMNISEARCH_MAX_QUERY_CHARS", 1_000),
            max_upload_bytes=_env_int("OMNISEARCH_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_image_pixels=_env_int("OMNISEARCH_MAX_IMAGE_PIXELS", 40_000_000),
            api_version=os.environ.get("OMNISEARCH_API_VERSION", "v1"),
            protocol_version=os.environ.get("OMNISEARCH_PROTOCOL_VERSION", "retrieval_eval_v1"),
            latency_enabled=_env_bool("OMNISEARCH_LATENCY_ENABLED", True),
            log_query_content=_env_bool("OMNISEARCH_LOG_QUERY_CONTENT", False),
        )
        config.validate_settings()
        return config

    def validate_settings(self) -> None:
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("OMNISEARCH_DEVICE must be auto, cpu, or mps")
        if self.batch_size <= 0 or self.text_max_length <= 0:
            raise ValueError("batch_size and text_max_length must be positive")
        if not 1 <= self.default_top_k <= self.max_top_k <= 50:
            raise ValueError("top-k settings must satisfy 1 <= default <= max <= 50")
        if self.max_query_chars <= 0 or self.max_upload_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError("query, upload, and image-pixel limits must be positive")
        if self.api_version != "v1":
            raise ValueError("only API version v1 is supported")

    def artifact_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "model_id": self.model_id,
            "device_preference": self.device,
            "default_top_k": self.default_top_k,
            "max_top_k": self.max_top_k,
            "max_query_chars": self.max_query_chars,
            "max_upload_bytes": self.max_upload_bytes,
            "max_image_pixels": self.max_image_pixels,
            "latency_enabled": self.latency_enabled,
            "log_query_content": self.log_query_content,
            "default_system": "Phase 7 full-FT CLIP + Phase 10 cached embeddings + FAISS Flat exact search",
            "reranker_enabled": False,
            "approximate_index_default": False,
            "paths_relative_to_root": {
                "checkpoint": str(self.checkpoint_path.relative_to(self.root)) if self.checkpoint_path.is_relative_to(self.root) else str(self.checkpoint_path),
                "manifest": str(self.manifest_path.relative_to(self.root)) if self.manifest_path.is_relative_to(self.root) else str(self.manifest_path),
                "image_root": str(self.image_root.relative_to(self.root)) if self.image_root.is_relative_to(self.root) else str(self.image_root),
                "cache_dir": str(self.cache_dir.relative_to(self.root)) if self.cache_dir.is_relative_to(self.root) else str(self.cache_dir),
                "image_index": str(self.image_index_path.relative_to(self.root)) if self.image_index_path.is_relative_to(self.root) else str(self.image_index_path),
                "caption_index": str(self.caption_index_path.relative_to(self.root)) if self.caption_index_path.is_relative_to(self.root) else str(self.caption_index_path),
            },
        }

    def public_info(self, dimension: int | None = None, device: str | None = None) -> dict[str, Any]:
        return {
            "project": "OmniSearch",
            "model_family": "CLIP ViT-B/32",
            "model_id": self.model_id,
            "embedding_dimension": dimension,
            "retrieval_backend": "FAISS Flat exact inner-product search",
            "supported_query_modes": ["text-to-image", "image-to-text"],
            "default_top_k": self.default_top_k,
            "max_top_k": self.max_top_k,
            "api_version": self.api_version,
            "protocol_version": self.protocol_version,
            "device": device,
            "reranker_enabled": False,
            "content_safety_filtering": "NOT IMPLEMENTED",
            "research_system": True,
        }
