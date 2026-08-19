"""Small CPU-only retrieval service for artifact-free Streamlit deployment.

This is a deployment fallback, not a benchmark path. The validated local
service is still used whenever its checkpoint, COCO gallery, caches, and
indexes exist. Community Cloud receives a compact public COCO validation
gallery so the existing UI remains demonstrable without committing restricted
or multi-GB local artifacts.
"""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import urlopen

import numpy as np

from ..api.config import ServiceConfig
from ..api.errors import MalformedImageError, RetrievalExecutionError
from ..clip_baseline import (
    _model_inputs,
    _output_rows,
    encode_images,
    load_clip_runtime,
    normalize_embeddings,
)

CLOUD_MODEL_ID = "openai/clip-vit-base-patch32"
CLOUD_ASSET_DIR = Path("assets/cloud_demo")
CLOUD_GALLERY = (
    (
        "000000038829.jpg",
        "A man riding a bicycle with a boy on the back of it.",
        "https://images.cocodataset.org/val2017/000000038829.jpg",
    ),
    (
        "000000017029.jpg",
        "A dog jumping to catch a red frisbee in a garden.",
        "https://images.cocodataset.org/val2017/000000017029.jpg",
    ),
    (
        "000000015335.jpg",
        "A group of people sitting at a table with food.",
        "https://images.cocodataset.org/val2017/000000015335.jpg",
    ),
    (
        "000000001532.jpg",
        "A street scene with focus on the street signs on an overpass.",
        "https://images.cocodataset.org/val2017/000000001532.jpg",
    ),
    (
        "000000001584.jpg",
        "A red double decker bus driving down a city street.",
        "https://images.cocodataset.org/val2017/000000001584.jpg",
    ),
)


def local_artifacts_available(config: ServiceConfig) -> bool:
    """Return whether the full validated service can be used unchanged."""

    required_files = (
        config.phase20_report_path,
        config.phase20_recommendations_path,
        config.phase20_provenance_path,
        config.checkpoint_path,
        config.manifest_path,
        config.image_index_path,
        config.image_index_metadata_path,
        config.caption_index_path,
        config.caption_index_metadata_path,
        config.cache_dir / "metadata.json",
        config.cache_dir / "images.npy",
        config.cache_dir / "captions.npy",
        config.cache_dir / "image_ids.json",
        config.cache_dir / "caption_ids.json",
    )
    return config.image_root.is_dir() and all(path.is_file() for path in required_files)


class CompactCloudDemoService:
    """CPU retrieval over a tiny public COCO gallery for Community Cloud."""

    system_identifier = "cloud_zero_shot_clip_compact_gallery"
    retrieval_backend = "FAISS Flat exact inner-product search"
    deployment_mode = "compact_cloud_demo"

    def __init__(self, config: ServiceConfig) -> None:
        self._temporary_gallery = TemporaryDirectory(prefix="omnisearch-cloud-demo-")
        gallery_root = Path(self._temporary_gallery.name)
        self.config = replace(
            config,
            image_root=gallery_root,
            model_id=CLOUD_MODEL_ID,
            device="cpu",
        )
        self._runtime: Any | None = None
        self._image_index: Any | None = None
        self._caption_index: Any | None = None
        self._image_metadata: dict[str, dict[str, Any]] = {}
        self._caption_metadata: dict[str, dict[str, Any]] = {}
        self._image_ids: tuple[str, ...] = ()
        self._caption_ids: tuple[str, ...] = ()
        self._preview_cache: dict[str, bytes] = {}
        self._model_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._runtime is not None and self._image_index is not None and self._caption_index is not None

    @property
    def device(self) -> str:
        return "cpu"

    def _make_gallery(self) -> list[tuple[str, str]]:
        from PIL import Image, UnidentifiedImageError

        root = self.config.image_root
        root.mkdir(parents=True, exist_ok=True)
        local_root = self.config.root / "data/raw/coco2017/val2017"
        for filename, _, source_url in CLOUD_GALLERY:
            destination = root / filename
            if not destination.is_file():
                source_path = local_root / filename
                if source_path.is_file():
                    payload = source_path.read_bytes()
                else:
                    try:
                        with urlopen(source_url, timeout=30) as response:
                            payload = response.read(self.config.max_upload_bytes)
                    except (OSError, TimeoutError) as error:
                        raise RuntimeError(f"could not download the compact demo image {filename}") from error
                if not payload:
                    raise RuntimeError(f"compact demo image {filename} was empty")
                destination.write_bytes(payload)
            try:
                with Image.open(destination) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                raise RuntimeError(f"compact demo image {filename} is unreadable") from error
        return [(filename, caption) for filename, caption, _ in CLOUD_GALLERY]

    def _load_public_assets(self, runtime: Any) -> tuple[Any, Any] | None:
        asset_root = self.config.root / CLOUD_ASSET_DIR
        metadata_path = asset_root / "gallery.json"
        embeddings_path = asset_root / "embeddings.npz"
        if not metadata_path.is_file() or not embeddings_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(embeddings_path, allow_pickle=False) as arrays:
                image_vectors = np.asarray(arrays["image_embeddings"], dtype=np.float32)
                caption_vectors = np.asarray(arrays["caption_embeddings"], dtype=np.float32)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise RuntimeError("the public compact gallery assets could not be read") from error
        if metadata.get("model_id") != CLOUD_MODEL_ID:
            raise RuntimeError("the public compact gallery model identity is incompatible")
        images = metadata.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError("the public compact gallery metadata is empty")
        captions = [caption for image in images for caption in image.get("captions", [])]
        if image_vectors.shape != (len(images), runtime.embedding_dimension):
            raise RuntimeError("public image embeddings have an incompatible shape")
        if caption_vectors.shape != (len(captions), runtime.embedding_dimension):
            raise RuntimeError("public caption embeddings have an incompatible shape")
        import faiss

        image_index = faiss.IndexFlatIP(runtime.embedding_dimension)
        caption_index = faiss.IndexFlatIP(runtime.embedding_dimension)
        image_index.add(image_vectors)
        caption_index.add(caption_vectors)
        self._image_ids = tuple(str(image["image_id"]) for image in images)
        self._caption_ids = tuple(str(caption["caption_id"]) for caption in captions)
        self._image_metadata = {
            str(image["image_id"]): {
                "image_id": str(image["image_id"]),
                "filename": "",
                "image_url": str(image["image_url"]),
                "caption_count": len(image.get("captions", [])),
            }
            for image in images
        }
        self._caption_metadata = {
            str(caption["caption_id"]): {
                "caption_id": str(caption["caption_id"]),
                "image_id": str(caption["image_id"]),
                "text": str(caption["text"]),
            }
            for caption in captions
        }
        return image_index, caption_index

    def load(self) -> dict[str, Any]:
        if self.ready:
            return self.info()
        try:
            import faiss

            runtime = load_clip_runtime(CLOUD_MODEL_ID, requested_device="cpu")
            public_assets = self._load_public_assets(runtime)
            if public_assets is None:
                gallery = self._make_gallery()
                image_items = [(f"demo-image-{index}", self.config.image_root / filename) for index, (filename, _) in enumerate(gallery)]
                image_batch = encode_images(image_items, runtime, batch_size=5)
                caption_batch = self._encode_captions(gallery, runtime)
                image_vectors = np.asarray(image_batch.embeddings, dtype=np.float32)
                caption_vectors = np.asarray(caption_batch, dtype=np.float32)
                image_index = faiss.IndexFlatIP(runtime.embedding_dimension)
                caption_index = faiss.IndexFlatIP(runtime.embedding_dimension)
                image_index.add(image_vectors)
                caption_index.add(caption_vectors)
                self._image_ids = tuple(item_id for item_id, _ in image_items)
                self._caption_ids = tuple(f"demo-caption-{index}" for index in range(len(gallery)))
                self._image_metadata = {
                    item_id: {"image_id": item_id, "filename": filename}
                    for item_id, (filename, _) in zip(self._image_ids, gallery, strict=True)
                }
                self._caption_metadata = {
                    caption_id: {
                        "caption_id": caption_id,
                        "image_id": self._image_ids[index],
                        "text": gallery[index][1],
                    }
                    for index, caption_id in enumerate(self._caption_ids)
                }
            else:
                image_index, caption_index = public_assets
            self._runtime = runtime
            self._image_index = image_index
            self._caption_index = caption_index
            return self.info()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("the compact CPU demo could not load its public CLIP runtime") from error

    def _encode_captions(self, gallery: list[tuple[str, str]], runtime: Any) -> tuple[tuple[float, ...], ...]:
        from ..clip_baseline import encode_texts

        batch = encode_texts(
            [(f"demo-caption-{index}", caption) for index, (_, caption) in enumerate(gallery)],
            runtime,
            batch_size=5,
        )
        return batch.embeddings

    def _encode_text(self, query: str) -> np.ndarray:
        assert self._runtime is not None
        with self._model_lock, self._runtime.torch.inference_mode():
            processed = self._runtime.processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._runtime.text_max_length,
            )
            inputs = _model_inputs(processed, self._runtime.device)
            rows = _output_rows(self._runtime.model.get_text_features(**inputs))
        return np.asarray(normalize_embeddings(rows), dtype=np.float32)

    def _encode_image(self, image: Any) -> np.ndarray:
        assert self._runtime is not None
        with self._model_lock, self._runtime.torch.inference_mode():
            processed = self._runtime.processor(images=[image], return_tensors="pt")
            inputs = _model_inputs(processed, self._runtime.device)
            rows = _output_rows(self._runtime.model.get_image_features(pixel_values=inputs["pixel_values"]))
        return np.asarray(normalize_embeddings(rows), dtype=np.float32)

    def _search(self, index: Any, vector: np.ndarray, ids: tuple[str, ...], metadata: dict[str, dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        limit = min(int(top_k), len(ids))
        scores, positions = index.search(vector, limit)
        return [
            {"id": ids[int(position)], "rank": rank, "score": float(score), "metadata": dict(metadata[ids[int(position)]])}
            for rank, (position, score) in enumerate(zip(positions[0], scores[0], strict=True), start=1)
        ]

    def info(self) -> dict[str, Any]:
        return {
            "project": "OmniSearch",
            "model_family": "CLIP ViT-B/32",
            "model_id": CLOUD_MODEL_ID,
            "embedding_dimension": getattr(self._runtime, "embedding_dimension", 512),
            "retrieval_backend": self.retrieval_backend,
            "supported_query_modes": ["text-to-image", "image-to-text"],
            "default_top_k": self.config.default_top_k,
            "max_top_k": self.config.max_top_k,
            "api_version": self.config.api_version,
            "protocol_version": self.config.protocol_version,
            "device": self.device,
            "deployment_mode": self.deployment_mode,
            "gallery_size": len(self._image_ids) or 5,
            "gallery_note": "Public COCO validation gallery; not the full validated benchmark.",
        }

    def get_image_preview(self, image_id: str) -> bytes:
        """Fetch one public result image server-side for reliable UI rendering."""

        if image_id in self._preview_cache:
            return self._preview_cache[image_id]
        metadata = self._image_metadata.get(image_id)
        if metadata is None or not metadata.get("image_url"):
            raise RuntimeError("image preview is unavailable")
        source_url = str(metadata["image_url"])
        urls = (source_url, source_url.replace("https://", "http://", 1))
        last_error: BaseException | None = None
        for url in urls:
            try:
                with urlopen(url, timeout=15) as response:
                    payload = response.read(self.config.max_upload_bytes)
                self._decode_image(payload).close()
                self._preview_cache[image_id] = payload
                return payload
            except (OSError, TimeoutError, MalformedImageError) as error:
                last_error = error
        raise RuntimeError("public image preview could not be downloaded") from last_error

    def _decode_image(self, payload: bytes) -> Any:
        from PIL import Image, UnidentifiedImageError

        if not payload or len(payload) > self.config.max_upload_bytes:
            raise MalformedImageError("image payload exceeds the configured upload limit")
        try:
            with Image.open(io.BytesIO(payload)) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP"}:
                    raise MalformedImageError("unsupported image format")
                probe.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                return image.convert("RGB")
        except MalformedImageError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise MalformedImageError("uploaded file is not a valid supported image") from error

    def search_text_to_image(self, query: str, top_k: int, request_id: str) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("compact demo service is not ready")
        started = time.perf_counter()
        try:
            results = self._search(self._image_index, self._encode_text(query.strip()), self._image_ids, self._image_metadata, top_k)
        except (RuntimeError, TypeError, ValueError) as error:
            raise RetrievalExecutionError("text retrieval failed") from error
        return self._response("text-to-image", query.strip(), results, started, request_id)

    def search_image_to_text(self, payload: bytes, top_k: int, request_id: str) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("compact demo service is not ready")
        started = time.perf_counter()
        image = self._decode_image(payload)
        try:
            results = self._search(self._caption_index, self._encode_image(image), self._caption_ids, self._caption_metadata, top_k)
        except (RuntimeError, TypeError, ValueError) as error:
            raise RetrievalExecutionError("image retrieval failed") from error
        finally:
            image.close()
        return self._response("image-to-text", None, results, started, request_id)

    def _response(self, query_type: str, query: str | None, results: list[dict[str, Any]], started: float, request_id: str) -> dict[str, Any]:
        return {
            "query_type": query_type,
            "query": query,
            "results": results,
            "model_system": self.system_identifier,
            "retrieval_backend": self.retrieval_backend,
            "latency_ms": {"total_server_ms": max(0.0, (time.perf_counter() - started) * 1000.0)},
            "request_id": request_id,
        }

    def close(self) -> None:
        self._runtime = None
        self._image_index = None
        self._caption_index = None
        self._preview_cache.clear()
        self._temporary_gallery.cleanup()
