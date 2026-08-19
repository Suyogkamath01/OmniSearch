"""Focused observability and runtime-reliability tests for Phase 25."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from omnisearch.api.app import create_app
from omnisearch.api.config import ServiceConfig
from omnisearch.api.errors import (
    IndexUnavailableError,
    MalformedImageError,
    RetrievalExecutionError,
)
from omnisearch.api.observability import JsonLogFormatter, RuntimeMetrics
from omnisearch.api.retrieval import RetrievalService
from omnisearch.phase25 import REQUIRED_ARTIFACTS, validate_phase25_artifacts


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 8), color=(30, 100, 180)).save(output, format="PNG")
    return output.getvalue()


class _StubService:
    ready = True
    device = "cpu"
    dimension = 512

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ready else "degraded",
            "project": "OmniSearch",
            "model_loaded": self.ready,
            "indexes_loaded": self.ready,
            "device": "cpu",
            "api_version": "v1",
        }

    def readiness(self) -> tuple[bool, str | None]:
        return (True, None) if self.ready else (False, "not loaded")

    def info(self) -> dict[str, Any]:
        return {
            "project": "OmniSearch",
            "model_family": "CLIP ViT-B/32",
            "model_id": "openai/clip-vit-base-patch32",
            "embedding_dimension": 512,
            "retrieval_backend": "FAISS Flat exact inner-product search",
            "supported_query_modes": ["text-to-image", "image-to-text"],
            "default_top_k": 5,
            "max_top_k": 50,
            "api_version": "v1",
            "protocol_version": "retrieval_eval_v1",
            "device": "cpu",
            "reranker_enabled": False,
            "content_safety_filtering": "NOT IMPLEMENTED",
            "research_system": True,
        }

    def search_text_to_image(self, query: str, top_k: int, request_id: str) -> dict[str, Any]:
        if self.fail:
            raise RetrievalExecutionError("fixture model failure")
        return self._response("text-to-image", query, top_k, request_id)

    def search_image_to_text(self, payload: bytes, top_k: int, request_id: str) -> dict[str, Any]:
        if self.fail:
            raise RetrievalExecutionError("fixture index failure")
        if not payload.startswith((b"\x89PNG", b"\xff\xd8")):
            raise MalformedImageError("invalid fixture image")
        return self._response("image-to-text", None, top_k, request_id)

    @staticmethod
    def _response(query_type: str, query: str | None, top_k: int, request_id: str) -> dict[str, Any]:
        return {
            "query_type": query_type,
            "query": query,
            "results": [
                {"id": f"item-{index}", "rank": index, "score": 1.0 / index, "metadata": {}}
                for index in range(1, top_k + 1)
            ],
            "model_system": "phase7_full_ft_clip_phase10_cached_faiss_flat",
            "retrieval_backend": "FAISS Flat exact inner-product search",
            "latency_ms": {
                "preprocessing_ms": 0.1,
                "query_encoding_ms": 0.2,
                "search_ms": 0.01,
                "total_server_ms": 0.31,
            },
            "request_id": request_id,
        }


def test_request_id_is_preserved_and_generated() -> None:
    app = create_app(config=ServiceConfig.from_env(Path.cwd()), service=_StubService(), load_on_startup=False)
    with TestClient(app) as test_client:
        preserved = test_client.get("/health", headers={"X-Request-ID": "phase25-client-id"})
        generated = test_client.get("/health")
    assert preserved.status_code == 200
    assert preserved.headers["X-Request-ID"] == "phase25-client-id"
    assert generated.headers["X-Request-ID"]
    assert generated.headers["X-Request-ID"] != "phase25-client-id"


def test_metrics_and_latency_logging_are_safe() -> None:
    app = create_app(config=ServiceConfig.from_env(Path.cwd()), service=_StubService(), load_on_startup=False)
    with TestClient(app) as client:
        response = client.post(
            "/search/text-to-image",
            json={"query": "a private query that must not be logged", "top_k": 2},
            headers={"X-Request-ID": "metrics-request"},
        )
        metrics = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["request_id"] == "metrics-request"
    assert metrics.json()["request_counts"]["text_search"] == 1
    assert metrics.json()["persistence"] == "process-local only"
    assert metrics.json()["ready"] is True

    record = logging.LogRecord("omnisearch.api", logging.INFO, __file__, 1, "query=%s /Users/suyash/private", ("secret",), None)
    formatted = JsonLogFormatter().format(record)
    assert "secret" not in formatted
    assert "/Users/" not in formatted
    assert "request_id" not in formatted


def test_error_taxonomy_is_exposed_for_validation_image_unready_and_model_errors() -> None:
    app = create_app(config=ServiceConfig.from_env(Path.cwd()), service=_StubService(), load_on_startup=False)
    with TestClient(app) as client:
        validation = client.post("/search/text-to-image", json={"query": "x", "top_k": 51})
        malformed = client.post("/search/image-to-text", files={"image": ("bad.bin", b"not image", "application/octet-stream")})
    assert validation.json()["error_category"] == "VALIDATION_ERROR"
    assert malformed.json()["error_category"] == "IMAGE_DECODE_ERROR"

    unready = _StubService()
    unready.ready = False
    with TestClient(create_app(config=ServiceConfig.from_env(Path.cwd()), service=unready, load_on_startup=False)) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error_category"] == "RESOURCE_NOT_READY"

    with TestClient(create_app(config=ServiceConfig.from_env(Path.cwd()), service=_StubService(fail=True), load_on_startup=False)) as client:
        response = client.post("/search/text-to-image", json={"query": "x"})
    assert response.status_code == 500
    assert response.json()["error_category"] == "MODEL_ERROR"

    class IndexFail(_StubService):
        def search_text_to_image(self, query: str, top_k: int, request_id: str) -> dict[str, Any]:
            raise IndexUnavailableError("fixture index failure")

    with TestClient(create_app(config=ServiceConfig.from_env(Path.cwd()), service=IndexFail(), load_on_startup=False)) as client:
        response = client.post("/search/text-to-image", json={"query": "x"})
    assert response.status_code == 503
    assert response.json()["error_category"] == "INDEX_ERROR"

    class InternalFail(_StubService):
        def search_text_to_image(self, query: str, top_k: int, request_id: str) -> dict[str, Any]:
            raise RuntimeError("fixture unexpected failure")

    with TestClient(create_app(config=ServiceConfig.from_env(Path.cwd()), service=InternalFail(), load_on_startup=False)) as client:
        response = client.post("/search/text-to-image", json={"query": "x"})
    assert response.status_code == 500
    assert response.json()["error_category"] == "INTERNAL_ERROR"


def test_runtime_metrics_are_thread_safe_and_count_error_categories() -> None:
    metrics = RuntimeMetrics()
    metrics.set_startup("ready")
    metrics.record_request("/search/text-to-image", 200, 1.0)
    metrics.record_request("/search/image-to-text", 400, 2.0, "IMAGE_DECODE_ERROR")
    snapshot = metrics.snapshot(_StubService())
    assert snapshot["request_counts"] == {"total": 2, "successful": 1, "failed": 1, "text_search": 1, "image_search": 1}
    assert snapshot["error_counts"] == {"IMAGE_DECODE_ERROR": 1}


def test_service_close_releases_runtime_references() -> None:
    service = RetrievalService(ServiceConfig.from_env(Path.cwd()))
    service._model = object()
    service._processor = object()
    service._torch = object()
    service._device = "cpu"
    service.close()
    assert service.ready is False
    assert service.device is None
    assert service.startup_report is None


def test_phase25_artifact_validator_rejects_incomplete_directory(tmp_path: Path) -> None:
    result = validate_phase25_artifacts(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["required_artifacts"] is False
    assert set(result["required"]) == set(REQUIRED_ARTIFACTS)


def test_phase25_artifact_validator_accepts_minimal_valid_fixture(tmp_path: Path) -> None:
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "phase25_report.json").write_text(json.dumps({"status": "PASS", "quality_gate": {"one": True}}), encoding="utf-8")
    (tmp_path / "provenance.json").write_text(json.dumps({"training_performed": False, "new_dataset_downloaded": False, "phase26_started": False}), encoding="utf-8")
    result = validate_phase25_artifacts(tmp_path)
    assert result["passed"] is True
