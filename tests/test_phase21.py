from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from omnisearch.api.app import create_app
from omnisearch.api.config import ServiceConfig
from omnisearch.api.errors import MalformedImageError, StartupValidationError
from omnisearch.api.retrieval import RetrievalService
from omnisearch.phase21 import audit_phase20


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 8), color=(30, 100, 180)).save(output, format="PNG")
    return output.getvalue()


class StubService:
    ready = True
    device = "cpu"
    dimension = 512

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "project": "OmniSearch",
            "model_loaded": True,
            "indexes_loaded": True,
            "device": "cpu",
            "api_version": "v1",
        }

    def readiness(self) -> tuple[bool, str | None]:
        return True, None

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
        return self._response("text-to-image", query, top_k, request_id)

    def search_image_to_text(self, payload: bytes, top_k: int, request_id: str) -> dict[str, Any]:
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


@pytest.fixture()
def client() -> TestClient:
    config = ServiceConfig.from_env(Path.cwd())
    return TestClient(create_app(config=config, service=StubService(), load_on_startup=False))


def test_service_endpoints_and_response_schema(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert health.headers["X-Frame-Options"] == "DENY"
    assert health.headers["Referrer-Policy"] == "no-referrer"
    assert health.headers["Cache-Control"] == "no-store"
    assert client.get("/ready").status_code == 200
    info = client.get("/info")
    assert info.status_code == 200
    assert info.json()["retrieval_backend"] == "FAISS Flat exact inner-product search"
    assert "/Users/" not in info.text

    text = client.post("/search/text-to-image", json={"query": "a blue object", "top_k": 3})
    assert text.status_code == 200
    assert len(text.json()["results"]) == 3
    assert text.json()["latency_ms"]["search_ms"] >= 0

    image = client.post(
        "/search/image-to-text",
        files={"image": ("fixture.png", _image_bytes(), "image/png")},
        data={"top_k": "2"},
    )
    assert image.status_code == 200
    assert image.json()["query_type"] == "image-to-text"


def test_input_validation_and_safe_errors(client: TestClient) -> None:
    assert client.post("/search/text-to-image", json={"query": "   "}).status_code == 422
    assert client.post("/search/text-to-image", json={"query": "query", "top_k": 51}).status_code == 422
    malformed = client.post(
        "/search/image-to-text",
        files={"image": ("bad.bin", b"not image", "application/octet-stream")},
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"] == "malformed_image"
    assert "Traceback" not in malformed.text
    wrong_media_type = client.post(
        "/search/image-to-text",
        files={"image": ("fixture.png", _image_bytes(), "text/plain")},
    )
    assert wrong_media_type.status_code == 400
    assert wrong_media_type.json()["error_category"] == "IMAGE_DECODE_ERROR"


def test_unready_service_returns_503() -> None:
    class Unready(StubService):
        ready = False

        def readiness(self) -> tuple[bool, str | None]:
            return False, "not loaded"

    app = create_app(config=ServiceConfig.from_env(Path.cwd()), service=Unready(), load_on_startup=False)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/search/text-to-image", json={"query": "query"}).status_code == 503


@pytest.mark.local_data
def test_phase20_focused_audit_passes() -> None:
    audit = audit_phase20(Path.cwd())
    assert audit["passed"] is True
    assert audit["audit_result"] == "PRE-PHASE AUDIT: Phase 20 PASS"


def test_missing_resources_fail_startup_validation(tmp_path: Path) -> None:
    service = RetrievalService(ServiceConfig.from_env(tmp_path))
    with pytest.raises(StartupValidationError, match="missing Phase 20 service resources"):
        service.validate_startup()


def test_incompatible_index_metadata_fails_startup(tmp_path: Path) -> None:
    service = RetrievalService(ServiceConfig.from_env(Path.cwd()))
    bad_metadata = tmp_path / "bad.metadata.json"
    bad_metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(StartupValidationError, match="could not load compatible FAISS Flat index"):
        service._load_index(
            service.config.image_index_path,
            bad_metadata,
            ["not-the-corpus"],
            "image_group",
            "wrong-manifest-hash",
            512,
        )


def test_image_decoder_rejects_malformed_payload() -> None:
    service = RetrievalService(ServiceConfig.from_env(Path.cwd()))
    with pytest.raises(MalformedImageError):
        service.decode_image(b"not an image")


def test_image_decoder_uses_configured_pixel_limit() -> None:
    base = ServiceConfig.from_env(Path.cwd())
    service = RetrievalService(replace(base, max_image_pixels=10))
    with pytest.raises(MalformedImageError, match="dimensions"):
        service.decode_image(_image_bytes())
