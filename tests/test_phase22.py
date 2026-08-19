from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from omnisearch.api.errors import MalformedImageError, ResourceUnavailableError
from omnisearch.phase22 import audit_phase21
from omnisearch.ui.adapter import (
    format_results,
    run_image_search,
    run_text_search,
    safe_error_message,
    validate_text_query,
    validate_top_k,
)


class FakeService:
    config = SimpleNamespace(max_query_chars=20)

    def search_text_to_image(self, query: str, top_k: int, request_id: str) -> dict[str, Any]:
        return self._response("text-to-image", query, top_k, request_id)

    def search_image_to_text(self, payload: bytes, top_k: int, request_id: str) -> dict[str, Any]:
        if payload == b"bad":
            raise MalformedImageError("bad image")
        return self._response("image-to-text", None, top_k, request_id)

    @staticmethod
    def _response(query_type: str, query: str | None, top_k: int, request_id: str) -> dict[str, Any]:
        rows = []
        for index in range(1, top_k + 1):
            if query_type == "text-to-image":
                metadata = {"image_id": f"image-{index}", "filename": f"{index}.jpg"}
            else:
                metadata = {"caption_id": f"caption-{index}", "image_id": f"image-{index}", "text": f"caption {index}"}
            rows.append({"id": f"item-{index}", "rank": index, "score": 1.0 / index, "metadata": metadata})
        return {
            "query_type": query_type,
            "query": query,
            "results": rows,
            "latency_ms": {"total_server_ms": 0.1, "preprocessing_ms": 0.01, "query_encoding_ms": 0.05, "search_ms": 0.01},
            "request_id": request_id,
        }


def test_ui_validation_bounds_and_normalization() -> None:
    assert validate_text_query("  a bicycle  ") == "a bicycle"
    assert validate_top_k(20) == 20
    with pytest.raises(ValueError, match="non-empty"):
        validate_text_query("   ")
    with pytest.raises(ValueError, match="between"):
        validate_top_k(21)


def test_adapter_delegates_both_modes_and_formats_stable_fields() -> None:
    service = FakeService()
    text = run_text_search(service, "  a bicycle ", 2, request_id="text-test")
    image = run_image_search(service, b"valid", 2, request_id="image-test")
    assert text["query"] == "a bicycle"
    assert text["request_id"] == "text-test"
    assert text["latency_ms"]["ui_service_call_wall_ms"] >= 0
    assert text["latency_ms"]["ui_adapter_overhead_ms"] >= 0
    assert format_results(text, "text-to-image")[0]["filename"] == "1.jpg"
    assert format_results(image, "image-to-text")[0]["text"] == "caption 1"


def test_adapter_rejects_malformed_image_and_safe_messages() -> None:
    with pytest.raises(MalformedImageError):
        run_image_search(FakeService(), b"bad", 1)
    assert "traceback" not in safe_error_message(MalformedImageError("bad")).casefold()
    assert "not ready" in safe_error_message(ResourceUnavailableError("not ready")).casefold()


def test_adapter_enforces_upload_size_before_service_call() -> None:
    service = FakeService()
    service.config.max_upload_bytes = 3
    with pytest.raises(MalformedImageError, match="at most"):
        run_image_search(service, b"1234", 1)


def test_adapter_handles_unready_service_without_retrieval_logic() -> None:
    class Unready(FakeService):
        def search_text_to_image(self, query: str, top_k: int, request_id: str) -> dict[str, Any]:
            raise ResourceUnavailableError("not ready")

    with pytest.raises(ResourceUnavailableError):
        run_text_search(Unready(), "query", 1)


@pytest.mark.local_data
def test_phase21_dependency_audit_passes_before_ui() -> None:
    audit = audit_phase21(Path.cwd())
    assert audit["passed"] is True
    assert audit["audit_result"] == "PRE-PHASE AUDIT: Phase 21 PASS"
    assert audit["phase23_started"] is False
