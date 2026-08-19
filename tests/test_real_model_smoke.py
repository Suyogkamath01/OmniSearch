"""Optional real-resource smoke; excluded from the default fast suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnisearch.api.config import ServiceConfig
from omnisearch.api.retrieval import RetrievalService
from omnisearch.manifest import read_manifest

pytestmark = [pytest.mark.real_model, pytest.mark.slow, pytest.mark.local_data]


def test_real_model_bidirectional_retrieval() -> None:
    """Exercise both directions against the already validated local resources."""

    root = Path.cwd()
    config = ServiceConfig.from_env(root)
    required = {
        "checkpoint": config.checkpoint_path,
        "manifest": config.manifest_path,
        "image_root": config.image_root,
        "image_index": config.image_index_path,
        "caption_index": config.caption_index_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        pytest.fail(
            "Missing local real-model smoke resources: "
            + ", ".join(missing)
            + ". Run the validated Phase 21/22 setup or set the OMNISEARCH_* paths."
        )
    service = RetrievalService(config)
    service.load()
    manifest = read_manifest(config.manifest_path)
    image_path = next(
        config.image_root / str(record.filename)
        for record in manifest.records
        if record.filename and (config.image_root / record.filename).is_file()
    )
    text_response = service.search_text_to_image("a person outdoors", 5, "phase23-real-text")
    image_response = service.search_image_to_text(image_path.read_bytes(), 5, "phase23-real-image")
    assert service.ready is True
    assert len(text_response["results"]) == 5
    assert len(image_response["results"]) == 5
