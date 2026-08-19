"""Focused checks for the artifact-free Streamlit deployment path."""

from pathlib import Path

from omnisearch.api.config import ServiceConfig
from omnisearch.ui.adapter import format_result
from omnisearch.ui.cloud_demo import CompactCloudDemoService, local_artifacts_available


def test_cloud_fallback_is_selected_when_full_artifacts_are_absent(tmp_path: Path) -> None:
    config = ServiceConfig.from_env(tmp_path)

    assert not local_artifacts_available(config)
    service = CompactCloudDemoService(config)
    try:
        assert service.config.device == "cpu"
        assert service.config.model_id == "openai/clip-vit-base-patch32"
        assert service.info()["deployment_mode"] == "compact_cloud_demo"
        assert service.info()["gallery_size"] == 5
    finally:
        service.close()


def test_cloud_fallback_detects_a_complete_artifact_shape(tmp_path: Path) -> None:
    config = ServiceConfig.from_env(tmp_path)
    config.cache_dir.mkdir(parents=True)
    config.image_root.mkdir(parents=True)
    config.phase20_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.image_index_path.parent.mkdir(parents=True, exist_ok=True)
    config.caption_index_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (
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
    ):
        path.touch()

    assert local_artifacts_available(config)


def test_cloud_image_results_preserve_remote_preview_urls() -> None:
    row = format_result(
        {
            "id": "image-1",
            "rank": 1,
            "score": 0.8,
            "metadata": {"image_id": "image-1", "image_url": "https://example.test/image.jpg"},
        },
        "text-to-image",
    )

    assert row["image_url"] == "https://example.test/image.jpg"
