import json
import tempfile
import unittest
from pathlib import Path

from omnisearch.coco_acquisition import (
    COCO_CAPTION_ANNOTATIONS_URL,
    COCO_TERMS_URL,
    COCO_VAL_IMAGES_URL,
    build_coco_manifest,
    parse_coco_captions,
)
from omnisearch.evaluation import run_phase4_real_migration
from omnisearch.manifest import (
    CaptionRecord,
    DatasetManifest,
    ImageRecord,
    validate_manifest,
    write_manifest,
)
from omnisearch.splitting import assert_no_split_leakage, assign_image_grouped_splits


class CocoDatasetTests(unittest.TestCase):
    def test_parser_preserves_image_groups_and_variable_official_counts(self) -> None:
        payload = {
            "images": [
                {"id": 2, "file_name": "000000000002.jpg"},
                {"id": 1, "file_name": "000000000001.jpg"},
            ],
            "annotations": [
                {"id": 11, "image_id": 2, "caption": "Two"},
                {"id": 12, "image_id": 2, "caption": "Two"},
                {"id": 13, "image_id": 1, "caption": "One"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "captions.json"
            archive_path = root / "val2017.zip"
            annotation_path.write_text(json.dumps(payload), encoding="utf-8")
            archive_path.write_bytes(b"archive fixture")
            records = parse_coco_captions(annotation_path)
            manifest = build_coco_manifest(records, annotation_path, archive_path)

        self.assertEqual([record.image_id for record in records], ["1", "2"])
        self.assertEqual(records[1].captions[0].caption_id, "coco2017_val#11")
        self.assertEqual(manifest.dataset_id, "coco2017_val")
        self.assertEqual(manifest.source_url, COCO_CAPTION_ANNOTATIONS_URL)
        self.assertEqual(manifest.terms_url, COCO_TERMS_URL)
        self.assertEqual(manifest.metadata["image_archive_url"], COCO_VAL_IMAGES_URL)

    def test_coco_validation_allows_extra_captions_but_detects_missing(self) -> None:
        payload = {
            "images": [
                {"id": 1, "file_name": "000000000001.jpg"},
                {"id": 2, "file_name": "000000000002.jpg"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "one"},
                {"id": 2, "image_id": 1, "caption": "one"},
                {"id": 3, "image_id": 1, "caption": "one"},
                {"id": 4, "image_id": 1, "caption": "one"},
                {"id": 5, "image_id": 1, "caption": "one"},
                {"id": 6, "image_id": 1, "caption": "one"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = parse_coco_captions(path)
        report = validate_manifest(
            records,
            expected_captions_per_image=None,
            allowed_image_extensions={".jpg", ".jpeg"},
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.missing_caption_image_ids, ("2",))
        self.assertTrue(report.duplicate_caption_groups)

    def test_coco_splits_are_image_grouped_and_caption_disjoint(self) -> None:
        records = tuple(
            record
            for record in parse_coco_captions_from_records()
        )
        split = assign_image_grouped_splits(records, seed=42)
        assert_no_split_leakage(split)
        self.assertEqual(len(split), 3)

    def test_real_phase4_rankings_migrate_through_canonical_evaluator(self) -> None:
        records = (
            ImageRecord(
                "1",
                "000000000001.jpg",
                (CaptionRecord("coco2017_val#1", "one"),),
                split="test",
            ),
            ImageRecord(
                "2",
                "000000000002.jpg",
                (CaptionRecord("coco2017_val#2", "two"),),
                split="test",
            ),
        )
        manifest = DatasetManifest(
            "coco2017_val",
            "test",
            "https://example.test/captions",
            "https://example.test/terms",
            "fixture",
            None,
            records,
            {},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            report_path = root / "phase4.json"
            write_manifest(manifest, manifest_path)
            report_path.write_text(
                json.dumps(
                    {
                        "scope": {"real_dataset_evaluation": True},
                        "evaluation": {"split": "test"},
                        "model": {"model_id": "fixture/clip", "device": "cpu"},
                        "rankings": {
                            "text_to_image": {
                                "coco2017_val#1": [
                                    {"id": "1", "score": 1.0},
                                    {"id": "2", "score": 0.0},
                                ],
                                "coco2017_val#2": [
                                    {"id": "2", "score": 1.0},
                                    {"id": "1", "score": 0.0},
                                ],
                            },
                            "image_to_text": {
                                "1": [
                                    {"id": "coco2017_val#1", "score": 1.0},
                                    {"id": "coco2017_val#2", "score": 0.0},
                                ],
                                "2": [
                                    {"id": "coco2017_val#2", "score": 1.0},
                                    {"id": "coco2017_val#1", "score": 0.0},
                                ],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = run_phase4_real_migration(
                report_path, manifest_path, root / "output", bootstrap_resamples=10
            )
        self.assertEqual(result["status"], "migrated_real")
        self.assertEqual(
            {item["task"] for item in result["results"]},
            {"text_to_image", "image_to_text"},
        )


def parse_coco_captions_from_records():
    from omnisearch.manifest import CaptionRecord, ImageRecord

    return (
        ImageRecord(
            str(index),
            f"{index:012d}.jpg",
            (CaptionRecord(f"{index}#0", f"caption {index}"),),
        )
        for index in range(3)
    )
