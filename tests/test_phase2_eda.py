import base64
import tempfile
import unittest
from pathlib import Path

from omnisearch.eda import (
    caption_statistics,
    duplicate_caption_analysis,
    image_header_metadata,
    image_statistics,
    leakage_reaudit,
    run_eda,
    stable_sample,
)
from omnisearch.manifest import (
    CaptionRecord,
    DatasetManifest,
    ImageRecord,
    write_manifest,
)

VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def record(
    image_id: str, split: str | None = None, filename: str | None = None
) -> ImageRecord:
    captions = tuple(
        CaptionRecord(f"{image_id}#{index}", f"caption {index} for {image_id}")
        for index in range(5)
    )
    return ImageRecord(image_id, filename or f"{image_id}.jpg", captions, split=split)


class Phase2EdaTests(unittest.TestCase):
    def test_caption_statistics_and_duplicate_analysis_are_deterministic(self) -> None:
        records = (
            ImageRecord(
                "one",
                "one.jpg",
                (
                    CaptionRecord("one#0", "A dog."),
                    CaptionRecord("one#1", "one two three four"),
                ),
            ),
            ImageRecord(
                "two",
                "two.jpg",
                (
                    CaptionRecord("two#0", " a   dog. "),
                    CaptionRecord("two#1", ""),
                ),
            ),
        )

        stats = caption_statistics(records, short_threshold=3, long_threshold=3)
        duplicates = duplicate_caption_analysis(records)

        self.assertEqual(stats["caption_count"], 4)
        self.assertEqual(stats["short_caption_count"], 3)
        self.assertEqual(stats["long_caption_count"], 1)
        self.assertEqual(stats["empty_caption_count"], 1)
        self.assertEqual(stats["normalized_duplicate_caption_group_count"], 1)
        self.assertEqual(stats["normalized_duplicate_caption_groups_across_images"], 1)
        self.assertEqual(duplicates["cross_image_group_count"], 1)

    def test_image_statistics_reports_fixture_integrity_and_dimensions(self) -> None:
        records = (
            record("valid", filename="valid.png"),
            record("duplicate", filename="duplicate.png"),
            record("corrupt", filename="corrupt.png"),
            record("missing", filename="missing.png"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid.png").write_bytes(VALID_PNG)
            (root / "duplicate.png").write_bytes(VALID_PNG)
            (root / "corrupt.png").write_bytes(b"not an image")

            stats = image_statistics(records, root)
            metadata = image_header_metadata(root / "valid.png")

        self.assertEqual(stats["status"], "completed")
        self.assertEqual(stats["images_decoded"], 2)
        self.assertEqual(stats["missing_image_count"], 1)
        self.assertEqual(stats["corrupted_image_count"], 1)
        self.assertEqual(stats["exact_duplicate_group_count"], 1)
        self.assertEqual(stats["width"]["min"], 1.0)
        self.assertEqual(stats["height"]["max"], 1.0)
        self.assertEqual(metadata["width"], 1)
        self.assertEqual(metadata["height"], 1)

    def test_leakage_reaudit_distinguishes_critical_and_benign_overlap(self) -> None:
        train = record("train-image", split="train")
        test = record("test-image", split="test")
        test = ImageRecord(
            test.image_id,
            test.filename,
            (CaptionRecord("test-image#0", train.captions[0].text),)
            + test.captions[1:],
            split=test.split,
        )

        findings = leakage_reaudit((train, test), None)["findings"]
        self.assertFalse(any(item["severity"] == "CRITICAL" for item in findings))
        self.assertTrue(
            any(item["severity"] == "POTENTIAL/BENIGN" for item in findings)
        )

        cross_split_same_image = (train, train.with_split("test"))
        critical = leakage_reaudit(cross_split_same_image, None)["findings"]
        self.assertTrue(any(item["severity"] == "CRITICAL" for item in critical))

    def test_stable_sample_is_reproducible(self) -> None:
        records = tuple(record(str(index), split="train") for index in range(8))
        self.assertEqual(
            stable_sample(records, seed=42, sample_size=4),
            stable_sample(records, seed=42, sample_size=4),
        )
        self.assertEqual(len(stable_sample(records, seed=42, sample_size=4)), 4)

    def test_run_eda_writes_machine_and_human_readable_artifacts(self) -> None:
        records = tuple(
            record(str(index), split="train" if index < 2 else "validation")
            for index in range(3)
        )
        manifest = DatasetManifest(
            dataset_id="fixture",
            dataset_version="fixture-v1",
            source_url="https://example.test/fixture",
            terms_url="https://example.test/terms",
            source_snapshot_marker="fixture",
            source_sha256="fixture-sha256",
            records=records,
            metadata={"fixture": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_dir = root / "phase2"
            write_manifest(manifest, manifest_path)

            report = run_eda(
                manifest_path,
                config_path=Path("configs/default.toml"),
                output_dir=output_dir,
                sample_size=2,
            )

            self.assertTrue((output_dir / "phase2_report.json").exists())
            self.assertTrue((output_dir / "phase2_report.md").exists())
            self.assertTrue((output_dir / "sample_selection.json").exists())
            self.assertTrue(
                (output_dir / "figures" / "caption_word_length.svg").exists()
            )

            missing_image_report = run_eda(
                manifest_path,
                config_path=Path("configs/default.toml"),
                output_dir=root / "phase2-missing-images",
                image_root=root / "missing-images",
                sample_size=2,
            )

        self.assertTrue(report["scope"]["real_metadata_eda"])
        self.assertFalse(report["scope"]["real_image_eda"])
        self.assertFalse(missing_image_report["scope"]["real_image_eda"])
        self.assertEqual(
            report["qualitative_gallery"]["status"], "not_generated_no_image_root"
        )
        self.assertEqual(report["provenance"]["sample_size"], 2)


if __name__ == "__main__":
    unittest.main()
