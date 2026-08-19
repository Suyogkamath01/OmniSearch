import base64
import tempfile
import unittest
from pathlib import Path

from omnisearch.image_validation import validate_image_records
from omnisearch.manifest import CaptionRecord, ImageRecord

# A tiny valid PNG fixture. Image bytes are test data, not dataset content.
VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def record(image_id: str) -> ImageRecord:
    captions = tuple(CaptionRecord(f"{image_id}#{i}", f"caption {i}") for i in range(5))
    return ImageRecord(image_id, f"{image_id}.png", captions)


class ImageValidationTests(unittest.TestCase):
    def test_missing_corrupt_and_exact_duplicate_images_are_reported(self) -> None:
        records = (
            record("valid"),
            record("duplicate"),
            record("corrupt"),
            record("missing"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid.png").write_bytes(VALID_PNG)
            (root / "duplicate.png").write_bytes(VALID_PNG)
            (root / "corrupt.png").write_bytes(b"not an image")
            report = validate_image_records(records, root)

        self.assertEqual(report.status, "completed")
        self.assertEqual(report.missing_image_ids, ("missing",))
        self.assertEqual(report.corrupted_image_ids, ("corrupt",))
        self.assertEqual(len(report.exact_duplicate_groups), 1)

    def test_no_image_root_is_explicitly_not_run(self) -> None:
        report = validate_image_records((record("one"),), None)
        self.assertEqual(report.status, "not_run_no_image_root")
