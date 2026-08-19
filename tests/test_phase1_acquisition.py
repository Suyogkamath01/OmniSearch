import tempfile
import unittest
from pathlib import Path

from omnisearch.acquisition import build_manifest, parse_caption_html
from omnisearch.manifest import ImageRecord, validate_manifest, write_manifest
from omnisearch.preprocessing import normalize_text

FIXTURE_HTML = """
<html><body>
<a href="123.jpg">123.jpg</a><ul>
<li> A Caption with  spaces. </li><li>Second caption.</li>
<li>Third caption.</li><li>Fourth caption.</li><li>Fifth caption.</li>
</ul>
<td>Image Not Found</td><ul>
<li>Missing image caption one.</li><li>Missing image caption two.</li>
<li>Missing image caption three.</li><li>Missing image caption four.</li>
<li>Missing image caption five.</li>
</ul>
<a href="456.jpg">456.jpg</a><ul>
<li>Another caption.</li><li>Second caption.</li><li>Third caption.</li>
<li>Fourth caption.</li><li>Fifth caption.</li>
</ul>
</body></html>
"""


class AcquisitionTests(unittest.TestCase):
    def test_caption_html_parser_preserves_image_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.html"
            path.write_text(FIXTURE_HTML, encoding="utf-8")
            records = parse_caption_html(
                path, "https://example.test/data/flickr30k.html"
            )

        self.assertEqual(
            [record.image_id for record in records],
            ["123", "missing-source-image-000002", "456"],
        )
        self.assertEqual([len(record.captions) for record in records], [5, 5, 5])
        self.assertEqual(records[0].captions[0].caption_id, "123#0")
        self.assertEqual(records[0].image_url, "https://example.test/data/123.jpg")
        self.assertIsNone(records[1].filename)
        self.assertFalse(records[1].source_image_id_available)

    def test_normalization_is_conservative_and_deterministic(self) -> None:
        self.assertEqual(
            normalize_text("  Café\u00a0IN  the  Park! "), "café in the park!"
        )

    def test_manifest_validation_detects_duplicate_ids_and_missing_captions(
        self,
    ) -> None:
        records = (
            ImageRecord("same", "same.jpg", ()),
            ImageRecord("same", "same.jpg", ()),
        )
        report = validate_manifest(records)
        self.assertFalse(report.passed)
        self.assertEqual(report.duplicate_image_ids, {"same": 2})
        self.assertEqual(report.missing_caption_image_ids, ("same", "same"))

    def test_manifest_validation_detects_structural_errors(self) -> None:
        records = (ImageRecord("bad/id", "bad.png", (), split="unknown"),)
        report = validate_manifest(records)
        self.assertFalse(report.passed)
        self.assertTrue(any("invalid image ID" in error for error in report.errors))
        self.assertTrue(any("invalid split" in error for error in report.errors))

    def test_manifest_fingerprint_is_reproducible_for_same_source_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "captions.html"
            source.write_text(FIXTURE_HTML, encoding="utf-8")
            records = parse_caption_html(
                source, "https://example.test/data/flickr30k.html"
            )
            first = build_manifest(
                records, source, source_last_modified="fixed-source-marker"
            )
            second = build_manifest(
                records, source, source_last_modified="fixed-source-marker"
            )
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            first_digest = write_manifest(first, first_path)
            second_digest = write_manifest(second, second_path)

        self.assertEqual(first_digest, second_digest)
