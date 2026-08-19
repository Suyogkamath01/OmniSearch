import tempfile
import unittest
from pathlib import Path

from omnisearch.baselines import (
    BM25Index,
    ImageHistogramIndex,
    TextDocument,
    TfidfIndex,
    average_precision,
    colour_histogram_descriptor,
    evaluate_rankings,
    histogram_intersection,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_phase3,
)
from omnisearch.manifest import (
    CaptionRecord,
    DatasetManifest,
    ImageRecord,
    write_manifest,
)


def image_record(
    image_id: str, filename: str | None = None, split: str = "test"
) -> ImageRecord:
    captions = tuple(
        CaptionRecord(f"{image_id}#{index}", f"caption {index} for {image_id}")
        for index in range(5)
    )
    return ImageRecord(image_id, filename or f"{image_id}.jpg", captions, split=split)


class Phase3BaselineTests(unittest.TestCase):
    def test_tfidf_ranking_is_deterministic_and_excludes_query(self) -> None:
        documents = (
            TextDocument("one", "image-one", "red apple"),
            TextDocument("two", "image-two", "blue car"),
            TextDocument("three", "image-three", "red apple tree"),
        )
        index = TfidfIndex().fit(documents)
        first = index.search("red apple", top_k=3, exclude_doc_id="one")
        second = index.search("red apple", top_k=3, exclude_doc_id="one")

        self.assertEqual(first, second)
        self.assertEqual(first[0].item_id, "three")
        self.assertNotIn("one", [item.item_id for item in first])
        self.assertEqual(index.search("unseen-token", top_k=3), [])

    def test_bm25_ranking_and_empty_query(self) -> None:
        documents = (
            TextDocument("one", "image-one", "red apple apple"),
            TextDocument("two", "image-two", "blue car"),
            TextDocument("three", "image-three", "red apple tree"),
        )
        index = BM25Index(k1=1.5, b=0.75).fit(documents)
        results = index.search("red apple", top_k=2)

        self.assertEqual(results[0].item_id, "one")
        self.assertEqual(index.search("", top_k=2), [])
        with self.assertRaises(ValueError):
            BM25Index(b=2.0)

    def test_retrieval_metrics_match_known_binary_relevance_example(self) -> None:
        ranking = ["a", "x", "b"]
        relevant = {"a", "b"}

        self.assertAlmostEqual(precision_at_k(ranking, relevant, 1), 1.0)
        self.assertAlmostEqual(recall_at_k(ranking, relevant, 2), 0.5)
        self.assertAlmostEqual(reciprocal_rank(ranking, relevant), 1.0)
        self.assertAlmostEqual(average_precision(ranking, relevant), (1 + 2 / 3) / 2)
        self.assertAlmostEqual(
            ndcg_at_k(ranking, relevant, 2), 1 / (1 + 1 / 1.584962500721156)
        )
        evaluated = evaluate_rankings({"q": ranking}, {"q": relevant}, ks=(1, 2))
        self.assertEqual(evaluated["queries_evaluated"], 1)
        self.assertAlmostEqual(
            evaluated["mean_reciprocal_rank"]
            if "mean_reciprocal_rank" in evaluated
            else evaluated["mrr"],
            1.0,
        )
        self.assertEqual(
            evaluate_rankings({"q": []}, {"q": set()})["queries_evaluated"], 0
        )

    def test_ppm_colour_descriptor_and_histogram_similarity(self) -> None:
        ppm_red = b"P6\n2 1\n255\n" + bytes((255, 0, 0, 255, 0, 0))
        ppm_blue = b"P6\n2 1\n255\n" + bytes((0, 0, 255, 0, 0, 255))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "red.ppm").write_bytes(ppm_red)
            (root / "blue.ppm").write_bytes(ppm_blue)
            red = colour_histogram_descriptor(root / "red.ppm")
            blue = colour_histogram_descriptor(root / "blue.ppm")
            records = (
                image_record("red", "red.ppm"),
                image_record("blue", "blue.ppm"),
            )
            index = ImageHistogramIndex().fit(records, root)
            results = index.search("red", top_k=1)

        self.assertEqual(len(red), 120)
        self.assertAlmostEqual(histogram_intersection(red, red), 1.0)
        self.assertLess(histogram_intersection(red, blue), 0.5)
        self.assertEqual(index.stats()["items_indexed"], 2)
        self.assertEqual(results[0].item_id, "blue")

    def test_descriptor_rejects_malformed_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ppm"
            path.write_bytes(b"not a ppm")
            with self.assertRaises(ValueError):
                colour_histogram_descriptor(path)

    def test_run_phase3_writes_report_and_respects_no_image_root(self) -> None:
        records = tuple(image_record(str(index)) for index in range(3))
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
            output_dir = root / "phase3"
            write_manifest(manifest, manifest_path)
            report = run_phase3(
                manifest_path,
                config_path=Path("configs/default.toml"),
                output_dir=output_dir,
                max_text_queries=4,
            )

            self.assertTrue((output_dir / "phase3_report.json").exists())
            self.assertTrue((output_dir / "phase3_report.md").exists())
            self.assertTrue((output_dir / "qualitative_examples.json").exists())
            self.assertTrue((output_dir / "baseline_comparison.csv").exists())

            missing_image_report = run_phase3(
                manifest_path,
                config_path=Path("configs/default.toml"),
                output_dir=root / "phase3-missing-images",
                image_root=root / "missing-images",
                max_text_queries=4,
            )

        self.assertTrue(report["scope"]["real_text_baselines"])
        self.assertFalse(report["scope"]["real_image_baseline"])
        self.assertEqual(report["image_baseline"]["status"], "not_run_no_image_root")
        self.assertFalse(missing_image_report["scope"]["real_image_baseline"])
        self.assertEqual(
            missing_image_report["image_baseline"]["status"],
            "not_completed_no_valid_images",
        )
        self.assertEqual(len(report["baselines"]), 2)
        self.assertTrue(report["baselines"][0]["experiment_id"].startswith("phase3_"))
        self.assertEqual(report["baselines"][0]["split"], "test")
        self.assertTrue(report["baselines"][0]["artifact_paths"])


if __name__ == "__main__":
    unittest.main()
