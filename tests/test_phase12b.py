from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omnisearch.circo import (
    CircoSchemaError,
    ComposedQuery,
    load_circo_gallery,
    load_circo_queries,
    split_query_ids,
    validate_queries_against_gallery,
)
from omnisearch.evaluation import ranking_from_scores
from omnisearch.phase12b import (
    _circo_metrics,
    run_phase12b,
    validate_phase12b_config,
)


class CircoAdapterTests(unittest.TestCase):
    def test_loads_multiple_ground_truths_and_excludes_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation_path = root / "val.json"
            annotation_path.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "reference_img_id": 10,
                            "relative_caption": "with a blue object",
                            "target_img_id": 20,
                            "gt_img_ids": [20, 30],
                            "semantic_aspects": ["addition"],
                            "shared_concept": "object",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            gallery_path = root / "image_info.json"
            gallery_path.write_text(
                json.dumps({"images": [{"id": item, "file_name": f"{item:012d}.jpg"} for item in (10, 20, 30)]}),
                encoding="utf-8",
            )
            queries = load_circo_queries(annotation_path)
            gallery = load_circo_gallery(gallery_path)
            summary = validate_queries_against_gallery(queries, gallery)
            self.assertEqual(queries[0].ground_truth_image_ids, frozenset({"20", "30"}))
            self.assertEqual(summary["queries_with_multiple_ground_truths"], 1)
            self.assertEqual(gallery.path_for("20", root).name, "000000000020.jpg")

    def test_reference_ground_truth_is_rejected(self) -> None:
        with self.assertRaises(CircoSchemaError):
            ComposedQuery("0", "val", "10", "change", "10", frozenset({"10"}), (), None)

    def test_selection_holdout_is_deterministic_and_disjoint(self) -> None:
        queries = tuple(
            ComposedQuery(str(index), "val", str(index + 10), "change", str(index + 100), frozenset({str(index + 100)}), (), None)
            for index in range(6)
        )
        left, right = split_query_ids(queries, 42, 0.5)
        left_again, right_again = split_query_ids(queries, 42, 0.5)
        self.assertEqual(tuple(item.query_id for item in left), tuple(item.query_id for item in left_again))
        self.assertEqual(tuple(item.query_id for item in right), tuple(item.query_id for item in right_again))
        self.assertTrue({item.query_id for item in left}.isdisjoint(item.query_id for item in right))


class CircoMetricTests(unittest.TestCase):
    def test_official_map_and_target_recall_handle_multiple_targets(self) -> None:
        query = ComposedQuery("0", "val", "10", "change", "20", frozenset({"20", "30"}), (), None)
        ranking = ranking_from_scores(
            query_id="0",
            task="image_to_image",
            candidates=[("20", 1.0), ("40", 0.9), ("30", 0.8)],
            relevant_ids=query.ground_truth_image_ids,
            system_id="fixture",
            experiment_id="fixture",
            candidate_count=3,
            candidate_corpus_id="fixture",
            relevance_definition="CIRCO",
        )
        metrics, _ = _circo_metrics((ranking,), (query,), (3,))
        self.assertAlmostEqual(metrics["map_at_3"], (1.0 + (2.0 / 3.0)) / 2.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["any_ground_truth_recall_at_3"], 1.0)


class Phase12BTests(unittest.TestCase):
    def test_config_requires_metric_coverage_and_sorted_alphas(self) -> None:
        base = {
            "batch_size": 2,
            "top_k": 50,
            "bootstrap_resamples": 10,
            "latency_query_limit": 4,
            "latency_repeats": 1,
            "seed": 42,
            "selection_metric": "map_at_10",
            "selection_fraction": 0.5,
            "alpha_values": [0.25, 0.5, 0.75],
            "metric_ks": [5, 10, 25, 50],
            "phase7_checkpoint": "artifacts/phase7/best_checkpoint.pt",
        }
        validate_phase12b_config(base)
        with self.assertRaises(ValueError):
            validate_phase12b_config({**base, "metric_ks": [5, 10], "top_k": 5})
        with self.assertRaises(ValueError):
            validate_phase12b_config({**base, "alpha_values": [0.5, 0.25]})

    def test_smoke_writes_separate_phase12b_artifacts_and_no_audit_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase12b"
            report = run_phase12b(Path("configs/default.toml"), output, smoke=True)
            self.assertEqual(report["phase"], "Phase 12B — Proper Composed Image Retrieval Evaluation")
            self.assertEqual(report["quality_gate"]["status"], "SMOKE_ONLY")
            self.assertFalse(report["ready_for_phase13"])
            self.assertTrue((output / "phase12b_report.json").exists())
            self.assertTrue((output / "paired_comparisons.json").exists())
            self.assertFalse((output / "phase12b_audit.md").exists())
            self.assertIn("map_at_10", json.loads((output / "evaluation_results.json").read_text())[0]["metrics"])


if __name__ == "__main__":
    unittest.main()
