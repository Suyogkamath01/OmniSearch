from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from omnisearch.phase12 import (
    _score_rows,
    _weighted_fusion,
    run_phase12,
    validate_phase12_config,
)


class Phase12FusionTests(unittest.TestCase):
    def test_alpha_validation_and_bounds(self) -> None:
        base = {
            "batch_size": 2,
            "bootstrap_resamples": 10,
            "top_k": 5,
            "latency_query_limit": 4,
            "latency_repeats": 1,
            "warmup_queries": 1,
            "qualitative_examples": 2,
            "seed": 42,
            "selection_metric": "mean_mrr",
            "alpha_values": [0.25, 0.5, 0.75],
            "tier_sizes": [10],
            "phase7_checkpoint": "artifacts/phase7/best_checkpoint.pt",
        }
        validate_phase12_config(base)
        with self.assertRaises(ValueError):
            validate_phase12_config({**base, "alpha_values": [0.5, 0.25]})
        with self.assertRaises(ValueError):
            validate_phase12_config({**base, "alpha_values": [0.0, 0.5]})

    def test_weighted_fusion_normalizes_and_has_modality_limits(self) -> None:
        image = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        text = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
        image_only = _weighted_fusion(image, text, 1.0)
        text_only = _weighted_fusion(image, text, 0.0)
        fused = _weighted_fusion(image, text, 0.5)
        self.assertTrue(np.allclose(image_only, image))
        self.assertTrue(np.allclose(text_only, text))
        self.assertTrue(np.allclose(np.linalg.norm(fused, axis=1), 1.0))
        with self.assertRaises(ValueError):
            _weighted_fusion(image, np.ones((1, 4), dtype=np.float32), 0.5)
        with self.assertRaises((ValueError, FloatingPointError)):
            _weighted_fusion(np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32), text, 0.5)

    def test_late_score_rows_are_deterministic(self) -> None:
        ids, scores = _score_rows(
            np.asarray([[0.5, 0.5, 0.2]], dtype=np.float32),
            ("b", "a", "c"),
            3,
        )
        self.assertEqual(ids, [["a", "b", "c"]])
        self.assertTrue(np.allclose(scores, [[0.5, 0.5, 0.2]]))


class Phase12SmokeTests(unittest.TestCase):
    def test_smoke_writes_controls_and_gate(self) -> None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase12"
            report = run_phase12(Path("configs/default.toml"), output, smoke=True)
            gate = report["quality_gate"]
            self.assertEqual(gate["status"], "SMOKE_ONLY")
            self.assertTrue(gate["image_only_control"])
            self.assertTrue(gate["text_only_control"])
            self.assertTrue(gate["early_fusion"])
            self.assertTrue(gate["late_fusion"])
            self.assertTrue(gate["no_fabricated_ground_truth"])
            self.assertTrue((output / "phase12_report.json").exists())
            self.assertTrue((output / "paired_comparisons.json").exists())
            self.assertFalse((output / "phase12_audit.md").exists())
            results = json.loads((output / "test_results.json").read_text())
            variants = {item["variant_id"] for item in results}
            self.assertIn("image_only", variants)
            self.assertIn("text_only", variants)
            self.assertTrue(any(item["method"] == "early" for item in results))
            self.assertTrue(any(item["method"] == "late" for item in results))


if __name__ == "__main__":
    unittest.main()
