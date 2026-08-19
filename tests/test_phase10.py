from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from omnisearch.phase10 import (
    ExactIndex,
    build_persisted_index,
    load_persisted_index,
    normalize_vectors,
    run_phase10,
    validate_normalized_vectors,
)


class Phase10VectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors = normalize_vectors(
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.8, 0.6, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.7, 0.7, 0.0, 0.0],
                    [0.6, 0.8, 0.0, 0.0],
                    [0.5, 0.5, 0.5, 0.5],
                    [-0.5, -0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            )
        )
        self.ids = tuple(f"item-{index:02d}" for index in range(len(self.vectors)))
        self.source = {
            "model_id": "test-model",
            "checkpoint_sha256": "checkpoint-hash",
            "manifest_sha256": "manifest-hash",
            "protocol_version": "retrieval_eval_v1",
        }

    def test_exact_search_is_deterministic_and_clamps_k(self) -> None:
        index = ExactIndex(self.vectors, self.ids)
        ids, scores = index.search(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), 100)
        self.assertEqual(len(ids[0]), len(self.ids))
        self.assertEqual(ids[0][0], "item-00")
        self.assertEqual(ids[0][1], "item-02")
        self.assertTrue(np.all(np.diff(scores[0].astype(np.float32)) <= 1e-6))

    def test_normalized_similarity_contract_rejects_bad_vectors(self) -> None:
        with self.assertRaises(ValueError):
            validate_normalized_vectors(np.zeros((2, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            normalize_vectors(np.asarray([[1.0, np.nan, 0.0, 0.0]], dtype=np.float32))
        with self.assertRaises(ValueError):
            ExactIndex(np.empty((0, 4), dtype=np.float32), ())

    def test_metadata_save_load_and_stale_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "flat"
            built = build_persisted_index(
                self.vectors,
                self.ids,
                "exact_numpy",
                {},
                self.source,
                "tier-test",
                "image_group",
                base,
                42,
            )
            loaded = load_persisted_index(
                built,
                self.ids,
                {
                    "dataset_manifest_sha256": "manifest-hash",
                    "tier": "tier-test",
                    "candidate_unit": "image_group",
                    "embedding_dimension": 4,
                    "candidate_count": len(self.ids),
                },
            )
            original_ids, original_scores = built.index.search(self.vectors[:2], 5)
            loaded_ids, loaded_scores = loaded.search(self.vectors[:2], 5)
            self.assertEqual(original_ids.tolist(), loaded_ids.tolist())
            self.assertTrue(np.allclose(np.asarray(original_scores, dtype=np.float32), np.asarray(loaded_scores, dtype=np.float32)))
            with self.assertRaises(ValueError):
                load_persisted_index(
                    built,
                    self.ids,
                    {
                        "dataset_manifest_sha256": "stale-manifest",
                        "tier": "tier-test",
                        "candidate_unit": "image_group",
                        "embedding_dimension": 4,
                        "candidate_count": len(self.ids),
                    },
                )

    @unittest.skipUnless(importlib.util.find_spec("faiss") is not None, "FAISS optional dependency is unavailable")
    def test_faiss_flat_matches_exact_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "faiss-flat"
            built = build_persisted_index(
                self.vectors,
                self.ids,
                "faiss_flat",
                {},
                self.source,
                "tier-test",
                "image_group",
                base,
                42,
            )
            loaded = load_persisted_index(
                built,
                self.ids,
                {
                    "dataset_manifest_sha256": "manifest-hash",
                    "tier": "tier-test",
                    "candidate_unit": "image_group",
                    "embedding_dimension": 4,
                    "candidate_count": len(self.ids),
                },
            )
            exact_ids, _ = ExactIndex(self.vectors, self.ids).search(self.vectors[:3], 5)
            faiss_ids, _ = loaded.search(self.vectors[:3], 5)
            self.assertEqual(exact_ids.tolist(), faiss_ids.tolist())

    @unittest.skipUnless(importlib.util.find_spec("hnswlib") is not None, "hnswlib optional dependency is unavailable")
    def test_hnsw_save_load_returns_valid_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "hnsw"
            built = build_persisted_index(
                self.vectors,
                self.ids,
                "hnswlib",
                {"M": 8, "ef_construction": 40, "ef_search": 20},
                self.source,
                "tier-test",
                "image_group",
                base,
                42,
            )
            loaded = load_persisted_index(
                built,
                self.ids,
                {
                    "dataset_manifest_sha256": "manifest-hash",
                    "tier": "tier-test",
                    "candidate_unit": "image_group",
                    "embedding_dimension": 4,
                    "candidate_count": len(self.ids),
                },
            )
            ids, scores = loaded.search(self.vectors[:2], 5)
            self.assertEqual(ids.shape, (2, 5))
            self.assertEqual(scores.shape, (2, 5))
            self.assertEqual(ids[0][0], "item-00")

    def test_smoke_benchmark_writes_schema_and_uses_both_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = run_phase10(
                Path("configs/default.toml"), Path(temporary) / "phase10", smoke=True
            )
            self.assertEqual(report["quality_gate"]["status"], "SMOKE_ONLY")
            self.assertTrue(report["quality_gate"]["faiss_actually_ran"])
            self.assertTrue(report["quality_gate"]["hnsw_actually_ran"])
            self.assertTrue(report["quality_gate"]["index_save_load_validated"])
            self.assertEqual(report["pre_phase_audit"], "Phase 9 PASS")
            self.assertTrue((Path(temporary) / "phase10" / "phase10_report.json").exists())
            json.loads((Path(temporary) / "phase10" / "phase10_report.json").read_text())


if __name__ == "__main__":
    unittest.main()
