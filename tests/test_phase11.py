from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from omnisearch.evaluation import evaluate_rankings, ranking_from_scores
from omnisearch.phase11 import (
    PairwiseMLPReranker,
    _candidate_recall,
    _fixture_manifest_and_arrays,
    _oracle_rankings,
    _rerank_rows,
    run_phase11,
    validate_phase11_config,
)


class Phase11UnitTests(unittest.TestCase):
    def test_config_rejects_invalid_depths_and_later_checkpoint(self) -> None:
        base = {
            "batch_size": 2,
            "epochs": 1,
            "hidden_dim": 8,
            "latency_query_limit": 4,
            "latency_repeats": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "margin": 0.1,
            "candidate_depths": [10, 25],
            "selection_metric": "mean_mrr",
            "bootstrap_resamples": 10,
            "max_train_images": 4,
            "tier_sizes": [10],
            "phase7_checkpoint": "artifacts/phase7/best_checkpoint.pt",
        }
        validate_phase11_config(base)
        with self.assertRaises(ValueError):
            validate_phase11_config({**base, "candidate_depths": [25, 10]})
        with self.assertRaises(ValueError):
            validate_phase11_config({**base, "phase7_checkpoint": "artifacts/phase8/best.pt"})

    def test_pair_features_are_explicit_finite_interactions(self) -> None:
        model = PairwiseMLPReranker(dimension=4, hidden_dim=8)
        query = torch.eye(4, dtype=torch.float32)
        candidate = torch.flip(query, dims=(0,))
        features = model.pair_features(query, candidate)
        self.assertEqual(tuple(features.shape), (4, 9))
        self.assertTrue(bool(torch.isfinite(features).all().item()))
        scores = model(query, candidate)
        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(bool(torch.isfinite(scores).all().item()))

    def test_candidate_recall_and_oracle_are_bounded(self) -> None:
        rows: dict[str, Any] = {
            "query_ids": ("q1", "q2"),
            "candidate_rows": [["positive", "other"], ["other", "positive"]],
            "score_rows": [[0.9, 0.8], [0.9, 0.8]],
            "relevant": {"q1": {"positive"}, "q2": {"positive"}},
            "candidate_count": 2,
            "candidate_corpus_id": "fixture-corpus",
        }
        recall = _candidate_recall(rows, 1)
        self.assertEqual(recall["candidate_hit_rate"], 0.5)
        oracle = _oracle_rankings("text_to_image", rows, 2, "oracle")
        stage1 = evaluate_rankings(
            tuple(
                ranking_from_scores(
                    query_id=query_id,
                    task="text_to_image",
                    candidates=list(zip(ids, scores)),
                    relevant_ids=rows["relevant"][query_id],
                    system_id="stage1",
                    experiment_id="fixture",
                    candidate_count=2,
                    candidate_corpus_id="fixture-corpus",
                )
                for query_id, ids, scores in zip(rows["query_ids"], rows["candidate_rows"], rows["score_rows"])
            )
        )
        oracle_metrics = evaluate_rankings(oracle)
        self.assertGreaterEqual(oracle_metrics["recall_at_1"], stage1["recall_at_1"])
        self.assertEqual(oracle[0].system_id, "oracle_analysis_not_a_model")

    def test_reranking_is_deterministic_and_preserves_candidate_set(self) -> None:
        torch.manual_seed(42)
        model = PairwiseMLPReranker(dimension=3, hidden_dim=8)
        rows: dict[str, Any] = {
            "query_ids": ("q1",),
            "candidate_rows": [["a", "b", "c"]],
            "score_rows": [[0.9, 0.8, 0.7]],
            "relevant": {"q1": {"b"}},
            "candidate_count": 3,
            "candidate_corpus_id": "fixture-corpus",
        }
        vectors = {
            "a": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "b": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "c": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        }
        query = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
        first = _rerank_rows(model, rows, query, vectors, 2, torch.device("cpu"))
        second = _rerank_rows(model, rows, query, vectors, 2, torch.device("cpu"))
        self.assertEqual(first["candidate_rows"], second["candidate_rows"])
        self.assertEqual(set(first["candidate_rows"][0]), {"a", "b"})
        self.assertEqual(len(first["score_rows"][0]), 2)


class Phase11SmokeTests(unittest.TestCase):
    def test_smoke_writes_gate_and_split_isolated_artifacts(self) -> None:
        # FAISS and PyTorch can load separate OpenMP runtimes on macOS CI hosts.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase11"
            report = run_phase11(Path("configs/default.toml"), output, smoke=True)
            self.assertEqual(report["quality_gate"]["status"], "SMOKE_ONLY")
            self.assertTrue(report["quality_gate"]["reranker_train_only"])
            self.assertTrue(report["quality_gate"]["test_isolation"])
            self.assertTrue(report["quality_gate"]["paired_statistics"])
            self.assertTrue((output / "phase11_report.json").exists())
            self.assertTrue((output / "reranker_checkpoint.pt").exists())
            self.assertFalse((output / "phase11_audit.md").exists())
            index_manifest = json.loads((output / "index_manifest.json").read_text())
            self.assertTrue(any(item["split"] == "train" for item in index_manifest))
            self.assertTrue(all(item["tier"] == "training" or item["split"] in {"validation", "test"} for item in index_manifest))
            _, arrays = _fixture_manifest_and_arrays()
            self.assertEqual(arrays["images"].shape[1], report["reranker"]["dimension"])


if __name__ == "__main__":
    unittest.main()
