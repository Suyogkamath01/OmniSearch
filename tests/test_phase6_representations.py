import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from omnisearch.representations import (
    TEXT_SPECS,
    VISION_SPECS,
    RepresentationBatch,
    load_text_runtime,
    load_vision_runtime,
    normalize_rows,
    rank_embedding_batches,
    read_embedding_cache,
    select_device,
    write_embedding_cache,
)


class FakeTorch:
    class backends:
        class mps:
            @staticmethod
            def is_available():
                return False

    @staticmethod
    @contextmanager
    def inference_mode():
        yield


class Phase6RepresentationTests(unittest.TestCase):
    def test_device_fallback_and_model_modality_contract(self) -> None:
        self.assertEqual(select_device("auto", FakeTorch()), "cpu")
        self.assertEqual(select_device("cpu", FakeTorch()), "cpu")
        with self.assertRaises(RuntimeError):
            select_device("mps", FakeTorch())
        with self.assertRaises(ValueError):
            load_text_runtime(VISION_SPECS["resnet18_native"])
        with self.assertRaises(ValueError):
            load_vision_runtime(TEXT_SPECS["minilm_mean"])

    def test_normalization_rejects_nonfinite_and_zero_rows(self) -> None:
        self.assertEqual(normalize_rows([[3.0, 4.0]]), ((0.6, 0.8),))
        self.assertEqual(normalize_rows([[3.0, 4.0]], "none"), ((3.0, 4.0),))
        with self.assertRaises(ValueError):
            normalize_rows([[0.0, 0.0]])
        with self.assertRaises(ValueError):
            normalize_rows([[float("inf"), 1.0]])

    def test_batch_shape_and_finiteness_contract(self) -> None:
        self.assertEqual(RepresentationBatch(("a",), ((1.0, 0.0),)).dimension, 2)
        with self.assertRaises(ValueError):
            RepresentationBatch(("a", "a"), ((1.0,), (1.0,)))
        with self.assertRaises(ValueError):
            RepresentationBatch(("a",), ((float("nan"),),))
        with self.assertRaises(ValueError):
            RepresentationBatch(("a", "b"), ((1.0,), (1.0, 0.0)))

    def test_same_space_similarity_is_deterministic_and_cross_space_is_rejected(
        self,
    ) -> None:
        queries = RepresentationBatch(("q",), ((1.0, 0.0),))
        candidates = RepresentationBatch(("b", "a"), ((1.0, 0.0), (0.0, 1.0)))
        ranked = rank_embedding_batches(
            queries, candidates, "text:model", "text:model", top_k=2
        )
        self.assertEqual([item["id"] for item in ranked["q"]], ["b", "a"])
        with self.assertRaises(ValueError):
            rank_embedding_batches(queries, candidates, "text:model", "image:model")

    def test_cache_round_trip_and_stale_metadata_rejection(self) -> None:
        batch = RepresentationBatch(("a", "b"), ((0.6, 0.8), (1.0, 0.0)))
        metadata = {
            "model_key": "fixture",
            "model_id": "fixture/model",
            "manifest_sha256": "manifest",
            "split": "test",
            "pooling": "mean",
            "normalization": "l2",
            "embedding_dimension": 2,
            "device": "cpu",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_embedding_cache(path, batch, metadata)
            self.assertEqual(read_embedding_cache(path, metadata), batch)
            with self.assertRaises(ValueError):
                read_embedding_cache(path, {**metadata, "split": "validation"})
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    '"ids_sha256":', '"ids_sha256_bad":', 1
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_embedding_cache(path, metadata)


if __name__ == "__main__":
    unittest.main()
