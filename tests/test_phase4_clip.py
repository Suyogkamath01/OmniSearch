import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from omnisearch.clip_baseline import (
    ClipRuntime,
    EmbeddingBatch,
    encode_texts,
    exact_rank,
    normalize_embeddings,
    rank_metrics,
    read_embedding_cache,
    select_device,
    write_embedding_cache,
)


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


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


class FakeProcessor:
    def __call__(self, **kwargs):
        values = kwargs.get("text") or kwargs.get("images")
        return {"values": values}


class FakeModel:
    def _features(self, values):
        return FakeTensor(
            [
                [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]
                for index, _ in enumerate(values)
            ]
        )

    def get_text_features(self, **inputs):
        return self._features(inputs["values"])


def runtime() -> ClipRuntime:
    return ClipRuntime(
        FakeModel(), FakeProcessor(), FakeTorch(), "cpu", "fixture/clip", 77, 2
    )


class Phase4ClipTests(unittest.TestCase):
    def test_device_falls_back_to_cpu_without_mps(self) -> None:
        self.assertEqual(select_device("auto", FakeTorch()), "cpu")
        self.assertEqual(select_device("cpu", FakeTorch()), "cpu")
        with self.assertRaises(RuntimeError):
            select_device("mps", FakeTorch())

    def test_normalization_rejects_invalid_vectors(self) -> None:
        vectors = normalize_embeddings([[3.0, 4.0]])
        self.assertEqual(vectors, ((0.6, 0.8),))
        with self.assertRaises(ValueError):
            normalize_embeddings([[0.0, 0.0]])
        with self.assertRaises(ValueError):
            normalize_embeddings([[float("nan"), 1.0]])

    def test_fake_text_embedding_shape_order_and_finiteness(self) -> None:
        batch = encode_texts(
            [("a", "red square"), ("b", "blue square"), ("empty", "")],
            runtime(),
            batch_size=1,
        )
        self.assertEqual(batch.ids, ("a", "b"))
        self.assertEqual(batch.dimension, 2)
        self.assertEqual(batch.skipped[0]["reason"], "empty_text")
        self.assertEqual(batch.embeddings[0], (1.0, 0.0))

    def test_exact_normalized_similarity_and_deterministic_ranking(self) -> None:
        candidates = EmbeddingBatch(("x", "y"), ((1.0, 0.0), (0.0, 1.0)))
        first = exact_rank((1.0, 0.0), candidates, top_k=2)
        second = exact_rank((1.0, 0.0), candidates, top_k=2)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["id"], "x")
        self.assertAlmostEqual(first[0]["score"], 1.0)

    def test_multiple_caption_relevance_metrics(self) -> None:
        rankings = {
            "image-1": [{"id": "cap-2", "score": 0.9}, {"id": "cap-1", "score": 0.8}],
            "image-2": [{"id": "cap-3", "score": 0.9}, {"id": "other", "score": 0.8}],
        }
        relevance = {"image-1": {"cap-1", "cap-2"}, "image-2": {"cap-4", "cap-5"}}
        metrics = rank_metrics(rankings, relevance, ks=(1, 2))
        self.assertEqual(metrics["queries_evaluated"], 2)
        self.assertAlmostEqual(metrics["recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_2"], 0.5)

    def test_cache_round_trip_and_stale_metadata_detection(self) -> None:
        batch = EmbeddingBatch(("a",), ((1.0, 0.0),))
        metadata = {
            "model_id": "fixture/clip",
            "manifest_sha256": "manifest",
            "split": "test",
            "embedding_dimension": 2,
            "normalization": "l2",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_embedding_cache(path, batch, metadata)
            restored = read_embedding_cache(path, metadata)
            self.assertEqual(restored, batch)
            with self.assertRaises(ValueError):
                read_embedding_cache(path, {**metadata, "split": "train"})


if __name__ == "__main__":
    unittest.main()
