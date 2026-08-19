import json
import tempfile
import unittest
from pathlib import Path

from omnisearch.evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    average_precision,
    bootstrap_ci,
    build_result,
    compare_systems,
    evaluate_rankings,
    make_protocol,
    ndcg_at_k,
    precision_at_k,
    ranking_from_scores,
    recall_at_k,
    run_phase4_migration,
    run_ranking_file,
    validate_result,
)


def ranking(
    query_id: str,
    order: list[str],
    relevant: set[str],
    system_id: str = "system",
) -> RankingRecord:
    return ranking_from_scores(
        query_id=query_id,
        task="text_to_text",
        candidates=[
            (item_id, float(len(order) - index)) for index, item_id in enumerate(order)
        ],
        relevant_ids=relevant,
        system_id=system_id,
        experiment_id="gold",
        candidate_count=len(order),
        candidate_corpus_id="fixture:test:3",
        relevance_definition="fixture relevance",
    )


class Phase5EvaluationTests(unittest.TestCase):
    def test_protocol_covers_all_canonical_tasks(self) -> None:
        for task in (
            "text_to_image",
            "image_to_text",
            "text_to_text",
            "image_to_image",
        ):
            protocol = make_protocol(task)
            self.assertEqual(protocol["protocol_version"], PROTOCOL_VERSION)
            self.assertEqual(protocol["task"], task)
            self.assertEqual(protocol["k_values"], [1, 5, 10])

        with self.assertRaises(ValueError):
            make_protocol("text_to_text", ks=(1.5,))
        with self.assertRaises(ValueError):
            make_protocol("text_to_text", ks=(True,))

    def test_known_binary_metrics_and_perfect_ranking(self) -> None:
        perfect = ranking("q", ["a", "b", "x"], {"a", "b"})
        self.assertEqual(precision_at_k(perfect, 1), 1.0)
        self.assertAlmostEqual(precision_at_k(perfect, 5), 0.4)
        self.assertEqual(recall_at_k(perfect, 1), 0.5)
        self.assertEqual(recall_at_k(perfect, 5), 1.0)
        self.assertEqual(average_precision(perfect), 1.0)
        self.assertEqual(ndcg_at_k(perfect, 2), 1.0)

        metrics = evaluate_rankings((perfect,), ks=(1, 5))
        self.assertEqual(metrics["queries_total"], 1)
        self.assertEqual(metrics["queries_evaluated"], 1)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["map"], 1.0)
        self.assertEqual(metrics["rank_statistics"]["mean_first_relevant_rank"], 1.0)

    def test_truncated_rank_and_multiple_positive_average_precision(self) -> None:
        rank_five = ranking("q5", ["x1", "x2", "x3", "x4", "a"], {"a"})
        self.assertAlmostEqual(precision_at_k(rank_five, 5), 0.2)
        self.assertEqual(recall_at_k(rank_five, 5), 1.0)
        self.assertAlmostEqual(average_precision(rank_five), 0.2)
        self.assertAlmostEqual(ndcg_at_k(rank_five, 5), 1 / __import__("math").log2(6))

        multiple = ranking("qm", ["a", "x", "b"], {"a", "b"})
        self.assertAlmostEqual(average_precision(multiple), (1 + 2 / 3) / 2)

    def test_no_relevance_is_reported_not_silently_scored(self) -> None:
        no_labels = ranking("q", ["a", "b"], set())
        metrics = evaluate_rankings((no_labels,))
        self.assertEqual(metrics["queries_without_relevance"], 1)
        self.assertIsNone(metrics["mrr"])
        self.assertIsNone(metrics["recall_at_1"])
        no_relevance = bootstrap_ci((no_labels,), "recall", 1)
        self.assertEqual(no_relevance["status"], "not_evaluated_no_relevant_queries")
        self.assertNotIn("estimate", no_relevance)

    def test_ties_are_deterministic_and_malformed_rankings_fail(self) -> None:
        tied = ranking_from_scores(
            "q",
            "text_to_text",
            [("b", 1.0), ("a", 1.0)],
            {"a"},
            "s",
            "e",
            candidate_count=2,
            candidate_corpus_id="c",
        )
        self.assertEqual(tied.candidate_ids, ("a", "b"))
        with self.assertRaises(ValueError):
            RankingRecord.from_mapping(
                {
                    **tied.to_dict(),
                    "candidate_ids": ["b", "a"],
                    "scores": [1.0, 1.0],
                }
            )
        with self.assertRaises(ValueError):
            RankingRecord.from_mapping(
                {
                    **tied.to_dict(),
                    "candidate_ids": ["a", "a"],
                }
            )
        with self.assertRaises(ValueError):
            RankingRecord.from_mapping(
                {
                    **tied.to_dict(),
                    "scores": [float("nan"), 1.0],
                }
            )

    def test_bootstrap_is_query_level_and_seed_reproducible(self) -> None:
        records = (
            ranking("q1", ["a", "x"], {"a"}),
            ranking("q2", ["x", "a"], {"a"}),
            ranking("q3", ["a", "x"], {"a"}),
        )
        first = bootstrap_ci(records, "recall", 1, resamples=100, seed=17)
        second = bootstrap_ci(records, "recall", 1, resamples=100, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["unit"], "query")
        self.assertLessEqual(first["lower"], first["estimate"])
        self.assertLessEqual(first["estimate"], first["upper"])

    def test_paired_comparison_rejects_unsafe_pairs_and_reports_deltas(self) -> None:
        left = (ranking("q1", ["a", "x", "b"], {"a", "b"}, "left"),)
        right = (ranking("q1", ["x", "a", "b"], {"a", "b"}, "right"),)
        metadata = {
            "task": "text_to_text",
            "dataset_id": "fixture",
            "split": "test",
            "protocol_version": PROTOCOL_VERSION,
            "candidate_corpus_id": "fixture:test:3",
            "relevance_definition": "fixture relevance",
        }
        comparison = compare_systems(
            left,
            right,
            {**metadata, "system_id": "left"},
            {**metadata, "system_id": "right"},
            ks=(1,),
            bootstrap_resamples=50,
        )
        self.assertEqual(comparison["status"], "comparable")
        self.assertIn("recall_at_1", comparison["paired_query_deltas"])
        with self.assertRaises(ValueError):
            compare_systems(
                left,
                right,
                {**metadata, "system_id": "left"},
                {**metadata, "system_id": "right", "split": "validation"},
            )
        with self.assertRaises(ValueError):
            compare_systems(
                left,
                (ranking("q2", ["x", "a", "b"], {"a", "b"}, "right"),),
                {**metadata, "system_id": "left"},
                {**metadata, "system_id": "right"},
            )

    def test_result_schema_and_canonical_cli_artifacts(self) -> None:
        record = ranking("q", ["a", "x"], {"a"})
        protocol = make_protocol("text_to_text", ks=(1, 5))
        result = build_result(
            (record,),
            protocol,
            {"dataset_id": "fixture", "dataset_version": "v1", "subset": "gold"},
            "test",
            "gold_experiment",
            "gold_system",
            {"learned": False},
            42,
            {"recall_at_1": bootstrap_ci((record,), "recall", 1, resamples=10)},
            {"seconds": 0.001},
            {"device": "cpu"},
            {"source": "fixture"},
        )
        validate_result(result)
        with self.assertRaises(ValueError):
            validate_result({**result, "project": "Other"})
        with self.assertRaises(ValueError):
            validate_result(
                {**result, "protocol": {**protocol, "protocol_version": "old"}}
            )
        with self.assertRaises(ValueError):
            validate_result(
                {**result, "metrics": {**result["metrics"], "recall_at_1": 1.1}}
            )
        with self.assertRaises(ValueError):
            validate_result(
                {
                    **result,
                    "protocol": {**protocol, "k_values": [True]},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "rankings.json"
            input_path.write_text(
                json.dumps(
                    {
                        "task": "text_to_text",
                        "system_id": "fixture_cli",
                        "experiment_id": "fixture_cli_exp",
                        "dataset": {
                            "dataset_id": "fixture",
                            "dataset_version": "v1",
                            "subset": "gold",
                        },
                        "split": "test",
                        "queries": [record.to_dict()],
                    }
                ),
                encoding="utf-8",
            )
            output = run_ranking_file(
                input_path, root / "output", bootstrap_resamples=10, seed=42
            )
            self.assertEqual(output["protocol"]["protocol_version"], PROTOCOL_VERSION)
            self.assertTrue((root / "output" / "evaluation_result.json").exists())
            self.assertTrue((root / "output" / "protocol.json").exists())

    def test_phase4_fixture_migration_is_explicitly_separated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "phase4_report.json"
            fixture_path = root / "fixture.json"
            report_path.write_text(
                json.dumps({"scope": {"real_flickr30k_evaluation": False}}),
                encoding="utf-8",
            )
            fixture_path.write_text(
                json.dumps(
                    {
                        "fixture_only": True,
                        "model_id": "fixture/clip",
                        "device": "cpu",
                        "text_to_image_rankings": {
                            "red-caption": [
                                {"id": "red-image", "score": 1.0},
                                {"id": "blue-image", "score": 0.0},
                            ],
                        },
                        "image_to_text_rankings": {
                            "red-image": [
                                {"id": "red-caption", "score": 1.0},
                                {"id": "blue-caption", "score": 0.0},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            migration = run_phase4_migration(
                report_path, fixture_path, root / "output", bootstrap_resamples=10
            )
            self.assertEqual(
                migration["phase4_real_flickr30k_status"],
                "not_migrated_no_authorized_image_root",
            )
            self.assertEqual(
                migration["phase4_fixture_status"], "migrated_fixture_only"
            )
            self.assertEqual(
                {item["task"] for item in migration["phase4_fixture_results"]},
                {"text_to_image", "image_to_text"},
            )
            self.assertTrue(
                (root / "output" / "phase4_fixture_text_to_image.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
