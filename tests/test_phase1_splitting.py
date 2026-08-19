import unittest

from omnisearch.manifest import CaptionRecord, ImageRecord
from omnisearch.splitting import (
    assert_no_split_leakage,
    assign_image_grouped_splits,
    select_tier,
    split_counts,
)


def records(count: int) -> tuple[ImageRecord, ...]:
    return tuple(
        ImageRecord(
            str(index),
            f"{index}.jpg",
            tuple(
                CaptionRecord(f"{index}#{caption}", f"caption {caption}")
                for caption in range(5)
            ),
        )
        for index in range(count)
    )


class SplittingTests(unittest.TestCase):
    def test_split_is_deterministic_and_image_grouped(self) -> None:
        source = records(20)
        first = assign_image_grouped_splits(source, seed=42)
        second = assign_image_grouped_splits(source, seed=42)
        self.assertEqual(first, second)
        assert_no_split_leakage(first)
        self.assertEqual(
            sum(value["images"] for value in split_counts(first).values()), 20
        )
        self.assertEqual(
            sum(value["captions"] for value in split_counts(first).values()), 100
        )

    def test_leakage_check_rejects_image_crossing_splits(self) -> None:
        source = records(1)[0]
        with self.assertRaisesRegex(ValueError, "split leakage"):
            assert_no_split_leakage(
                (source.with_split("train"), source.with_split("test"))
            )

    def test_tier_selection_keeps_whole_image_records(self) -> None:
        split = assign_image_grouped_splits(records(120), seed=42)
        tier = select_tier(split, "tier1", seed=42)
        self.assertEqual(len(tier), 100)
        self.assertTrue(all(len(item.captions) == 5 for item in tier))
        self.assertEqual(select_tier(split, "tier1", seed=42), tier)

    def test_tier_selection_excludes_unresolved_source_records_by_default(self) -> None:
        source = list(records(4))
        source[-1] = ImageRecord(
            "missing-source-image-000001",
            None,
            source[-1].captions,
            source_image_id_available=False,
        )
        selected = select_tier(source, "tier3", seed=42)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(record.source_image_id_available for record in selected))
        self.assertEqual(
            len(select_tier(source, "tier3", seed=42, include_unresolved=True)), 4
        )
