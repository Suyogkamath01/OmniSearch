import copy
import unittest
from pathlib import Path

import torch

from omnisearch.manifest import CaptionRecord, ImageRecord
from omnisearch.phase7 import TrainingPair
from omnisearch.phase9 import (
    DEFAULT_PHASE9_CONFIG,
    HardNegativeRecord,
    _hard_negative_training_records,
    _mine_from_embeddings,
    _validate_mined_manifest,
    hard_negative_clip_loss,
    validate_phase9_config,
)


def fixture_records() -> tuple[ImageRecord, ...]:
    return tuple(
        ImageRecord(
            image_id=f"image-{index}",
            filename=f"{index}.jpg",
            split="train",
            captions=(
                CaptionRecord(f"caption-{index}-a", text),
                CaptionRecord(f"caption-{index}-b", f"another view of {text}"),
            ),
        )
        for index, text in enumerate(("a red ball", "a blue car", "a green tree"))
    )


class Phase9Tests(unittest.TestCase):
    def test_config_requires_train_only_static_mining(self) -> None:
        validate_phase9_config(DEFAULT_PHASE9_CONFIG)
        with self.assertRaises(ValueError):
            validate_phase9_config({**DEFAULT_PHASE9_CONFIG, "mining_split": "test"})
        with self.assertRaises(ValueError):
            validate_phase9_config({**DEFAULT_PHASE9_CONFIG, "candidate_pool_size": 1})
        with self.assertRaises(ValueError):
            validate_phase9_config({**DEFAULT_PHASE9_CONFIG, "hard_negative_ratio": 0})

    def test_mining_is_deterministic_and_excludes_same_image_and_aliases(self) -> None:
        records = fixture_records()
        pairs = tuple(
            TrainingPair(
                image_id=record.image_id,
                caption_id=record.captions[0].caption_id,
                text=record.captions[0].text,
                image_path=record.filename or "",
            )
            for record in records
        )
        image_embeddings = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=torch.float32
        )
        text_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        first, first_stats = _mine_from_embeddings(
            pairs,
            records,
            [record.image_id for record in records],
            image_embeddings,
            [caption.caption_id for record in records for caption in record.captions],
            text_embeddings,
            seed=42,
            candidate_pool_size=2,
            strategy="static_frozen_clip_top5_hash_sample",
        )
        second, second_stats = _mine_from_embeddings(
            pairs,
            records,
            [record.image_id for record in records],
            image_embeddings,
            [caption.caption_id for record in records for caption in record.captions],
            text_embeddings,
            seed=42,
            candidate_pool_size=2,
            strategy="static_frozen_clip_top5_hash_sample",
        )
        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)
        for item in first:
            self.assertNotEqual(item.image_id, item.negative_image_id)
            negative_caption_image = next(
                record.image_id
                for record in records
                if any(caption.caption_id == item.negative_caption_id for caption in record.captions)
            )
            self.assertNotEqual(item.image_id, negative_caption_image)
            self.assertNotIn(item.negative_image_id, {item.image_id})
            self.assertEqual(len(item.negative_image_pool_ids), 2)
            self.assertEqual(len(item.negative_caption_pool_ids), 2)

    def test_mined_manifest_rejects_stale_or_same_image_rows(self) -> None:
        record = HardNegativeRecord(
            image_id="image-1",
            positive_caption_id="caption-1",
            positive_caption_text="positive",
            negative_image_id="image-2",
            negative_image_score=0.8,
            negative_image_rank=1,
            negative_image_pool_ids=("image-2", "image-3"),
            negative_caption_id="caption-2",
            negative_caption_text="negative",
            negative_caption_score=0.7,
            negative_caption_rank=1,
            negative_caption_pool_ids=("caption-2", "caption-3"),
            mining_strategy="static_frozen_clip_top5_hash_sample",
        )
        payload = {
            "phase9_schema_version": 1,
            "manifest_sha256": "manifest",
            "mining_split": "train",
            "mining_strategy": "static_frozen_clip_top5_hash_sample",
            "records": [record.to_dict()],
        }
        self.assertEqual(len(_validate_mined_manifest(payload, "manifest")), 1)
        with self.assertRaises(ValueError):
            _validate_mined_manifest(payload, "stale")
        same_image = copy.deepcopy(payload)
        same_image["records"][0]["negative_image_id"] = "image-1"
        with self.assertRaises(ValueError):
            _validate_mined_manifest(same_image, "manifest")

    def test_hard_negative_loss_preserves_diagonal_targets_and_gradients(self) -> None:
        image = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        text = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        hard_image = torch.tensor([[0.8, 0.2]], requires_grad=True)
        hard_text = torch.tensor([[0.2, 0.8]], requires_grad=True)
        logit_scale = torch.tensor(1.0, requires_grad=True)
        loss, details = hard_negative_clip_loss(
            image, text, hard_image, hard_text, logit_scale
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(details["hard_negative_count"], 1.0)
        before = image.detach().clone()
        optimizer = torch.optim.SGD(
            [image, text, hard_image, hard_text, logit_scale], lr=0.1
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in optimizer.param_groups[0]["params"]))
        optimizer.step()
        self.assertFalse(torch.equal(before, image.detach()))

    def test_mixed_ratio_is_exact_even_when_subset_hashes_are_skewed(self) -> None:
        records = fixture_records()
        pairs = tuple(
            TrainingPair(
                image_id=record.image_id,
                caption_id=record.captions[0].caption_id,
                text=record.captions[0].text,
                image_path=record.filename or "",
            )
            for record in records
        )
        mined = tuple(
            HardNegativeRecord(
                image_id=record.image_id,
                positive_caption_id=record.captions[0].caption_id,
                positive_caption_text=record.captions[0].text,
                negative_image_id=records[(index + 1) % len(records)].image_id,
                negative_image_score=0.5,
                negative_image_rank=1,
                negative_image_pool_ids=(records[(index + 1) % len(records)].image_id, records[(index + 2) % len(records)].image_id),
                negative_caption_id=records[(index + 1) % len(records)].captions[0].caption_id,
                negative_caption_text=records[(index + 1) % len(records)].captions[0].text,
                negative_caption_score=0.5,
                negative_caption_rank=1,
                negative_caption_pool_ids=(records[(index + 1) % len(records)].captions[0].caption_id, records[(index + 2) % len(records)].captions[0].caption_id),
                mining_strategy="static_frozen_clip_top5_hash_sample",
            )
            for index, record in enumerate(records)
        )
        selected = _hard_negative_training_records(
            pairs,
            mined,
            image_root=Path("."),
            records_by_image={record.image_id: record for record in records},
            ratio=0.5,
            seed=42,
            epoch=0,
        )
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
