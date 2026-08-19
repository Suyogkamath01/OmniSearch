import tempfile
import unittest
from pathlib import Path

import torch

from omnisearch.manifest import CaptionRecord, ImageRecord
from omnisearch.phase7 import (
    _load_checkpoint,
    _save_checkpoint,
    build_training_pairs,
    contrastive_targets,
    symmetric_clip_loss,
    validate_phase7_config,
)


def grouped_records() -> tuple[ImageRecord, ...]:
    return tuple(
        ImageRecord(
            image_id,
            f"{image_id}.jpg",
            (
                CaptionRecord(f"{image_id}#0", f"caption zero {image_id}"),
                CaptionRecord(f"{image_id}#1", f"caption one {image_id}"),
            ),
            split="train",
        )
        for image_id in ("image-a", "image-b", "image-c")
    )


class Phase7TrainingTests(unittest.TestCase):
    def test_symmetric_loss_is_finite_and_backpropagates(self) -> None:
        image = torch.randn(3, 4, requires_grad=True)
        text = torch.randn(3, 4, requires_grad=True)
        logit_scale = torch.tensor(2.0, requires_grad=True)

        loss, details = symmetric_clip_loss(image, text, logit_scale)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(details["image_to_text_loss"], 0.0)
        self.assertGreater(details["text_to_image_loss"], 0.0)
        self.assertIsNotNone(image.grad)
        self.assertIsNotNone(text.grad)
        self.assertIsNotNone(logit_scale.grad)
        self.assertTrue(torch.isfinite(image.grad).all().item())

    def test_contrastive_targets_are_diagonal(self) -> None:
        self.assertTrue(
            torch.equal(contrastive_targets(4, torch.device("cpu")), torch.arange(4))
        )
        with self.assertRaises(ValueError):
            contrastive_targets(0, torch.device("cpu"))

    def test_training_pairs_are_one_caption_per_image_and_reproducible(self) -> None:
        first = build_training_pairs(grouped_records(), Path("images"), 42, epoch=0)
        second = build_training_pairs(grouped_records(), Path("images"), 42, epoch=0)
        next_epoch = build_training_pairs(grouped_records(), Path("images"), 42, epoch=1)

        self.assertEqual(first, second)
        self.assertEqual({pair.image_id for pair in first}, {"image-a", "image-b", "image-c"})
        self.assertEqual(len(first), len({pair.image_id for pair in first}))
        self.assertEqual(len(next_epoch), len(first))
        self.assertTrue(all(pair.image_path.startswith("images/") for pair in first))

    def test_config_rejects_split_boundary_and_invalid_batch(self) -> None:
        valid = {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "learning_rate": 1e-6,
            "weight_decay": 0.01,
            "precision": "fp32",
            "selection_metric": "mean_recall_at_5",
        }
        validate_phase7_config(valid)
        with self.assertRaises(ValueError):
            validate_phase7_config({**valid, "test_split": "validation"})
        with self.assertRaises(ValueError):
            validate_phase7_config({**valid, "batch_size": 0})

    def test_checkpoint_round_trip_restores_weights_and_metadata(self) -> None:
        model = torch.nn.Linear(3, 2)
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save_checkpoint(path, model, {"epoch": 3, "selected": True})
            for parameter in model.parameters():
                parameter.data.zero_()
            metadata = _load_checkpoint(path, model)

        self.assertEqual(metadata, {"epoch": 3, "selected": True})
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))


if __name__ == "__main__":
    unittest.main()
