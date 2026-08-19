import tempfile
import unittest
from pathlib import Path

import torch

from omnisearch.evaluation import ranking_from_scores
from omnisearch.phase8 import (
    _assert_frozen_contract,
    _merged_equivalence,
    _parameter_summary,
    _retention,
    validate_phase8_config,
)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional Phase 8 dependency
    PEFT_AVAILABLE = False


class TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.q_proj(values) + self.v_proj(values)


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = TinyAttention()
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(
        self,
        values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        actual = values if values is not None else input_ids
        if actual is None:
            actual = inputs_embeds
        if actual is None:
            raise ValueError("a tiny test input is required")
        return self.attention(actual)


def ranking(query: str, order: list[str]) -> object:
    return ranking_from_scores(
        query_id=query,
        task="text_to_image",
        candidates=[(item, float(len(order) - index)) for index, item in enumerate(order)],
        relevant_ids={"positive"},
        system_id="system",
        experiment_id="experiment",
        candidate_count=len(order),
        candidate_corpus_id="fixture:test",
    )


class Phase8Tests(unittest.TestCase):
    def test_config_rejects_unsafe_selection_and_adapter_settings(self) -> None:
        valid = {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "precision": "fp32",
            "selection_metric": "mean_recall_at_5",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj"],
            "lora_bias": "none",
        }
        validate_phase8_config(valid)
        with self.assertRaises(ValueError):
            validate_phase8_config({**valid, "test_split": "validation"})
        with self.assertRaises(ValueError):
            validate_phase8_config({**valid, "lora_rank": 0})
        with self.assertRaises(ValueError):
            validate_phase8_config({**valid, "lora_target_modules": ["all_layers"]})

    def test_retention_refuses_misleading_near_zero_denominator(self) -> None:
        value = _retention(
            {"recall_at_5": 0.5},
            {"recall_at_5": 0.5},
            {"recall_at_5": 0.6},
            ("recall_at_5",),
        )["recall_at_5"]
        self.assertIsNone(value["lora_fraction_of_full_improvement"])
        self.assertIn("not summarized", value["retention_note"])

    def test_merged_equivalence_checks_ids_and_scores(self) -> None:
        left = {"text_to_image": (ranking("q", ["positive", "other"]),), "image_to_text": ()}
        right = {"text_to_image": (ranking("q", ["positive", "other"]),), "image_to_text": ()}
        result = _merged_equivalence(left, right)
        self.assertTrue(result["ranking_ids_equal"])
        self.assertTrue(result["equivalent_within_tolerance"])

    @unittest.skipUnless(PEFT_AVAILABLE, "Phase 8 PEFT dependency is not installed")
    def test_lora_injection_freezes_base_and_supports_save_load_merge(self) -> None:
        base = TinyEncoder()
        before = {
            "q_proj": base.attention.q_proj.weight.detach().clone(),
            "v_proj": base.attention.v_proj.weight.detach().clone(),
            "logit_scale": base.logit_scale.detach().clone(),
        }
        config = LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(base, config)
        _assert_frozen_contract(model)
        summary = _parameter_summary(model)
        self.assertGreater(summary["adapter_trainable_parameters"], 0)
        self.assertLess(summary["trainable_percentage_of_base"], 100.0)
        values = torch.randn(2, 4)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        loss = model(values).pow(2).mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(model.get_base_model().attention.q_proj.base_layer.weight, before["q_proj"]))
        self.assertTrue(torch.equal(model.get_base_model().attention.v_proj.base_layer.weight, before["v_proj"]))
        self.assertTrue(torch.equal(model.get_base_model().logit_scale, before["logit_scale"]))
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            restored = PeftModel.from_pretrained(TinyEncoder(), adapter_dir)
            restored.eval()
            merged = restored.merge_and_unload()
            self.assertTrue(torch.isfinite(merged(values)).all().item())


if __name__ == "__main__":
    unittest.main()
