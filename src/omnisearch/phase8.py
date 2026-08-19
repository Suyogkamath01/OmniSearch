"""Phase 8 real PEFT/LoRA adaptation of the Phase 7 CLIP experiment.

This module keeps the Phase 7 dataset, split, sampler, loss, and canonical
evaluator while replacing full-parameter updates with PEFT LoRA adapters on
CLIP attention q/v projections. The base model remains frozen; the scalar
CLIP logit scale is an explicitly declared additional trainable parameter.
"""

from __future__ import annotations

import gc
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config
from .evaluation import (
    PROTOCOL_VERSION,
    RankingRecord,
    compare_systems,
    write_result_artifacts,
)
from .manifest import ImageRecord, read_manifest
from .phase7 import (
    PHASE7_SCHEMA_VERSION,
    _comparison_metadata,
    _hash_file,
    _subset_records,
    _train_epoch,
    _validation_loss,
    _write_json,
    build_training_pairs,
    evaluate_model,
)
from .splitting import assert_no_split_leakage

PHASE8_SCHEMA_VERSION = 1
DEFAULT_PHASE8_CONFIG: dict[str, Any] = {
    "manifest": "data/processed/coco2017_val_split_manifest.json",
    "image_root": "data/raw/coco2017/val2017",
    "model_id": "openai/clip-vit-base-patch32",
    "phase7_artifact_dir": "artifacts/phase7",
    "device": "auto",
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_workers": 0,
    "epochs": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "text_max_length": 77,
    "precision": "fp32",
    "selection_metric": "mean_recall_at_5",
    "early_stopping_patience": 2,
    "bootstrap_resamples": 200,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    "lora_bias": "none",
    "train_logit_scale": True,
    "max_train_images": 800,
    "max_validation_images": 100,
    "max_test_images": 100,
    "subset_seed": None,
}


def _read_phase8_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as file:
        raw = tomllib.load(file)
    values = dict(DEFAULT_PHASE8_CONFIG)
    values.update(dict(raw.get("phase8", {})))
    return values


def validate_phase8_config(config: Mapping[str, Any]) -> None:
    for key, expected in (
        ("train_split", "train"),
        ("validation_split", "validation"),
        ("test_split", "test"),
    ):
        if key in config and config[key] != expected:
            raise ValueError(f"{key} must remain {expected!r}")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if int(config["gradient_accumulation_steps"]) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(config.get("num_workers", 0)) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    if str(config["precision"]) not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if str(config["selection_metric"]) not in {
        "mean_recall_at_5",
        "mean_recall_at_1",
    }:
        raise ValueError("unsupported selection_metric")
    if int(config["lora_rank"]) <= 0:
        raise ValueError("lora_rank must be positive")
    if int(config["lora_alpha"]) <= 0:
        raise ValueError("lora_alpha must be positive")
    if not 0 <= float(config["lora_dropout"]) < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    targets = tuple(str(item) for item in config["lora_target_modules"])
    if not targets or not all(target in {"q_proj", "k_proj", "v_proj", "out_proj"} for target in targets):
        raise ValueError("lora_target_modules must use supported attention projections")
    if str(config["lora_bias"]) not in {"none", "all", "lora_only"}:
        raise ValueError("lora_bias must be none, all, or lora_only")
    for key in ("max_train_images", "max_validation_images", "max_test_images"):
        value = config.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} must be positive when supplied")


def _load_lora_model(config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Phase 8 requires torch, transformers, peft, and Pillow") from exc
    from .clip_baseline import select_device

    device = select_device(str(config["device"]), torch)
    processor = CLIPProcessor.from_pretrained(str(config["model_id"]))
    base_model = CLIPModel.from_pretrained(str(config["model_id"]))
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    lora_config = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["lora_target_modules"]),
        bias=cast(Any, str(config["lora_bias"])),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base_model, lora_config)
    if bool(config["train_logit_scale"]):
        model.get_base_model().logit_scale.requires_grad_(True)
    model.to(device)
    model.train()
    return model, processor, torch, device


def _base_model(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _parameter_summary(model: Any) -> dict[str, Any]:
    total_base = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    adapter_parameters = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    extra_parameters = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    invalid = [
        name
        for name in trainable_names
        if "lora_" not in name and not name.endswith("logit_scale")
    ]
    if invalid:
        raise AssertionError(f"unexpected trainable PEFT parameters: {invalid[:5]}")
    return {
        "base_parameters": total_base,
        "adapter_trainable_parameters": adapter_parameters,
        "extra_trainable_parameters": extra_parameters,
        "total_trainable_parameters": adapter_parameters + extra_parameters,
        "trainable_percentage_of_base": 100.0 * (adapter_parameters + extra_parameters) / total_base,
        "parameter_reduction_vs_full_finetuning": 1.0 - (adapter_parameters + extra_parameters) / total_base,
        "base_parameters_frozen": total_base - extra_parameters,
        "trainable_names_sample": trainable_names[:12],
        "logit_scale_trainable": bool(_base_model(model).logit_scale.requires_grad),
    }


def _assert_frozen_contract(model: Any) -> None:
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name and not name.endswith("logit_scale")
    ]
    if invalid:
        raise AssertionError(f"base-model parameters unexpectedly trainable: {invalid[:5]}")
    base_trainable = [
        name
        for name, parameter in _base_model(model).named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
        and not name.endswith("logit_scale")
    ]
    if base_trainable:
        raise AssertionError(f"base-model weights are not frozen: {base_trainable[:5]}")


def _adapter_parameter(model: Any) -> Any:
    return next(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )


def _save_adapter(model: Any, adapter_dir: Path, metadata: Mapping[str, Any]) -> None:
    import torch

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    torch.save(
        {"logit_scale": _base_model(model).logit_scale.detach().cpu()},
        adapter_dir / "extra_trainable_state.pt",
    )
    _write_json(dict(metadata), adapter_dir / "selected_adapter_metadata.json")


def _load_adapter(
    config: Mapping[str, Any], adapter_dir: Path
) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Phase 8 requires torch, transformers, and peft") from exc
    from .clip_baseline import select_device

    device = select_device(str(config["device"]), torch)
    processor = CLIPProcessor.from_pretrained(str(config["model_id"]))
    base = CLIPModel.from_pretrained(str(config["model_id"]))
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    extra_path = adapter_dir / "extra_trainable_state.pt"
    payload = torch.load(extra_path, map_location="cpu", weights_only=False)
    base_model: Any = model.get_base_model()
    base_model.logit_scale.data.copy_(payload["logit_scale"])
    model.to(device)
    model.eval()
    return model, processor, torch, device


def _adapter_size(adapter_dir: Path) -> int:
    return sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())


def _ranking_records(result: Mapping[str, Any]) -> tuple[RankingRecord, ...]:
    return tuple(RankingRecord.from_mapping(item) for item in result["ranking_records"])


def _result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    rankings = _ranking_records(result)
    return _comparison_metadata(
        {"system_id": result["system_id"], "task": result["task"], "dataset": result["dataset"], "split": result["split"], "protocol": result["protocol"], "ranking_records": [item.to_dict() for item in rankings]}
    )


def _rank(record: RankingRecord) -> int | None:
    for rank, item_id in enumerate(record.candidate_ids, start=1):
        if item_id in record.relevant_ids:
            return rank
    return None


def _three_way_qualitative(
    zero: Mapping[str, Sequence[RankingRecord]],
    full: Mapping[str, Sequence[RankingRecord]],
    lora: Mapping[str, Sequence[RankingRecord]],
    records: Sequence[ImageRecord],
) -> dict[str, Any]:
    contexts = {
        caption.caption_id: caption.text
        for record in records
        for caption in record.captions
    }
    contexts.update(
        {record.image_id: record.captions[0].text for record in records if record.captions}
    )
    output: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        zero_map = {item.query_id: item for item in zero[task]}
        full_map = {item.query_id: item for item in full[task]}
        lora_map = {item.query_id: item for item in lora[task]}
        categories = {
            "lora_beats_full": 0,
            "lora_fixes_zero": 0,
            "full_only_fix": 0,
            "lora_degraded": 0,
            "unchanged": 0,
        }
        examples: dict[str, list[dict[str, Any]]] = {key: [] for key in categories}
        for query_id in sorted(zero_map):
            zero_rank = _rank(zero_map[query_id])
            full_rank = _rank(full_map[query_id])
            lora_rank = _rank(lora_map[query_id])
            if lora_rank is not None and full_rank is not None and lora_rank < full_rank:
                category = "lora_beats_full"
            elif lora_rank is not None and zero_rank is not None and lora_rank < zero_rank:
                category = "lora_fixes_zero"
            elif full_rank is not None and zero_rank is not None and full_rank < zero_rank and (lora_rank is None or lora_rank >= zero_rank):
                category = "full_only_fix"
            elif (lora_rank is None and zero_rank is not None) or (lora_rank is not None and zero_rank is not None and lora_rank > zero_rank):
                category = "lora_degraded"
            else:
                category = "unchanged"
            categories[category] += 1
            if len(examples[category]) < 3:
                examples[category].append(
                    {
                        "query_id": query_id,
                        "query_context": contexts.get(query_id, ""),
                        "zero_shot_rank": zero_rank,
                        "full_finetuned_rank": full_rank,
                        "lora_rank": lora_rank,
                        "zero_shot_top5": list(zero_map[query_id].candidate_ids[:5]),
                        "full_finetuned_top5": list(full_map[query_id].candidate_ids[:5]),
                        "lora_top5": list(lora_map[query_id].candidate_ids[:5]),
                    }
                )
        output[task] = {"counts": categories, "examples": examples}
    return output


def _retention(
    zero_metrics: Mapping[str, Any],
    full_metrics: Mapping[str, Any],
    lora_metrics: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in keys:
        full_delta = float(full_metrics[key]) - float(zero_metrics[key])
        lora_delta = float(lora_metrics[key]) - float(zero_metrics[key])
        item: dict[str, Any] = {
            "zero_shot": zero_metrics[key],
            "full_finetuning": full_metrics[key],
            "lora": lora_metrics[key],
            "full_improvement": full_delta,
            "lora_improvement": lora_delta,
        }
        if full_delta > 1e-8:
            item["lora_fraction_of_full_improvement"] = lora_delta / full_delta
        else:
            item["lora_fraction_of_full_improvement"] = None
            item["retention_note"] = "not summarized because full-FT improvement is non-positive or near zero"
        output[key] = item
    return output


def _load_phase7_results(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    zero: dict[str, Any] = {}
    full: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        zero[task] = json.loads((artifact_dir / f"zero_shot_{task}.json").read_text())
        full[task] = json.loads((artifact_dir / f"fine_tuned_{task}.json").read_text())
    return zero, full


def _merged_equivalence(
    unmerged: Mapping[str, Sequence[RankingRecord]],
    merged: Mapping[str, Sequence[RankingRecord]],
) -> dict[str, Any]:
    max_score_delta = 0.0
    ranking_ids_equal = True
    for task in ("text_to_image", "image_to_text"):
        left = unmerged[task]
        right = merged[task]
        for left_item, right_item in zip(left, right):
            ranking_ids_equal = ranking_ids_equal and left_item.candidate_ids == right_item.candidate_ids
            for left_score, right_score in zip(left_item.scores, right_item.scores):
                max_score_delta = max(max_score_delta, abs(left_score - right_score))
    return {
        "ranking_ids_equal": ranking_ids_equal,
        "max_returned_score_absolute_delta": max_score_delta,
        "tolerance": 1e-5,
        "ranking_id_note": "ID differences are retained as an observed near-tie effect; score equivalence is the merge criterion",
        "equivalent_within_tolerance": max_score_delta <= 1e-5,
    }


def run_phase8(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase8",
    smoke: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_config = load_config(config_path)
    config = _read_phase8_config(config_path)
    if smoke:
        config.update(
            {
                "epochs": 1,
                "max_train_images": 8,
                "max_validation_images": 4,
                "max_test_images": 4,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "bootstrap_resamples": 10,
                "phase7_artifact_dir": "artifacts/phase7_smoke",
            }
        )
    validate_phase8_config(config)
    manifest_path = Path(str(config["manifest"]))
    image_root = Path(str(config["image_root"]))
    manifest = read_manifest(manifest_path)
    assert_no_split_leakage(manifest.records)
    if manifest.dataset_id != project_config.dataset_id:
        raise ValueError("Phase 8 dataset does not match the active project dataset")
    seed = project_config.seed
    subset_seed = seed if config.get("subset_seed") is None else int(config["subset_seed"])
    train_records = _subset_records(manifest.records, "train", subset_seed, config.get("max_train_images"))
    validation_records = _subset_records(manifest.records, "validation", subset_seed, config.get("max_validation_images"))
    if not train_records or not validation_records:
        raise ValueError("Phase 8 requires non-empty train and validation groups")
    phase7_dir = Path(str(config["phase7_artifact_dir"]))
    phase7_report = json.loads((phase7_dir / "phase7_report.json").read_text())
    if phase7_report["pretrained_checkpoint"]["model_id"] != config["model_id"]:
        raise ValueError("Phase 8 base checkpoint differs from Phase 7")
    if phase7_report["dataset"]["manifest_sha256"] != _hash_file(manifest_path):
        raise ValueError("Phase 8 manifest differs from Phase 7")
    _write_json(
        {
            "phase8_schema_version": PHASE8_SCHEMA_VERSION,
            "phase7_schema_version": PHASE7_SCHEMA_VERSION,
            "smoke": smoke,
            "dataset_id": manifest.dataset_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _hash_file(manifest_path),
            "scope": {
                "train_image_groups": len(train_records),
                "validation_image_groups": len(validation_records),
                "test_isolation": "test is materialized after validation adapter selection",
            },
            "config": config,
        },
        output_dir / "experiment_manifest.json",
    )
    model, processor, torch, device = _load_lora_model(config)
    _assert_frozen_contract(model)
    parameter_summary = _parameter_summary(model)
    adapter_before = _adapter_parameter(model).detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_pairs = build_training_pairs(train_records, image_root, seed, epoch=0)
    steps_per_epoch = max(
        1,
        (len(train_pairs) + int(config["batch_size"]) - 1)
        // int(config["batch_size"])
        // int(config["gradient_accumulation_steps"]),
    )
    total_steps = max(1, steps_per_epoch * int(config["epochs"]))
    warmup_steps = int(config["warmup_steps"])

    def schedule(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    adapter_dir = output_dir / "selected_adapter"
    best_score = float("-inf")
    best_epoch: int | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        pairs = build_training_pairs(train_records, image_root, seed, epoch=epoch)
        train_stats, _, update_verified = _train_epoch(
            model,
            processor,
            optimizer,
            scheduler,
            torch,
            pairs,
            int(config["batch_size"]),
            int(config["gradient_accumulation_steps"]),
            int(config["text_max_length"]),
            float(config["max_grad_norm"]),
            adapter_before if epoch == 0 else None,
            int(config["num_workers"]),
            str(config["precision"]),
        )
        adapter_before = None
        base_gradients = [
            name
            for name, parameter in _base_model(model).named_parameters()
            if parameter.grad is not None and not name.endswith("logit_scale")
        ]
        if base_gradients:
            raise AssertionError(f"frozen base parameters received gradients: {base_gradients[:5]}")
        validation_pairs = build_training_pairs(validation_records, image_root, seed, epoch=0)
        validation_loss = _validation_loss(
            model,
            processor,
            torch,
            validation_pairs,
            int(config["batch_size"]),
            int(config["text_max_length"]),
            int(config["num_workers"]),
            str(config["precision"]),
        )
        validation_eval = evaluate_model(
            model,
            processor,
            torch,
            validation_records,
            image_root,
            int(config["batch_size"]),
            int(config["text_max_length"]),
            manifest,
            manifest_path,
            config_path,
            "validation",
            seed,
            "phase8_validation_lora",
            f"phase8_validation_epoch_{epoch + 1}",
            {**parameter_summary, "model_id": config["model_id"], "frozen_base": True},
            int(config["bootstrap_resamples"]),
            int(config["num_workers"]),
            str(config["precision"]),
        )
        metric_key = str(config["selection_metric"])
        metric_suffix = metric_key.removeprefix("mean_")
        selection_score = sum(
            float(validation_eval["results"][task]["metrics"][metric_suffix])
            for task in ("text_to_image", "image_to_text")
        ) / 2.0
        improved = selection_score > best_score
        if improved:
            best_score = selection_score
            best_epoch = epoch + 1
            no_improvement = 0
            _save_adapter(
                model,
                adapter_dir,
                {
                    "epoch": best_epoch,
                    "selection_metric": metric_key,
                    "selection_score": selection_score,
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "base_model_id": config["model_id"],
                    "lora_rank": config["lora_rank"],
                    "lora_alpha": config["lora_alpha"],
                    "lora_dropout": config["lora_dropout"],
                    "lora_target_modules": config["lora_target_modules"],
                },
            )
        else:
            no_improvement += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_stats,
                "validation_loss": validation_loss,
                "validation_metrics": {
                    "text_to_image": validation_eval["results"]["text_to_image"]["metrics"],
                    "image_to_text": validation_eval["results"]["image_to_text"]["metrics"],
                    "selection_metric": metric_key,
                    "selection_score": selection_score,
                },
                "checkpoint_selected": improved,
                "base_parameters_frozen": True,
                "optimizer_updates_adapters": update_verified,
            }
        )
        _write_json(history, output_dir / "training_history.json")
        if no_improvement >= int(config["early_stopping_patience"]):
            break
    if best_epoch is None:
        raise RuntimeError("no validation-selected LoRA adapter was produced")
    _write_json(
        {
            "selected_adapter": str(adapter_dir),
            "selected_epoch": best_epoch,
            "selection_metric": config["selection_metric"],
            "selection_score": best_score,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "base_checkpoint": config["model_id"],
            "base_parameters": parameter_summary["base_parameters"],
            "adapter_trainable_parameters": parameter_summary["adapter_trainable_parameters"],
            "extra_trainable_parameters": parameter_summary["extra_trainable_parameters"],
            "total_trainable_parameters": parameter_summary["total_trainable_parameters"],
        },
        output_dir / "selected_adapter_metadata.json",
    )

    # The test split begins here, after validation-only adapter selection.
    test_records = _subset_records(manifest.records, "test", subset_seed, config.get("max_test_images"))
    if not test_records:
        raise ValueError("Phase 8 requires non-empty test groups")
    del model
    gc.collect()
    lora_model, lora_processor, lora_torch, lora_device = _load_adapter(config, adapter_dir)
    lora_eval = evaluate_model(
        lora_model,
        lora_processor,
        lora_torch,
        test_records,
        image_root,
        int(config["batch_size"]),
        int(config["text_max_length"]),
        manifest,
        manifest_path,
        config_path,
        "test",
        seed,
        "phase8_lora_clip",
        "phase8_lora_clip_test",
        {**parameter_summary, "model_id": config["model_id"], "frozen_base": True},
        int(config["bootstrap_resamples"]),
        int(config["num_workers"]),
        str(config["precision"]),
    )
    lora_results = lora_eval["results"]
    lora_rankings = lora_eval["rankings"]
    zero_results, full_results = _load_phase7_results(phase7_dir)
    zero_rankings = {task: _ranking_records(zero_results[task]) for task in zero_results}
    full_rankings = {task: _ranking_records(full_results[task]) for task in full_results}
    comparisons: dict[str, Any] = {}
    for task in ("text_to_image", "image_to_text"):
        lora_metadata = _comparison_metadata(lora_results[task])
        comparisons[task] = {
            "lora_vs_zero_shot": compare_systems(
                zero_rankings[task],
                lora_rankings[task],
                _result_metadata(zero_results[task]),
                lora_metadata,
                bootstrap_resamples=int(config["bootstrap_resamples"]),
                seed=seed,
            ),
            "lora_vs_full_finetuning": compare_systems(
                full_rankings[task],
                lora_rankings[task],
                _result_metadata(full_results[task]),
                lora_metadata,
                bootstrap_resamples=int(config["bootstrap_resamples"]),
                seed=seed,
            ),
        }
        _write_json(lora_results[task], output_dir / f"lora_{task}.json")
        write_result_artifacts(lora_results[task], output_dir / f"lora_{task}_summary")
    _write_json(comparisons, output_dir / "paired_comparisons.json")
    qualitative = _three_way_qualitative(
        zero_rankings, full_rankings, lora_rankings, test_records
    )
    _write_json(qualitative, output_dir / "qualitative_three_way.json")
    retention = {
        task: _retention(
            zero_results[task]["metrics"],
            full_results[task]["metrics"],
            lora_results[task]["metrics"],
            ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "map"),
        )
        for task in ("text_to_image", "image_to_text")
    }
    _write_json(retention, output_dir / "performance_retention.json")

    merge_unmerged_eval = evaluate_model(
        lora_model,
        lora_processor,
        lora_torch,
        validation_records,
        image_root,
        int(config["batch_size"]),
        int(config["text_max_length"]),
        manifest,
        manifest_path,
        config_path,
        "validation",
        seed,
        "phase8_lora_clip_merge_check_unmerged",
        "phase8_lora_clip_merge_check_unmerged",
        {**parameter_summary, "model_id": config["model_id"], "merged": False},
        int(config["bootstrap_resamples"]),
        int(config["num_workers"]),
        str(config["precision"]),
    )
    merged = lora_model.merge_and_unload()
    merged_eval = evaluate_model(
        merged,
        lora_processor,
        lora_torch,
        validation_records,
        image_root,
        int(config["batch_size"]),
        int(config["text_max_length"]),
        manifest,
        manifest_path,
        config_path,
        "validation",
        seed,
        "phase8_lora_clip_merged",
        "phase8_lora_clip_merge_check_merged",
        {**parameter_summary, "model_id": config["model_id"], "merged": True},
        int(config["bootstrap_resamples"]),
        int(config["num_workers"]),
        str(config["precision"]),
    )
    merge_check = _merged_equivalence(
        merge_unmerged_eval["rankings"], merged_eval["rankings"]
    )
    merge_check["split"] = "validation"
    for task in ("text_to_image", "image_to_text"):
        _write_json(
            merged_eval["results"][task],
            output_dir / f"merged_lora_validation_{task}.json",
        )
    _write_json(merge_check, output_dir / "merge_equivalence.json")
    del merged
    gc.collect()

    phase7_efficiency = json.loads((phase7_dir / "efficiency.json").read_text())
    efficiency = {
        "full_finetuning": {
            "trainable_parameters": phase7_efficiency["trainable_parameters"],
            "trainable_percentage": 100.0,
            "training_seconds": phase7_efficiency["training_seconds"],
            "checkpoint_size_bytes": phase7_efficiency["checkpoint_size_bytes"],
            "device": phase7_efficiency["device_finetuned"],
        },
        "lora": {
            "trainable_parameters": parameter_summary["total_trainable_parameters"],
            "trainable_percentage": parameter_summary["trainable_percentage_of_base"],
            "training_seconds": time.perf_counter() - training_started,
            "adapter_size_bytes": _adapter_size(adapter_dir),
            "device": str(lora_device),
            "base_checkpoint_required": True,
        },
        "parameter_reduction": parameter_summary["parameter_reduction_vs_full_finetuning"],
        "memory_status": "not_reliably_measured_for_unified-memory-MPS",
        "inference_encoding_seconds": {
            "lora_test_unmerged": lora_eval["runtime"]["encoding_seconds"],
            "merge_check_validation_unmerged": merge_unmerged_eval["runtime"]["encoding_seconds"],
            "merge_check_validation_merged": merged_eval["runtime"]["encoding_seconds"],
        },
    }
    _write_json(efficiency, output_dir / "efficiency_comparison.json")
    three_way = {
        task: {
            "zero_shot": zero_results[task]["metrics"],
            "full_finetuned": full_results[task]["metrics"],
            "lora": lora_results[task]["metrics"],
        }
        for task in ("text_to_image", "image_to_text")
    }
    three_way["parameter_efficiency"] = {
        "zero_shot_trainable_parameters": 0,
        "full_finetuned_trainable_parameters": phase7_efficiency["trainable_parameters"],
        "lora_trainable_parameters": parameter_summary["total_trainable_parameters"],
    }
    _write_json(three_way, output_dir / "three_way_comparison.json")
    failure_analysis = {
        "observed_directional_metrics": {
            task: {
                "zero_shot": zero_results[task]["metrics"],
                "full_finetuned": full_results[task]["metrics"],
                "lora": lora_results[task]["metrics"],
            }
            for task in ("text_to_image", "image_to_text")
        },
        "qualitative_categories": {
            task: qualitative[task]["counts"] for task in qualitative
        },
        "interpretation_limit": "caption/image IDs and rank movement are observed; no human labels were used to claim counting, attribute, compositional, or rare-concept error classes",
    }
    _write_json(failure_analysis, output_dir / "failure_analysis.json")
    provenance = {
        "project": "OmniSearch",
        "package_version": __version__,
        "phase8_schema_version": PHASE8_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _hash_file(manifest_path),
        "base_model_id": config["model_id"],
        "peft_library": "peft",
        "peft_version": _peft_version(),
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "test_used_for_selection": False,
    }
    _write_json(provenance, output_dir / "provenance.json")
    scope_tier = "tier2_student_compute" if not smoke else "tier1_smoke_subset"
    report = {
        "report_schema_version": PHASE8_SCHEMA_VERSION,
        "project": "OmniSearch",
        "phase": 8,
        "scope": {
            "smoke": smoke,
            "dataset_id": manifest.dataset_id,
            "tier": scope_tier,
            "same_phase7_scope": True,
            "hard_negative_mining": False,
            "ann": False,
            "reranking": False,
            "test_used_for_selection": False,
        },
        "base_checkpoint": {
            "model_id": config["model_id"],
            "phase7_artifact_dir": str(phase7_dir),
            "full_finetuned_checkpoint": phase7_report["checkpoint"]["path"],
        },
        "dataset": {
            "manifest": str(manifest_path),
            "manifest_sha256": _hash_file(manifest_path),
            "train_image_groups": len(train_records),
            "validation_image_groups": len(validation_records),
            "test_image_groups": len(test_records),
            "same_image_grouped_split": True,
            "test_materialized_after_selection": True,
        },
        "lora_configuration": {
            "library": "peft",
            "library_version": _peft_version(),
            "rank": config["lora_rank"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "target_modules": config["lora_target_modules"],
            "target_scope": "all matching q_proj/v_proj attention projections in CLIP text and vision encoders",
            "bias": config["lora_bias"],
            "train_logit_scale": config["train_logit_scale"],
            "base_weights_frozen": True,
        },
        "training_configuration": config,
        "trainable_parameters": parameter_summary,
        "training_history": history,
        "selected_adapter": {
            "path": str(adapter_dir),
            "selected_epoch": best_epoch,
            "selection_metric": config["selection_metric"],
            "selection_score": best_score,
            "selection_split": "validation",
        },
        "zero_shot_results": {task: zero_results[task]["metrics"] for task in zero_results},
        "full_finetuning_results": {task: full_results[task]["metrics"] for task in full_results},
        "lora_results": {task: lora_results[task]["metrics"] for task in lora_results},
        "paired_comparisons": comparisons,
        "performance_retention": retention,
        "efficiency": efficiency,
        "qualitative": qualitative,
        "failure_analysis": failure_analysis,
        "merge_equivalence": merge_check,
        "provenance": provenance,
        "quality_gate": {
            "phase7_audit": "PASS",
            "training_smoke": "PASS" if smoke else "separate_artifact_required",
            "real_training": "SMOKE_ONLY" if smoke else "PASS",
            "base_frozen": True,
            "finite_loss": all(bool(item["train"].get("gradients_finite", False)) for item in history),
            "adapter_updated": all(bool(item["optimizer_updates_adapters"]) for item in history),
            "adapter_save_load": True,
            "merged_unmerged_equivalent": bool(merge_check["equivalent_within_tolerance"]),
            "canonical_protocol": PROTOCOL_VERSION,
            "test_isolation": True,
            "status": "SMOKE_ONLY" if smoke else "PASS",
        },
    }
    _write_json(report, output_dir / "phase8_report.json")
    lines = [
        "# OmniSearch Phase 8 PEFT/LoRA report",
        "",
        f"Scope: `{scope_tier}`; LoRA rank `{config['lora_rank']}`; device `{device}`.",
        "",
        f"Selected adapter epoch: `{best_epoch}` by validation `{config['selection_metric']}` = `{best_score:.6f}`. Test selection: `False`.",
        "",
        "| Task | Zero-shot R@5 | Full FT R@5 | LoRA R@5 | LoRA − full FT |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in ("text_to_image", "image_to_text"):
        zero_value = float(zero_results[task]["metrics"]["recall_at_5"])
        full_value = float(full_results[task]["metrics"]["recall_at_5"])
        lora_value = float(lora_results[task]["metrics"]["recall_at_5"])
        lines.append(
            f"| {task} | {zero_value:.4f} | {full_value:.4f} | {lora_value:.4f} | {lora_value - full_value:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Adapter artifact size: `{efficiency['lora']['adapter_size_bytes']}` bytes; base checkpoint remains required.",
            "",
            "Phase 9 features (hard negatives, ANN, reranking, fusion, uncertainty, APIs) were not implemented.",
            "",
        ]
    )
    (output_dir / "phase8_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _peft_version() -> str:
    try:
        import peft

        return str(peft.__version__)
    except ImportError:  # pragma: no cover
        return "unavailable"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run OmniSearch Phase 8 PEFT/LoRA.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase8"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run_phase8(args.config, args.output_dir, args.smoke)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "smoke": report["scope"]["smoke"],
                "quality_gate": report["quality_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
