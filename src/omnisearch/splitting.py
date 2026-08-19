"""Deterministic image-grouped splits, leakage checks, and tier selection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from .manifest import ImageRecord

SPLIT_NAMES = ("train", "validation", "test")
# Tier sizes are image-group counts. Tier 3 is the complete selected source
# release; no tier ever truncates an image's captions.
TIER_LIMITS: dict[str, int | None] = {"tier1": 100, "tier2": 1000, "tier3": None}


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def assign_image_grouped_splits(
    records: Iterable[ImageRecord],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[ImageRecord, ...]:
    """Assign every image and all its captions to exactly one split."""

    records = tuple(records)
    if abs(sum(ratios) - 1.0) > 1e-9 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("split ratios must be positive and sum to 1")
    image_ids = [record.image_id for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("cannot split a manifest containing duplicate image IDs")
    ordered_ids = sorted(image_ids, key=lambda image_id: _stable_key(seed, image_id))
    train_count = int(len(ordered_ids) * ratios[0])
    validation_count = int(len(ordered_ids) * ratios[1])
    if len(ordered_ids) >= 3:
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)
    split_by_id = {image_id: "train" for image_id in ordered_ids[:train_count]}
    split_by_id.update(
        {
            image_id: "validation"
            for image_id in ordered_ids[train_count : train_count + validation_count]
        }
    )
    split_by_id.update(
        {image_id: "test" for image_id in ordered_ids[train_count + validation_count :]}
    )
    return tuple(record.with_split(split_by_id[record.image_id]) for record in records)


def assert_no_split_leakage(records: Iterable[ImageRecord]) -> None:
    """Raise if image IDs or caption IDs occur across split boundaries."""

    image_splits: defaultdict[str, set[str]] = defaultdict(set)
    caption_splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.split not in SPLIT_NAMES:
            raise ValueError(f"record {record.image_id} has no valid split")
        image_splits[record.image_id].add(record.split)
        for caption in record.captions:
            caption_splits[caption.caption_id].add(record.split)
    leaking_images = {
        key: sorted(value) for key, value in image_splits.items() if len(value) > 1
    }
    leaking_captions = {
        key: sorted(value) for key, value in caption_splits.items() if len(value) > 1
    }
    if leaking_images or leaking_captions:
        raise ValueError(
            f"split leakage detected: images={leaking_images}, captions={leaking_captions}"
        )


def split_counts(records: Iterable[ImageRecord]) -> dict[str, dict[str, int]]:
    output = {split: {"images": 0, "captions": 0} for split in SPLIT_NAMES}
    for record in records:
        if record.split not in output:
            raise ValueError(f"invalid split: {record.split}")
        output[record.split]["images"] += 1
        output[record.split]["captions"] += len(record.captions)
    return output


def select_tier(
    records: Iterable[ImageRecord],
    tier: str,
    seed: int = 42,
    include_unresolved: bool = False,
) -> tuple[ImageRecord, ...]:
    """Select whole image records, never individual captions.

    By default, metadata-only groups without a source image ID are excluded
    from candidate image tiers. They remain in the full metadata manifest.
    """

    if tier not in TIER_LIMITS:
        raise ValueError(f"unknown tier {tier}; expected one of {sorted(TIER_LIMITS)}")
    records = tuple(
        record
        for record in records
        if include_unresolved or record.source_image_id_available
    )
    limit = TIER_LIMITS[tier]
    ordered = sorted(records, key=lambda record: _stable_key(seed, record.image_id))
    return tuple(ordered if limit is None else ordered[: min(limit, len(ordered))])
