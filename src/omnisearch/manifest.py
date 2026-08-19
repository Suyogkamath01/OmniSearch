"""Manifest schema, caption parsing, validation, and dataset statistics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .preprocessing import normalize_text

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaptionRecord:
    caption_id: str
    text: str

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.text)

    @property
    def normalized_hash(self) -> str:
        return hashlib.sha256(self.normalized_text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, str]:
        return {
            "caption_id": self.caption_id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "normalized_hash": self.normalized_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaptionRecord:
        return cls(caption_id=str(value["caption_id"]), text=str(value["text"]))


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    filename: str | None
    captions: tuple[CaptionRecord, ...]
    image_url: str | None = None
    split: str | None = None
    source_image_id_available: bool = True

    def with_split(self, split: str) -> ImageRecord:
        return replace(self, split=split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "image_url": self.image_url,
            "split": self.split,
            "source_image_id_available": self.source_image_id_available,
            "captions": [caption.to_dict() for caption in self.captions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImageRecord:
        return cls(
            image_id=str(value["image_id"]),
            filename=str(value["filename"]) if value.get("filename") else None,
            image_url=str(value["image_url"]) if value.get("image_url") else None,
            split=str(value["split"]) if value.get("split") else None,
            source_image_id_available=bool(
                value.get("source_image_id_available", True)
            ),
            captions=tuple(
                CaptionRecord.from_dict(item) for item in value.get("captions", [])
            ),
        )


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    dataset_version: str
    source_url: str
    terms_url: str
    source_snapshot_marker: str
    source_sha256: str | None
    records: tuple[ImageRecord, ...]
    metadata: dict[str, Any]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source_url": self.source_url,
            "terms_url": self.terms_url,
            "source_snapshot_marker": self.source_snapshot_marker,
            "source_sha256": self.source_sha256,
            "metadata": self.metadata,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetManifest:
        schema_version = int(value.get("schema_version", 0))
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema version: {schema_version}")
        return cls(
            dataset_id=str(value["dataset_id"]),
            dataset_version=str(value["dataset_version"]),
            source_url=str(value["source_url"]),
            terms_url=str(value["terms_url"]),
            source_snapshot_marker=str(
                value.get(
                    "source_snapshot_marker",
                    value.get("accessed_at_utc", "not_recorded"),
                )
            ),
            source_sha256=str(value["source_sha256"])
            if value.get("source_sha256")
            else None,
            records=tuple(
                ImageRecord.from_dict(item) for item in value.get("records", [])
            ),
            metadata=dict(value.get("metadata", {})),
            schema_version=schema_version,
        )


@dataclass(frozen=True)
class ValidationReport:
    """Validation findings; duplicate captions are findings, not automatic errors."""

    passed: bool
    records_checked: int
    unique_image_ids: int
    total_captions: int
    duplicate_image_ids: dict[str, int]
    missing_caption_image_ids: tuple[str, ...]
    caption_count_mismatches: dict[str, int]
    duplicate_caption_groups: dict[str, tuple[str, ...]]
    duplicate_caption_texts: dict[str, str]
    records_without_source_image_id: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(manifest: DatasetManifest, path: Path | str) -> str:
    """Write a canonical JSON manifest and return its SHA-256 digest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )
    output.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return digest


def read_manifest(path: Path | str) -> DatasetManifest:
    with Path(path).open("r", encoding="utf-8") as file:
        return DatasetManifest.from_dict(json.load(file))


def write_validation_report(report: ValidationReport, path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_manifest(
    records: Iterable[ImageRecord],
    expected_captions_per_image: int | None = 5,
    allowed_image_extensions: frozenset[str] | set[str] | None = None,
) -> ValidationReport:
    """Validate common image-caption structure.

    ``expected_captions_per_image`` may be ``None`` for datasets whose source
    schema permits a variable number of captions.  The default extension set
    preserves the historical Flickr30k JPEG contract; adapters for other
    datasets should pass the extensions declared by that source.
    """
    records = tuple(records)
    extensions = {
        extension.casefold() for extension in (allowed_image_extensions or {".jpg", ".jpeg"})
    }
    id_counts = Counter(record.image_id for record in records)
    duplicate_image_ids = {
        key: count for key, count in sorted(id_counts.items()) if count > 1
    }
    missing_caption_image_ids = tuple(
        record.image_id for record in records if not record.captions
    )
    caption_count_mismatches = (
        {
            record.image_id: len(record.captions)
            for record in records
            if len(record.captions) != expected_captions_per_image
        }
        if expected_captions_per_image is not None
        else {}
    )

    caption_occurrences: defaultdict[str, list[str]] = defaultdict(list)
    duplicate_caption_texts: dict[str, str] = {}
    records_without_source_image_id = tuple(
        record.image_id for record in records if not record.source_image_id_available
    )
    errors: list[str] = []
    warnings: list[str] = []

    if not records:
        errors.append("manifest contains no image records")
    if duplicate_image_ids:
        errors.append("duplicate image IDs detected")
    if missing_caption_image_ids:
        errors.append("one or more image records have no captions")
    if caption_count_mismatches:
        errors.append(
            f"caption count mismatch; expected {expected_captions_per_image} per image"
        )

    for record in records:
        if not record.image_id or "/" in record.image_id or "\\" in record.image_id:
            errors.append(f"invalid image ID: {record.image_id!r}")
        if record.source_image_id_available:
            if not record.filename or Path(record.filename).name != record.filename:
                errors.append(
                    f"source-identified record has invalid filename: {record.image_id}"
                )
            elif Path(record.filename).suffix.casefold() not in extensions:
                errors.append(
                    f"source-identified record has unsupported image extension: {record.image_id}"
                )
        elif record.filename is not None:
            errors.append(
                f"unresolved source record must not have a filename: {record.image_id}"
            )
        if record.split not in {None, "train", "validation", "test"}:
            errors.append(f"invalid split for {record.image_id}: {record.split!r}")
        caption_ids = [caption.caption_id for caption in record.captions]
        if len(caption_ids) != len(set(caption_ids)):
            errors.append(f"duplicate caption IDs within image {record.image_id}")
        for caption in record.captions:
            if not caption.caption_id:
                errors.append(f"empty caption ID for {record.image_id}")
            if not caption.text.strip():
                errors.append(f"empty caption text for {caption.caption_id}")
            normalized = caption.normalized_text
            if normalized:
                caption_occurrences[normalized].append(caption.caption_id)

    duplicate_caption_groups: dict[str, tuple[str, ...]] = {}
    for normalized_text, caption_ids in sorted(caption_occurrences.items()):
        if len(caption_ids) > 1:
            duplicate_caption_groups[normalized_text] = tuple(sorted(caption_ids))
            duplicate_caption_texts[
                hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            ] = normalized_text
    if duplicate_caption_groups:
        warnings.append(
            f"{len(duplicate_caption_groups)} normalized duplicate caption groups detected"
        )
    if records_without_source_image_id:
        warnings.append(
            f"{len(records_without_source_image_id)} records have captions but no source image ID"
        )

    return ValidationReport(
        passed=not errors,
        records_checked=len(records),
        unique_image_ids=len(id_counts),
        total_captions=sum(len(record.captions) for record in records),
        duplicate_image_ids=duplicate_image_ids,
        missing_caption_image_ids=missing_caption_image_ids,
        caption_count_mismatches=caption_count_mismatches,
        duplicate_caption_groups=duplicate_caption_groups,
        duplicate_caption_texts=duplicate_caption_texts,
        records_without_source_image_id=records_without_source_image_id,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def dataset_statistics(
    records: Iterable[ImageRecord], validation: ValidationReport | None = None
) -> dict[str, Any]:
    records = tuple(records)
    captions_per_image = Counter(len(record.captions) for record in records)
    split_counts = Counter(record.split or "unassigned" for record in records)
    return {
        "records": len(records),
        "unique_image_ids": len({record.image_id for record in records}),
        "records_with_source_image_id": sum(
            record.source_image_id_available for record in records
        ),
        "records_without_source_image_id": sum(
            not record.source_image_id_available for record in records
        ),
        "captions": sum(len(record.captions) for record in records),
        "captions_per_image": {
            str(key): value for key, value in sorted(captions_per_image.items())
        },
        "split_image_counts": dict(sorted(split_counts.items())),
        "split_caption_counts": {
            split: sum(
                len(record.captions)
                for record in records
                if (record.split or "unassigned") == split
            )
            for split in sorted(split_counts)
        },
        "validation": validation.to_dict() if validation else None,
    }
