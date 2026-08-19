"""Schema and access adapter for the official CIRCO benchmark.

The adapter keeps CIRCO's composed-query contract separate from the COCO
image-caption manifest used by the earlier phases.  It validates the released
validation annotations, preserves multiple ground-truth image IDs, and never
silently turns a reference image into a positive target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CIRCO_REPOSITORY_URL = "https://github.com/miccunifi/CIRCO"
CIRCO_OFFICIAL_SITE = "https://circo.micc.unifi.it/"
CIRCO_LICENSE_URL = "https://github.com/miccunifi/CIRCO/blob/main/LICENSE"
CIRCO_LICENSE_NOTE = (
    "The official CIRCO repository states that its material is CC BY-NC 4.0. "
    "The underlying COCO images retain their originating Flickr copyright and "
    "terms; this project uses them only through the official COCO access path."
)
CIRCO_VAL_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/miccunifi/CIRCO/main/annotations/val.json"
)
CIRCO_TEST_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/miccunifi/CIRCO/main/annotations/test.json"
)
COCO_UNLABELED_IMAGES_URL = "http://images.cocodataset.org/zips/unlabeled2017.zip"
COCO_UNLABELED_INFO_URL = (
    "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip"
)
CIRCO_SCHEMA_VERSION = 1


class CircoSchemaError(ValueError):
    """Raised when released CIRCO metadata violates its documented schema."""


def normalize_image_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CircoSchemaError(f"image ID must be an integer or string: {value!r}")
    text = str(value).strip()
    if not text or not text.isdigit():
        raise CircoSchemaError(f"image ID must be a non-negative integer string: {value!r}")
    return str(int(text))


@dataclass(frozen=True)
class ComposedQuery:
    """One CIRCO query and its released relevance labels."""

    query_id: str
    split: str
    reference_image_id: str
    modification_text: str
    target_image_id: str | None
    ground_truth_image_ids: frozenset[str]
    semantic_aspects: tuple[str, ...]
    shared_concept: str | None

    def __post_init__(self) -> None:
        if not self.query_id:
            raise CircoSchemaError("query_id is required")
        if self.split not in {"val", "test"}:
            raise CircoSchemaError("CIRCO split must be val or test")
        if not self.modification_text.strip():
            raise CircoSchemaError(f"empty modification text for query {self.query_id}")
        if self.split == "val" and not self.ground_truth_image_ids:
            raise CircoSchemaError(f"validation query has no ground truth: {self.query_id}")
        if self.reference_image_id in self.ground_truth_image_ids:
            raise CircoSchemaError(
                f"reference image is a CIRCO target for query {self.query_id}; "
                "reference copying is not permitted by the Phase 12B contract"
            )
        if self.target_image_id is not None and self.target_image_id not in self.ground_truth_image_ids:
            raise CircoSchemaError(f"target image is not in ground truth for query {self.query_id}")

    @property
    def reference_filename(self) -> str:
        return f"{int(self.reference_image_id):012d}.jpg"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "split": self.split,
            "reference_image_id": self.reference_image_id,
            "reference_filename": self.reference_filename,
            "modification_text": self.modification_text,
            "target_image_id": self.target_image_id,
            "ground_truth_image_ids": sorted(self.ground_truth_image_ids),
            "semantic_aspects": list(self.semantic_aspects),
            "shared_concept": self.shared_concept,
        }


@dataclass(frozen=True)
class CircoGallery:
    """The official COCO-unlabeled candidate image corpus metadata."""

    image_ids: tuple[str, ...]
    filenames: dict[str, str]

    def __post_init__(self) -> None:
        if not self.image_ids:
            raise CircoSchemaError("CIRCO gallery is empty")
        if len(set(self.image_ids)) != len(self.image_ids):
            raise CircoSchemaError("CIRCO gallery has duplicate image IDs")
        if set(self.image_ids) != set(self.filenames):
            raise CircoSchemaError("CIRCO gallery IDs and filenames differ")

    def path_for(self, image_id: str, image_root: Path) -> Path:
        try:
            filename = self.filenames[str(image_id)]
        except KeyError as exc:
            raise CircoSchemaError(f"image is absent from CIRCO gallery: {image_id}") from exc
        return image_root / filename


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CIRCO metadata file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CircoSchemaError(f"invalid JSON in CIRCO metadata: {path}") from exc


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_circo_queries(
    annotation_path: Path | str,
    split: str = "val",
    require_ground_truth: bool = True,
) -> tuple[ComposedQuery, ...]:
    """Load and validate official CIRCO query annotations.

    Validation annotations contain `target_img_id` and `gt_img_ids`; test
    annotations intentionally do not contain released ground-truth targets.
    """

    if split not in {"val", "test"}:
        raise ValueError("CIRCO split must be val or test")
    raw = _load_json(Path(annotation_path))
    if not isinstance(raw, list) or not raw:
        raise CircoSchemaError("CIRCO annotations must be a non-empty JSON list")
    queries: list[ComposedQuery] = []
    seen_ids: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise CircoSchemaError("every CIRCO annotation must be an object")
        required = {"id", "reference_img_id", "relative_caption"}
        missing = sorted(required - set(entry))
        if missing:
            raise CircoSchemaError(f"CIRCO query is missing fields: {', '.join(missing)}")
        query_id = str(entry["id"])
        if query_id in seen_ids:
            raise CircoSchemaError(f"duplicate CIRCO query ID: {query_id}")
        seen_ids.add(query_id)
        raw_gt = entry.get("gt_img_ids", [])
        if raw_gt is None:
            raw_gt = []
        if not isinstance(raw_gt, list):
            raise CircoSchemaError(f"gt_img_ids must be a list for query {query_id}")
        gt_ids = tuple(normalize_image_id(value) for value in raw_gt)
        if len(set(gt_ids)) != len(gt_ids):
            raise CircoSchemaError(f"duplicate ground-truth IDs for query {query_id}")
        target_value = entry.get("target_img_id")
        target_id = normalize_image_id(target_value) if target_value is not None else None
        if require_ground_truth and (not gt_ids or target_id is None):
            raise CircoSchemaError(
                f"ground-truth labels are unavailable for required query {query_id}"
            )
        if split == "val" and target_id != (gt_ids[0] if gt_ids else None):
            raise CircoSchemaError(
                f"CIRCO validation target must be the first ground truth for query {query_id}"
            )
        raw_aspects = entry.get("semantic_aspects", [])
        if not isinstance(raw_aspects, list) or not all(str(value).strip() for value in raw_aspects):
            raise CircoSchemaError(f"semantic_aspects must be a list of strings for query {query_id}")
        queries.append(
            ComposedQuery(
                query_id=query_id,
                split=split,
                reference_image_id=normalize_image_id(entry["reference_img_id"]),
                modification_text=str(entry["relative_caption"]).strip(),
                target_image_id=target_id,
                ground_truth_image_ids=frozenset(gt_ids),
                semantic_aspects=tuple(str(value) for value in raw_aspects),
                shared_concept=_string_or_none(entry.get("shared_concept")),
            )
        )
    return tuple(queries)


def load_circo_gallery(info_path: Path | str) -> CircoGallery:
    """Load `image_info_unlabeled2017.json` from the official COCO release."""

    raw = _load_json(Path(info_path))
    entries = raw.get("images") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise CircoSchemaError("COCO unlabeled image info must contain a non-empty images list")
    filenames: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry or "file_name" not in entry:
            raise CircoSchemaError("COCO unlabeled image info has an invalid image entry")
        image_id = normalize_image_id(entry["id"])
        filename = str(entry["file_name"]).strip()
        if not filename or Path(filename).name != filename:
            raise CircoSchemaError(f"invalid COCO unlabeled filename for image {image_id}")
        if image_id in filenames:
            raise CircoSchemaError(f"duplicate COCO unlabeled image ID: {image_id}")
        filenames[image_id] = filename
    return CircoGallery(tuple(sorted(filenames)), filenames)


def validate_queries_against_gallery(
    queries: tuple[ComposedQuery, ...], gallery: CircoGallery
) -> dict[str, Any]:
    """Verify every reference and labeled target exists in the candidate corpus."""

    missing_references = sorted({query.reference_image_id for query in queries if query.reference_image_id not in gallery.filenames})
    missing_targets = sorted({image_id for query in queries for image_id in query.ground_truth_image_ids if image_id not in gallery.filenames})
    if missing_references or missing_targets:
        raise CircoSchemaError(
            f"CIRCO queries reference absent gallery images: references={missing_references[:5]}, "
            f"targets={missing_targets[:5]}"
        )
    return {
        "query_count": len(queries),
        "gallery_count": len(gallery.image_ids),
        "queries_with_multiple_ground_truths": sum(len(query.ground_truth_image_ids) > 1 for query in queries),
        "mean_ground_truth_count": sum(len(query.ground_truth_image_ids) for query in queries) / len(queries) if queries else 0.0,
        "reference_images_excluded_from_ground_truth": True,
    }


def split_query_ids(
    queries: tuple[ComposedQuery, ...], seed: int, selection_fraction: float
) -> tuple[tuple[ComposedQuery, ...], tuple[ComposedQuery, ...]]:
    """Create a deterministic labeled selection/holdout split inside CIRCO val."""

    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be strictly between zero and one")
    import hashlib

    ordered = sorted(
        queries,
        key=lambda query: hashlib.sha256(f"{seed}\0{query.query_id}".encode()).hexdigest(),
    )
    selection_count = min(max(1, round(len(ordered) * selection_fraction)), len(ordered) - 1)
    selection_ids = {query.query_id for query in ordered[:selection_count]}
    selection = tuple(query for query in queries if query.query_id in selection_ids)
    holdout = tuple(query for query in queries if query.query_id not in selection_ids)
    return selection, holdout
