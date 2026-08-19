"""Official MS COCO 2017 validation-caption acquisition and manifest adapter.

This module intentionally handles only the official COCO endpoints.  It does
not scrape Flickr, use mirrors, or silently substitute another image source.
The project scope is the complete official ``val2017`` image/caption release;
the repository creates deterministic internal train/validation/test partitions
from those image groups for comparable retrieval experiments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .acquisition import sha256_file
from .manifest import CaptionRecord, DatasetManifest, ImageRecord

COCO_OFFICIAL_DATA_PAGE = "https://cocodataset.org/dataset/download.htm"
COCO_TERMS_URL = "https://cocodataset.org/dataset/termsofuse.htm"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_CAPTION_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_VAL_CAPTIONS_FILENAME = "captions_val2017.json"
COCO_IMAGE_LICENSE_NOTE = (
    "COCO annotations and website are CC BY 4.0; COCO does not own image "
    "copyrights, so image use remains subject to the originating Flickr terms."
)


@dataclass(frozen=True)
class CocoSourceCheck:
    name: str
    url: str
    status_code: int | None
    content_type: str | None
    content_length: int | None
    last_modified: str | None
    etag: str | None
    accessible: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _head(url: str, name: str) -> CocoSourceCheck:
    request = Request(
        url, headers={"User-Agent": "OmniSearch/COCO-acquisition-preflight"}, method="HEAD"
    )
    try:
        with urlopen(request, timeout=30) as response:
            headers = response.headers
            length = headers.get("Content-Length")
            return CocoSourceCheck(
                name=name,
                url=url,
                status_code=getattr(response, "status", None),
                content_type=headers.get_content_type(),
                content_length=int(length) if length and length.isdigit() else None,
                last_modified=headers.get("Last-Modified"),
                etag=headers.get("ETag"),
                accessible=200 <= response.status < 400,
            )
    except HTTPError as exc:
        return CocoSourceCheck(
            name,
            url,
            exc.code,
            exc.headers.get_content_type(),
            None,
            exc.headers.get("Last-Modified"),
            exc.headers.get("ETag"),
            False,
            str(exc),
        )
    except (OSError, URLError) as exc:
        return CocoSourceCheck(name, url, None, None, None, None, None, False, str(exc))


def preflight_coco() -> dict[str, Any]:
    """Check only official COCO pages and archive headers before download."""

    checks = (
        _head(COCO_OFFICIAL_DATA_PAGE, "official_download_page"),
        _head(COCO_TERMS_URL, "official_terms"),
        _head(COCO_VAL_IMAGES_URL, "val2017_images_archive"),
        _head(COCO_CAPTION_ANNOTATIONS_URL, "trainval_caption_annotations_archive"),
    )
    return {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "MS COCO 2017 val2017 captions",
        "official_data_page": COCO_OFFICIAL_DATA_PAGE,
        "terms_url": COCO_TERMS_URL,
        "license_note": COCO_IMAGE_LICENSE_NOTE,
        "checks": [check.to_dict() for check in checks],
        "all_required_sources_accessible": all(check.accessible for check in checks),
    }


def download_coco_archive(output: Path | str, url: str) -> dict[str, Any]:
    """Download one allowlisted official archive atomically with provenance."""

    if url not in {COCO_VAL_IMAGES_URL, COCO_CAPTION_ANNOTATIONS_URL}:
        raise ValueError("download URL is not an allowlisted official COCO archive")
    check = _head(url, "requested_coco_archive")
    if not check.accessible:
        raise RuntimeError(f"official COCO archive is not accessible: {check.to_dict()}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    request = Request(url, headers={"User-Agent": "OmniSearch/COCO-acquisition"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
    temporary.replace(output_path)
    return {
        "url": url,
        "output": str(output_path),
        "sha256": sha256_file(output_path),
        "content_length": output_path.stat().st_size,
        "expected_content_length": check.content_length,
        "last_modified": check.last_modified,
        "etag": check.etag,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }


def parse_coco_captions(
    annotation_path: Path | str,
    image_root: Path | str | None = None,
) -> tuple[ImageRecord, ...]:
    """Convert official COCO caption JSON into the common image-caption schema."""

    with Path(annotation_path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise TypeError("COCO caption JSON must contain images and annotations lists")

    captions_by_image: dict[int, list[CaptionRecord]] = {}
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise TypeError("COCO annotation entry is not an object")
        image_id = int(annotation["image_id"])
        caption_id = str(annotation["id"])
        captions_by_image.setdefault(image_id, []).append(
            CaptionRecord(f"coco2017_val#{caption_id}", str(annotation["caption"]))
        )

    records: list[ImageRecord] = []
    for image in images:
        if not isinstance(image, Mapping):
            raise TypeError("COCO image entry is not an object")
        image_id = int(image["id"])
        captions = tuple(captions_by_image.get(image_id, ()))
        filename = str(image["file_name"])
        records.append(
            ImageRecord(
                image_id=str(image_id),
                filename=filename,
                captions=captions,
                image_url=f"http://images.cocodataset.org/val2017/{filename}",
                source_image_id_available=True,
            )
        )
    records.sort(key=lambda record: int(record.image_id))
    return tuple(records)


def build_coco_manifest(
    records: tuple[ImageRecord, ...] | list[ImageRecord],
    annotation_path: Path | str,
    image_archive_path: Path | str,
    annotation_url: str = COCO_CAPTION_ANNOTATIONS_URL,
) -> DatasetManifest:
    """Build a provenance-rich manifest for the downloaded COCO val release."""

    annotation_path = Path(annotation_path)
    image_archive_path = Path(image_archive_path)
    records = tuple(records)
    return DatasetManifest(
        dataset_id="coco2017_val",
        dataset_version="2017-val-captions",
        source_url=annotation_url,
        terms_url=COCO_TERMS_URL,
        source_snapshot_marker="COCO-2017-val-caption-release",
        source_sha256=sha256_file(annotation_path),
        records=records,
        metadata={
            "dataset_family": "MS COCO",
            "official_data_page": COCO_OFFICIAL_DATA_PAGE,
            "official_source_split": "val2017",
            "caption_structure": (
                "nominally five captions per image; the official val annotation "
                "file contains 4,987 images with 5, 12 with 6, and 1 with 7"
            ),
            "nominal_captions_per_image": 5,
            "expected_captions_per_image": None,
            "image_archive_url": COCO_VAL_IMAGES_URL,
            "image_archive_sha256": sha256_file(image_archive_path),
            "caption_annotation_url": annotation_url,
            "caption_annotation_sha256": sha256_file(annotation_path),
            "caption_license_note": COCO_IMAGE_LICENSE_NOTE,
            "image_rights_note": (
                "COCO does not own image copyrights; originating Flickr terms apply."
            ),
            "image_count": len(records),
            "caption_count": sum(len(record.captions) for record in records),
            "internal_split_protocol": "deterministic_sha256_image_group_split_seed_42",
        },
    )
