"""Image integrity checks with an optional Pillow decoder."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .manifest import ImageRecord


@dataclass(frozen=True)
class ImageFileCheck:
    image_id: str
    path: str
    exists: bool
    readable: bool
    decodable: bool
    method: str
    sha256: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageValidationReport:
    status: str
    image_root: str | None
    checks: tuple[ImageFileCheck, ...]
    missing_image_ids: tuple[str, ...]
    unreadable_image_ids: tuple[str, ...]
    corrupted_image_ids: tuple[str, ...]
    exact_duplicate_groups: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "image_root": self.image_root,
            "images_checked": len(self.checks),
            "checks": [check.to_dict() for check in self.checks],
            "missing_image_ids": list(self.missing_image_ids),
            "unreadable_image_ids": list(self.unreadable_image_ids),
            "corrupted_image_ids": list(self.corrupted_image_ids),
            "exact_duplicate_groups": self.exact_duplicate_groups,
        }


def _signature_is_plausible(data: bytes, suffix: str) -> bool:
    suffix = suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    if suffix == ".png":
        return (
            len(data) >= 33
            and data[:8] == b"\x89PNG\r\n\x1a\n"
            and data[12:16] == b"IHDR"
            and data[-8:-4] == b"IEND"
        )
    if suffix == ".gif":
        return (
            len(data) >= 14 and data[:6] in {b"GIF87a", b"GIF89a"} and data[-1:] == b";"
        )
    if suffix == ".webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return len(data) > 0


def _pillow_check(path: Path) -> tuple[bool, str | None] | None:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            image.verify()
        # `verify()` checks container structure but may not decode every pixel.
        # Reopen and load the image so truncated/corrupt payloads are caught.
        with Image.open(path) as image:
            image.load()
        return True, None
    except (OSError, ValueError, SyntaxError) as exc:
        return False, str(exc)


def check_image_file(image_id: str, path: Path | str) -> ImageFileCheck:
    path = Path(path)
    if not path.exists():
        return ImageFileCheck(
            image_id, str(path), False, False, False, "none", None, "missing file"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ImageFileCheck(
            image_id, str(path), True, False, False, "none", None, str(exc)
        )

    digest = hashlib.sha256(data).hexdigest()
    pillow_result = _pillow_check(path)
    if pillow_result is not None:
        decodable, error = pillow_result
        return ImageFileCheck(
            image_id, str(path), True, True, decodable, "pillow", digest, error
        )

    decodable = _signature_is_plausible(data, path.suffix)
    return ImageFileCheck(
        image_id,
        str(path),
        True,
        True,
        decodable,
        "signature",
        digest,
        None
        if decodable
        else "file signature or terminator is invalid; install Pillow for full decoding",
    )


def validate_image_records(
    records: Iterable[ImageRecord], image_root: Path | str | None
) -> ImageValidationReport:
    if image_root is None:
        return ImageValidationReport("not_run_no_image_root", None, (), (), (), (), {})

    root = Path(image_root)
    checks_list: list[ImageFileCheck] = []
    for record in records:
        if record.filename is None:
            checks_list.append(
                ImageFileCheck(
                    record.image_id,
                    "",
                    False,
                    False,
                    False,
                    "none",
                    None,
                    "source image ID unavailable in official metadata",
                )
            )
        else:
            checks_list.append(
                check_image_file(record.image_id, root / record.filename)
            )
    checks = tuple(checks_list)
    missing = tuple(check.image_id for check in checks if not check.exists)
    unreadable = tuple(
        check.image_id for check in checks if check.exists and not check.readable
    )
    corrupted = tuple(
        check.image_id for check in checks if check.readable and not check.decodable
    )
    by_hash: defaultdict[str, list[str]] = defaultdict(list)
    for check in checks:
        if check.sha256:
            by_hash[check.sha256].append(check.image_id)
    duplicate_groups = {
        digest: tuple(sorted(image_ids))
        for digest, image_ids in sorted(by_hash.items())
        if len(image_ids) > 1
    }
    return ImageValidationReport(
        "completed",
        str(root),
        checks,
        missing,
        unreadable,
        corrupted,
        duplicate_groups,
    )
