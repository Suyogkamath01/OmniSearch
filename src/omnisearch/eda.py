"""Reproducible Phase 2 exploratory data analysis.

This module is deliberately dependency-free. It analyses caption metadata and
split structure locally, and analyses image bytes only when an authorized
``--image-root`` is supplied. It never downloads images and never imports a
learned model.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import statistics
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config
from .image_validation import validate_image_records
from .manifest import (
    CaptionRecord,
    ImageRecord,
    read_manifest,
    validate_manifest,
)
from .splitting import SPLIT_NAMES

TOKEN_PATTERN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.UNICODE)
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "yours",
    ]
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": _quantile(values, 0.10),
        "p90": _quantile(values, 0.90),
    }


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text for descriptive statistics only."""

    return TOKEN_PATTERN.findall(text.casefold())


def _caption_rows(
    records: Iterable[ImageRecord],
) -> list[tuple[ImageRecord, CaptionRecord]]:
    return [(record, caption) for record in records for caption in record.captions]


def caption_statistics(
    records: Iterable[ImageRecord],
    short_threshold: int = 3,
    long_threshold: int = 30,
) -> dict[str, Any]:
    """Compute text statistics while retaining raw-caption provenance."""

    rows = _caption_rows(records)
    word_lengths: list[int] = []
    char_lengths: list[int] = []
    frequencies: Counter[str] = Counter()
    frequency_of_frequency: Counter[int] = Counter()
    stopword_count = 0
    numeric_token_count = 0
    punctuation_count = 0
    total_characters = 0
    empty_caption_ids: list[str] = []
    near_empty_caption_ids: list[str] = []
    unusual_character_caption_ids: list[str] = []
    short_caption_ids: list[str] = []
    long_caption_ids: list[str] = []

    for _, caption in rows:
        raw_text = caption.text
        normalized = raw_text.casefold()
        tokens = tokenize(normalized)
        word_length = len(tokens)
        char_length = len(raw_text)
        word_lengths.append(word_length)
        char_lengths.append(char_length)
        frequencies.update(tokens)
        total_characters += char_length
        stopword_count += sum(token in STOPWORDS for token in tokens)
        numeric_token_count += sum(
            any(character.isdigit() for character in token) for token in tokens
        )
        punctuation_count += sum(
            unicodedata_category(character).startswith("P") for character in raw_text
        )

        if not raw_text.strip():
            empty_caption_ids.append(caption.caption_id)
        if word_length <= 1:
            near_empty_caption_ids.append(caption.caption_id)
        if word_length < short_threshold:
            short_caption_ids.append(caption.caption_id)
        if word_length > long_threshold:
            long_caption_ids.append(caption.caption_id)
        if any(
            (not character.isprintable()) or (ord(character) > 127)
            for character in raw_text
        ):
            unusual_character_caption_ids.append(caption.caption_id)

    frequency_of_frequency.update(frequencies.values())
    total_tokens = sum(frequencies.values())
    top_tokens = [
        {"token": token, "count": count}
        for token, count in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )[:50]
    ]
    rare_tokens = sorted(token for token, count in frequencies.items() if count == 1)
    duplicate_groups = duplicate_caption_analysis(records)

    return {
        "caption_count": len(rows),
        "word_length": distribution(word_lengths),
        "character_length": distribution(char_lengths),
        "total_tokens": total_tokens,
        "vocabulary_size": len(frequencies),
        "lexical_diversity_type_token_ratio": (len(frequencies) / total_tokens)
        if total_tokens
        else None,
        "token_frequency_of_frequency": {
            str(count): frequency
            for count, frequency in sorted(frequency_of_frequency.items())
        },
        "top_tokens": top_tokens,
        "rare_token_count": len(rare_tokens),
        "rare_tokens_sample": rare_tokens[:100],
        "stopword_tokens": stopword_count,
        "stopword_fraction": (stopword_count / total_tokens) if total_tokens else None,
        "numeric_token_count": numeric_token_count,
        "numeric_token_caption_count": sum(
            any(
                any(character.isdigit() for character in token)
                for token in tokenize(caption.text)
            )
            for _, caption in rows
        ),
        "punctuation_character_count": punctuation_count,
        "punctuation_character_fraction": (punctuation_count / total_characters)
        if total_characters
        else None,
        "empty_caption_count": len(empty_caption_ids),
        "empty_caption_ids_sample": empty_caption_ids[:50],
        "near_empty_caption_count": len(near_empty_caption_ids),
        "near_empty_caption_ids_sample": near_empty_caption_ids[:50],
        "short_caption_threshold_words": short_threshold,
        "short_caption_count": len(short_caption_ids),
        "long_caption_threshold_words": long_threshold,
        "long_caption_count": len(long_caption_ids),
        "long_caption_ids_sample": long_caption_ids[:50],
        "unusual_character_caption_count": len(unusual_character_caption_ids),
        "unusual_character_caption_ids_sample": unusual_character_caption_ids[:50],
        "normalized_duplicate_caption_group_count": duplicate_groups["group_count"],
        "normalized_duplicate_caption_groups_across_images": duplicate_groups[
            "cross_image_group_count"
        ],
    }


def unicodedata_category(character: str) -> str:
    # Local import keeps the module's public surface small and makes the
    # category rule explicit in one place.
    import unicodedata

    return unicodedata.category(character)


def duplicate_caption_analysis(records: Iterable[ImageRecord]) -> dict[str, Any]:
    occurrences: defaultdict[str, list[dict[str, str | None]]] = defaultdict(list)
    for record in records:
        for caption in record.captions:
            normalized = caption.normalized_text
            if normalized:
                occurrences[normalized].append(
                    {
                        "image_id": record.image_id,
                        "caption_id": caption.caption_id,
                        "split": record.split,
                    }
                )

    groups = {
        text: values for text, values in sorted(occurrences.items()) if len(values) > 1
    }
    cross_image = {
        text: values
        for text, values in groups.items()
        if len({value["image_id"] for value in values}) > 1
    }
    cross_split = {
        text: values
        for text, values in groups.items()
        if len({value["split"] for value in values}) > 1
    }
    return {
        "group_count": len(groups),
        "cross_image_group_count": len(cross_image),
        "cross_split_group_count": len(cross_split),
        "groups_sample": [
            {"normalized_text": text, "occurrences": values}
            for text, values in list(groups.items())[:50]
        ],
        "cross_split_groups_sample": [
            {"normalized_text": text, "occurrences": values}
            for text, values in list(cross_split.items())[:50]
        ],
    }


def _jpeg_dimensions(data: bytes) -> tuple[int, int, int | None] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    position = 2
    sof_markers = (
        set(range(0xC0, 0xC4))
        | set(range(0xC5, 0xC8))
        | set(range(0xC9, 0xCC))
        | set(range(0xCD, 0xD0))
    )
    while position + 9 < len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[position : position + 2])[0]
        if segment_length < 2 or position + segment_length > len(data):
            break
        if marker in sof_markers and segment_length >= 8:
            height, width = struct.unpack(">HH", data[position + 3 : position + 7])
            channels = data[position + 7]
            return width, height, channels
        position += segment_length
    return None


def image_header_metadata(path: Path) -> dict[str, Any] | None:
    """Read dimensions/channels without a learned model or mandatory Pillow."""

    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 26:
        width, height = struct.unpack(">II", data[16:24])
        color_type = data[25]
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
        mode = {0: "L", 2: "RGB", 3: "P", 4: "LA", 6: "RGBA"}.get(color_type)
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "mode": mode,
            "method": "png_header",
        }
    if data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return {
            "width": width,
            "height": height,
            "channels": 3,
            "mode": "P",
            "method": "gif_header",
        }
    if (
        data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and len(data) >= 30
        and data[12:16] == b"VP8X"
    ):
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return {
            "width": width,
            "height": height,
            "channels": None,
            "mode": None,
            "method": "webp_header",
        }
    jpeg = _jpeg_dimensions(data)
    if jpeg:
        width, height, channels = jpeg
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "mode": (
                {1: "L", 3: "RGB", 4: "CMYK"}.get(channels)
                if channels is not None
                else None
            ),
            "method": "jpeg_header",
        }
    return None


def image_statistics(
    records: Iterable[ImageRecord],
    image_root: Path | str | None,
    small_dimension: int = 64,
    large_file_bytes: int = 10_000_000,
) -> dict[str, Any]:
    records = tuple(records)
    validation = validate_image_records(records, image_root)
    if image_root is None:
        return {
            "status": "not_run_no_image_root",
            "image_root": None,
            "records_expected": len(records),
            "images_decoded": 0,
            "width": distribution([]),
            "height": distribution([]),
            "aspect_ratio": distribution([]),
            "file_size_bytes": distribution([]),
            "modes": {},
            "channels": {},
            "small_dimension_threshold": small_dimension,
            "large_file_threshold_bytes": large_file_bytes,
            "small_image_count": 0,
            "large_file_count": 0,
            "extreme_aspect_ratio_count": 0,
            "missing_image_count": None,
            "unreadable_image_count": None,
            "corrupted_image_count": None,
            "exact_duplicate_group_count": None,
            "exact_duplicate_groups_sample": [],
            "near_duplicate_status": "not_evaluated_no_perceptual_hash",
        }

    root = Path(image_root)
    widths: list[float] = []
    heights: list[float] = []
    aspects: list[float] = []
    sizes: list[float] = []
    modes: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    small_count = 0
    large_count = 0
    extreme_count = 0
    observations: list[dict[str, Any]] = []
    checks_by_id = {check.image_id: check for check in validation.checks}
    for record in records:
        if record.filename is None:
            continue
        path = root / record.filename
        check = checks_by_id.get(record.image_id)
        if check is None or not check.decodable:
            continue
        try:
            metadata = image_header_metadata(path)
            size = path.stat().st_size
        except (OSError, ValueError):
            continue
        if not metadata or not metadata.get("width") or not metadata.get("height"):
            continue
        width = int(metadata["width"])
        height = int(metadata["height"])
        aspect = width / height
        widths.append(width)
        heights.append(height)
        aspects.append(aspect)
        sizes.append(size)
        mode = metadata.get("mode")
        channel_count = metadata.get("channels")
        if mode:
            modes[str(mode)] += 1
        if channel_count is not None:
            channels[str(channel_count)] += 1
        small_count += min(width, height) < small_dimension
        large_count += size > large_file_bytes
        extreme_count += aspect < 0.5 or aspect > 2.0
        observations.append(
            {
                "image_id": record.image_id,
                "split": record.split,
                "width": width,
                "height": height,
                "aspect_ratio": aspect,
                "file_size_bytes": size,
            }
        )

    return {
        "status": "completed",
        "image_root": str(root),
        "records_expected": len(records),
        "images_decoded": len(observations),
        "width": distribution(widths),
        "height": distribution(heights),
        "aspect_ratio": distribution(aspects),
        "file_size_bytes": distribution(sizes),
        "modes": dict(sorted(modes.items())),
        "channels": dict(sorted(channels.items())),
        "small_dimension_threshold": small_dimension,
        "large_file_threshold_bytes": large_file_bytes,
        "small_image_count": small_count,
        "large_file_count": large_count,
        "extreme_aspect_ratio_count": extreme_count,
        "missing_image_count": len(validation.missing_image_ids),
        "unreadable_image_count": len(validation.unreadable_image_ids),
        "corrupted_image_count": len(validation.corrupted_image_ids),
        "exact_duplicate_group_count": len(validation.exact_duplicate_groups),
        "exact_duplicate_groups_sample": [
            {"sha256": digest, "image_ids": list(image_ids)}
            for digest, image_ids in list(validation.exact_duplicate_groups.items())[
                :50
            ]
        ],
        "near_duplicate_status": "not_evaluated_no_perceptual_hash",
        "observations": observations[:1000],
    }


def split_analysis(
    records: Iterable[ImageRecord],
    image_root: Path | str | None,
    short_threshold: int,
    long_threshold: int,
    small_dimension: int,
    large_file_bytes: int,
) -> dict[str, Any]:
    records = tuple(records)
    output: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        split_records = tuple(record for record in records if record.split == split)
        output[split] = {
            "image_record_count": len(split_records),
            "caption_statistics": caption_statistics(
                split_records, short_threshold, long_threshold
            ),
            "image_statistics": image_statistics(
                split_records,
                image_root,
                small_dimension,
                large_file_bytes,
            ),
        }
    unassigned = tuple(record for record in records if record.split not in SPLIT_NAMES)
    output["unassigned"] = {
        "image_record_count": len(unassigned),
        "caption_statistics": caption_statistics(
            unassigned, short_threshold, long_threshold
        ),
    }
    return output


def leakage_reaudit(
    records: Iterable[ImageRecord], image_root: Path | str | None
) -> dict[str, Any]:
    """Independent split-overlap analysis; repeated language is not critical leakage."""

    records = tuple(records)
    split_ids = {
        split: {record.image_id for record in records if record.split == split}
        for split in SPLIT_NAMES
    }
    split_caption_ids = {
        split: {
            caption.caption_id
            for record in records
            if record.split == split
            for caption in record.captions
        }
        for split in SPLIT_NAMES
    }
    image_overlaps: list[dict[str, Any]] = []
    caption_id_overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            common_images = sorted(split_ids[left] & split_ids[right])
            common_caption_ids = sorted(
                split_caption_ids[left] & split_caption_ids[right]
            )
            if common_images:
                image_overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "count": len(common_images),
                        "sample": common_images[:20],
                    }
                )
            if common_caption_ids:
                caption_id_overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "count": len(common_caption_ids),
                        "sample": common_caption_ids[:20],
                    }
                )

    duplicate_captions = duplicate_caption_analysis(records)
    findings: list[dict[str, Any]] = []
    if image_overlaps:
        findings.append(
            {
                "severity": "CRITICAL",
                "type": "image_id_overlap",
                "details": image_overlaps,
            }
        )
    if caption_id_overlaps:
        findings.append(
            {
                "severity": "CRITICAL",
                "type": "caption_id_overlap",
                "details": caption_id_overlaps,
            }
        )
    if duplicate_captions["cross_split_group_count"]:
        findings.append(
            {
                "severity": "POTENTIAL/BENIGN",
                "type": "normalized_caption_overlap",
                "count": duplicate_captions["cross_split_group_count"],
                "details": "Repeated generic or identical language is not image leakage by itself.",
            }
        )

    exact_image_overlap: dict[str, Any] = {"status": "not_evaluated_no_image_root"}
    if image_root is not None:
        image_report = validate_image_records(records, image_root)
        hash_splits: defaultdict[str, set[str]] = defaultdict(set)
        checks_by_id = {check.image_id: check for check in image_report.checks}
        for record in records:
            check = checks_by_id.get(record.image_id)
            if check and check.sha256 and record.split:
                hash_splits[check.sha256].add(record.split)
        cross_split_hashes = {
            digest: sorted(splits)
            for digest, splits in hash_splits.items()
            if len(splits) > 1
        }
        exact_image_overlap = {
            "status": "completed",
            "cross_split_exact_hash_count": len(cross_split_hashes),
            "cross_split_exact_hashes_sample": list(cross_split_hashes.items())[:20],
        }
        if cross_split_hashes:
            findings.append(
                {
                    "severity": "CRITICAL",
                    "type": "exact_image_hash_overlap",
                    "details": exact_image_overlap,
                }
            )

    return {
        "image_id_overlap": image_overlaps,
        "caption_id_overlap": caption_id_overlaps,
        "exact_image_overlap": exact_image_overlap,
        "normalized_caption_overlap": {
            "cross_split_group_count": duplicate_captions["cross_split_group_count"],
            "classification": "POTENTIAL/BENIGN; repeated language is not automatically leakage",
            "examples": duplicate_captions["cross_split_groups_sample"],
        },
        "near_duplicate_image_overlap": "not_evaluated_no_perceptual_hash",
        "findings": findings,
    }


def structural_alignment(
    records: Iterable[ImageRecord], validation: Any
) -> dict[str, Any]:
    records = tuple(records)
    captions_without_record = 0
    caption_ids: set[str] = set()
    duplicate_caption_ids: list[str] = []
    for record in records:
        for caption in record.captions:
            if caption.caption_id in caption_ids:
                duplicate_caption_ids.append(caption.caption_id)
            caption_ids.add(caption.caption_id)
    return {
        "record_count": len(records),
        "caption_count": sum(len(record.captions) for record in records),
        "records_without_captions": len(validation.missing_caption_image_ids),
        "captions_without_valid_record": captions_without_record,
        "caption_count_anomaly_records": len(validation.caption_count_mismatches),
        "duplicate_caption_id_count": len(duplicate_caption_ids),
        "unresolved_source_image_id_count": len(
            validation.records_without_source_image_id
        ),
        "duplicate_caption_groups": len(validation.duplicate_caption_groups),
        "cross_image_duplicate_caption_groups": duplicate_caption_analysis(records)[
            "cross_image_group_count"
        ],
        "semantic_alignment_claim": "not_inferred_from structural metadata",
    }


def stable_sample(
    records: Iterable[ImageRecord], seed: int, sample_size: int
) -> list[dict[str, Any]]:
    records = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}\0{record.image_id}".encode()
        ).hexdigest(),
    )
    return [
        {
            "image_id": record.image_id,
            "filename": record.filename,
            "split": record.split,
            "source_image_id_available": record.source_image_id_available,
            "captions": [caption.text for caption in record.captions],
        }
        for record in records[:sample_size]
    ]


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(values)
    if not counts:
        return {}
    maximum = max(counts)
    if maximum <= 40:
        return {str(key): counts[key] for key in range(maximum + 1) if counts[key]}
    compact: Counter[str] = Counter()
    for key, value in counts.items():
        compact[str(key) if key < 40 else "40+"] += value
    return dict(sorted(compact.items(), key=lambda item: (item[0] == "40+", item[0])))


def _svg_bar_chart(
    values: Mapping[str, int],
    title: str,
    x_label: str,
    y_label: str,
    output: Path,
    max_bars: int = 40,
) -> None:
    items = list(values.items())[:max_bars]
    width, height = 960, 560
    left, right, top, bottom = 90, 30, 70, 120
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum = max((value for _, value in items), default=1)
    bar_width = chart_width / max(len(items), 1)
    bars: list[str] = []
    labels: list[str] = []
    for index, (label, value) in enumerate(items):
        x = left + index * bar_width + 2
        bar_height = chart_height * value / maximum
        y = top + chart_height - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 4, 1):.2f}" height="{bar_height:.2f}" fill="#285f8f"/>'
        )
        if len(items) <= 20 or index % max(1, len(items) // 20) == 0:
            labels.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{height - bottom + 22}" text-anchor="middle" font-size="11">{html.escape(str(label))}</text>'
            )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-family="sans-serif">{html.escape(title)}</text>
<line x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}" stroke="#222"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>
{"".join(bars)}
{"".join(labels)}
<text x="{width / 2}" y="{height - 28}" text-anchor="middle" font-size="14" font-family="sans-serif">{html.escape(x_label)}</text>
<text x="20" y="{height / 2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 20 {height / 2})">{html.escape(y_label)}</text>
</svg>
'''
    output.write_text(svg, encoding="utf-8")


def generate_figures(report: Mapping[str, Any], output_dir: Path) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    caption = report["caption_statistics"]
    figures: list[str] = []
    word_hist = {
        key: int(value)
        for key, value in _histogram_from_distribution(
            report["caption_statistics"], "word_length_histogram"
        ).items()
    }
    # The complete histogram is stored separately in the report; these SVGs
    # are small deterministic summaries of the actual observed metadata.
    _svg_bar_chart(
        word_hist,
        "Caption word-length distribution",
        "Words per caption",
        "Captions",
        figure_dir / "caption_word_length.svg",
    )
    figures.append("figures/caption_word_length.svg")
    _svg_bar_chart(
        {item["token"]: item["count"] for item in caption["top_tokens"][:20]},
        "Most frequent caption tokens",
        "Token",
        "Occurrences",
        figure_dir / "top_tokens.svg",
        max_bars=20,
    )
    figures.append("figures/top_tokens.svg")
    _svg_bar_chart(
        {
            str(key): value
            for key, value in report["dataset_statistics"]["captions_per_image"].items()
        },
        "Captions per image-group",
        "Captions per group",
        "Image groups",
        figure_dir / "captions_per_image.svg",
        max_bars=20,
    )
    figures.append("figures/captions_per_image.svg")

    image = report["image_statistics"]
    if image["status"] == "completed":
        for field, title, label in (
            ("width", "Image width distribution", "Width (pixels)"),
            ("height", "Image height distribution", "Height (pixels)"),
            ("aspect_ratio", "Image aspect-ratio distribution", "Aspect ratio"),
        ):
            values = image[field]
            if values["count"]:
                # Continuous image values are summarized with a compact
                # min/median/max chart rather than invented raw bins.
                bars = {
                    "min": int(values["min"]),
                    "median": int(values["median"]),
                    "max": int(values["max"]),
                }
                filename = f"image_{field}.svg"
                _svg_bar_chart(
                    bars,
                    title,
                    label,
                    "Observed value",
                    figure_dir / filename,
                    max_bars=3,
                )
                figures.append(f"figures/{filename}")
    return figures


def _histogram_from_distribution(report: Mapping[str, Any], key: str) -> dict[str, int]:
    # Kept as a separate helper so the report schema can hold a full histogram
    # without coupling the SVG writer to internal counters.
    return {str(k): int(v) for k, v in report.get(key, {}).items()}


def _with_histograms(
    caption_stats: dict[str, Any], records: Iterable[ImageRecord]
) -> dict[str, Any]:
    rows = _caption_rows(records)
    word_lengths = [len(tokenize(caption.text)) for _, caption in rows]
    char_lengths = [len(caption.text) for _, caption in rows]
    caption_stats = dict(caption_stats)
    caption_stats["word_length_histogram"] = _histogram(word_lengths)
    caption_stats["character_length_histogram"] = _histogram(char_lengths)
    return caption_stats


def _markdown_report(report: Mapping[str, Any], figure_paths: list[str]) -> str:
    dataset_id = report["provenance"]["dataset_id"]
    local = report["locally_verified_information"]
    image = report["image_statistics"]
    leakage = report["leakage_reaudit"]
    lines = [
        "# OmniSearch Phase 2 EDA and data-quality report",
        "",
        f"Generated: `{report['provenance']['generated_at_utc']}`",
        "",
        "## Scope and truth status",
        "",
        f"- Real metadata/text EDA: **{report['scope']['real_metadata_eda']}**",
        f"- Real image EDA: **{report['scope']['real_image_eda']}**",
        f"- Learned model used: **{report['scope']['learned_model_used']}**",
        f"- Qualitative image gallery: **{report['qualitative_gallery']['status']}**",
        "",
        "## Published dataset information",
        "",
        "Published counts are context only and are not substituted for local measurements.",
        "",
        f"- Reported image count: `{report['published_dataset_information']['reported_image_count']}`",
        f"- Reported caption count: `{report['published_dataset_information']['reported_caption_count']}`",
        "",
        "## Locally verified metadata",
        "",
        f"- Caption groups analysed: `{local['caption_groups_analysed']}`",
        f"- Captions analysed: `{local['captions_analysed']}`",
        f"- Source image IDs present: `{local['source_image_ids_available']}`",
        f"- Unresolved source-image groups: `{local['unresolved_source_image_groups']}`",
        f"- Schema-valid records: `{local['schema_valid_records']}`",
        "",
        "## Image analysis",
        "",
        f"- Status: **{image['status']}**",
        f"- Images decoded: `{image['images_decoded']}`",
        f"- Missing/corrupted/unreadable counts: `{image['missing_image_count']}` / `{image['corrupted_image_count']}` / `{image['unreadable_image_count']}`",
        f"- Near-duplicate analysis: `{image['near_duplicate_status']}`",
        "",
        "## Leakage re-audit",
        "",
        f"- Critical findings: `{sum(finding['severity'] == 'CRITICAL' for finding in leakage['findings'])}`",
        f"- Normalized caption overlap: `{leakage['normalized_caption_overlap']['cross_split_group_count']}` groups, classified as potential/benign linguistic overlap unless image IDs also overlap",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- [{path}]({path})" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            f"Image EDA is reported for `{dataset_id}` only when a local image root was supplied. Lexical statistics cannot establish semantic image-caption alignment.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_eda(
    manifest_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = "artifacts/phase2",
    image_root: Path | str | None = None,
    seed: int | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Run the canonical Phase 2 analysis and write JSON/Markdown/SVG artifacts."""

    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    manifest = read_manifest(manifest_path)
    records = manifest.records
    actual_seed = config.seed if seed is None else seed
    eda_config = _read_eda_config(config_path)
    actual_sample_size = (
        sample_size if sample_size is not None else int(eda_config["sample_size"])
    )
    expected_captions = manifest.metadata.get("expected_captions_per_image", 5)
    validation = validate_manifest(
        records,
        expected_captions_per_image=(
            int(expected_captions) if expected_captions is not None else None
        ),
        allowed_image_extensions={".jpg", ".jpeg", ".png", ".gif", ".webp"},
    )
    overall_caption_stats = _with_histograms(
        caption_statistics(
            records,
            int(eda_config["short_caption_words"]),
            int(eda_config["long_caption_words"]),
        ),
        records,
    )
    image_report = image_statistics(
        records,
        image_root,
        int(eda_config["small_image_dimension"]),
        int(eda_config["large_file_bytes"]),
    )
    splits = split_analysis(
        records,
        image_root,
        int(eda_config["short_caption_words"]),
        int(eda_config["long_caption_words"]),
        int(eda_config["small_image_dimension"]),
        int(eda_config["large_file_bytes"]),
    )
    for split in SPLIT_NAMES:
        if split in splits:
            splits[split]["caption_statistics"] = _with_histograms(
                splits[split]["caption_statistics"],
                tuple(record for record in records if record.split == split),
            )
    dataset_stats = {
        "records": len(records),
        "caption_groups": len(records),
        "captions": sum(len(record.captions) for record in records),
        "captions_per_image": Counter(len(record.captions) for record in records),
        "split_image_counts": Counter(
            record.split or "unassigned" for record in records
        ),
        "split_caption_counts": {
            split: sum(
                len(record.captions) for record in records if record.split == split
            )
            for split in SPLIT_NAMES
        },
    }
    dataset_stats = json.loads(
        json.dumps(dataset_stats, default=lambda value: dict(value))
    )
    leakage = leakage_reaudit(records, image_root)
    alignment = structural_alignment(records, validation)
    samples = stable_sample(records, actual_seed, actual_sample_size)
    provenance = {
        "project": "OmniSearch",
        "package": "omnisearch",
        "project_version": __version__,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "source_sha256": manifest.source_sha256,
        "seed": actual_seed,
        "sample_size": actual_sample_size,
        "image_root": str(image_root) if image_root is not None else None,
    }
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "provenance": provenance,
        "scope": {
            "real_metadata_eda": True,
            "real_image_eda": image_report["status"] == "completed"
            and image_report["images_decoded"] > 0,
            "learned_model_used": False,
        },
        "published_dataset_information": {
            "reported_image_count": manifest.metadata.get(
                "image_count", len(records)
            ),
            "reported_caption_count": manifest.metadata.get(
                "caption_count", sum(len(record.captions) for record in records)
            ),
            "source": manifest.source_url,
        },
        "locally_verified_information": {
            "caption_groups_analysed": len(records),
            "captions_analysed": sum(len(record.captions) for record in records),
            "source_image_ids_available": sum(
                record.source_image_id_available for record in records
            ),
            "unresolved_source_image_groups": sum(
                not record.source_image_id_available for record in records
            ),
            "schema_valid_records": len(records) if validation.passed else None,
            "validation_passed": validation.passed,
        },
        "dataset_statistics": dataset_stats,
        "caption_statistics": overall_caption_stats,
        "image_statistics": image_report,
        "structural_alignment": alignment,
        "split_analysis": splits,
        "leakage_reaudit": leakage,
        "qualitative_gallery": {
            "status": "not_generated_no_image_root"
            if image_root is None
            else "metadata_sample_only",
            "sample_selection": samples,
            "note": "No image gallery is claimed without real image bytes.",
        },
        "data_quality": {
            "schema_errors": list(validation.errors),
            "schema_warnings": list(validation.warnings),
            "missing_image_count": image_report["missing_image_count"],
            "corrupted_image_count": image_report["corrupted_image_count"],
            "unreadable_image_count": image_report["unreadable_image_count"],
            "exact_duplicate_image_group_count": image_report[
                "exact_duplicate_group_count"
            ],
            "near_duplicate_status": image_report["near_duplicate_status"],
            "caption_anomalies": {
                "empty": overall_caption_stats["empty_caption_count"],
                "near_empty": overall_caption_stats["near_empty_caption_count"],
                "short": overall_caption_stats["short_caption_count"],
                "long": overall_caption_stats["long_caption_count"],
                "unusual_characters": overall_caption_stats[
                    "unusual_character_caption_count"
                ],
            },
        },
    }
    figure_paths = generate_figures(report, output_dir)
    report["figures"] = figure_paths
    (output_dir / "phase2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "phase2_report.md").write_text(
        _markdown_report(report, figure_paths), encoding="utf-8"
    )
    (output_dir / "sample_selection.json").write_text(
        json.dumps(
            {"provenance": provenance, "samples": samples}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _read_eda_config(config_path: Path) -> dict[str, int | str]:
    import tomllib

    with config_path.open("rb") as file:
        raw = tomllib.load(file)
    eda = raw.get("eda", {})
    return {
        "short_caption_words": int(eda.get("short_caption_words", 3)),
        "long_caption_words": int(eda.get("long_caption_words", 30)),
        "small_image_dimension": int(eda.get("small_image_dimension", 64)),
        "large_file_bytes": int(eda.get("large_file_bytes", 10_000_000)),
        "sample_size": int(eda.get("sample_size", 12)),
        "near_duplicate_status": str(
            eda.get("near_duplicate_status", "not_evaluated_no_perceptual_hash")
        ),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OmniSearch Phase 2 metadata/image EDA."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/coco2017_val_split_manifest.json"),
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()
    report = run_eda(
        manifest_path=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        image_root=args.image_root,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "real_metadata_eda": report["scope"]["real_metadata_eda"],
                "real_image_eda": report["scope"]["real_image_eda"],
                "caption_groups": report["locally_verified_information"][
                    "caption_groups_analysed"
                ],
                "captions": report["locally_verified_information"]["captions_analysed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
