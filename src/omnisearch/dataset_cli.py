"""Dataset acquisition, manifest, validation, split, and tier CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .acquisition import (
    OFFICIAL_CAPTION_HTML_URL,
    build_manifest,
    download_official_metadata,
    parse_caption_html,
    preflight_flickr30k,
)
from .coco_acquisition import (
    COCO_CAPTION_ANNOTATIONS_URL,
    build_coco_manifest,
    parse_coco_captions,
    preflight_coco,
)
from .image_validation import validate_image_records
from .manifest import (
    DatasetManifest,
    dataset_statistics,
    read_manifest,
    validate_manifest,
    write_manifest,
    write_validation_report,
)
from .splitting import assert_no_split_leakage, assign_image_grouped_splits, select_tier


def _write_json(value: object, path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _manifest_with_records(manifest: DatasetManifest, records) -> DatasetManifest:
    return replace(manifest, records=tuple(records))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset acquisition and Phase 1 validation workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="check official terms and endpoint headers")
    subparsers.add_parser(
        "preflight-coco", help="check official COCO pages and archive headers"
    )

    acquire = subparsers.add_parser(
        "acquire-metadata", help="download official caption metadata only"
    )
    acquire.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser(
        "build-manifest", help="parse official caption HTML into a JSON manifest"
    )
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--source-last-modified",
        default=None,
        help="Stable source snapshot marker from acquisition preflight/metadata artifact.",
    )

    coco_build = subparsers.add_parser(
        "build-coco-manifest", help="parse official COCO captions into a manifest"
    )
    coco_build.add_argument("--annotations", type=Path, required=True)
    coco_build.add_argument("--images-archive", type=Path, required=True)
    coco_build.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", help="validate schema/captions and optionally local images"
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--image-root", type=Path, default=None)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--require-images", action="store_true")

    split = subparsers.add_parser(
        "split", help="assign deterministic image-grouped splits"
    )
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)

    stats = subparsers.add_parser(
        "stats", help="write minimum acquisition/split statistics"
    )
    stats.add_argument("--manifest", type=Path, required=True)
    stats.add_argument("--output", type=Path, required=True)

    tiers = subparsers.add_parser("tiers", help="write deterministic tier manifests")
    tiers.add_argument("--manifest", type=Path, required=True)
    tiers.add_argument("--output-dir", type=Path, required=True)
    tiers.add_argument("--seed", type=int, default=42)
    tiers.add_argument("--include-unresolved", action="store_true")

    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight_flickr30k(), indent=2))
        return 0
    if args.command == "preflight-coco":
        print(json.dumps(preflight_coco(), indent=2))
        return 0
    if args.command == "acquire-metadata":
        print(json.dumps(download_official_metadata(args.output), indent=2))
        return 0
    if args.command == "build-manifest":
        records = parse_caption_html(args.source, OFFICIAL_CAPTION_HTML_URL)
        manifest = build_manifest(
            records, args.source, source_last_modified=args.source_last_modified
        )
        digest = write_manifest(manifest, args.output)
        print(
            json.dumps({"records": len(records), "manifest_sha256": digest}, indent=2)
        )
        return 0
    if args.command == "build-coco-manifest":
        records = parse_coco_captions(args.annotations)
        manifest = build_coco_manifest(
            records,
            args.annotations,
            args.images_archive,
            annotation_url=COCO_CAPTION_ANNOTATIONS_URL,
        )
        digest = write_manifest(manifest, args.output)
        print(
            json.dumps(
                {
                    "dataset_id": manifest.dataset_id,
                    "records": len(records),
                    "captions": sum(len(record.captions) for record in records),
                    "manifest_sha256": digest,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "validate":
        manifest = read_manifest(args.manifest)
        expected = manifest.metadata.get("expected_captions_per_image", 5)
        report = validate_manifest(
            manifest.records,
            expected_captions_per_image=int(expected) if expected is not None else None,
            allowed_image_extensions={".jpg", ".jpeg", ".png", ".gif", ".webp"},
        )
        image_report = validate_image_records(manifest.records, args.image_root)
        output = {"manifest": report.to_dict(), "images": image_report.to_dict()}
        write_validation_report(report, args.output)
        image_report_path = args.output.with_name(args.output.stem + "_images.json")
        _write_json(image_report.to_dict(), image_report_path)
        print(json.dumps(output, indent=2))
        images_ready = (
            image_report.status == "completed"
            and all(record.filename is not None for record in manifest.records)
            and not image_report.missing_image_ids
            and not image_report.unreadable_image_ids
            and not image_report.corrupted_image_ids
        )
        if not report.passed or (args.require_images and not images_ready):
            return 1
        return 0
    if args.command == "split":
        manifest = read_manifest(args.manifest)
        split_records = assign_image_grouped_splits(manifest.records, seed=args.seed)
        assert_no_split_leakage(split_records)
        digest = write_manifest(
            _manifest_with_records(manifest, split_records), args.output
        )
        print(
            json.dumps(
                {
                    "split_counts": dataset_statistics(split_records)[
                        "split_image_counts"
                    ],
                    "manifest_sha256": digest,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "stats":
        manifest = read_manifest(args.manifest)
        expected = manifest.metadata.get("expected_captions_per_image", 5)
        report = validate_manifest(
            manifest.records,
            expected_captions_per_image=int(expected) if expected is not None else None,
            allowed_image_extensions={".jpg", ".jpeg", ".png", ".gif", ".webp"},
        )
        _write_json(dataset_statistics(manifest.records, report), args.output)
        return 0
    if args.command == "tiers":
        manifest = read_manifest(args.manifest)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for tier in ("tier1", "tier2", "tier3"):
            selected = select_tier(
                manifest.records,
                tier,
                args.seed,
                include_unresolved=args.include_unresolved,
            )
            tier_metadata = dict(manifest.metadata)
            tier_metadata.update(
                {
                    "tier": tier,
                    "tier_seed": args.seed,
                    "tier_filter": "all_records"
                    if args.include_unresolved
                    else "source_image_id_available_only",
                }
            )
            tier_manifest = replace(
                manifest,
                records=selected,
                metadata=tier_metadata,
            )
            path = args.output_dir / f"{tier}.json"
            outputs[tier] = {
                "records": len(tier_manifest.records),
                "sha256": write_manifest(tier_manifest, path),
            }
        print(json.dumps(outputs, indent=2))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
