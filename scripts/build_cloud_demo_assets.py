"""Build a deterministic, public, zero-shot CLIP demo gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omnisearch.clip_baseline import encode_images, encode_texts, load_clip_runtime

MODEL_ID = "openai/clip-vit-base-patch32"
PRIORITY_TERMS = ("bicycle", "dog", "frisbee", "sitting at a table", "street")


def select_records(manifest_path: Path, image_limit: int) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    prioritized = [
        record
        for record in records
        if any(term in " ".join(caption["text"].lower() for caption in record["captions"]) for term in PRIORITY_TERMS)
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in [*prioritized, *records]:
        image_id = str(record["image_id"])
        if image_id in seen:
            continue
        selected.append(record)
        seen.add(image_id)
        if len(selected) == image_limit:
            return selected
    raise ValueError(f"manifest contains only {len(selected)} usable image groups")


def build_assets(manifest_path: Path, image_root: Path, output_dir: Path, image_limit: int) -> None:
    records = select_records(manifest_path, image_limit)
    runtime = load_clip_runtime(MODEL_ID, requested_device="auto")
    image_items = [(str(record["image_id"]), image_root / str(record["filename"])) for record in records]
    image_batch = encode_images(image_items, runtime, batch_size=32)
    if image_batch.skipped or len(image_batch.ids) != len(records):
        raise RuntimeError(f"image encoding skipped {len(image_batch.skipped)} records")
    caption_items = [
        (str(caption["caption_id"]), str(caption["text"]))
        for record in records
        for caption in record["captions"]
    ]
    caption_batch = encode_texts(caption_items, runtime, batch_size=64)
    if caption_batch.skipped or len(caption_batch.ids) != len(caption_items):
        raise RuntimeError(f"caption encoding skipped {len(caption_batch.skipped)} records")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "source": "COCO 2017 validation images and captions",
        "image_count": len(records),
        "caption_count": len(caption_items),
        "images": [
            {
                "image_id": str(record["image_id"]),
                "filename": str(record["filename"]),
                "image_url": f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{record['filename']}",
                "captions": [
                    {
                        "caption_id": str(caption["caption_id"]),
                        "image_id": str(record["image_id"]),
                        "text": str(caption["text"]),
                    }
                    for caption in record["captions"]
                ],
            }
            for record in records
        ],
    }
    (output_dir / "gallery.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "embeddings.npz",
        image_embeddings=np.asarray(image_batch.embeddings, dtype=np.float32),
        caption_embeddings=np.asarray(caption_batch.embeddings, dtype=np.float32),
    )
    print(json.dumps({"images": len(records), "captions": len(caption_items), "output": str(output_dir)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/coco2017_val_split_manifest.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/coco2017/val2017"))
    parser.add_argument("--output", type=Path, default=Path("assets/cloud_demo"))
    parser.add_argument("--image-limit", type=int, default=1_000)
    args = parser.parse_args()
    build_assets(args.manifest, args.image_root, args.output, args.image_limit)


if __name__ == "__main__":
    main()
