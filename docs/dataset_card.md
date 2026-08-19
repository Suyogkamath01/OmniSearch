# MS COCO 2017 val2017 dataset card

Status: **primary dataset acquired, validated, split, and used for real Phase 2–6 runs**.

## Source and access method

The active benchmark is the complete official `val2017` image release and its official caption annotations. The source page is the [COCO download page](https://cocodataset.org/dataset/download.htm); usage terms are on the [COCO terms page](https://cocodataset.org/dataset/termsofuse.htm).

The exact local scope is `coco2017_val`, not the full COCO train/val corpus:

- image archive: `http://images.cocodataset.org/zips/val2017.zip`
- caption archive: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
- extracted captions: `data/raw/coco2017/annotations/captions_val2017.json`
- extracted images: `data/raw/coco2017/val2017/`

Both official pages and both official archives returned HTTP 200 during the pre-download check. The archive hashes are recorded in the manifest metadata and Phase 1 artifacts.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `val2017.zip` | 815,585,330 | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `annotations_trainval2017.zip` | 252,907,541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |

## License and usage notes

COCO states that its annotations and website are available under CC BY 4.0, but that COCO does not own the copyrights to the images. Image use therefore remains subject to the originating Flickr terms and applicable rights. The repository keeps metadata, manifests, checksums, and local artifacts; image bytes are ignored by version control and are not redistributed.

People reproducing the project must obtain COCO from the official source and
accept the source terms themselves. This repository does not grant rights to
the underlying photographs, does not bundle a redistribution copy, and does
not treat a successful download as proof that every image is free of
third-party restrictions. The same rule applies to any future Flickr30k or
CIRCO acquisition.

## Local size and caption structure

The manifest contains 5,000 unique image IDs and 25,014 caption records. The official annotation file is nominally five captions per image, with the actual local distribution measured as:

| Captions/image | Image groups |
|---:|---:|
| 5 | 4,987 |
| 6 | 12 |
| 7 | 1 |

No image group is missing captions. There are 133 normalized duplicate-caption groups; duplicates are reported and retained because repeated captions are source data, not automatically invalid records.

## Split and tiers

The official source partition is `val2017`. Because it has no caption-retrieval train/validation/test partition, the project creates an internal deterministic image-grouped split using SHA-256 ordering and seed 42:

| Internal split | Images | Captions |
|---|---:|---:|
| Train | 4,000 | 20,011 |
| Validation | 500 | 2,502 |
| Test | 500 | 2,501 |

All captions for an image stay in one split. The leakage checker verifies that neither image IDs nor caption IDs cross boundaries.

| Tier | Image groups | Purpose |
|---|---:|---|
| Tier 1 | 100 | smoke/correctness checks |
| Tier 2 | 1,000 | student-GPU/limited-compute experiments |
| Tier 3 | 5,000 | largest locally feasible COCO configuration |

Phase 7's executed Tier 2 configuration uses the canonical split manifest and
selects 800 train, 100 validation, and 100 test image groups by the same
seeded image-ID ordering. This keeps the total scope at 1,000 groups while
retaining all three split roles; it is recorded separately from the historical
`tiers/tier2.json` source-tier artifact.

## Verified Phase 1 results

Pillow decoded all 5,000 local images. The validation report found zero missing, unreadable, or corrupted images and zero exact byte-duplicate image groups. Exact duplicate detection is SHA-256 over file bytes; visually identical images encoded with different bytes are not detected. Perceptual near-duplicate detection was not run.

The canonical artifacts are [validation](../artifacts/coco_phase1/validation.json) and [statistics](../artifacts/coco_phase1/statistics.json). The generated split manifest at `data/processed/coco2017_val_split_manifest.json` remains local-only because it is a generated dataset-scale file.

## Historical Flickr30k record

The original official Flickr30k caption metadata, manifest, audits, and metadata-only artifacts remain in the repository. Its official Box image path was unavailable during preflight, so Flickr30k is no longer required for the primary benchmark and is retained as an optional future external validation set. Historical Flickr artifacts are not relabeled as COCO.

## Known limitations

- `val2017` is a source validation partition, not an official retrieval test partition; the internal split is a reproducible project protocol.
- The benchmark is limited to 5,000 images because full COCO train images are not appropriate for this host's current storage/compute budget.
- Caption relevance is metadata-defined as all captions belonging to the same image; it is not human semantic judgment.
- Exact hashing misses visually equivalent files with different encodings.
- COCO does not grant image copyright ownership; use remains rights-sensitive.
- The source image population and captions have known demographic, cultural, and annotation biases that aggregate retrieval metrics cannot resolve.
