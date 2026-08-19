# Dataset decision

## Migration decision

The active primary benchmark is **official MS COCO 2017 `val2017` captions**. This migration was necessary because the official Flickr30k image archive linked by the Illinois distribution page returned HTTP 404. No unofficial mirror, scrape, or fabricated image set was used.

The repository preserves the historical Flickr30k metadata pipeline and artifacts. Flickr30k remains an optional future external-validation dataset if an authorized image archive becomes available.

## Why COCO is currently primary

COCO provides a documented official download path, official terms page, image bytes and caption annotations that were actually downloaded, and a manageable 5,000-image validation release for this machine. It supports the same image-caption group abstraction and multi-positive retrieval relevance needed by the existing experiments.

The local scope is deliberately explicit: `coco2017_val` means the complete official `val2017` release, with a project-defined deterministic internal split. It does not claim the size or coverage of full COCO.

## Candidate comparison

| Candidate | Strength | Main risk | Decision |
|---|---|---|---|
| MS COCO 2017 val2017 | Official access path; real local images and captions; 5,000-image feasible scope | Source val partition is not an official retrieval split; image rights remain separate | **Primary** |
| Flickr30k | Strong established caption-retrieval benchmark and historical project work | Authorized image archive unavailable through official path | Optional future external validation |
| Full MS COCO train/val | Larger and more statistically useful | Current storage/compute budget does not justify full image acquisition | Future tier if budget changes |
| Conceptual Captions/WIT | Larger or multilingual web-scale extensions | URL/provenance/rights and compute complexity | Compute-dependent extension |

## Acceptance criteria met

The active dataset has an official source and terms record, verified archive hashes, a schema-valid manifest, local image decoding results, duplicate and missingness reports, deterministic tiers, and image-grouped leakage checks.

The manifests retain original caption text and add conservative normalized text only for duplicate detection and later preprocessing. No model-specific image transform is part of the dataset layer.
