# Phase 0 compute and environment audit

Audit date: 2026-08-16 (local time, Asia/Kolkata)

## Observed environment

The following facts were measured from the workspace host during Phase 0:

| Item | Observed value |
|---|---|
| Operating system | macOS Darwin 25.5.0, arm64 |
| Machine | MacBook Air, Mac15,12 |
| CPU | Apple M3, 8 CPU cores (4 performance, 4 efficiency) |
| Integrated GPU | Apple M3 GPU, 8 cores, Metal 4 |
| Unified memory | 8 GB |
| Free disk after COCO acquisition | approximately 8.5 GiB on a 228 GiB volume |
| Default `python3` | CPython 3.14.4 |
| Project-compatible interpreter available | CPython 3.12.12 through `uv` |
| Package manager | `uv` 0.9.13 |
| Git | 2.46.2 |
| ML packages detected | none of torch, transformers, datasets, NumPy, pandas, Pillow, scikit-learn, FAISS, hnswlib, Streamlit, FastAPI, or pytest |
| Repository state at start | empty directory, not a Git repository |

The GPU is hardware-present, but MPS availability and model throughput were
**not measured in this Phase 0 baseline** because PyTorch was not installed.
Later measured MPS runs are recorded in the Phase 4 and Phase 6 artifacts; no
performance or training claim follows from the Phase 0 hardware description.

## Constraints and implications

1. Eight GB of unified memory makes frozen encoders, small batches, image-size limits, and cache discipline the default. Full fine-tuning of large vision-language models is not a local assumption.
2. Approximately 18 GiB free disk is a hard planning constraint. Raw images, model caches, generated embeddings, and indexes must not all be retained at maximum size. Phase 1 must measure actual download size before committing to a full split.
3. Python 3.14 is not the project target. The repository pins the development target to Python 3.12 because that interpreter is already available and is the safer compatibility baseline for PyTorch and related packages. This is a compatibility decision, not a claim about every package's future support.
4. No third-party packages are installed, so Phase 0 uses only the Python standard library. Installing the research extra is deferred until Phase 1 and should be done in an isolated `uv` environment.

## Experiment tiers

| Tier | Purpose | Planned scope | Local status |
|---|---|---|---|
| Tier 1 | correctness/smoke test | 100 image groups, frozen features, tiny batches, deterministic checks | manifest generated; smoke tier available |
| Tier 2 | student research | 1,000 COCO image groups, frozen CLIP/transformer encoders, one or a few seeds, CPU or MPS where stable | manifest generated; full Phase 4–6 run used Tier 3 |
| Tier 3 | largest local comparison | 5,000 official COCO val2017 image groups and 25,014 captions | acquired and executed through Phase 6 |

Tier 3 support must never be reported as Tier 3 execution. Run metadata will record the actual device, package versions, split checksum, seed, and artifact paths.

## Resource policy

- Do not download any dataset automatically from import, test, or application startup.
- Use an explicit data-acquisition command in Phase 1.
- Keep original source URLs and checksums in manifests; do not commit image bytes or model weights.
- Prefer feature caches over duplicate image copies.
- Fail early when free disk is below a configured safety margin.
- Record actual wall-clock time and peak memory for benchmarks; never infer them from hardware specifications.
