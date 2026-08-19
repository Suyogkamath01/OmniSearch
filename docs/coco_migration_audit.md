# COCO migration and Phase 0–6 audit

Audit date: 2026-08-16 (Asia/Kolkata)

This is the active foundation record after migrating the primary benchmark
from Flickr30k to official COCO. Historical phase checkpoint notes were
consolidated into the permanent data-validation, experiment, reproducibility,
and limitations documents; machine-readable artifacts remain intact.

## Foundation checklist

| Area | Result | Evidence |
|---|---|---|
| Phase 0 configuration/design | PASS | `configs/default.toml`, design/architecture/compute docs, bootstrap |
| Official COCO access/legal preflight | PASS | `preflight-coco` output; official download and terms pages; two HTTP 200 archive checks |
| Acquisition provenance | PASS | exact archive sizes and SHA-256 values in `coco2017_val_manifest.json` metadata |
| Manifest/schema validation | PASS | 5,000 unique image groups; 25,014 captions; no missing-caption groups or duplicate image IDs |
| Real image validation | PASS | Pillow decoded 5,000/5,000; zero missing, unreadable, corrupt, or exact duplicate groups |
| Image-grouped split/leakage gate | PASS | 4,000/500/500 image groups; image and caption split assertion passes |
| Tier reproducibility | PASS | 100 / 1,000 / 5,000 whole-image manifests with checksums |
| Phase 2 | PASS | real COCO metadata and image EDA in `artifacts/coco/phase2` |
| Phase 3 | PASS | real COCO TF-IDF/BM25 and RGB histogram baseline in `artifacts/coco/phase3` |
| Phase 4 | PASS | frozen CLIP on MPS; 500 images and 2,501 captions in `artifacts/coco/phase4` |
| Phase 5 | PASS | canonical `retrieval_eval_v1`; fresh COCO text and real CLIP ranking migration |
| Phase 6 | PASS | frozen text comparisons and 500-image ResNet/ViT/CLIP representation evidence |
| Documentation/tests | PASS | migration docs, dataset card, technical defense, 49 tests, Ruff, mypy |

## Measured COCO facts

- Official image archive: 815,585,330 bytes; SHA-256 `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`.
- Official annotation archive: 252,907,541 bytes; SHA-256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`.
- Manifest: 5,000 images and 25,014 captions.
- Caption cardinalities: 4,987 groups with 5, 12 with 6, and 1 with 7.
- Internal split: train 4,000/20,011, validation 500/2,502, test 500/2,501 (images/captions).
- Validation: 0 missing, 0 unreadable, 0 corrupted, 0 exact duplicate groups.
- Normalized duplicate captions: 133 groups, retained as warnings.

## Executed model evidence

Phase 4 real CLIP metrics on the internal test split:

| Task | Queries | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Text → image | 2,501 | 0.6182 | 0.8948 | 0.9580 | 0.7341 |
| Image → text | 500 | 0.7880 | 0.9560 | 0.9880 | 0.8606 |

Phase 5 text baseline migration (2,501 test-caption queries): TF-IDF R@1/R@10
`0.1102/0.4309`; BM25 R@1/R@10 `0.1141/0.4387`.

Phase 6 text R@1/R@10: MiniLM `0.1463/0.6330`, DistilBERT
`0.0916/0.3393`, and CLIP text `0.1473/0.5814`. Phase 6 also encoded all
500 real test images with each selected vision representation and preserved
nearest-neighbor evidence. Formal image-to-image metrics were not computed.

## Score

Foundation score: **95/100**.

The deduction is explicit rather than a model-performance judgment: 3 points
for using the complete feasible `val2017` source release with a project-defined
internal split instead of an official COCO retrieval split, and 2 points for
exact-only duplicate detection without perceptual near-duplicate analysis.

## Quality gate

**PASS for the COCO Phase 0–6 foundation.** The former authorized Flickr30k
image blocker no longer blocks the primary benchmark because the project has a
documented, official, actually downloaded COCO alternative. Phase 7–9 training
experiments were executed afterward and are summarized in
`docs/experiments.md`; this foundation record does not replace their detailed
machine-readable artifacts.

## Readiness decision

**READY FOR PHASE 7: YES (historical foundation decision)**

This means the active COCO foundation is ready for a separately audited Phase
7. It does not mean Phase 7 has started, and it does not waive the limitations
above.
