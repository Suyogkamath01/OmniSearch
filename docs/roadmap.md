# Implementation roadmap

This roadmap is deliberately incremental. A phase is complete only when its implementation, tests, experiments, artifacts, and limitations are reported honestly.

**Current status (2026-08-18):** Phases 1–27 are complete for the declared
scopes. Phase 12B is implemented and formally closed **PARTIAL / NON-BLOCKING
/ STORAGE-DEFERRED** because its official CIRCO gallery could not fit safely
on the local volume. No CIRCO performance claim is made. Phase 27 is the final
planned project phase; it completed the documentation, portfolio material,
claims, links, and release-file audit without changing the Git index.
Historical Flickr30k work is
preserved. Phase 9 includes a train-only static hard-negative manifest and
explicit false-negative limitations. Phase 10 includes exact, FAISS, and
hnswlib vector benchmarks at 100/1,000/5,000-image tiers. Permanent experiment
and reproducibility records are in `docs/experiments.md` and
`docs/reproducibility.md`.

| Phase | Deliverable | Classification | Completion evidence |
|---:|---|---|---|
| 0 | research design, compute audit, architecture, bootstrap | CORE | repository check passes; design documents exist; no unexecuted result is claimed |
| 1 | dataset acquisition, validation, preprocessing, leakage checks | CORE | official COCO val2017 downloaded; 5,000 images/25,014 captions; Pillow validation, checksums, tiers, and leakage gate pass |
| 2 | multimodal EDA | CORE | real COCO metadata and image EDA executed; zero missing/corrupt/unreadable images; exact duplicate scan complete |
| 3 | classical text/image baselines | CORE | TF-IDF/BM25 and handcrafted image baseline executed on real COCO test data |
| 4 | frozen zero-shot CLIP baseline | CORE | frozen ViT-B/32 MPS run completed on 500 COCO test images and 2,501 captions |
| 5 | unified retrieval evaluator | CORE | `retrieval_eval_v1` migrated fresh COCO Phase 3 and real Phase 4 rankings; fixture history retained |
| 6 | transformer text/vision representation experiments | CORE | frozen MiniLM/DistilBERT/CLIP text and real 500-image ResNet/ViT/CLIP representation evidence completed |
| 7 | CLIP fine-tuning and contrastive learning | CORE | Tier 1 smoke and Tier 2 MPS run; full-parameter checkpoint, validation-only selection, held-out test result, paired bootstrap deltas, and qualitative before/after artifact |
| 8 | PEFT/LoRA | ADVANCED | real PEFT 0.20.0 rank-8 Tier 2 run; 491,521 trainable parameters, adapter artifact, validation selection, canonical three-way comparison, paired bootstrap, and efficiency evidence |
| 9 | hard-negative mining | ADVANCED | real static frozen-CLIP train-only top-5 mining on 800 groups; 50% mixed objective, Tier 2 full-FT run, paired top-k comparison, early-rank analysis, and false-negative audit |
| 10 | FAISS/HNSW retrieval and scalability | ADVANCED / COMPUTE-DEPENDENT | real exact/FAISS/hnswlib comparison, validation-only parameter selection, fidelity/semantic/latency/storage/build evidence at 100/1,000/5,000 groups |
| 11 | two-stage retrieval and reranking | ADVANCED | exact FAISS Flat Stage 1, train-only pairwise MLP Stage 2, candidate-depth selection, paired statistics, oracle ceilings, and Tier 2/Tier 3 quality-latency evidence; executed reranker result was negative |
| 12 | image + text query fusion | ADVANCED | controlled image-plus-text-to-image evaluation, image/text controls, early/late alpha grid, paired comparisons, modality dominance, qualitative conflict/compositional analysis, and latency evidence |
| 12B | proper composed image retrieval correction | ADVANCED / PARTIAL, NON-BLOCKING | CIRCO adapter, official multi-ground-truth metrics, exact-search protocol, tests, preflight, and closure implemented; official archive not downloaded, so no real benchmark result is claimed |
| 13 | statistical validation and multiple seeds | CORE / PASS | predeclared seeds 42/123/2026, fixed-manifest full-FT/LoRA/hard-negative runs, paired query bootstrap, permutation tests, Holm correction, effect sizes, stability classifications, compute and storage provenance |
| 14 | controlled ablations | CORE / PASS | Phase 13 audit PASS; mandatory zero-shot/full-FT, full-FT/LoRA, hard-negative, and reranker comparisons reused; one actual ratio-25 hard-negative ablation; efficiency, qualitative, statistical, contribution, and provenance artifacts under `artifacts/phase14` |
| 15 | robustness and distribution shift | CORE / PASS | evaluation-only zero-shot and full-FT CLIP robustness over 10 text and 14 image corruption conditions, paired bootstrap, rank stability, and a predeclared 5-vs-5 aspect-ratio shift under `artifacts/phase15` |
| 16 | error analysis and failure taxonomy | CORE / PASS | fixed-scope query records, rank transitions, explicit mechanical/heuristic taxonomy, deterministic examples, corrective priorities, and artifacts under `artifacts/phase16` |
| 17 | confidence, uncertainty, selective retrieval | ADVANCED / PASS | validation-only calibration, discrimination, reliability/ECE/Brier, selective coverage-risk/AURC, high-confidence errors, robustness limitation, and artifacts under `artifacts/phase17` |
| 18 | explainability and embedding interpretation | ADVANCED / PASS | deterministic token deletion, 3x3 region occlusion, counterfactuals, local faithfulness checks, high-confidence-error explanations, explicit limitations, and artifacts under `artifacts/phase18` |
| 19 | responsible AI and bias/exposure analysis | CORE / PASS | COCO-derived caption-length, lexical-rarity, object-complexity, and aspect-ratio strata; bootstrap uncertainty; confidence/high-confidence-error review; privacy, safety, multilingual, accessibility, and misuse limitations; system card and artifacts under `artifacts/phase19` |
| 20 | efficiency and resource optimization | OPTIONAL / COMPUTE-DEPENDENT / PASS | retained-artifact resource inventory, model/checkpoint sizes, measured model-load and query-stage latency, cache and float16 analysis, exact-versus-ANN decision, cost/quality frontier, configuration recommendations, cleanup recommendations, and artifacts under `artifacts/phase20`; no training or download |
| 21 | FastAPI retrieval service | CORE engineering / PASS | application factory, startup artifact/provenance validation, full-FT + cached-embedding + FAISS Flat reuse, health/readiness/info, text→image and image→text endpoints, input/error handling, privacy-aware logging, real-model smoke, warm latency benchmark, and artifacts under `artifacts/phase21` |
| 22 | Streamlit/Gradio application | CORE engineering / PASS | Streamlit direct adapter over the Phase 21 RetrievalService, bounded text→image and image→captions controls, visible status/errors/privacy limitations, resource reuse, real browser/AppTest smoke, UI latency, screenshots, provenance, and artifacts under `artifacts/phase22`; no training or download |
| 23 | testing, CI, reproducibility hardening | CORE engineering / PASS | marker-aware test inventory, compact GitHub Actions workflow, frozen-lockfile and clean-environment validation, entrypoint/API/UI checks, artifact/provenance validation, path/secret/large-file audits, and artifacts under `artifacts/phase23`; no training or download |
| 24 | deployment and packaging hardening | CORE engineering / PASS | native uv deployment, preflight, local artifact manifest, API/UI smoke, and packaging evidence under `artifacts/phase24` |
| 25 | observability and runtime reliability | CORE engineering / PASS | request IDs, structured logs, error taxonomy, reliability/concurrency/soak checks, and artifacts under `artifacts/phase25` |
| 26 | end-to-end system validation and final benchmark | CORE engineering / PASS | frozen final configuration, integrity checks, real API/UI/deployment validation, cold/warm latency, reliability, claims, limitations, scorecard, and artifacts under `artifacts/phase26` |
| 27 | final documentation and portfolio release preparation | CORE / PASS | final README, architecture/results summary, portfolio material, claims/link consistency audit, release-file classification, and artifacts under `artifacts/phase27` |

## Feature classification summary

### CORE

COCO acquisition and validation, image-grouped leakage protection, multimodal EDA, classical baselines, frozen CLIP baseline, unified bidirectional evaluation, transformer representation comparisons, contrastive fine-tuning, statistical validation, ablations, robustness, error analysis, responsible-AI analysis, API/app integration, tests, CI, documentation, and final audits. Historical Flickr30k remains optional external validation.

### ADVANCED

LoRA/PEFT, hard-negative mining, approximate indexes, reranking, multimodal query fusion, uncertainty/selective retrieval, embedding-space interpretation, and richer responsible-exposure analyses. These are included because they answer specific research questions, not because they increase feature count.

### OPTIONAL

Compression, multilingual WIT analysis, large web-scale pretraining, elaborate dashboards, and additional model families that do not answer a declared question. An optional feature may be omitted without weakening the core study.

### COMPUTE-DEPENDENT

Full COCO/WIT/CC3M acquisition, large-batch or full-model fine-tuning, large-scale ANN benchmarks, broad multi-seed sweeps, and full external distribution-shift studies. Code support for a tier does not imply that tier was executed.

## Phase 0 stop condition

Phase 0 ends after the standard-library bootstrap check and tests pass. Dataset bytes, model downloads, training, retrieval indexes, API code, and dashboards belong to later phases.
