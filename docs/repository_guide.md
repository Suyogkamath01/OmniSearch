# OmniSearch repository guide

This is a file-by-file map of the public OmniSearch repository. The captions
are intentionally specific to this project: they explain why a file is here,
what question it answers, or which part of the demo it supports.

The guide covers the public tracked surface. The full dataset, fine-tuned
checkpoint, embedding arrays, FAISS binary indexes, Hugging Face cache, and
runtime logs remain local-only because they are large, machine-specific, or
not appropriate for a public source import.

## Project controls and runtime entry points

| File | Personalised caption |
| --- | --- |
| `.dockerignore` | Keeps local datasets, checkpoints, caches, and generated runtime material out of a portable container build. |
| `.env.example` | The safe starting point for wiring a local checkpoint, indexes, ports, device, and offline mode without committing secrets. |
| `.github/workflows/ci.yml` | The clean-checkout contract: it installs the project, runs the fixture suite, and checks the public package surface. |
| `.gitignore` | Separates reusable source and small evidence from private datasets, model weights, caches, indexes, logs, and machine-local state. |
| `.python-version` | Pins the intended interpreter family so local development and CI begin from the same Python baseline. |
| `Dockerfile` | A portability path for the API/UI stack; it deliberately mounts large model artifacts instead of baking them into the image. |
| `README.md` | The human entry point: what OmniSearch does, what actually worked, how to run it, and where the evidence boundary is. |
| `configs/default.toml` | Reproducible defaults for data, representations, evaluation, serving, and resource-aware experiment settings. |
| `pyproject.toml` | The project contract for dependencies, optional extras, tooling, test markers, and concise CLI commands. |
| `uv.lock` | The resolved dependency graph used to make installs repeatable rather than dependent on whatever happens to be newest. |
| `data/raw/.gitkeep` | Preserves the expected raw-data mount point while keeping the restricted/large dataset out of Git. |
| `data/interim/.gitkeep` | Leaves a predictable place for validated intermediate data without pretending those files are part of the public release. |
| `data/processed/.gitkeep` | Documents the processed-data location used by the pipeline while keeping generated arrays and indexes local-only. |

## Core retrieval and data lifecycle

| File | Personalised caption |
| --- | --- |
| `src/omnisearch/__init__.py` | Gives the package a small public identity and version surface. |
| `src/omnisearch/__main__.py` | Provides the package-level command entry when OmniSearch is invoked as a module. |
| `src/omnisearch/acquisition.py` | Defines the dataset acquisition workflow, access checks, manifests, and safe failure behaviour before downloads. |
| `src/omnisearch/coco_acquisition.py` | Handles the reproducible MS COCO metadata/image setup used by the validated benchmark. |
| `src/omnisearch/manifest.py` | Records dataset provenance, versions, schema facts, and file-level identity so results can be traced back to inputs. |
| `src/omnisearch/image_validation.py` | Checks image readability, dimensions, duplicates, and missing files before they can contaminate evaluation. |
| `src/omnisearch/preprocessing.py` | Keeps text normalization and model-agnostic image preprocessing behind small, testable interfaces. |
| `src/omnisearch/splitting.py` | Implements deterministic image-grouped train/validation/test splitting so captions from one image never cross boundaries. |
| `src/omnisearch/eda.py` | Produces only the acquisition and split statistics needed to understand dataset shape and validation health. |
| `src/omnisearch/dataset_cli.py` | Turns acquisition and validation operations into repeatable commands rather than one-off notebook steps. |
| `src/omnisearch/bootstrap.py` | Checks the repository contract and local runtime assumptions before expensive work or service startup. |
| `src/omnisearch/representations.py` | Defines embedding generation, normalization, cache metadata, and representation identity shared by evaluation and serving. |
| `src/omnisearch/baselines.py` | Holds the classical retrieval baselines that establish how much the learned multimodal system adds. |
| `src/omnisearch/clip_baseline.py` | Runs the frozen CLIP reference path used before comparing against fine-tuning. |
| `src/omnisearch/evaluation.py` | Computes bidirectional retrieval metrics with image/caption relevance handled explicitly. |
| `src/omnisearch/circo.py` | Isolates the composed-image retrieval adapter and its honest unavailable-gallery boundary. |
| `src/omnisearch/deployment.py` | Centralizes deployment configuration, preflight checks, artifact compatibility, and device selection. |
| `src/omnisearch/config.py` | Loads layered configuration and environment overrides without embedding a developer’s absolute paths. |
| `src/omnisearch/final_audit.py` | Runs the final consistency checks that connect code, evidence, claims, documentation, and release classification. |

## Service and interface files

| File | Personalised caption |
| --- | --- |
| `src/omnisearch/api/__init__.py` | Marks the HTTP layer as a small package around the shared retrieval service. |
| `src/omnisearch/api/app.py` | Builds the FastAPI application, health/readiness endpoints, request limits, and retrieval routes. |
| `src/omnisearch/api/config.py` | Translates deployment environment variables into typed API settings such as host, port, paths, and device. |
| `src/omnisearch/api/errors.py` | Gives invalid inputs and unavailable runtime resources predictable API-level error responses. |
| `src/omnisearch/api/observability.py` | Adds request IDs and structured runtime logging without turning the demo into a hosted telemetry product. |
| `src/omnisearch/api/retrieval.py` | Owns model/index loading and the reusable text-to-image/image-to-text retrieval operations. |
| `src/omnisearch/api/schemas.py` | Defines the stable request and response shapes used by the API and its smoke tests. |
| `src/omnisearch/ui/__init__.py` | Keeps the Streamlit-facing package small and importable. |
| `src/omnisearch/ui/adapter.py` | Adapts the service results into UI-friendly records and keeps presentation concerns out of retrieval code. |
| `src/omnisearch/ui/streamlit_app.py` | Provides the local reviewer demo with text search, image upload, readiness status, and ranked results. |

## Research and phase entry points

These modules preserve the project’s reproducible research history. They are
not twenty-two separate services; the production-shaped path is the shared
service in `api/retrieval.py`, FastAPI, Streamlit, and the deployment checks.

| File | Personalised caption |
| --- | --- |
| `src/omnisearch/phase6.py` | Compares representation choices and establishes the frozen-encoder reference. |
| `src/omnisearch/phase7.py` | Runs the full fine-tuning comparison that produced the selected serving checkpoint. |
| `src/omnisearch/phase8.py` | Tests parameter-efficient LoRA adaptation against the full fine-tuning reference. |
| `src/omnisearch/phase9.py` | Evaluates hard-negative mining and records where harder negatives help or hurt. |
| `src/omnisearch/phase10.py` | Compares exact and approximate index families with persistence and latency checks. |
| `src/omnisearch/phase11.py` | Tests the learned reranking idea and preserves the negative result that kept it out of serving. |
| `src/omnisearch/phase12.py` | Evaluates composed and multimodal query strategies, including fusion and sensitivity checks. |
| `src/omnisearch/phase12b.py` | Closes the storage-deferred CIRCO path without inventing a score for an unavailable gallery. |
| `src/omnisearch/phase13.py` | Measures seed stability, bootstrap uncertainty, and whether conclusions survive repeated runs. |
| `src/omnisearch/phase14.py` | Performs ablations to separate useful components from features that add complexity without evidence. |
| `src/omnisearch/phase15.py` | Tests robustness under corruption and distribution-shift conditions. |
| `src/omnisearch/phase16.py` | Builds the failure taxonomy and links retrieval errors to likely system causes. |
| `src/omnisearch/phase17.py` | Measures confidence, calibration, and selective retrieval behaviour. |
| `src/omnisearch/phase18.py` | Evaluates retrieval explanations, sensitivity, and faithfulness rather than treating similarity as an explanation. |
| `src/omnisearch/phase19.py` | Records responsible-AI scope, group comparisons, safety limitations, and mitigation recommendations. |
| `src/omnisearch/phase20.py` | Studies latency, memory, precision, and storage trade-offs for a student-scale deployment. |
| `src/omnisearch/phase21.py` | Validates the FastAPI retrieval service and its warm request behaviour. |
| `src/omnisearch/phase22.py` | Validates the Streamlit interface and preserves the three public demo screenshots. |
| `src/omnisearch/phase23.py` | Hardens tests, CI, reproducibility checks, secret checks, and release file classification. |
| `src/omnisearch/phase24.py` | Packages the native/Docker deployment paths and validates preflight, startup, and consistency. |
| `src/omnisearch/phase25.py` | Adds runtime observability, failure handling, concurrency checks, and shutdown evidence. |
| `src/omnisearch/phase26.py` | Performs end-to-end validation and freezes the final system configuration and benchmark claims. |
| `src/omnisearch/phase27.py` | Produces the final portfolio, interview, documentation, and public-release preparation records. |

## Documentation

| File | Personalised caption |
| --- | --- |
| `docs/architecture.md` | Explains how acquisition, encoders, indexes, service code, and interfaces fit together. |
| `docs/coco_migration_audit.md` | Records the migration from the earlier dataset path to the validated COCO benchmark. |
| `docs/compute_audit.md` | Keeps the hardware, runtime, and resource assumptions behind the experiments visible. |
| `docs/data_validation.md` | Describes schema, image, caption, duplicate, split, and leakage checks. |
| `docs/dataset_card.md` | States dataset source, access notes, caption structure, split policy, size evidence, and limitations. |
| `docs/dataset_migration.md` | Explains the practical steps and constraints involved in moving the working dataset. |
| `docs/dataset_selection.md` | Gives the reasoning for the benchmark choice and its fit to the project question. |
| `docs/evaluation.md` | Defines retrieval relevance, metrics, directionality, and evaluation protocol. |
| `docs/experimental_methodology.md` | Describes how experiments are configured, compared, repeated, and recorded. |
| `docs/experiments.md` | Summarises the experiment sequence and the evidence retained for each decision. |
| `docs/github_release_recommendations.md` | Captures the release hygiene checklist used before making the repository public. |
| `docs/interview_summary.md` | Turns the engineering decisions and trade-offs into an interview-ready project explanation. |
| `docs/limitations.md` | Makes the unsupported claims explicit: scale, safety, fairness, multilinguality, and public production. |
| `docs/mathematical_foundations.md` | Provides the compact mathematical basis for normalized embeddings, similarity, and retrieval metrics. |
| `docs/portfolio_summary.md` | Presents the project as a coherent portfolio case study rather than a list of disconnected phases. |
| `docs/reproducibility.md` | Explains environment setup, artifact boundaries, deterministic controls, and how to repeat the validated paths. |
| `docs/research_design.md` | Connects the research questions to controls, baselines, ablations, and decision criteria. |
| `docs/resume_bullets.md` | Provides concise, evidence-backed ways to describe the work without overstating it. |
| `docs/roadmap.md` | Shows what was completed, what remains deferred, and which future work is genuinely open. |
| `docs/system_card.md` | Describes intended use, evaluation scope, safety boundaries, limitations, and operational assumptions. |
| `docs/technical_defense.md` | Gives the reasoning behind leakage protection, model/index choices, deployment decisions, and honest caveats. |
| `docs/repository_guide.md` | This personalised map: a quick way to understand the purpose of each public file without reading the whole history first. |

## Tests

The tests are deliberately split between cheap fixture checks and explicitly
marked local-data/real-model checks. That lets a clean public checkout verify
the software contract without pretending that private model artifacts exist.

| File | Personalised caption |
| --- | --- |
| `tests/conftest.py` | Shared fixtures for tiny deterministic datasets, images, captions, and service-shaped test inputs. |
| `tests/test_bootstrap.py` | Protects the repository contract and the clean-checkout assumptions used by CI. |
| `tests/test_coco_dataset.py` | Checks COCO metadata parsing and dataset structure on controlled fixtures. |
| `tests/test_phase1_acquisition.py` | Verifies safe acquisition planning, access checks, manifests, and non-silent blockers. |
| `tests/test_phase1_images.py` | Exercises missing, unreadable, duplicate, and exact-duplicate image validation. |
| `tests/test_phase1_preprocessing.py` | Checks basic caption normalization and model-agnostic preprocessing boundaries. |
| `tests/test_phase1_splitting.py` | Proves deterministic image-grouped splits and cross-split leakage failures. |
| `tests/test_phase2_eda.py` | Guards the minimum statistics needed to validate the dataset and split. |
| `tests/test_phase3_baselines.py` | Checks the classical retrieval references before learned models are compared. |
| `tests/test_phase4_clip.py` | Verifies the frozen CLIP baseline contract and representation metadata. |
| `tests/test_phase5_evaluation.py` | Protects bidirectional metric calculations and relevance handling. |
| `tests/test_phase6_representations.py` | Checks embedding generation, normalization, caching, and representation identity. |
| `tests/test_phase7.py` | Validates fine-tuning configuration, evidence loading, and selected-model claims. |
| `tests/test_phase8.py` | Checks LoRA experiment metadata and comparison boundaries. |
| `tests/test_phase9.py` | Checks hard-negative mining records, quality safeguards, and result structure. |
| `tests/test_phase10.py` | Checks exact/approximate index metadata, persistence, and benchmark records. |
| `tests/test_phase11.py` | Protects reranker comparisons and the recorded negative decision. |
| `tests/test_phase12.py` | Checks multimodal/composed-query evaluation records and fusion behaviour. |
| `tests/test_phase12b.py` | Ensures the storage-deferred CIRCO closure remains explicit and non-fabricated. |
| `tests/test_phase13.py` | Checks seed, bootstrap, and statistical-evidence artifacts. |
| `tests/test_phase14.py` | Validates ablation tables and component-value classifications. |
| `tests/test_phase15.py` | Checks corruption, shift, and robustness evidence. |
| `tests/test_phase16.py` | Protects failure taxonomy structure and error-link records. |
| `tests/test_phase17.py` | Checks confidence, calibration, and selective-retrieval evidence. |
| `tests/test_phase18.py` | Checks explanation records, faithfulness evidence, and sensitivity outputs. |
| `tests/test_phase19.py` | Checks responsible-AI matrices, group definitions, and limitation records. |
| `tests/test_phase20.py` | Checks resource, precision, latency, and efficiency evidence. |
| `tests/test_phase21.py` | Verifies API dependency audits, health/readiness, schemas, and retrieval smoke paths. |
| `tests/test_phase22.py` | Verifies the UI dependency audit, launch contract, and screenshot manifest. |
| `tests/test_phase23.py` | Checks CI, clean-environment, secret, large-file, and documentation-link evidence. |
| `tests/test_phase24.py` | Checks deployment preflight, artifact manifests, startup, and consistency evidence. |
| `tests/test_phase25.py` | Checks observability, reliability, concurrency, failure-injection, and shutdown evidence. |
| `tests/test_phase26.py` | Checks final benchmark, integrity, reliability, claims, and system-manifest evidence. |
| `tests/test_phase27.py` | Checks final documentation, portfolio, release classification, and link evidence. |
| `tests/test_real_model_smoke.py` | Optional local-data smoke coverage for the actual checkpoint and indexes; omitted in ordinary clean CI. |
| `tests/__init__.py` | Keeps the tests importable as a small package for the project’s tooling. |

## Public evidence and screenshots

These are intentionally small, reviewable records. They are not substitutes
for the local dataset or checkpoint; they let a reader inspect the claimed
setup and final measurements without downloading the whole experiment store.

| File | Personalised caption |
| --- | --- |
| `artifacts/coco_phase1/statistics.json` | Small acquisition/split statistics record showing what the public validation summary actually contains. |
| `artifacts/coco_phase1/validation.json` | Public schema and validation summary for the dataset setup. |
| `artifacts/phase12b/closure.json` | The compact record explaining why CIRCO remains storage-deferred and why no score is reported. |
| `artifacts/phase22/ui_landing.png` | A real capture of the local Streamlit landing state and service readiness. |
| `artifacts/phase22/ui_text_to_image.png` | A real capture of ranked image results for a text query. |
| `artifacts/phase22/ui_image_to_text_mode.png` | A real capture of the image-to-caption mode and upload affordance. |
| `artifacts/phase24/deployment_manifest.json` | Machine-readable description of the supported deployment artifacts and runtime identity. |
| `artifacts/phase26/claims_audit.json` | Final cross-check of which project claims are supported, qualified, or unavailable. |
| `artifacts/phase26/final_benchmark.json` | Canonical held-out retrieval measurements used in the README. |
| `artifacts/phase26/final_latency.json` | The local warm-request and startup timing record behind the runtime numbers. |
| `artifacts/phase26/final_reliability.json` | Final service reliability and failure-handling evidence. |
| `artifacts/phase26/final_scorecard.json` | A compact scorecard tying quality, efficiency, and limitations together. |
| `artifacts/phase26/final_system_manifest.json` | Identity record for the final model, indexes, configuration, and evaluation path. |
| `artifacts/phase26/phase26_report.json` | The structured end-to-end validation report for the completed system. |
| `artifacts/phase27/artifact_validation.json` | Release-time check that public evidence files are present and valid. |
| `artifacts/phase27/claims_validation.json` | Release-time check against inflated or unsupported wording. |
| `artifacts/phase27/documentation_consistency.json` | Cross-document consistency check for metrics, status, limitations, and commands. |
| `artifacts/phase27/final_project_summary.json` | Structured portfolio summary of the final project state. |
| `artifacts/phase27/github_release_recommendations.json` | Release preparation notes for a clean public GitHub import. |
| `artifacts/phase27/interview_summary.json` | Structured interview-facing explanation of the system and its trade-offs. |
| `artifacts/phase27/link_validation.json` | Evidence that the release-facing internal documentation links resolve. |
| `artifacts/phase27/phase27_report.json` | The final documentation and portfolio phase report. |
| `artifacts/phase27/portfolio_summary.json` | Machine-readable portfolio framing tied back to real artifacts. |
| `artifacts/phase27/pre_phase_audit.json` | The audit record that phase 27 started from the expected final state. |
| `artifacts/phase27/provenance.json` | Provenance for the final documentation and release records. |
| `artifacts/phase27/release_file_classification.json` | Explicit public/local-only classification used to keep large resources out of Git. |
| `artifacts/phase27/resume_bullets.json` | Evidence-backed resume language with the project’s limitations still attached. |
| `artifacts/final_audit/artifact_validation.json` | Final repository-wide artifact presence and format check. |
| `artifacts/final_audit/claims_consistency.json` | Final check that README, docs, and evidence tell the same story. |
| `artifacts/final_audit/documentation_audit.json` | Final documentation completeness and natural-language review record. |
| `artifacts/final_audit/final_scores.json` | Final score summary with only measured values retained. |
| `artifacts/final_audit/full_audit.json` | The broad final audit result tying the release checks together. |
| `artifacts/final_audit/improvement_pass.json` | Record of the last honest improvement pass before public release. |
| `artifacts/final_audit/phase_status.json` | Completion/deferred-status map for the declared project phases. |
| `artifacts/final_audit/provenance.json` | Provenance for the final audit bundle. |

Empty `.gitkeep` files under `artifacts/experiments/`, `artifacts/logs/`,
`artifacts/metrics/`, and `artifacts/models/` are directory placeholders for
clean checkout tests. They do not claim that the corresponding local outputs
are publicly included.

## How to navigate it quickly

Start with `README.md`, then `docs/architecture.md` and
`docs/reproducibility.md`. For the actual service path, read
`src/omnisearch/api/retrieval.py`, `src/omnisearch/api/app.py`, and
`src/omnisearch/ui/streamlit_app.py`. For evidence, begin with the Phase 26
benchmark/system manifest and the three Phase 22 screenshots. For the honest
boundaries, read `docs/limitations.md`, `docs/system_card.md`, and
`docs/technical_defense.md`.
