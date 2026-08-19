# Reproducibility

## Fixed inputs

- Python: project constraint `>=3.12,<3.14`; the executed real runs used the
  local Python 3.12 environment.
- Seed: 42.
- Dataset manifest:
  `data/processed/coco2017_val_split_manifest.json`.
- Manifest SHA-256:
  `09a2c1e56eb1a628b2ead16f064510d713f81aff5ee2f2d09b4ca8993bba3b43`.
- Active image root: `data/raw/coco2017/val2017`.
- Canonical evaluator: `retrieval_eval_v1`.
- Base model: `openai/clip-vit-base-patch32`.
- Recorded hardware: Apple MPS; unified-memory peak usage was not measured
  reliably.

## Reproduction commands

Install the declared development and experiment dependencies with `uv`:

```bash
uv sync --extra dev --extra phase4 --extra research --extra phase8 --extra retrieval
```

Run smoke and real experiments:

```bash
PYTHONPATH=src uv run python -m omnisearch.phase7 --config configs/default.toml --output-dir artifacts/phase7 --smoke
PYTHONPATH=src uv run python -m omnisearch.phase8 --config configs/default.toml --output-dir artifacts/phase8 --smoke
PYTHONPATH=src uv run python -m omnisearch.phase9 --config configs/default.toml --output-dir artifacts/phase9 --smoke
PYTHONPATH=src uv run python -m omnisearch.phase10 --config configs/default.toml --output-dir artifacts/phase10_smoke --smoke
```

The real Tier 2 commands are the same without `--smoke`. Phase 9 mining is
static and train-only; its generated manifest records the source manifest
hash, model, strategy, candidate pool, seed, and protection rules.

Run the real Phase 10 benchmark with:

```bash
PYTHONPATH=src uv run python -m omnisearch.phase10 \
  --config configs/default.toml --output-dir artifacts/phase10
```

Phase 10 reuses the cached normalized embeddings when the manifest and Phase 7
checkpoint hashes match. Its latency measurements use the first deterministic
128 query vectors (or all queries for smaller tiers), repeated three times;
semantic metrics still use every declared validation or test query. The real
run uses validation fidelity and search latency for configuration selection and
reports held-out test results afterward.

Run Phase 11 with:

```bash
PYTHONPATH=src uv run python -m omnisearch.phase11 \
  --config configs/default.toml --output-dir artifacts/phase11
```

On the recorded macOS host, FAISS and PyTorch can load separate OpenMP
runtimes. The actual run therefore used the process-level compatibility flag
below; this is an environment workaround, not a model or data setting:

```bash
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE uv run python -m omnisearch.phase11 \
  --config configs/default.toml --output-dir artifacts/phase11
```

Phase 11 trains only a shallow pairwise MLP on the 800 Tier 2 train image
groups, using exact FAISS Flat candidate retrieval. Candidate depths 10, 25,
and 50 are evaluated on validation; depth 10 was selected by mean MRR. The
test results use separate test-only candidate indexes and are not used for
fitting or selection. Query encoding, Stage-1 search, and reranking latency
are recorded separately. The reranker checkpoint, training history, index
metadata, paired bootstrap comparisons, oracle analysis, and failure findings
are all under `artifacts/phase11/`.

## Artifact contract

Each experiment records configuration, manifest hash, split, seed, model,
device, protocol version, training history where applicable, selection
decision, and output paths. Phase 10 additionally records embedding generation
time, exact/ANN fidelity, search latency, throughput, build/load/serialization
time, raw vector storage, serialized index size, library versions, and index
metadata. Full checkpoints, embeddings, and indexes are ignored by Git; they
remain local artifacts and are not silently replaced by placeholders. Interrupted
checkpoint saves are not valid results; Phase 9 uses atomic checkpoint
replacement and validates the final checkpoint before test evaluation.

The canonical regression checks are:

```bash
PYTHONPATH=src uv run pytest -q
uv run ruff check src tests
uv run mypy src/omnisearch
PYTHONPATH=src uv run python -m compileall -q src tests
```

The Phase 11 focused checks were also run with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run pytest -q tests/test_phase11.py
```

## Selection discipline

All model, mining, hardness, and checkpoint decisions use train/validation
data only. The held-out test image groups and captions are materialized only
after selection, and paired comparisons use identical query IDs, relevance
sets, candidate corpora, and `retrieval_eval_v1` metadata.

The Phase 11 reranker is a ranking-score model, not a calibrated classifier.
Its negative held-out result is retained as evidence: candidate recall and an
analysis-only oracle show the available ceiling, while paired test deltas
show that this shallow interaction head degraded the validated Stage-1
baseline. Phase 12 fusion is documented below. Phase 12B is closed
PARTIAL/NON-BLOCKING and Phase 13 is complete under `artifacts/phase13`.

Run Phase 12 with:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
KMP_DUPLICATE_LIB_OK=TRUE uv run python -m omnisearch.phase12 \
  --config configs/default.toml --output-dir artifacts/phase12
```

The offline flags are intentional: Phase 12 reuses the locally cached CLIP
checkpoint already validated by Phase 7 and prevents an unrelated Hugging
Face safetensors-conversion request from racing with checkpoint loading. The
OpenMP compatibility flag is required on this macOS host when FAISS and
PyTorch are loaded in one process.

Phase 12 uses one associated caption per image group for the controlled
image-plus-text-to-image task. Image-only and text-only controls share the
same split-specific candidate corpus and query IDs. Early fusion computes
`normalize(alpha * image + (1-alpha) * text)`; late fusion combines the two
cosine score vectors. Alpha is selected from `{0.25, 0.50, 0.75}` on Tier 2
validation only. Test data is used only for final reporting and paired
comparisons. Compositional edit strings are qualitative-only and have no
benchmark relevance labels.

Phase 12B preflight and smoke command:

```bash
PYTHONPATH=src uv run python -m omnisearch.phase12b \
  --config configs/default.toml --output-dir artifacts/phase12b
PYTHONPATH=src uv run python -m omnisearch.phase12b \
  --config configs/default.toml --output-dir artifacts/phase12b_smoke --smoke
```

Phase 12B uses the official CIRCO repository and COCO 2017 unlabeled image
path. The real preflight recorded the official archive size as 20,126,613,414
bytes and insufficient current storage, so it stopped before download.
`artifacts/phase12b/preflight.json`, `phase12b_report.json`, and `closure.json`
are the durable provenance artifacts. No unofficial mirror or existing
labeled COCO validation image is substituted.
When the authorized archive is available, the adapter uses the released CIRCO
validation labels, deterministic selection/holdout alpha selection, exact
FAISS Flat retrieval, official mAP/Recall semantics, and no CIRCO test-label
claim because those labels are withheld.

Run Phase 13 with the cached model stack:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
KMP_DUPLICATE_LIB_OK=TRUE uv run python -m omnisearch.phase13 \
  --config configs/default.toml --output-dir artifacts/phase13
```

Phase 13 predeclares seeds `42, 123, 2026`, reuses compatible seed-42 Phase
7/8/9 results, and trains fresh seeds 123 and 2026 on the same seed-42
selected 800/100/100 Tier-2 image groups. Phase 9 uses the verified fixed
seed-42 mined-negative manifest. Zero-shot is a single non-trained reference.
Per-seed rankings, histories, bootstrap comparisons, permutation tests, Holm
correction, stability classifications, and cleanup provenance are under
`artifacts/phase13`.

Run the controlled Phase 14 ablations with the cached model stack:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
KMP_DUPLICATE_LIB_OK=TRUE uv run python -m omnisearch.phase14 \
  --output-dir artifacts/phase14
```

Phase 14 first requires `artifacts/phase14/pre_phase_audit.json` to record
`PHASE 13 AUDIT: PASS`. It reuses Phase 13 and Phase 11 evidence and trains
only the declared seed-42 hard-negative-ratio-25 ablation. Its fixed config,
result rankings, paired bootstrap, efficiency data, qualitative comparisons,
component classification, final table, provenance, and report are under
`artifacts/phase14`. The generated ratio-25 checkpoint was removed only after
result verification; its SHA256 and byte count are preserved in provenance.

Run the evaluation-only Phase 15 robustness protocol:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
KMP_DUPLICATE_LIB_OK=TRUE uv run python -m omnisearch.phase15 \
  --output-dir artifacts/phase15
```

Phase 15 reuses the fixed manifest and Phase 7 zero-shot/full-FT artifacts.
It writes the corruption configuration and metadata-only aspect-ratio shift
definition before model evaluation. Ten text conditions and fourteen image
conditions are generated in memory; candidates remain clean, and no training,
model download, or new dataset is performed. Required results, paired
bootstrap intervals, rank stability, qualitative examples, and provenance are
under `artifacts/phase15`.

## Phase 21 API reproduction contract

Install the dedicated API dependencies and run the tested single-process
service from the repository root:

```bash
uv sync --extra api --extra phase4 --extra retrieval
KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run uvicorn omnisearch.api.app:create_app --factory \
  --host 127.0.0.1 --port 8000
```

The OpenMP flag is required on this macOS host when FAISS and PyTorch are
loaded in one process. The offline flags ensure the model cache is reused and
the service never downloads a checkpoint at startup. The application factory
does not load resources at import time; the lifespan loads and validates the
Phase 7 full-FT checkpoint, Phase 10 float32 cache, metadata, and Tier 3 FAISS
Flat indexes once per process. A model lock serializes ordinary inference
reads while indexes remain reusable read-only objects; no multiprocessing or
request-result cache is introduced.

The API is versioned in metadata as `v1`. `GET /health` is lightweight and may
report a degraded process; `GET /ready` is the resource-readiness gate; and
`GET /info` exposes safe model/backend metadata without filesystem paths.
`POST /search/text-to-image` accepts `{\"query\": \"...\", \"top_k\": 5}`.
`POST /search/image-to-text` accepts a JPEG, PNG, or WEBP multipart upload and
`top_k`. Uploads are decoded in memory and are not persisted. Responses use
Pydantic schemas and include ranked IDs, scores, metadata, request IDs, and
preprocessing/encoding/search/server-total latency fields.

The real Phase 21 run used TestClient for five warm requests per direction and
also verified the uvicorn command as a separate process. Its compact evidence
is under `artifacts/phase21/`; it records hashes, startup checks, endpoint
smoke results, OpenAPI, latency, and the fact that no training or download
occurred. Content-safety filtering is not implemented, and the API is not a
public-deployment authorization.

## Phase 22 Streamlit reproduction contract

Phase 22 is a thin local UI over the Phase 21 `RetrievalService`; it does not
reimplement CLIP, preprocessing, FAISS, or model/index lifecycle. Install and
launch it with:

```bash
uv sync --extra app --extra phase4 --extra retrieval
KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run streamlit run src/omnisearch/ui/streamlit_app.py \
  --server.headless true --server.port 8501
```

`st.cache_resource` owns one service instance per Streamlit process, so normal
reruns reuse the loaded model and indexes. The UI top-k control is bounded to
1–20 even though the underlying API contract allows a wider bound. Text and
image requests report the Phase 21 backend timing alongside direct UI service
call wall time; browser rendering is outside that latency measurement.

The real Phase 22 evidence is under `artifacts/phase22/`: the pre-phase audit,
UI configuration, direct real-resource smoke, AppTest image-to-text action,
browser text-to-image screenshots, latency, provenance, screenshot manifest,
and final report. No training or download occurred. Uploads are decoded in
memory and are not persisted by the UI. The content-safety limitation remains
explicitly **NOT IMPLEMENTED**. On the validation Chrome profile, setting a
local file through the extension file chooser was blocked by file-URL
permissions; the real Streamlit AppTest covered the image-to-text action with
the same local corpus image bytes.

## Phase 23 reproducibility hardening

Phase 23 defines three explicit reproducibility levels. Level 1 is the
default, data-free engineering path:

```bash
uv sync --frozen --extra test
uv run pytest -q
uv run ruff check src tests
uv run mypy src/omnisearch
uv run python -m compileall -q src tests
```

Level 2 is the optional real-model smoke. It requires the retained local
Phase 7 checkpoint, Phase 10 indexes and embedding caches, validated manifest,
and corpus image files; it is intentionally excluded from the default suite:

```bash
uv sync --frozen --extra api --extra phase4 --extra retrieval
KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run pytest -m real_model
```

Level 3 is a full experiment rerun using the phase-specific commands and
dataset/model artifacts documented above. It is compute-, storage-, hardware-,
and cache-dependent. The frozen `uv.lock`, `.python-version`, manifest hashes,
phase configuration, and provenance files constrain the environment, but they
cannot make MPS/native-kernel behavior identical across machines.

The compact GitHub Actions workflow at `.github/workflows/ci.yml` installs the
test extra with `uv sync --frozen`, runs the default fast suite, Ruff, mypy,
and bytecode compilation, and intentionally performs no COCO, CIRCO, model, or
index download. Its structure was validated locally in Phase 23; no hosted CI
run is claimed here. A separate pre-commit configuration was not added because
the repository has no existing hook policy and CI already centralizes the same
Ruff contract; adding another formatter/linter entrypoint would create a second
source of truth.

## Phase 24 deployment reproduction contract

The primary deployment path is native `uv`, because it can use Apple MPS while
retaining a CPU fallback. Install the frozen deployment extra and use the
short launchers:

```bash
uv sync --frozen --extra deployment
set -a; source .env.example; set +a
OMNISEARCH_OFFLINE=1 uv run omnisearch-preflight
OMNISEARCH_OFFLINE=1 uv run omnisearch-api
# or independently:
OMNISEARCH_OFFLINE=1 uv run omnisearch-ui
```

`omnisearch-ui` is an in-process Streamlit demo over the Phase 21
`RetrievalService`; it does not require the API process. `omnisearch-api`
launches the validated FastAPI factory. `omnisearch-preflight` checks Python,
dependencies, selected device, permissions, free disk, checkpoint/cache/index
compatibility, and local model-cache availability without loading model
weights. Host, ports, paths, device, and offline mode are environment-driven.

Portable components are source, tests, fixtures, `.python-version`, `uv.lock`,
and configuration. The checkpoint, manifest/image root, embedding cache, FAISS
indexes, Phase 20 provenance, and offline Hugging Face cache are local-artifact
dependent. MPS/CPU selection, cold start, warm latency, and native-library
behavior are hardware-dependent. The Phase 24 manifest records relative paths,
hashes, sizes, identity, and these portability classes under
`artifacts/phase24/deployment_manifest.json`.

On macOS the launchers set `KMP_DUPLICATE_LIB_OK=TRUE` and bounded native
thread variables as a centralized FAISS/PyTorch compatibility workaround.
This is not a general deployment guarantee. The optional Dockerfile is a
CPU-only portability path with runtime-mounted artifacts; Docker was not
built/run here because the daemon was unavailable and ordinary Docker cannot
provide Apple MPS. No Docker latency or cloud deployment claim is made.

## Phase 20 reproduction contract

Phase 20 is an evaluation-only consolidation of retained Phase 7–19
artifacts. It performs no training, dataset download, new model-family
initialization, or new index build. With the local model cache available, the
resource analysis can be rerun with:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 uv run omnisearch-phase20
```

The command writes only the declared JSON evidence under `artifacts/phase20`.
It audits Phase 19 first, reuses the Phase 10 exact/ANN tables and Phase 11
encoding/reranking measurements, and labels values as measured, calculated,
or consolidated components. The model-load submeasurement uses the local
cached `openai/clip-vit-base-patch32` weights and restores the retained Phase
7 checkpoint; it does not run a forward pass or optimizer. If the local model
cache is unavailable, the artifact records model-load profiling as not
measured rather than downloading.

The Phase 20 float16 check converts the canonical float32 embedding arrays in
memory, accumulates in float32, and reports ranking/quality agreement. It does
not overwrite the canonical cache or persist a new representation. Peak MPS
memory is explicitly not treated as measured. The Phase 20 artifact validator
records that no Phase 21 work was included in that historical Phase 20 run.

## Phase 25 observability and runtime reliability contract

Phase 25 reuses the native Phase 24 launch path and does not retrain, download
new data/models, or change retrieval. Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run omnisearch-phase25
```

The command starts the real API, checks `/health` and `/ready`, exercises 20
text and 20 image requests, verifies request-ID preservation and structured
latency fields, injects representative validation/readiness/image/model and
startup-resource failures, then runs a small concurrency check and short
soak. The Streamlit health endpoint is also checked through the existing UI
launcher. Evidence is written to `artifacts/phase25`.

Logs are JSON on process stdout. Default fields are operational metadata only;
raw query text, uploads, secrets, and reusable machine paths are excluded.
Runtime counters are in-process and reset on restart; collecting, rotating,
alerting on, and retaining logs/metrics remains the deployment environment's
responsibility. `X-Request-ID` is preserved when it is printable and at most
64 characters, otherwise the service generates a UUID. MPS remains the local
preferred device with CPU fallback, and the Phase 24 macOS OpenMP workaround
continues to be centralized in the launcher.

## Phase 26 final validation contract

Phase 26 is reproducible only when the local Phase 7 checkpoint, validated COCO
image root and manifest, Phase 10 float32 embedding cache and exact indexes,
Phase 20 provenance, and local Hugging Face cache (for offline mode) are
available. It does not download data, retrain, tune the test set, or create a
new model/index.

Run the real final path with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run omnisearch-phase26
```

The command performs the focused Phase 25 audit, native preflight, integrity
checks, API launch, `/health` and `/ready` checks, text→image and image→text
requests, a fresh Streamlit health smoke, repeated/concurrent requests, and
separate cold/warm timing. It writes only the declared JSON files under
`artifacts/phase26/`. `omnisearch-phase26` reuses the Phase 7 held-out metrics
as retained measurements; it does not recompute them.

The recorded run used native `uv` on macOS with MPS selected. Cold start was
11.139 seconds. Warm server mean/median/p95 was 12.31/11.62/15.46 ms for
text→image and 27.50/24.48/37.00 ms for image→text. These are local hardware
measurements, not cross-machine guarantees or public SLOs. Docker is an
optional CPU portability path and was not validated for this MPS-backed demo.
Phase 12B remains partial/storage-deferred. Phase 27 final documentation and
portfolio preparation is complete.
