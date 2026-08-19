# OmniSearch system card

## System and intended use

OmniSearch is a research image-text retrieval system built around a frozen
zero-shot CLIP baseline and a one-epoch full-parameter CLIP fine-tuning
checkpoint. The evaluated protocol retrieves COCO images for text queries and
COCO captions for image queries. It is intended for reproducible research and
error analysis, not autonomous decisions or unsupervised action.

The quantitative evidence is limited to the declared COCO `val2017`-derived,
image-grouped split, the retained 100-image Tier 2 test selection, and the
`openai/clip-vit-base-patch32` model family. COCO same-image caption relevance
is a metadata proxy, not a complete human relevance judgment.

## Evaluation evidence

Phase 7 established the zero-shot/full-FT retrieval rankings. Phase 15 added
synthetic robustness and a small aspect-ratio stress test. Phase 16 analysed
mechanical rank failures and heuristic failure categories. Phase 17 added
validation-only confidence calibration, selective retrieval, and
high-confidence-error records. Phase 18 added local token deletion and coarse
image-region perturbation evidence; these are sensitivity tests, not causal
explanations. Phase 19 adds dataset-derived group comparisons, confidence
disparity observations, privacy/safety analysis, and mitigation guidance.

## Responsible-use limitations

- No protected-attribute labels were fabricated or inferred. The Phase 19
  strata are caption length, train-caption lexical rarity, a small object
  complexity heuristic, and image aspect ratio. Their disparities are
  descriptive and are not a fairness conclusion.
- The test corpus is English COCO text. Multilingual performance, accessibility
  usability, and generalization to other domains were not evaluated.
- High-confidence retrieval errors remain. Confidence is a bounded empirical
  estimate over this protocol, not a guarantee of correctness.
- No content-safety classifier, moderation policy evaluation, red-team study,
  or human safety review was run. Results must not be treated as a safety
  screen.
- The benchmark uses public-source COCO material, but image rights remain with
  originating Flickr sources. Any new data requires provenance, consent,
  retention, deletion, and access-control review.
- Image-text retrieval could be misused for sensitive search, profiling, or
  surveillance. No public service, private-data ingestion, or deployment
  authorization is provided by this repository.

## Efficiency and deployment configuration evidence

Phase 20 reused the retained Phase 7–19 evidence and performed no training or
download. The recommended quality configuration is full-FT CLIP with cached
embeddings and exact FAISS Flat retrieval. At the current approximately
5,000-image scale, FAISS Flat preserved exact neighbor fidelity while tested
IVF/HNSW alternatives traded fidelity and semantic Recall@5 for speed. Warm
retrieval is dominated by query encoding; caching removes that stage from the
warm path. The Phase 11 reranker is disabled because its roughly 1 ms/query
overhead accompanied negative held-out quality deltas.

LoRA is an optional constrained-adaptation configuration. Its measured
1,988,122-byte adapter and 491,521 trainable parameters are not the total
deployable model size: the base CLIP model remains required. A frozen
zero-shot CLIP plus cached embeddings and exact search remains the lightweight
baseline. Phase 20 evaluated float16 cached embeddings in memory only; the
canonical float32 cache was not replaced. Peak MPS memory was not reliably
measured and is intentionally not stated numerically.

## Human oversight and mitigations

High-impact or sensitive retrieval requires human review and an abstention
path. A deployment would need policy-specific content filtering, abuse and
security testing, access controls, audit logs, monitoring by declared data
strata, and governance-approved human relevance data before making claims
about fairness or safety.

## Reproducibility

Machine-readable Phase 19 evidence is under `artifacts/phase19/`; Phase 20
resource evidence is under `artifacts/phase20/`. The reports record source
hashes, the fixed split, group rules, and the fact that Phase 20 performed no
training or dataset download. These efficiency measurements do not change the
responsible-use limitations above.

## Phase 21 service interface

Phase 21 exposes the recommended full-FT CLIP + cached float32 embeddings +
FAISS Flat exact-search configuration through a FastAPI application factory.
The service validates Phase 20 report status, checkpoint/cache/index hashes,
manifest identity, dimensions, candidate IDs, and model identity before
loading the model. Resources are loaded once per process and reused; the
reranker and approximate indexes are not enabled. The interface provides
`/health`, `/ready`, `/info`, `/search/text-to-image`, and
`/search/image-to-text` with Pydantic request/response schemas and measured
latency stages.

The default is privacy-aware: uploaded images are decoded in memory and not
stored, raw query text is not logged, client-controlled filesystem/model paths
are not accepted, and request logs contain endpoint/request ID/latency/result
count/error category only. Upload size, image format, query length, and top-k
are bounded. This does not provide authentication, rate limiting, abuse
controls, a private-data audit, or content moderation. Content-safety
filtering remains **NOT IMPLEMENTED**; the API is a research interface and
not a safe-deployment gate.

## Phase 22 interactive demo

Phase 22 adds a local Streamlit interface over the same Phase 21
`RetrievalService`. The interface exposes text→image and image→captions modes,
bounded top-k, ranked IDs/scores/metadata, backend/UI latency fields, and a
visible readiness panel. Streamlit resource caching keeps one service/model
instance per UI process; no retrieval logic or model-specific transform is
duplicated in the app.

The demo is research-only and local. Uploaded images are decoded in memory and
not written to disk by the UI. It has no authentication, rate limiting,
moderation, private-data audit, or production retention policy. Scores are not
confidence probabilities. **CONTENT-SAFETY FILTERING: NOT IMPLEMENTED.**
Phase 22 evidence, including real text results, image-to-text AppTest results,
latency, and screenshots, is under `artifacts/phase22`.

## Phase 24 deployment boundary

The validated deployment path is native `uv` with the Phase 21 FastAPI service
or the Phase 22 Streamlit UI. Both reuse the same full-FT CLIP checkpoint,
float32 embedding cache, and exact FAISS Flat indexes. `omnisearch-preflight`
checks local artifact presence, hashes/compatibility, permissions, disk space,
runtime dependencies, model-cache availability for offline mode, and MPS/CPU
device selection before launch. The Phase 24 deployment manifest records the
required local artifacts using configurable relative paths.

The local API smoke loaded MPS resources, passed `/health` and `/ready`, and
returned text→image and image→text results. Native API cold start was 6.10
seconds on the recorded Apple macOS host; warm retrieval remains a separate
measurement. CPU fallback is supported. macOS launchers centralize the
documented `KMP_DUPLICATE_LIB_OK=TRUE` FAISS/PyTorch compatibility workaround.

An optional Dockerfile describes a CPU-only mounted-artifact path, but Docker
was not built or run and cannot provide Apple MPS in the current ordinary
container setup. This repository is not production-hardened: authentication,
rate limiting, TLS/reverse proxy, upload abuse controls, content safety,
privacy governance, and operational load testing remain required before public
deployment.

## Phase 25 observability boundary

The existing API now emits structured JSON request records with timestamp,
level, endpoint, status, request ID, total latency, retained latency stages,
and a small error taxonomy. It returns the same request ID in the
`X-Request-ID` response header and in successful retrieval/error bodies where
applicable. Raw queries, uploads, secrets, and reusable filesystem paths are
excluded by default. `/metrics` is a process-local snapshot of request/error
counters and startup/readiness state, not an external monitoring system.

Startup readiness still means the validated model/checkpoint, both retrieval
indexes, metadata, and related Phase 20 resources are loaded and compatible;
`/health` remains available while the service is degraded. Phase 25 exercised
failure taxonomy, request IDs, CPU fallback selection, concurrency, and a
short soak on the native local deployment. It does not establish public
availability, alerting, durable observability, or a production SLO. Shutdown
reference cleanup is implemented, while a nonfatal macOS resource-tracker
warning remains a recorded lower-level runtime limitation when observed.

## Final portfolio release status

The final serving configuration is Phase 7 full-FT CLIP with cached float32
embeddings and exact FAISS Flat search. Phase 26 measured the native MPS path,
and Phase 27 checked that the README, evaluation notes, claims, limitations,
links, and release-file recommendations agree with those artifacts. The
project is suitable for a research/demo portfolio presentation. It is not a
public production service, and Phase 12B remains PARTIAL / NON-BLOCKING /
STORAGE-DEFERRED with no CIRCO performance claim.

## Use, privacy, and human oversight

The intended use is local research, reproducible evaluation, and an
interactive portfolio demonstration over the declared COCO-derived corpus.
The system is not intended for identity decisions, surveillance, employment,
credit, medical decisions, moderation, or unsupervised decisions about people.

Uploaded images are decoded in memory and are not written to disk by the API
or Streamlit adapter. Default structured logs retain request IDs, endpoint,
status, latency, and error category, but not raw query text, image bytes,
secrets, or reusable filesystem paths. Operators should still avoid sending
private or sensitive images to a local demo process and should place a
separate retention and access policy around logs or a reverse proxy.

Scores are similarity values, not confidence or safety judgments. A human
should review results before using them outside the research scope, especially
when queries involve people, sensitive attributes, or ambiguous content.
Content-safety filtering, demographic fairness evaluation, multilingual
coverage, accessibility testing, and a private-data audit were not evaluated.
Those are open risks, not implied capabilities.

The source data carries provenance and rights limitations: COCO annotations
are documented under their stated terms, while image rights remain tied to the
originating photographs and applicable terms. The project reports benchmark
behavior only for its English, metadata-defined COCO scope; it does not claim
that the system is representative, safe, or fair outside that scope.

## Phase 26 final system scorecard

The final frozen system is Phase 7 full-FT CLIP with the Phase 10 float32
embedding cache and exact FAISS Flat indexes. Reranking is disabled by
default; LoRA, hard negatives, ANN, and fusion remain optional research
components. Phase 26 verified artifact hashes, dimensions, dtype,
normalization, candidate units, model identity, and the image-grouped test
selection boundary before launching the service.

The real native macOS run selected MPS. API cold start through `/ready` was
11.139 seconds. Ten warm server requests per direction measured text→image
mean/median/p95 of 12.31/11.62/15.46 ms and image→text
27.50/24.48/37.00 ms. The API passed health/readiness, text/image smoke,
repeated-query, and limited four-worker checks; Streamlit health and clean
shutdown passed. These are local research/demo results, not public SLOs.

The retained held-out quality metrics are text→image
R@1/R@5/R@10/MRR = 0.8263/0.9880/1.0000/0.8946 and image→text =
0.1837/0.8143/0.9303/0.9517. Claims are limited to the declared English COCO
scope. No protected-group fairness, multilingual, content-safety, causal
explainability, million-scale ANN, or CIRCO result is claimed. Phase 12B is
still partial/storage-deferred, and public internet deployment requires
authentication, rate limiting, TLS/reverse proxy, durable monitoring, content
safety, and upload-abuse controls.
