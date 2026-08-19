# Limitations

- The active COCO benchmark is the complete feasible `val2017` release with
  an internal retrieval split, not an official COCO retrieval partition.
- Phase 7–9 fine-tuning evidence is one seed, one epoch, and Tier 2 only.
  Results do not establish multi-seed stability, Tier 3 behavior, or general
  superiority.
- COCO caption relevance is metadata-defined. It is not human semantic
  judgment, and cross-image image-to-image relevance labels are unavailable.
- Exact byte hashing catches identical files but not perceptual duplicates.
  Phase 9 additionally protects exact caption aliases, but semantic
  near-positives across distinct image IDs remain a false-negative risk.
- Phase 9's mining-quality sample uses deterministic IDs, ranks, scores, and
  a lexical-overlap screen. It does not claim human labels for visual
  similarity, attributes, actions, or compositional correctness.
- The hard-negative experiment runs one static frozen-CLIP strategy with a
  top-5 pool and 50% explicit-negative ratio. Random explicit negatives and
  separately human-labeled semantic negatives were not trained as additional
  variants because they would expand the compute budget and duplicate the
  standard in-batch comparator.
- Apple unified-memory MPS peak memory was not measured reliably. Checkpoint
  sizes and measured wall times are local run observations, not universal
  hardware claims.
- LoRA adapters require the base checkpoint for reconstruction.
- The authorized Flickr30k image archive remains unavailable; historical
  Flickr30k metadata artifacts are not COCO evidence.
- Phase 11 uses one shallow reranker architecture, one seed, one train-only
  Tier 2 fitting scope, and candidate depths 10/25/50. It does not establish
  that reranking is generally ineffective; the executed configuration was a
  negative result.
- The Phase 11 reranker was trained with pairwise metadata-defined positives,
  not human relevance judgments. Multiple captions for one image are treated
  as relevant for image-to-text evaluation, and distinct images with similar
  content can still be false negatives.
- Candidate recall is an upper-bound diagnostic, not a quality guarantee. The
  oracle analysis assumes perfect ordering of the candidates that Stage 1
  returned and is explicitly not a model.
- Phase 11 latency measurements exclude model load and are local Apple MPS
  observations. The macOS run required a process-level OpenMP compatibility
  flag for FAISS/PyTorch; this is an environment detail, not a benchmark
  quality claim.
- The reranker output is an uncalibrated score. No probability, confidence,
  threshold, or selective-retrieval interpretation is supported.
- Phase 12's quantitative task is a controlled same-image identity protocol;
  image-only retrieval has an inherent self-retrieval ceiling. Its metrics do
  not establish arbitrary compositional or attribute-edit retrieval quality.
- Early and late fusion were evaluated only in the validated CLIP dual-encoder
  space with one small alpha grid. A learned fusion model, cross-encoder, and
  broader alpha sweep were intentionally skipped.
- Conflict and compositional examples have no correctness labels. Their top
  results support qualitative inspection only, not claims about color,
  object-count, scene, or attribute correctness.
- Phase 12 re-encodes only one deterministic caption per image for the
  controlled joint-query task. It does not evaluate all possible textual
  modifications or multimodal natural-language interactions.
- Phase 12B's CIRCO adapter and benchmark-specific mAP/Recall implementation
  are present, but the official COCO unlabeled image archive was not
  downloaded because the local volume could not safely hold the archive,
  extracted gallery, index, and runtime headroom. Consequently Phase 12B has
  no real retrieval, statistical, qualitative, or latency results. It is
  formally PARTIAL/NON-BLOCKING and is excluded from Phase 13.
- CIRCO's official test ground truth is withheld. The intended local protocol
  uses a deterministic selection/holdout partition within labeled validation;
  this is not an official test-server result.
- CIRCO's CC BY-NC 4.0 repository terms do not remove the originating Flickr
  copyright and terms that apply to the COCO images.
- Phase 13 uses only three training seeds and the compact one-epoch Tier-2
  scope. Its sample standard deviations describe these runs; they are not a
  precise estimate of the population distribution of training outcomes.
- The Phase 13 paired bootstrap measures query-level uncertainty, while the
  seed table measures training variability. They are intentionally not
  collapsed into one interval.
- The recorded MPS runs are seeded but not claimed bitwise deterministic.
  Python hash randomization and possible MPS kernel nondeterminism remain
  limitations.
- Phase 13 evaluated the fixed local COCO retrieval task only. It did not
  provide real CIRCO evidence, Tier-3 evidence, robustness, calibration, or
  external-distribution evidence; Phase 15 adds only the narrowly scoped
  synthetic robustness and aspect-ratio stress tests documented above.
- Phase 14's new 25% hard-negative-ratio run uses one seed and is exploratory;
  the 0% and 50% ratio points are reused parent artifacts rather than a
  balanced multi-seed ratio sweep.
- Ablation classifications are conditional on the fixed COCO identity-based
  retrieval task. They do not prove that a component is universally useful or
  harmful on other datasets, model sizes, or compositional benchmarks.
- The Phase 11 reranker comparison is a preserved negative result under its
  selected candidate-depth protocol; it is not evidence about a different
  reranker architecture.
- Phase 15 evaluates only zero-shot and full-parameter CLIP on the fixed
  100-image/501-caption COCO Tier-2 test selection. Its corruptions are
  synthetic, query-only perturbations; candidates remain clean and no
  corrupted dataset is materialized.
- Phase 15's aspect-ratio shift compares five extreme-ratio image groups with
  five near-square controls selected from the same test subset. This is a
  small metadata-defined stress test, not external-domain generalisation.
- Phase 15 bootstrap intervals quantify paired query uncertainty for clean vs
  corrupted rankings. The shift groups are disjoint, so their comparison is
  descriptive and unpaired. Rank displacement is censored when the relevant
  item is absent from either returned top-10 list.
- Phase 16's semantic-looking taxonomy categories are heuristic labels derived
  from observable caption/query features; they are not human annotations and
  should not be read as measured semantic error prevalence.
- Phase 16 retains only the evaluated top-10 candidate rankings for exact
  rank analysis. Failures beyond that depth are censored lower bounds, and
  score margins are similarity diagnostics rather than calibrated confidence.
- The Phase 11 reranker artifact preserves aggregate metrics and qualitative
  examples but not complete paired per-query Stage-1 and reranked rankings;
  exact intersection counts for reranker-induced failures are unavailable.
- Phase 16's Phase 15 robustness links use the retained diagnostic worst-case
  examples and condition summaries, not a newly materialized corrupted
  population. They therefore identify hypotheses and priorities rather than
  estimating a general corruption failure distribution.
- Phase 17 calibrates on one fixed 100-image validation tier and evaluates on
  the matching retained 100-image test tier. It does not establish calibration
  across datasets, seeds, model families, or external domains.
- Phase 17's softmax mass and entropy are computed over retained top-10 scores,
  not the full candidate corpus. Raw CLIP similarity is not a probability, and
  calibrated confidence is only a bounded empirical estimate for this scope.
- Image-to-text uncertainty evidence is limited by only 7 zero-shot and 8
  full-FT top-1 test failures; its bootstrap intervals are consequently wide.
- Phase 15 did not retain aligned clean/corrupted score arrays. Phase 17
  therefore reports no confidence-drop conclusion for shortening or
  occlusion, despite reusing their aggregate robustness results.
- Selective thresholds are offline validation-derived recommendations only;
  they are not hard-coded into an inference API and have not been tested for
  operational abstention costs.
- High-confidence errors and taxonomy-linked confidence use COCO's existing
  same-image relevance proxy. They are not human judgments of semantic
  correctness.
- Phase 18 token importance is local deletion sensitivity. Removing a token
  can change grammar, tokenization, and meaning, so a positive score delta is
  not a causal or human-semantic attribution.
- Phase 18 region sensitivity is a coarse 3×3 mean-colour occlusion map. It is
  perturbation- and grid-dependent, does not identify objects, and is not
  equivalent to attention, gradient saliency, or a complete explanation.
- CLIP similarity is distributed across representation dimensions. A token or
  region with high local sensitivity need not be the human-important concept,
  and low sensitivity does not prove irrelevance.
- Phase 18 uses a small deterministic explanation sample. Its faithfulness and
  casing-consistency checks support only the tested local examples and do not
  establish global explanation reliability.
- Phase 18 high-confidence-error explanations attach observable sensitivities
  but do not establish a single underlying failure cause. Phase 16 taxonomy
  labels remain heuristic.
- Phase 19 groups are dataset-derived observables only: caption length,
  train-caption lexical rarity, a small object/count complexity heuristic,
  and image aspect ratio. They are not protected attributes, and measured
  gaps are descriptive performance disparities rather than fairness findings.
- Phase 19 suppresses strong interpretation for groups with fewer than 20
  retained test queries. The image-to-text aspect strata have only one
  eligible group, so no disparity conclusion is supported there.
- Phase 19's bootstrap intervals quantify query-sample uncertainty within the
  fixed retained Tier 2 test rows. They do not account for new datasets,
  seeds, model families, annotator disagreement, or deployment shift.
- No content-safety classifier, moderation evaluation, red-team exercise,
  private-data privacy audit, accessibility study, or multilingual evaluation
  was run. The system card therefore records these as unresolved deployment
  risks or unevaluated limitations rather than safety evidence.
- Phase 19 high-confidence-error review reuses COCO's same-image relevance
  proxy. A high-confidence retrieval error is a reliability observation, not
  a determination that an image or caption is harmful.

### Phase 20 efficiency limitations

- Peak memory is **PEAK MEMORY NOT RELIABLY MEASURED**. Apple unified-memory
  and MPS allocation behavior was not captured with a reliable peak metric, so
  no fabricated memory number is reported.
- The model-load timing is one offline local MPS profile and is hardware,
  cache, and process-state dependent. It is reported separately from query
  latency and is not a universal service-startup benchmark.
- The cold/warm cache comparison consolidates the retained Phase 10/11 cache,
  encoding, and FAISS Flat measurements on the same local hardware. It is a
  reproducible project profile, not a production SLO or a cross-hardware
  benchmark.
- Float16 storage was tested in memory only. The quality sample retained the
  same R@1/R@5 in both directions, but this does not establish behavior for
  every query, hardware backend, serialization format, or future model.
- The Pareto frontier has only one quality metric and one training-cost metric
  over the tested configurations. It is a fixed-scope decision aid, not global
  optimality and not a substitute for deployment load testing.
- Exact/ANN conclusions reuse the Phase 10 approximately 5,000-image corpus
  and fidelity protocol. Larger corpora may justify ANN after a fresh
  validation and held-out fidelity check.
- LoRA adapter bytes are not total deployment bytes because the base model is
  required. The disk-cleanup artifact is advisory only; no files were deleted
  automatically.

### Phase 21 service limitations

- The service was verified as a single-process local MPS application. No
  production concurrency throughput, autoscaling, network latency, or uptime
  SLO was established.
- The real API benchmark used five warm TestClient requests per direction.
  Its wall-clock values include TestClient overhead; server timing excludes
  network latency. The result is a local engineering profile, not a public
  service guarantee.
- The macOS FAISS/PyTorch process requires the documented
  `KMP_DUPLICATE_LIB_OK=TRUE` compatibility workaround. This is an environment
  risk and should be replaced with a clean single-runtime environment before
  deployment.
- The API has upload and query bounds, safe error responses, and no permanent
  upload storage by default, but it does not implement authentication,
  rate-limiting, abuse prevention, audit-log governance, or private-data
  retention controls.
- Content-safety filtering is **NOT IMPLEMENTED**. Retrieval relevance,
  confidence, or a successful HTTP response must not be interpreted as a
  moderation or safety decision.
- The service exposes only text→image and image→text. Multimodal fusion,
  composed CIRCO retrieval, and image→image semantics were not added; the
  Phase 22 UI remains a thin adapter over those two directions.

### Phase 22 interactive demo limitations

- The Streamlit app is a local research/demo interface over the Phase 21
  service, not a production deployment. It has no authentication, rate
  limiting, abuse controls, moderation, or uptime/concurrency guarantee.
- UI latency evidence measures warm direct in-process service-call wall time
  and separates backend timing from adapter overhead; browser rendering and
  human interaction time are not represented as backend latency.
- The app supports only text→image and image→captions over the retained COCO
  artifact. It does not add fusion, composed CIRCO retrieval, calibrated
  confidence, or image→image retrieval.
- Uploaded bytes are decoded in memory and not persisted by the UI, but the
  process and host remain responsible for access control and memory hygiene.
- The validation Chrome profile rejected automated local file selection because
  the browser extension lacked file-URL access. The image-to-text UI path was
  still exercised through the real Streamlit AppTest harness with an actual
  corpus image; ordinary users may need to grant the browser extension's
  file-URL permission for automated local-file smoke reproduction.
- Content-safety filtering is **NOT IMPLEMENTED**; relevance scores and
  rendered captions are not safety decisions.

### Phase 23 testing and reproducibility limitations

- The GitHub Actions workflow was structurally and locally validated, but no
  hosted GitHub run was executed in this workspace.
- The clean-environment check used an isolated temporary uv environment on the
  same machine and package cache; it is not an independent operating-system or
  hardware reproduction.
- The default suite excludes real-model/local-data smoke by design. The
  optional smoke depends on retained checkpoints, indexes, manifest, image
  bytes, and compatible local model packages.
- The secret scan is a basic regex/configuration check, not a complete
  credential or history audit. Exact duplicate detection and large-file
  classification likewise do not identify semantic image duplicates.
- The current workspace has no committed Git index, so tracked/untracked
  classification is reported honestly from the available Git state and ignore
  rules; this does not simulate a remote repository review.

### Phase 24 deployment and packaging limitations

- Native `uv` is the only deployment path actually validated. The API and UI
  smoke ran on the recorded Apple macOS host with MPS; CPU fallback is
  supported but was not used for the full deployment smoke.
- API cold start was measured at 6.10 seconds in one local run. It is hardware,
  cache, process-state, and model-version dependent, not a service SLO.
- The deployment requires approximately 1.6 GB of local checkpoint, cache,
  index, manifest, and image artifacts, plus the local Hugging Face cache in
  offline mode. These remain outside the package/image.
- Docker was not built or run because the Docker daemon was unavailable.
  Ordinary Docker cannot provide Apple MPS, so the included Dockerfile is only
  CPU portability scaffolding and carries no performance claim.
- The deployment manifest hashes the retained local artifacts, but it cannot
  guarantee identical numerical results or latency across hardware and
  native-library versions.
- Public deployment remains unauthorized and incomplete without
  authentication, rate limiting, TLS/reverse proxy, upload abuse controls,
  content safety, privacy governance, monitoring, and production load tests.

### Phase 25 observability and runtime-reliability limitations

- Structured logs and `/metrics` are lightweight process-local mechanisms.
  They do not provide durable storage, aggregation, alerting, retention
  policy, tracing backend, or a production SLO.
- The real evidence is a local native Apple macOS run: 20 text requests, 20
  image requests, a four-worker concurrency sample, and a short 20-second
  alternating soak. It is not a long-duration uptime, autoscaling, network,
  or multi-process benchmark.
- CPU fallback is validated as a device-selection behavior, while the full
  reliability smoke uses the available local MPS path. MPS/native-library
  latency and shutdown behavior can differ on other systems.
- `RetrievalService.close()` and Streamlit exit cleanup release owned
  references. A nonfatal Python `resource_tracker` leaked-semaphore warning
  was observed in the prior deployment evidence and is investigated in the
  Phase 25 shutdown artifact; it is not silently treated as a clean-library
  guarantee.
- Failure injection covers representative malformed input, invalid settings,
  unready service, model failure, missing checkpoint/index, and incompatible
  metadata. It does not prove resilience to every native FAISS, PyTorch,
  operating-system, or abrupt-kill failure mode.

### Phase 26 final validation limitations

- Final quality metrics are retained Phase 7 held-out measurements, not a
  newly rerun Phase 26 test. Live API smoke validates serving identity and
  deterministic behavior, but its 5,000-image corpus is not the 100-image
  Phase 7 test candidate set.
- Cold start was 11.139 seconds and warm latency was measured on one native
  macOS MPS host. CPU fallback is supported, but no full CPU deployment
  benchmark or cross-platform equivalence claim was made.
- Repeated-query and four-worker evidence is deliberately limited. It is not
  a long-duration uptime, autoscaling, network, multi-process, or production
  throughput study. A nonfatal macOS `resource_tracker` warning may still
  appear during process cleanup.
- Phase 17 confidence/selective-retrieval values are validation-threshold
  diagnostics and the API does not expose calibrated abstention. Phase 15
  robustness uses controlled synthetic corruption; Phase 16 categories are
  heuristic; Phase 18 perturbations are local sensitivity, not causal
  explanation.
- Protected-group fairness, multilingual access, content safety, accessibility,
  and external-domain generalization were not evaluated. Public deployment is
  not production-ready without authentication, rate limiting, TLS/reverse
  proxy, durable monitoring, content safety, privacy governance, and upload
  abuse controls.
- Phase 12B remains `PARTIAL / NON-BLOCKING`: the authorized CIRCO image archive
  was not downloaded because of storage limits, so no CIRCO score is claimed.
