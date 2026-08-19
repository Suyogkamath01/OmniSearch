# Executed experiments

This document is the concise experiment chronology. Machine-readable evidence
remains in the `artifacts/` directories; numbers below are summaries of those
artifacts, not planned or illustrative results.

## Active benchmark

The active benchmark is official COCO 2017 `val2017`, represented by the
5,000-image, 25,014-caption manifest with an internal deterministic
4,000/500/500 image-group split. The historical Flickr30k path is preserved
as a documented optional external dataset because its authorized image
archive was unavailable.

## Baselines and representations

COCO Phase 1–6 executed acquisition validation, minimum EDA, TF-IDF/BM25 and
RGB-histogram baselines, frozen CLIP, the unified evaluator, and frozen text
and vision representation comparisons. Formal image-to-image relevance was
not invented because COCO captions do not provide cross-image labels.

## Phase 7: full fine-tuning

The Tier 2 CLIP run used pretrained `openai/clip-vit-base-patch32`, 800/100/100
train/validation/test image groups, one epoch, seed 42, and full-parameter
AdamW updates. Test Recall@5 was 0.9880 text-to-image and 0.8143 image-to-text.
The checkpoint and paired comparisons are under `artifacts/phase7/`.

## Phase 8: PEFT/LoRA

The real rank-8 PEFT run used the same scope and evaluator. It trained 491,521
parameters versus 151,277,313 for full fine-tuning, a 99.6751% reduction.
LoRA Recall@5 was 0.9780 text-to-image and 0.7983 image-to-text. It therefore
did not match full fine-tuning on this run. The adapter, three-way comparison,
paired bootstrap, and validation merge check are under `artifacts/phase8/`.

## Phase 9: static hard-negative mining

The canonical hard-negative experiment starts from pretrained zero-shot CLIP;
Phase 7 full fine-tuning remains the standard quality baseline and Phase 8
LoRA remains the efficiency comparison. Negatives were mined once from the
800 train image groups with frozen CLIP, using a valid top-5 non-positive pool
and deterministic hash sampling. Same-image captions, exact normalized
caption aliases, positive IDs, and known exact image duplicates were excluded.

The run used 50% explicit mined rows in addition to ordinary in-batch
negatives. Validation mean Recall@5 selected the epoch-1 checkpoint; the test
split was opened afterward.

| System | Text→image R@1 | Text→image R@5 | Image→text R@1 | Image→text R@5 |
|---|---:|---:|---:|---:|
| Zero-shot | 0.8144 | 0.9780 | 0.1857 | 0.8003 |
| Full FT | 0.8263 | 0.9880 | 0.1837 | 0.8143 |
| LoRA | 0.8184 | 0.9780 | 0.1837 | 0.7983 |
| Hard-negative FT | 0.8244 | 0.9800 | 0.1877 | 0.8203 |

Hard-negative minus full-FT Recall@5 was -0.0080 with 95% CI
[-0.0200,+0.0020] for text-to-image and +0.0060 with CI
[-0.0080,+0.0260] for image-to-text. Both intervals cross zero. The run
therefore does not establish superiority over standard full fine-tuning.

Mining took 59.99 seconds. The recorded training-loop wall time was 417.77
seconds including validation/checkpoint-selection overhead; final test
encoding took 5.69 seconds. The full checkpoint was 605,243,647 bytes and the
mined-negative manifest was 737,623 bytes.

The complete Phase 9 evidence is under `artifacts/phase9/`, including the
mined manifest, mining statistics, training history, paired comparisons,
top-rank analysis, false-negative audit, qualitative sample, and provenance.

## Phase 10: exact versus approximate vector retrieval

Phase 10 did not train a model. The primary embedding source was the strongest
validated shared-space representation already available: the Phase 7 full-
fine-tuned `openai/clip-vit-base-patch32` checkpoint. Its 5,000 image and
25,014 caption embeddings were generated once on MPS, L2-normalized, cached,
and reused by every exact and ANN index. The manifest hash, checkpoint hash,
embedding dimension, dtype, and protocol are in `artifacts/phase10/`.

The benchmark used deterministic 100-, 1,000-, and 5,000-image-group corpora
with the active 80/10/10 train/validation/test proportions. Validation groups
were used for ANN configuration selection; test groups were evaluated after
selection. Semantic metrics use all queries in each declared split. Search
latency uses a deterministic first-128-query sample for larger tiers,
repeated three times, and excludes embedding-generation time.

The exact NumPy index is the trusted reference. FAISS Flat, FAISS IVF-Flat
with `nprobe` 1 and 8, and hnswlib HNSW with `M=16`, `efConstruction=100`, and
`efSearch` 8 and 32 all ran successfully. The exact and approximate systems
use the same vectors, queries, candidate IDs, relevance sets, and
`retrieval_eval_v1` ranking contract.

### Tier 3 held-out summary

The following is the 5,000-image-group corpus with its held-out test queries.
Approximate R@10 is neighbor fidelity against exact search; semantic R@5 is
the COCO image-caption retrieval metric.

| Task / configuration | Mean search ms | ANN neighbor R@10 | Semantic R@5 | Exact semantic R@5 |
|---|---:|---:|---:|---:|
| Text→image exact NumPy | 0.965 | 1.000 | 0.5902 | 0.5902 |
| Text→image FAISS Flat | 0.288 | 1.000 | 0.5902 | 0.5902 |
| Text→image IVF nprobe=1 | 0.033 | 0.5046 | 0.3798 | 0.5902 |
| Text→image IVF nprobe=8 | 0.049 | 0.9056 | 0.5578 | 0.5902 |
| Text→image HNSW efSearch=8 | 0.139 | 0.8353 | 0.5290 | 0.5902 |
| Text→image HNSW efSearch=32 | 0.242 | 0.9617 | 0.5730 | 0.5902 |
| Image→text exact NumPy | 12.735 | 1.000 | 0.3578 | 0.3578 |
| Image→text FAISS Flat | 1.533 | 1.000 | 0.3578 | 0.3578 |
| Image→text IVF nprobe=1 | 0.046 | 0.5178 | 0.2266 | 0.3578 |
| Image→text IVF nprobe=8 | 0.089 | 0.9312 | 0.3402 | 0.3578 |
| Image→text HNSW efSearch=8 | 0.154 | 0.7312 | 0.2866 | 0.3578 |
| Image→text HNSW efSearch=32 | 0.262 | 0.9062 | 0.3323 | 0.3578 |

These timings are measured search-only timings on the recorded Apple MPS
machine; they are not end-to-end user latency. FAISS Flat is exact in the
algorithmic sense and is included as a faster exact-engine comparison. At
Tier 3, the strict validation threshold of 0.99 neighbor Recall@10 selected
FAISS Flat for both directions. The result is therefore not a claim that
approximation is superior at this scale.

### Findings and research questions

- Approximate search can be much faster: at Tier 3 IVF nprobe=1 was about 29x
  faster than exact NumPy for text→image and about 276x faster for image→text,
  but its neighbor fidelity was about 0.505 and 0.518.
- Raising IVF `nprobe` improved fidelity to 0.906/0.931 while remaining about
  20x/136x faster than exact NumPy, but it still missed the declared 0.99
  fidelity threshold.
- HNSW showed the expected `efSearch` trade-off: efSearch=32 improved fidelity
  over efSearch=8 but remained below 0.99 at Tier 3.
- Semantic retrieval changed when approximation dropped neighbors. The
  machine-readable qualitative artifact contains deterministic exact matches,
  lower-rank changes, top-rank changes, and observed ANN-caused semantic
  top-5 misses; it does not cherry-pick only successes.
- HNSW had the largest measured build/storage burden at Tier 3. IVF build time
  was higher than Flat because training coarse centroids is an up-front cost.
  Raw vector storage was 10,240,000 bytes for 5,000 image vectors and
  51,228,672 bytes for 25,014 caption vectors; serialized index sizes and
  metadata are recorded per direction and configuration.
- At the current 5,000-image scale, exact search is sufficient when strict
  embedding-space fidelity is required. ANN is valuable as a measured future
  scaling path, not as an unsupported claim of current semantic superiority.
  No million-scale result is inferred.

The complete Phase 10 evidence is under `artifacts/phase10/`: embedding
source/generation metadata, exact baseline, benchmark results, validation
selection, persisted-index checks, scaling results, quality-latency frontier,
qualitative approximation examples, failure analysis, provenance, and the
phase report. No `phase10_audit.md` file was created.

## Phase 11: two-stage retrieval and reranking

The pre-phase audit of the actual Phase 10 report, source, tests, and
provenance passed. Phase 10 used the validated Phase 7 full-fine-tuned CLIP
space, real FAISS 1.15.0/hnswlib 0.8.0 runs, validation-only ANN selection,
and a held-out test scope. Phase 11 therefore reused that source without
training a new Stage-1 retriever. The Phase 10 audit finding is recorded in
the Phase 11 report rather than in a separate `phase11_audit.md` file.

Stage 1 is persisted exact FAISS `IndexFlatIP`, separately materialized for
train, validation, and test candidate corpora. Stage 2 is a 65,729-parameter
pairwise MLP with 512-dimensional query/candidate inputs and a 64-unit hidden
layer. Its features are the elementwise product, absolute difference, and
cosine similarity; its output is an uncalibrated ranking score, not a
probability. It was fit on 4,801 train-only pairs from 800 Tier 2 image
groups, for three AdamW epochs with seed 42 and softplus margin loss. The
validation and test sets were not used for fitting.

Tier 2 validation selected candidate depth 10 by mean MRR:

| Depth | mean validation MRR | mean validation R@5 | mean rerank seconds/query |
|---:|---:|---:|---:|
| 10 | 0.6412 | 0.7145 | 0.001145 |
| 25 | 0.5077 | 0.5418 | 0.001133 |
| 50 | 0.4093 | 0.4142 | 0.001118 |

The selected depth produced a negative held-out result. Stage 1 is the
validated reference and Stage 2 is the reranked system:

| Tier | Direction | Stage 1 R@1 | Reranked R@1 | Δ R@1 | Stage 1 R@5 | Reranked R@5 | Δ R@5 | Stage 1 MRR | Reranked MRR | Δ MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tier 2 | Text→image | 0.8263 | 0.3533 | -0.4731 | 0.9880 | 0.8523 | -0.1357 | 0.8946 | 0.5569 | -0.3377 |
| Tier 2 | Image→text | 0.1837 | 0.1457 | -0.0380 | 0.8143 | 0.6190 | -0.1953 | 0.9517 | 0.8387 | -0.1130 |
| Tier 3 | Text→image | 0.6373 | 0.2403 | -0.3970 | 0.8988 | 0.7265 | -0.1723 | 0.7493 | 0.4420 | -0.3074 |
| Tier 3 | Image→text | 0.1571 | 0.1020 | -0.0551 | 0.6209 | 0.4831 | -0.1378 | 0.8611 | 0.6881 | -0.1730 |

Paired query bootstrap intervals (200 resamples, 95%) confirmed the
degradation. The R@1 intervals were [-0.5230,-0.4271] and
[-0.1695,-0.0658] for Tier 2 text→image and image→text, and
[-0.4207,-0.3710] and [-0.0656,-0.0471] for the corresponding Tier 3
comparisons. The exact machine-readable artifact records full R@1, R@5, and
MRR intervals with explicit system IDs.

Candidate recall explains what the reranker could and could not recover. At
the selected depth on the held-out test set, text→image candidate hit/fraction
was 1.0000/1.0000 for Tier 2 and 0.9636/0.9636 for Tier 3. Image→text
candidate hit rate was 1.0000 and 0.9860, while candidate recall fraction was
0.9303 and 0.7813, respectively. The corresponding oracle analysis placed
all available relevant candidates first and is explicitly labeled
`ORACLE ANALYSIS — NOT A MODEL`; at depth 10 its test R@5 ceilings were 1.0000,
0.9303, 0.9636, and 0.7813 in the same row order as the table above.

Selected-depth quality/latency measurements separated model query encoding,
Stage-1 search, and Stage-2 reranking. Model load time was excluded; re-
encoding agreed with the Phase 10 cache within 5.96e-7 maximum absolute
error. On the recorded Apple MPS run:

| Tier | Direction | Encoding ms/query | Stage 1 ms/query | Rerank ms/query | End-to-end ms/query |
|---|---|---:|---:|---:|---:|
| Tier 2 | Text→image | 2.931 | 0.027 | 1.033 | 3.991 |
| Tier 2 | Image→text | 18.357 | 0.048 | 1.034 | 19.439 |
| Tier 3 | Text→image | 3.068 | 0.049 | 1.022 | 4.139 |
| Tier 3 | Image→text | 19.690 | 0.157 | 1.299 | 21.146 |

The qualitative artifact contains promoted, demoted, unchanged, regression,
and candidate-miss slots. All four selected test rows changed ordering; the
machine-readable failure analysis counts 3,602 ordering changes, 2,783 top-1
changes, and four test-row candidate-miss categories. The dominant observed
failure is reranker ordering failure: relevant candidates were present but
the shallow score displaced them. This phase does not claim that the model
classifies relevance or that its scores are calibrated probabilities.

The complete Phase 11 evidence is under `artifacts/phase11/`: configuration,
train-only checkpoint and history, pair statistics, persisted index manifest,
candidate recall, Stage-1 and reranked results, paired bootstrap comparisons,
oracle analysis, latency breakdown, qualitative examples, failure analysis,
provenance, and the phase report. Phase 12 follows in the next section.

## Phase 12: controlled same-image fusion sanity check

The Phase 11 audit passed from the actual report, source, artifacts, docs, and
regression suite. Phase 12 uses the same validated Phase 7 full-fine-tuned
CLIP checkpoint and Phase 10 cache. No unrelated Sentence-BERT/ResNet spaces
are combined, and no learned fusion model is trained.

The quantitative protocol is intentionally controlled: one image and one
deterministically selected caption belonging to that image form a joint
`(image_query, text_query)`; the target is the same image group in the
split-specific image corpus. This is a controlled aligned-signal identity task,
not a benchmark for arbitrary instructions such as “same object but red.”
The latter are stored only as qualitative examples without correctness
labels. The experiment evaluates image-only, text-only, early weighted-vector
fusion, and late score fusion with alpha values 0.25, 0.50, and 0.75, where
alpha weights the image modality.

Tier 2 validation gave every fusion configuration perfect identity MRR and
R@5. The deterministic tie-break selected late score fusion with alpha 0.25;
early and late fusion produced identical rankings in this protocol because
normalizing a weighted query only multiplies all candidate cosine scores by a
query-level constant.

Held-out results:

| Tier | Variant | R@1 | R@5 | R@10 | MRR | Mean first rank |
|---|---|---:|---:|---:|---:|---:|
| Tier 2 | Image-only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| Tier 2 | Text-only | 0.7900 | 0.9900 | 1.0000 | 0.8672 | 1.460 |
| Tier 2 | Best fusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| Tier 3 | Image-only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| Tier 3 | Text-only | 0.6740 | 0.9120 | 0.9640 | 0.7722 | 1.832 |
| Tier 3 | Best fusion | 0.9960 | 1.0000 | 1.0000 | 0.9980 | 1.004 |

Paired bootstrap comparisons use 200 resamples and report fusion minus the
control. Against text-only, Tier 2 deltas were +0.2100 R@1, +0.0100 R@5,
and +0.1328 MRR; Tier 3 deltas were +0.3220, +0.0880, and +0.2258. Their
R@1 confidence intervals were [0.1400,0.2900] and [0.2800,0.3640], and their
MRR intervals were [0.0871,0.1872] and [0.1965,0.2529]. Against image-only,
Tier 2 deltas were exactly zero; Tier 3 deltas were -0.0040 R@1, 0 R@5,
and -0.0020 MRR, with intervals including zero. The improvement claim is
therefore only relative to the weaker text-only control under this identity
protocol; fusion does not improve the image-only ceiling.

Modality dominance analysis showed the expected alpha effect. At Tier 3,
best-fusion top-10 overlap with image-only was 0.5087, 0.6728, and 0.8340 at
alpha 0.25, 0.50, and 0.75, while overlap with text-only was 0.6655, 0.5033,
and 0.4106. The same monotonic pattern occurred at Tier 2. Image-only top-1
changed in 0%, 0%, and 0% of Tier 2 queries and 0.4%, 0%, and 0% of Tier 3
queries; text-only top-1 changed in 32.2–32.6% of Tier 3 queries. This shows
that both modalities affect ranking, but the image signal dominates the
controlled identity target.

Selected-fusion latency on MPS, with model load excluded, was 33.452 ms
image encoding + 3.029 ms text encoding + 0.010 ms fusion + 0.028 ms exact
retrieval per Tier 2 query, with 36.519 ms end-to-end; Tier 3 measured
19.271 ms + 3.078 ms + 0.021 ms + 0.100 ms, with 22.470 ms end-to-end.
Fusion computation is negligible compared with encoding.
Cache re-encoding maximum absolute error was below 6e-7.

The qualitative artifacts contain four `QUALITATIVE CONFLICT ANALYSIS`
queries formed by pairing an image with a caption from a different image, and
four `QUALITATIVE COMPOSITIONAL QUERY` examples (“similar scene at night,”
“same object but red,” “with two people,” and “indoors”). They record IDs,
texts, paths, and top results only; they do not fabricate correctness labels.
Diagnostic sensitivity found that changing text while holding the image fixed
changed selected-fusion top-1 for 53.8% of Tier 3 queries, while changing the
image while holding text fixed changed it for 64.6%.

The complete Phase 12 evidence is under `artifacts/phase12/`: fusion configs,
validation selection, controls and fusion results, paired comparisons,
overlap/dominance analysis, query sensitivity, latency, conflict and
compositional examples, failure findings, provenance, and the phase report.
Phase 13 statistical validation follows this controlled result and does not
include Phase 12B.

## Phase 12B: proper composed image retrieval evaluation

Phase 12B was implemented to correct the central limitation of the original
Phase 12. The original experiment remains valid as a controlled same-image
fusion sanity check, but its image-only ceiling is not evidence for real
compositional retrieval.

The selected benchmark is CIRCO, the official open-domain composed image
retrieval benchmark built from COCO 2017 unlabeled images. The official
repository, annotation API, official evaluation site, license statement, and
COCO unlabeled archive path were verified before any dataset download. CIRCO
material is stated by its repository to be CC BY-NC 4.0; the underlying COCO
images retain their originating Flickr terms and copyrights. The verified
repository commit was `ba9a9346a8840513bc5d0beccdaf6dd0f5c3c6fa`.

The real run was closed before download. The official unlabeled image archive
advertised a content length of 20,126,613,414 bytes (18.74 GiB); the current
preflight records insufficient safe local storage for the archive, extracted
gallery, and index. No unofficial mirror and no substitute COCO validation
image corpus were used. The reproducible preflight, report, and formal
closure are in `artifacts/phase12b/preflight.json`,
`artifacts/phase12b/phase12b_report.json`, and
`artifacts/phase12b/closure.json`.

The implemented adapter accepts CIRCO `reference_img_id`,
`relative_caption`, `target_img_id`, `gt_img_ids`, query ID, split, and
benchmark-provided `semantic_aspects`. It rejects a reference image appearing
in the released ground-truth set. CIRCO's official metric semantics are
implemented separately from `retrieval_eval_v1`: mAP@K uses all `gt_img_ids`,
while official Recall@K tests the released `target_img_id`; multiple ground
truths are not collapsed to one label. The intended real protocol uses a
deterministic selection/holdout partition inside labeled CIRCO validation so
alpha selection is not evaluated on the same queries. The official CIRCO test
ground truth is withheld and no test result is claimed.

The comparison ladder is image-only, text-only, early weighted-vector fusion,
and late weighted-score fusion at alpha 0.25, 0.50, and 0.75. Exact FAISS
Flat is the intended candidate index. Because the image archive is absent,
there are no real image-only, text-only, early-fusion, late-fusion,
statistical, dominance, qualitative, or latency results for Phase 12B.
Fixture smoke execution validates the code path only and is not benchmark
evidence.

Phase 12B quality gate: **PARTIAL / NON-BLOCKING**. The official access path
and metric/adapter implementation pass, but genuine ground-truth evaluation,
controls, fusion results, and real latency remain unevaluated. This limitation
does not block Phase 13 because Phase 13 evaluates the fixed local COCO
retrieval systems and explicitly excludes Phase 12B.

## Phase 13: statistical validation and multiple seeds

Phase 13 predeclared the exact training seed plan `{42, 123, 2026}` before
examining new results. Seed 42 was reused only because its Phase 7, 8, and 9
artifacts matched the fixed manifest and Tier-2 protocol. Seeds 123 and 2026
were freshly trained for full fine-tuning, rank-8 LoRA, and the existing
static hard-negative configuration. The same seed-42-selected 800/100/100
image-group subsets were used for every seed; only stochastic training state
varied. Phase 12B was excluded.

Primary metrics below are mean ± sample standard deviation across the three
trained seeds:

| System | Text→image R@1 | Text→image R@5 | Image→text R@1 | Image→text R@5 |
|---|---:|---:|---:|---:|
| Full FT | 0.8197 ± 0.0061 | 0.9854 ± 0.0046 | 0.9200 ± 0.0000 | 1.0000 ± 0.0000 |
| LoRA | 0.8150 ± 0.0042 | 0.9780 ± 0.0000 | 0.9300 ± 0.0100 | 1.0000 ± 0.0000 |
| Hard-negative FT | 0.8230 ± 0.0023 | 0.9814 ± 0.0023 | 0.9300 ± 0.0173 | 1.0000 ± 0.0000 |

Full FT's text→image R@5 gain over zero-shot was positive for all three
seeds, but its image→text R@5 did not change. LoRA matched zero-shot on both
R@5 measures and remained below full FT on text→image R@5 for every seed.
Hard-negative training gave a small positive text→image R@5 delta over
zero-shot for every seed, but its comparison with full FT was seed-sensitive;
image→text R@5 was unchanged. No Holm-corrected primary permutation test was
rejected at α=0.05.

The paired query bootstrap used 200 resamples per seed and the paired
sign-flip permutation test used 2,000 permutations. These query-level
intervals are not pooled with the three-seed variability estimates. Mean
training time was approximately 171 s for full FT, 83 s for LoRA, and 345 s
for hard-negative FT on the recorded MPS host. The complete per-seed rows,
histories, comparisons, corrections, win/loss/tie counts, and provenance are
under `artifacts/phase13/`.

## Phase 14: controlled ablations

Phase 13 was audited independently before Phase 14 substantive work. The
audit passed: Phase 12 remains PASS, Phase 12B remains PARTIAL/NON-BLOCKING,
no real CIRCO result is claimed, the fixed COCO manifest and seed-42-selected
Tier-2 split are intact, and the Phase 13 mean/sample-standard-deviation and
paired-test artifacts were recomputed successfully. The durable audit is
`artifacts/phase14/pre_phase_audit.json`; no `phase14_audit.md` was created.

The ablation study reuses the three-seed Phase 13 evidence for full FT versus
zero-shot, full FT versus rank-8 LoRA, and standard full FT versus 50%
hard-negative FT. It also reuses the Phase 11 Stage-1 versus Stage-1 plus
reranker paired result. All comparisons use the same COCO manifest
(`09a2c1e56eb1a628b2ead16f064510d713f81aff5ee2f2d09b4ca8993bba3b43`),
`retrieval_eval_v1`, and Tier-2 800/100/100 image-group scope.

One new true ablation was run because it changes only the explicit
hard-negative ratio: 25% hard-negative FT, seed 42, one epoch, with the
existing frozen seed-42 mined manifest. The 0% point is the existing standard
full-FT result and the 50% point is the existing Phase 9/13 result; no
redundant retraining was performed.

| Comparison | T→I R@1 | T→I R@5 | I→T R@1 | I→T R@5 | Primary delta | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| Full FT − zero-shot | 0.8197 | 0.9854 | 0.9200 | 1.0000 | T→I R@5 +0.0073 | KEEP for T→I quality; not uniform |
| Full FT − LoRA | 0.8197 | 0.9854 | 0.9200 | 1.0000 | T→I R@5 +0.0073 | LoRA is OPTIONAL for efficiency |
| Hard-negative 50% − standard FT | 0.8230 | 0.9814 | 0.9300 | 1.0000 | T→I R@5 −0.0040 | OPTIONAL; mixed gain and added cost |
| Stage 1 + reranker − Stage 1 | 0.3533 | 0.8523 | 0.1457 | 0.6190 | T→I R@5 −0.1357 | REMOVE / NOT RECOMMENDED |

The first three rows are Phase 13 means; the reranker row is the preserved
Phase 11 Tier-2 test result. Full FT's text-to-image R@5 gain over zero-shot
is positive for all three seeds. LoRA uses 491,521 trainable parameters
versus 151,277,313 for full FT and roughly half the recorded training time,
but remains below full FT on text-to-image R@5 for every seed. Hard-negative
training is more expensive and its comparison with standard FT is
seed-sensitive. The reranker degrades every reported ranking metric in both
directions.

The new ratio-25 run produced T→I R@1/R@5 = 0.8303/0.9820 and I→T R@1/R@5
= 0.9400/1.0000 at seed 42. Paired bootstrap intervals crossed zero for the
ratio-50 versus ratio-25 primary deltas; this is exploratory single-seed
evidence, not a ratio-selection claim. Ten deterministic qualitative query
comparisons are retained using fixed evenly spaced sorted test IDs, and
Phase 11 promoted/demoted/regression cases are preserved separately. Full
artifact details are under `artifacts/phase14/`.

## Phase 15: robustness and distribution shift

Phase 14 was audited before Phase 15 and passed. Phase 12B remains
`PARTIAL / NON-BLOCKING`; no CIRCO result is included. Phase 15 froze the
same COCO manifest (`09a2c1e56eb1a628b2ead16f064510d713f81aff5ee2f2d09b4ca8993bba3b43`),
the seed-42-selected 100 test image groups and 501 caption queries, and the
`retrieval_eval_v1` evaluator. The only evaluated systems were cached
zero-shot CLIP and the cached Phase 7 full-parameter checkpoint. No training,
model download, new model family, or corrupted dataset was used.

The evaluation generated ten text-query conditions (casing, punctuation,
typo noise, word deletion, and shortening at low/high severity) and fourteen
image-query conditions (resize, blur, JPEG, brightness, crop, noise, and
occlusion at low/high severity). Candidates stayed clean. Every corruption
was deterministic and retained its original query ID, family, severity, and
seed in `artifacts/phase15/corruption_manifest.json`.

The clean references were the actual Phase 7 test rankings:

| System | Text→image R@1/R@5 | Image→text R@1/R@5 |
|---|---:|---:|
| Zero-shot | 0.8144 / 0.9780 | 0.9300 / 1.0000 |
| Full FT | 0.8263 / 0.9880 | 0.9200 / 1.0000 |

The strongest observed text-query degradation for both systems was high
severity shortening: text→image R@1 fell by 0.3553 for zero-shot and 0.3473
for full FT; R@5 fell by 0.2535 and 0.2575, respectively. The strongest
image-query degradation was high occlusion: image→text R@1 fell by 0.3000
for both systems. These are observed condition deltas, not claims of
statistical superiority between the two models. Paired 200-resample query
bootstrap intervals are in `bootstrap_comparisons.json`; rank preservation,
top-5 overlap, and censored relevant-rank displacement are in
`rank_stability.json`.

The controlled shift was declared from image dimensions before model metrics:
five extreme-ratio groups versus five near-square controls, selected by
ascending image ID. Text→image R@1 was 0.96 on the shifted groups versus
1.00 on controls for both systems (delta −0.04); image→text metrics were
1.00/1.00/1.00/MRR 1.00 for both groups and both systems. Because the groups
are disjoint and small, this is descriptive stress-test evidence, not an
external generalisation result. The complete machine-readable record is in
`artifacts/phase15/`.

Phase 15 quality gate: **PASS**. The phase is evaluation-only and does not
claim robustness beyond the tested synthetic conditions or the small
metadata-defined shift.

## Phase 16: error analysis and failure taxonomy

Phase 15 was audited before analysis and passed. Phase 16 reused the fixed
seed-42 COCO test scope, the Phase 7 zero-shot/full-FT rankings, the Phase 9
hard-negative rankings, the Phase 11 reranker summaries and qualitative
examples, and the Phase 15 robustness artifacts. It did not download data,
train a model, or begin Phase 17.

Failure definitions were fixed by rank: top-1 failure means the first result
is not relevant, top-5 failure means no relevant result is in the first five,
and severe failure means the first relevant result is absent from the retained
top-10 ranking. Ranks outside top-10 are therefore censored lower bounds, not
invented exact ranks. Low, medium, and high severity correspond to ranks
2--5, 6--10, and censored/greater-than-10, respectively.

The analysis produced 1,202 query-level records, keeping text-to-image and
image-to-text separate and retaining both zero-shot and full-FT systems. It
records transitions, rank movement, score margins, fixed-scope query
features, and explicit taxonomy label provenance. Mechanical labels describe
observable text/caption structure; heuristic labels such as object, spatial,
attribute, action, and scene-context categories are marked as heuristic and
are not human semantic ground truth.

On text-to-image, zero-shot had 93/501 top-1 failures and 11/501 top-5
failures; full FT had 87/501 and 6/501. Severe failures were 4 for zero-shot
and 0 for full FT. On image-to-text, top-1 failures were 7/100 for zero-shot
and 8/100 for full FT; both systems had zero top-5 failures. Full FT improved
the text-to-image failure rate at the first five ranks, but two image-to-text
queries regressed at top-1 while one improved.

Phase 15 links show that retained worst-condition examples are diagnostic,
not a full corrupted-query population: high shortening produced the largest
text degradation and high occlusion the largest image degradation. Phase 11
links preserve the actual aggregate reranker degradation, while exact
per-query Stage-1-success-to-reranker-break intersections are unavailable
because the canonical Phase 11 artifact did not retain complete paired
rankings. Phase 9 links compare the real hard-negative rankings with the
zero-shot/full-FT records and expose both fixes and regressions.

The deterministic qualitative sample contains ten examples spanning easy
successes, near misses, severe failures, zero/full transitions, robustness
failures, reranker-induced changes, hard-negative changes, and ambiguous
multi-caption cases. Corrective priorities are to inspect lexical/object and
spatial confusion, preserve the top-5 gains of full FT, treat shortening and
occlusion as robustness targets, and avoid the evaluated reranker protocol.
These are analysis priorities rather than new training claims.

Phase 16 quality gate: **PASS**. Complete machine-readable artifacts are
under `artifacts/phase16/`; the phase remains evaluation-only.

## Phase 17: uncertainty, confidence, and calibration

Phase 16 was checked before this analysis and passed. Phase 17 uses the same
seed-42 COCO Tier-2 held-out scope: 100 validation image groups for fitting
calibration and 100 retained test image groups for final evaluation, with 501
caption queries and 100 image queries per system/direction. No model was
trained and no dataset was downloaded. Validation inference reused the
existing CLIP model and Phase 7 full-FT checkpoint; test rankings and scores
were reused from Phase 7.

Raw CLIP similarities remain retrieval scores, not probabilities. The tested
confidence proxies were top-1 score, top-1/top-2 margin, retained-top-10
softmax top-1 mass at temperature 1.0, and one-minus-normalized retained-top-10
entropy. A validation-only ROC-AUC selection chose top-1/top-2 margin for both
text-to-image systems and entropy confidence for both image-to-text systems.
The selected proxy was transformed with a validation-fitted logistic
calibrator; test labels were not used for fitting or threshold selection.

Held-out discrimination for calibrated confidence was:

| System | Direction | ROC-AUC (95% bootstrap interval) | PR-AUC |
|---|---|---:|---:|
| Zero-shot | text→image | 0.826 (0.780, 0.863) | 0.955 |
| Full FT | text→image | 0.839 (0.795, 0.881) | 0.962 |
| Zero-shot | image→text | 0.856 (0.694, 0.988) | 0.988 |
| Full FT | image→text | 0.917 (0.819, 0.999) | 0.992 |

The image-to-text intervals are wide because only 7 zero-shot and 8 full-FT
test queries are top-1 failures. Held-out ECE/Brier were 0.029/0.119 for
zero-shot text→image and 0.040/0.111 for full FT; image→text was
0.101/0.079 for zero-shot and 0.068/0.051 for full FT. These are metrics of
the explicitly transformed confidence, not raw similarity.

Validation-selected abstention thresholds at the 50% target coverage point
were supported on held-out test data for all four system/direction pairs. Test
coverage was 0.517 for both text→image systems and 0.470/0.440 for zero-shot
/full-FT image→text. Test selective top-1 accuracy was 0.954/0.969 for
zero-shot/full FT text→image and 1.000 for both image→text conditions. AURC
was 0.0575/0.0495 for zero-shot/full FT text→image and 0.0146/0.0115 for
zero-shot/full FT image→text. This supports offline flagging at these
validation-derived thresholds, not a production API policy.

There were 15 retained high-confidence top-1 errors at the fixed diagnostic
threshold of 0.80. Phase 17 also stores ten deterministic qualitative examples
including high- and low-confidence successes/errors and fine-tuning confidence
changes. Phase 15 did not retain aligned clean/corrupted score arrays, so no
confidence-drop claim is made for shortening or occlusion; their aggregate
robustness failures remain linked with that limitation.

Phase 17 quality gate: **PASS**. Machine-readable artifacts are under
`artifacts/phase17/`; Phase 18 has not started.

## Phase 18: explainability and retrieval interpretation

Phase 17 was audited before this analysis and passed. Phase 18 reused its
validation-only confidence records, calibrated confidence, high-confidence
errors, Phase 7 test rankings/checkpoint, and the fixed COCO Tier-2 test
scope. No model was retrained and no dataset was downloaded. The analysis
used deterministic selected queries from both directions and both systems.

The core explanation record reports the query, top retrieved item, relevant
item(s) where retained, top-1 score, observed relevant score, top-1/top-2
margin, target rank, calibrated confidence, and success/severity. These are
observable retrieval facts, not claims about a complete causal explanation of
CLIP.

Text explanations used whitespace/punctuation token deletion. For token `j`,

`importance_j = relevant_score(original) - relevant_score(without j)`.

Rank degradation is the perturbed first-relevant rank minus the original rank.
The run evaluated 226 real token deletions across 10 deterministic text query
IDs per system. The largest observed sensitivities were content-bearing words
such as `ocean`, `pheasants`, `birds`, `door`, and `Goodmayes`; this is a
sample-level association, not a semantic attribution ground truth. In the
selected sample, action/object heuristic tokens generally had positive mean
score sensitivity, while the two observed attribute-token cases were slightly
negative. Stopword-like terms were usually lower-impact but were not uniformly
neutral.

Image explanations used a deterministic 3×3 mean-colour rectangular occlusion
grid. The run produced 72 real region perturbations across four image query
IDs per system. The largest sensitivities were spatially localized; for
example, image `463730` showed strong changes in the upper-right and central
grid cells. These cells are not object labels and the map is not a complete
saliency explanation.

Text counterfactuals removed matched object/action/attribute/spatial heuristic
tokens where available, producing 42 edits. Image counterfactuals reused the
72 region occlusions. The labels remain heuristic and are explicitly not
human semantic annotations.

The perturbation-faithfulness check compared the most and least score-
sensitive deletion/occlusion within each example. Score- and rank-order
support were 100% for the 10 zero-shot/full-FT token examples and the four
zero-shot/full-FT image-region examples. Mean score-degradation differences
were 0.0463/0.0471 for zero-shot/full-FT tokens and 0.0319/0.0289 for
zero-shot/full-FT regions. This is evidence for local perturbation sensitivity
under the declared procedure, not proof of causal faithfulness.

Under uppercase casing perturbation, the mean top-three token overlap was
1.00 across eight selected records. This compact consistency result is
descriptive and does not establish explanation stability generally.

Seven selected Phase 17 high-confidence errors received attached token or
region sensitivities. The qualitative set contains 12 deterministic examples
covering successful and failed text→image and image→text retrieval, high- and
low-confidence errors, and full-FT rank improvements/regressions. All
explanations use cautious observable-evidence language.

Phase 18 quality gate: **PASS**. Machine-readable artifacts are under
`artifacts/phase18/`.

## Phase 19: responsible AI, bias, and safety analysis

Phase 18 was audited before this analysis and passed. Phase 19 reused the
retained Phase 17 test confidence rows, Phase 17 high-confidence errors and
validation-only selective thresholds, Phase 16 taxonomy fields, Phase 18
local explanation links, the Phase 7 ranking/checkpoint hashes, and the
canonical COCO manifest. It did not train, download data, or run a new model.

The analysis used 1,202 retained test query rows: 501 text→image and 100
image→text queries for each of zero-shot and full-FT. It declared a minimum
group size of 20 and used deterministic 200-resample percentile bootstrap
intervals. Groups were derived only from caption length (short/medium/long),
train-caption lexical rarity (rare/medium/common), a fixed object/count
complexity heuristic, and image aspect ratio. No protected labels were
fabricated or inferred. Small groups remain descriptive only.

On the fixed test scope, full FT minus zero-shot top-1 was +0.012 for
text→image and -0.010 for image→text; full FT minus zero-shot MRR was +0.008
and -0.005 respectively. These are held-out system comparisons, not causal or
fairness claims. The largest eligible text→image top-1 gaps were 0.166/0.153
for zero-shot/full-FT aspect strata and 0.113/0.111 for zero-shot/full-FT
lexical-rarity strata. Image→text aspect comparisons had only one eligible
group, so they were marked **INSUFFICIENT EVIDENCE**. These gaps are measured
performance disparities conditional on the COCO same-image relevance proxy,
not evidence of unfairness.

Phase 19 retained all 15 Phase 17 high-confidence top-1 errors at the 0.80
calibrated-confidence threshold; seven have attached Phase 18 local token or
region evidence. The artifact records explicitly decline to determine whether
any example is harmful because no content-safety classifier or human safety
review was run. The responsible-AI matrix therefore separates measured
retrieval risk from potential deployment risks: privacy/rights, misuse,
content safety, multilingual access, accessibility, and generalization beyond
COCO are limitations or deployment concerns, not completed safety evaluations.

Phase 19 quality gate: **PASS**. Machine-readable artifacts are under
`artifacts/phase19/`; the concise system card is `docs/system_card.md`.

## Phase 20: efficiency and resource optimization

Phase 19 was audited before this analysis and passed: its report, validator,
provenance, responsible-AI matrix, system card, and the retained Phase 7–11,
17–19 resource artifacts were readable and internally consistent. Phase 20 is
evaluation-only. It did not train, download data, build a new index, or add a
model family.

The canonical resource comparison covers frozen zero-shot CLIP, Phase 7 full
fine-tuning, Phase 8 rank-8 LoRA, and Phase 9 hard-negative full fine-tuning.
The CLIP base has 151,277,313 parameters; its calculated fp32 parameter payload
is 605,109,252 bytes. The full-FT checkpoint is 605,242,499 measured bytes.
The LoRA adapter is only 1,988,122 measured bytes and has 491,521 trainable
parameters, but the base CLIP checkpoint is still required, so the adapter is
not the total deployable model size. The hard-negative checkpoint is
605,243,647 measured bytes. Peak memory was not reported as a number because
Apple unified-memory/MPS peak usage was not reliably measured.

One offline cached model-load profile was run on MPS. Base CLIP load took
5.964 seconds and base plus full-FT state restoration took 6.590 seconds;
there was no forward pass or optimizer. These cold-start numbers are kept
separate from query latency. In the retained Tier 2 profile, text encoding is
2.885 ms/query calculated from 1.445 seconds over 501 queries, image encoding
is 20.459 ms/query calculated from 2.046 seconds over 100 queries, and FAISS
Flat search is 0.074 ms/query for text→image and 0.288 ms/query for
image→text. Thus cold encode+search is 2.959 ms/query and 20.748 ms/query,
while warm cached search is only the search component. Reranking adds about
1.0 ms/query and reduced held-out R@1/R@5 in both directions, so it is
disabled by the recommended configuration.

Caching was quantified as cold encode+search versus warm cached search. The
ratios are 39.9× for text→image and 72.0× for image→text; model load is not
included in either ratio. The persisted cache contains 5,000 image vectors
and 25,014 caption vectors at 512 dimensions in float32. Dense payload sizes
are calculated as 9.77 MiB and 48.86 MiB, with measured `.npy` sizes of
10,240,128 and 51,228,800 bytes. A float16 cache was evaluated in memory with
float32 accumulation: R@1/R@5 were unchanged in both directions, top-1
agreement was 1.0, top-10 agreement was 0.986 for text→image and 0.970 for
image→text, and calculated dense storage would be halved. No float16 cache
file was persisted or adopted in production.

Phase 10's exact-versus-ANN evidence was reused rather than rebuilt. FAISS
Flat retained exact neighbor recall of 1.0 at Tier 3 in both directions and
is the current default. IVF/HNSW settings were faster but lost measurable
neighbor and/or semantic fidelity under the retained threshold, so ANN is
optional scale-out evidence rather than the default at approximately 5,000
images. The quality/cost frontier uses the retained three-seed training
costs: full FT mean 171.3 seconds, LoRA 83.4 seconds (0.487× full FT), and
hard-negative FT 345.2 seconds before its measured mining overhead, or 365.2
seconds consolidated (2.015× full FT training cost). The hard-negative result
is dominated by full FT in the fixed mean-R@5 versus cost comparison; LoRA is
an optional constrained-adaptation path, not an automatic quality win.

The recommended configurations are: (1) quality default — full-FT CLIP,
cached embeddings, and FAISS Flat; (2) efficiency option — base CLIP with an
optional LoRA adapter, cached embeddings, and FAISS Flat; and (3) lightweight
baseline — frozen zero-shot CLIP, cached embeddings, and exact search. No
production optimization was switched on; Phase 20 measured existing cache
reuse and evaluated float16 safely in memory only. Disk cleanup is advisory:
keep canonical checkpoints, the selected adapter, cache, exact indexes, and
scientific evidence; approximate indexes and exploratory checkpoints are
regenerable/optional; no deletion was performed.

Phase 20 quality gate: **PASS**. Machine-readable artifacts are under
`artifacts/phase20/`. At the time of that Phase 20 run, Phase 21 had not yet
started; the subsequent Phase 21 API and Phase 22 Streamlit demo are recorded
in their own artifact directories and reproduction contracts.

## Phase 26: end-to-end system validation and final benchmark

Phase 26 first audited the Phase 25 PASS gate, then froze the final quality
configuration: Phase 7 full-FT CLIP, 512-dimensional L2-normalized float32
embeddings, cached corpus vectors, exact FAISS Flat inner-product search, and
reranking disabled. No model training, dataset download, or test tuning
occurred.

The retained Phase 7 held-out test results remain the final quality numbers:
text→image R@1/R@5/R@10/MRR are 0.8263/0.9880/1.0000/0.8946 over 501 caption
queries and 100 image candidates; image→text values are
0.1837/0.8143/0.9303/0.9517 over 100 image queries and 501 captions. These are
marked `REUSED_MEASURED_PHASE7`, not presented as a new Phase 26 benchmark.

The actual native deployment selected MPS and reached `/ready` in 11.139
seconds. Ten warm requests per direction were measured after startup.
Text→image server latency mean/median/p95 was 12.31/11.62/15.46 ms;
image→text was 27.50/24.48/37.00 ms. Health/readiness, three deterministic
text queries, a real image query, Streamlit health, repeated-query stability,
and an eight-request four-worker check passed. Result consistency is scoped to
model/backend identity and deterministic smoke IDs because the live service
corpus has 5,000 images while the Phase 7 test candidate set has 100.

Phase 26 consolidates Phase 15 robustness, Phase 16 failures, Phase 17
confidence/selective retrieval, Phase 18 perturbation evidence, Phase 19
responsible-AI limitations, and Phase 20 efficiency decisions. Phase 12B
remains `PARTIAL` because the official CIRCO image archive was not downloaded
under the storage constraint. Evidence is in `artifacts/phase26/`; the
quality gate is **PASS**.
