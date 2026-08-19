# Unified retrieval evaluation

Phase 5 defines one versioned evaluation contract for OmniSearch:
`retrieval_eval_v1`. The implementation is in `src/omnisearch/evaluation.py`
and is the single metric/protocol path used by migrated Phase 3 baselines and
Phase 4 COCO and fixture rankings.

## Tasks and relevance

| Task | Query | Candidate | Relevance used here |
|---|---|---|---|
| `text_to_image` | caption | image group | the image group that owns the caption |
| `image_to_text` | image group | caption | every declared caption owned by that image |
| `text_to_text` | caption | caption | same-image captions, with the query caption excluded |
| `image_to_image` | image group | image group | COCO caption metadata supplies no cross-image relevance label; metrics remain unevaluated unless labels are supplied |

The evaluator does not infer semantic relevance from caption similarity,
model scores, or image appearance. A producer must supply an explicit
relevance set for every query. Queries with no relevance are counted and
reported separately rather than silently treated as failures or successes.

## Ranking contract

Each query is represented by a `RankingRecord` containing the query ID,
ordered candidate IDs, scores, relevance IDs, candidate corpus ID, candidate
count, and relevance definition. Scores are sorted in descending order; exact
ties are resolved by candidate ID ascending. This makes rankings stable across
runs and prevents a hidden sort implementation from changing a metric.

The ranked list may be truncated. Precision@K is hits in the returned prefix
divided by K, so omitted positions count as non-hits. Recall@K is the fraction
of the declared relevance set retrieved in the prefix. AP uses the declared
relevance-set size as its denominator over the returned list; when only top-K
results are provided, the result is explicitly reported as truncated MAP. The
report also includes MRR, binary NDCG@K, and first-relevant-rank hit/miss
statistics.

All aggregates are macro averages over queries with declared relevance. This
avoids large candidate groups dominating the result and keeps paired system
comparisons at the same query unit.

## Uncertainty and comparisons

Confidence intervals use deterministic query-level bootstrap resampling. A
paired comparison is accepted only when query IDs, task, split, dataset,
protocol version, candidate corpus, candidate counts, relevance definitions,
and relevance sets agree. The comparison reports right-minus-left point
deltas and paired query-bootstrap intervals. It never compares different
query samples as if they were paired.

## Artifacts and CLI

The canonical command is:

```text
PYTHONPATH=src python -m omnisearch.evaluation --source rankings --rankings PATH --output-dir artifacts/phase5
```

Phase 3 migration is run with `--source phase3`; the Phase 4 fixture/schema
migration is run with `--source phase4`. Results are JSON with schema version
1, plus protocol JSON, Markdown, and CSV summaries where applicable. Every
result records dataset/manifest provenance, split, seed, runtime, hardware,
and the protocol version.

The current real-data artifacts cover COCO `text_to_text`, `text_to_image`, and
`image_to_text`. The Phase 4 fixture results remain explicitly marked
`fixture_only`; the real COCO results are separate and are not fixture claims.

## Phase 15 robustness and distribution shift

Phase 15 keeps the Phase 7 test selection, manifest digest, candidate
corpora, relevance sets, and `retrieval_eval_v1` contract fixed. It evaluates
the zero-shot CLIP checkpoint and the selected full-parameter Phase 7
checkpoint only; no model is retrained. Text corruptions modify text queries
for text-to-image retrieval, while image corruptions modify image queries for
image-to-text retrieval. Candidate embeddings remain clean in both cases.

The predeclared text families are casing, punctuation removal, deterministic
character noise, word deletion, and shortening, each at low and high
severity. The image families are downscale/restore, Gaussian blur, JPEG
round-trip, brightness, center crop/restore, additive noise, and central
occlusion, again at two severities. All variants are generated in memory and
identified by query ID, family, severity, and deterministic seed. For each
condition the artifacts report absolute `corrupted - clean` deltas,
clean-relative degradation where defined, retention, paired query-bootstrap
intervals for R@1/R@5, and top-1/top-5 rank stability.

The distribution shift is declared before metric computation: five extreme
aspect-ratio image groups (`width/height <= 0.75` or `>= 1.3333`) are compared
with five near-square groups (`0.9 <= width/height <= 1.1`) from the same fixed
test subset. The group sets are disjoint, so their comparison is descriptive
and unpaired rather than a paired significance claim. This is a controlled
stress test, not external-domain evidence.

## Phase 16 error analysis and failure taxonomy

Phase 16 is a fixed-scope, artifact-only analysis of the Phase 7 zero-shot
and full-FT rankings. It keeps text-to-image and image-to-text separate and
defines top-1, top-5, severe, rank-severity, regression, robustness, reranker,
and hard-negative failures before counting them. A relevant rank absent from a
retained top-10 list is represented as censored (`rank > 10`), not assigned an
unobserved exact value.

Each record includes the query identity, relevance IDs, candidate IDs and
scores, observed rank properties, transition status, and label provenance.
Mechanical features are limited to observable properties such as caption
length, token rarity, lexical overlap, and caption-set structure. Higher-level
object, spatial, attribute, action, and scene-context labels are heuristic
proxies and are explicitly not human semantic annotations. Root-cause layers
are therefore hypotheses tied to observable evidence, not causal conclusions.

The fixed analysis contains 501 text queries and 100 image queries for each of
two systems and both directions. It retains score-margin categories as
diagnostic similarity gaps rather than probabilities, reports rank movement
with censoring, and uses paired bootstrap summaries where an aligned query
comparison is available. Phase 15 robustness and Phase 9 hard-negative links
are reported with their actual coverage. Phase 11 reranker links preserve its
aggregate/qualitative evidence and disclose that complete paired per-query
Stage-1/reranked rankings were not retained.

## Phase 17 uncertainty, confidence, and selective retrieval

Phase 17 evaluates whether ranking structure predicts the deterministic
top-1 correctness target. A retrieval score is the model's similarity value;
it is not a probability. A confidence proxy is an interpretable score-derived
signal, and a calibrated confidence is a bounded transformation fitted on
validation labels only.

The analysis uses four proxies: top-1 score, `s1 - s2` top-1/top-2 margin,
softmax top-1 mass over the retained top-10 candidates, and
one-minus-normalized entropy of that same softmax distribution. The softmax
quantities are experimental concentration measures over the retained list,
not probabilities over the full candidate corpus. Raw scales are compared
only within each system/direction; cross-system confidence comparisons use the
separately fitted transformation.

For each system and direction, validation selects the proxy by ROC-AUC and
fits a two-parameter logistic transformation after validation-only
standardization. The held-out test records then receive calibrated confidence
without refitting. Reliability bins use ten predetermined bins; empty bins
are omitted from the table and recorded explicitly. ECE is the count-weighted
absolute gap between mean confidence and empirical correctness, and Brier
score is reported only for the bounded transformed confidence. ROC-AUC and
PR-AUC measure discrimination, not calibration.

Selective retrieval accepts a query when calibrated confidence is at least a
threshold and otherwise flags it for abstention. Thresholds are selected on
validation at predetermined target coverages of 100%, 90%, 80%, 70%, and 50%.
The test report contains coverage, selective top-1 accuracy, and risk for the
same validation-derived threshold. Risk-coverage curves accept queries in
descending confidence order. AURC uses the rectangle rule: the mean risk of
all descending-confidence prefixes, where lower is better.

Phase 17 also stores high-confidence errors, low-confidence correct results,
confidence by Phase 16 heuristic failure category, and deterministic examples.
The Phase 15 artifacts contain aligned ranking outcomes but not aligned clean
and corrupted score arrays, so robustness-confidence relationships are
explicitly marked unavailable rather than inferred.

## CIRCO composed-retrieval metrics

Phase 12B uses a separate benchmark adapter because CIRCO's relevance contract
has multiple ground truths and a released target image. For a query with
ground-truth set `G`, target image `g*`, and ranked candidates `r`, CIRCO
mAP@K follows the official evaluation implementation:

`AP@K = sum_i precision@i * 1[r_i in G] / min(|G|, K)`.

Official CIRCO Recall@K is the binary indicator that `g*` appears in the
returned prefix. The adapter also records any-ground-truth recall as a
secondary diagnostic, but does not substitute it for official Recall@K. The
reference image is rejected if it appears in `G`, and all target IDs remain
distinct. These metrics are implemented in `src/omnisearch/phase12b.py` and
are tested independently from the COCO identity metrics.

The official CIRCO test labels are withheld. Phase 12B therefore uses a
deterministic selection/holdout partition inside labeled validation for local
method selection and held-out comparison. A blocked preflight is recorded when
the official COCO unlabeled image archive cannot be acquired compliantly; no
test-server score or fabricated local test score is reported.

## Phase 9 paired evidence

The hard-negative run uses the same query IDs, candidate corpora, relevance
sets, and `retrieval_eval_v1` contract as the Phase 7 full-FT and Phase 8 LoRA
artifacts. Its primary paired comparison is hard-negative full FT versus
standard Phase 7 full FT, with a secondary comparison versus zero-shot CLIP.
Recall@5 deltas are reported with query counts, seed 42, and 200 bootstrap
resamples. Intervals crossing zero are treated as inconclusive; no result is
called reliable solely because its point estimate is positive.

## Phase 13 statistical validation

Phase 13 keeps the canonical `retrieval_eval_v1` query unit and fixed COCO
manifest while varying only the training seed. For each trained system and
seed, retained ranking records support R@1, R@5, R@10, and MRR in both
directions. The zero-shot result is a single non-trained reference, not a
fabricated three-seed sample.

For a paired query metric `m`, the comparison is
`delta_q = m_comparison(q) - m_baseline(q)`. Query-level bootstrap resamples
the same query IDs 200 times to form a 95% interval. A separate paired
sign-flip randomization test uses 2,000 permutations. Holm–Bonferroni is
applied to the declared primary family; none of the 60 seed/direction/metric
tests was rejected at α=0.05. Training-seed mean and sample standard
deviation are computed across exactly three seeds and are descriptive, not a
population estimate.

## Phase 14 ablation evaluation

Phase 14 compares component-on and component-off systems under compatible
COCO/Tier-2 protocols. The full-system-minus-ablation delta is the primary
contribution quantity. Existing Phase 13 bootstrap, permutation, and Holm
artifacts are reused; only the new single-seed hard-negative-ratio ablation
receives fresh paired-query bootstrap intervals. No new multiple-comparison
claim is made for that exploratory run.

The controlled findings are directional. Full fine-tuning adds a stable
text-to-image R@5 gain over zero-shot across all three seeds, but does not
improve image-to-text R@5. LoRA is substantially cheaper in trainable
parameters and roughly half the training time, but is below full FT on the key
text-to-image R@5 metric. Hard negatives have mixed value relative to full FT
and higher cost. The Phase 11 reranker is a strong negative result: its paired
Tier-2 test deltas are negative for R@1, R@5, R@10, and MRR in both directions.
The ratio-25 ablation is single-seed and does not establish a preferred ratio.

## Phase 26 final evaluation boundary

Phase 26 does not alter the held-out evaluation protocol. It reuses the Phase
7 `retrieval_eval_v1` test artifacts after verifying that test data was not
used for checkpoint selection. Text→image uses caption queries, image-group
candidates, and the associated image as relevant; image→text uses image-group
queries, caption candidates, and all captions for that image as relevant.
The retained metrics are R@1/R@5/R@10/MRR = 0.8263/0.9880/1.0000/0.8946 for
text→image and 0.1837/0.8143/0.9303/0.9517 for image→text.

Phase 26 additionally validates the real native deployment path. It checks
checkpoint, manifest, cache, index, dimension, dtype, normalization, and
candidate-unit identity; launches the actual API and Streamlit processes;
checks health/readiness and both directions; and records warm latency and
limited reliability evidence. Live API smoke results are not substituted for
the Phase 7 test metrics because the service uses the full 5,000-image corpus
while the final test artifact declares 100 image candidates.

The confidence, robustness, failure, explainability, and responsible-AI
findings are consolidated as scope-qualified evidence. In particular,
calibrated confidence is not exposed as an API abstention guarantee,
robustness rows use controlled synthetic perturbations, Phase 16 taxonomy
categories are heuristic, Phase 18 perturbations are local sensitivity rather
than causal explanations, and protected-group fairness, multilingual access,
and content safety remain unevaluated/not implemented. Phase 12B has no
quantitative result because its official image archive remains storage-
deferred. See `artifacts/phase26/final_benchmark.json` and
`artifacts/phase26/claims_audit.json`.

### Why image-to-text R@1 and MRR are not interchangeable

For `image_to_text`, the relevance set contains every declared caption owned
by the query image. `recall_at_1` therefore measures the fraction of that
caption set retrieved in a one-item prefix. At rank 1, at most one caption can
be retrieved, so a query with five relevant captions contributes about 0.2 to
Recall@1 even when its first result is relevant. `precision_at_1` answers a
different question: whether the single returned caption is relevant.

MRR uses the rank of the first relevant caption only. It can therefore be high
when the first relevant caption usually appears at rank 1, even though the
top-1 prefix contains only one part of the full relevance set. The retained
image-to-text result illustrates this distinction: Recall@1 is `0.1837`,
precision@1 is `0.92`, MRR is `0.9517`, and the mean first-relevant rank is
`1.14`. These values use the declared `retrieval_eval_v1` definitions; they
are not alternative or post-hoc metric calculations.
