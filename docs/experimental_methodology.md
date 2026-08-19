# Experimental methodology

## Comparison ladder

Every later experiment should add one controlled capability at a time:

1. classical text/image baseline;
2. frozen pretrained CLIP zero-shot baseline;
3. transformer text and vision representation comparisons;
4. contrastive fine-tuning;
5. parameter-efficient adaptation where justified;
6. hard-negative mining;
7. exact versus approximate retrieval;
8. candidate reranking;
9. text/image query fusion;
10. multi-seed, ablation, robustness, uncertainty, and responsible-AI analyses.

Phase 3 evaluates TF-IDF and BM25 over the held-out COCO test caption corpus
using a deterministic 256-query subset. Its relevance definition is limited to
the other captions for the same image group. The handcrafted image descriptor
is executed on the real COCO test image groups. No cross-modal classical
baseline is forced when the alignment would be artificial.

The frozen CLIP result is a particularly important reference: later gains must be compared against it, not only against the weakest classical model. The current real run evaluates all 500 internal COCO test images and their 2,501 captions.

## Split and fitting protocol

The split key is `image_id`. All captions for an image remain together, including the small number of official COCO groups with more than five captions. Training-only fitting includes vocabulary construction, feature scaling, learned projection heads, negative-mining statistics, and thresholds. The validation set is used for model selection and early stopping; the test set is opened once per declared experiment family. An external set, including future Flickr30k, is never used to tune the primary model.

## Metrics

For image-to-text retrieval, each image query has up to five relevant caption IDs. For text-to-image retrieval, each caption query has one relevant image ID. The evaluator will report:

- Recall@1, Recall@5, and Recall@10;
- median rank and mean rank;
- NDCG@10 using the declared relevance map;
- mean reciprocal rank as a secondary ranking measure;
- index build time, warm-query p50/p95 latency, peak memory, and index size for retrieval benchmarks.

Metrics are reported separately for each direction and then summarized; a single averaged number must not hide a directional failure. For approximate indexes, recall is evaluated against exact-search neighbors under the same embeddings.

## Uncertainty and comparison

The main interval unit is the image query group, not an independently sampled caption that shares the same image. Later multi-seed experiments will report the seed list and per-seed scores. Paired bootstrap or a paired permutation test will compare two systems on the same queries. Multiple comparisons will be declared before the ablation family is run and corrected where appropriate. Statistical significance will not be used as a substitute for practical effect size.

## Robustness and error analysis

After the primary benchmark is stable, evaluate image corruption/resize changes, caption perturbations, near-duplicate or hard-negative subsets, and the external dataset if available. Error analysis will sample false positives and false negatives by a fixed, documented rule and create a failure taxonomy. It must include qualitative examples with IDs and source terms, not only a gallery of favorable cases.

## Reproducibility contract

Each experiment artifact must include:

- Git revision and dirty-worktree status;
- Python and dependency versions;
- device and batch configuration;
- random seed(s);
- dataset source, access date, manifest checksum, and split checksum;
- model name and revision/checksum;
- preprocessing configuration;
- metric implementation version;
- actual start/end time and output paths.

No artifact may contain invented metrics. If a run is interrupted or a dependency is unavailable, it is recorded as not evaluated.

## Phase 7 fine-tuning protocol

Phase 7 uses the canonical COCO split manifest with a deterministic Tier 2
scope of 800 train, 100 validation, and 100 test image groups. The seed is 42;
per-split limits are selected by a SHA-256 ordering of image IDs, so the
selection is reproducible and never truncates a caption group. Training uses
one deterministic caption per image per epoch. Validation uses all captions
for validation image groups and selects the checkpoint by mean Recall@5 across
text-to-image and image-to-text retrieval. Test groups are materialized only
after selection.

The executed configuration is full-parameter CLIP fine-tuning with AdamW,
one epoch, learning rate `1e-6`, weight decay `0.01`, batch size 2, gradient
accumulation 4, effective batch size 8, fp32, and MPS. The smoke run also
executes forward, symmetric loss, backward, finite-gradient, update,
checkpoint, validation, and held-out evaluation checks. The real run records
the training history, selected checkpoint, runtime, paired bootstrap deltas,
and before/after qualitative categories in `artifacts/phase7/`.

This is a one-seed, one-epoch Tier 2 result. It does not support claims about
Tier 3 fine-tuning, multi-seed stability, PEFT, hard negatives, ANN search,
reranking, or production latency.

## Phase 11 two-stage retrieval protocol

Phase 11 starts from the validated Phase 10 embedding cache and uses exact
FAISS Flat retrieval as Stage 1. The candidate corpus is materialized
separately for train, validation, and test image groups; no candidate from a
different split is searched. A shallow pairwise MLP reranker receives
candidate-specific product, absolute-difference, and cosine features. It is
fit on 800 Tier 2 train image groups only, with all captions for a query image
treated as known positives in the image-to-text direction and same-image
positive candidates excluded when selecting negatives.

Candidate depths 10, 25, and 50 are evaluated on Tier 2 validation. Mean MRR
selects the depth, with reranking latency and smaller depth used only as
deterministic tie-breakers. The selected depth is then frozen before Tier 2
and Tier 3 test evaluation. Stage 1 and Stage 2 use identical queries,
candidate corpora, relevance sets, and `retrieval_eval_v1` metadata in paired
comparisons. The study reports R@1, R@5, MRR, candidate recall, candidate hit
rate, an analysis-only oracle upper bound, and paired bootstrap intervals.

Query encoding, Stage-1 search, and Stage-2 reranking are timed separately.
Encoding is re-run from the Phase 7 checkpoint to verify agreement with the
Phase 10 cache, while model-load time is excluded. Qualitative reporting
includes promoted, demoted, unchanged, regression, and candidate-miss
categories. A negative reranker result is a valid outcome; the implementation
must not silently replace it with a different model or depth after seeing the
test scores.

The actual Phase 11 run selected depth 10 but degraded both retrieval
directions at both tiers. This result is recorded in `artifacts/phase11/` and
does not authorize Phase 12 query fusion.

## Phase 12B composed-retrieval protocol

Phase 12B is a separate correction experiment, not a replacement of the
controlled Phase 12 same-image sanity check. It uses the official CIRCO
validation annotations and the COCO 2017 unlabeled image gallery. Each query
is `(reference image, relative modification text)` and the candidate target
is one of the released `gt_img_ids`; the reference image is never accepted as
a target unless a benchmark explicitly labels it, which CIRCO does not.

The adapter validates query IDs, reference IDs, modification text, target IDs,
semantic aspects, gallery membership, and multiple ground truths. The official
CIRCO metric definitions are preserved: mAP@K uses the complete ground-truth
set and Recall@K uses the annotation's target image. Image-only, text-only,
early fusion, and late fusion share one candidate gallery and one query set.
Alpha values 0.25, 0.50, and 0.75 are selected on a deterministic subset of
labeled validation queries; the disjoint validation holdout is used for the
reported local comparison. The official CIRCO test labels are unavailable, so
no official test result is claimed.

Exact FAISS Flat retrieval is used to avoid ANN approximation effects. Paired
query bootstrap comparisons are defined for the best fusion against each
unimodal control. Qualitative examples are selected by deterministic category
rules and accompanied by category counts so failures are not hidden. Visual
causes such as background dominance or fine-grained mismatch are not asserted
unless supported by benchmark labels or explicit manual inspection.

Phase 12B is formally closed PARTIAL/NON-BLOCKING. The current preflight
records that the official unlabeled image archive is 18.74 GiB and that the
active volume cannot safely hold the archive, extracted gallery, and index.
The implementation records this limitation and does not silently substitute
the already downloaded labeled COCO validation images. Phase 13 therefore
uses the fixed local COCO manifest and excludes Phase 12B.

## Phase 13 statistical-validation protocol

Phase 13 predeclares seeds 42, 123, and 2026. The COCO manifest and the
seed-42-selected Tier-2 subset are fixed across all systems and seeds. Full
fine-tuning, LoRA rank 8 (`alpha=16`, dropout 0.05, `q_proj`/`v_proj`), and
the existing static top-5 hard-negative strategy are rerun without per-seed
hyperparameter tuning. Compatible seed-42 artifacts are reused; seeds 123
and 2026 are actually trained. Zero-shot is evaluated once as a non-trained
reference.

The two uncertainty layers remain separate: paired query bootstrap measures
uncertainty over the fixed test queries, while mean/sample-standard-deviation
tables measure variation across the three training seeds. Primary comparisons
are full FT vs zero-shot, LoRA vs zero-shot, LoRA vs full FT, hard-negative FT
vs zero-shot, and hard-negative FT vs full FT, prioritizing R@1 and R@5 in
each direction. Paired permutation tests and Holm–Bonferroni correction are
recorded, but statistical significance is not treated as a substitute for
effect size or practical importance.

## Phase 12 multimodal fusion protocol

Phase 12 defines a primary `image + text -> image` query. For each image
group, one caption is selected deterministically by the seed and the pair is
used as a joint query. The target is the same image group in the same
split-specific candidate corpus. This controlled identity task is used only
because the COCO caption metadata does not provide credible arbitrary
compositional labels.

The comparison ladder contains image-only, text-only, early normalized
weighted-embedding fusion, and late score fusion. Alpha values 0.25, 0.50,
and 0.75 are selected on Tier 2 validation by mean MRR; the test split is not
used for alpha or method selection. Exact persisted FAISS Flat retrieval is
used for every candidate corpus to avoid adding ANN approximation effects.
Early and late fusion share query IDs, candidate IDs, relevance sets, and the
canonical `retrieval_eval_v1` metadata. Best-fusion comparisons against both
controls use paired bootstrap resampling.

Modality dominance is measured with top-1 changes and top-K Jaccard overlap
against each control as alpha changes. Query sensitivity holds one modality
fixed while deterministically shifting the other; these shifted queries have
no correctness labels. Conflict-query and compositional-query examples are
stored with explicit qualitative labels, IDs, source text, paths, and top
results. They are not added to the benchmark metrics.

Latency separates image encoding, text encoding, fusion arithmetic, exact
retrieval, and total query time. Model loading is excluded, and re-encoded
vectors are checked against the validated Phase 10 cache. No learned fusion
module is trained because the compact early/late controlled comparison is
sufficient for the declared research question and avoids adding an
underpowered trainable component.

## Phase 8 PEFT protocol

Phase 8 uses the same canonical split manifest and Tier 2 image-group scope as
Phase 7: 800 train, 100 validation, and 100 test image groups, seed 42, and
the same candidate corpora and multi-caption relevance definitions. The base
checkpoint is `openai/clip-vit-base-patch32`. LoRA uses PEFT 0.20.0, rank 8,
alpha 16, dropout 0.05, `q_proj`/`v_proj` targets, no bias adaptation, and a
trainable logit-scale scalar. Its learning rate is `1e-4`, intentionally
declared separately from Phase 7's full-FT learning rate of `1e-6`.

Validation mean Recall@5 selects the adapter. The final test is evaluated once
after selection, using the same `retrieval_eval_v1` evaluator as zero-shot and
full FT. Paired bootstrap comparisons use identical query IDs and relevance
sets. The experiment includes a real-model smoke test, a real Tier 2 run,
adapter save/load reconstruction, merged/unmerged numerical comparison, and a
three-way zero-shot/full-FT/LoRA comparison.

Only one rank was run because the host has 8 GB unified memory and the study's
central comparison is parameter efficiency rather than a broad rank sweep.
This is a declared limitation, not evidence that rank 8 is globally optimal.

## Phase 9 hard-negative protocol

Phase 9 starts from pretrained zero-shot CLIP so that the standard Phase 7
full-FT run remains a direct quality baseline. A frozen pretrained CLIP model
mines once from the 800 train image groups. For each selected positive pair,
the candidate pool is the top five valid non-positives by cosine similarity;
one candidate is selected deterministically from that pool using the project
seed and stable IDs. The pool excludes the positive image, all same-image
captions, exact normalized caption aliases, and known exact image duplicates.

The executed objective keeps the ordinary diagonal in-batch positives and
adds explicit mined image and caption negatives for 50% of training rows. The
strategy is static because re-mining after every update would add complexity
and compute cost without a clean Tier 2 comparison. Standard in-batch
negatives are represented by the Phase 7 baseline; explicit random negatives
were not run as a redundant extra variant, and separately human-labeled
semantic negatives are unavailable.

Mining uses train data only. Validation mean Recall@5 selects the checkpoint;
test is materialized once afterward. The real run records mining time,
training-loop wall time, candidate ranks/scores, false-negative protections,
paired bootstrap comparisons against Phase 7 full FT and zero-shot CLIP, and
early-rank analysis under `retrieval_eval_v1`.

## Phase 14 controlled-ablation protocol

An ablation removes or changes one component while keeping the parent
system's dataset, split, evaluator, candidate/relevance contract, model
backbone, and selection policy fixed. This separates component contribution
from a general system comparison. Phase 14 therefore reuses compatible Phase
13 multi-seed rows and the Phase 11 reranker comparison instead of retraining
them.

The only new training changes the explicit hard-negative ratio from the
existing 50% setting to 25%. It uses seed 42, the same Tier-2 image groups,
the same pretrained CLIP starting point, the same static top-5 mining method,
and the same frozen mined-negative manifest. The standard full-FT result is
the 0% reference. The new ratio uses paired query bootstrap intervals with
200 resamples, but because it has one seed it is exploratory and receives no
corrected significance claim. Phase 12B remains excluded and no CIRCO work is
reopened.

For every comparable system, the canonical metrics are Recall@1, Recall@5,
Recall@10, and MRR in both text-to-image and image-to-text directions. The
component effect is reported as the full-system metric minus the ablated
metric. Seed variability and query bootstrap uncertainty remain separate.
Ten qualitative queries are selected by fixed evenly spaced positions in the
sorted test-query IDs; selection is not based on whether a component wins.

## Phase 15 robustness protocol

Phase 15 is an evaluation-only extension of the fixed Phase 7 protocol. It
uses the same manifest, seed-42-selected 100 test image groups, 501 caption
queries, candidate corpora, relevance sets, and `retrieval_eval_v1` contract.
The evaluated systems are zero-shot CLIP and the selected Phase 7 full-FT
checkpoint. The LoRA and hard-negative systems are omitted to keep the
robustness comparison focused and computationally bounded; this is not a
claim that they are robust or non-robust.

Text perturbations are applied only to text queries in text-to-image
retrieval. Image perturbations are applied only to image queries in
image-to-text retrieval. Candidates are always clean. Casing, punctuation,
typo noise, word deletion, shortening, resize, blur, JPEG, brightness, crop,
noise, and occlusion are each evaluated at predeclared low and high
severity. Perturbations are generated in memory from stable query/family/
severity seeds, so no derived image dataset is created.

For each condition and direction, the clean and corrupted query IDs are
aligned. The primary quantities are the absolute delta
`m_corrupted - m_clean`, relative degradation
`(m_clean - m_corrupted) / m_clean` when the clean value is nonzero, and
retention `m_corrupted / m_clean`. Paired query bootstrap uses 200 resamples
for R@1 and R@5. Rank stability reports top-1 preservation, top-5 overlap,
and first-relevant-rank displacement; displacement is explicitly censored
when the first relevant item is outside either returned top-10 list.

Distribution shift is defined before metrics using only test image metadata:
five groups with aspect ratio at most 0.75 or at least 1.3333 are compared
with five groups between 0.9 and 1.1. The groups are selected by ascending
image ID and form separate candidate/query subsets. Since the control and
shift groups are disjoint, the comparison is descriptive and unpaired.
