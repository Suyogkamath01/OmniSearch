# Technical defense notes

This document is the defense outline for the major techniques planned in OmniSearch. The active benchmark is official COCO 2017 `val2017`; real COCO Phase 1–6 artifacts are complete. Historical Flickr30k image access remains a documented optional future path.

## Dataset migration and leakage defense

The project migrated from Flickr30k to COCO because the authorized Flickr30k
image archive was unavailable through its official distribution path. COCO was
checked through its official download and terms pages before download. The old
Flickr metadata and artifacts remain historical and are never relabeled as
COCO.

### Why image-grouped splitting is required

An image is the semantic unit shared by its captions. COCO has multiple captions
for each image, and the retrieval ground truth treats all captions for that
image as positives. Therefore the image ID is the split group. Assigning the
group once ensures that visual content and all of its textual descriptions are
seen in only one partition.

### What leakage would look like

Leakage occurs if an image ID appears in more than one split, if one of its
caption IDs appears in more than one split, or if a caption is duplicated across
splits in a way that exposes the same labeled example. The manifest checker
records image and caption split sets and fails on any cross-split membership.
The Phase 1 COCO check found no image or caption leakage.

### Why caption-level random splitting is invalid

Randomly assigning individual captions can put four descriptions of one image
in training and the fifth in test. A model can then recognize the image or its
near-duplicate wording from training, making the test score optimistic. It also
violates the multi-positive retrieval protocol: the test caption's associated
image group is no longer held out. Caption-level splitting is therefore not an
acceptable alternative, even when caption strings are unique.

### Duplicate-detection limitations

Exact image duplicates are detected by SHA-256 over local file bytes. This
finds identical files but misses resized, recompressed, cropped, color-adjusted,
or otherwise visually equivalent images. Normalized duplicate captions are
reported as warnings and retained because a repeated source caption is not
automatically an error. Perceptual near-duplicate detection was not run in
Phase 1, so exact-zero duplicate results must not be interpreted as semantic
uniqueness.

## Shared image-text representation

**What:** map images and text into a common vector space so related items have high similarity.

**Why:** a shared space enables one retrieval interface for text-to-image and image-to-text search and makes representation quality separable from the index implementation.

**How:** an image encoder `f(I)` and a text encoder `g(T)` produce vectors. After normalization, cosine similarity is `f(I) · g(T)`. Retrieval ranks candidates by this score.

**Concepts:** metric spaces, dot products, normalization, nearest-neighbor search, and representation invariance.

**Alternatives:** independent classifiers, generative captioning, late-fusion feature concatenation, or cross-encoders that score each pair directly. The dual encoder is chosen for efficient retrieval; a cross-encoder remains a later reranking option.

**Actual result:** not yet evaluated.

**Limitation:** a single global vector may miss fine-grained relations and can encode dataset biases.

## Contrastive learning

**What:** learn to make matched image-text pairs more similar than mismatched pairs.

**Why:** retrieval needs relative ordering, so a ranking-oriented objective is more aligned with the task than a closed-set class label.

**How:** for a batch of matched pairs, compute a similarity matrix. A symmetric InfoNCE-style loss applies cross-entropy to the correct text for each image and the correct image for each text, with temperature `tau` controlling concentration.

**Concepts:** positive/negative pairs, softmax classification over in-batch negatives, temperature scaling, and mutual alignment.

**Alternatives:** triplet loss, margin ranking loss, binary matching loss, or generative likelihood. InfoNCE is compact and naturally provides bidirectional training, but it relies on the assumption that other batch items are valid negatives.

**Actual result:** not yet evaluated.

**Limitation:** false negatives and batch composition can distort the objective.

## Hard-negative mining

**What:** deliberately select plausible but incorrect candidates, such as captions for visually similar images.

**Why:** uniform negatives can make the task too easy and fail to train the ranking boundary that users notice near the top of a result list.

**How:** mine candidates using a frozen or earlier model, exclude known positives, and train on a versioned negative set. Mining must use training data only.

**Alternatives:** uniform in-batch negatives, random-caption negatives, or human-curated negatives. Mining is chosen only if it improves a declared top-k metric without leakage.

**Actual result:** not yet evaluated.

**Limitation:** stale or biased mining can reinforce the current model's blind spots.

## Approximate nearest-neighbor search

**What:** retrieve near neighbors without comparing a query against every vector.

**Why:** exact search is a correctness baseline but scales linearly with the corpus size.

**How:** Phase 10 uses the normalized 512-dimensional embeddings from the
validated Phase 7 full-FT CLIP checkpoint. NumPy exact search computes every
query-candidate inner product. FAISS `IndexFlatIP` is an exact implementation
reference; FAISS IVF-Flat restricts scoring to `nprobe` inverted-list cells;
hnswlib searches a graph whose maximum connectivity is `M` and whose
construction/search breadth is controlled by `efConstruction` and `efSearch`.
Because vectors are L2-normalized, inner product equals cosine similarity.

**ANN fidelity:** overlap with the exact embedding-space top-K neighbors. This
is not semantic retrieval Recall@K: semantic Recall@K checks declared
image-caption relevance, while ANN fidelity checks reproduction of exact
neighbors.

**Actual result:** real FAISS and hnswlib runs completed at 100, 1,000, and
5,000 image-group tiers, in both text-to-image and image-to-text directions.
At Tier 3, exact NumPy search took about 0.965 ms/query for text-to-image and
12.735 ms/query for image-to-text on the measured deterministic latency
sample. IVF with `nprobe=1` took about 0.033 ms and 0.046 ms respectively, but
top-10 neighbor fidelity was only 0.505 and 0.518. IVF with `nprobe=8`
improved fidelity to 0.906 and 0.931 at about 0.050 ms and 0.089 ms. HNSW
with `M=16, efConstruction=100, efSearch=32` reached 0.962 and 0.906
fidelity at about 0.242 ms and 0.262 ms. FAISS Flat preserved the exact
semantic metrics and 1.0 held-out top-10 neighbor fidelity in the Tier 3
artifact.

**Selection:** the declared validation threshold was 0.99 mean top-10
neighbor recall. At the larger tiers this selected FAISS Flat, not an
approximate configuration. This is evidence that exact search is sufficient
for strict fidelity at the current 5,000-image scale, while approximate
indexes demonstrate a future speed/quality trade-off.

**Alternatives:** brute-force matrix multiplication, inverted files, product
quantization, and tree-based indexes. The selected index must be justified by
measured latency, memory/storage, build cost, and neighbor fidelity.

**Limitation:** approximate search can silently drop relevant neighbors and
adds another source of configuration variance. The present benchmark does not
claim million-scale behavior, product quantization, distributed indexing, or
human semantic relevance labels.

## Reranking

**What:** retrieve a small candidate set with a fast dual encoder, then score candidates with a more expressive pair model.

**Why:** a cross-modal scorer can model token-region or fine-grained interactions that a single vector cannot, while avoiding full-corpus quadratic cost.

**How:** first-stage top-M candidates are rescored; only the final top-K is returned. The candidate count, model, and latency budget must be reported.

**Alternatives:** larger dual encoders, learned score calibration, reciprocal rank fusion, or no reranking. Reranking is justified only by a paired quality/latency comparison.

**Actual result:** not yet evaluated.

**Limitation:** it can overfit the benchmark and may make interactive use slow.

## Responsible retrieval analysis

**What:** inspect whether retrieval quality or exposure differs across documented content strata.

**Why:** aggregate retrieval scores can conceal systematic failures and stereotyped associations.

**How:** define strata before looking at test results where feasible, report support counts, compare distributions with uncertainty, and avoid inferring protected attributes from images. Any metadata used must have a defensible provenance.

**Alternatives:** omit subgroup analysis, use only aggregate metrics, or conduct a qualitative audit. This project keeps both quantitative and qualitative analysis while documenting what cannot be inferred.

**Actual result:** not yet evaluated.

**Limitation:** COCO metadata and captions do not provide a complete or authoritative representation of people or identity; absence of an observed disparity is not proof of fairness.

## Image-grouped splitting and leakage protection

**Why it is required:** COCO associates multiple captions with the same photograph. The photograph is the independent visual unit, so the split key must be `image_id` (or an equivalent source-group key). All captions for one image must be assigned together.

**What invalid caption-level splitting would do:** If captions were randomly split as independent rows, a training caption could describe the exact image used in validation or test. A model could then retrieve or memorize image-specific language instead of demonstrating generalisation to unseen images. The resulting Recall@K would be optimistically biased.

**What leakage looks like:** the same image ID appears in more than one split; a caption ID appears in more than one split; or preprocessing/fitting uses validation/test captions. The Phase 1 leakage checker fails on cross-split image or caption IDs, and the split command refuses manifests with duplicate image IDs.

**How this implementation prevents it:** split assignment first creates one stable mapping from image-group ID to `train`, `validation`, or `test`, then copies that label to the entire record. A group-aware assertion is run after assignment and before the split manifest is written.

**Limitations:** the historical Flickr30k metadata contains 769 unresolved groups; the active COCO manifest has no unresolved image IDs. COCO's source `val2017` partition is not itself an official retrieval train/test protocol, so the project declares its internal split explicitly.

## Duplicate and corruption detection

Exact duplicate detection hashes local image bytes. This catches identical files but not perceptual duplicates or files that depict the same content after recompression. When Pillow is installed, `Image.verify()` is used; otherwise the fallback checks common file signatures and terminators and is deliberately reported as weaker. Missing, unreadable, and corrupted files are separate findings so an inaccessible path is not confused with a decoder failure.

## Why Phase 2 EDA is required

EDA is a quality-control stage, not a substitute for retrieval experiments. For this dataset it verifies the unit of analysis, caption multiplicity, text anomalies, split balance, file integrity, and whether the local bytes match the manifest assumptions before any model is allowed to consume them. The Phase 2 report therefore separates published dataset context from measurements made on the local manifest.

## Multimodal EDA versus tabular EDA

Tabular checks can summarize row counts and missing fields, but an image-text dataset has two coupled modalities and a group structure. Text EDA covers token and character lengths, unusual characters, duplicate captions, and vocabulary. Image EDA must additionally cover file existence, readability, decoding, dimensions, aspect ratios, file sizes, exact byte duplicates, and—when deliberately added—perceptual near duplicates. Structural checks then connect image-group records to their caption sets without claiming semantic alignment from metadata alone. The current COCO run executed both metadata/text and local image checks.

## Repeated captions are not automatically leakage

Crowdsourced captions can be identical or near-identical across different photographs, especially for generic scenes. The report counts normalized duplicate-caption groups and labels cross-split language overlap as potential/benign unless the image ID or caption ID itself crosses splits. This avoids deleting legitimate data or falsely declaring leakage from common language. A repeated caption still deserves review because it can make a retrieval benchmark easier, but it is not equivalent to exposing the same image in multiple splits.

## Split analysis and leakage re-audit

The split report is computed after grouping by image ID and reports image-group and caption counts independently for train, validation, and test. The leakage re-audit independently intersects image IDs and caption IDs across every split pair. Any non-empty intersection is a critical finding. Exact image-hash overlap is a second, byte-level check when image files exist. A lack of observed overlap in metadata does not prove that two recompressed files are visually different; that is why perceptual duplicate analysis remains a separate limitation.

## Duplicate-detection limitations

Exact SHA-256 hashing finds byte-identical files only. It misses resized, recompressed, cropped, edited, or format-converted copies. Perceptual hashing could identify some visually similar files but introduces thresholds, collision behavior, and its own reproducibility requirements; it was intentionally not implemented in this phase. Signature-based validation without Pillow is weaker than a full decoder. Consequently, the current report claims exact byte-level COCO integrity only; it does not claim perceptual or semantic image uniqueness.

## What the current EDA does not establish

Caption length, word frequency, and duplicate counts do not establish image-caption semantic correctness, demographic properties, or retrieval quality. Those questions require image bytes, declared evaluation protocols, and later experiments. No model, embedding, benchmark, or qualitative image conclusion is inferred from the metadata-only Phase 2 run.

## Phase 3 classical baselines

### TF-IDF

For a document (d) and term (t), TF-IDF weights a term by its within-document frequency and its inverse document frequency. The implementation uses sublinear term frequency (1+log(tf)), smoothed IDF

`log((1 + N) / (1 + df)) + 1`,

and L2 normalization. Rare terms receive more weight because their document frequency is lower. Cosine similarity is then the dot product of the normalized sparse vectors. This is a transparent lexical baseline, but it cannot recognize synonyms, paraphrases, or visual similarity without shared words.

### BM25

BM25 scores a query term using inverse document frequency, term-frequency saturation, and document-length normalization. The implemented Okapi form uses `k1=1.5` and `b=0.75`. `k1` controls how quickly repeated occurrences stop adding evidence; `b` controls the strength of normalization for long documents. Unlike the TF-IDF baseline, BM25 does not L2-normalize a document vector and explicitly adjusts term contributions for document length. It remains lexical and therefore has the same semantic limitations for synonyms and paraphrases.

### Relevance and retrieval metrics

The Phase 3 text experiment uses a deliberately limited relevance definition: for a held-out test caption query, the other four captions belonging to the same image group are relevant. The query caption itself is excluded. This supports Precision@K, Recall@K, MRR, MAP, and binary NDCG@K for caption-to-caption retrieval. It is not a human relevance judgment and does not establish cross-modal retrieval quality. Image-to-image metrics are not reported because Flickr30k does not provide labels saying that two different images are relevant to one another.

### Handcrafted image descriptor

The classical image baseline computes a global plus 2x2 spatial RGB colour histogram with eight bins per channel and uses histogram intersection. The descriptor captures coarse appearance and spatial colour layout, but not objects, actions, composition semantics, or text. It can therefore retrieve images with similar colours or backgrounds while confusing their content. The implementation supports dependency-free PPM fixtures and an optional Pillow decoder for other formats; the real Flickr30k run is blocked because its image root is unavailable.

### Why weak baselines come first

TF-IDF, BM25, and handcrafted image descriptors establish a measured quality-versus-complexity floor. Without them, a later transformer result cannot show whether its extra computation improves retrieval over a meaningful simple method. Their characteristic lexical and appearance failures also motivate learned representations without pretending that a sophisticated model is automatically better.

## Phase 4 frozen zero-shot CLIP baseline

CLIP is a pretrained multimodal model with an image encoder and a text encoder trained to place paired images and texts near one another in a shared representation space. OmniSearch uses one frozen `openai/clip-vit-base-patch32` checkpoint through Transformers. No weights, projection head, temperature, or adapter are updated in Phase 4.

For an image (x), the image encoder produces (f(x)); for text (t), the text encoder produces (g(t)). The implementation L2-normalizes both outputs and uses their dot product, which equals cosine similarity after normalization. This permits exact text-to-image and image-to-text ranking without FAISS/HNSW. The checkpoint's ViT-B/32 image preprocessing and CLIP BPE tokenizer are supplied by the model processor.

This is stronger than TF-IDF or colour histograms because the shared space was learned from broad image-text data and can transfer visual-language associations without COCO training. It is still not a guarantee of semantic correctness: CLIP can fail on counting, negation, fine-grained attributes, spatial relations, rare concepts, bias, and visually similar but incorrect content. The current real COCO evaluation covers 500 test images and 2,501 captions; the fixture run remains separately labelled.

Zero-shot evaluation must remain separate from fine-tuning. A later fine-tuning phase may optimize a task-specific objective, but doing so here would destroy the baseline's purpose as an off-the-shelf reference.

## Phase 5 unified retrieval evaluation

Phase 5 fixes the measurement contract before additional representation
experiments are introduced. The versioned protocol is `retrieval_eval_v1` and
the canonical implementation is `src/omnisearch/evaluation.py`. It accepts
rankings produced by a system; it does not train a model or introduce an
index.

The four task schemas are text-to-image, image-to-text, text-to-text, and
image-to-image. Each producer must declare query IDs, candidate IDs and
corpus, scores, relevance IDs, candidate count, and a relevance definition.
For Flickr30k, text-to-text uses the other captions for the same image group;
text-to-image uses the owning image; image-to-text treats the image's captions
as multiple positives. Flickr30k metadata does not provide defensible
different-image relevance labels, so image-to-image is not scored without an
external declared label source.

The evaluator reports macro query-level Precision@K, Recall@K, binary NDCG@K,
MRR, MAP, and first-relevant-rank statistics. Precision@K divides by K,
including omitted positions in a truncated list as non-hits. Recall@K divides
by the complete declared relevance set. MAP is labelled as truncated when the
producer supplies only top-K candidates. No-relevance queries are counted
separately. Score ties use candidate-ID ascending order so a metric cannot
change because of incidental dictionary or sort order.

Uncertainty is query-level bootstrap with a recorded seed and resample count.
System comparisons are paired only when query IDs, split, candidate corpus,
candidate counts, protocol, relevance definition, and relevance sets match.
The comparison then reports per-query right-minus-left deltas with paired
bootstrap intervals. This prevents a favorable difference from being caused
by different query samples or incompatible relevance maps.

Phase 5 freshly reran the COCO Phase 3 TF-IDF and BM25 test-caption baselines
under this contract. It migrated the historical Phase 4 fixture smoke through
the same schema and also migrated the real COCO Phase 4 rankings. Image-to-
image remains label-limited, not data-limited.

## Phase 6 frozen representation experiments

Phase 6 answers a narrower question than fine-tuning: how do compact, already
pretrained representations behave when extracted without changing a single
weight? The text matrix contains MiniLM-L6 mean pooling as the compact
sentence-level semantic encoder and mean-pooled DistilBERT as a generic
pretraining control. The vision matrix contains ResNet-18 for a small CNN
inductive bias and ViT-Base/16 for a patch-based global-attention comparison.
The existing frozen CLIP text and vision components are retained as the
canonical shared-space reference, not silently mixed with the unrelated
unimodal spaces.

For a transformer hidden-state sequence H and attention mask m, mean pooling
and normalization are

`z = sum_l(m_l H_l) / max(1, sum_l m_l)` and `z_hat = z / ||z||_2`.

Self-attention forms queries, keys, and values and computes
`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`. Residual connections and
normalization stabilize repeated transformer blocks. A CLS vector is a
learned summary token; mean pooling uses all unmasked tokens and is explicit
here so the generic DistilBERT control is not presented as a
sentence-similarity model. MiniLM was chosen because it is much smaller than
base encoders and is intended as a practical semantic embedding; DistilBERT
tests whether generic contextual features alone are sufficient.

CNN and ViT encode images with different inductive biases. A CNN applies local
shared filters and builds larger receptive fields through depth; ResNet skip
connections make a small residual network easier to optimize and its native
global pooled feature is used here. ViT first divides an image into fixed-size
patch tokens, adds positional information, and applies global self-attention;
the native pooled output is retained. The qualitative fixture comparison is
descriptive only; real COCO image representation evidence is reported
separately and does not create an image-to-image relevance label.

All extraction uses deterministic ID order, fixed batches, model processors,
inference mode, frozen `eval()` models, finite-value checks, L2 normalization,
and metadata-checked embedding caches. The cache records model ID/revision,
manifest hash, split, pooling, normalization, dimension, device, and ID hash.
Exact ranking is performed within one declared vector space. Direct cosine
comparison between a MiniLM text vector and a ResNet or ViT image vector is
invalid even if dimensions happen to match. Only CLIP's own projected image
and text components have a learned shared space.

The actual Phase 6 COCO text run used 256 deterministic queries and all 2,501
held-out test captions. MiniLM, DistilBERT, and CLIP text were compared with
the canonical `retrieval_eval_v1` evaluator and against TF-IDF/BM25. The
vision run also encoded all 500 real COCO test images with ResNet-18,
ViT-Base/16, and CLIP vision. Therefore real representation evidence exists,
but no formal image-to-image metric or image relevance claim is made.

## Phase 6 limitations and interpretation

The same-image caption relevance map is a useful reproducible proxy, but it is
not a human semantic judgment: five captions for one photograph are not
necessarily interchangeable, and captions for different photographs can be
semantically similar. A transformer winning over BM25 on a low lexical-overlap
query is diagnostic evidence of possible paraphrase sensitivity, not proof of
visual correctness. Conversely, BM25 can win when exact words are strong
signals. The recorded qualitative categories and examples are deterministic
sample diagnostics, not an EDA expansion or a claim of general superiority.

Model parameter bytes and observed load/encoding times are hardware- and
cache-dependent measurements from the M3/MPS run. They are reported for
reproducibility, not as universal throughput claims. No fine-tuning, LoRA,
adapter, projection training, alignment training, ANN index, reranking,
service, or dashboard is part of Phase 6.

## Phase 7 full-parameter CLIP fine-tuning

Phase 7 uses the pretrained `openai/clip-vit-base-patch32` checkpoint and
updates all 151,277,313 parameters, including `model.logit_scale`. It is a
conventional full-parameter run; LoRA, PEFT, adapters, hard-negative mining,
ANN indexes, and reranking are deliberately outside this phase.

For each training epoch, one caption is selected deterministically from each
image group. This makes every image-text pair in a batch have a unique image
ID, so another caption belonging to the same image is never treated as an
in-batch negative. Validation uses all captions for the selected validation
image groups and is used for checkpoint selection. The test image groups are
materialized only after the selected checkpoint is fixed.

With normalized image and text projections `u_i` and `v_j`, the batch score
matrix is `S_ij = exp(s) u_i^T v_j`, where `s` is the trainable CLIP logit
scale and the exponential is capped at 100 for numerical safety. The loss is

`L = 1/2 [ CE(S, diag_targets) + CE(S^T, diag_targets) ]`.

This symmetric objective matches the two retrieval directions evaluated by the
project. A caption-level random split would invalidate the comparison because
captions for one photograph would put the same visual group in both fitting
and evaluation data. The Phase 7 loader therefore consumes the already
image-grouped split manifest and applies deterministic limits separately to
train, validation, and test.

The executed Tier 2 run used 800 train, 100 validation, and 100 test image
groups from the 5,000-image COCO split manifest, with batch size 2,
gradient accumulation 4 (effective batch 8), one epoch, AdamW learning rate
`1e-6`, weight decay `0.01`, fp32, and MPS. The selected checkpoint was chosen
by validation mean Recall@5. On the held-out Tier 2 test subset, fine-tuning
changed text-to-image Recall@1/5 from 0.8144/0.9780 to 0.8263/0.9880 and
image-to-text Recall@1/5 from 0.1857/0.8003 to 0.1837/0.8143. The paired
Recall@5 deltas were +0.0100 (95% bootstrap CI [0.0020, 0.0200]) and +0.0140
(95% CI [-0.0080, 0.0340]), respectively. These are one-seed, one-epoch
Tier 2 results, not evidence that fine-tuning universally improves CLIP.

The smoke and real runs explicitly checked finite loss, finite gradients,
parameter updates, checkpoint creation, validation-only selection, and
bidirectional canonical evaluation. The qualitative artifact reports
improved/unchanged/degraded counts rather than only favorable examples. The
main limitations are the small compute-limited fine-tuning scope, one seed,
one epoch, a source `val2017` partition rather than an official retrieval
train/test partition, and the known limits of exact duplicate detection.

## Phase 8 parameter-efficient fine-tuning and LoRA

LoRA replaces a full update to a weight matrix with a frozen base matrix plus
a trainable low-rank update. OmniSearch uses the established PEFT library
(`peft` 0.20.0), rank 8, alpha 16, dropout 0.05, no bias adaptation, and
targets the `q_proj` and `v_proj` attention projections in both the CLIP text
and vision encoders. These projections are defensible targets because they
control query/value transformations in attention while avoiding adapters on
every possible parameter. The CLIP logit scale is an explicitly declared
additional trainable scalar; all other base parameters remain frozen.

The executed Tier 2 experiment uses the same COCO manifest, image-grouped
split, 800/100/100 scope, seed, candidate sets, relevance definitions, and
`retrieval_eval_v1` evaluator as Phase 7. Validation mean Recall@5 selects the
adapter; test data is materialized afterward. The adapter is saved separately
from the base model, with a small extra state file for the trainable logit
scale. The measured adapter artifact is 1,988,122 bytes; reconstructing it
still requires the base CLIP checkpoint.

The LoRA run trains 491,521 parameters, or 0.3249% of the 151,277,313 full-FT
parameters: a 99.6751% reduction. On this one-seed, one-epoch Tier 2 run,
LoRA text-to-image R@5 is 0.9780, equal to zero-shot and below full FT's
0.9880. Image-to-text R@5 is 0.7983, slightly below zero-shot 0.8003 and
full FT 0.8143. The paired
Recall@5 deltas are -0.0100 with CI
[-0.0180,-0.0040] for text-to-image and -0.0160 with CI
[-0.0340,+0.0041] for image-to-text. These results show the storage and
parameter advantage clearly, but do not show performance parity with full FT.

LoRA training took approximately 86 seconds on MPS versus approximately 193
seconds for the recorded full-FT run. Unified-memory MPS peak memory was not
treated as reliably measurable by this experiment. Merged and unmerged LoRA
validation inference had maximum returned-score difference below 1e-5 and
identical ranking IDs on the final run; the equivalence artifact records the
validation-only merge check. The main limitations are the single rank-8
configuration, one seed, one epoch, and the compute-limited Tier 2 scope. No
hard negatives, ANN search, reranking, fusion, uncertainty, API, or UI was
introduced.

## Phase 9 hard-negative mining

Contrastive learning needs negatives because the positive pair alone does not
teach the model which other candidates should be separated. Ordinary CLIP
training gets many negatives from the other rows in a batch. A hard negative
is a non-positive pair that the current mining model scores highly, so its
logit creates stronger discrimination pressure than an easy random mismatch.

OmniSearch uses static mining: frozen pretrained CLIP encodes only the train
image groups and captions, ranks valid non-positives, and samples
deterministically from the top five. This is reproducible and cheaper to
defend than dynamic re-mining after every optimizer update. The candidate
pool excludes same-image captions, exact normalized caption aliases, positive
IDs, and known exact image duplicates. Distinct images can still depict the
same scene, so exclusion rules reduce but do not eliminate false negatives.

The hard-negative objective retains the diagonal positive targets and adds
selected mined image/caption logits to both directional denominators. Harder
negative logits increase the gradient pressure to separate confusable pairs,
but harder is not always better: a semantically equivalent image with a
different ID is a mislabeled negative, and over-emphasizing it can hurt broad
generalization. Mining bias can also make the model specialize to the frozen
miner's notion of similarity.

The executed Tier 2 run used exactly 400 of 800 rows as explicit mined rows,
in addition to ordinary in-batch negatives. It changed image-to-text
Recall@5 from full FT's 0.8143 to 0.8203, while text-to-image Recall@5 was
0.9800 versus 0.9880 for full FT. Paired bootstrap intervals crossed zero in
both directions, so this is not evidence of superiority. Mining took 59.99
seconds and the recorded training-loop wall time was 417.77 seconds including
validation/checkpoint-selection overhead. The cost must be justified by
quality gains; here the result is directionally positive for image-to-text but
inconclusive at this compute budget.

## Phase 11 two-stage retrieval and reranking

Two-stage retrieval separates broad candidate generation from a more
expensive candidate-specific scoring function. For a query `q`, Stage 1
returns a fixed candidate prefix `C_N(q)` from the validated shared embedding
space. Stage 2 can reorder only that prefix; it cannot recover a relevant item
that Stage 1 omitted. This is why candidate recall must be reported before
interpreting a reranker result.

Phase 11 uses exact FAISS Flat Stage 1 over the Phase 7 full-fine-tuned CLIP
embeddings. Exact FAISS was chosen because Phase 10 showed it preserved the
validated neighbor results while being faster than the NumPy reference at the
current scale. Separate train, validation, and test candidate indexes make
the fitting and evaluation scopes explicit. No new Stage-1 model was trained.

The Stage-2 model is deliberately shallow: it receives the query/candidate
elementwise product, absolute difference, and cosine similarity, then emits a
scalar ranking score through a 64-unit MLP. The product and absolute
difference are candidate-specific interactions beyond simply sorting by
cosine. The output is not a probability, because no calibration or probabilistic
loss was fit. Pairwise softplus margin training used only 4,801 train pairs
from 800 Tier 2 train image groups; validation selected the candidate depth and
the test split remained untouched.

The selected depth was 10. The reranker changed every evaluated test ordering
and reduced R@1, R@5, and MRR in both directions at both Tier 2 and Tier 3.
This is a useful negative result: a pairwise objective can learn a training
ordering while misaligning with the multi-caption retrieval metric. The
candidate recall/oracle artifacts distinguish three causes: a Stage-1 miss
that no reranker can repair, a present candidate demoted by Stage 2, and a
semantic error shared by both systems under the metadata-defined relevance
map. The oracle is not a model; it simply moves available relevant candidates
to the front to show the candidate-set ceiling.

Latency is also structurally different across stages. Query encoding uses the
CLIP checkpoint and is measured separately from FAISS search and MLP scoring.
At the selected depth, reranking added about 1.02–1.30 ms per query on the
recorded MPS run, while image-query encoding remained the dominant component.
These are local measurements with model load excluded, not universal service
latency.

The Phase 10 audit passed before this experiment. Phase 11's quality gate is
therefore a methodological PASS despite the model-quality regression: the
retriever was validated, training and selection scopes were isolated, paired
statistics and oracle ceilings were generated, and the negative result is
reported without relabeling scores as probabilities. Phase 12 query fusion
was evaluated next under a controlled identity protocol.

## Phase 12 multimodal query fusion

A multimodal query is represented as `Q = (x, t)`, where `x` is an image and
`t` is a text modification or description. Phase 12 uses CLIP's validated
shared projected space for both components, so the vectors have compatible
semantics and dimensions. Combining an unrelated text encoder with a CNN
feature would require an explicit learned alignment and is outside this
phase.

The primary quantitative task is deliberately controlled: `x` and one of its
captions form the query, and the corresponding image group is the target in
the same split-specific image corpus. This gives a defensible identity
relevance set while avoiding fabricated labels for arbitrary edits. It also
creates a known ceiling: image-only retrieval can often return the source
image at rank one. Therefore fusion is compared against both image-only and
text-only controls, and a fusion result that merely matches image-only is not
presented as a semantic improvement.

Early fusion computes a weighted query vector and normalizes after the sum:

`q_early = normalize(alpha * q_image + (1-alpha) * q_text)`.

The post-sum normalization is required because the dot product with a
normalized candidate should represent cosine similarity. Late fusion keeps
the two scores separate:

`s_late(c) = alpha * (q_image^T c) + (1-alpha) * (q_text^T c)`.

Here alpha is the image weight. For normalized image and text query vectors,
early and late fusion have identical candidate ordering for a fixed alpha in
this dual-encoder setting: the early query's normalization divides every
candidate score by the same positive query norm. They may differ in raw score
scale, but not in rank. This explains the observed identical metrics and is a
mathematical property, not evidence that all fusion architectures are
equivalent.

The validation grid used alpha 0.25, 0.50, and 0.75. The selected late
alpha=0.25 configuration improved over text-only on the controlled task but
did not exceed image-only. Increasing alpha monotonically increased top-10
overlap with image-only and decreased overlap with text-only. The sensitivity
analysis also changed the text or image component independently and measured
top-1 rank changes without assigning correctness to shifted queries.

Conflict queries pair an image with a caption from another image and are
labeled `QUALITATIVE CONFLICT ANALYSIS`. Compositional strings such as “same
object but red” are labeled `QUALITATIVE COMPOSITIONAL QUERY`. Neither is a
quantitative benchmark because COCO captions do not say whether an arbitrary
attribute edit is satisfied. Their purpose is to expose modality dominance,
text modification neglect, and representation drift for visual inspection.

Fusion arithmetic was negligible relative to CLIP encoding: the selected
configuration measured about 0.010–0.022 ms for fusion and 0.027–0.104 ms for
exact retrieval, while image encoding cost about 19.5–24.6 ms/query and text
encoding about 2.3–3.0 ms/query on the recorded MPS host. Model load was
excluded. No learned fusion module, query expansion, uncertainty method,
service, or UI was added.

## Phase 12B: genuine composed-retrieval correction

The original Phase 12 identity task has a known structural ceiling: the query
image is itself the target. Phase 12B keeps that result, but separately
implements the intended composed task on CIRCO:

`(reference image, modification text) -> target image(s)`.

CIRCO is appropriate because its released validation annotations contain a
relative caption, a reference image, a target image, multiple ground-truth
images, and benchmark-provided semantic aspects. The adapter rejects a
reference image appearing in the ground-truth set, preventing reference-image
copying from being counted as success. Its metric implementation follows the
official semantics rather than the COCO same-image evaluator: mAP@K uses all
ground truths, while Recall@K uses the annotation's target image.

Phase 12B compares image-only, text-only, early vector fusion, and late score
fusion over one exact candidate gallery. Alpha selection is validation-only;
the local implementation deterministically separates CIRCO labeled validation
queries into selection and holdout subsets because official test labels are
not released. Paired bootstrap comparisons and modality-overlap analysis are
defined at the query level.

This correction is formally PARTIAL/NON-BLOCKING because of acquisition
capacity, not because of an ambiguous benchmark source. The official
repository and access path were verified, but the official `unlabeled2017.zip`
archive and extracted gallery do not fit the current volume with safe
headroom. Downloading an unofficial mirror or substituting the labeled COCO
validation images would change the benchmark and invalidate the claim. The
machine-readable preflight and closure record this limitation, and no Phase
12B quantitative result is claimed.

## Phase 13: why multiple seeds matter

A random seed initializes or controls several stochastic choices: model
adapter initialization, pair ordering, caption sampling, negative-row
selection, and framework random-number generators. A single trained result
cannot show whether an observed gain is a stable property of the protocol or
an accident of one stochastic trajectory. Phase 13 therefore predeclared
seeds 42, 123, and 2026 and kept the manifest, image-group split, Tier-2
subset, optimizer, epoch count, and checkpoint-selection rule fixed.

The zero-shot model is a non-trained deterministic reference and is reported
once. Full FT, rank-8 LoRA, and the existing static hard-negative system are
the trained systems. Seed 42 was reused only after manifest, scope, protocol,
and quality-gate compatibility checks; seeds 123 and 2026 were actually
trained. Phase 9 uses one verified seed-42 mined-negative manifest so the
comparison isolates training randomness rather than mixing in a new mining
procedure.

The mean summarizes the three observed seed scores, while the sample standard
deviation describes their spread using `n-1` in the denominator. With only
three seeds this is descriptive evidence, not a precise population estimate.
The paired query bootstrap instead resamples identical test queries and asks
how uncertain a system delta is over those queries. The two uncertainty
sources answer different questions and are not collapsed into one number.

For a paired query, the permutation null hypothesis is that the sign of its
metric delta is exchangeable around zero. The implementation uses 2,000
paired sign-flip permutations and records the p-value. Because many primary
comparisons are made, Holm–Bonferroni correction is applied to the declared
family. A corrected non-rejection is not proof of equality; it means the
available evidence does not support a reliable difference at the declared
threshold. Effect sizes and win/loss/tie counts remain necessary.

The actual result is directional: full FT's text-to-image R@5 improvement
over zero-shot was positive across all three seeds, but image-to-text R@5 did
not change. LoRA matched zero-shot at R@5 and stayed below full FT on
text-to-image R@5. Hard-negative training had a small stable text-to-image
R@5 gain over zero-shot, but its comparison against full FT was seed-sensitive.
No corrected primary permutation test was rejected. These conclusions are
limited to the one-epoch Tier-2 COCO protocol and do not include CIRCO.

## Phase 14: why controlled ablations matter

An ablation study asks what changes when one component is removed or altered
while the rest of the protocol stays fixed. A full-system comparison alone can
confound dataset, seed, checkpoint, evaluator, or preprocessing changes. Phase
14 therefore declares the changed component and unchanged components for every
comparison, reuses compatible Phase 13 multi-seed evidence, and preserves the
Phase 11 reranker result as a negative control.

The contribution convention is

`Delta_component = metric_full_system - metric_ablated_system`.

Positive deltas do not automatically imply a useful component: the direction,
metric, seed spread, paired-query interval, and computation cost all matter.
Full fine-tuning has a stable text-to-image R@5 gain over zero-shot in this
study, so it is retained for that objective. LoRA is an efficiency option with
lower adaptation capacity and lower quality on the key text-to-image R@5
comparison. Hard negatives are optional because their gains versus standard
full FT are mixed while mining/training costs rise. The Phase 11 reranker is
not recommended because it consistently degraded the held-out ranking metrics.

Negative ablations are useful: they prevent feature count from being mistaken
for scientific value and can justify removing complexity. The ratio-25 run
shows why uncertainty matters: its paired intervals overlap zero and its one
seed cannot select a hard-negative ratio. Effects can also interact, so these
results do not claim that component deltas are additive or universal.

## Phase 15: why robustness is a separate claim

High clean retrieval scores do not imply that a system preserves its ranking
under malformed text, image degradation, or a changed input distribution.
Robustness therefore has to be evaluated with the clean test protocol fixed
and the perturbation rule declared before looking at the resulting metrics.
Phase 15 does this for zero-shot and full-FT CLIP only. It does not retrain on
the corruptions, tune a model to them, or replace clean candidates with a
corrupted gallery.

Caption-level random changes and image-level degradations answer different
failure questions. Text corruption tests whether the text encoder preserves
the relevant image under reduced lexical signal; image corruption tests
whether the visual encoder preserves the relevant caption group. Query IDs,
candidate IDs, relevance sets, and evaluator semantics remain aligned so
the clean-minus-corrupted difference is a paired query comparison rather than
a change in benchmark composition.

The aspect-ratio analysis is a controlled distribution-shift stress test, not
an external-domain result. The shifted and control groups are selected from
the fixed test subset by image metadata before metric computation, have equal
size, and are disjoint. This prevents selecting a shift because it produced a
large error, but the small sample and same-dataset origin limit what can be
generalised. Its descriptive comparison is intentionally not treated as a
paired significance test.

The result supports a specific diagnosis: high shortening and high central
occlusion caused the largest observed degradations in this scope, while
moderate casing changes were nearly neutral. These findings motivate the
next error-analysis phase, but they do not establish a universal corruption
ranking or prove that fine-tuning improves robustness. The full condition
matrix, bootstrap intervals, rank stability, and qualitative examples remain
available for audit under `artifacts/phase15`.

## Phase 16: why failure analysis is rank- and evidence-aware

Aggregate Recall@K says how often a relevant item enters a prefix, but it does
not explain why a query failed, whether fine-tuning changed the failure, or
whether a robustness condition caused a rank displacement. Phase 16 therefore
starts from the fixed Phase 7 query rankings and records top-1, top-5, and
severe (`rank > 10` or absent from the retained top-10) failures separately.
This makes an improvement at rank five distinguishable from a top-1
regression, while avoiding invented ranks beyond the observed candidate list.

Text-to-image and image-to-text are analysed independently because their
queries, relevance sets, and failure mechanisms differ. A caption-level
random split would make captions for one image appear in both training and
evaluation and could let image identity solve the task; image-grouped records
avoid that leakage. The same identity discipline is retained when linking
Phase 9 hard-negative and Phase 15 robustness evidence.

The taxonomy separates mechanical observations from interpretation. Length,
lexical overlap, rarity, caption count, and score margin can be computed from
the retained records. Labels such as object, spatial, attribute, action, and
scene context are heuristic proxies, explicitly marked as such, because this
repository has no human error annotations. The resulting root-cause layers
are corrective hypotheses, not causal or prevalence claims.

Phase 11 is handled conservatively: its aggregate reranker metrics and
qualitative cases are linked, but the canonical artifact did not preserve all
paired Stage-1/reranked query lists, so an exact intersection such as
“Stage-1 success became reranker failure” is not reported. Likewise, Phase 15
worst-condition examples are useful diagnostic evidence but do not constitute
a full corrupted test population. These provenance boundaries keep the
failure priorities auditable: investigate lexical/object and spatial errors,
shortening and occlusion sensitivity, and the evaluated reranker regression
without presenting hypotheses as ground truth.

## Phase 17: confidence, calibration, and selective retrieval

A retrieval score is a similarity or ranking value produced by the model. It
has no probability semantics by itself. A confidence proxy is a deterministic
function of the returned scores, such as the top-1 score or the margin

`m = s1 - s2`.

Phase 17 also tests score concentration over the retained top-10 list with a
temperature-one softmax and normalized entropy. These are experimental
confidence signals, not probabilities over all candidates. Because score
scales can differ by system and direction, raw values are not compared
directly across those conditions.

Calibration asks whether a bounded confidence value agrees with empirical
correctness. The selected proxy is standardized and transformed by a
two-parameter logistic calibrator using validation labels only. The test split
is then evaluated once with the persisted transformation. For reliability
bin `b`, mean confidence is compared with the observed top-1 accuracy. With
`n_b` queries in a bin and `N` total queries,

`ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|`.

The associated Brier score is `mean((confidence - correctness)^2)` and is
meaningful here only because the transformed confidence is explicitly bounded
and evaluated against binary targets. Neither ECE nor Brier is applied to raw
cosine similarity. ROC-AUC and PR-AUC answer a different question: whether
confidence ranks correct queries above incorrect ones. They measure
discrimination, not calibration.

Selective prediction adds an abstention option to offline evaluation. At a
fixed threshold, accepted queries have calibrated confidence at or above the
threshold. Coverage and selective risk are

`C = N_accepted / N`  and  `R = N_incorrect_accepted / N_accepted`.

The risk-coverage curve accepts queries from highest to lowest confidence; a
useful signal generally lowers risk as coverage is reduced, but monotonicity
is not assumed. AURC is calculated as the mean risk of every descending-
confidence prefix, so it summarizes the curve without replacing it.

Thresholds are chosen only at predetermined coverage points using validation
labels. The held-out test reports the resulting coverage and risk but cannot
change the threshold. This preserves the distinction between uncertainty
analysis and a production abstention policy.

The phase stores both dangerous high-confidence errors and low-confidence
correct results. The first exposes failures that a confidence gate would miss;
the second exposes over-abstention. Image-to-text has few failures, so its
confidence findings are descriptive with wide uncertainty. Phase 15 cannot be
linked at score level because its corrupted score arrays were not retained;
the absence is recorded rather than filled with a robustness claim.

## Phase 18: explainability without causal overclaiming

Interpretability describes how a model is structured; explainability here means
providing evidence about one retrieval decision. Phase 18 uses local
perturbations because they can be reproduced from the same checkpoint and
query. They do not reveal a complete causal computation inside CLIP.

For a text token or image region `j`, the local sensitivity is

`importance_j = s(x) - s(x_without_j)`.

Positive importance means the observed relevant score decreased after the
feature was removed; negative importance means it increased. Rank sensitivity
is the corresponding change in first-relevant rank. The sign and magnitude
are conditional on the perturbation, preprocessing, candidate set, and
relevance proxy.

Token occlusion deletes one whitespace/punctuation token or a heuristic
counterfactual word group. Region occlusion replaces one cell of a fixed 3×3
grid with its local mean colour. These are score-sensitivity explanations,
not token attribution truth or image saliency truth. A high score delta can
reflect a changed sentence or an artificial visual boundary rather than a
human semantic cause.

The faithfulness check is deliberately modest: within each selected example,
the most sensitive perturbation is compared with the least sensitive one. If
the former produces a larger score/rank degradation, the local signal is
consistent with the perturbation test. Because the comparison is selected
from the same perturbation table and uses a small sample, it is not a global
faithfulness guarantee.

Counterfactual edits remove object, action, attribute, or spatial words when
small heuristic lexicons identify them. Such labels guide a controlled edit;
they do not become semantic ground truth. High-confidence errors are retained
because a locally persuasive explanation can accompany a wrong retrieval.
Correct and low-confidence cases are retained as controls against presenting
only attractive explanations.

The explanation record separates observable score/rank evidence, local
perturbation sensitivity, and heuristic interpretation. It therefore avoids
claims such as “the model thinks” or “this proves the reason.” The appropriate
claim is that a token or region was associated with a score/rank change under
the declared perturbation.

## Phase 19: responsible AI without fabricated fairness labels

Responsible-AI analysis must distinguish a measurable retrieval disparity from
an unfairness conclusion. The retained COCO evaluation has image IDs,
captions, split assignments, ranks, scores, and confidence records, but it
does not provide lawful, consented, task-appropriate protected-group labels.
Phase 19 therefore uses only reproducible dataset-derived strata: caption
length, lexical rarity computed from train captions, a fixed object/count
complexity heuristic, and image aspect ratio. These strata can reveal where
this benchmark performs differently; they cannot identify people or establish
discrimination. No demographic status was inferred from pixels or caption
words.

The predeclared minimum group size is 20 retained test queries. Smaller groups
are emitted for transparency but are marked descriptive-only and cannot drive
a strong comparison. For eligible groups, top-1/top-5 rate, MRR, failure rate,
confidence, high-confidence-error rate, and validation-derived selective
rejection are reported with compact deterministic percentile bootstrap
intervals. A gap is recorded as a measured performance disparity, not as a
fairness judgment. The uncertainty is query-sample uncertainty within this
fixed COCO/Tier-2 scope, not population uncertainty across users or domains.

Caption-level random splitting would be invalid for these analyses for the
same reason it is invalid for model evaluation: captions from one image are
not independent examples. If captions for one image entered different splits,
group performance could reflect image-specific memorization and captions from
the same source image could appear on both sides of a comparison. Phase 1's
image-grouped manifest and leakage checks keep all captions of one image in
one split; Phase 19 consumes that manifest rather than rebuilding groups.

The confidence review retains 15 high-confidence top-1 errors from Phase 17.
This is a concrete reliability risk because an abstention rule can miss a
wrong result when confidence is high. Phase 18 local perturbation evidence is
attached to seven of these records, but it does not identify a single cause.
The analysis also compares zero-shot and full-FT systems on the same held-out
queries; the difference is a conditional benchmark comparison, not a causal
or fairness improvement claim.

Privacy and safety are intentionally separated. COCO source-rights and
originating-Flickr terms are provenance constraints; they do not amount to a
private-data audit. Content safety asks whether a result or query is harmful;
Phase 19 ran no moderation classifier, human safety review, or red-team test,
so content safety is **NOT EVALUATED**. Potential privacy, profiling,
surveillance, accessibility, multilingual, and misuse risks are documented as
deployment concerns with proportionate mitigations. The appropriate current
posture is research-only use, human review for sensitive retrieval, and no
claim of safe deployment.

## Phase 20: efficiency without overstating optimization

Phase 20 answers a resource question using retained evidence rather than
claiming a new training result. The focused pre-phase audit records
`PRE-PHASE AUDIT: Phase 19 PASS`; all Phase 19 responsible-use limitations
remain in force. The analysis performs no training, download, new model-family
experiment, or new index build.

The first distinction is total model cost versus update cost. The full CLIP
checkpoint has about 151.3 million parameters and a calculated 577.1 MiB
fp32 parameter payload; its measured Phase 7 checkpoint is 605,242,499 bytes.
The LoRA adapter is 1,988,122 bytes and updates 491,521 parameters, but it
still requires the base CLIP checkpoint. Calling the adapter the model would
understate deployment storage. Peak memory is not stated because it was not
reliably measured on MPS unified memory.

The second distinction is cold start versus query work. The offline local
MPS profile measured 5.964 seconds to load base CLIP and 6.590 seconds to load
it plus the full-FT state dict. Those values are separate from query latency.
For the retained Tier 2 profile, text encoding costs 2.885 ms/query and image
encoding costs 20.459 ms/query after dividing measured batch totals by the
declared query counts. FAISS Flat search costs 0.074 ms/query for text→image
and 0.288 ms/query for image→text. Therefore the cold encode-plus-search
profiles are 2.959 ms/query and 20.748 ms/query, while warm cached retrieval
contains only the search stage. This identifies query encoding, especially
image encoding, as the primary steady-state bottleneck and model loading as a
cold-start bottleneck.

Caching is valuable because it changes the critical path: the retained profile
shows approximately 39.9× and 72.0× cold-to-warm ratios for text→image and
image→text. This is not a claim that all service latency disappears; request
preprocessing, serialization, concurrency, and I/O were not fully benchmarked.
The cache has 5,000 image vectors and 25,014 caption vectors at 512 dimensions
in float32. The dense payload calculation is 9.77 MiB plus 48.86 MiB before
metadata/index overhead, and the selected FAISS Flat files are measured
separately.

Float16 is a bounded optional optimization. Converting the same cache to
float16 in memory and accumulating search in float32 preserved R@1/R@5 in the
tested text→image and image→text samples; top-1 agreement was 1.0 and top-10
agreement was 0.986/0.970. The calculated dense payload halves, but no float16
cache was persisted, so the production recommendation remains unchanged.

Exact search versus ANN is a scale decision, not a speed contest. At this
approximately 5,000-image scale, Phase 10 selected FAISS Flat under the
retained 0.99 neighbor-fidelity criterion. IVF/HNSW configurations were faster
but lost neighbor and/or semantic fidelity. Hence the current recommendation
is exact FAISS Flat, with ANN retained as optional scale-out evidence pending a
fresh fidelity check at larger scale.

The quality-efficiency comparison also explains why more components are not
automatically better. Full FT averaged 171.3 seconds across the retained
three-seed runs; LoRA averaged 83.4 seconds, about 0.487× full FT; hard-negative
FT averaged 345.2 seconds before measured mining overhead and 365.2 seconds
when consolidated, about 2.015× full-FT training cost. The hard-negative point
is dominated by full FT in the fixed mean-R@5/cost frontier. The Phase 11
reranker adds about 1 ms/query but reduced held-out quality, so it is disabled.
The practical quality default is full FT plus caching plus FAISS Flat; LoRA is
optional for constrained adaptation, and zero-shot remains the lightweight
baseline. Phase 20 therefore implements analysis and recommendations, not an
unverified production switch.

## Phase 21: serving the validated retrieval stack

Phase 21 wraps the scientific retrieval path in a small REST API without
changing the model or metric definition. A REST endpoint is a stable HTTP
resource/action boundary: the client sends a request representation and the
server returns a machine-readable response or a typed HTTP error. FastAPI
provides routing, request parsing, OpenAPI generation, and lifecycle hooks;
Pydantic validates the JSON and multipart-derived fields and serializes stable
response schemas.

The application factory `create_app()` constructs the ASGI application without
loading the model at Python import time. Its lifespan is the startup lifecycle:
it validates the Phase 20 report, hashes, manifest, cache, index metadata,
dimensions, IDs, and model identity, then loads the full-FT checkpoint and
FAISS Flat indexes once. This is dependency/resource injection in practical
terms: route functions receive the already-created `RetrievalService` through
a FastAPI dependency rather than constructing a model per request. The
retrieval service is independent of routing, so a later UI can reuse
`search_text_to_image()` and `search_image_to_text()` without duplicating
model or index logic.

Health and readiness answer different questions. `/health` is lightweight
process status and reports whether model/index objects are loaded; `/ready` is
the gate for answering retrieval requests and returns an unavailable response
until validated resources are ready. `/info` exposes safe metadata such as
model family, dimension, backend, supported modes, version, and the explicit
content-safety status, but not local filesystem paths.

The text endpoint accepts a bounded query and top-k, encodes the query with
the loaded full-FT CLIP model, searches the cached image index, and returns
rank, score, ID, and available image metadata. The image endpoint accepts a
bounded multipart payload, decodes the actual bytes with Pillow rather than
trusting the filename or MIME type, encodes the image, and searches the
caption index. Malformed images and invalid top-k values are client errors;
missing/incompatible resources are readiness errors; model/index failures are
safe internal retrieval errors without tracebacks in the response.

Cold and warm latency must not be conflated. Cold service behavior includes
startup model/index loading. Warm request behavior begins after resources are
loaded and reports preprocessing, query encoding, search, and total server
time. Phase 21 measured five warm local TestClient requests per direction;
wall-clock and server-only summaries are separated in
`artifacts/phase21/api_latency.json`. The dominant warm component remains CLIP
query encoding, while exact FAISS Flat search is comparatively small. A model
lock prevents concurrent inference calls from mutating shared runtime state;
the development deployment intentionally avoids multiprocessing complexity.

### Phase 21 research questions

- **RQ21.1:** Yes, the validated stack served stable real text→image and
  image→text requests after startup provenance checks.
- **RQ21.2:** API/TestClient overhead is reported separately from server total;
  the real five-request benchmark recorded both distributions.
- **RQ21.3:** Warm query encoding dominates; FAISS Flat search remains the
  smaller stage.
- **RQ21.4:** Yes for ordinary single-process read requests: model and indexes
  are loaded once and reused, with inference serialized by a lock.
- **RQ21.5:** Authentication, rate limiting, abuse controls, private-data
  governance, and content-safety filtering remain before any public service;
  content-safety filtering is explicitly **NOT IMPLEMENTED**.

## Phase 22: interactive demonstration boundary

Phase 22 adds a user-facing surface without changing the scientific retrieval
path. Streamlit is a suitable lightweight local framework here because the
deliverable is a reproducible research demo rather than a production web
deployment. The app calls the Phase 21 `RetrievalService` directly, while
`st.cache_resource` keeps one loaded service per process across Streamlit
reruns. This makes the dependency boundary explicit: the UI owns controls,
status, formatting, and error presentation; the service owns model loading,
image decoding, embedding, index search, and latency fields.

The two directions are deliberately symmetric at the interface level. A text
query is bounded and passed to `search_text_to_image()`, with ranked image IDs,
scores, filenames, and local previews. An uploaded JPEG, PNG, or WEBP is read
as request-scoped bytes and passed to `search_image_to_text()`, with ranked
caption text and source image IDs. The UI stores only ordinary session state
for the current query and latest result response; it does not write raw upload
bytes or raw query logs to disk.

The top-k control is limited to 1–20 for a compact screen even though the
underlying Phase 21 API accepts up to 50. Empty text, missing image, malformed
image, unavailable resources, and service failures become short UI messages
without tracebacks. A status panel shows model, backend, device, readiness,
default top-k, and API version so a screenshot is interpretable. Confidence and
fusion are not displayed because Phase 22 does not add a new calibrated score
or fusion experiment; retrieval similarities remain scores, not probabilities.

Phase 22 measures backend timing from the service response and direct UI
service-call wall time separately. This avoids presenting browser rendering or
Streamlit rerun time as a model latency claim. The real local evidence under
`artifacts/phase22` combines browser text-to-image results and status/error
checks with a real Streamlit AppTest image-to-text action. The validation
Chrome profile blocked automated local file selection because file-URL access
was disabled; that environment limitation is recorded rather than hidden.

The interface states **CONTENT-SAFETY FILTERING: NOT IMPLEMENTED**. Therefore
the demo is not a moderation tool, safe-deployment gate, or authorization to
expose the corpus or model to untrusted users.

## Phase 23: testing, CI, and reproducibility boundary

Phase 23 hardens the engineering path after the Phase 22 dependency audit. The
test inventory separates deterministic unit tests, module/service integration
tests, artifact validation, and an explicitly excluded real-model smoke. The
default marker expression is `not real_model and not slow and not local_data`,
so a normal test run does not silently require local checkpoints, COCO/CIRCO
bytes, or large retrieval indexes. API and UI tests use small fake services at
their boundaries; the real model smoke remains a separate claim.

Image-grouped retrieval datasets make this split discipline essential. One
image has multiple captions, so a caption-level random split can place one
caption in training and another caption for the same image in test. The model
then sees the test image or its near-equivalent caption during training, and
Recall/MRR can look better without measuring generalization to unseen image
groups. Leakage can also occur when an image ID, caption ID, exact duplicate
image, or derived embedding/index crosses split boundaries. The Phase 1
manifest and leakage checks therefore operate on image groups; Phase 23 keeps
the regression contract around those artifacts rather than weakening it to
caption-level sampling.

The compact GitHub Actions job installs from the frozen `uv.lock`, then runs
pytest, Ruff, mypy, and `compileall`. It deliberately excludes COCO, CIRCO,
Hugging Face, checkpoint, and index downloads so CI remains a bounded code
quality signal. A clean-environment check repeats installation, import,
default tests, and bootstrap in an isolated temporary uv environment. The
project's `.python-version` pins the interpreter family, while the lockfile
pins resolved packages; neither guarantees identical MPS kernels, native
thread behavior, or hardware latency.

Phase 23 also validates a small set of current Phase 21/22 contracts, scans
reusable source/config/documentation scopes for machine-specific absolute
paths and obvious credentials, checks Markdown relative links, and classifies
large files. Raw datasets, checkpoints, indexes, and heavy generated records
are local-only and protected by `.gitignore`; compact JSON provenance and
reports remain reviewable. No secret scanner proves the absence of secrets,
and no duplicate-image detector proves semantic identity: byte hashes catch
exact byte duplicates only, while resized, recompressed, cropped, or
near-duplicate images require a separate perceptual/semantic audit. Historical
artifacts may contain old host paths as provenance; reusable source and current
documentation must not depend on them.

### Phase 23 research questions

- **RQ23.1:** Yes. The default fast path runs without large data/model/index
  downloads and is reproducible from project metadata and the frozen lockfile.
- **RQ23.2:** Real-model smoke is available but intentionally requires the
  local retained resource set; it is not presented as a data-free CI result.
- **RQ23.3:** Yes for the declared local workflow: lock validation, tests,
  lint, typing, compilation, entrypoint help, artifact validation, and clean
  environment checks are recorded under `artifacts/phase23`.
- **RQ23.4:** Yes. API and UI boundary behavior is covered by mocked/fake
  services, while real-resource API/UI evidence remains in Phases 21–22.
- **RQ23.5:** Reproducibility remains limited by hardware, MPS/native kernels,
  local cache availability, large artifact storage, and the absence of a
  remotely executed hosted-CI result.

## Phase 24: deployment and packaging boundary

The primary deployment path is native `uv`. This is the appropriate local
target because it preserves the validated Phase 21 service and can select
Apple MPS with an explicit CPU fallback. The Streamlit process is a thin
in-process UI over the same `RetrievalService`; it does not require a second
API process. The API launcher invokes the existing FastAPI application
factory, so deployment does not introduce a new serving or retrieval stack.

The deployment manifest separates three kinds of reproducibility. Source,
tests, fixtures, `.python-version`, `uv.lock`, and configuration are portable.
The full-FT checkpoint, checkpoint metadata, COCO manifest/image root, Phase 10
float32 embedding cache, FAISS Flat indexes/metadata, Phase 20 provenance, and
offline Hugging Face cache are local-artifact dependent. Device selection,
MPS/native kernels, cold start, and warm latency are hardware dependent. The
manifest stores relative paths, sizes, and SHA-256 identities, so a clean
machine can discover exactly what must be mounted or provisioned without
embedding this host's absolute paths.

Preflight is intentionally separate from model loading. It checks Python and
dependency availability, artifact existence/readability, free disk, selected
device, local model-cache availability when offline, and Phase 20/index/cache
compatibility. It fails with missing-artifact names and an actionable next
step. The actual API smoke then measures process start through `/ready`, while
warm text/image requests are recorded separately; cold startup must not be
reported as retrieval latency.

The launcher centralizes environment behavior. On macOS it sets
`KMP_DUPLICATE_LIB_OK=TRUE` and bounded native thread variables as a known
FAISS/PyTorch OpenMP compatibility workaround. Offline mode sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after the model cache exists;
first-time setup may leave offline mode unset. Host, API/UI ports, root,
artifact paths, device, and offline mode are configuration inputs. Port
conflicts produce an error and do not terminate unrelated processes.

The optional Dockerfile is CPU-only portability scaffolding. It installs the
deployment extra from the frozen lockfile and mounts `artifacts/` and `data/`
at runtime instead of baking large files into the image. Docker was not built
or run in this phase because the local daemon was unavailable, and ordinary
Docker cannot expose Apple MPS. Therefore no Docker performance or deployment
success claim is made. Native MPS evidence and container CPU evidence must not
be conflated.

This remains a research/demo deployment, not a public production service.
Authentication, rate limiting, TLS/reverse-proxy configuration, upload abuse
controls, content safety, privacy governance, monitoring, concurrency/load
testing, and operational rollback are still required before internet exposure.

### Phase 24 research questions

- **RQ24.1:** Yes. Native `uv` launchers, frozen dependencies, environment
  overrides, and preflight provide a reproducible path outside the current
  interactive shell.
- **RQ24.2:** The checkpoint, manifest/image root, embedding cache, FAISS Flat
  indexes/metadata, Phase 20 artifacts, and offline model cache remain local.
- **RQ24.3:** Native API cold start was measured at 6.10 seconds on the Apple
  macOS host with MPS selected; this excludes warm request latency.
- **RQ24.4:** Yes for the retained Phase 22 image-result prefix and repeated
  deployment requests; model/backend identity and deterministic IDs matched.
- **RQ24.5:** Docker was not practical to validate here because its daemon was
  unavailable and it cannot provide MPS; it remains an explicitly unvalidated
  CPU portability path.
- **RQ24.6:** Public deployment needs auth, rate limiting, TLS, upload abuse
  protection, content safety, privacy governance, and production operations
  validation.

## Phase 25: observability and runtime reliability boundary

Phase 25 keeps the Phase 24 native deployment and adds a deliberately small
runtime contract. The API middleware creates or preserves a bounded printable
request ID, returns it in `X-Request-ID`, and places it in retrieval bodies
and structured logs. The log formatter emits JSON with timestamp, level,
endpoint, status, latency, the existing preprocessing/encoding/search/total
latency terms when available, and an error category. It does not log raw
queries, uploaded bytes, secrets, or reusable absolute paths. This makes an
individual local request traceable without turning research logs into a data
collection channel.

The error taxonomy distinguishes `VALIDATION_ERROR`, `RESOURCE_NOT_READY`,
`MODEL_ERROR`, `IMAGE_DECODE_ERROR`, `STARTUP_ERROR`, and
`INTERNAL_ERROR`. `/health` is deliberately lightweight and remains callable
when resources are degraded. `/ready` is the load gate: it is successful only
after the validated model and both compatible indexes are available. The
process-local `/metrics` snapshot reports total/success/failure and text/image
request counters, error counts, device, model identity, startup/shutdown
state, and uptime. It resets on restart and is not a Prometheus backend or an
availability guarantee.

The Phase 25 smoke launches the real API and UI path, then runs 20 text and
20 image requests, checks result/request-ID consistency, injects malformed
image, invalid top-k, unready, model, missing checkpoint/index, and incompatible
metadata failures, and runs a four-worker sample plus a 20-second alternating
soak. The latency comparison uses the Phase 24 warm samples and excludes the
first two Phase 25 requests as warm-up. This is an engineering regression
check, not a user-facing latency SLO.

Shutdown analysis matters because Phase 24 recorded a nonfatal
`resource_tracker` leaked-semaphore warning. The service now clears its model,
processor, index, and device references on lifespan shutdown, and the UI
registers equivalent process-exit cleanup. If the warning persists, it is
reported as lower-level Python/native-library cleanup noise while clean exit,
readiness after startup, and request behavior remain separately measured; the
warning is not suppressed to manufacture a clean result.

### Phase 25 research questions

- **RQ25.1:** Yes for the local native path: startup checks, request IDs,
  structured logs, counters, health/readiness, failure taxonomy, and lifecycle
  cleanup are implemented.
- **RQ25.2:** Runtime counters are process-local; durable telemetry, log
  collection, retention, and alerting must be supplied by deployment.
- **RQ25.3:** Cold start is measured from native process start through HTTP
  `/ready`; warm retrieval stages remain separate from startup.
- **RQ25.4:** The real smoke compares model/backend identity and deterministic
  result IDs with the retained Phase 24/22 canonical outputs.
- **RQ25.5:** A small four-worker check and short soak provide bounded runtime
  evidence only; no production concurrency or uptime claim is made.
- **RQ25.6:** Public production still requires authentication, rate limiting,
  TLS/reverse proxy, upload-abuse controls, content safety, privacy governance,
  durable monitoring, and operational load/rollback validation.

## Phase 26: end-to-end validation and final release boundary

Phase 26 answers whether the validated system can be run as a reproducible
research/demo outside the current shell. The answer is yes for the native
`uv` path when the declared local artifacts are present. The preflight checks
runtime/dependencies, disk and permissions, device selection, and checkpoint,
manifest, cache, index, and metadata compatibility. The final integrity
artifact then records the exact checkpoint and manifest hashes, 512-dimensional
float32 normalized embeddings, 5,000 image/25,014 caption counts, Flat index
types, candidate units, and the fact that test data did not select the
checkpoint.

The frozen architecture remains:

`Streamlit UI -> RetrievalService/API -> Phase 7 full-FT CLIP -> cached embeddings -> FAISS Flat exact search`.

This keeps the quality and serving decisions aligned. Reranking is disabled
because Phase 11 measured a negative quality result; ANN is a future scale
option rather than a default claim; LoRA and hard negatives are optional
research components; fusion is not part of the final serving path. Full-FT
quality is reported from the retained Phase 7 held-out protocol, while live
deployment evidence is intentionally limited to serving identity, health,
both request directions, deterministic smoke results, and lifecycle behavior.

The native run on the actual macOS host selected MPS and reached `/ready` in
11.139 seconds. Warm server latency, with cold start excluded, was:

- text→image: mean 12.31 ms, median 11.62 ms, p95 15.46 ms;
- image→text: mean 27.50 ms, median 24.48 ms, p95 37.00 ms.

The latency artifact keeps preprocessing, query encoding, search, total
server, and client-wall measurements separate. Cold start is not mixed into
warm request latency. A four-worker eight-request check and repeated-query
stability check passed, but this is not a production throughput, autoscaling,
uptime, or network benchmark. The nonfatal macOS `resource_tracker` warning
remains a lower-level cleanup limitation when observed.

Docker is not the primary path because ordinary Docker does not provide Apple
MPS; the included path is CPU-only portability scaffolding and was not run in
Phase 26. Native MPS latency must not be represented as container latency.
Likewise, a local launch is not public production: authentication, rate
limiting, TLS/reverse proxy, durable telemetry, content safety, privacy
governance, and upload-abuse controls remain required.

### Phase 26 research questions

- **RQ26.1:** Yes for the native configured-artifact research/demo path; the
  concise command is `KMP_DUPLICATE_LIB_OK=TRUE uv run omnisearch-phase26`.
- **RQ26.2:** The full-FT checkpoint, COCO image root/manifest, float32 cache,
  Flat indexes/metadata, Phase 20 provenance, and offline Hugging Face model
  cache remain local-artifact dependencies.
- **RQ26.3:** Cold start was 11.139 seconds on the recorded macOS MPS host;
  warm timings are recorded separately.
- **RQ26.4:** Model/backend identity and deterministic smoke IDs matched the
  prior canonical service. Exact equality to the Phase 7 100-candidate test
  corpus is not claimed because the live service uses 5,000 images.
- **RQ26.5:** Docker was not validated for the MPS-backed demo and is only an
  optional CPU portability path.
- **RQ26.6:** Public production needs auth, rate limiting, TLS/reverse proxy,
  upload-abuse controls, content safety, privacy governance, durable
  monitoring, and operational rollback/load validation.
