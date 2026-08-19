# Mathematical foundations

## Shared CLIP representation

Let (f(x) in mathbb{R}^d) be the image embedding for image (x), and
let (g(t) in mathbb{R}^d) be the text embedding for text (t). Phase 4
uses the frozen pretrained CLIP encoders; it does not train either function.

Each vector is L2-normalized:

\[
\hat f(x)=\frac{f(x)}{\lVert f(x)\rVert_2},\qquad
\hat g(t)=\frac{g(t)}{\lVert g(t)\rVert_2}.
\]

The exact retrieval score is:

\[
s(x,t)=\hat f(x)^\top\hat g(t).
\]

Because both vectors have unit norm, this dot product equals cosine
similarity. The implementation rejects zero, NaN, and infinite embeddings.

## Ranking and Recall@K

For a query (q), candidates are sorted by decreasing (s(q,c)), with a
stable ID tie-break. If (R(q)) is the set of relevant candidates, then:

\[
\mathrm{Recall@K}(q)=
\mathbf{1}\left[\operatorname{rank}_q(R(q))\le K\right].
\]

The reported aggregate is the mean of this indicator over evaluated queries.
For text-to-image retrieval, the image corresponding to the query caption is
relevant. For image-to-text retrieval, all captions belonging to the query
image are relevant; this is important because COCO has multiple captions per
image group.

## Similarity matrix

Given normalized query rows (Q\in\mathbb{R}^{m\times d}) and candidate rows
(C\in\mathbb{R}^{n\times d}), the exact score matrix is:

\[
S=QC^\top.
\]

Phase 4 computes the equivalent pairwise scores directly over feasible
evaluation sets. No approximate nearest-neighbor index is introduced.

## Contrastive pretraining intuition

CLIP was pretrained by making matched image-text pairs score higher than
mismatched pairs within a batch. Conceptually, for temperature \(\tau\), an
image-to-text term has the form:

\[
\mathcal{L}_{i\rightarrow t}=-\frac{1}{B}
\sum_{i=1}^{B}\log
\frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{B}\exp(S_{ij}/\tau)}.
\]

A symmetric text-to-image term is averaged with it. This explains why a
frozen model can transfer to zero-shot retrieval. OmniSearch does not run
this loss, update weights, tune temperature, or use COCO captions for
training in Phase 4.

## Evaluation status

The actual run produced 512-dimensional normalized embeddings on MPS for 256
COCO test captions and real image candidates. A two-image/two-caption fixture
also exercises both retrieval directions and remains stored as fixture-only
evidence; it must not be interpreted as a COCO benchmark result.

## Unified retrieval metrics

For query (q), let (L_q=(c_1,ldots,c_m)) be the returned ranking and let
(R_q) be its declared relevance set. With (I_k(q)={c_i:ileq K}):

\[
P@K(q)=\frac{|I_k(q)\cap R_q|}{K},\qquad
R@K(q)=\frac{|I_k(q)\cap R_q|}{|R_q|}.
\]

The denominator of Precision@K is K even when fewer than K candidates are
returned. Queries with (|R_q|=0) are excluded from metric averages and
reported as no-relevance queries.

The first relevant rank is

\[
r_q=\min\{i:c_i\in R_q\},
\]

when such an (i) exists. Reciprocal rank is (1/r_q), or zero for a miss,
and MRR is its macro mean. Average precision over a returned list is:

\[
AP(q)=\frac{1}{|R_q|}\sum_{i=1}^{m}
P@i(q)\,\mathbf{1}[c_i\in R_q].
\]

If (m<K) or the list is intentionally top-K truncated, AP/MAP is a
truncated estimate and the artifact records that scope. For binary relevance,

\[
DCG@K(q)=\sum_{i=1}^{K}\frac{\mathbf{1}[c_i\in R_q]}{\log_2(i+1)},
\qquad NDCG@K(q)=\frac{DCG@K(q)}{IDCG@K(q)}.
\]

The evaluator uses macro query means over queries with declared relevance. A
query-level bootstrap samples those query metric values with replacement. For
systems A and B evaluated on the same query set, paired deltas are

\[
\Delta(q)=M_B(q)-M_A(q),
\]

and the reported estimate is the mean of (Delta(q)), with a bootstrap over
queries. This is valid only when the query IDs, relevance sets, split, and
candidate corpus are held fixed.

## Frozen transformer text representations

Given token hidden states H and binary attention mask m, the Phase 6
mean-pooled representation is

`z = sum_l(m_l H_l) / max(1, sum_l m_l)`, followed by
`z_hat = z / ||z||_2`.

The denominator excludes padding. For normalized query and candidate matrices
Q and C, the exact same-space score matrix is `S = Q C^T`. Phase 6 rejects
mismatched space identifiers, so this equation is not used to compare
arbitrary text and image encoders. CLIP is the exception because its
checkpoint learned matching projection spaces.

Transformer self-attention is

`Attention(Q,K,V) = softmax((QK^T) / sqrt(d_k)) V`.

## Phase 10 nearest-neighbor retrieval

Let normalized candidate vectors be (C={c_1,ldots,c_n}) and a normalized
query be (qinmathbb{R}^d). Exact top-(K) retrieval is:

\[
\operatorname{TopK}_{\mathrm{exact}}(q)=
\operatorname{arg,topK}_{i\in\{1,\ldots,n\}} q^\top c_i.
\]

For unit vectors, (q^\top c_i=\cos(q,c_i)). With float32 storage, the raw
vector cost is (4nd) bytes before IDs and index metadata. Phase 10 records
both this calculated raw storage and the serialized index size.

An approximate index returns

\[
\operatorname{TopK}_{\mathrm{ANN}}(q)\approx
\operatorname{TopK}_{\mathrm{exact}}(q).
\]

Its embedding-space fidelity at K is the overlap

\[
\operatorname{NeighborRecall@K}=
\frac{|\operatorname{TopK}_{\mathrm{ANN}}(q)\cap
\operatorname{TopK}_{\mathrm{exact}}(q)|}{K},
\]

averaged over queries. This is distinct from semantic retrieval Recall@K,
which uses the declared image-caption relevance set.

Inverted-file (IVF) search learns (nlist) coarse cells and probes only
(nprobe) of them. Increasing (nprobe) usually raises neighbor fidelity and
work; decreasing it usually lowers both latency and fidelity. IVF-Flat keeps
the original vectors inside the selected cells, so its approximation comes
from cell pruning rather than vector compression.

HNSW builds a layered proximity graph. (M) limits graph connectivity,
(efConstruction) controls graph-building breadth, and (efSearch) controls
the breadth explored per query. Larger (efSearch) generally improves the
chance of finding exact neighbors at a latency cost. Neither IVF nor HNSW
guarantees the true nearest neighbor; the exact index is the correctness
reference.

Mean pooling is an explicit, reproducible choice. CLS pooling would instead
select the first token state; native model pooling is used for the ViT image
control and CLIP's learned projections.

## CNN and ViT image representations

A convolutional layer applies a shared local kernel over spatial positions,
building increasingly large receptive fields through depth. ResNet adds a
residual mapping `y = F(x) + x`, which supports stable optimization in deep
CNN stacks. The Phase 6 ResNet representation is the model's native global
pooled feature followed by L2 normalization.

ViT converts an image into patch tokens, adds positional information, and
applies transformer attention over all patches. This supplies global token
interaction earlier than a local CNN but relies more heavily on pretraining and
data-scale inductive bias. Phase 6 retains the model-native pooled
representation and compares it qualitatively on a two-image fixture only; no
image relevance labels were invented.

## Phase 6 statistical scope

The text metrics are query means over 256 deterministic test-caption queries,
with the other captions for the same image group as relevance and the query
caption excluded. Bootstrap intervals resample queries with the configured
seed and resample count. Comparisons to TF-IDF and BM25 are paired because the
query IDs, candidate corpus, split, and relevance map are held fixed. Cache
hits do not alter scores; they only avoid repeating the same frozen extraction.

## Phase 7 symmetric CLIP fine-tuning

Let `u_i = f_theta(x_i) / ||f_theta(x_i)||_2` and
`v_i = g_phi(t_i) / ||g_phi(t_i)||_2` be the normalized projected image and
caption embeddings for a batch of `B` unique image groups. Let `s` be the
trainable scalar logit-scale parameter. The similarity logits are

`S_ij = exp(min(s, log(100))) u_i^T v_j`.

The positive target for row `i` is column `i`. The symmetric contrastive loss
used in Phase 7 is

`L_i2t = -(1/B) sum_i log softmax(S_i,:)_i`,

`L_t2i = -(1/B) sum_i log softmax(S^T_i,:)_i`,

`L = (L_i2t + L_t2i) / 2`.

All encoder, projection, and logit-scale parameters are trainable in this
phase. With learning rate `eta`, the conceptual update is
`theta <- theta - eta * grad_theta L` and `phi <- phi - eta * grad_phi L`.
Gradient accumulation performs several micro-batches before one optimizer
update; the executed Tier 2 run used 4 micro-batches of 2 pairs, for an
effective batch size of 8. The implementation clips the global gradient norm
to 1.0 and rejects non-finite loss or gradients.

The one-caption-per-image training sampler is important to the objective: if
two captions from the same image were placed in one diagonal batch, they
would be incorrectly treated as negatives for one another. Full multi-caption
relevance is retained for validation and test ranking, where all captions of
the query image are relevant in the image-to-text direction.

Checkpoint selection maximizes validation mean Recall@5 across the two
directions. The test split is not loaded until after that selection. Therefore
the reported test delta compares the selected fine-tuned checkpoint with a
fresh frozen copy of the same pretrained checkpoint under the same
`retrieval_eval_v1` candidate corpora and relevance maps.

## Phase 8 LoRA update

For a frozen base weight matrix `W` with shape `d_out x d_in`, LoRA uses

`W' = W + Delta W`,

`Delta W = (alpha / r) B A`,

where `A` has shape `r x d_in`, `B` has shape `d_out x r`, and the update rank
is at most `r`. Only `A` and `B` receive gradients; `W` is frozen. The adapter
parameter count is therefore `r(d_in + d_out)` instead of `d_in*d_out` for
that matrix. Dropout may be applied to the adapter branch during training.

Phase 8 uses `r=8`, `alpha=16`, and targets CLIP attention `q_proj` and
`v_proj` modules in both encoders. The trainable logit-scale scalar is stored
as explicitly declared extra adapter state. The measured total is 491,521
trainable parameters versus 151,277,313 for Phase 7 full fine-tuning.

The contrastive objective remains the Phase 7 symmetric loss; LoRA changes the
parameterization of the update, not the retrieval labels or the objective.
The practical trade-off is a much smaller adaptation artifact and fewer
updated parameters, at the cost of a restricted update subspace. A low-rank
adapter can miss coordinated changes that full fine-tuning can express, which
is consistent with the observed Tier 2 R@5 regressions.

## Hard-negative contrastive objective

Let normalized positive image/text embeddings be `v_i` and `t_i`, and let
`h^v_j` and `h^t_j` be mined non-positive embeddings for selected rows. With
temperature scale `s`, the image-to-text logits are

`L^(I->T)_ij = s * v_i^T [t_1,...,t_B,h^t_1,...,h^t_H]_j`.

The target for image `i` remains the diagonal index `i`; mined columns are
additional negatives, not new positives. The text-to-image direction is

`L^(T->I)_ij = s * t_i^T [v_1,...,v_B,h^v_1,...,h^v_H]_j`.

The symmetric loss is the mean of the two cross-entropies. A high negative
logit increases the denominator and its gradient pressure. If a mined pair is
actually semantically positive, the same objective applies the wrong penalty;
this is the false-negative failure mode. Same-image and exact caption aliases
are therefore removed before mining, while cross-image semantic equivalence
remains a documented limitation.

## Phase 11 candidate reranking

Let `q` be a normalized query vector and let `D` be the split-specific
candidate corpus. Stage 1 uses inner product, which equals cosine similarity
for normalized vectors:

`s_1(q,d) = q^T d`,

`C_N(q) = TopN_{d in D} s_1(q,d)`.

The candidate recall at depth `N` is the fraction of the declared relevant
set that appears in `C_N(q)`, averaged over queries. Candidate hit rate is the
indicator that at least one relevant item appears. The former matters for
image-to-text, where an image has multiple relevant captions; the latter
shows whether a query has any recoverable positive at all.

Phase 11's candidate-specific interaction vector is

`phi(q,d) = [q ⊙ d, |q-d|, q^T d]`,

and the reranker emits

`s_2(q,d) = MLP_theta(phi(q,d))`.

The pairwise training loss for a positive `d+` and a train-only negative `d-`
is the softplus margin objective

`L(theta) = log(1 + exp(m - s_2(q,d+) + s_2(q,d-)))`,

with margin `m=0.1`. At inference, Stage 2 sorts only `C_N(q)` by `s_2`; it
does not change the candidate universe. The Phase 11 implementation uses the
reranker score for ordering and does not interpret it as a probability.

The oracle upper bound is an analysis-only ranking that places every relevant
candidate in `C_N(q)` before every non-relevant candidate. It is not learned
and must not be compared as a model. Its Recall@K reports the maximum quality
available from the selected candidate set under the declared relevance map.

The two-stage latency decomposition is

`T_end_to_end = T_encode(q) + T_stage1_search(q,N) + T_rerank(q,C_N(q))`.

Phase 11 measures these components separately. For a fixed MLP, reranking
work is approximately linear in `N` and the pair-feature dimension, while
Stage-1 exact search is linear in corpus size for a flat index. The recorded
negative quality result shows that lower loss on train-only pairwise examples
does not imply improved Recall@1, Recall@5, or MRR on the multi-caption test
protocol.

## Phase 12 multimodal query fusion

Let `q_i` and `q_t` be normalized CLIP image and text query embeddings, and
let `c_j` be a normalized candidate image embedding. Alpha is the image
weight, with `0 <= alpha <= 1`.

Early weighted-embedding fusion is

`q_fused = normalize(alpha * q_i + (1-alpha) * q_t)`.

The candidate score is

`s_early(c_j) = q_fused^T c_j`.

The explicit post-sum normalization prevents the magnitude of the weighted
sum from changing the interpretation of the score and keeps inner product
equal to cosine similarity in the candidate space.

Late score fusion is

`s_late(c_j) = alpha * (q_i^T c_j) + (1-alpha) * (q_t^T c_j)`.

Because

`s_early(c_j) = [alpha(q_i^T c_j) + (1-alpha)(q_t^T c_j)] / ||alpha q_i + (1-alpha)q_t||_2`,

the denominator is independent of candidate `c_j`. Thus early and late
fusion have identical ranking order for the same alpha when both query and
candidate vectors are normalized and the same cosine space is used. Their
numeric scores differ by a positive query-specific factor, so ranking metrics
match even though raw score values need not.

Image-only and text-only are the alpha endpoints, evaluated as explicit
controls rather than selected fusion configurations. The controlled
same-image identity relevance set permits paired R@K/MRR comparisons, while
arbitrary compositional text modifications have no quantitative relevance
function and are therefore excluded from metric claims.

## Phase 12B CIRCO metrics

For a CIRCO query with ranked candidates `r_1, ..., r_K`, released target
`g*`, and multiple-ground-truth set `G`, the benchmark-specific metrics are:

`AP@K = (1 / min(|G|, K)) * sum_{i=1..K} P(i) * 1[r_i in G]`,

where `P(i)` is precision through rank `i`. Official Recall@K is
`1[g* in {r_1, ..., r_K}]`. Thus a valid alternative ground truth contributes
to mAP but does not replace CIRCO's official single-target Recall@K. The
reference image is excluded from `G` by schema validation.

The Phase 12B alpha-selection protocol uses a deterministic split of labeled
CIRCO validation queries into `V_select` and disjoint `V_holdout`; it does not
use withheld CIRCO test labels. This prevents selecting alpha on the same
queries used for the local comparison while preserving the benchmark's
multiple-target semantics.

## Phase 13 statistical validation

For a metric value `x_s` from seed `s` and `n=3` predeclared seeds, the sample
mean is

`x_bar = (1/n) * sum_s x_s`.

The sample standard deviation is

`s_x = sqrt((1/(n-1)) * sum_s (x_s - x_bar)^2)`.

For paired query `q`, a comparison delta is

`delta_q = m_B(q) - m_A(q)`,

where `A` is the baseline and `B` is the comparison system. The observed
paired effect is the macro mean `delta_bar` over the identical query IDs.
Query bootstrap repeatedly samples those query IDs with replacement and
recomputes `delta_bar`; the empirical 2.5th and 97.5th percentiles form the
reported 95% interval. This measures query-level uncertainty, not training
variability.

The paired permutation test randomly flips the sign of each observed
`delta_q` under the null that paired differences are exchangeable around zero.
The two-sided p-value is the fraction of randomized absolute means at least
as extreme as the observed absolute mean, with a small finite-resample
correction. Holm–Bonferroni orders the p-values from smallest to largest and
controls the family-wise error rate by comparing each ordered value with its
rank-dependent threshold. Statistical non-rejection does not imply practical
equivalence; absolute deltas, relative changes where meaningful, and
win/loss/tie counts are reported separately.

## Phase 14 ablation effects

For a full system `F` and an ablated system `A`, the component contribution
for metric `m` is

`Delta_m = m(F) - m(A)`.

When the same query IDs and relevance sets are retained, the paired query
effect is

`Delta_q = m_F(q) - m_A(q)`,

and the reported point effect is the macro mean over queries. Resampling the
paired query vectors gives uncertainty around that delta. Across training
seeds, the mean and sample standard deviation describe a separate source of
variation; they are not pooled with the query bootstrap interval.

Component effects need not be additive. For components `X` and `Y`, the
effect of `X+Y` can differ from `effect(X) + effect(Y)` because the learned
representation, optimization path, negative distribution, and candidate
ranking interact. Phase 14 therefore interprets each comparison only under
its declared fixed parent protocol and does not extrapolate beyond the
observed COCO/Tier-2 scope.

## Phase 15 robustness quantities

For query `q`, clean metric value `m_clean(q)`, and corrupted value
`m_corrupt(q)`, the condition-level absolute change is

`Delta_m = mean_q[m_corrupt(q) - m_clean(q)]`.

When the clean aggregate is nonzero, relative degradation and retention are

`D_m = (m_clean - m_corrupt) / m_clean`,

`R_m = m_corrupt / m_clean`.

The Phase 15 bootstrap resamples the same query IDs with replacement and
recomputes `Delta_m`, so its interval measures uncertainty over aligned test
queries. It is not a training-seed interval and does not make the disjoint
aspect-ratio shift groups paired.

For clean and corrupted returned rankings `L_clean` and `L_corrupt`, top-K
overlap is

`Overlap_K = |prefix_K(L_clean) intersect prefix_K(L_corrupt)| / K`.

If the first relevant ranks are `r_clean` and `r_corrupt` and both are
observed in the returned top-10 lists, rank displacement is
`r_corrupt - r_clean`. Missing first hits are reported as censored rather than
assigned an arbitrary rank. This keeps rank-stability summaries honest when
the evaluator stores only a top-10 prefix.

The declared distribution shift compares two disjoint group sets selected by
aspect ratio. For a direction and metric, its descriptive effect is

`Delta_shift = mean_{q in shifted}[m(q)] - mean_{q in control}[m(q)]`.

It is a between-group descriptive difference, not a paired query estimate.
