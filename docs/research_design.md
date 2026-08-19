# Research design

## Working title

**OmniSearch — Multimodal Semantic Search & Representation Learning**

## Core research question

How effectively can learned text-image representations retrieve semantically relevant information across modalities, and how do representation, fine-tuning, reranking, and retrieval strategies affect relevance, robustness, fairness, efficiency, and generalisation?

## Active benchmark and migration

The active benchmark is official MS COCO 2017 `val2017`, scoped locally to its
5,000 images and 25,014 caption records. The internal retrieval split is
image-grouped and deterministic. Historical Flickr30k metadata and artifacts
are preserved, but its unavailable authorized image archive means it is not a
required primary dependency.

## Sub-questions

1. How do classical text/image representations compare with a frozen pretrained CLIP-style dual encoder on image-to-text and text-to-image retrieval?
2. Does task-specific contrastive adaptation improve held-out retrieval quality over frozen representations, and what is the cost in robustness or compute?
3. Do hard negatives and second-stage reranking improve top-k relevance, especially for visually or linguistically similar distractors?
4. Can image and text query embeddings be fused into a useful multimodal query without making the system difficult to calibrate or explain?
5. How do index type, candidate count, and model size trade retrieval quality against memory, build time, and query latency?
6. Which groups of queries or concepts fail disproportionately, and how should those failures limit the system's claims?

## Falsifiable hypotheses

These are hypotheses to test, not results:

- **H1 — pretrained alignment:** a frozen pretrained shared embedding will outperform the classical baseline on Recall@K and NDCG@K for both retrieval directions.
- **H2 — task adaptation:** contrastive fine-tuning on the training split will improve in-domain held-out retrieval over the frozen encoder, but may reduce performance on an external or distribution-shift set.
- **H3 — hard negatives:** training with semantically confusable negatives will improve ranking among the top candidates more than uniform negatives, particularly at small K.
- **H4 — reranking trade-off:** a cross-modal reranker will improve NDCG@K or Recall@K over first-stage retrieval at a measurable latency and memory cost.
- **H5 — approximate retrieval:** FAISS/HNSW will reduce query latency or memory relative to exact search at sufficiently large index sizes, with a quality loss that can be measured rather than assumed.
- **H6 — uneven performance:** retrieval quality will vary across observable content or language-related strata; aggregate scores alone will conceal some failure modes.

## Primary outcomes

The primary quality outcomes are bidirectional Recall@1, Recall@5, Recall@10, median rank, and NDCG@10. Retrieval will use all valid captions for an image as relevant positives. Secondary outcomes are mean reciprocal rank, mAP where the relevance definition is appropriate, calibration/selective-retrieval measures, and resource metrics.

No target score is set in Phase 0. Baseline ordering and confidence intervals will be established only from executed experiments.

## Research boundaries

The primary unit is cross-modal retrieval, not image caption generation, visual question answering, or a general-purpose chatbot. A natural-language search interface is a later presentation layer for the measured retrieval system. Large-scale pretraining is out of scope unless external compute and a clear comparison budget become available.

## Portfolio fit

The project deliberately avoids duplicating the existing recommender, streaming-fraud, tabular-health, churn, and generic MLOps projects. Its defensible contribution is a coherent sequence of representation-learning experiments with retrieval evaluation, statistical comparison, robustness analysis, and a thin engineering layer that serves the measured model.
