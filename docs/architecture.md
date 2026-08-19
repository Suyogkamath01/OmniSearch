# System architecture

The system is organized around reproducible experiment boundaries rather than around the eventual UI.

```text
official COCO metadata/images
        |
        v
data acquisition -> validation -> image-grouped split -> processed manifest
        |                                      |
        v                                      v
 text/image preprocessing                 experiment runner
        |                                      |
        +-------------> encoders -------------+
                         |                    |
           classical / CLIP / tuned dual encoder
                         |
                 normalized embeddings
                         |
             exact / FAISS / HNSW index
                         |
          candidate retrieval -> optional reranker
                         |
       unified evaluator -> metrics + uncertainty + artifacts
                         |
                 API / interactive app (Phase 21–22 adapters)
```

## Module responsibilities

### Data layer

Acquisition is explicit and resumable. The active adapter handles official COCO `val2017`; the historical Flickr30k adapter remains separate. Validation produces a common manifest with a stable `image_id`, filename, source URL, caption IDs, original caption text, split, and source-ID availability. Local image checksums and decoder status live in the separate image-validation report. The splitter groups by `image_id`, so captions cannot cross train/validation/test boundaries.

### Representation layer

The first baseline will be classical and intentionally understandable: TF-IDF text features and a fixed visual feature extractor or low-dimensional image descriptor, with a clearly documented cross-modal matching rule. Later encoders will expose the same `encode_text` and `encode_image` contract, return normalized vectors with recorded dimension, and never decide how metrics are calculated.

### Retrieval layer

Retrieval accepts query vectors and an index interface. Exact cosine search is the correctness reference. FAISS and HNSW are later interchangeable implementations behind the same interface. Index build and query measurement are separated from model encoding so latency claims can be attributed correctly.

### Evaluation layer

The evaluator consumes ranked IDs and a relevance map. It computes both directions with multi-positive ground truth and emits versioned JSON/CSV artifacts containing configuration, split checksum, seed, package versions, device, and metric definitions.

### Serving layer

FastAPI and Streamlit are adapters over the retrieval service. They must not contain training logic or alter metric definitions. The app is a demonstration of a measured research artifact, not evidence of model quality by itself.

## Data and leakage controls

- Split at image level before fitting vocabulary, normalization, hard-negative mining, or learned parameters.
- Fit TF-IDF and any statistics only on training captions.
- Keep validation for model selection and configuration; reserve test for the declared evaluation.
- Treat the five captions for one image as a group in split checks and uncertainty resampling.
- Record preprocessing and split versions in artifact metadata.
- Do not use test captions for prompt engineering, threshold selection, or error-driven training.

## Phase 1 implementation boundary

Acquisition/manifest utilities, validation, deterministic splitting, tier handling, preprocessing interfaces, tests, and documentation are implemented. Model-specific transforms, ANN indexes, APIs, dashboards, and training remain outside the migration scope.

## Final serving path

The released research/demo path is narrower than the full experiment tree:

```text
text -> CLIP text encoder -> L2-normalized 512D vector -> IndexFlatIP -> images
image -> CLIP image encoder -> L2-normalized 512D vector -> IndexFlatIP -> captions
                                      ^
                         cached corpus embeddings and metadata
                                      ^
                  RetrievalService -> FastAPI or Streamlit
```

The Phase 7 full-FT checkpoint supplies the encoders. Phase 10 supplies the
float32 cache and exact indexes. The API and UI are adapters over that service;
they do not retrain, rerank, or change the relevance protocol. The Phase 11
reranker is retained as a negative experiment but is disabled in the final
configuration. ANN and fusion remain research or future-scale options.

## Scale handoff

The retrieval boundary is split into query encoding, cached normalized
vectors, index loading, and a small search adapter. The current deployment
loads persisted FAISS Flat indexes once and reuses them for every request.
Index identity, candidate IDs, dimensions, and manifest/checkpoint hashes are
checked before the service becomes ready.

Phase 10 also measured FAISS IVF-Flat and HNSW alternatives at the same
declared scale. They were faster in search but did not meet the retained
neighbor-fidelity threshold in the tested settings. They remain measured
scale-out evidence rather than defaults. A future larger-corpus deployment
can replace the persisted backend behind the existing index-loading boundary,
but it must repeat candidate-fidelity and retrieval-quality checks before
changing the default. No million-item benchmark was run.

The handoff rule is: keep Flat while exact search is acceptable for latency
and memory; evaluate ANN when the corpus or service load makes that trade-off
necessary; then compare it against the exact reference on the same query and
candidate protocol.
