# Portfolio summary

I built OmniSearch to study text-to-image and image-to-text retrieval as a
complete ML system rather than only a model-training exercise. The project
uses CLIP representations, full fine-tuning, normalized cached embeddings,
and exact FAISS Flat search, then exposes the same retrieval service through
FastAPI and Streamlit. I compared zero-shot CLIP, full fine-tuning, LoRA,
hard negatives, approximate indexes, reranking, robustness, confidence, and
explanation methods under fixed evaluation protocols. The final held-out COCO
results were 0.8263 text-to-image R@1 and 0.9880 R@5. I also measured the real
native MPS deployment: it reached readiness in 11.139 seconds, with warm mean
server latency of 12.31 ms for text-to-image and 27.50 ms for image-to-text.
The most useful conclusion was negative: the tested reranker made retrieval
worse, so the simpler exact-search system remained the final design.

## How to read the repository

The `phase*.py` modules are reproducibility entry points for the research
history, not separate production services. The final user-facing path is the
shared `RetrievalService`, exposed through FastAPI and Streamlit and supported
by the deployment, evaluation, and configuration modules. Keeping the phase
entry points preserves repeatability without making the repository's main
navigation depend on the number of experiments.
