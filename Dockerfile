# Optional CPU portability image. Local artifacts are mounted at runtime;
# Apple MPS is not available in ordinary Docker containers.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra deployment

ENV OMNISEARCH_ROOT=/app \
    OMNISEARCH_DEVICE=cpu \
    OMNISEARCH_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "omnisearch-api"]
