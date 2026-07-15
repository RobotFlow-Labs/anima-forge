# FORGE — Multi-stage Dockerfile
# Stage 1: Base (CPU validation) — for CI and local development
# Stage 2: CUDA — for GPU training, inference, and TensorRT export

# ============================================================
# STAGE: base — CPU/MLX development and testing
# ============================================================
FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml README.md uv.lock hatch_build.py ./
COPY src/ src/

# Cache dependencies independently of the per-commit project provenance layer.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --group dev --no-install-project

# Preserve exact source provenance without copying the private Git history into
# the image. Builds fail closed unless the caller supplies a full commit SHA.
ARG FORGE_GIT_SHA
RUN printf '%s\n' "${FORGE_GIT_SHA}" | grep -Eq '^[0-9a-f]{40}$' || \
    (echo "FORGE_GIT_SHA must be a full 40-character lowercase Git commit SHA" >&2; exit 2)
ENV FORGE_GIT_SHA=${FORGE_GIT_SHA}
RUN uv sync --locked --group dev

COPY configs/ configs/
COPY tests/ tests/
COPY scripts/ scripts/
COPY docs/ docs/
COPY .claude/skills/forge/ .claude/skills/forge/
COPY .github/ .github/
COPY marketing/README.md marketing/README.md
COPY marketing/release/ marketing/release/
COPY benchmarks/.gitkeep benchmarks/.gitkeep
COPY .dockerignore .gitignore CHANGELOG.md CITATION.cff CODE_OF_CONDUCT.md ./
COPY COMPATIBILITY.md CONTRIBUTING.md LICENSE SECURITY.md docker-compose.yml ./
COPY Dockerfile Dockerfile
COPY install.sh install.ps1 ./

# Default: run tests
CMD ["uv", "run", "pytest", "tests/", "-v", "--tb=short"]

# ============================================================
# STAGE: cuda — GPU training and inference
# ============================================================
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS cuda

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv python install 3.12
ENV UV_PYTHON=3.12

# Copy project files
COPY pyproject.toml README.md uv.lock hatch_build.py ./
COPY src/ src/

# CUDA, TensorRT, and every teacher runtime are mandatory Linux dependencies.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --group dev --no-install-project

ARG FORGE_GIT_SHA
RUN printf '%s\n' "${FORGE_GIT_SHA}" | grep -Eq '^[0-9a-f]{40}$' || \
    (echo "FORGE_GIT_SHA must be a full 40-character lowercase Git commit SHA" >&2; exit 2)
ENV FORGE_GIT_SHA=${FORGE_GIT_SHA}
RUN uv sync --locked --group dev

COPY configs/ configs/
COPY tests/ tests/
COPY scripts/ scripts/

# Mount point for models
VOLUME /models

ENV FORGE_MODEL_DIR=/models
ENV FORGE_DEVICE=cuda

ENTRYPOINT ["uv", "run", "forge"]
CMD ["--help"]
