# Multi-stage build for the audit service (§10).
#
# The blueprint's sketch assumes requirements.txt and pip; this project uses uv and a lockfile,
# so the shape is the same and the commands are not. Two things dominate the image size and both
# are handled deliberately:
#
#   torch          517 MB on the default index, because sentence-transformers pulls it for the
#                  local MiniLM embeddings. The Linux default wheel bundles CUDA, which is dead
#                  weight in a CPU service -- the cpu index is roughly 200 MB instead.
#   the models     MiniLM (~87 MB) and FlashRank's TinyBERT (~3 MB) are baked in at build time.
#                  A container that downloads a model on first use is a container whose first
#                  audit fails behind a firewall, and this project's whole retrieval path is
#                  otherwise offline.
#
# chroma_db (74 MB) is deliberately NOT copied. It is a build artefact of `finguard-store`, not
# source, and mounting it means re-indexing the corpus does not mean rebuilding the image:
#
#   docker run -p 8000:8000 \
#     -v "$PWD/chroma_db:/app/chroma_db" \
#     -e OPENAI_API_KEY=sk-... \
#     finguard
#
# The mount is deliberately NOT :ro. ChromaDB is SQLite underneath, and SQLite opens a journal
# even to read -- a read-only mount fails with "attempt to write a readonly database" and the
# health probe correctly refuses to serve. The container runs as uid 1000, so the host directory
# has to be writable by it.

# --- builder ----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependencies first, in their own layer: the lockfile changes far less often than src/ does, so
# an ordinary code change re-runs neither the resolve nor the download.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Swap the CUDA torch that the lockfile resolves to for the CPU build: ~200 MB against 517 MB
# plus roughly a gigabyte of nvidia-* CUDA runtime, none of which a CPU service can use.
#
# Three ordering rules, each learned from a failed build:
#   * it must come AFTER the last `uv sync`, or the sync restores the locked CUDA build and the
#     override is silently undone (the layer log showed nvidia-cusparselt going back in);
#   * torch must be UNINSTALLED first, because the CPU and CUDA wheels carry the same version
#     number -- `uv pip install` sees the requirement already satisfied and does nothing;
#   * the nvidia-* packages must go in the same step, not after. Removed once torch is already
#     importing them, the next import dies on `libcublasLt.so not found`.
#
# Not pinned in pyproject.toml because the cpu index is a deployment decision: developers on
# macOS already get a CPU build, and forcing the index there would break anyone with a GPU.
RUN . /app/.venv/bin/activate && \
    CUDA_PKGS="$(uv pip list --python /app/.venv | awk '/^nvidia-|^triton /{print $1}')" && \
    uv pip uninstall --python /app/.venv torch $CUDA_PKGS && \
    uv pip install --python /app/.venv \
        --index-url https://download.pytorch.org/whl/cpu torch && \
    /app/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Bake the models. Both go to well-known caches that the runner stage copies wholesale.
ENV HF_HOME=/opt/models/hf
RUN /app/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && \
    /app/.venv/bin/python -c "\
from flashrank import Ranker; Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/opt/models/flashrank')"

# --- runner -----------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# libgomp is torch's OpenMP runtime; the slim image omits it and the import fails without it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The service writes only to a temp dir for uploads, which it deletes after each run.
RUN useradd --create-home --uid 1000 finguard
WORKDIR /app

COPY --from=builder --chown=finguard:finguard /app/.venv /app/.venv
COPY --from=builder --chown=finguard:finguard /opt/models /opt/models
COPY --chown=finguard:finguard src ./src
COPY --chown=finguard:finguard pyproject.toml README.md ./

# A writable home for the two things that are written at runtime. /app is root-owned and the
# service is uid 1000, so both have to point somewhere else or the first flush is a PermissionError.
RUN mkdir -p /var/cache/finguard && chown finguard:finguard /var/cache/finguard

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models/hf \
    HF_HUB_OFFLINE=1 \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    EMBEDDING_CACHE_DIR=/var/cache/finguard/embeddings

USER finguard
EXPOSE 8000

# /health checks that the collection is queryable, not merely that the process is up -- a service
# whose vector store is missing would accept audits and then refuse every retrieval.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
