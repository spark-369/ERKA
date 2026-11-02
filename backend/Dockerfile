# =============================================================================
# Backend Core API image: FastAPI + LangChain + Celery + ChromaDB + transformers
#
# Multi-stage build:
#   1. builder  - compiles wheels (psycopg2, tokenizers, chroma deps...)
#   2. runtime  - slim image with only the installed virtualenv
#
# The same image runs the API (uvicorn/gunicorn) and the Celery worker, so both
# share one pinned dependency set.
# =============================================================================

# ------------------------------------------------------------------ builder ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain needed by psycopg2 / tokenizers / chroma-hnswlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        libpq-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU-only PyTorch first (smaller, and the default index pulls CUDA wheels).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# --------------------------------------------------------------- ml caches ---
# Pre-download the embedding model and the Qwen2.5-0.5B weights so the first
# request is not slowed down by a cold Hugging Face download. Set
# PREFETCH_MODELS=false to skip this (smaller build, slower first run).
ARG PREFETCH_MODELS=true
ARG LLM_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV HF_HOME=/opt/hf

RUN if [ "$PREFETCH_MODELS" = "true" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}', cache_folder='/opt/hf')" \
        && python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('${LLM_MODEL_ID}', cache_dir='/opt/hf'); \
AutoModelForCausalLM.from_pretrained('${LLM_MODEL_ID}', cache_dir='/opt/hf')" ; \
    else \
        echo "Skipping model prefetch"; mkdir -p /opt/hf ; \
    fi

# ------------------------------------------------------------------ runtime ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CHROMA_PERSIST_DIR=/data/chroma \
    UPLOAD_DIR=/data/uploads

# Runtime shared libraries only (libpq for psycopg2, libgomp for torch).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf

WORKDIR /app
COPY --chown=appuser:appuser . /app

# Writable runtime data (Chroma persistence, uploads, HF cache).
RUN mkdir -p /data/chroma /data/uploads /data/hf \
 && chown -R appuser:appuser /data /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# ``tini`` reaps zombies and forwards signals so Celery/uvicorn shut down cleanly.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: production-grade ASGI server. Override the command to run a Celery
# worker from the same image, e.g.:
#   celery -A app.workers.worker:celery_app worker --loglevel=info
#
# ``--timeout`` must cover a worst-case generation: the full 8192-token answer
# budget on CPU can take several minutes, and a shorter timeout kills the worker
# mid-stream (the client sees a truncated reply). It is also used as the SSE
# keep-alive budget, so it is generous by design.
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "900", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
