# ScholarMind AI — CPU-Optimized Multi-Role Production Dockerfile (FastAPI + Streamlit)
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCHOLARMIND_ENV=production \
    SCHOLARMIND_PROFILE=production \
    SCHOLARMIND_DATA_DIR=/app/data \
    SCHOLARMIND_ARTIFACTS_DIR=/app/artifacts \
    HF_HOME=/app/artifacts/hf_cache \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Minimal OS runtime libraries (libgomp1 for PyTorch/FAISS CPU, curl for container healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch & Python dependencies first for optimal Docker layer caching
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source, configs, saved lightweight benchmark JSONs & plots
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY artifacts/experiments ./artifacts/experiments
COPY data/reports ./data/reports
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Install scholarmind package without re-downloading dependencies
RUN pip install --no-deps --no-build-isolation . \
    && sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && useradd --create-home --shell /bin/sh appuser \
    && mkdir -p /app/data/store /app/artifacts/hf_cache /app/artifacts/checkpoints /tmp/matplotlib \
    && chown -R appuser:appuser /app /tmp/matplotlib

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD sh -c 'if [ "${APP_MODE:-api}" = "ui" ]; then curl -fsS http://127.0.0.1:${SCHOLARMIND_UI_PORT:-8501}/_stcore/health; else curl -fsS http://127.0.0.1:${SCHOLARMIND_PORT:-8000}/health; fi'

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
