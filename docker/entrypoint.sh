#!/usr/bin/env sh
set -eu

MODE="${APP_MODE:-api}"
HOST="${SCHOLARMIND_HOST:-0.0.0.0}"

# Ensure persistent volume directories exist
mkdir -p "${SCHOLARMIND_DATA_DIR:-/app/data}/store" "${HF_HOME:-/app/artifacts/hf_cache}"

if [ "$MODE" = "ui" ]; then
    PORT="${SCHOLARMIND_UI_PORT:-8501}"
    echo "[ScholarMind] Starting Streamlit UI on ${HOST}:${PORT} (API=${SCHOLARMIND_API_URL:-local})..."
    exec streamlit run src/scholarmind/ui/app.py \
        --server.address="${HOST}" \
        --server.port="${PORT}" \
        --server.headless=true \
        --browser.gatherUsageStats=false
else
    PORT="${SCHOLARMIND_PORT:-8000}"
    echo "[ScholarMind] Starting FastAPI Backend on ${HOST}:${PORT} (profile=${SCHOLARMIND_PROFILE:-production})..."
    exec uvicorn scholarmind.api.main:app \
        --host "${HOST}" \
        --port "${PORT}" \
        --workers 1
fi
