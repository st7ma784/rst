# ── RST C-backed backend ─────────────────────────────────────────────────────
#
# Builds on top of superdarn_rstbase:latest (already published by
# docker-build-push.yml) to avoid re-compiling the full RST library chain.
# Only the Python app layer is added here.
#
# Build context must be the repo root:
#   docker build -f webapp/backend/Dockerfile.rst -t siw-backend .

# ── Stage 1: extract compiled RST artifacts ──────────────────────────────────
FROM st7ma784/superdarn_rstbase:latest AS rst-base

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 zlib1g curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled RST binaries and libraries from the base image.
COPY --from=rst-base /opt/rst/codebase/bin/linux/ /opt/rst/bin/
COPY --from=rst-base /opt/rst/codebase/lib/linux/ /opt/rst/lib/

ENV RST_BINPATH=/opt/rst/bin \
    LD_LIBRARY_PATH=/opt/rst/lib

WORKDIR /app
COPY webapp/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY webapp/backend/ .

ENV DATA_DIR=/data \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN adduser --disabled-password --gecos "" --uid 1000 appuser \
    && mkdir -p /data && chown appuser:appuser /data
USER appuser

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
