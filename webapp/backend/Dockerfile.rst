# ── RST reference backend ─────────────────────────────────────────────────────
# Compiles the RST C libraries (best-effort) then runs the FastAPI server
# with BACKEND_TYPE=rst, which delegates to the compiled binaries via subprocess.
#
# Build context must be the repo root:
#   docker build -f webapp/backend/Dockerfile.rst -t siw-backend-rst .
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran curl \
        libhdf5-dev libnetcdf-dev libpng-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── RST library build (best-effort; some targets may not compile) ─────────────
COPY codebase/ /opt/rst/codebase/
COPY build/    /opt/rst/build/

ENV RST_ROOT=/opt/rst \
    SYSTEM=linux

RUN mkdir -p /opt/rst/codebase/lib/${SYSTEM} /opt/rst/codebase/bin/${SYSTEM} \
    && find /opt/rst/codebase -maxdepth 6 -name makefile | sort | \
       xargs -I{} sh -c 'make -C "$(dirname {})" -f makefile 2>/dev/null || true'

ENV IPATH=/opt/rst/codebase/include \
    LIBPATH=/opt/rst/codebase/lib/${SYSTEM} \
    BINPATH=/opt/rst/codebase/bin/${SYSTEM} \
    PATH=/opt/rst/codebase/bin/${SYSTEM}:${PATH} \
    LD_LIBRARY_PATH=/opt/rst/codebase/lib/${SYSTEM}:${LD_LIBRARY_PATH}

# ── Python layer ──────────────────────────────────────────────────────────────
COPY webapp/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pythonv2/ /pythonv2/
COPY webapp/backend/ .

ENV PYTHONPATH=/pythonv2 \
    BACKEND_TYPE=rst \
    RST_BINPATH=/opt/rst/codebase/bin/${SYSTEM} \
    DATA_DIR=/data \
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
