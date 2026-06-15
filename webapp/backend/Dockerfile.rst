# ── RST C-backed backend (multi-stage, libgrdopt-linked) ────────────────────
#
# Single image for the rancher-deployed rst-workbench. The FastAPI in
# webapp/backend subprocesses to three native binaries:
#   make_fit  — raw ACF → fitacf
#   make_grid — fitacf  → grid
#   map_grd   — grid    → map (linked against libgrdopt for the
#                              sort/locate speedups from libgrdopt
#                              Phase B/C round-8 closeout).
#
# Build context must be the repo root:
#   docker build -f webapp/backend/Dockerfile.rst -t siw-backend .
#
# Multi-stage: builder compiles only the dep chain the three binaries
# need (in correct order, no `|| true` masking), runtime is python:3.11-slim
# + libgomp + the produced /opt/rst/{bin,lib}. Estimated final image
# size: ~400 MB (vs ~1.5 GB for the previous build-tools-included image).

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rst
COPY codebase/      codebase/
COPY build/         build/
COPY .profile/      .profile/
COPY .profile.bash  .

ENV RSTPATH=/opt/rst SYSTEM=linux
SHELL ["/bin/bash", "-c"]

# Build the library dependency chain in dependency order. This mirrors
# .github/workflows/grid-search-test.yml round-7 and
# .github/workflows/fitacf-search-test.yml. No `|| true` — failures
# are fatal so a broken image never ships.
RUN source .profile.bash && set -e && \
    for d in \
        codebase/base/src.lib/task/rtypes.1.5 \
        codebase/base/src.lib/math/rmath.1.8 \
        codebase/general/src.lib/time.1.7 \
        codebase/base/src.lib/task/convert.1.11 \
        codebase/base/src.lib/task/option.1.7 \
        codebase/general/src.lib/dmap.1.25 \
        codebase/general/src.lib/rfile.1.9 \
        codebase/superdarn/src.lib/tk/radar.1.22 \
        codebase/superdarn/src.lib/tk/raw.1.22 \
        codebase/superdarn/src.lib/tk/scan.1.7 \
        codebase/superdarn/src.lib/tk/cfit.1.19 \
        codebase/superdarn/src.lib/tk/fit.1.35 \
        codebase/superdarn/src.lib/tk/elevation.1.0 \
        codebase/analysis/src.lib/mpfit/mpfit.1.5 \
        codebase/superdarn/src.lib/tk/fitacf.2.5 \
        codebase/superdarn/src.lib/tk/fitacf_v3.0 \
        codebase/superdarn/src.lib/tk/lmfit.1.0 \
        codebase/superdarn/src.lib/tk/fitacfex.1.3 \
        codebase/superdarn/src.lib/tk/fitacfex2.1.0 \
        codebase/superdarn/src.lib/tk/grid.1.24 \
        codebase/superdarn/src.lib/tk/grid.1.24_optimized.1 \
        ; do \
        if [ ! -d "${d}/src" ]; then echo "missing: ${d}/src"; exit 1; fi; \
        (cd "${d}/src" && make); \
    done

# Build the three binaries the FastAPI subprocesses to. map_grd's
# makefile is patched (in-tree, see map_grd.1.16/makefile) to link
# -lgrdopt.1 ahead of -lgrd.1 so the sort/locate fast path lights up.
RUN source .profile.bash && set -e && \
    for b in \
        codebase/superdarn/src.bin/tk/tool/make_fit \
        codebase/superdarn/src.bin/tk/tool/make_grid.2.0 \
        codebase/superdarn/src.bin/tk/tool/map_grd.1.16 \
        ; do \
        if [ ! -f "${b}/makefile" ]; then echo "missing: ${b}/makefile"; exit 1; fi; \
        (cd "${b}" && make); \
    done

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 zlib1g curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the compiled artifacts the runtime needs.
COPY --from=builder /opt/rst/bin /opt/rst/bin
COPY --from=builder /opt/rst/lib /opt/rst/lib

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
