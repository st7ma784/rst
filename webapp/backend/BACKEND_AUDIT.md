# Helm backend image audit — lightweight RST + libgrdopt

Date: 2026-05-25. Companion to `codebase/superdarn/src.lib/tk/grid.1.24_optimized.1/AUDIT.md` (Round-8).

## What the FastAPI backend actually needs from RST

`webapp/backend/services/processor_rst.py` invokes exactly **three** RST binaries via subprocess:

| Constant         | Binary       | Purpose                          |
|------------------|--------------|----------------------------------|
| `_BIN_MAKE_FIT`  | `make_fit`   | raw ACF → fitacf                 |
| `_BIN_MAKE_GRID` | `make_grid`  | fitacf → gridded data            |
| `_BIN_MAP_GRD`   | `map_grd`    | grid → map / multi-station combine |

Everything else in RST is unused at runtime — `Dockerfile.rst` currently
builds the entire codebase (~200+ targets), then ships build tools in
the final image.

## Transitive library dependencies of the three binaries

From the per-binary makefiles:

- **`make_fit`** (`src.bin/tk/tool/make_fit/makefile`):
  `oldraw, oldfit, cfit, rscan, radar, fitacf.1, fitacf.3.0, fitacfex2,
  fitacfex, lmfit, elevation, raw, fit, dmap, opt, rtime, rcnv, rmath, mpfit`
- **`make_grid`** (`src.bin/tk/tool/make_grid.2.0/makefile`):
  `oldfit, gtabw, oldgtabw, gtable, rpos, filter, cfit, fit, rscan, radar,
  dmap, opt, rtime, rcnv, aacgm, igrf, aacgm_v2, igrf_v2, astalg, channel`
- **`map_grd`** (`src.bin/tk/tool/map_grd.1.16/makefile`):
  `cnvmap, oldcnvmap, oldgrd, grd, dmap, radar, rfile, rtime, mlt, aacgm,
  mlt_v2, aacgm_v2, igrf_v2, astalg, opt, rcnv`

Only **`map_grd`** uses `libgrd`. **That is the only binary whose
performance is touched by libgrdopt.** `make_grid` writes grid output but
does not link the grid library.

## Bugs in the current `Dockerfile.rst`

1. **Build order is random with all failures swallowed.**
   `find ... -name makefile | sort | xargs -I{} sh -c '... || true'`
   runs makefiles in find's path order with `|| true` on each. This is
   the exact anti-pattern we just fixed in `.github/workflows/grid-search-test.yml`
   (round-7). `grid.1.24_optimized.1` attempts to compile before
   `rtypes.1.5` installs `rtypes.h`; the failure is silently masked.
   Whatever ships works by accident.

2. **No build/runtime separation.**
   `build-essential` and `gfortran` (and their dev headers) live in the
   final image. Easy ~600 MB to shed via multi-stage build.

3. **No libgrdopt in the resulting image, even if RST builds.**
   `map_grd`'s makefile still links `-lgrd`, not `-lgrdopt`. The optimized
   library would have to be either substituted (`-lgrdopt -lgrd` for the
   I/O symbols libgrdopt delegates) or made a drop-in replacement of
   `libgrd.so.1` via symlink. Currently neither happens.

4. **Unrelated heavy build deps.**
   `libhdf5-dev libnetcdf-dev libpng-dev` are pulled in. None of the
   three binaries we actually need link against these — they're
   inherited from the upstream RST install script. Drop them.

5. **`Dockerfile.k8s` is unrelated:** it's a pythonv2-only image (no C
   build, no `BACKEND_TYPE=rst`). Not relevant to this audit.

## Recommendation — `Dockerfile.rst-lite` (multi-stage)

Two-stage build. Builder compiles only the libraries the three binaries
need, in dependency order. Runtime image contains only Python + `libgomp` +
`libz` + the built `bin/` and `lib/`.

Sketch:

```dockerfile
# ── Stage 1: builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/rst
COPY codebase/ codebase/
COPY build/    build/
COPY .profile/ .profile/
COPY .profile.bash .
ENV RSTPATH=/opt/rst SYSTEM=linux
SHELL ["/bin/bash", "-c"]

# Build libraries in dependency order. Same pattern as
# .github/workflows/grid-search-test.yml round-7 — no || true.
RUN source .profile.bash && set -e && \
    for d in \
        codebase/base/src.lib/task/rtypes.1.5 \
        codebase/base/src.lib/math/rmath.1.8 \
        codebase/general/src.lib/time.1.7 \
        codebase/base/src.lib/task/convert.1.11 \
        codebase/base/src.lib/task/option.1.7 \
        codebase/general/src.lib/dmap.1.25 \
        codebase/general/src.lib/rfile.1.9 \
        # ... plus aacgm, igrf, mlt, astalg, channel, radar, etc.
        # ... plus all libraries listed in the three makefiles above.
        codebase/superdarn/src.lib/tk/grid.1.24 \
        codebase/superdarn/src.lib/tk/grid.1.24_optimized.1 \
        ; do \
        (cd "${d}/src" && make); \
    done && \
    for b in \
        codebase/superdarn/src.bin/tk/tool/make_fit \
        codebase/superdarn/src.bin/tk/tool/make_grid.2.0 \
        codebase/superdarn/src.bin/tk/tool/map_grd.1.16 \
        ; do \
        (cd "${b}/src" && make); \
    done

# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 zlib1g curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/rst/bin /opt/rst/bin
COPY --from=builder /opt/rst/lib /opt/rst/lib
ENV LD_LIBRARY_PATH=/opt/rst/lib RST_BINPATH=/opt/rst/bin

WORKDIR /app
COPY webapp/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pythonv2/ /pythonv2/
COPY webapp/backend/ .
ENV PYTHONPATH=/pythonv2 BACKEND_TYPE=rst DATA_DIR=/data \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN adduser --disabled-password --gecos "" --uid 1000 appuser \
    && mkdir -p /data && chown appuser:appuser /data
USER appuser
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
```

Estimated impact:
- Image size: ~1.5 GB → ~400 MB (rough estimate; depends on how many
  libs the full list above ends up needing).
- Build is **deterministic** — fails on the first real error instead of
  silently shipping a half-built image.
- Adding `make.code map_grd.1.16` after the per-lib loop with a patched
  `LIBS=-lgrdopt.1 -lgrd.1 ...` gets `map_grd` linked against the
  optimized library.

## Getting `map_grd` to actually use `libgrdopt`

The optimized library is **API-compatible** with libgrd for the
operations it implements (Sort/Average/Integrate/Merge/Copy/Add/Locate
all pass equivalence — see Round-8 in `AUDIT.md`). It does **not**
re-implement file I/O (`GridFseek`, `GridIndexFload`, `GridFread`); those
remain in libgrd.

Two options:

1. **Link both.** Patch `map_grd.1.16/makefile`:
   ```
   LIBS= -lcnvmap.1 -loldcnvmap.1 -loldgrd.1 -lgrdopt.1 -lgrd.1 -ldmap.1 \
         ...
   ```
   `libgrdopt`'s sort/locate/etc. wins resolution because it appears
   first; libgrd supplies the missing I/O symbols. **No source change.**

2. **Drop-in replacement via `libgrd.so` symlink in the image:**
   ```
   ln -sf libgrdopt.1.so /opt/rst/lib/libgrd.1.so
   ```
   Cleaner runtime, but requires libgrdopt to export *every* symbol
   libgrd does. It currently doesn't (no `GridFread` etc.), so this
   would need finishing the I/O surface — out of Phase B/C scope.

Recommend **option 1** for now. Two-line change to the `map_grd`
makefile, runs through the existing CI on the next push.

## What this audit does not do

- Does **not** rewrite `Dockerfile.rst` — that's a follow-up PR with
  Docker layer cache verification + image size measurement.
- Does **not** patch `map_grd.1.16/makefile` — same reason. Should
  ship behind a feature flag or env-driven build arg so we can A/B
  the optimized vs reference grid library in production.
- Does **not** fix the unrelated `nginx: host not found in upstream
  "backend"` error in the frontend pod — that's task #12, separate
  concern (nginx envsubst template + Helm Service name mismatch).

## Action items (for follow-up PRs)

1. Write `webapp/backend/Dockerfile.rst-lite` per the sketch above;
   verify build succeeds, image size drops, all three binaries run.
2. Patch `codebase/superdarn/src.bin/tk/tool/map_grd.1.16/makefile`
   to add `-lgrdopt.1` ahead of `-lgrd.1` in `LIBS`.
3. Add a CI job that builds `Dockerfile.rst-lite` on every push to
   `webapp/backend/**` or any RST library tree we depend on.
4. Update `helm/superdarn-workbench/values.yaml` to point
   `backendType: rst` at the new lite image once verified.
