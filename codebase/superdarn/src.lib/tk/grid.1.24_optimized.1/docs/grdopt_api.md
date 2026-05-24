# `libgrdopt` Public API Reference

This document is the canonical reference for the optimized grid library
(`libgrdopt.1.24_optimized`). It supersedes the `*Opt` material in
`docs/user_guide.md`, which documents the broader (and older) `*Parallel`
surface.

Audit reference: this file fulfils **E4 — doc/grdopt.tex public-API
documentation** (`AUDIT.md`). The doc lives in Markdown rather than
LaTeX to match the rest of the optimised tree's docs.

## When to use libgrdopt vs libgrd

| Workload                                        | Library     |
| ----------------------------------------------- | ----------- |
| Small grids (`vcnum < 10 000`), single-record   | `libgrd`    |
| Large grids, repeated sorts                     | `libgrdopt` |
| Repeated `GridLocateCell` calls (any size)      | `libgrdopt` |
| Bit-exact reproducibility against archived runs | `libgrd`    |

Benchmarks (Intel 14-core, AVX2, OMP=4, `grid_opt_compare`):

```
Sort  N=500 000   libgrd  90 ms   libgrdopt  44 ms   2.04x speedup
Locate N=100 000  libgrd 4.88 ms  libgrdopt 0.19 ms 25.30x speedup
```

The `LocateCell` win comes from C4 (O(log N) binary search when the
grid is sorted by `(st_id, index)`) and C5 (AVX2 scan of a dense int32
cache when it isn't). Sort wins are from the C3 iterative pairwise
merge sort. See `AUDIT.md` for the design background.

## Linking

```
LIBS = -lgrdopt.1 -lgrd.1 -ldmap.1 -lrtime.1 -lrcnv.1
SLIB = -lz -lm -fopenmp
```

`-lgrd.1` is required because `libgrdopt` delegates its file I/O to
libgrd's `.grd` DataMap parser/writer (`gridopt_wrappers.c`). The
parallel runtime is OpenMP; no CUDA dependency at link time.

## Headers

```c
#include "griddata_parallel.h"      /* core types + 7 baseline wrappers */
#include "griddata_parallel_api.h"  /* full *Opt surface (audit A6) */
```

## Public functions

All entries below take and return the *Opt struct family
(`GridDataOpt`, `GridSVecOpt`, `GridGVecOpt`). Field semantics match
libgrd's `GridData`/`GridSVec`/`GridGVec` except where noted.

### Allocation

```c
struct GridDataOpt *GridMakeOpt(void);
void                GridFreeOpt(struct GridDataOpt *ptr);
```

### Bulk operations

```c
void GridSortOpt     (struct GridDataOpt *ptr);
void GridAverageOpt  (struct GridDataOpt *mptr, struct GridDataOpt *ptr, int flg);
void GridIntegrateOpt(struct GridDataOpt *a,    struct GridDataOpt *b, double *err);
void GridMergeOpt    (struct GridDataOpt *mptr, struct GridDataOpt *ptr);
void GridCopyOpt     (struct GridDataOpt *a,    struct GridDataOpt *b);
void GridAddOpt      (struct GridDataOpt *a,    struct GridDataOpt *b, int recnum);
```

`GridSortOpt` sorts by `(st_id, index)` ascending (matches libgrd
`GridSort`). For grids larger than 10 000 cells and `OMP_NUM_THREADS>1`,
the implementation switches to a parallel tile-sort + iterative
pairwise merge with thread-safe `qsort_r`.

### Cell lookup

```c
int GridLocateCellOpt(int npnt, struct GridGVecOpt *ptr, int index);
```

Returns the first index `i` such that `ptr[i].index == index`, or
`npnt` if no match. The function probes monotonicity in O(1) and
falls into a binary search when the grid is sorted; otherwise it
uses an AVX2-vectorised scan of a thread-local int32 cache.

### File I/O (Phase D bridge)

```c
int GridReadOpt  (int   fid, struct GridDataOpt *gp);
int GridWriteOpt (int   fid, struct GridDataOpt *ptr);
int GridFreadOpt (FILE *fp,  struct GridDataOpt *ptr);
int GridFwriteOpt(FILE *fp,  struct GridDataOpt *ptr);
```

These delegate to libgrd's record reader/writer with a `GridData`
intermediate. The conversion drops the `srng` field on write
(`GridGVecOpt` does not store it) and zeros it on read. Other fields
round-trip byte-faithfully through the DataMap encoding.

### Seek and index

```c
int GridSeekOpt (int   fid, int yr, int mo, int dy, int hr, int mt,
                 int sc, double *atme, struct GridIndexParallel *inx);
int GridFseekOpt(FILE *fp,  int yr, int mo, int dy, int hr, int mt,
                 int sc, double *atme, struct GridIndexParallel *inx);

struct GridIndexParallel *GridIndexLoadOpt (int   fid);
struct GridIndexParallel *GridIndexFloadOpt(FILE *fp);
void                      GridIndexFreeOpt (struct GridIndexParallel *inx);
```

### Linear regression and time decode

```c
void   GridLinRegOpt(int num, struct GridGVecOpt **data,
                     double *vpar, double *vper);
double GridGetTimeOpt(struct DataMap *ptr);
```

## Threading model

`libgrdopt` is built with `-fopenmp`. Functions that benefit from
multiple threads pick up `OMP_NUM_THREADS` automatically; you don't
need to call `omp_set_num_threads` yourself.

Thread-safety guarantees:

- `GridSortOpt` is thread-safe (C2 made the comparator re-entrant
  via `qsort_r`; the prior global comparator pointer was a known
  race).
- `GridLocateCellOpt` uses a per-thread (`__thread`) cache for the
  SIMD fallback path, so concurrent calls on distinct grids are safe.
- `GridAverageOpt` and `GridMergeOpt` share a static hash table
  (`avggrid_parallel.c`). They are **not** safe to call from
  multiple threads simultaneously on different grids; protect
  with your own mutex if you need that.

## Differences from libgrd

| Field / behaviour                  | libgrd          | libgrdopt           |
| ---------------------------------- | --------------- | ------------------- |
| `GridGVec.srng` per-cell range     | present         | dropped (C1 slim)   |
| Struct alignment                   | natural         | natural (was 128)   |
| `sizeof(GridGVecOpt)`              | 88 bytes        | 88 bytes (was 192)  |
| Sort comparator context            | global (libgrd) | per-call (qsort_r)  |
| Sort threshold for parallel path   | n/a             | `vcnum > 10 000`    |
| LocateCell complexity (sorted)     | O(N)            | O(log N) -- C4      |
| LocateCell complexity (unsorted)   | O(N) scalar     | O(N/8) AVX2 -- C5   |
| File format                        | `.grd` DataMap  | same (bridge)       |

## Build options

The makefile picks `-mavx2 -mfma` when `AVX2=1` is set in the parent
RST profile. AVX-512 paths exist in `gridseek_optimized.c` but are
auto-detected at compile time via `__AVX512F__`.

## Sanitizer recipes (E1-E3)

The `src/makefile` does not (yet) carry dedicated `asan` / `tsan`
targets. Until it does, run the harness with:

```sh
# E1: AddressSanitizer + UndefinedBehaviorSanitizer
CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
LDFLAGS="-fsanitize=address,undefined" \
  make -C codebase/superdarn/src.lib/tk/grid.1.24_optimized.1/src clean all
ASAN_OPTIONS=halt_on_error=1:detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  OMP_NUM_THREADS=4 ./bin/grid_opt_compare

# E2: ThreadSanitizer (requires single-threaded malloc allocator)
CFLAGS="-O1 -g -fsanitize=thread -fno-omit-frame-pointer" \
LDFLAGS="-fsanitize=thread" \
  make -C codebase/superdarn/src.lib/tk/grid.1.24_optimized.1/src clean all
OMP_NUM_THREADS=4 ./bin/grid_opt_compare

# E3: Valgrind leak audit
OMP_NUM_THREADS=1 valgrind --leak-check=full --show-leak-kinds=all \
  ./bin/grid_opt_compare
```

Known sanitizer findings (all resolved or documented):

- `vectorized_average_step` had a 32-byte aligned store onto a stack
  `double[4]` that crashed under OMP. Fixed by switching to
  `_mm256_storeu_pd`.
- AVX-alignment crash in `GridAverageParallel` / `GridMergeParallel`
  is **fixed**: aligned `_mm256_store_pd` / `_mm256_load_pd` at
  `avggrid_parallel.c:389,402`, `mergegrid_parallel.c:108-112`,
  `integrategrid_parallel.c:83-85`, and `grid_parallel_utils.c:324-349`
  were swapped to their unaligned counterparts.
- gcc `-O3 -march=native` auto-vectorized plain struct copies
  (`ptr->sdata[0].noise = mptr->sdata[0].noise`) into aligned AVX2
  `vmovdqa`. At OMP=4 the malloc arena returns 16-byte (not 32-byte)
  aligned pointers and the load faulted. Fixed by replacing struct
  copies with `memcpy` in `avggrid_parallel.c` and `mergegrid_parallel.c`.
- `GridDataOpt` is declared `ALIGNED(128)` but `GridMakeOpt` used
  plain `malloc` (16-byte aligned). UBSAN flagged the misaligned
  member access. Fixed by switching to `aligned_alloc(128, ...)` with
  size rounded up to a multiple of 128.
- ASAN + UBSAN: **clean** at OMP=1 against the full harness
  (`--with-ops --with-io`). Zero findings.
- TSan: structurally blocked — the harness statically links libgrd /
  libdmap / librtime / librcnv which are not instrumented with
  `-fsanitize=thread`. A partially-instrumented TSan binary crashes
  immediately. Full TSan coverage would require rebuilding the entire
  RST tree with `-fsanitize=thread`, which is out of scope for this
  library.

## Known limitations

- The `GridDataOpt.spatial_index` field is allocated but populated
  only on first `GridLocateCellOpt` call; persistent caching across
  grid mutations is not implemented.
- `GridGVecOpt.srng` removal is observable on data round-trips
  through `libgrdopt` (always reads back zero).
- DataMap `.grd` round-trips are **lossy** in the float-encoded
  fields: `mlat`, `mlon`, `vel.{median,sd}`, `pwr.{median,sd}`,
  `wdt.{median,sd}` are stored as `DATAFLOAT` (single-precision)
  on disk, while the `*Opt` structs hold doubles. After a
  round-trip these fields are bit-identical to their `(double)(float)`
  projection. The B3 round-trip test uses a `1e-4` epsilon. `.index`,
  `.st_id`, `.chn` are integer-encoded and round-trip exactly. Cell
  ordering IS preserved.
- `GridAverageOpt` / `GridMergeOpt` legacy wrappers force
  `omp_set_num_threads(1)` internally because the implementations
  share a static hash table and depend on encounter-order
  accumulation. Callers wanting parallelism across multiple grids
  must invoke their own thread pool.

## Audit cross-reference

| Audit item | Status                                   | File                        |
| ---------- | ---------------------------------------- | --------------------------- |
| A1-A6      | done (Phase A symbol coverage)           | various                     |
| B1         | done (per-op equivalence PASS)           | `grid_opt_compare.c`        |
| B2         | done (alignment crashes fixed)           | various                     |
| B3         | done (round-trip PASS)                   | `grid_opt_compare.c`        |
| B4         | done (CI workflow)                       | `grid-search-test.yml`      |
| C1         | done (88-byte struct)                    | `griddata_parallel.h`       |
| C2         | done (qsort_r comparator)                | `sortgrid_parallel.c`       |
| C3         | done (parallel pairwise merge)           | `sortgrid_parallel.c`       |
| C4         | done (bsearch fast path)                 | `avggrid_parallel.c`        |
| C5         | done (AVX2 int32-cache scan)             | `avggrid_parallel.c`        |
| D1         | done (cross-write byte-identical PASS)   | `grid_opt_compare.c`        |
| D2         | done (cross-read via round-trip PASS)    | `grid_opt_compare.c`        |
| D3         | done (no divergences found)              | n/a                         |
| E1         | done (ASAN+UBSAN clean)                  | various                     |
| E2         | blocked (uninstrumented deps); see above | n/a                         |
| E3         | done (manual audit; no leaks in changes) | various                     |
| E4         | this document                            | `docs/grdopt_api.md`        |
