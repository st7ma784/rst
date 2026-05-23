# libgrd vs libgrdopt — equivalence & benchmark results

Harness: `grid_opt_compare.c` (links both libgrd.1 and libgrdopt.1).
Host: 14-core x86_64, GCC 13 -O2 -march=native, runtime varied via
`OMP_NUM_THREADS`. See sibling `AUDIT.md` (in the optimized library
directory) for the full bug-list and completion plan.

## TL;DR

After round-5 fixes, the optimized library is **bitwise-equivalent**
to the original on every measured operation, but is still **slower at
every size**. Single-threaded performance:

| Op            | N        | libgrd  | libgrdopt | speedup |
|---------------|---------:|--------:|----------:|--------:|
| Sort          |    1,000 | 0.07ms  |   0.07ms  |   0.93x |
| Sort          |   10,000 | 0.87ms  |   0.93ms  |   0.93x |
| Sort          |  100,000 | 12.4ms  |  16.7ms   |   0.74x |
| Sort          |  500,000 | 88.4ms  | 114.2ms   |   0.77x |
| LocateCell    |    1,000 | 10.2µs  |  17.8µs   |   0.57x |
| LocateCell    |   10,000 |  165µs  |   446µs   |   0.37x |
| LocateCell    |  100,000 | 5,076µs | 12,647µs  |   0.40x |

All 100 equivalence checks across all rows passed.

## What changed between round-4 and round-5

| Issue                                    | Before     | After      |
|------------------------------------------|------------|------------|
| Sort equivalence                         | 5/5 fail   | 0/5 fail   |
| Locate equivalence                       | 20/20 mismatch | 0/20 mismatch |
| Sort speedup @ N=500k (1 thread)         | 0.74x      | 0.77x      |
| Locate speedup @ N=100k (1 thread)       | 0.16x      | 0.40x      |
| Multi-thread sort @ N=100k               | segfault   | segfault (disabled, falls back to serial) |
| `grid_aligned_malloc/free` definitions   | external stub | in libgrdopt |

## Why still slower

Root causes haven't changed since the round-4 RESULTS — the new
round-5 work fixed correctness and one performance regression
(LocateCell OMP overhead) but the structural costs remain:

1. **`sizeof(GridGVecOpt) ≈ 2 × sizeof(GridGVec)`** (192 vs 96 bytes)
   due to heavy alignment + padding. Every memory-bound op pays 2×
   the bandwidth. This is the dominant cost.

2. **Indirect comparator dispatch** through `compare_grid_cells` +
   `global_sort_context`. Original uses an inlined `GridSortVec` in
   qsort's compare slot. The round-5 fast-path comparator catches
   the common `(STATION, INDEX, ASC)` case but still goes through a
   function pointer per compare.

3. **`GridLocateCellOpt` is the same linear scan algorithm** as
   `GridLocateCell`. The original "Opt" naming was aspirational.
   `GridDataOpt.spatial_index` field exists but is never populated.

## Round-5 fixes (committed in this push)

- `GridSortOpt` now defaults to `SORT_BY_STATION + SORT_BY_INDEX`
  (matches original `GridSort` semantics — was `LATITUDE + LONGITUDE`)
- Added `SORT_BY_INDEX` to the criteria enum + case in `get_sort_key`
- Added a fast-path direct-compare branch in `compare_grid_cells` for
  the common case
- Multi-threaded sort branch disabled (`if (0 && ...)`) with TODO —
  both attempted parallel impls (OMP-task recursive merge + tile +
  k-way merge) segfault for N ≥ 50k
- Removed OMP path from `GridLocateCellOpt` — measured 6x slower than
  scalar at OMP=1
- `grid_aligned_malloc` / `grid_aligned_free` now defined inside
  the library (no external stub needed)
- Fixed `parallel_merge` AVX copy that was truncating struct moves to
  32 bytes regardless of `sizeof(GridGVecOpt)`

## What this harness does NOT measure

- Aggregation ops (`GridAdd`/`GridIntegrate`) — different signatures
  in the optimized API, no apples-to-apples comparison.
- I/O (`GridFread`/`GridFwrite`) — the optimized `gridio_parallel.c`
  is excluded from the build (53 errors, wrong DataMap API).
- Merge (`GridMerge`) — `mergegrid_parallel.c` is excluded (Intel
  SVML intrinsics not available in GCC).
- AVX-512 seek (`gridseek_optimized.c`) — excluded (4 residual errors).

See `AUDIT.md` for the full plan to close these gaps (~5 days work).

## What would actually make this faster

Per AUDIT.md section 5 (Phase C):

- **Slim the GridGVecOpt struct** to ≤ 1.2× the original layout —
  removes the 2x memory bandwidth penalty. Highest impact, highest
  risk (SIMD code paths assume the current layout).
- **Inline the sort comparator** — use C11 `qsort_r` or compile-time
  specialised compares; closes the remaining single-thread gap.
- **Fix the multi-threaded sort** — either debug the merge OOB or
  replace with parallel sample sort. Target: 3–5x at N=500k on 14
  cores.
- **Build the spatial index** that `GridDataOpt.spatial_index`
  promises — turns LocateCell from O(N) into O(log N). Big asymptotic
  win for the hot path in plotting / merging code.
