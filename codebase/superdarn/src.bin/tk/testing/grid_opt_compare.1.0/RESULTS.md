# libgrd vs libgrdopt — equivalence & benchmark results

Harness: `/tmp/grid_bench/grid_bench.c` (links both libgrd.1 and libgrdopt.1).
Host: 14-core CPU, GCC -O2, runtime varied via `OMP_NUM_THREADS`.

## TL;DR

The "optimized" library is **slower than the original on every operation
measured, at every input size**, and produces **non-equivalent results**
for sorting. The performance regression is reproducible and structural.

## Sort

| N       | original (ms) | libgrdopt (ms) | speedup | equivalent? |
|---------|--------------:|---------------:|--------:|:-----------:|
| 1,000   |          0.07 |           0.07 |   0.95x | ✗           |
| 10,000  |          0.87 |           0.97 |   0.91x | ✗           |
| 100,000 |         12.82 |          17.43 |   0.74x | ✗           |
| 500,000 |         90.11 |         121.94 |   0.74x | ✗           |

OMP_NUM_THREADS=1 and =14 produce identical timings, confirming the
parallel path is never taken.

## LocateCell (linear search, 100 probes / iter)

| N       | original (µs) | libgrdopt (µs) | speedup | mismatches |
|---------|--------------:|---------------:|--------:|-----------:|
| 1,000   |         10.23 |          17.26 |   0.59x | 20/20      |
| 10,000  |        159.16 |         459.37 |   0.35x | 20/20      |
| 100,000 |       5036.02 |       13378.10 |   0.38x | 20/20      |

## Why the regression

1. **Different sort key**. `GridSort` sorts by `(st_id, index)`. `GridSortOpt`
   wraps `GridSortParallel(grid, NULL)` which calls
   `GridSortParallelEx(SORT_BY_LATITUDE, SORT_BY_LONGITUDE)`. The "optimized"
   API is not semantically equivalent to the original.

2. **Parallel branch is dead**. `GridSortParallelEx`:
       int num_threads = config ? config->num_threads : 1;
       if (grid->vcnum > 10000 && num_threads > 1) { /* parallel sort */ }
   The wrapper `GridSortOpt` passes `config=NULL`, so `num_threads=1` and
   the parallel-merge-sort branch is never reached. The "optimized" sort
   is always a single-threaded qsort.

3. **Larger struct, worse cache density**. `sizeof(GridGVec)` is ~96 bytes;
   `sizeof(GridGVecOpt)` is ~192 bytes (ALIGNED(128), nested aligned
   sub-structs, padding fields). Sorting twice the bytes, walking
   twice the bytes per probe — explains the 0.74x on sort and 0.35-0.59x
   on locate.

4. **Indirect comparator**. Original qsort calls `GridSortVec` (direct
   field compare). Optimized qsort calls `parallel_sort_compare` which
   reads `global_sort_context`, switches on criteria enum, computes a
   double sort key, then compares. Per-comparison overhead is multiples
   higher.

5. **LocateCell algorithm is unchanged**. `GridLocateCellOpt` is
   character-for-character the same linear search as `GridLocateCell` —
   the only delta is the larger struct it walks. There is no "SIMD"
   optimization despite the file name.

## What would actually help

- Pass a non-NULL `GridProcessingConfig` with `num_threads > 1` and run
  inputs >= 10k records to engage the parallel merge sort branch.
- Add a `GridSortByStation` entry in the criteria enum and have `GridSortOpt`
  use it, so the comparison is apples-to-apples.
- Reduce GridGVecOpt alignment/padding (currently 2x the original) or
  add a slim representation for sort/locate hot paths.
- Replace `GridLocateCellOpt`'s linear scan with the spatial index
  the optimized GridData already provisions (`spatial_index` field is
  defined but never populated).

## What this doesn't measure

- Aggregation operations (`GridAddParallel`, `GridIntegrateParallel`)
  could not be compared because they have different signatures than
  the original (`GridAdd(a, b, recnum)` vs `GridAddParallel(target,
  source, tolerance)`).
- I/O (`GridFread`/`GridFwrite`) — the optimized `gridio_parallel.c`
  is excluded from the build (didn't compile; wrong DataMap API).
- Merge (`GridMerge`) — optimized `mergegrid_parallel.c` is excluded
  (depends on Intel SVML intrinsics not available in GCC).
