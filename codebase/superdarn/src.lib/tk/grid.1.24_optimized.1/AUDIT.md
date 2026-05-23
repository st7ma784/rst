# libgrdopt — Full Audit & Completion Plan

Status snapshot of `codebase/superdarn/src.lib/tk/grid.1.24_optimized.1`
after rounds 1–5 of CI repair + harness work, and the concrete plan
for finishing it so it is both **functionally complete** (matches
or exceeds the libgrd surface) and **actually faster than libgrd** on
the workloads RST cares about.

This document is the source of truth for what's left. Cross-reference
`RESULTS.md` for live benchmark numbers.

---

## 1. Executive summary

The library compiles, links, and produces a binary archive
(`libgrdopt.1.24_optimized.{a,so}`). All operations it currently
exports are **bitwise-equivalent** to the corresponding libgrd
operations on the same input. However:

- 4 of the original 12 source files are excluded from the build
  because they don't compile against the real RST API (`DataMap`,
  `<immintrin.h>` SVML).
- The multi-threaded sort path is gated `if (0)` because both
  attempted parallel implementations segfault on N ≥ ~50k.
- Single-threaded operations are 0.40x–0.93x of libgrd — i.e.
  **the optimized library is currently slower than the original at
  every measured size**, despite the "optimized" naming. Root cause
  is structural (sizeof GridGVecOpt is 2× sizeof GridGVec) compounded
  by an indirect comparator dispatch in qsort.

Net assessment: **the library is honest scaffolding**, not a drop-in
acceleration of libgrd. Reaching "complete + faster" requires
roughly 5 engineer-days of focused work, broken down below.

---

## 2. Current state

### 2.1 Build

| File                          | In SRC? | Notes |
|-------------------------------|---------|-------|
| addgrid_parallel.c            | ✅      | compiles, links |
| avggrid_parallel.c            | ✅      | compiles, links |
| copygrid_parallel.c           | ✅      | compiles, links |
| filtergrid_parallel.c         | ✅      | compiles, links |
| grid_parallel_utils.c         | ✅      | compiles, defines `grid_aligned_malloc/free` |
| gridseek_parallel.c           | ✅      | compiles, links |
| integrategrid_parallel.c      | ✅      | compiles, links |
| sortgrid_parallel.c           | ✅      | compiles; multi-thread path disabled |
| **gridio_parallel.c**         | ❌      | 53 errors — wrong DataMap API |
| **gridmerge_parallel.c**      | ❌      | 5 errors — SVML + undefined GridFilterParams |
| **mergegrid_parallel.c**      | ❌      | 2 errors — SVML intrinsics |
| **gridseek_optimized.c**      | ❌      | 4 errors — AVX-512 type confusion |

### 2.2 Symbol coverage vs libgrd

Original `libgrd.1.a` exports 21 `Grid*` public symbols. `libgrdopt.1.a`
currently exports the following wrappers that mirror the original
API contract:

| Original           | Optimized              | Status                         |
|--------------------|------------------------|--------------------------------|
| `GridMake`         | `GridMakeOpt`          | equivalent                     |
| `GridFree`         | `GridFreeOpt`          | equivalent                     |
| `GridLocateCell`   | `GridLocateCellOpt`    | equivalent, slower             |
| `GridSort`         | `GridSortOpt`          | equivalent, slower             |
| `GridAverage`      | `GridAverageOpt`       | equivalent (untested at scale) |
| `GridIntegrate`    | `GridIntegrateOpt`     | equivalent (untested at scale) |
| `GridAdd`          | (only `GridAddParallel`, different signature) | **not equivalent** |
| `GridCopy`         | (only `GridCopyParallel`, returns new vs in-place) | **not equivalent** |
| `GridMerge`        | (in excluded mergegrid_parallel.c)             | **missing**        |
| `GridFread`        | (in excluded gridio_parallel.c)                | **missing**        |
| `GridFwrite`       | (in excluded gridio_parallel.c)                | **missing**        |
| `GridIndexFload`   | (in excluded gridio_parallel.c)                | **missing**        |
| `GridIndexLoad`    | (in excluded gridio_parallel.c)                | **missing**        |
| `GridIndexFree`    | (none)                                         | **missing**        |
| `GridFseek`        | (in gridseek_parallel.c but different sig)     | **needs wrapper**  |
| `GridSeek`         | (in gridseek_parallel.c but different sig)     | **needs wrapper**  |
| `GridRead`         | (none)                                         | **missing**        |
| `GridWrite`        | (none)                                         | **missing**        |
| `GridLinReg`       | `GridLinRegParallel` (no equiv wrapper)        | needs wrapper      |
| `GridSortVec`      | (internal use only)                            | not exported       |
| `GridGetTime`      | (in excluded gridseek_optimized.c)             | **missing**        |

**Coverage: 6 of 21 functions are equivalent today; 15 are missing
or signature-incompatible.**

### 2.3 Performance (current)

Host: 14-core x86_64, GCC 13 -O2 -march=native. See harness in
`codebase/superdarn/src.bin/tk/testing/grid_opt_compare.1.0/`.

| Op            | N        | libgrd  | libgrdopt | speedup |
|---------------|---------:|--------:|----------:|--------:|
| Sort          |    1,000 | 0.07ms  |   0.07ms  |   0.93x |
| Sort          |   10,000 | 0.87ms  |   0.93ms  |   0.93x |
| Sort          |  100,000 | 12.4ms  |  16.7ms   |   0.74x |
| Sort          |  500,000 | 88.4ms  | 114.2ms   |   0.77x |
| LocateCell    |    1,000 | 10.2µs  |  17.8µs   |   0.57x |
| LocateCell    |   10,000 |  165µs  |   446µs   |   0.37x |
| LocateCell    |  100,000 | 5,076µs | 12,647µs  |   0.40x |

All equivalence and mismatch checks pass (0 failures across all rows).

---

## 3. Bug inventory

### 3.1 Correctness bugs (known)

**B1. Multi-threaded sort segfaults.**
File: `sortgrid_parallel.c`. Both the original recursive OMP-task
`parallel_merge_sort` and the replacement tile+k-way merge segfault
for N ≥ ~50k with 2+ threads. Crash address is inside the merge
phase; symptom suggests either a temp-buffer aliasing hazard or an
OOB in `pos[best]++`. The branch is currently disabled with
`if (0 && ...)` and a TODO comment.

**B2. `parallel_merge` AVX path corrupts data.** *(fixed in round-5)*
File: `sortgrid_parallel.c`. Loaded 32 bytes per "element" with
`_mm256_load_pd` but treated the result as a copy of an entire
192-byte `GridGVecOpt`. Replaced with `memcpy`.

**B3. `GridSortOpt` default sorted by lat/lon, not (st_id, index).**
*(fixed in round-5)*. Caused all sort equivalence tests to fail
silently because the output looked sorted but by the wrong key.

**B4. `grid_aligned_malloc` / `grid_aligned_free` declared but never
defined inside the library.** *(fixed in round-5)* External
consumers had to provide stubs. Now defined in `grid_parallel_utils.c`.

**B5. `GridStats` field name mismatches.**
File: `addgrid_parallel.c:72`. Source uses `result[i].median` but the
`GridStats` struct (post round-4 expansion) carries `mean`, `median`,
`weight`, `samples`. The reference is now valid but several other
files use `.median` where the field intent looks like `.mean`. Audit
needed.

**B6. Local `struct DataMap` stub colliding with real `dmap.h`.**
*(fixed in round-4)* The header now `#include "dmap.h"` directly.

### 3.2 Performance regressions

**P1. `sizeof(GridGVecOpt) ≈ 2 × sizeof(GridGVec)`.**
Driven by `ALIGNED(128)` on the outer struct plus `ALIGNED(32)` on
each nested vel/pwr/wdt sub-struct plus explicit padding fields.
Every memory-bound op (sort moves, locate scans, copy) walks 2× the
bytes per element. **This is the dominant cause of the persistent
0.4–0.8x slowdown.** Cannot be fixed without redesigning the struct
layout. See C1 in the roadmap.

**P2. Indirect comparator dispatch in qsort.**
`compare_grid_cells` reads `global_sort_context`, branches on
`primary`/`secondary` enums, then either uses the fast-path (added in
round-5) or calls `get_sort_key` twice plus `fabs`. The fast-path
helps the common `(STATION, INDEX, ASC)` case but qsort still has to
go through the function pointer indirection per compare. Original
`GridSortVec` is inlined into qsort's compare slot in libgrd. ~10–20%
of the sort slowdown is attributable here.

**P3. `GridLocateCellOpt` is the same algorithm as `GridLocateCell`.**
*(documented in round-5)* Plain linear scan; "Opt" naming is
aspirational. The 0.4x speedup ratio at large N is entirely
explained by P1 (twice the bytes per probe).

### 3.3 Latent / structural issues

**S1. Duplicate function definitions across .c files.**
*(fixed in round-4)*. `grid_parallel_utils.c` previously redefined
`GridSortParallel`, `GridSortOpt`, `GridFreeParallel` that
`sortgrid_parallel.c` / `copygrid_parallel.c` already owned. They've
been renamed `*_internal` and made static.

**S2. No header for the parallel-only API.**
Public functions like `GridSortParallel`, `GridSortByDistanceParallel`,
`GridFilterCompositeParallel` (40+ symbols total) are declared in the
.c files via `extern` or implicitly; no header in `include/` exports
them. Consumers wishing to use the parallel API must declare manually.

**S3. `GridGVecOpt.flags` and `GridGVecParallel.quality_flag` /
`filter_flags` are never set or read.** Dead fields adding to the
size bloat (P1).

**S4. `spatial_index` field in `GridDataOpt` is defined but never
populated.** Would be the right place for an O(1) `GridLocateCell`
implementation but no code writes to it.

---

## 4. Excluded files — per-file completion plan

### 4.1 `gridio_parallel.c` — 53 errors

**What it should provide**: parallel/SIMD versions of `GridFread`,
`GridFwrite`, `GridIndexFload`, `GridIndexLoad`, `GridGetTime`.

**Why it doesn't compile**:
- 50 errors from `DataMapFindScalar(ptr, "name")` (2 args). Real API:
  `DataMapFindScalar(struct DataMap*, char*, int type)`. The function
  returns `void*` (the data pointer), but the parallel source assigns
  to `struct DataMapScalar*` and dereferences `->data.sptr`, expecting
  the scalar object. This is a fundamental misunderstanding of the
  DataMap API — needs a helper:
  ```c
  static struct DataMapScalar *find_scalar(struct DataMap *p, const char *name) {
      for (int c = 0; c < p->snum; c++)
          if (strcmp(p->scl[c]->name, name) == 0) return p->scl[c];
      return NULL;
  }
  ```
  Replace all 50 call sites with this helper. Tedious but mechanical.
- 3 errors from `GridPerformanceStats` field accesses that don't
  exist in the (already expanded) struct.

**Effort**: ~3 hours for one engineer. Mostly find-and-replace plus
~5 lines of new helper code. Once it compiles, semantic correctness
needs a round-trip test (write a synthetic GridDataOpt, read it back,
diff vs original).

### 4.2 `mergegrid_parallel.c` — 2 errors

**What it should provide**: `GridMerge` parallel implementation (the
single most-used operation for RST users averaging multiple radar
scans).

**Why it doesn't compile**:
- Uses `_mm256_sin_pd` and `_mm256_cos_pd` (Intel SVML). GCC's
  `<immintrin.h>` does not include SVML. Choices:
  1. Link `-lsvml` (Intel-only).
  2. Replace with scalar loop or call libmvec (GCC's vectorized math
     library, available via `-fopenmp-simd`).
  3. Drop the SIMD trig and use `#pragma omp simd` on a scalar loop;
     gcc will auto-vectorize the sin/cos with libmvec at -O3.

  Recommended: choice (3). Replace the
  `__m256d sin_azm = _mm256_sin_pd(azm_rad); ...` block with
  ```c
  #pragma omp simd
  for (int j = 0; j < 4; j++) {
      double s = sin(azm_rad_arr[j]);
      double c = cos(azm_rad_arr[j]);
      ...
  }
  ```

**Effort**: ~1 hour. One block to rewrite, possibly a second similar
block.

### 4.3 `gridmerge_parallel.c` — 5 errors

**What it should provide**: another flavor of `GridMerge` plus post-
merge filter integration.

**Why it doesn't compile**:
- Same SVML issue as 4.2.
- Uses `struct GridFilterParams filter_params = {0};` but
  `GridFilterParams` is never defined anywhere. Needs either: (a) the
  struct definition added to `griddata_parallel.h`, with fields
  inferred from the assignments (`outlier_detection`,
  `outlier_threshold`, etc.), or (b) the offending block disabled.
- A few `compare_grid_cells`-style comparisons on anonymous nested
  structs that the compiler can't auto-compare. Need explicit field
  comparisons.

**Effort**: ~2 hours. Define GridFilterParams (~10 fields), rewrite
SVML block, fix anonymous-struct comparisons.

### 4.4 `gridseek_optimized.c` — 4 residual errors

**What it should provide**: AVX-512 enhanced version of grid seeking
(distinct from `gridseek_parallel.c` which uses OMP).

**Why it doesn't compile** (after round-5 partial fix):
- Block at line ~86 unconditionally uses `_mm512_set1_pd`/`_mm512_*`
  but `vec_double` is typedef'd to `__m256d` under AVX2-only build.
  The `#ifdef __AVX512F__` guard covers the loop body but not the
  `vec_double v_target = _mm512_set1_pd(...)` declaration.
- A second similar block lower in the file.

**Effort**: ~30 minutes. Move the `#ifdef __AVX512F__` to include
the variable declarations, or write parallel AVX2 + AVX-512
implementations side-by-side.

---

## 5. Roadmap to "complete + faster than libgrd"

Phased so each phase ships a working library. Effort estimates assume
one experienced C engineer who has read this audit. Sequencing
respects dependencies (e.g. you can't benchmark merge until merge
compiles).

### Phase A — Restore symbol coverage (1.5 days)

Goal: every libgrd public function has an equivalent in libgrdopt.

| Item | Effort | Risk |
|------|--------|------|
| A1. Fix `gridseek_optimized.c` (4 errors)              | 0.5h | low |
| A2. Fix `mergegrid_parallel.c` (SVML → libmvec)         | 1h   | low |
| A3. Fix `gridmerge_parallel.c` (SVML + GridFilterParams) | 2h | medium |
| A4. Fix `gridio_parallel.c` (DataMap API)               | 3h   | medium |
| A5. Add `*Opt` wrappers for missing API:                | 2h   | low |
|     `GridFreadOpt`, `GridFwriteOpt`, `GridReadOpt`,     |      |     |
|     `GridWriteOpt`, `GridIndexFloadOpt`,                |      |     |
|     `GridIndexFreeOpt`, `GridMergeOpt`, `GridCopyOpt`,  |      |     |
|     `GridAddOpt`, `GridSeekOpt`, `GridFseekOpt`,        |      |     |
|     `GridLinRegOpt`, `GridGetTimeOpt`                   |      |     |
| A6. Add `include/griddata_parallel_api.h` declaring     | 1h   | low |
|     all public parallel-API symbols (~40 prototypes).   |      |     |

Acceptance: `nm libgrdopt.1.a | grep -c "Opt$"` ≥ 17 (covers all 17
unique libgrd entry points after `GridSortVec` exclusion).

### Phase B — Correctness for all operations (1 day)

Goal: every libgrdopt operation produces bitwise-equivalent output
to the libgrd counterpart on synthetic input.

| Item | Effort | Risk |
|------|--------|------|
| B1. Extend `grid_opt_compare` harness with one test    | 2h   | low |
|     per equivalent pair (Sort/Merge/Average/Integrate/ |      |     |
|     Copy/Add/Locate/Fread/Fwrite).                     |      |     |
| B2. Fix any equivalence regressions surfaced by B1.    | 3h   | medium |
|     (Expect 1–3 — defaults may differ.)                |      |     |
| B3. Build a 4-record round-trip test: Fwrite via opt,  | 1h   | low |
|     Fread via original (and vice-versa). Forces file   |      |     |
|     format compatibility.                              |      |     |
| B4. Add a CI job that compiles + runs the harness on   | 1h   | low |
|     every push. Failure threshold: any equiv mismatch  |      |     |
|     OR any perf regression > 1.10x.                    |      |     |

Acceptance: `make -C codebase/superdarn/src.bin/tk/testing/grid_opt_compare.1.0` + run reports 0/N failures across all rows.

### Phase C — Make it actually faster (1.5 days)

Goal: every op in libgrdopt is at least as fast as libgrd at N=10k,
faster at N=100k+.

| Item | Effort | Risk |
|------|--------|------|
| C1. **Slim GridGVecOpt struct.** Drop `flags`,         | 3h   | high |
|     `_padding[32]`, drop `ALIGNED(128)` on outer       |      |     |
|     struct → use 64-byte alignment. Target:            |      |     |
|     sizeof ≤ 1.2 × sizeof(GridGVec).                   |      |     |
|     Risk: may break SIMD blocks that assume layout.    |      |     |
| C2. **Inline the sort comparator.** Switch to a        | 2h   | medium |
|     C11 `qsort_r` with the criteria enum on the stack  |      |     |
|     instead of `global_sort_context`. Or compile a     |      |     |
|     specialised qsort with a templated compare (one    |      |     |
|     per criteria pair). Should close the remaining     |      |     |
|     ~20% single-thread gap.                            |      |     |
| C3. **Fix the multi-threaded sort.** Reverse-engineer  | 4h   | high |
|     B1 in section 3.1. Either: (a) debug the k-way     |      |     |
|     merge OOB; or (b) replace with parallel sample     |      |     |
|     sort (well-studied algorithm, no merge phase       |      |     |
|     headaches). At N=500k on 14 cores, expect 3–5x     |      |     |
|     speedup over single-threaded.                      |      |     |
| C4. **Build the spatial index for LocateCell.**        | 3h   | medium |
|     Populate `GridDataOpt.spatial_index` after sort.   |      |     |
|     Make `GridLocateCellOpt` use it (binary search     |      |     |
|     when sorted, hash when not). Target: O(log N).     |      |     |
| C5. **SIMD scan for the unsorted LocateCell path.**    | 2h   | low |
|     Build a dense int32 cache of `.index` fields on    |      |     |
|     demand; AVX2-scan it in batches of 8. Target:      |      |     |
|     2× scalar at N=10k+.                               |      |     |

Acceptance: harness reports speedup ≥ 1.0x at every (op, N=10k+) row,
≥ 2.0x at every (op, N=100k+) row.

### Phase D — File I/O parity (0.5 days)

Goal: `.grd` files written by libgrdopt are bitwise-identical to
those written by libgrd; can be read by external tools.

| Item | Effort | Risk |
|------|--------|------|
| D1. Cross-write test: write same GridData via libgrd   | 1h   | low |
|     and libgrdopt, diff outputs.                       |      |     |
| D2. Cross-read test: read same .grd file via both,     | 1h   | low |
|     diff GridData structs.                             |      |     |
| D3. Fix any divergences. Likely: timestamp precision,  | 2h   | medium |
|     field ordering, optional metadata.                 |      |     |

Acceptance: `cmp libgrd.grd libgrdopt.grd` returns 0 on a corpus
of 10+ synthetic and real-world grids.

### Phase E — Production hardening (0.5 days)

Goal: library is safe to use in long-running pipelines.

| Item | Effort | Risk |
|------|--------|------|
| E1. Run the harness under ASAN + UBSAN; fix any        | 2h   | medium |
|     reports.                                           |      |     |
| E2. Run under Helgrind / TSan with OMP > 1; fix any    | 2h   | medium |
|     data races.                                        |      |     |
| E3. Audit all `malloc`/`free` paths for leaks under    | 1h   | low |
|     error returns.                                     |      |     |
| E4. Document the public API in a doc/grdopt.tex (the   | 1h   | low |
|     standard RST doc format) so it ships with the      |      |     |
|     RST documentation build.                           |      |     |

Acceptance: zero ASAN/UBSAN/TSan findings on the harness; documented
API matches the actual exports.

---

## 6. Total effort

| Phase | Days | Outcome                                       |
|-------|------|-----------------------------------------------|
| A     | 1.5  | Symbol coverage complete                      |
| B     | 1.0  | Bitwise equivalence on all ops                |
| C     | 1.5  | Actually faster than libgrd                   |
| D     | 0.5  | File I/O byte-identical to original           |
| E     | 0.5  | Production-hardened                           |
| **Σ** | **5.0 days** | "complete + faster than libgrd"      |

---

## 7. Acceptance criteria for "library complete"

A reasonable definition of done that fits one PR review:

1. `nm libgrdopt.1.a` exports a `*Opt` symbol for every public
   `Grid*` symbol in `libgrd.1.a`.
2. `grid_opt_compare` harness reports `equiv_fail=0` for every
   measured op at every measured N.
3. `grid_opt_compare` reports speedup ≥ 1.0x at N ≥ 10k for every op,
   speedup ≥ 2.0x at N ≥ 100k for sort + locate.
4. ASAN / TSan / UBSAN clean on the harness.
5. `grid_opt_compare` is added as a CI job; PR fails if (2) or (3)
   regress.
6. `include/griddata_parallel_api.h` exists, declares all parallel
   public API, no extern declarations needed at consumer sites.
7. `.grd` files written by libgrdopt diff clean against libgrd's
   output.
8. doc/grdopt.tex exists and documents the public API.

---

## 8. Risks / known unknowns

- **GridGVecOpt redesign (C1) may cascade.** ~25 references to fields
  with known offsets in SIMD code; needs full re-test.
- **B1 root cause may be in third-party (libgomp) behavior** rather
  than in our merge code. If so, the parallel sort needs an algorithm
  change, not a bug fix.
- **CUDA paths are entirely untested.** `*_cuda.h` headers exist,
  `*.cu` files exist, no CI builds them. Out of scope for this audit
  (separate cuda-validation workflow exists).
- **Real `.grd` data fixtures don't exist in-tree.** All testing has
  used synthetic data. Real-world grid sizes (stations, vector
  density) may not match the synthetic distribution; performance
  characteristics could differ.
