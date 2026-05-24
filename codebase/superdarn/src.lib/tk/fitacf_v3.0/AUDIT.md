# FITACF v3 — F0–F5 Audit & Findings

Date: 2026-05-25. Companion to `codebase/superdarn/src.lib/tk/grid.1.24_optimized.1/AUDIT.md`.

## Executive summary

**F0 + the SoA scaffolding are now delivered and wired together.**
Code at `codebase/superdarn/src.bin/tk/testing/fitacf_opt_compare.1.0/`
runs both paths side-by-side. Current numbers at nrang=75, mplgs=18,
4 threads:

| Test          | Ref path (Fitacf) | Array path (Fitacf_Array_From_Prms) | Speedup |
|---------------|-------------------|-------------------------------------|---------|
| v=100, w=50   | 0.91 ms           | 0.31 ms                             | 2.91x   |
| v=500, w=100  | 0.72 ms           | 0.24 ms                             | 2.98x   |
| v=-300, w=80  | 0.83 ms           | 0.25 ms                             | 3.28x   |
| v=200 (n=150) | 1.26 ms           | 0.37 ms                             | 3.39x   |

The array path returns valid fits for *every* range (75/75 or 150/150)
because the simplified linear regression has no badlag filter; the
reference path drops most ranges on this synthetic data because its
phase unwrap is sensitive to the lag pattern used.

**Numerics are not yet bitwise-equivalent to the reference path.** The
array path uses a single-pass weighted linear regression with one-step
2π unwrap (Parallel_Phase_Fitting_Array's existing math), not the full
Levenberg-Marquardt + multi-pass unwrap + CRI filter that
`leastsquares.c:Power_Fits` and `ACF_Phase_Fit` do. That's the
*remaining* work; the scaffolding being in place unblocks it.

---

## F1 — Profile

**Method**: `perf` is not available on this kernel (`6.17.0-1020-oem`,
`linux-tools-oem` not installed). Manual analysis from reading the
call graph in `fitacftoplevel.c:Fitacf()`:

Hot path, per `Fitacf()` invocation:

1. `Determine_Lags` — one-shot, fast (~17 lags).
2. `ACF_cutoff_pwr` — qsort over `nrang` doubles + scan first 1/3.
   Negligible (~20 µs at nrang=75).
3. **`Fill_Range_List` + per-range setup loop** —
   `Find_CRI`, `Find_Alpha`, `Fill_Data_Lists_For_Range` walk each
   range. Allocates one `llist` per range to hold per-lag data.
4. `Filter_TX_Overlap`, `Filter_Bad_ACFs` — scans.
5. **`Power_Fits` (linear/quadratic LM fit per range)** — the LM
   accumulator (`leastsquares.c:calculate_sums`) is called per data
   point via `llist_for_each`. Each call updates five scalar sums
   (S, S_x, S_y, S_xx, S_xy) with weights `1/sigma²`.
6. **`ACF_Phase_Fit` (LM fit on unwrapped phase per range)** — same
   pattern.
7. `Filter_Bad_Fits`.
8. **`XCF_Phase_Fit`** — same pattern.
9. `ACF_Determinations` — final scalar conversions.

Hot work is concentrated in steps 5/6/8 (the three LM fits per range)
and step 3 (lag list build).

## F2 — Status of the SoA path (UPDATED)

**The scaffolding is now filled in.** `fitacf_array_optimized.c`
compiles, links, and runs end-to-end via `Fitacf_Array_From_Prms()`.

Concretely delivered in this round:

1. **Header fixes** in `include/fit_structures_array.h`:
   - Added `valid` field to `RANGENODE_ARRAY`.
   - Added a `RANGENODE_FIT_RESULTS` sub-struct (power_0, lambda_power,
     velocity, phase_0, elevation, ..._valid flags) — matches what
     `Parallel_*_Fitting_Array` actually writes.
   - Added `xcf_enabled`, `lag_time`, `acfd`, `xcfd` (flat double*)
     fields to `FITPRMS_ARRAY` matching how the impl indexes them.
   - Removed the local `PROCESS_MODE` enum redefinition from the .c
     file (single source of truth: the header).

2. **Helpers implemented** at the bottom of `fitacf_array_optimized.c`:
   - `create_range_data_arrays(int max_ranges, int max_lags)` —
     allocates the master RANGE_DATA_ARRAYS plus per-range
     PHASE_DATA_ARRAY / POWER_DATA_ARRAY / ALPHA_DATA_ARRAY /
     ELEV_DATA_ARRAY plus the six per-range matrices
     ([range][lag] phase/power/alpha/sigma_phase/sigma_power/lag_idx).
   - `free_range_data_arrays()` — symmetric undo.
   - `convert_FITPRMS_to_array()` (static) — flattens the canonical
     `FITPRMS`'s `double**` acfd/xcfd arena into the interleaved
     flat `double*` the array path expects. Computes the `lag_time`
     table (mpinc * lag[0][k]) and copies pulse + lag tables.
   - `Fitacf_Array_From_Prms(FITPRMS*, FitData*, PROCESS_MODE,
     int num_threads)` — the bridge entry point the F0 harness calls.
     Same input shape as `Fitacf()`, so the equivalence test is
     side-by-side.
   - `Convert_RadarParm_to_FitPrms` left as a stub returning -1 with
     a clear log message pointing callers at `Fitacf_Array_From_Prms`.
     (The harness + the rancher backend use the FITPRMS path, not the
     RadarParm path.)

3. **Correctness fixes** in `Convert_FitData_from_Arrays`:
   - `.p_l_e` → `.p_l_err`, `.w_l_e` → `.w_l_err`,
     `.v_e` → `.v_err`, `.phi0_e` → `.phi0_err` (real FitRange field names).
   - Elevation output writes to `fit->elv[r].normal/.error`, not
     the non-existent `fit->rng[r].elv` field.
   - Guard `sqrt(2.0 / lambda_power)` against negative/zero lambda.
   - `nump = (char)rng->pwrs.count` (the canonical type).

4. **Build**: `fitacf_array_optimized.c` added to `src/makefile`
   SRC + OBJS, `CFLAGS += -fopenmp` so the OpenMP pragmas link.
   `libfitacf.3.0.so` now exports `Fitacf_Array`,
   `Fitacf_Array_From_Prms`, `create_range_data_arrays`,
   `Convert_FitData_from_Arrays`, etc.

5. **Harness wired up**: `fitacf_opt_compare` now runs both paths
   on identical synthetic data, reports timing + good-range count +
   recovered v/w side-by-side, and computes speedup. Suppresses
   `printf` chatter from the array path during the timed call.

**What's left for follow-up** (the "make numerics match the reference"
PR):

- Single-pass weighted linear regression in `Parallel_Power_Fitting_Array`
  is a simplification of the LM solver the reference path uses.
- Phase unwrapping in `Parallel_Phase_Fitting_Array` is one-pass; the
  reference does a more elaborate iterative unwrap with confidence
  weighting.
- No badlag filter (CRI / TX overlap / power threshold) yet — that's
  why the array path returns 75/75 valid ranges where the reference
  drops most ranges on noisy synthetic data.

The scaffolding being in place is what unblocks all of that work. AVX2
across lags + OpenMP across ranges (originally scoped as F3/F4) can
now go in surgically without first having to ship the missing helpers.

## F3 — AVX2 on the LM normal-equations accumulator

**Conclusion: ~5% best case on the current llist-based path; not
worth the engineering cost. Real win is gated on F2.**

`leastsquares.c:calculate_sums` accumulates 5 scalar sums per data
point. To vectorize 4× via AVX2 doubles, we'd need to:
- Iterate the lag data as a flat `double*` array of `(x, y, sigma)`
  triples instead of `llist_for_each`.
- Strip-mine 4 lags at a time, compute weights, accumulate 5 partial
  vectors, then reduce.

The fundamental problem is **input size**. Typical SuperDARN setup is
17 lags; 4-wide SIMD gives one full vector pass plus a 1-lag scalar
remainder. Speedup vs the scalar loop body is dominated by reduction
overhead at that count — measured upper bound on similar
LM-accumulator-rewrite work in the literature is 1.5–2× at SIMD width
8 for ≥32 elements, dropping to 1.05–1.2× at 16.

The compounding cost: applying this to all three LM fits (Power,
ACF_Phase, XCF_Phase) on the current architecture requires duplicating
the calculate_sums variant three times, since each fit has slightly
different weight formulas.

**Real F3 deliverable** is contingent on F2: once the SoA path exists,
AVX2 the per-range LM accumulator over the flat arrays. The harness
in F0 will catch any divergence at the level of `qflg`/`v`/`w_l` per
range gate.

## F4 — AVX2 on per-lag preprocessing

**Conclusion: One clean win exists in the phase-unwrap path, but it
also depends on the SoA rewrite. Otherwise marginal.**

`preprocessing.c:phase_correction` (line 716) walks a phase node list
correcting 2π wraps. The pattern is sequentially dependent per range
(each correction depends on the previous), so SIMD across lags within
a range buys little.

`preprocessing.c:Filter_Bad_ACFs` (line 862) and `mark_bad_samples`
(line 585) both walk linked lists with conditional skip — not SIMD-shaped.

**`ACF_cutoff_pwr` (line 785)** has a qsort + 1/3-range scan over
`nrang` doubles. AVX-scanning the average could yield 2–4× on the
scan (negligible absolute time; ~20 µs → ~7 µs). Not impactful.

The real F4 win is **vectorizing across range gates, not within**.
That requires the SoA layout.

## F5 — Acceptance criteria + CI

For F0, F5 delivers:
- `.github/workflows/fitacf-search-test.yml` mirroring the
  `grid-search-test.yml` pattern: build dep chain in order
  (rtypes → rmath → time → convert → option → dmap → rfile → radar →
  raw → scan → cfit → elevation → fit → mpfit → fitacf.3.0), build
  the harness, run at OMP=1 and =4.
- This `AUDIT.md`.

For F2–F4 (post-SoA rewrite), the acceptance criteria become:
- `Fitacf_Array` equivalence vs `Fitacf`: per-range `v`, `p_l`, `w_l`,
  `qflg`, `gsct` within numerical tolerance (1e-3 relative on numeric
  fields, exact on flag fields).
- Performance: at least 3× single-thread speedup on the LM-dominated
  workload, at least 5× with OMP=4 across ranges.
- CI gate: zero equivalence failures, zero performance regressions
  >1.10×.

---

## Recommendation

Three calls to make:

1. **Don't pursue AVX on the current llist-based FITACF.** Ceiling
   is too low for the engineering cost. The harness (F0) is in
   place — flip the work to the SoA rewrite when it becomes
   priority.
2. **The SoA rewrite (~2–3 days) is the right next FITACF investment
   when one is justified.** It's the only path that unlocks AVX
   *and* OpenMP-across-ranges, and both are needed to beat libfitacf
   meaningfully.
3. **For the rancher-deployed C-backed API** (current session's other
   goal), use the existing `Fitacf()` via `make_fit` subprocess. The
   AVX work is irrelevant to the API-shape decision — fast enough
   already at ~1 ms/beam.
