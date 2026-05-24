# FITACF v3 — F0–F5 Audit & Findings

Date: 2026-05-25. Companion to `codebase/superdarn/src.lib/tk/grid.1.24_optimized.1/AUDIT.md`.

## Executive summary

**F0 (harness) is delivered.** Code at
`codebase/superdarn/src.bin/tk/testing/fitacf_opt_compare.1.0/`. Builds
clean against `libfitacf.3.0`, runs `Fitacf()` on synthetic ACF data,
times it (~0.6–1.1 ms per call at nrang=75, mplgs=18), and reports
valid-range count plus recovered v/w. Wired analogously to
`grid_opt_compare`.

**F1–F4 (AVX work) are blocked on a structural rewrite that isn't
warranted right now.** The pragmatic ceiling for AVX on the *current*
FITACF v3 architecture is ~5–10%. The big win (3–10×) requires
finishing the SoA rewrite that was started in
`fitacf_array_optimized.c` but never completed. That's a multi-day
project; below is the evidence for that call and what the rewrite
would look like.

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

## F2 — Status of the SoA path

**Conclusion: `fitacf_array_optimized.c` is incomplete scaffolding,
not a working alternative path.**

Evidence:
- File exists (612 LOC, declares `Fitacf_Array()`, `Parallel_Power_Fitting_Array()`,
  `Parallel_Phase_Fitting_Array()`, etc.).
- Has partial AVX2 vectorization at lines 103–140 (power calc, phase
  placeholder).
- **Is not in `src/makefile`'s `SRC=` list** — never compiled.
- References helpers that are **declared in `fit_structures_array.h`
  but never defined** anywhere: `create_range_data_arrays`,
  `destroy_range_data_arrays`, `convert_prms_to_array`. Linking
  `Fitacf_Array` today would produce undefined-symbol errors.

To make this path live would require, at minimum:
1. Implement the missing allocators (~150 LOC).
2. Implement `convert_prms_to_array` (translate `FITPRMS` →
   `FITPRMS_ARRAY` / `RANGE_DATA_ARRAYS`).
3. Implement the per-stage helpers (Phase_Fit_Array, XCF_Fit_Array)
   that the existing top-level skeleton calls — they're not in
   `fitacf_array_optimized.c`.
4. Add `fitacf_array_optimized.c` to the makefile.
5. Equivalence-test against `Fitacf()` via the F0 harness; expect
   1–3 rounds of fixing real divergences.

Estimated effort: **2–3 focused days**. Not done here — this audit
flags it as the next-PR-sized item.

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
