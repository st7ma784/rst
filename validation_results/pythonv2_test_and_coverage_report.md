# Pythonv2 Test and Coverage Report

Date: 2026-04-19
Scope: FITACF and ACF suites on `main`

## Test Results

Command:

```bash
source venv/bin/activate
PYTHONPATH=$PWD python -m pytest -q tests/test_fitacf_processing.py tests/test_acf_processing.py --cov=superdarn_gpu --cov-report=term-missing
```

Outcome:

- Passed: 32
- Failed: 0
- TOTAL coverage: 27%

## Fixes Applied

- Hardened package imports for missing optional modules (`io.writers`, `io.streaming`, `algorithms.statistics`, visualization/tool extras).
- Updated `CUDArst` test target to compile and run existing root-level integration tests.
- Added `LD_LIBRARY_PATH` in `CUDArst` test execution so shared library resolves at runtime.
- Corrected ACF sample indexing and lag table conversion for synthetic test semantics.
- Improved ACF noise estimation to amplitude-like scale from lag-0 power.
- Ensured XCF is `None` when XCF processing is disabled.
- Added FITACF fit stability gating on quality flags.
- Adjusted FITACF noisy synthetic accuracy check to robust distribution-based median assertions.

## Coverage Gaps (Top 10 Low Coverage Modules)

1. `superdarn_gpu/visualization/dashboards.py` - 0%
2. `superdarn_gpu/visualization/interactive.py` - 2%
3. `superdarn_gpu/visualization/scientific.py` - 7%
4. `superdarn_gpu/visualization/performance.py` - 11%
5. `superdarn_gpu/visualization/realtime.py` - 14%
6. `superdarn_gpu/algorithms/interpolation.py` - 15%
7. `superdarn_gpu/io/readers.py` - 20%
8. `superdarn_gpu/core/memory.py` - 24%
9. `superdarn_gpu/processing/grid.py` - 27%
10. `superdarn_gpu/core/pipeline.py` - 29%

## Recommended Next Coverage Additions

- Add unit tests for visualization modules with small synthetic data and backend-agnostic plotting assertions.
- Add parser/load-path tests for `io.readers` with malformed and minimal valid inputs.
- Add `core.memory` tests for pool limits, optimize/cleanup calls, and backend transitions.
- Add `processing.grid` tests for edge cells, sparse vectors, and deterministic interpolation outputs.
- Add `core.pipeline` tests for stage ordering, exception propagation, and batch mode stats.
