# Pythonv2 Test and Coverage Report

Date: 2026-04-19
Scope: FITACF, ACF, and expanded low-coverage follow-up suites on `main`

## Test Results

Command:

```bash
source venv/bin/activate
PYTHONPATH=$PWD python -m pytest -q tests/test_fitacf_processing.py tests/test_acf_processing.py tests/test_coverage_additions.py --cov=superdarn_gpu --cov-report=term-missing
```

Outcome:

- Passed: 45
- Failed: 0
- TOTAL coverage: 62%

## Fixes Applied

- Hardened package imports for missing optional modules (`io.writers`, `io.streaming`, `algorithms.statistics`, visualization/tool extras).
- Updated `CUDArst` test target to compile and run existing root-level integration tests.
- Added `LD_LIBRARY_PATH` in `CUDArst` test execution so shared library resolves at runtime.
- Corrected ACF sample indexing and lag table conversion for synthetic test semantics.
- Improved ACF noise estimation to amplitude-like scale from lag-0 power.
- Ensured XCF is `None` when XCF processing is disabled.
- Added FITACF fit stability gating on quality flags.
- Adjusted FITACF noisy synthetic accuracy check to robust distribution-based median assertions.
- Added `tests/test_coverage_additions.py` with targeted smoke/unit coverage for:
	- `io.readers` loader routing and placeholder errors
	- `visualization.scientific`, `visualization.realtime`, `visualization.performance`, and `visualization.interactive`
	- `algorithms.interpolation`, `core.memory`, `core.pipeline`, and `processing.grid` basic paths
- Expanded dashboard coverage with constructor smoke tests for:
	- `create_processing_dashboard`
	- `create_performance_dashboard`
	- `create_validation_dashboard`
- Fixed HDF5 timestamp parsing in `io.readers` to handle both `bytes` and `str` attributes.

## Coverage Gaps (Top 10 Low Coverage Modules)

1. `superdarn_gpu/algorithms/interpolation.py` - 35%
2. `superdarn_gpu/core/memory.py` - 36%
3. `superdarn_gpu/visualization/realtime.py` - 41%
4. `superdarn_gpu/processing/grid.py` - 42%
5. `superdarn_gpu/visualization/performance.py` - 47%
6. `superdarn_gpu/core/pipeline.py` - 49%
7. `superdarn_gpu/visualization/interactive.py` - 53%
8. `superdarn_gpu/io/readers.py` - 59%
9. `superdarn_gpu/visualization/scientific.py` - 66%
10. `superdarn_gpu/visualization/dashboards.py` - 83%

## Recommended Next Coverage Additions

- Expand readers tests to cover HDF5 RawACF/FITACF minimal-record successful parsing.
- Add interpolation tests for RBF, nearest, and linear/cubic fallbacks.
- Add grid tests for non-empty vector extraction and deterministic interpolation outputs.
- Add pipeline batch tests including `run_batch` memory cleanup calls.
