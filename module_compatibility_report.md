# SuperDARN Module Compatibility Report

Generated: Fri Jul 25 10:47:04 PM BST 2025

## Executive Summary

This report details the build status and drop-in replacement capability of all SuperDARN modules, including CUDA variants.

## CUDA Environment

✅ CUDA is available and functional
nvcc: NVIDIA (R) Cuda compiler driver
GPU: NVIDIA GeForce RTX 3090

## Module Build Status

| Module | Standard Build | CUDA Build | Tests | Drop-in Ready |
|--------|---------------|------------|-------|---------------|
| raw.1.22 | ✅ SUCCESS | ➖ N/A | ✅ SUCCESS | ❌ |
| fitacf_v3.0 | ✅ SUCCESS | ✅ SUCCESS | ✅ SUCCESS | ✅ |
| radar.1.22 | ✅ SUCCESS | ➖ N/A | ✅ SUCCESS | ❌ |
| acf.1.16_optimized.2.0 | ✅ SUCCESS | ✅ SUCCESS | ✅ SUCCESS | ✅ |
| fitacf.2.5 | ✅ SUCCESS | ✅ SUCCESS | ✅ SUCCESS | ✅ |
| grid.1.24_optimized.1 | ✅ SUCCESS | ✅ SUCCESS | ✅ SUCCESS | ✅ |
| cfit.1.19 | ✅ SUCCESS | ➖ N/A | ✅ SUCCESS | ❌ |
| lmfit_v2.0 | ✅ SUCCESS | ✅ SUCCESS | ✅ SUCCESS | ✅ |
| grid.1.24 | ✅ SUCCESS | ➖ N/A | ✅ SUCCESS | ❌ |
| binplotlib.1.0_optimized.2.0 | ✅ SUCCESS | ✅ SUCCESS | ✅ SUCCESS | ✅ |
| fit.1.35 | ✅ SUCCESS | ✅ SUCCESS | ✅ SUCCESS | ✅ |

## Detailed Issues

### All modules built successfully! 🎉

## Build Logs

- Full build log: `/home/user/rst/build_report.log`
- Test log: `/home/user/rst/test_report.log`

## Next Steps

1. Fix any failed builds identified above
2. Ensure all CUDA variants provide identical APIs
3. Run performance benchmarks comparing CPU vs CUDA implementations
4. Update documentation for drop-in replacement usage
