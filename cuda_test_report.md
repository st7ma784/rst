# CUDA Module Test Report

Generated: Fri Jul 25 10:46:43 PM BST 2025

## Test Summary
- **Total modules tested**: 8
- **Modules passed**: 0  
- **Modules failed**: 8
- **Success rate**: 0%

## Module Status

### fitacf_v3.0
- ✅ CUDA makefile present

### fit.1.35
- ❌ No CUDA makefile

### grid.1.24_optimized.1
- ❌ No CUDA makefile

### lmfit_v2.0
- ✅ CUDA makefile present

### acf.1.16_optimized.2.0
- ✅ CUDA makefile present

### binplotlib.1.0_optimized.2.0
- ✅ CUDA makefile present

### fitacf.2.5
- ✅ CUDA makefile present

### cuda_common
- ✅ CUDA makefile present


## Next Steps
1. Address any build failures
2. Run performance benchmarks
3. Test drop-in compatibility
4. Update documentation

## Usage Instructions
Each CUDA-enabled module provides three build variants:
- **CPU**: Standard CPU implementation
- **CUDA**: GPU-accelerated implementation
- **Compatibility**: Automatic CPU/GPU selection

To use CUDA versions, link with the appropriate library variant:
```bash
# Link with CUDA version
-l<module_name>.cuda

# Link with compatibility version  
-l<module_name>.compat
```
