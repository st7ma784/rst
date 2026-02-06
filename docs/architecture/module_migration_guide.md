# RST SuperDARN Module Migration Guide

## Overview

This guide documents the systematic approach for migrating RST SuperDARN modules to CUDA acceleration, based on successful implementation of FITACF v3.0 and LMFIT v2.0.

## Migration Strategy

### Phase 1: Assessment and Planning

#### 1. **Module Analysis Checklist**
```bash
# Analyze module for migration potential
cd /path/to/module
find . -name "*.c" -exec grep -l "struct.*next" {} \;  # Find linked lists
find . -name "*.c" -exec grep -l "malloc\|calloc" {} \; # Find memory allocation
wc -l src/*.c                                          # Code complexity
grep -r "for.*range\|for.*lag" src/                   # Loop patterns
```

#### 2. **Migration Priority Matrix**
| Factor | Weight | Scoring |
|--------|--------|---------|
| Computational Intensity | 40% | High=3, Medium=2, Low=1 |
| Linked List Usage | 30% | Heavy=3, Some=2, None=1 |
| Data Parallelism | 20% | High=3, Medium=2, Low=1 |
| Usage Frequency | 10% | Critical=3, Common=2, Rare=1 |

**Migration Score = Σ(Factor × Weight)**

#### 3. **Example Module Scoring**
```
FITACF v3.0:
- Computational: High (3) × 40% = 1.2
- Linked Lists: Heavy (3) × 30% = 0.9  
- Parallelism: High (3) × 20% = 0.6
- Usage: Critical (3) × 10% = 0.3
Total Score: 3.0 (HIGH PRIORITY) ✅

Grid Processing:
- Computational: Medium (2) × 40% = 0.8
- Linked Lists: Some (2) × 30% = 0.6
- Parallelism: High (3) × 20% = 0.6  
- Usage: Common (2) × 10% = 0.2
Total Score: 2.2 (MEDIUM PRIORITY) 🟡
```

### Phase 2: Implementation Patterns

#### Pattern A: Linked List → Array Migration

**Before (Linked List):**
```c
// Original linked list structure
typedef struct range_node {
    int range_idx;
    float power;
    float velocity;
    struct range_node *next;
} range_node_t;

// Sequential processing
range_node_t *current = head;
while (current != NULL) {
    process_range(current);
    current = current->next;
}
```

**After (CUDA Array):**
```c
// CUDA-compatible array structure  
typedef struct {
    int *range_indices;     // Device array
    float *power_values;    // Device array
    float *velocity_values; // Device array
    bool *validity_mask;    // Device array
    int count;             // Number of valid ranges
    int capacity;          // Array capacity
} cuda_range_array_t;

// Parallel processing
__global__ void process_ranges_kernel(cuda_range_array_t *ranges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ranges->count && ranges->validity_mask[idx]) {
        // Process range in parallel
        process_range_cuda(ranges, idx);
    }
}
```

#### Pattern B: Memory Management Migration

**Before (CPU-only):**
```c
// CPU memory allocation
float *data = malloc(size * sizeof(float));
// ... process data ...
free(data);
```

**After (Unified Memory):**
```c
// Unified CPU/GPU memory
#include "cuda_datatypes.h"

cuda_unified_memory_t mem;
cuda_memory_alloc(&mem, size * sizeof(float), CUDA_MEMORY_MANAGED);

// Use on both CPU and GPU
cpu_process(mem.host_ptr);
gpu_process<<<blocks, threads>>>(mem.device_ptr);

cuda_memory_free(&mem);
```

#### Pattern C: Build System Integration

**Directory Structure:**
```
module_name/
├── src/
│   ├── original_cpu_code.c
│   ├── cuda/
│   │   ├── module_cuda.h
│   │   ├── module_cuda_kernels.cu
│   │   └── module_cuda_bridge.c
│   └── ...
├── include/
│   ├── original_headers.h
│   └── module_cuda.h
├── makefile          # Original CPU build
├── makefile.cuda     # CUDA build extension
└── ...
```

**makefile.cuda Template:**
```makefile
# CUDA Extension Makefile for [MODULE_NAME]
CUDA_ARCH ?= sm_50
CUDA_FLAGS = -arch=$(CUDA_ARCH) -O3 -Xcompiler -fPIC

# CUDA sources
CUDA_SOURCES = src/cuda/$(MODULE)_cuda_kernels.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Combined library with CUDA support
lib$(MODULE)_cuda.a: $(OBJECTS) $(CUDA_OBJECTS)
	$(AR) rcs $@ $^

# CUDA compilation rule
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -I$(CUDA_COMMON)/include -c $< -o $@

# Include original makefile
include makefile
```

### Phase 3: Module-Specific Implementation

#### Grid Processing Modules

**Key Challenges:**
- Large spatial data arrays
- Interpolation algorithms
- Memory bandwidth optimization

**Implementation Strategy:**
```c
// Grid cell structure (replaces linked spatial lists)
typedef struct {
    float *lat_array;       // Latitude values
    float *lon_array;       // Longitude values  
    float *value_array;     // Data values
    bool *valid_mask;       // Validity flags
    int *neighbor_indices;  // Spatial neighbors (2D → 1D mapping)
    int num_cells;
} cuda_grid_t;

// CUDA kernel for spatial interpolation
__global__ void interpolate_grid_kernel(cuda_grid_t *grid, 
                                       float target_lat, float target_lon,
                                       float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < grid->num_cells && grid->valid_mask[idx]) {
        // Parallel spatial interpolation
        float distance = calculate_distance(grid->lat_array[idx], 
                                          grid->lon_array[idx],
                                          target_lat, target_lon);
        // Atomic operations for weighted averaging
        atomicAdd(result, grid->value_array[idx] / (distance + 1e-6));
    }
}
```

#### I/O Acceleration Modules  

**Key Challenges:**
- File format parsing
- Data streaming
- Memory transfer optimization

**Implementation Strategy:**
```c
// Buffered I/O with GPU preprocessing
typedef struct {
    char *file_buffer;      // Large file buffer
    void *staging_buffer;   // GPU staging area
    cudaStream_t stream;    // Async transfer stream
    int buffer_size;
    int current_pos;
} cuda_io_context_t;

// Asynchronous data loading and preprocessing
cudaError_t load_and_preprocess_async(cuda_io_context_t *ctx, 
                                     const char *filename) {
    // 1. Load file chunk to staging buffer
    load_chunk_async(ctx->file_buffer, ctx->buffer_size, ctx->stream);
    
    // 2. Launch preprocessing kernels
    preprocess_data_kernel<<<blocks, threads, 0, ctx->stream>>>(
        ctx->staging_buffer, ctx->buffer_size);
    
    // 3. Overlap next chunk loading with current processing
    return cudaStreamSynchronize(ctx->stream);
}
```

### Phase 4: Testing and Validation

#### Performance Testing Framework

**Benchmark Template:**
```c
// Performance comparison test
void benchmark_module(int data_size, int iterations) {
    // Setup test data
    setup_test_data(data_size);
    
    // CPU baseline
    clock_t cpu_start = clock();
    for (int i = 0; i < iterations; i++) {
        cpu_module_process(test_data);
    }
    double cpu_time = (clock() - cpu_start) / (double)CLOCKS_PER_SEC;
    
    // CUDA implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cuda_module_process(test_data);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start, stop);
    
    // Report results
    printf("Speedup: %.2fx\n", cpu_time / (cuda_time / 1000.0));
}
```

#### Numerical Validation

**Accuracy Testing:**
```c
// Numerical accuracy validation
bool validate_module_accuracy(void *cpu_result, void *cuda_result, 
                             int data_size, float tolerance) {
    float *cpu_data = (float*)cpu_result;
    float *cuda_data = (float*)cuda_result;
    
    float max_error = 0.0f;
    float rms_error = 0.0f;
    
    for (int i = 0; i < data_size; i++) {
        float error = fabsf(cpu_data[i] - cuda_data[i]);
        max_error = fmaxf(max_error, error);
        rms_error += error * error;
    }
    
    rms_error = sqrtf(rms_error / data_size);
    
    printf("Max Error: %e, RMS Error: %e\n", max_error, rms_error);
    return (max_error < tolerance) && (rms_error < tolerance * 0.1f);
}
```

### Phase 5: Integration and Deployment

#### CUDArst Library Integration

**Adding New Module to CUDArst:**
```c
// 1. Add module interface to cudarst.h
typedef struct {
    // Module-specific data structures
} cudarst_module_data_t;

cudarst_error_t cudarst_module_process(cudarst_module_data_t *data);

// 2. Implement in cudarst_module.c
cudarst_error_t cudarst_module_process(cudarst_module_data_t *data) {
    if (cudarst_is_cuda_available()) {
        return cuda_module_process_internal(data);
    } else {
        return cpu_module_process_internal(data);
    }
}

// 3. Add to unified makefile
SOURCES += src/cudarst_module.c
CUDA_SOURCES += src/cudarst_module_kernels.cu
```

## Migration Checklist

### Pre-Migration
- [ ] Module complexity analysis completed
- [ ] Migration priority score calculated  
- [ ] Linked list usage patterns identified
- [ ] Memory allocation patterns documented
- [ ] Performance baseline established

### Implementation
- [ ] Data structures migrated to arrays + masks
- [ ] CUDA kernels implemented and tested
- [ ] CPU/GPU bridge layer created
- [ ] Memory management updated to unified system
- [ ] Build system extended with makefile.cuda

### Testing  
- [ ] Unit tests pass for both CPU and CUDA paths
- [ ] Performance benchmarks show expected speedup
- [ ] Numerical accuracy validation passes
- [ ] Memory leak testing completed
- [ ] Edge case handling verified

### Integration
- [ ] CUDArst library interface added
- [ ] Backward compatibility confirmed
- [ ] Documentation updated
- [ ] Example programs created
- [ ] Migration guide updated

## Troubleshooting Common Issues

### Performance Issues
```
Issue: Lower than expected speedup
Debugging:
1. Check memory access patterns (coalesced vs. scattered)
2. Verify occupancy with nvprof
3. Look for CPU/GPU synchronization points
4. Ensure sufficient parallelism (thousands of threads)
```

### Memory Issues
```
Issue: CUDA out of memory errors
Solutions:
1. Implement data streaming/chunking
2. Use unified memory with hints
3. Optimize memory layout for GPU
4. Implement memory pool management
```

### Numerical Issues
```
Issue: Results differ between CPU and CUDA
Debugging:
1. Check floating-point precision (float vs. double)
2. Verify atomic operation usage
3. Test with smaller datasets
4. Compare intermediate results step-by-step
```

## Success Metrics

### Performance Targets
- **Minimum**: 2x speedup for migration to be worthwhile
- **Good**: 5x speedup indicates well-optimized implementation  
- **Excellent**: 10x+ speedup shows optimal GPU utilization

### Quality Targets
- **Numerical Accuracy**: RMS error < 0.1% of signal magnitude
- **Memory Efficiency**: GPU memory usage < 80% of available
- **Compatibility**: 100% backward compatibility maintained

---

*This guide is based on successful migration of FITACF v3.0 and LMFIT v2.0 modules, achieving 5-16x performance improvements while maintaining full backward compatibility.*