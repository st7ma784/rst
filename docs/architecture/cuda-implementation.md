# CUDA Implementation

Technical documentation of the CUDA acceleration implementation in RST.

## Design Philosophy

The CUDA implementation follows these principles:

1. **Drop-in Replacement** - Same API, automatic acceleration
2. **Graceful Fallback** - Works without GPU
3. **Numerical Equivalence** - Results match CPU within tolerance
4. **Module Independence** - Each module accelerated separately

## Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Application Code                             │
│                 (unchanged from original)                       │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│                   CUDArst Interface Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ cuda_fitacf  │  │  cuda_grid   │  │  cuda_raw    │  ...    │
│  │  _process()  │  │  _process()  │  │  _process()  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼──────────────────┐
│                 Runtime Decision Layer                          │
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │  if (cuda_available && data_size > threshold)       │     │
│    │      → use_cuda_backend()                           │     │
│    │  else                                               │     │
│    │      → use_cpu_backend()                            │     │
│    └─────────────────────────────────────────────────────┘     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌──────────────────┐                   ┌──────────────────┐
│   CUDA Backend   │                   │   CPU Backend    │
│  ┌────────────┐  │                   │  ┌────────────┐  │
│  │  Kernels   │  │                   │  │  Original  │  │
│  │  Memory    │  │                   │  │   Code     │  │
│  │  Streams   │  │                   │  └────────────┘  │
│  └────────────┘  │                   └──────────────────┘
└──────────────────┘
```

### CUDA Kernel Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU Execution Model                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Grid of Blocks                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Block(0,0)  Block(1,0)  Block(2,0)  ...  Block(N,0) │    │
│  │    │            │            │              │        │    │
│  │    ▼            ▼            ▼              ▼        │    │
│  │ ┌──────┐    ┌──────┐    ┌──────┐       ┌──────┐    │    │
│  │ │Range0│    │Range1│    │Range2│  ...  │RangeN│    │    │
│  │ │Lag 0 │    │Lag 0 │    │Lag 0 │       │Lag 0 │    │    │
│  │ │Lag 1 │    │Lag 1 │    │Lag 1 │       │Lag 1 │    │    │
│  │ │ ...  │    │ ...  │    │ ...  │       │ ...  │    │    │
│  │ └──────┘    └──────┘    └──────┘       └──────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Each block processes one range gate                         │
│  Each thread within block processes one lag                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Unified Memory Manager

```c
// include/cuda_common/cuda_memory_manager.h

typedef struct {
    void *host_ptr;           // CPU memory pointer
    void *device_ptr;         // GPU memory pointer
    size_t size;              // Allocation size
    cuda_memory_type_t type;  // Memory type
    bool host_valid;          // Host data current?
    bool device_valid;        // Device data current?
} cuda_unified_memory_t;

// Memory operations
cuda_error_t cuda_memory_alloc(cuda_unified_memory_t *mem, 
                               size_t size,
                               cuda_memory_type_t type);

cuda_error_t cuda_memory_sync_to_device(cuda_unified_memory_t *mem);
cuda_error_t cuda_memory_sync_to_host(cuda_unified_memory_t *mem);
cuda_error_t cuda_memory_free(cuda_unified_memory_t *mem);
```

### 2. CUDA-Compatible Data Structures

```c
// Linked list replacement
typedef struct {
    cuda_unified_memory_t data_memory;   // Array of elements
    cuda_unified_memory_t mask_memory;   // Validity mask
    size_t element_size;                 // Element size
    size_t capacity;                     // Max elements
    size_t count;                        // Valid elements
} cuda_list_t;

// Range gate batch
typedef struct {
    float *power;           // [n_ranges]
    float *velocity;        // [n_ranges]
    float *width;           // [n_ranges]
    bool  *valid;           // [n_ranges]
    int   *lag_indices;     // [n_ranges × n_lags]
    float *acf_real;        // [n_ranges × n_lags]
    float *acf_imag;        // [n_ranges × n_lags]
    int    n_ranges;
    int    n_lags;
} cuda_range_batch_t;
```

### 3. Kernel Implementations

#### ACF Processing Kernel

```cuda
// codebase/superdarn/src.lib/tk/fitacf_v3.0/src/cuda_kernels.cu

__global__ void process_acf_kernel(
    const float *acf_real,    // Input: ACF real parts
    const float *acf_imag,    // Input: ACF imaginary parts
    const bool  *valid_mask,  // Input: Validity mask
    float *power,             // Output: Power values
    float *phase,             // Output: Phase values
    int n_ranges,             // Number of ranges
    int n_lags                // Number of lags
) {
    // Each block handles one range
    int range_idx = blockIdx.x;
    // Each thread handles one lag
    int lag_idx = threadIdx.x;
    
    if (range_idx >= n_ranges || lag_idx >= n_lags)
        return;
    
    int idx = range_idx * n_lags + lag_idx;
    
    if (!valid_mask[idx])
        return;
    
    float real = acf_real[idx];
    float imag = acf_imag[idx];
    
    // Compute power and phase
    power[idx] = real * real + imag * imag;
    phase[idx] = atan2f(imag, real);
}
```

#### Parallel Reduction Kernel

```cuda
__global__ void reduce_statistics_kernel(
    const float *values,
    const bool *valid_mask,
    float *sum_out,
    float *sum_sq_out,
    int *count_out,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Load and accumulate
    float sum = 0.0f, sum_sq = 0.0f;
    int count = 0;
    
    if (i < n && valid_mask[i]) {
        float v = values[i];
        sum = v;
        sum_sq = v * v;
        count = 1;
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    sdata[tid + blockDim.x] = sum_sq;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(sum_out, sdata[0]);
        atomicAdd(sum_sq_out, sdata[blockDim.x]);
        atomicAdd(count_out, count);
    }
}
```

### 4. CPU-CUDA Bridge

```c
// codebase/superdarn/src.lib/tk/fitacf_v3.0/src/cuda_cpu_bridge.c

// Transparent interface that selects backend
int fitacf_process(FitACFData *data) {
    // Check CUDA availability
    if (cuda_is_available() && 
        data->n_ranges >= MIN_CUDA_RANGES &&
        !getenv("RST_DISABLE_CUDA")) {
        
        return cuda_fitacf_process(data);
    }
    
    // Fall back to CPU
    return cpu_fitacf_process(data);
}

// CUDA implementation
static int cuda_fitacf_process(FitACFData *data) {
    cuda_range_batch_t batch;
    
    // 1. Convert to CUDA-friendly format
    convert_to_batch(data, &batch);
    
    // 2. Allocate GPU memory
    cuda_batch_alloc(&batch);
    
    // 3. Transfer to GPU
    cuda_batch_to_device(&batch);
    
    // 4. Execute kernels
    int blocks = batch.n_ranges;
    int threads = min(batch.n_lags, 256);
    
    process_acf_kernel<<<blocks, threads>>>(
        batch.acf_real, batch.acf_imag,
        batch.valid, batch.power, batch.phase,
        batch.n_ranges, batch.n_lags
    );
    
    // 5. Transfer results back
    cuda_batch_to_host(&batch);
    
    // 6. Convert back to original format
    convert_from_batch(&batch, data);
    
    // 7. Cleanup
    cuda_batch_free(&batch);
    
    return 0;
}
```

## Module Implementations

### FITACF v3.0

**Kernels:**
- `process_acf_kernel` - ACF power/phase computation
- `fit_model_kernel` - Parallel model fitting
- `noise_estimation_kernel` - Statistical noise calculation
- `bad_lag_detection_kernel` - Quality filtering

**Performance:**
- 8-16x speedup on typical data
- Best for >50 ranges

### Grid 1.24

**Kernels:**
- `grid_locate_kernel` - Parallel cell location (O(1) vs O(n))
- `grid_merge_kernel` - Data merging with shared memory
- `grid_stats_kernel` - Statistical reduction
- `grid_sort_kernel` - Thrust-based sorting

**Performance:**
- 5-10x speedup
- Best for large grids (>1000 cells)

### ACF 1.16

**Kernels:**
- `acf_compute_kernel` - Core ACF calculation
- `power_spectrum_kernel` - Power computation
- `bad_sample_kernel` - Quality detection
- `normalize_kernel` - Data normalization

**Performance:**
- 20-60x speedup
- Excellent for long time series

## Memory Optimization

### Coalesced Access

```cuda
// BAD: Strided access (slow)
value = data[threadIdx.x * stride + offset];

// GOOD: Coalesced access (fast)
value = data[blockIdx.x * blockDim.x + threadIdx.x];
```

### Shared Memory Usage

```cuda
__global__ void optimized_kernel(...) {
    // Shared memory for block-local data
    __shared__ float shared_data[256];
    
    // Load to shared memory (coalesced)
    shared_data[threadIdx.x] = global_data[global_idx];
    __syncthreads();
    
    // Work with shared memory (fast)
    float result = process(shared_data[threadIdx.x]);
    
    // Write back (coalesced)
    output[global_idx] = result;
}
```

### Memory Pooling

```c
// Pre-allocate memory pools
typedef struct {
    cuda_unified_memory_t pools[NUM_POOLS];
    size_t pool_sizes[NUM_POOLS];
    bool pool_in_use[NUM_POOLS];
} cuda_memory_pool_t;

void *cuda_pool_alloc(cuda_memory_pool_t *pool, size_t size) {
    // Find suitable pool
    for (int i = 0; i < NUM_POOLS; i++) {
        if (!pool->pool_in_use[i] && 
            pool->pool_sizes[i] >= size) {
            pool->pool_in_use[i] = true;
            return pool->pools[i].device_ptr;
        }
    }
    // Fallback to new allocation
    return cuda_malloc(size);
}
```

## Stream Processing

### Overlapping Transfers and Compute

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Batch 1: Transfer while computing previous
cudaMemcpyAsync(d_batch1, h_batch1, size, 
                cudaMemcpyHostToDevice, stream1);

// Batch 0: Compute (overlaps with batch 1 transfer)
process_kernel<<<blocks, threads, 0, stream2>>>(d_batch0);

// Synchronize
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

## Error Handling

```c
// Comprehensive error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return CUDA_ERROR; \
    } \
} while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
```

## Testing and Validation

### Numerical Accuracy Testing

```c
// Compare CPU vs CUDA results
int validate_results(float *cpu, float *cuda, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - cuda[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    // Accept < 0.1% difference (floating point tolerance)
    return (max_diff / max_value) < 0.001f;
}
```

### Benchmark Framework

```bash
# Run benchmarks
cd CUDArst
./tests/run_benchmarks.sh

# Output:
# Operation          CPU(ms)    CUDA(ms)   Speedup
# ─────────────────────────────────────────────────
# ACF Processing     452.3      28.1       16.1x
# Power Computation  387.2      30.9       12.5x
# Statistics         521.4      42.1       12.4x
```

## Next Steps

- [Data Structures](data-structures.md) - Detailed transformation guide
- [Migration Patterns](migration-patterns.md) - How to accelerate modules
- [Benchmarks](../guides/benchmarks.md) - Performance analysis
