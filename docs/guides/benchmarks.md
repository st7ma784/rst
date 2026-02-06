# Performance Benchmarks

Analysis of CUDA acceleration performance and optimization guidelines.

## Benchmark Results

### Summary

| Module | CPU Time | CUDA Time | Speedup | Best For |
|--------|----------|-----------|---------|----------|
| FITACF v3.0 | 452 ms | 28 ms | **16.1x** | >50 ranges |
| ACF Processing | 387 ms | 19 ms | **20.4x** | Long series |
| Grid Operations | 234 ms | 32 ms | **7.3x** | Large grids |
| Statistical Reduction | 521 ms | 42 ms | **12.4x** | All sizes |
| Convection Mapping | 1.2 s | 45 ms | **26.7x** | High-order |

### Detailed Results

#### FITACF v3.0

```
Data Size vs Speedup
─────────────────────────────────────────
Ranges    CPU(ms)    CUDA(ms)    Speedup
─────────────────────────────────────────
10        4.2        8.1         0.5x
25        12.3       9.4         1.3x
50        45.2       11.2        4.0x
75        98.7       14.5        6.8x
100       167.3      18.9        8.9x
150       312.4      23.1        13.5x
200       452.1      28.3        16.0x
─────────────────────────────────────────
```

**Observations:**
- CUDA overhead dominates for small datasets (<25 ranges)
- Break-even point: ~25 ranges
- Maximum speedup: ~16x at 200+ ranges

#### ACF Computation

```
Sample Count vs Speedup
─────────────────────────────────────────
Samples   CPU(ms)    CUDA(ms)    Speedup
─────────────────────────────────────────
100       2.1        4.2         0.5x
500       23.4       5.8         4.0x
1000      87.3       8.9         9.8x
5000      421.2      14.3        29.5x
10000     1687.4     28.7        58.8x
─────────────────────────────────────────
```

**Observations:**
- O(n²) CPU vs O(n) CUDA for correlation
- Break-even: ~200 samples
- Exceptional speedup for large datasets

#### Grid Processing

```
Cells vs Speedup
─────────────────────────────────────────
Cells     CPU(ms)    CUDA(ms)    Speedup
─────────────────────────────────────────
100       5.4        12.3        0.4x
500       28.7       15.4        1.9x
1000      67.2       18.9        3.6x
5000      312.4      28.7        10.9x
10000     687.3      42.1        16.3x
─────────────────────────────────────────
```

---

## Hardware Comparison

### GPU Configurations Tested

| GPU | Architecture | VRAM | Performance |
|-----|--------------|------|-------------|
| GTX 1080 | Pascal | 8 GB | 1.0x (baseline) |
| RTX 2080 | Turing | 8 GB | 1.4x |
| RTX 3080 | Ampere | 10 GB | 2.1x |
| RTX 4080 | Ada | 16 GB | 2.8x |
| Tesla V100 | Volta | 32 GB | 1.8x |
| A100 | Ampere | 40 GB | 3.2x |

### CPU Comparison

| CPU | Cores | Single-Thread | Multi-Thread |
|-----|-------|---------------|--------------|
| Intel i7-8700 | 6 | 1.0x | 4.2x |
| Intel i9-10900K | 10 | 1.2x | 7.1x |
| AMD Ryzen 9 5900X | 12 | 1.4x | 9.3x |
| AMD EPYC 7742 | 64 | 1.1x | 42.1x |

---

## Profiling

### Using NVIDIA Nsight

```bash
# Profile specific operation
nsys profile -o profile_output ./make_fit large_file.rawacf

# Analyze results
nsys stats profile_output.nsys-rep
```

### Key Metrics

| Metric | Target | Poor |
|--------|--------|------|
| GPU Utilization | >80% | <50% |
| Memory Bandwidth | >70% | <40% |
| Occupancy | >50% | <25% |
| Kernel Time | >90% total | <70% total |

### Profiling Output Example

```
Kernel Statistics:
═══════════════════════════════════════════════════════════════
Kernel                         Time(ms)   Calls   Avg(ms)   %
═══════════════════════════════════════════════════════════════
process_acf_kernel            12.34      1000    0.012    44.2%
fit_model_kernel               8.21      1000    0.008    29.4%
statistical_reduction_kernel   4.56       500    0.009    16.3%
memory_transfer                2.82        --       --     10.1%
═══════════════════════════════════════════════════════════════
```

---

## Optimization Guide

### Data Size Thresholds

Configure CUDA usage based on data size:

```c
// Recommended thresholds
#define CUDA_FITACF_THRESHOLD    25   // ranges
#define CUDA_ACF_THRESHOLD      200   // samples
#define CUDA_GRID_THRESHOLD     500   // cells
#define CUDA_MAP_THRESHOLD       50   // harmonics
```

### Memory Optimization

#### 1. Use Pinned Memory

```c
// Faster transfers with pinned memory
cudaHostAlloc(&host_ptr, size, cudaHostAllocDefault);
// ~2x faster than pageable memory
```

#### 2. Batch Transfers

```c
// BAD: Many small transfers
for (int i = 0; i < n; i++) {
    cudaMemcpy(d_ptr + i, h_ptr + i, 1, ...);  // Slow
}

// GOOD: Single large transfer
cudaMemcpy(d_ptr, h_ptr, n, cudaMemcpyHostToDevice);  // Fast
```

#### 3. Use Streams

```c
// Overlap transfer and compute
cudaMemcpyAsync(d_batch1, h_batch1, size, 
                cudaMemcpyHostToDevice, stream1);
kernel<<<blocks, threads, 0, stream2>>>(d_batch0);
```

### Kernel Optimization

#### 1. Optimal Block Size

```c
// Query device for optimal block size
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize,
    my_kernel, 0, 0
);
```

#### 2. Shared Memory Usage

```cuda
// Use shared memory for frequently accessed data
__shared__ float shared_data[256];

// Load once, use many times
shared_data[threadIdx.x] = global_data[idx];
__syncthreads();

// Multiple reads from shared memory (fast)
for (int i = 0; i < 10; i++) {
    result += shared_data[some_index];
}
```

#### 3. Memory Coalescing

```cuda
// BAD: Strided access
float val = data[threadIdx.x * stride];  // Uncoalesced

// GOOD: Sequential access
float val = data[blockIdx.x * blockDim.x + threadIdx.x];  // Coalesced
```

---

## Running Benchmarks

### Built-in Benchmark Suite

```bash
# Run all benchmarks
cd scripts
./comprehensive_cuda_performance.sh

# Compare CPU vs CUDA
python compare_performance.py --output results.csv

# Generate dashboard
python generate_performance_dashboard.py
```

### Custom Benchmark

```c
// benchmark_custom.c

#include <time.h>
#include "fitacf.h"

double benchmark_fitacf(int n_iterations) {
    // Load test data
    RawACF *data = load_test_data();
    
    clock_t start = clock();
    
    for (int i = 0; i < n_iterations; i++) {
        fitacf_process(data);
    }
    
    clock_t end = clock();
    
    return (double)(end - start) / CLOCKS_PER_SEC / n_iterations;
}

int main() {
    printf("Average time: %.3f ms\n", 
           benchmark_fitacf(100) * 1000);
    return 0;
}
```

---

## When NOT to Use CUDA

CUDA adds overhead. Avoid for:

1. **Small datasets** (<25 ranges, <200 samples)
2. **Single operations** (batch multiple operations instead)
3. **Memory-limited tasks** (VRAM exhaustion)
4. **Sequential algorithms** (inherently non-parallel)

### Quick Decision Guide

```
IF data_size < threshold:
    USE CPU
ELIF memory_required > GPU_VRAM:
    USE CPU
ELIF algorithm is sequential:
    USE CPU
ELSE:
    USE CUDA
```

---

## Troubleshooting Performance

### Symptom: No Speedup

**Causes:**
1. Data too small → Increase batch size
2. Memory transfer dominated → Use pinned memory
3. Low occupancy → Adjust block size

### Symptom: Slower Than CPU

**Causes:**
1. Below threshold → Increase `CUDA_THRESHOLD`
2. Memory bottleneck → Profile memory bandwidth
3. Kernel overhead → Batch operations

### Symptom: Variable Performance

**Causes:**
1. GPU throttling → Check temperatures
2. Other GPU processes → Use exclusive mode
3. Memory fragmentation → Restart application
