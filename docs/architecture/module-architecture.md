# Module Architecture

Detailed technical documentation for each CUDA-accelerated module.

## Module Overview

| Module | Purpose | CUDA Status | Performance |
|--------|---------|-------------|-------------|
| [fitacf_v3.0](#fitacf-v30) | ACF Fitting | ✅ Production | 8-16x |
| [lmfit_v2.0](#lmfit-v20) | Levenberg-Marquardt | ✅ Production | 3-8x |
| [grid.1.24](#grid124) | Spatial Gridding | ✅ Complete | 5-10x |
| [raw.1.22](#raw122) | Raw Data I/O | ✅ Complete | 3-7x |
| [scan.1.7](#scan17) | Scan Processing | ✅ Complete | 4-8x |
| [fit.1.35](#fit135) | Fit Processing | ✅ Complete | 3-6x |
| [acf.1.16](#acf116) | ACF Computation | ✅ Complete | 20-60x |
| [iq.1.7](#iq17) | IQ Processing | ✅ Complete | 8-25x |
| [cnvmap.1.17](#cnvmap117) | Convection Mapping | ✅ Complete | 10-100x |

---

## FITACF v3.0

### Purpose
Core fitting algorithm that extracts physical parameters (velocity, power, spectral width) from auto-correlation functions.

### Location
```
codebase/superdarn/src.lib/tk/fitacf_v3.0/
├── src/
│   ├── fitacf.c                    # Original CPU implementation
│   ├── cuda_kernels.cu             # CUDA kernels
│   ├── cuda_advanced_kernels.cu    # Advanced optimization kernels
│   ├── cuda_llist.cu              # CUDA linked list replacement
│   ├── cuda_cpu_bridge.c          # Backend selection
│   └── cuda_fitacf_optimizer.c    # High-level interface
├── include/
│   ├── fitacf.h
│   └── cuda_llist.h
├── tests/
└── makefile.cuda
```

### CUDA Kernels

| Kernel | Function | Parallelism |
|--------|----------|-------------|
| `process_acf_kernel` | ACF power/phase | Range × Lag |
| `fit_model_kernel` | Model fitting | Range |
| `noise_estimation_kernel` | Noise calculation | Range |
| `bad_lag_detection_kernel` | Quality filtering | Range × Lag |
| `statistical_reduction_kernel` | Stats computation | Parallel reduction |

### Data Flow

```
Input: RAWACF
    │
    ▼
┌─────────────────────────────────────────────┐
│ 1. Convert linked lists → CUDA arrays       │
│    RangeList → CudaRangeBatch               │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 2. Transfer to GPU                          │
│    cudaMemcpyHostToDevice                   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 3. Execute kernels (parallel)              │
│    - bad_lag_detection_kernel              │
│    - process_acf_kernel                    │
│    - fit_model_kernel                      │
│    - noise_estimation_kernel               │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 4. Transfer results back                    │
│    cudaMemcpyDeviceToHost                  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 5. Convert back to original format          │
│    CudaFitResults → FitACF structures       │
└─────────────────────────────────────────────┘
    │
    ▼
Output: FITACF
```

### Key Optimizations

1. **Batch Processing**: All ranges processed simultaneously
2. **Shared Memory**: Lag data cached in shared memory
3. **Coalesced Access**: SOA layout for memory efficiency
4. **Stream Overlap**: Data transfer overlaps computation

---

## LMFIT v2.0

### Purpose
Levenberg-Marquardt least-squares fitting for ACF model parameters.

### CUDA Implementation

The LM algorithm is iterative, but inner loops are parallelized:

```cuda
// Parallel Jacobian computation
__global__ void compute_jacobian_kernel(
    const float *params,
    const float *data,
    float *jacobian,
    int n_params, int n_data
) {
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= n_data) return;
    
    // Each thread computes one row of Jacobian
    for (int p = 0; p < n_params; p++) {
        jacobian[data_idx * n_params + p] = 
            compute_partial_derivative(params, data_idx, p);
    }
}
```

### Performance Notes
- Best for large datasets (>1000 points)
- Iteration overhead limits maximum speedup
- ~3-8x typical improvement

---

## Grid 1.24

### Purpose
Spatial data gridding and merging operations.

### Location
```
codebase/superdarn/src.lib/tk/grid.1.24/
├── src/
│   └── cuda/
│       └── grid.1.24_cuda.cu
└── include/
    └── grid.1.24_cuda.h
```

### CUDA Kernels

| Kernel | Function | Complexity |
|--------|----------|------------|
| `grid_locate_cell_kernel` | Find cell for point | O(1) vs O(n) |
| `grid_merge_data_kernel` | Merge overlapping data | Parallel |
| `grid_statistics_kernel` | Compute cell statistics | Reduction |
| `grid_sort_kernel` | Sort by location | Thrust |

### Key Innovation

**Cell Location**: Original O(n) linear search replaced with O(1) spatial hash:

```cuda
__device__ int locate_cell_fast(float lat, float lon, GridConfig *cfg) {
    // Direct calculation instead of search
    int lat_idx = (int)((lat - cfg->lat_min) / cfg->lat_step);
    int lon_idx = (int)((lon - cfg->lon_min) / cfg->lon_step);
    return lat_idx * cfg->n_lon + lon_idx;
}
```

---

## Raw 1.22

### Purpose
Raw IQ data processing and format conversion.

### CUDA Kernels

| Kernel | Purpose |
|--------|---------|
| `interleave_iq_kernel` | Combine I/Q channels |
| `deinterleave_iq_kernel` | Separate I/Q channels |
| `threshold_filter_kernel` | Remove below-threshold |
| `time_search_kernel` | Binary time search |

### Memory Pattern

```
Input IQ (interleaved):
[ I0 Q0 I1 Q1 I2 Q2 ... ]

Output (deinterleaved):
I: [ I0 I1 I2 ... ]
Q: [ Q0 Q1 Q2 ... ]
```

---

## Scan 1.7

### Purpose
Radar scan organization and beam management.

### CUDA Kernels

| Kernel | Purpose |
|--------|---------|
| `process_beam_kernel` | Parallel beam processing |
| `classify_scatter_kernel` | Ground/iono scatter |
| `validate_beam_kernel` | Quality checks |
| `range_statistics_kernel` | Per-range stats |

### Scatter Classification

ML-like parallel algorithm for scatter type:

```cuda
__device__ int classify_scatter(
    float velocity, float width, float power,
    float elevation
) {
    // Feature-based classification
    float features[4] = {velocity, width, power, elevation};
    
    // Decision boundaries (learned from data)
    if (fabsf(velocity) < GROUND_VEL_THRESH &&
        width < GROUND_WIDTH_THRESH) {
        return GROUND_SCATTER;
    }
    return IONOSPHERIC_SCATTER;
}
```

---

## Fit 1.35

### Purpose
FIT data structure management and CFIT conversion.

### CUDA Kernels

| Kernel | Purpose |
|--------|---------|
| `validate_ranges_kernel` | Quality validation |
| `fit_to_cfit_kernel` | Format conversion |
| `compute_elevation_kernel` | Elevation angles |
| `process_range_kernel` | Range processing |

---

## ACF 1.16

### Purpose
Auto-correlation function computation from IQ samples.

### Location
```
codebase/superdarn/src.lib/tk/acf.1.16/
├── src/
│   └── cuda_acf_kernels.cu
└── include/
    └── cuda_acf.h
```

### CUDA Kernels

| Kernel | Purpose | Notes |
|--------|---------|-------|
| `compute_acf_kernel` | Core ACF | Highest speedup |
| `power_spectrum_kernel` | Power calculation | |
| `bad_sample_kernel` | Quality detection | |
| `normalize_acf_kernel` | Normalization | |

### Algorithm

```cuda
__global__ void compute_acf_kernel(
    const float *i_samples,
    const float *q_samples,
    float *acf_real,
    float *acf_imag,
    int n_samples,
    int n_lags
) {
    int lag = blockIdx.x;
    int sample = threadIdx.x;
    
    extern __shared__ float sdata[];
    
    // Compute correlation for this lag
    float sum_real = 0, sum_imag = 0;
    
    for (int s = sample; s < n_samples - lag; s += blockDim.x) {
        float i1 = i_samples[s], q1 = q_samples[s];
        float i2 = i_samples[s + lag], q2 = q_samples[s + lag];
        
        // Complex multiplication: (i1 + jq1) * conj(i2 + jq2)
        sum_real += i1 * i2 + q1 * q2;
        sum_imag += q1 * i2 - i1 * q2;
    }
    
    // Parallel reduction in shared memory
    // ... (reduction code)
    
    if (sample == 0) {
        acf_real[lag] = sdata[0];
        acf_imag[lag] = sdata[blockDim.x];
    }
}
```

### Performance

ACF computation shows highest speedup (20-60x) because:
- O(n²) complexity in original
- Perfect parallelization across lags
- Efficient shared memory reduction

---

## IQ 1.7

### Purpose
IQ data manipulation and statistics.

### CUDA Kernels

| Kernel | Purpose |
|--------|---------|
| `time_convert_kernel` | Time format conversion |
| `array_copy_kernel` | Efficient bulk copy |
| `encode_decode_kernel` | Data encoding |
| `flatten_kernel` | Structure flattening |
| `statistics_kernel` | IQ statistics |

---

## CnvMap 1.17

### Purpose
Ionospheric convection map generation using spherical harmonics.

### Location
```
codebase/superdarn/src.lib/tk/cnvmap.1.17/
├── src/
│   └── cuda/
│       └── cnvmap_cuda.cu
└── include/
    └── cnvmap_cuda.h
```

### CUDA Kernels

| Kernel | Purpose |
|--------|---------|
| `legendre_kernel` | Legendre polynomial eval |
| `spherical_harmonic_kernel` | SH coefficient fitting |
| `velocity_matrix_kernel` | Velocity field construction |
| `potential_eval_kernel` | Potential evaluation |

### Mathematical Background

Spherical harmonic fitting is compute-intensive:

$$\Phi(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l} a_{lm} Y_l^m(\theta, \phi)$$

CUDA parallelizes over:
- Grid evaluation points (thousands)
- Harmonic terms (hundreds)

### Performance

Shows highest potential speedup (10-100x) for:
- Large grids
- High-order expansions
- Multiple time steps

---

## Adding New Modules

See [Migration Patterns](migration-patterns.md) for how to CUDA-enable additional modules.
