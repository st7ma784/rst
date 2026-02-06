# RST SuperDARN CUDA Architecture Design

## Executive Summary

This document defines a **unified CUDA-compatible architecture** for the RST SuperDARN toolkit that preserves existing compilation structures while enabling systematic migration to GPU acceleration. The design leverages proven patterns from FITACF v3.0 and extends them across all modules.

## Architecture Principles

### 1. **Compilation Structure Preservation**
- **Maintain existing Makefile hierarchy**: Top-level → Common → Module-specific
- **Dual compilation support**: Standard `makefile` + `makefile.cuda` per module
- **Backward compatibility**: All CPU interfaces remain unchanged
- **Runtime selection**: Automatic CPU/CUDA selection based on hardware and data size

### 2. **Unified Data Structure Migration**
- **Standardized replacement pattern**: Linked lists → Array + Validity Mask structures
- **Memory management abstraction**: Unified CPU/GPU memory handling
- **Interface compatibility**: Same function signatures with internal CUDA acceleration

### 3. **Performance and Scalability**
- **Batch processing**: Process multiple range gates/beams simultaneously
- **Memory coalescing**: GPU-optimized memory access patterns
- **Asynchronous operations**: Overlap computation with data transfer

## Core Architecture Components

### A. Unified Memory Management Layer

```c
// /include/cuda_common/cuda_memory_manager.h

typedef struct {
    void *host_ptr;           // Host memory pointer
    void *device_ptr;         // Device memory pointer  
    size_t size;              // Size in bytes
    cuda_memory_type_t type;  // MANAGED, HOST_PINNED, DEVICE_ONLY
    bool host_valid;          // Host data is current
    bool device_valid;        // Device data is current
    int device_id;            // Target CUDA device
} cuda_unified_memory_t;

typedef enum {
    CUDA_MEMORY_MANAGED,      // CUDA managed memory (automatic)
    CUDA_MEMORY_HOST_PINNED,  // Pinned host memory for fast transfer
    CUDA_MEMORY_DEVICE_ONLY   // Device-only memory
} cuda_memory_type_t;

// Unified memory allocation
cuda_error_t cuda_memory_alloc(cuda_unified_memory_t *mem, size_t size, cuda_memory_type_t type);
cuda_error_t cuda_memory_free(cuda_unified_memory_t *mem);
cuda_error_t cuda_memory_sync_to_device(cuda_unified_memory_t *mem);
cuda_error_t cuda_memory_sync_to_host(cuda_unified_memory_t *mem);
```

### B. Standardized Data Structure Layer

```c
// /include/cuda_common/cuda_data_structures.h

// CUDA-compatible linked list replacement
typedef struct {
    cuda_unified_memory_t data_memory;    // Array of data elements
    cuda_unified_memory_t mask_memory;    // Boolean validity mask
    size_t element_size;                  // Size of each data element
    size_t capacity;                      // Maximum number of elements
    size_t current_size;                  // Current valid elements
    size_t iterator_pos;                  // Current iteration position
    bool is_sorted;                       // Optimization flag
    cuda_list_compare_func_t compare_fn;  // Comparison function pointer
} cuda_list_t;

// CUDA-compatible matrix structure
typedef struct {
    cuda_unified_memory_t data_memory;
    size_t element_size;
    size_t rows, cols;
    size_t row_stride;                    // For memory alignment
    cuda_matrix_layout_t layout;          // ROW_MAJOR, COL_MAJOR
} cuda_matrix_t;

// SuperDARN-specific range gate data
typedef struct {
    int range_gate_id;
    cuda_list_t acf_data;                 // ACF measurements
    cuda_list_t xcf_data;                 // XCF measurements  
    cuda_list_t power_data;               // Power measurements
    cuda_list_t phase_data;               // Phase measurements
    float noise_level;
    float quality_metric;
    bool is_valid;
} cuda_range_gate_t;

// Batch processing structure
typedef struct {
    cuda_range_gate_t *range_gates;
    size_t num_range_gates;
    cuda_unified_memory_t batch_memory;
    cuda_processing_config_t config;
} cuda_batch_processor_t;
```

### C. Module Interface Standardization

```c
// /include/cuda_common/cuda_module_interface.h

// Standard module interface pattern
typedef struct {
    const char *module_name;
    int major_version;
    int minor_version;
    cuda_capability_flags_t required_capabilities;
    
    // Function pointers for CPU/CUDA implementations
    int (*process_cpu)(void *input, void *output, void *params);
    int (*process_cuda)(void *input, void *output, void *params);
    int (*validate_parameters)(void *params);
    
    // Memory requirement estimation
    size_t (*estimate_memory_required)(void *params);
    bool (*should_use_cuda)(void *params);
} cuda_module_interface_t;

// Runtime compute mode selection
typedef enum {
    COMPUTE_MODE_AUTO,        // Automatic selection based on data size/hardware
    COMPUTE_MODE_CPU_ONLY,    // Force CPU implementation
    COMPUTE_MODE_CUDA_ONLY,   // Force CUDA implementation  
    COMPUTE_MODE_CPU_FALLBACK // Try CUDA, fallback to CPU on error
} cuda_compute_mode_t;

// Global configuration
typedef struct {
    cuda_compute_mode_t default_compute_mode;
    size_t cuda_threshold_elements;       // Minimum elements to justify CUDA
    float cuda_memory_limit_fraction;     // Maximum GPU memory to use (0.0-1.0)
    bool enable_performance_profiling;
    bool enable_result_validation;
    float validation_tolerance;
} cuda_global_config_t;
```

## Module-Specific Migration Patterns

### 1. FITACF Module Family

**Current State**: FITACF v3.0 has complete CUDA implementation
**Action**: Extend pattern to FITACF v2.5 and FITACFEX variants

```c
// fitacf_unified.h - New unified interface
typedef struct {
    cuda_module_interface_t base;
    
    // FITACF-specific processing functions
    int (*process_acf_cuda)(cuda_list_t *acf_data, fitacf_params_t *params, fitacf_results_t *results);
    int (*process_xcf_cuda)(cuda_list_t *xcf_data, fitacf_params_t *params, fitacf_results_t *results);
    int (*estimate_noise_cuda)(cuda_batch_processor_t *batch, float *noise_estimate);
    
    // Batch processing capabilities
    int (*process_range_gates_batch)(cuda_batch_processor_t *batch, fitacf_batch_results_t *results);
    
} fitacf_cuda_interface_t;

// Backward compatible wrapper
int Fitacf(FITPRMS *fit_prms, struct FitData *fit_data, int elv_version) {
    // Auto-detect best implementation
    if (cuda_should_use_gpu(fit_prms->nrang * fit_prms->mplgs)) {
        return Fitacf_CUDA(fit_prms, fit_data, elv_version);
    } else {
        return Fitacf_CPU(fit_prms, fit_data, elv_version);
    }
}
```

### 2. LMFIT Module Family

**Current State**: Uses extensive linked lists for optimization data
**Action**: Migrate to CUDA-compatible array structures

```c
// lmfit_unified.h
typedef struct {
    cuda_module_interface_t base;
    
    // Convert existing RANGENODE linked lists to CUDA arrays
    cuda_list_t acf_measurements;        // Replaces llist acf
    cuda_list_t xcf_measurements;        // Replaces llist xcf  
    cuda_list_t power_measurements;      // Replaces llist pwrs
    cuda_list_t phase_measurements;      // Replaces llist phases
    
    // CUDA-accelerated fitting functions
    int (*levenberg_marquardt_cuda)(cuda_matrix_t *jacobian, cuda_list_t *residuals, 
                                   lmfit_params_t *params, lmfit_results_t *results);
    int (*compute_jacobian_cuda)(cuda_list_t *data, cuda_matrix_t *jacobian);
    int (*apply_constraints_cuda)(cuda_list_t *data, lmfit_constraints_t *constraints);
    
} lmfit_cuda_interface_t;

// Migration strategy for RANGENODE
typedef struct {
    int range;
    cuda_unified_memory_t SC_pow_memory;
    double refrc_idx;
    
    // Migrated linked lists to CUDA arrays
    cuda_list_t acf;        // Was: llist acf
    cuda_list_t xcf;        // Was: llist xcf
    cuda_list_t scpwr;      // Was: llist scpwr
    cuda_list_t phases;     // Was: llist phases
    cuda_list_t pwrs;       // Was: llist pwrs
    cuda_list_t elev;       // Was: llist elev
    
    // CUDA-optimized fitting data
    LMFITDATA_CUDA* l_acf_fit;   // GPU-optimized version
    LMFITDATA_CUDA* q_acf_fit;   // GPU-optimized version
    LMFITDATA_CUDA* l_xcf_fit;   // GPU-optimized version  
    LMFITDATA_CUDA* q_xcf_fit;   // GPU-optimized version
    
    double prev_pow, prev_phase, prev_width;
} RANGENODE_CUDA;
```

### 3. Grid Processing Module

**Current State**: Grid v1.24 has partial CUDA optimization  
**Action**: Complete migration and standardize

```c
// grid_unified.h
typedef struct {
    cuda_module_interface_t base;
    
    // Grid-specific CUDA operations
    int (*interpolate_spatial_cuda)(cuda_matrix_t *input_grid, cuda_matrix_t *output_grid,
                                   grid_params_t *params);
    int (*apply_median_filter_cuda)(cuda_matrix_t *grid, grid_filter_params_t *params);
    int (*compute_statistics_cuda)(cuda_matrix_t *grid, grid_statistics_t *stats);
    
    // Batch processing for multiple time steps
    int (*process_time_series_cuda)(cuda_matrix_t *time_series_grids, 
                                   grid_time_params_t *params, grid_results_t *results);
    
} grid_cuda_interface_t;
```

## Build System Integration

### Enhanced Makefile Structure

```makefile
# /codebase/superdarn/build/cuda_module.mk
# Enhanced module makefile template with CUDA support

include $(TOPDIR)/build/common.mk

# CUDA-specific variables
CUDA_ARCH ?= sm_70,sm_75,sm_80  # Support recent GPU architectures
CUDA_LIBS := -lcudart -lcublas -lcusolver -lcufft
CUDA_INCLUDES := -I$(CUDA_PATH)/include -I$(TOPDIR)/include/cuda_common

# Detect CUDA availability
HAS_CUDA := $(shell which nvcc >/dev/null 2>&1 && echo 1 || echo 0)

# Source file detection
C_SRCS := $(wildcard $(SRC_DIR)/*.c)
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp) 
CUDA_SRCS := $(wildcard $(SRC_DIR)/*.cu)

# Object files
C_OBJS := $(C_SRCS:$(SRC_DIR)/%.c=$(BUILDDIR)/%.o)
CPP_OBJS := $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(BUILDDIR)/%.o)
CUDA_OBJS := $(CUDA_SRCS:$(SRC_DIR)/%.cu=$(BUILDDIR)/%.cu.o)

# Libraries to build
STATIC_LIB := $(LIBDIR)/lib$(MODULE_NAME).a
ifeq ($(HAS_CUDA),1)
    CUDA_LIB := $(LIBDIR)/lib$(MODULE_NAME).cuda.a
    UNIFIED_LIB := $(LIBDIR)/lib$(MODULE_NAME).unified.a
endif

# Default target
all: $(STATIC_LIB) $(if $(HAS_CUDA),$(CUDA_LIB) $(UNIFIED_LIB))

# Static library (CPU only)
$(STATIC_LIB): $(C_OBJS) $(CPP_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

# CUDA library (GPU acceleration)
ifneq ($(strip $(CUDA_SRCS)),)
$(CUDA_LIB): $(CUDA_OBJS)
	@mkdir -p $(@D) 
	$(AR) rcs $@ $^
endif

# Unified library (CPU + CUDA)
$(UNIFIED_LIB): $(C_OBJS) $(CPP_OBJS) $(CUDA_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

# CUDA compilation rules
$(BUILDDIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $(CUDA_INCLUDES) -c $< -o $@

# Enhanced clean target
clean:
	$(RM) -r $(BUILDDIR) $(STATIC_LIB) $(CUDA_LIB) $(UNIFIED_LIB)

.PHONY: all clean cuda-info

# CUDA capability detection
cuda-info:
	@echo "CUDA Detection Results:"
	@echo "  NVCC Available: $(HAS_CUDA)"
	@if [ "$(HAS_CUDA)" = "1" ]; then \
	    echo "  NVCC Version: $$(nvcc --version | grep release)"; \
	    echo "  CUDA Path: $(CUDA_PATH)"; \
	    echo "  Target Architectures: $(CUDA_ARCH)"; \
	fi
```

### Module-Specific Makefile Example

```makefile
# /codebase/superdarn/src.lib/tk/lmfit_v2.0/makefile.cuda
# LMFIT v2.0 CUDA-enabled makefile

MODULE_NAME := lmfit_v2.0
MODULE_VERSION := 2.0

# Include enhanced CUDA module template
include $(TOPDIR)/build/cuda_module.mk

# Module-specific CUDA flags
NVCCFLAGS += -DLMFIT_CUDA_ENABLED -DLMFIT_VERSION_MAJOR=2 -DLMFIT_VERSION_MINOR=0

# Module-specific includes
INCLUDES += -I$(TOPDIR)/include/superdarn -I./include

# Module-specific libraries
LDLIBS += -lm -lpthread $(CUDA_LIBS)

# Custom CUDA kernel compilation for complex kernels
$(BUILDDIR)/lmfit_optimization_kernels.cu.o: $(SRC_DIR)/lmfit_optimization_kernels.cu
	$(NVCC) $(NVCCFLAGS) -arch=sm_70,sm_75,sm_80 -maxrregcount=64 $(INCLUDES) -c $< -o $@

# Performance testing target
test-performance: $(UNIFIED_LIB)
	@echo "Running LMFIT v2.0 CUDA performance tests..."
	@./test/lmfit_cuda_benchmark --validate --profile
```

## Validation and Testing Framework

### Automated Result Validation

```c
// /include/cuda_common/cuda_validation.h

typedef struct {
    float tolerance_absolute;
    float tolerance_relative;
    size_t max_differences_reported;
    bool enable_statistical_comparison;
} cuda_validation_config_t;

typedef struct {
    size_t total_elements;
    size_t different_elements;
    float max_absolute_error;
    float max_relative_error;
    float mean_absolute_error;
    float rms_error;
    bool validation_passed;
} cuda_validation_results_t;

// Cross-validate CUDA results against CPU baseline
cuda_validation_results_t cuda_validate_results(
    void *cuda_results, void *cpu_results, size_t element_count,
    size_t element_size, cuda_validation_config_t *config
);

// Module-specific validation functions
int fitacf_validate_cuda_results(struct FitData *cuda_fit, struct FitData *cpu_fit);
int lmfit_validate_cuda_results(LMFITDATA *cuda_results, LMFITDATA *cpu_results);
int grid_validate_cuda_results(GridData *cuda_grid, GridData *cpu_grid);
```

### Performance Benchmarking

```c
// /include/cuda_common/cuda_benchmark.h

typedef struct {
    const char *test_name;
    size_t input_size;
    float cpu_time_ms;
    float cuda_time_ms;
    float speedup_factor;
    size_t memory_used_bytes;
    bool results_validated;
} cuda_benchmark_result_t;

// Comprehensive benchmarking suite
int cuda_run_module_benchmark(cuda_module_interface_t *module, 
                             void *test_data, cuda_benchmark_result_t *results);

// System-wide performance profiling
typedef struct {
    cuda_benchmark_result_t fitacf_results;
    cuda_benchmark_result_t lmfit_results;
    cuda_benchmark_result_t grid_results;
    cuda_benchmark_result_t overall_pipeline;
} cuda_system_benchmark_t;

int cuda_run_system_benchmark(cuda_system_benchmark_t *results);
```

## Migration Implementation Plan

### Phase 1: Infrastructure Setup (Weeks 1-2)
1. **Create unified header structure** in `/include/cuda_common/`
2. **Implement memory management layer** with unified allocation/synchronization
3. **Setup enhanced build system** with CUDA detection and dual compilation
4. **Create validation and benchmarking framework**

### Phase 2: Complete Existing Modules (Weeks 3-6) 
1. **LMFIT v2.0**: Migrate linked lists to CUDA arrays, implement Levenberg-Marquardt on GPU
2. **Grid v1.24**: Complete partial CUDA implementation, add batch processing
3. **ACF v1.16**: Finish auto-correlation CUDA kernels

### Phase 3: Create Unified Interface Layer (Weeks 7-8)
1. **Implement runtime CPU/CUDA selection** logic
2. **Create backward-compatible wrappers** for all existing function signatures  
3. **Add comprehensive testing suite** with automated validation
4. **Performance optimization** and memory usage tuning

### Phase 4: Integration and Validation (Weeks 9-10)
1. **End-to-end pipeline testing** with real SuperDARN data
2. **Cross-validation** of all CUDA implementations against CPU baselines
3. **Performance benchmarking** and optimization
4. **Documentation** and deployment preparation

## Expected Performance Improvements

Based on FITACF v3.0 results and projected improvements:

| Module | Current Speedup | Target Speedup | Bottleneck |
|--------|----------------|----------------|------------|
| FITACF v3.0 | 5-16x | 10-20x | Memory bandwidth |
| LMFIT v2.0 | None | 8-15x | Matrix operations |
| Grid v1.24 | 2-3x | 6-12x | Spatial interpolation |
| ACF v1.16 | None | 10-25x | FFT operations |
| **Overall Pipeline** | **3-5x** | **8-15x** | **Data transfers** |

## Risk Mitigation

### Technical Risks
- **Memory limitations**: Implement streaming for large datasets
- **CUDA compatibility**: Support graceful fallback to CPU implementations
- **Numerical precision**: Extensive validation with configurable tolerances

### Integration Risks  
- **Build system complexity**: Comprehensive testing on multiple environments
- **API compatibility**: Maintain 100% backward compatibility with existing interfaces
- **Performance regression**: Automated benchmarking in CI/CD pipeline

This architecture provides a **systematic, proven approach** to completing the CUDA migration while preserving all existing functionality and build processes.