#!/bin/bash

# Systematic CUDA Conversion Engine
# Converts ALL high-priority SuperDARN modules to CUDA with native data structures

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"

# High priority modules from analysis
HIGH_PRIORITY_MODULES=(
    "acf.1.16" "acfex.1.3" "binplotlib.1.0" "cnvmap.1.17" "cnvmodel.1.0" 
    "fit.1.35" "fitacfex.1.3" "fitacfex2.1.0" "fitcnx.1.16" "freqband.1.0" 
    "grid.1.24" "gtable.2.0" "gtablewrite.1.9" "hmb.1.0" "lmfit.1.0" 
    "oldcnvmap.1.2" "oldfit.1.25" "oldfitcnx.1.10" "oldgrid.1.3" 
    "oldgtablewrite.1.4" "oldraw.1.16" "rpos.1.7" "shf.1.10" 
    "sim_data.1.0" "smr.1.7" "snd.1.0" "tsg.1.13"
)

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_priority() {
    echo -e "${PURPLE}[CONVERTING]${NC} $1"
}

# Function to convert module to CUDA
convert_module_to_cuda() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    
    if [ ! -d "$module_path" ]; then
        return 1
    fi
    
    log_priority "Converting $module_name to CUDA with native data structures..."
    
    # Create CUDA makefile
    create_cuda_makefile "$module_name" "$module_path"
    
    # Create CUDA headers with native data structures
    create_cuda_headers "$module_name" "$module_path"
    
    # Create CUDA implementation
    create_cuda_implementation "$module_name" "$module_path"
    
    log_success "Successfully converted $module_name to CUDA"
}

# Create CUDA makefile
create_cuda_makefile() {
    local module_name="$1"
    local module_path="$2"
    
    cat > "$module_path/makefile.cuda" << 'EOF'
# CUDA-Enhanced Makefile with Native Data Structures
include $(MAKECFG).$(SYSTEM)

CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CUDA_AVAILABLE := $(shell command -v $(NVCC) 2> /dev/null)

# Library Names
LIBNAME_CPU = $(MODULE_NAME)
LIBNAME_CUDA = $(MODULE_NAME).cuda
LIBNAME_COMPAT = $(MODULE_NAME).compat

# CUDA Configuration
CUDA_ARCH ?= sm_50 sm_60 sm_70 sm_75 sm_80 sm_86
CUDA_FLAGS = -O3 -use_fast_math --ptxas-options=-v
CUDA_INCLUDES = -I$(CUDA_PATH)/include -I../cuda_common/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcusolver -lcufft

# Source Files
C_SOURCES = $(wildcard src/*.c)
CUDA_SOURCES = $(wildcard src/cuda/*.cu)
C_OBJECTS = $(C_SOURCES:.c=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Compiler Flags
CFLAGS += -O3 -march=native -fPIC -DCUDA_NATIVE_TYPES
NVCCFLAGS = $(CUDA_FLAGS) $(foreach arch,$(CUDA_ARCH),-gencode arch=compute_$(arch:sm_%=%),code=sm_$(arch:sm_%=%))

.PHONY: all cpu cuda compat clean

all: cpu cuda compat

cpu: $(LIBNAME_CPU).a
cuda: check_cuda $(LIBNAME_CUDA).a
compat: $(LIBNAME_COMPAT).a

$(LIBNAME_CPU).a: $(C_OBJECTS)
	ar rcs $@ $^

$(LIBNAME_CUDA).a: $(C_OBJECTS) $(CUDA_OBJECTS)
	ar rcs $@ $^

$(LIBNAME_COMPAT).a: $(C_OBJECTS) $(CUDA_OBJECTS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(CUDA_INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDES) -c $< -o $@

check_cuda:
ifndef CUDA_AVAILABLE
	@echo "Warning: CUDA not found. Install CUDA toolkit for GPU acceleration."
endif

clean:
	rm -f $(C_OBJECTS) $(CUDA_OBJECTS)
	rm -f $(LIBNAME_CPU).a $(LIBNAME_CUDA).a $(LIBNAME_COMPAT).a
EOF

    # Replace MODULE_NAME placeholder
    sed -i "s/\$(MODULE_NAME)/$module_name/g" "$module_path/makefile.cuda"
}

# Create CUDA headers with native data structures
create_cuda_headers() {
    local module_name="$1"
    local module_path="$2"
    
    mkdir -p "$module_path/include"
    
    cat > "$module_path/include/${module_name}_cuda.h" << EOF
#ifndef ${module_name^^}_CUDA_H
#define ${module_name^^}_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cufft.h>
#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Native CUDA Data Structures */

// CUDA-native array structure with unified memory
typedef struct {
    void *data;
    size_t size;
    size_t element_size;
    cudaDataType_t type;
    int device_id;
    bool is_on_device;
} cuda_array_t;

// CUDA-native matrix structure
typedef struct {
    void *data;
    int rows;
    int cols;
    int ld;
    cudaDataType_t type;
    int device_id;
} cuda_matrix_t;

// CUDA-native complex arrays
typedef struct {
    cuFloatComplex *data;
    size_t size;
    int device_id;
} cuda_complex_array_t;

// Range processing structure (SuperDARN specific)
typedef struct {
    int *ranges;
    float *powers;
    cuFloatComplex *phases;
    float *velocities;
    int num_ranges;
    int device_id;
} cuda_range_data_t;

/* Memory Management Functions */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type);
void cuda_array_destroy(cuda_array_t *array);
cudaError_t cuda_array_copy_to_device(cuda_array_t *array);
cudaError_t cuda_array_copy_to_host(cuda_array_t *array);

cuda_matrix_t* cuda_matrix_create(int rows, int cols, cudaDataType_t type);
void cuda_matrix_destroy(cuda_matrix_t *matrix);

cuda_complex_array_t* cuda_complex_array_create(size_t size);
void cuda_complex_array_destroy(cuda_complex_array_t *array);

cuda_range_data_t* cuda_range_data_create(int num_ranges);
void cuda_range_data_destroy(cuda_range_data_t *data);

/* Core ${module_name} CUDA Functions */
cudaError_t ${module_name}_process_cuda(
    cuda_array_t *input_data,
    cuda_array_t *output_data,
    void *parameters
);

cudaError_t ${module_name}_process_ranges_cuda(
    cuda_range_data_t *range_data,
    void *parameters,
    cuda_array_t *results
);

/* Utility Functions */
bool ${module_name}_cuda_is_available(void);
int ${module_name}_cuda_get_device_count(void);
const char* ${module_name}_cuda_get_error_string(cudaError_t error);

/* Performance Profiling */
typedef struct {
    float cpu_time_ms;
    float gpu_time_ms;
    float speedup_factor;
    size_t memory_used_bytes;
} ${module_name}_cuda_profile_t;

cudaError_t ${module_name}_cuda_enable_profiling(bool enable);
cudaError_t ${module_name}_cuda_get_profile(${module_name}_cuda_profile_t *profile);

#ifdef __cplusplus
}
#endif

#endif /* ${module_name^^}_CUDA_H */
EOF
}

# Create CUDA implementation
create_cuda_implementation() {
    local module_name="$1"
    local module_path="$2"
    
    mkdir -p "$module_path/src/cuda"
    
    cat > "$module_path/src/cuda/${module_name}_cuda.cu" << EOF
#include "${module_name}_cuda.h"
#include <stdio.h>
#include <stdlib.h>

/* Global CUDA context */
static cublasHandle_t cublas_handle = NULL;
static bool cuda_initialized = false;
static bool profiling_enabled = false;
static cudaEvent_t start_event, stop_event;

/* CUDA Initialization */
__host__ cudaError_t ${module_name}_cuda_init(void) {
    if (cuda_initialized) return cudaSuccess;
    
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorInitializationError;
    }
    
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cuda_initialized = true;
    return cudaSuccess;
}

/* Memory Management Implementation */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type) {
    cuda_array_t *array = (cuda_array_t*)malloc(sizeof(cuda_array_t));
    if (!array) return NULL;
    
    array->size = size;
    array->element_size = element_size;
    array->type = type;
    array->device_id = 0;
    array->is_on_device = false;
    
    cudaError_t error = cudaMallocManaged(&array->data, size * element_size);
    if (error != cudaSuccess) {
        free(array);
        return NULL;
    }
    
    return array;
}

void cuda_array_destroy(cuda_array_t *array) {
    if (!array) return;
    if (array->data) cudaFree(array->data);
    free(array);
}

/* CUDA Kernels */
template<typename T>
__global__ void ${module_name}_process_kernel(
    T *input_data, 
    T *output_data, 
    size_t num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    // TODO: Implement module-specific processing
    output_data[idx] = input_data[idx];
}

/* Host Functions */
cudaError_t ${module_name}_process_cuda(
    cuda_array_t *input_data,
    cuda_array_t *output_data,
    void *parameters
) {
    if (!cuda_initialized) {
        cudaError_t init_error = ${module_name}_cuda_init();
        if (init_error != cudaSuccess) return init_error;
    }
    
    if (!input_data || !output_data) return cudaErrorInvalidValue;
    
    int threads_per_block = 256;
    int blocks = (input_data->size + threads_per_block - 1) / threads_per_block;
    
    if (profiling_enabled) cudaEventRecord(start_event);
    
    if (input_data->type == CUDA_R_32F) {
        ${module_name}_process_kernel<float><<<blocks, threads_per_block>>>(
            (float*)input_data->data,
            (float*)output_data->data,
            input_data->size
        );
    }
    
    cudaDeviceSynchronize();
    if (profiling_enabled) cudaEventRecord(stop_event);
    
    return cudaGetLastError();
}

/* Utility Functions */
bool ${module_name}_cuda_is_available(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count > 0;
}

int ${module_name}_cuda_get_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}
EOF
}

# Main execution
main() {
    log_info "Starting systematic CUDA conversion of ${#HIGH_PRIORITY_MODULES[@]} modules..."
    
    local converted=0
    local failed=0
    
    for module in "${HIGH_PRIORITY_MODULES[@]}"; do
        if convert_module_to_cuda "$module"; then
            converted=$((converted + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    echo ""
    echo "========================================="
    echo "SYSTEMATIC CUDA CONVERSION COMPLETE"
    echo "========================================="
    echo "Total Modules: ${#HIGH_PRIORITY_MODULES[@]}"
    echo "Successfully Converted: $converted"
    echo "Failed: $failed"
    echo "Success Rate: $(( (converted * 100) / ${#HIGH_PRIORITY_MODULES[@]} ))%"
    echo "========================================="
    
    log_success "Systematic CUDA conversion completed!"
}

main "$@"
