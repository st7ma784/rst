# Migration Patterns

Guide for developers adding CUDA acceleration to RST modules.

## Overview

This document provides patterns and templates for migrating CPU-only modules to CUDA-accelerated implementations while maintaining backward compatibility.

## Migration Checklist

### Phase 1: Assessment

- [ ] Identify computational bottlenecks (profile the code)
- [ ] Find linked list usage patterns
- [ ] Analyze data parallelism opportunities
- [ ] Estimate potential speedup
- [ ] Check data dependencies

### Phase 2: Preparation

- [ ] Create `makefile.cuda` alongside existing `makefile`
- [ ] Create `include/*_cuda.h` header
- [ ] Design array-based data structures
- [ ] Plan CPU/CUDA bridge interface

### Phase 3: Implementation

- [ ] Implement array data structures
- [ ] Write conversion functions (linked list ↔ array)
- [ ] Implement CUDA kernels
- [ ] Create CPU-CUDA bridge
- [ ] Add automatic backend selection

### Phase 4: Validation

- [ ] Unit tests for kernels
- [ ] Numerical accuracy comparison
- [ ] Performance benchmarks
- [ ] Integration tests

## Quick Assessment

### Scoring Module Priority

```python
# Calculate migration priority score

def calculate_priority(module):
    score = 0
    
    # Computational intensity (40%)
    if module.has_nested_loops:
        score += 0.4 * 3  # High
    elif module.has_single_loops:
        score += 0.4 * 2  # Medium
    else:
        score += 0.4 * 1  # Low
    
    # Linked list usage (30%)
    if module.linked_list_count > 5:
        score += 0.3 * 3  # Heavy
    elif module.linked_list_count > 0:
        score += 0.3 * 2  # Some
    else:
        score += 0.3 * 1  # None
    
    # Data parallelism (20%)
    if module.independent_iterations:
        score += 0.2 * 3  # High
    elif module.partial_independence:
        score += 0.2 * 2  # Medium
    else:
        score += 0.2 * 1  # Low
    
    # Usage frequency (10%)
    if module.is_critical_path:
        score += 0.1 * 3  # Critical
    elif module.is_commonly_used:
        score += 0.1 * 2  # Common
    else:
        score += 0.1 * 1  # Rare
    
    return score  # Max 3.0
```

**Priority Levels:**
- 2.5-3.0: High priority, migrate first
- 2.0-2.5: Medium priority
- 1.0-2.0: Low priority, may not benefit

## Pattern Templates

### Template 1: Simple Kernel Migration

#### Original Code
```c
// module.c
void process_data(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = expensive_computation(data[i]);
    }
}
```

#### CUDA Kernel
```cuda
// cuda_module_kernels.cu

__device__ float expensive_computation_cuda(float x) {
    // Same computation, device version
    return result;
}

__global__ void process_data_kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = expensive_computation_cuda(data[i]);
    }
}
```

#### Bridge Code
```c
// cuda_module_bridge.c

void process_data(float *data, int n) {
    if (cuda_available() && n >= CUDA_THRESHOLD) {
        cuda_process_data(data, n);
    } else {
        cpu_process_data(data, n);
    }
}

static void cuda_process_data(float *data, int n) {
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (n + 255) / 256;
    process_data_kernel<<<blocks, 256>>>(d_data, n);
    
    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
```

### Template 2: Linked List Replacement

#### Original Code
```c
// Original linked list processing
typedef struct Node {
    float value;
    struct Node *next;
} Node;

float process_list(Node *head) {
    float sum = 0;
    for (Node *n = head; n != NULL; n = n->next) {
        if (is_valid(n->value)) {
            sum += process(n->value);
        }
    }
    return sum;
}
```

#### CUDA-Compatible Structure
```c
// cuda_module.h

typedef struct {
    float *values;      // Array of values
    bool *valid;        // Validity mask
    int capacity;       // Maximum elements
    int count;          // Current count
} CudaArray;

// Conversion functions
CudaArray *list_to_array(Node *head);
Node *array_to_list(CudaArray *arr);
```

#### Conversion Implementation
```c
// cuda_module_convert.c

CudaArray *list_to_array(Node *head) {
    // Count elements
    int count = 0;
    for (Node *n = head; n != NULL; n = n->next) count++;
    
    // Allocate array
    CudaArray *arr = malloc(sizeof(CudaArray));
    arr->values = malloc(count * sizeof(float));
    arr->valid = malloc(count * sizeof(bool));
    arr->capacity = count;
    arr->count = count;
    
    // Copy data
    int i = 0;
    for (Node *n = head; n != NULL; n = n->next) {
        arr->values[i] = n->value;
        arr->valid[i] = true;
        i++;
    }
    
    return arr;
}
```

#### CUDA Kernel
```cuda
// cuda_module_kernels.cu

__global__ void process_array_kernel(
    float *values, bool *valid, float *result, int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Load and process
    float val = 0;
    if (i < n && valid[i]) {
        val = process_cuda(values[i]);
    }
    sdata[tid] = val;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}
```

### Template 3: Makefile.cuda

```makefile
# makefile.cuda - CUDA build configuration

# CUDA settings
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= sm_75

# Compiler flags
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3
NVCC_FLAGS += -Xcompiler -fPIC
NVCC_FLAGS += -I$(CUDA_PATH)/include
NVCC_FLAGS += -I../../include

# Debug flags (optional)
ifdef DEBUG
NVCC_FLAGS += -g -G -DDEBUG
endif

# Source files
CUDA_SRCS = cuda_kernels.cu cuda_bridge.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

C_SRCS = module.c
C_OBJS = $(C_SRCS:.c=.o)

# Target
LIB = libmodule_cuda.a

# Rules
all: $(LIB)

$(LIB): $(CUDA_OBJS) $(C_OBJS)
	$(AR) rcs $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -I$(CUDA_PATH)/include -DUSE_CUDA -c $< -o $@

clean:
	rm -f $(CUDA_OBJS) $(C_OBJS) $(LIB)

test: $(LIB)
	cd tests && ./run_tests.sh

.PHONY: all clean test
```

### Template 4: Header File

```c
// module_cuda.h - CUDA interface header

#ifndef MODULE_CUDA_H
#define MODULE_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// CUDA availability check
int cuda_module_available(void);

// CUDA-compatible data structure
typedef struct {
    float *data;
    bool *valid;
    int count;
    int capacity;
} CudaModuleData;

// Memory management
CudaModuleData *cuda_module_alloc(int capacity);
void cuda_module_free(CudaModuleData *data);

// Data transfer
int cuda_module_to_device(CudaModuleData *data);
int cuda_module_to_host(CudaModuleData *data);

// Processing functions
int cuda_module_process(CudaModuleData *data);

// Conversion functions (for compatibility)
CudaModuleData *cuda_module_from_legacy(void *legacy_data);
void *cuda_module_to_legacy(CudaModuleData *data);

#ifdef __cplusplus
}
#endif

#endif // MODULE_CUDA_H
```

## Testing Patterns

### Numerical Accuracy Test

```c
// test_accuracy.c

int test_numerical_accuracy(void) {
    // Generate test data
    float *test_data = generate_test_data(1000);
    float *cpu_result = malloc(1000 * sizeof(float));
    float *cuda_result = malloc(1000 * sizeof(float));
    
    // Run CPU version
    memcpy(cpu_result, test_data, 1000 * sizeof(float));
    cpu_process(cpu_result, 1000);
    
    // Run CUDA version
    memcpy(cuda_result, test_data, 1000 * sizeof(float));
    cuda_process(cuda_result, 1000);
    
    // Compare results
    float max_diff = 0;
    for (int i = 0; i < 1000; i++) {
        float diff = fabsf(cpu_result[i] - cuda_result[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    // Accept < 0.01% relative error
    float tolerance = 0.0001f * max_value(cpu_result, 1000);
    
    printf("Max difference: %e (tolerance: %e)\n", max_diff, tolerance);
    
    return (max_diff <= tolerance) ? 0 : 1;
}
```

### Performance Benchmark

```c
// benchmark.c

void benchmark_module(void) {
    int sizes[] = {100, 1000, 10000, 100000};
    
    printf("Size\tCPU(ms)\tCUDA(ms)\tSpeedup\n");
    printf("────────────────────────────────────\n");
    
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        float *data = generate_test_data(n);
        
        // Benchmark CPU
        double cpu_start = get_time();
        for (int i = 0; i < 100; i++) {
            cpu_process(data, n);
        }
        double cpu_time = (get_time() - cpu_start) / 100;
        
        // Benchmark CUDA
        double cuda_start = get_time();
        for (int i = 0; i < 100; i++) {
            cuda_process(data, n);
        }
        double cuda_time = (get_time() - cuda_start) / 100;
        
        printf("%d\t%.2f\t%.2f\t%.1fx\n", 
               n, cpu_time * 1000, cuda_time * 1000,
               cpu_time / cuda_time);
        
        free(data);
    }
}
```

## Common Pitfalls

### 1. Race Conditions

**Problem:**
```cuda
// BAD: Multiple threads write to same location
__global__ void bad_kernel(int *counter) {
    *counter += 1;  // Race condition!
}
```

**Solution:**
```cuda
// GOOD: Use atomic operations
__global__ void good_kernel(int *counter) {
    atomicAdd(counter, 1);
}
```

### 2. Memory Coalescing

**Problem:**
```cuda
// BAD: Strided access
value = data[threadIdx.x * stride];
```

**Solution:**
```cuda
// GOOD: Coalesced access
value = data[blockIdx.x * blockDim.x + threadIdx.x];
```

### 3. Shared Memory Bank Conflicts

**Problem:**
```cuda
// BAD: All threads access same bank
shared[threadIdx.x * 32] = value;
```

**Solution:**
```cuda
// GOOD: Pad to avoid conflicts
shared[threadIdx.x * 33] = value;
```

### 4. Forgetting Synchronization

**Problem:**
```cuda
// BAD: No sync after shared memory write
shared[tid] = compute();
result = shared[other_tid];  // May read stale data
```

**Solution:**
```cuda
// GOOD: Synchronize threads
shared[tid] = compute();
__syncthreads();
result = shared[other_tid];
```

## Summary

### Migration Steps

1. **Profile** existing code to find bottlenecks
2. **Design** array-based data structures
3. **Implement** CUDA kernels for parallel operations
4. **Bridge** CPU and CUDA with automatic selection
5. **Test** numerical accuracy and performance
6. **Document** changes and usage

### Key Principles

- Preserve original API
- Automatic backend selection
- Graceful CPU fallback
- Numerical equivalence
- Comprehensive testing
