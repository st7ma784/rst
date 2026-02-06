# Data Structures: Old vs New

This document details the data structure transformations made to enable CUDA acceleration while maintaining API compatibility.

## Core Transformation: Linked Lists → Arrays

### The Problem

Original RST used linked lists extensively:

```c
// Original: Linked list node
struct RangeNode {
    RangeData data;
    struct RangeNode *next;  // Sequential dependency
};

// Traversal: O(n) sequential, unpredictable memory access
struct RangeNode *node = head;
while (node != NULL) {
    process(node->data);
    node = node->next;  // Pointer chase - bad for GPU
}
```

**Why this can't parallelize:**
- Each iteration depends on previous (pointer)
- Memory access is scattered (cache unfriendly)
- Cannot divide work among threads
- No way to predict memory locations

### The Solution

Replace with array + validity mask:

```c
// New: Array with validity mask
typedef struct {
    RangeData *data;      // Contiguous array
    bool *valid;          // Parallel validity mask
    int capacity;         // Maximum elements
    int count;            // Active elements
} CudaRangeArray;

// Traversal: O(1) per element, predictable access
#pragma omp parallel for  // Or CUDA kernel
for (int i = 0; i < array->count; i++) {
    if (array->valid[i]) {
        process(array->data[i]);  // Independent!
    }
}
```

## Transformation Patterns

### Pattern 1: Simple Linked List

**Before:**
```c
typedef struct ACFNode {
    int lag;
    float real, imag;
    struct ACFNode *next;
} ACFNode;

ACFNode *build_acf_list(RawData *raw) {
    ACFNode *head = NULL;
    for (int i = 0; i < raw->n_lags; i++) {
        ACFNode *node = malloc(sizeof(ACFNode));
        node->lag = i;
        node->real = raw->real[i];
        node->imag = raw->imag[i];
        node->next = head;
        head = node;
    }
    return head;
}
```

**After:**
```c
typedef struct {
    int *lag;            // Array of lags
    float *real;         // Array of real parts
    float *imag;         // Array of imaginary parts
    bool *valid;         // Validity mask
    int capacity;
    int count;
} CudaACFArray;

CudaACFArray *build_acf_array(RawData *raw) {
    CudaACFArray *arr = cuda_acf_alloc(raw->n_lags);
    
    // Bulk copy (can use cudaMemcpy)
    memcpy(arr->lag, raw->lag_indices, raw->n_lags * sizeof(int));
    memcpy(arr->real, raw->real, raw->n_lags * sizeof(float));
    memcpy(arr->imag, raw->imag, raw->n_lags * sizeof(float));
    
    // All valid initially
    memset(arr->valid, true, raw->n_lags * sizeof(bool));
    arr->count = raw->n_lags;
    
    return arr;
}
```

### Pattern 2: Dynamic Deletion

**Before (Linked List):**
```c
void remove_bad_lags(ACFNode **head, float threshold) {
    ACFNode *prev = NULL;
    ACFNode *curr = *head;
    
    while (curr != NULL) {
        ACFNode *next = curr->next;
        
        if (is_bad_lag(curr, threshold)) {
            // Remove from list
            if (prev == NULL) {
                *head = next;
            } else {
                prev->next = next;
            }
            free(curr);
        } else {
            prev = curr;
        }
        curr = next;
    }
}
```

**After (Validity Mask):**
```c
// CPU version
void remove_bad_lags_cpu(CudaACFArray *arr, float threshold) {
    for (int i = 0; i < arr->count; i++) {
        if (arr->valid[i] && is_bad_lag_value(arr, i, threshold)) {
            arr->valid[i] = false;  // Mark invalid, don't delete
        }
    }
}

// CUDA version
__global__ void remove_bad_lags_kernel(
    float *real, float *imag, bool *valid,
    float threshold, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && valid[i]) {
        float power = real[i] * real[i] + imag[i] * imag[i];
        if (power < threshold) {
            valid[i] = false;  // Parallel marking
        }
    }
}
```

### Pattern 3: Ordered Iteration

**Before:**
```c
typedef struct SortedNode {
    float key;
    DataType data;
    struct SortedNode *next;
} SortedNode;

void insert_sorted(SortedNode **head, SortedNode *new_node) {
    if (*head == NULL || (*head)->key > new_node->key) {
        new_node->next = *head;
        *head = new_node;
    } else {
        SortedNode *curr = *head;
        while (curr->next && curr->next->key < new_node->key) {
            curr = curr->next;
        }
        new_node->next = curr->next;
        curr->next = new_node;
    }
}
```

**After:**
```c
typedef struct {
    float *keys;
    DataType *data;
    int *sort_indices;   // Indirect sort
    bool *valid;
    int capacity;
    int count;
} CudaSortedArray;

// Use parallel sort (Thrust library)
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

void sort_array_cuda(CudaSortedArray *arr) {
    thrust::device_ptr<float> keys(arr->keys);
    thrust::device_ptr<int> indices(arr->sort_indices);
    
    // Initialize indices: 0, 1, 2, ...
    thrust::sequence(indices, indices + arr->count);
    
    // Sort by key (parallel)
    thrust::sort_by_key(keys, keys + arr->count, indices);
}
```

## Structure-of-Arrays vs Array-of-Structures

### Original (Array of Structures - AOS)

```c
typedef struct {
    float power;
    float velocity;
    float width;
    float error;
} FitResult;

FitResult results[N_RANGES];  // AOS layout

// Memory layout: p0 v0 w0 e0 | p1 v1 w1 e1 | p2 v2 w2 e2 | ...
//               └─ result 0 ─┘ └─ result 1 ─┘ └─ result 2 ─┘
```

**Problem:** When processing only powers, we still load velocity, width, error (wasted bandwidth).

### New (Structure of Arrays - SOA)

```c
typedef struct {
    float *power;      // [N_RANGES]
    float *velocity;   // [N_RANGES]
    float *width;      // [N_RANGES]
    float *error;      // [N_RANGES]
    int count;
} CudaFitResults;

// Memory layout:
// power:    p0 p1 p2 p3 p4 p5 p6 p7 ...  (contiguous)
// velocity: v0 v1 v2 v3 v4 v5 v6 v7 ...  (contiguous)
// width:    w0 w1 w2 w3 w4 w5 w6 w7 ...  (contiguous)
```

**Benefits:**
- Accessing powers loads only power data → better cache/bandwidth
- Coalesced memory access for GPU
- SIMD-friendly for CPU vectorization

### Conversion Functions

```c
// AOS to SOA conversion
void aos_to_soa(FitResult *aos, CudaFitResults *soa, int n) {
    for (int i = 0; i < n; i++) {
        soa->power[i] = aos[i].power;
        soa->velocity[i] = aos[i].velocity;
        soa->width[i] = aos[i].width;
        soa->error[i] = aos[i].error;
    }
    soa->count = n;
}

// SOA to AOS conversion (for API compatibility)
void soa_to_aos(CudaFitResults *soa, FitResult *aos, int n) {
    for (int i = 0; i < n; i++) {
        aos[i].power = soa->power[i];
        aos[i].velocity = soa->velocity[i];
        aos[i].width = soa->width[i];
        aos[i].error = soa->error[i];
    }
}
```

## Complete Example: Range Gate Processing

### Original Structure

```c
// Original linked list design
typedef struct RangeGate {
    int gate_number;
    
    // Nested linked list for ACF data
    struct ACFPoint {
        int lag;
        double real;
        double imag;
        struct ACFPoint *next;
    } *acf_head;
    
    // Nested linked list for XCF data  
    struct XCFPoint {
        int lag;
        double real;
        double imag;
        struct XCFPoint *next;
    } *xcf_head;
    
    // Fit results
    double power;
    double velocity;
    double width;
    
    struct RangeGate *next;
} RangeGate;
```

### CUDA-Compatible Structure

```c
// CUDA-friendly design
typedef struct {
    // Batch dimensions
    int n_ranges;
    int max_lags;
    
    // Range gate info (SOA)
    int *gate_numbers;           // [n_ranges]
    bool *range_valid;           // [n_ranges]
    
    // ACF data (2D SOA)
    double *acf_real;            // [n_ranges × max_lags]
    double *acf_imag;            // [n_ranges × max_lags]
    bool *acf_valid;             // [n_ranges × max_lags]
    int *acf_lag_counts;         // [n_ranges]
    
    // XCF data (2D SOA)
    double *xcf_real;            // [n_ranges × max_lags]
    double *xcf_imag;            // [n_ranges × max_lags]
    bool *xcf_valid;             // [n_ranges × max_lags]
    int *xcf_lag_counts;         // [n_ranges]
    
    // Fit results (SOA)
    double *power;               // [n_ranges]
    double *velocity;            // [n_ranges]
    double *width;               // [n_ranges]
    
    // Memory management
    cuda_unified_memory_t memory;
} CudaRangeGateBatch;

// Indexing helper
#define ACF_INDEX(range, lag, max_lags) ((range) * (max_lags) + (lag))
```

### Conversion Code

```c
// Convert original to CUDA format
CudaRangeGateBatch *convert_to_cuda(RangeGate *head) {
    // Count ranges and max lags
    int n_ranges = 0;
    int max_lags = 0;
    for (RangeGate *r = head; r != NULL; r = r->next) {
        n_ranges++;
        int lag_count = count_acf_points(r->acf_head);
        if (lag_count > max_lags) max_lags = lag_count;
    }
    
    // Allocate batch
    CudaRangeGateBatch *batch = cuda_batch_alloc(n_ranges, max_lags);
    
    // Copy data
    int range_idx = 0;
    for (RangeGate *r = head; r != NULL; r = r->next) {
        batch->gate_numbers[range_idx] = r->gate_number;
        batch->range_valid[range_idx] = true;
        
        // Copy ACF linked list to arrays
        int lag_idx = 0;
        for (ACFPoint *p = r->acf_head; p != NULL; p = p->next) {
            int idx = ACF_INDEX(range_idx, lag_idx, max_lags);
            batch->acf_real[idx] = p->real;
            batch->acf_imag[idx] = p->imag;
            batch->acf_valid[idx] = true;
            lag_idx++;
        }
        batch->acf_lag_counts[range_idx] = lag_idx;
        
        // Mark remaining lags as invalid
        for (; lag_idx < max_lags; lag_idx++) {
            batch->acf_valid[ACF_INDEX(range_idx, lag_idx, max_lags)] = false;
        }
        
        range_idx++;
    }
    
    return batch;
}
```

## Memory Layout Comparison

```
ORIGINAL (Linked List):
═══════════════════════

   head
    │
    ▼
┌────────┐    ┌────────┐    ┌────────┐
│Range 0 │───▶│Range 1 │───▶│Range 2 │───▶ NULL
│ gate=5 │    │ gate=6 │    │ gate=7 │
│ acf────┼─┐  │ acf────┼─┐  │ acf────┼─┐
└────────┘ │  └────────┘ │  └────────┘ │
           │             │             │
           ▼             ▼             ▼
        ┌─────┐       ┌─────┐       ┌─────┐
        │lag 0│──▶... │lag 0│──▶... │lag 0│──▶...
        └─────┘       └─────┘       └─────┘

Memory locations: SCATTERED (bad for cache, impossible for GPU)


CUDA (Array + Mask):
══════════════════════

gate_numbers:  [ 5 | 6 | 7 | ... ]     ← contiguous
range_valid:   [ T | T | T | ... ]     ← contiguous

acf_real:      [ r00 r01 r02 | r10 r11 r12 | r20 r21 r22 | ... ]
                └─ range 0 ─┘ └─ range 1 ─┘ └─ range 2 ─┘
               ← contiguous (perfect for coalesced GPU access)

acf_valid:     [ T   T   F  |  T   T   T  |  T   F   F  | ... ]
               ← validity mask replaces deletion
```

## Performance Impact

| Operation | Linked List | Array + Mask | Speedup |
|-----------|-------------|--------------|---------|
| Sequential access | O(n) | O(n) | 1x |
| Random access | O(n) | O(1) | n× |
| Deletion | O(1) | O(1)* | 1x |
| Insertion | O(1) | O(1)* | 1x |
| GPU transfer | Impossible | O(n) | ∞ |
| Parallel process | Impossible | O(1) | n× |

\* Amortized with occasional compaction

## Summary

### Key Transformations

1. **Linked List → Array + Validity Mask**
   - Enables parallel processing
   - Better memory locality
   - GPU-compatible

2. **AOS → SOA**
   - Better memory bandwidth
   - Coalesced GPU access
   - SIMD friendly

3. **Dynamic Allocation → Pre-allocated Pools**
   - Reduced allocation overhead
   - Better GPU memory management
   - Predictable memory usage

4. **Pointer-based → Index-based**
   - Works across CPU/GPU boundary
   - Serializable
   - Cache friendly
