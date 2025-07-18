/*
 * CUDA-Compatible Linked List Implementation
 * 
 * This file is part of the Radar Software Toolkit (RST).
 * 
 * Implements GPU-friendly array-based structures with validity masks
 * to replace traditional linked lists for CUDA acceleration.
 * 
 * Author: CUDA Conversion Project
 * Date: 2025
 */

#include "llist_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

/* === Helper Functions === */

/**
 * @brief Calculate number of uint32_t elements needed for bitmask
 */
static inline uint32_t mask_size_for_capacity(uint32_t capacity) {
    return (capacity + 31) / 32;  // Round up to nearest 32
}

/**
 * @brief Find next valid element starting from index
 */
CUDA_CALLABLE static uint32_t find_next_valid(const uint32_t* mask, uint32_t start, uint32_t capacity) {
    for (uint32_t i = start; i < capacity; i++) {
        if (llist_cuda_is_valid(mask, i)) {
            return i;
        }
    }
    return capacity; // Not found
}

/* === Core Implementation === */

CUDA_CALLABLE llist_cuda_t* llist_cuda_create(
    llist_cuda_comparator comp_func,
    llist_cuda_equal equal_func,
    uint32_t initial_capacity
) {
    if (initial_capacity == 0) {
        initial_capacity = 64; // Default capacity
    }
    if (initial_capacity > LLIST_CUDA_MAX_CAPACITY) {
        initial_capacity = LLIST_CUDA_MAX_CAPACITY;
    }

    llist_cuda_t* list = (llist_cuda_t*)malloc(sizeof(llist_cuda_t));
    if (!list) return NULL;

    // Initialize data arrays
    list->data = (void**)calloc(initial_capacity, sizeof(void*));
    list->valid_mask = (uint32_t*)calloc(mask_size_for_capacity(initial_capacity), sizeof(uint32_t));
    list->indices = (uint32_t*)malloc(initial_capacity * sizeof(uint32_t));

    if (!list->data || !list->valid_mask || !list->indices) {
        free(list->data);
        free(list->valid_mask);
        free(list->indices);
        free(list);
        return NULL;
    }

    // Initialize indices array (0, 1, 2, ...)
    for (uint32_t i = 0; i < initial_capacity; i++) {
        list->indices[i] = i;
    }

    // Initialize metadata
    list->capacity = initial_capacity;
    list->count = 0;
    list->allocated_count = 0;
    list->iter_pos = 0;
    list->comp_func = comp_func;
    list->equal_func = equal_func;
    list->is_gpu_allocated = false;
    list->gpu_data = NULL;
    list->gpu_valid_mask = NULL;
    list->gpu_indices = NULL;

    return list;
}

CUDA_CALLABLE void llist_cuda_destroy(
    llist_cuda_t* list,
    bool destroy_nodes,
    llist_cuda_node_func destructor
) {
    if (!list) return;

    // Destroy node data if requested
    if (destroy_nodes) {
        for (uint32_t i = 0; i < list->capacity; i++) {
            if (llist_cuda_is_valid(list->valid_mask, i) && list->data[i]) {
                if (destructor) {
                    destructor(list->data[i]);
                } else {
                    free(list->data[i]);
                }
            }
        }
    }

    // Free GPU memory if allocated
#ifdef __CUDACC__
    if (list->is_gpu_allocated) {
        cudaFree(list->gpu_data);
        cudaFree(list->gpu_valid_mask);
        cudaFree(list->gpu_indices);
    }
#endif

    // Free host memory
    free(list->data);
    free(list->valid_mask);
    free(list->indices);
    free(list);
}

CUDA_CALLABLE int llist_cuda_add_node(
    llist_cuda_t* list,
    void* node,
    int flags
) {
    if (!list || !node) return LLIST_CUDA_NULL_ARGUMENT;

    // Find first available slot
    uint32_t insert_index = list->capacity;
    for (uint32_t i = 0; i < list->capacity; i++) {
        if (!llist_cuda_is_valid(list->valid_mask, i)) {
            insert_index = i;
            break;
        }
    }

    if (insert_index >= list->capacity) {
        return LLIST_CUDA_CAPACITY_EXCEEDED;
    }

    // Add the node
    list->data[insert_index] = node;
    llist_cuda_set_valid(list->valid_mask, insert_index);
    list->count++;
    if (insert_index >= list->allocated_count) {
        list->allocated_count = insert_index + 1;
    }

    // Handle ordering for front/rear insertion
    if (flags & LLIST_CUDA_ADD_FRONT) {
        // Move all indices forward and insert at beginning
        for (uint32_t i = list->count; i > 0; i--) {
            list->indices[i] = list->indices[i-1];
        }
        list->indices[0] = insert_index;
    } else {
        // Add to rear - find position in indices array
        uint32_t pos = 0;
        for (uint32_t i = 0; i < list->count - 1; i++) {
            if (llist_cuda_is_valid(list->valid_mask, list->indices[i])) {
                pos++;
            }
        }
        list->indices[pos] = insert_index;
    }

    return LLIST_CUDA_SUCCESS;
}

CUDA_CALLABLE int llist_cuda_delete_node(
    llist_cuda_t* list,
    void* node,
    bool destroy_node,
    llist_cuda_node_func destructor
) {
    if (!list || !node) return LLIST_CUDA_NULL_ARGUMENT;

    // Find the node
    uint32_t found_index = list->capacity;
    for (uint32_t i = 0; i < list->allocated_count; i++) {
        if (llist_cuda_is_valid(list->valid_mask, i) && 
            list->data[i] == node) {
            found_index = i;
            break;
        }
    }

    if (found_index >= list->capacity) {
        return LLIST_CUDA_NODE_NOT_FOUND;
    }

    // Mark as invalid (mask-based deletion)
    llist_cuda_clear_valid(list->valid_mask, found_index);
    list->count--;

    // Destroy node data if requested
    if (destroy_node && list->data[found_index]) {
        if (destructor) {
            destructor(list->data[found_index]);
        } else {
            free(list->data[found_index]);
        }
    }

    list->data[found_index] = NULL;
    return LLIST_CUDA_SUCCESS;
}

CUDA_CALLABLE int llist_cuda_find_node(
    llist_cuda_t* list,
    void* data,
    void** found
) {
    if (!list || !data || !found) return LLIST_CUDA_NULL_ARGUMENT;
    if (!list->equal_func) return LLIST_CUDA_EQUAL_MISSING;

    for (uint32_t i = 0; i < list->allocated_count; i++) {
        if (llist_cuda_is_valid(list->valid_mask, i) && 
            list->data[i] && 
            list->equal_func(list->data[i], data)) {
            *found = list->data[i];
            return LLIST_CUDA_SUCCESS;
        }
    }

    return LLIST_CUDA_NODE_NOT_FOUND;
}

CUDA_CALLABLE uint32_t llist_cuda_size(const llist_cuda_t* list) {
    return list ? list->count : 0;
}

CUDA_CALLABLE bool llist_cuda_is_empty(const llist_cuda_t* list) {
    return list ? (list->count == 0) : true;
}

CUDA_CALLABLE void* llist_cuda_get_at_index(
    const llist_cuda_t* list,
    uint32_t index
) {
    if (!list || index >= list->count) return NULL;

    // Find the nth valid element
    uint32_t valid_count = 0;
    for (uint32_t i = 0; i < list->allocated_count; i++) {
        if (llist_cuda_is_valid(list->valid_mask, i)) {
            if (valid_count == index) {
                return list->data[i];
            }
            valid_count++;
        }
    }
    return NULL;
}

CUDA_CALLABLE uint32_t llist_cuda_count_valid(const uint32_t* mask, uint32_t capacity) {
    uint32_t count = 0;
    uint32_t mask_size = mask_size_for_capacity(capacity);
    
    for (uint32_t i = 0; i < mask_size; i++) {
        // Count set bits in each uint32_t
        uint32_t word = mask[i];
        while (word) {
            count += word & 1;
            word >>= 1;
        }
    }
    return count;
}

/* === Iterator Functions === */

CUDA_CALLABLE int llist_cuda_reset_iter(llist_cuda_t* list) {
    if (!list) return LLIST_CUDA_NULL_ARGUMENT;
    list->iter_pos = find_next_valid(list->valid_mask, 0, list->allocated_count);
    return LLIST_CUDA_SUCCESS;
}

CUDA_CALLABLE int llist_cuda_go_next(llist_cuda_t* list) {
    if (!list) return LLIST_CUDA_NULL_ARGUMENT;
    if (list->iter_pos >= list->allocated_count) return LLIST_CUDA_ERROR;
    
    list->iter_pos = find_next_valid(list->valid_mask, list->iter_pos + 1, list->allocated_count);
    return LLIST_CUDA_SUCCESS;
}

CUDA_CALLABLE int llist_cuda_get_iter(llist_cuda_t* list, void** item) {
    if (!list || !item) return LLIST_CUDA_NULL_ARGUMENT;
    if (list->iter_pos >= list->allocated_count) return LLIST_CUDA_ERROR;
    
    *item = list->data[list->iter_pos];
    return LLIST_CUDA_SUCCESS;
}

/* === GPU Memory Management === */

#ifdef __CUDACC__
int llist_cuda_to_gpu(llist_cuda_t* list) {
    if (!list) return LLIST_CUDA_NULL_ARGUMENT;
    if (list->is_gpu_allocated) return LLIST_CUDA_SUCCESS; // Already on GPU

    cudaError_t err;
    size_t data_size = list->capacity * sizeof(void*);
    size_t mask_size = mask_size_for_capacity(list->capacity) * sizeof(uint32_t);
    size_t indices_size = list->capacity * sizeof(uint32_t);

    // Allocate GPU memory
    err = cudaMalloc(&list->gpu_data, data_size);
    if (err != cudaSuccess) return LLIST_CUDA_MALLOC_ERROR;

    err = cudaMalloc(&list->gpu_valid_mask, mask_size);
    if (err != cudaSuccess) {
        cudaFree(list->gpu_data);
        return LLIST_CUDA_MALLOC_ERROR;
    }

    err = cudaMalloc(&list->gpu_indices, indices_size);
    if (err != cudaSuccess) {
        cudaFree(list->gpu_data);
        cudaFree(list->gpu_valid_mask);
        return LLIST_CUDA_MALLOC_ERROR;
    }

    // Copy data to GPU
    err = cudaMemcpy(list->gpu_data, list->data, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(list->gpu_valid_mask, list->valid_mask, mask_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(list->gpu_indices, list->indices, indices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    list->is_gpu_allocated = true;
    return LLIST_CUDA_SUCCESS;

cleanup:
    cudaFree(list->gpu_data);
    cudaFree(list->gpu_valid_mask);
    cudaFree(list->gpu_indices);
    return LLIST_CUDA_ERROR;
}

int llist_cuda_from_gpu(llist_cuda_t* list) {
    if (!list || !list->is_gpu_allocated) return LLIST_CUDA_NULL_ARGUMENT;

    cudaError_t err;
    size_t data_size = list->capacity * sizeof(void*);
    size_t mask_size = mask_size_for_capacity(list->capacity) * sizeof(uint32_t);
    size_t indices_size = list->capacity * sizeof(uint32_t);

    // Copy data from GPU
    err = cudaMemcpy(list->data, list->gpu_data, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return LLIST_CUDA_ERROR;

    err = cudaMemcpy(list->valid_mask, list->gpu_valid_mask, mask_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return LLIST_CUDA_ERROR;

    err = cudaMemcpy(list->indices, list->gpu_indices, indices_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return LLIST_CUDA_ERROR;

    return LLIST_CUDA_SUCCESS;
}
#else
// CPU-only stubs
int llist_cuda_to_gpu(llist_cuda_t* list) {
    return LLIST_CUDA_SUCCESS; // No-op on CPU
}

int llist_cuda_from_gpu(llist_cuda_t* list) {
    return LLIST_CUDA_SUCCESS; // No-op on CPU
}
#endif

/* === Batch Processing === */

llist_cuda_batch_t* llist_cuda_create_batch(
    llist_cuda_t** lists,
    uint32_t num_lists
) {
    if (!lists || num_lists == 0) return NULL;

    llist_cuda_batch_t* batch = (llist_cuda_batch_t*)malloc(sizeof(llist_cuda_batch_t));
    if (!batch) return NULL;

    // Find maximum elements per list
    uint32_t max_elements = 0;
    uint32_t total_elements = 0;
    for (uint32_t i = 0; i < num_lists; i++) {
        if (lists[i]) {
            if (lists[i]->count > max_elements) {
                max_elements = lists[i]->count;
            }
            total_elements += lists[i]->count;
        }
    }

    batch->lists = lists;
    batch->num_lists = num_lists;
    batch->max_elements_per_list = max_elements;

    // Allocate flattened arrays
    batch->flat_data = (void**)malloc(total_elements * sizeof(void*));
    batch->flat_valid_mask = (uint32_t*)malloc(mask_size_for_capacity(total_elements) * sizeof(uint32_t));
    batch->list_offsets = (uint32_t*)malloc(num_lists * sizeof(uint32_t));
    batch->list_counts = (uint32_t*)malloc(num_lists * sizeof(uint32_t));

    if (!batch->flat_data || !batch->flat_valid_mask || 
        !batch->list_offsets || !batch->list_counts) {
        llist_cuda_destroy_batch(batch);
        return NULL;
    }

    // Flatten the data
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_lists; i++) {
        batch->list_offsets[i] = offset;
        batch->list_counts[i] = lists[i] ? lists[i]->count : 0;
        
        if (lists[i]) {
            // Copy valid elements to flattened array
            uint32_t copied = 0;
            for (uint32_t j = 0; j < lists[i]->allocated_count && copied < lists[i]->count; j++) {
                if (llist_cuda_is_valid(lists[i]->valid_mask, j)) {
                    batch->flat_data[offset + copied] = lists[i]->data[j];
                    llist_cuda_set_valid(batch->flat_valid_mask, offset + copied);
                    copied++;
                }
            }
            offset += lists[i]->count;
        }
    }

    return batch;
}

void llist_cuda_destroy_batch(llist_cuda_batch_t* batch) {
    if (!batch) return;
    
    free(batch->flat_data);
    free(batch->flat_valid_mask);
    free(batch->list_offsets);
    free(batch->list_counts);
    free(batch);
}
