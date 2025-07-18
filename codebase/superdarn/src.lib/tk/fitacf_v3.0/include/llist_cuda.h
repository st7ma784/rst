/*
 * CUDA-Compatible Linked List Replacement
 * 
 * This file is part of the Radar Software Toolkit (RST).
 * 
 * Replaces traditional linked lists with GPU-friendly array-based structures
 * using validity masks instead of dynamic deletion.
 * 
 * Author: CUDA Conversion Project
 * Date: 2025
 */

#ifndef LLIST_CUDA_H_
#define LLIST_CUDA_H_

#include <stdbool.h>
#include <stdint.h>
#include "rtypes.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define CUDA_KERNEL __global__
#define CUDA_DEVICE __device__
#else
#define CUDA_CALLABLE
#define CUDA_KERNEL
#define CUDA_DEVICE
#endif

/* Maximum capacity for GPU memory coalescing - power of 2 for alignment */
#define LLIST_CUDA_MAX_CAPACITY 1024
#define LLIST_CUDA_WARP_SIZE 32
#define LLIST_CUDA_BLOCK_SIZE 256

/** Error codes compatible with original llist API */
typedef enum {
    LLIST_CUDA_SUCCESS = 0x00,
    LLIST_CUDA_NODE_NOT_FOUND,
    LLIST_CUDA_EQUAL_MISSING,
    LLIST_CUDA_COMPARATOR_MISSING,
    LLIST_CUDA_NULL_ARGUMENT,
    LLIST_CUDA_MALLOC_ERROR,
    LLIST_CUDA_CAPACITY_EXCEEDED,
    LLIST_CUDA_INVALID_INDEX,
    LLIST_CUDA_ERROR
} E_LLIST_CUDA;

/** Flags for operations */
#define LLIST_CUDA_ADD_FRONT    (1 << 0)
#define LLIST_CUDA_ADD_REAR     (~LLIST_CUDA_ADD_FRONT)
#define LLIST_CUDA_SORT_ASC     (1 << 0)
#define LLIST_CUDA_SORT_DESC    (~LLIST_CUDA_SORT_ASC)

/** Function pointer types - compatible with original API */
typedef void (*llist_cuda_node_func)(void* node);
typedef void (*llist_cuda_node_func_arg)(void* node, void* arg1, void* arg2);
typedef int (*llist_cuda_comparator)(const void* first, const void* second);
typedef bool (*llist_cuda_equal)(const void* first, const void* second);

/**
 * @brief CUDA-compatible array-based list structure
 * 
 * Replaces linked list with contiguous arrays for GPU memory coalescing.
 * Uses validity masks instead of dynamic deletion for GPU efficiency.
 */
typedef struct {
    /* Data storage - contiguous for GPU coalescing */
    void** data;                    /**< Array of data pointers */
    uint32_t* valid_mask;          /**< Bitmask for valid elements (32 elements per uint32_t) */
    uint32_t* indices;             /**< Sorted indices for ordered access */
    
    /* Metadata */
    uint32_t capacity;             /**< Maximum number of elements */
    uint32_t count;                /**< Current number of valid elements */
    uint32_t allocated_count;      /**< Total allocated slots (including invalid) */
    
    /* Iterator state */
    uint32_t iter_pos;             /**< Current iterator position */
    
    /* Function pointers */
    llist_cuda_comparator comp_func;
    llist_cuda_equal equal_func;
    
    /* GPU memory management */
    bool is_gpu_allocated;         /**< True if data is on GPU */
    void** gpu_data;               /**< GPU copy of data array */
    uint32_t* gpu_valid_mask;      /**< GPU copy of validity mask */
    uint32_t* gpu_indices;         /**< GPU copy of indices */
    
} llist_cuda_t;

/**
 * @brief Batch processing structure for GPU kernels
 * 
 * Enables processing multiple lists simultaneously on GPU
 */
typedef struct {
    llist_cuda_t** lists;          /**< Array of list pointers */
    uint32_t num_lists;            /**< Number of lists in batch */
    uint32_t max_elements_per_list; /**< Maximum elements in any list */
    
    /* Flattened data for GPU processing */
    void** flat_data;              /**< Flattened data array */
    uint32_t* flat_valid_mask;     /**< Flattened validity masks */
    uint32_t* list_offsets;        /**< Starting offset for each list */
    uint32_t* list_counts;         /**< Element count for each list */
    
} llist_cuda_batch_t;

/* === Core API Functions === */

/**
 * @brief Create a new CUDA-compatible list
 */
CUDA_CALLABLE llist_cuda_t* llist_cuda_create(
    llist_cuda_comparator comp_func,
    llist_cuda_equal equal_func,
    uint32_t initial_capacity
);

/**
 * @brief Destroy a CUDA list and optionally its data
 */
CUDA_CALLABLE void llist_cuda_destroy(
    llist_cuda_t* list,
    bool destroy_nodes,
    llist_cuda_node_func destructor
);

/**
 * @brief Add a node to the list
 */
CUDA_CALLABLE int llist_cuda_add_node(
    llist_cuda_t* list,
    void* node,
    int flags
);

/**
 * @brief Mark a node as invalid (mask-based deletion)
 */
CUDA_CALLABLE int llist_cuda_delete_node(
    llist_cuda_t* list,
    void* node,
    bool destroy_node,
    llist_cuda_node_func destructor
);

/**
 * @brief Find a node in the list
 */
CUDA_CALLABLE int llist_cuda_find_node(
    llist_cuda_t* list,
    void* data,
    void** found
);

/**
 * @brief Get the number of valid elements
 */
CUDA_CALLABLE uint32_t llist_cuda_size(const llist_cuda_t* list);

/**
 * @brief Check if list is empty
 */
CUDA_CALLABLE bool llist_cuda_is_empty(const llist_cuda_t* list);

/* === GPU-Specific Functions === */

/**
 * @brief Transfer list data to GPU memory
 */
int llist_cuda_to_gpu(llist_cuda_t* list);

/**
 * @brief Transfer list data from GPU memory
 */
int llist_cuda_from_gpu(llist_cuda_t* list);

/**
 * @brief Create a batch processing structure
 */
llist_cuda_batch_t* llist_cuda_create_batch(
    llist_cuda_t** lists,
    uint32_t num_lists
);

/**
 * @brief Destroy a batch processing structure
 */
void llist_cuda_destroy_batch(llist_cuda_batch_t* batch);

/* === Utility Functions === */

/**
 * @brief Get element at specific index (respecting validity mask)
 */
CUDA_CALLABLE void* llist_cuda_get_at_index(
    const llist_cuda_t* list,
    uint32_t index
);

/**
 * @brief Compact the list by removing invalid elements
 */
CUDA_CALLABLE int llist_cuda_compact(llist_cuda_t* list);

/**
 * @brief Sort the list using the comparator function
 */
CUDA_CALLABLE int llist_cuda_sort(llist_cuda_t* list, int flags);

/* === Iterator Functions === */

/**
 * @brief Reset iterator to beginning
 */
CUDA_CALLABLE int llist_cuda_reset_iter(llist_cuda_t* list);

/**
 * @brief Move iterator to next valid element
 */
CUDA_CALLABLE int llist_cuda_go_next(llist_cuda_t* list);

/**
 * @brief Get current iterator element
 */
CUDA_CALLABLE int llist_cuda_get_iter(llist_cuda_t* list, void** item);

/* === Mask Utility Functions === */

/**
 * @brief Set validity bit for element at index
 */
CUDA_CALLABLE static inline void llist_cuda_set_valid(uint32_t* mask, uint32_t index) {
    mask[index / 32] |= (1U << (index % 32));
}

/**
 * @brief Clear validity bit for element at index
 */
CUDA_CALLABLE static inline void llist_cuda_clear_valid(uint32_t* mask, uint32_t index) {
    mask[index / 32] &= ~(1U << (index % 32));
}

/**
 * @brief Check if element at index is valid
 */
CUDA_CALLABLE static inline bool llist_cuda_is_valid(const uint32_t* mask, uint32_t index) {
    return (mask[index / 32] & (1U << (index % 32))) != 0;
}

/**
 * @brief Count number of valid elements in mask
 */
CUDA_CALLABLE uint32_t llist_cuda_count_valid(const uint32_t* mask, uint32_t capacity);

#endif /* LLIST_CUDA_H_ */
