/*
 * CUDA Linked List Compatibility Layer
 * 
 * This file is part of the Radar Software Toolkit (RST).
 * 
 * Provides backward compatibility with original llist API while using
 * CUDA-compatible data structures underneath. Allows existing code to
 * compile without changes while gaining GPU acceleration.
 * 
 * Author: CUDA Conversion Project
 * Date: 2025
 */

#ifndef LLIST_COMPAT_H_
#define LLIST_COMPAT_H_

#include "llist_cuda.h"

/* === Compatibility Type Mappings === */

// Map original types to CUDA types
typedef llist_cuda_t* llist;
typedef void* llist_node;

// Map original enums to CUDA enums
typedef enum {
    LLIST_SUCCESS = LLIST_CUDA_SUCCESS,
    LLIST_NODE_NOT_FOUND = LLIST_CUDA_NODE_NOT_FOUND,
    LLIST_EQUAL_MISSING = LLIST_CUDA_EQUAL_MISSING,
    LLIST_COMPERATOR_MISSING = LLIST_CUDA_COMPARATOR_MISSING,
    LLIST_NULL_ARGUMENT = LLIST_CUDA_NULL_ARGUMENT,
    LLIST_MALLOC_ERROR = LLIST_CUDA_MALLOC_ERROR,
    LLIST_NOT_IMPLEMENTED = LLIST_CUDA_ERROR,
    LLIST_MULTITHREAD_ISSUE = LLIST_CUDA_ERROR,
    LLIST_ERROR = LLIST_CUDA_ERROR,
    LLIST_END_OF_LIST = LLIST_CUDA_ERROR
} E_LLIST;

// Map original flags
#define ADD_NODE_FRONT      LLIST_CUDA_ADD_FRONT
#define ADD_NODE_REAR       LLIST_CUDA_ADD_REAR
#define ADD_NODE_BEFORE     LLIST_CUDA_ADD_FRONT
#define ADD_NODE_AFTER      LLIST_CUDA_ADD_REAR
#define SORT_LIST_ASCENDING LLIST_CUDA_SORT_ASC
#define SORT_LIST_DESCENDING LLIST_CUDA_SORT_DESC

// Threading support flags (ignored in CUDA version)
#define MT_SUPPORT_TRUE  (1)
#define MT_SUPPORT_FALSE (0)

// Map original function pointer types
typedef llist_cuda_node_func node_func;
typedef llist_cuda_node_func_arg node_func_arg;
typedef llist_cuda_comparator comperator;  // Note: original had typo "comperator"
typedef llist_cuda_equal equal;

/* === Compatibility Function Wrappers === */

/**
 * @brief Create a list - compatible with original API
 */
static inline llist llist_create(comperator compare_func, equal equal_func, unsigned flags) {
    // Ignore threading flags for CUDA version
    return llist_cuda_create(compare_func, equal_func, 0);
}

/**
 * @brief Destroy a list - compatible with original API
 */
static inline void llist_destroy(llist list, bool destroy_nodes, node_func destructor) {
    llist_cuda_destroy(list, destroy_nodes, destructor);
}

/**
 * @brief Add a node - compatible with original API
 */
static inline int llist_add_node(llist list, llist_node node, int flags) {
    return llist_cuda_add_node(list, node, flags);
}

/**
 * @brief Delete a node - compatible with original API
 */
static inline int llist_delete_node(llist list, llist_node node, bool destroy_node, node_func destructor) {
    return llist_cuda_delete_node(list, node, destroy_node, destructor);
}

/**
 * @brief Find a node - compatible with original API
 */
static inline int llist_find_node(llist list, void* data, llist_node* found) {
    return llist_cuda_find_node(list, data, found);
}

/**
 * @brief Get list size - compatible with original API
 */
static inline int llist_size(llist list) {
    return (int)llist_cuda_size(list);
}

/**
 * @brief Check if empty - compatible with original API
 */
static inline bool llist_is_empty(llist list) {
    return llist_cuda_is_empty(list);
}

/**
 * @brief Get head node - compatible with original API
 */
static inline llist_node llist_get_head(llist list) {
    return llist_cuda_get_at_index(list, 0);
}

/**
 * @brief Get tail node - compatible with original API
 */
static inline llist_node llist_get_tail(llist list) {
    uint32_t size = llist_cuda_size(list);
    return size > 0 ? llist_cuda_get_at_index(list, size - 1) : NULL;
}

/**
 * @brief Push to front - compatible with original API
 */
static inline int llist_push(llist list, llist_node node) {
    return llist_cuda_add_node(list, node, LLIST_CUDA_ADD_FRONT);
}

/**
 * @brief Peek at front - compatible with original API
 */
static inline llist_node llist_peek(llist list) {
    return llist_get_head(list);
}

/**
 * @brief Pop from front - compatible with original API
 */
static inline llist_node llist_pop(llist list) {
    llist_node head = llist_get_head(list);
    if (head) {
        llist_cuda_delete_node(list, head, false, NULL);
    }
    return head;
}

/**
 * @brief Reset iterator - compatible with original API
 */
static inline int llist_reset_iter(llist list) {
    return llist_cuda_reset_iter(list);
}

/**
 * @brief Move iterator next - compatible with original API
 */
static inline int llist_go_next(llist list) {
    return llist_cuda_go_next(list);
}

/**
 * @brief Get iterator item - compatible with original API
 */
static inline int llist_get_iter(llist list, void** item) {
    return llist_cuda_get_iter(list, item);
}

/* === Complex Operations Requiring Special Handling === */

/**
 * @brief For-each operation - CPU fallback for compatibility
 * 
 * This provides CPU compatibility but doesn't use GPU acceleration.
 * For GPU acceleration, code should be converted to use batch processing.
 */
static inline int llist_for_each(llist list, node_func func) {
    if (!list || !func) return LLIST_NULL_ARGUMENT;
    
    llist_cuda_reset_iter(list);
    void* item;
    while (llist_cuda_get_iter(list, &item) == LLIST_CUDA_SUCCESS) {
        func(item);
        if (llist_cuda_go_next(list) != LLIST_CUDA_SUCCESS) break;
    }
    return LLIST_SUCCESS;
}

/**
 * @brief For-each with arguments - CPU fallback for compatibility
 * 
 * This is the most critical function to convert for GPU acceleration.
 * Provides CPU fallback but real performance gains require using
 * llist_cuda_launch_batch_process() instead.
 */
static inline int llist_for_each_arg(llist list, node_func_arg func, void* arg1, void* arg2) {
    if (!list || !func) return LLIST_NULL_ARGUMENT;
    
    llist_cuda_reset_iter(list);
    void* item;
    while (llist_cuda_get_iter(list, &item) == LLIST_CUDA_SUCCESS) {
        func(item, arg1, arg2);
        if (llist_cuda_go_next(list) != LLIST_CUDA_SUCCESS) break;
    }
    return LLIST_SUCCESS;
}

/**
 * @brief Sort list - uses CUDA implementation
 */
static inline int llist_sort(llist list, int flags) {
    return llist_cuda_sort(list, flags);
}

/**
 * @brief Get maximum element - uses CUDA reduction when available
 */
static inline int llist_get_max(llist list, llist_node* max) {
    if (!list || !max || !list->comp_func) return LLIST_COMPERATOR_MISSING;
    
    // Simple CPU implementation for compatibility
    // TODO: Use CUDA reduction kernel for better performance
    *max = NULL;
    llist_cuda_reset_iter(list);
    void* item;
    while (llist_cuda_get_iter(list, &item) == LLIST_CUDA_SUCCESS) {
        if (!*max || list->comp_func(item, *max) > 0) {
            *max = item;
        }
        if (llist_cuda_go_next(list) != LLIST_CUDA_SUCCESS) break;
    }
    return *max ? LLIST_SUCCESS : LLIST_NODE_NOT_FOUND;
}

/**
 * @brief Get minimum element - uses CUDA reduction when available
 */
static inline int llist_get_min(llist list, llist_node* min) {
    if (!list || !min || !list->comp_func) return LLIST_COMPERATOR_MISSING;
    
    // Simple CPU implementation for compatibility
    // TODO: Use CUDA reduction kernel for better performance
    *min = NULL;
    llist_cuda_reset_iter(list);
    void* item;
    while (llist_cuda_get_iter(list, &item) == LLIST_CUDA_SUCCESS) {
        if (!*min || list->comp_func(item, *min) < 0) {
            *min = item;
        }
        if (llist_cuda_go_next(list) != LLIST_CUDA_SUCCESS) break;
    }
    return *min ? LLIST_SUCCESS : LLIST_NODE_NOT_FOUND;
}

/**
 * @brief Reverse list - CPU implementation
 */
static inline int llist_reverse(llist list) {
    // Simple implementation - reverse the indices array
    if (!list) return LLIST_NULL_ARGUMENT;
    
    uint32_t size = llist_cuda_size(list);
    if (size <= 1) return LLIST_SUCCESS;
    
    // This would need to be implemented in llist_cuda.c
    // For now, return not implemented
    return LLIST_NOT_IMPLEMENTED;
}

/**
 * @brief Concatenate lists - CPU implementation
 */
static inline int llist_concat(llist first, llist second) {
    if (!first || !second) return LLIST_NULL_ARGUMENT;
    
    // Simple implementation - add all elements from second to first
    llist_cuda_reset_iter(second);
    void* item;
    while (llist_cuda_get_iter(second, &item) == LLIST_CUDA_SUCCESS) {
        llist_cuda_add_node(first, item, LLIST_CUDA_ADD_REAR);
        if (llist_cuda_go_next(second) != LLIST_CUDA_SUCCESS) break;
    }
    return LLIST_SUCCESS;
}

/**
 * @brief Merge lists - not implemented in original either
 */
static inline int llist_merge(llist first, llist second) {
    return LLIST_NOT_IMPLEMENTED;
}

/* === GPU Acceleration Hints === */

/**
 * @brief Check if GPU acceleration is available for this list
 */
static inline bool llist_cuda_acceleration_available(llist list) {
#ifdef __CUDACC__
    return list != NULL;
#else
    return false;
#endif
}

/**
 * @brief Transfer list to GPU for acceleration
 */
static inline int llist_enable_gpu_acceleration(llist list) {
    return llist_cuda_to_gpu(list);
}

/**
 * @brief Transfer list from GPU back to CPU
 */
static inline int llist_disable_gpu_acceleration(llist list) {
    return llist_cuda_from_gpu(list);
}

#endif /* LLIST_COMPAT_H_ */
