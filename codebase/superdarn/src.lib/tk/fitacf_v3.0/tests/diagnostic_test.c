#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Include the header
#include "../include/llist.h"

// Simple test data
typedef struct {
    int value;
} simple_data_t;

// Simple comparator
int simple_compare(llist_node first, llist_node second) {
    simple_data_t *a = (simple_data_t *)first;
    simple_data_t *b = (simple_data_t *)second;
    return a->value - b->value;
}

// Simple equality
bool simple_equal(llist_node first, llist_node second) {
    simple_data_t *a = (simple_data_t *)first;
    simple_data_t *b = (simple_data_t *)second;
    return a->value == b->value;
}

int main() {
    printf("=== SUPERDARN Linked List Diagnostic Test ===\n\n");
    
    // Step 1: Test list creation
    printf("Step 1: Creating list...\n");
    llist list = llist_create(simple_compare, simple_equal, MT_SUPPORT_FALSE);
    if (!list) {
        printf("ERROR: Failed to create list\n");
        return 1;
    }
    printf("✓ List created successfully\n");
    
    // Step 2: Test empty list properties
    printf("\nStep 2: Testing empty list...\n");
    printf("  Is empty: %s\n", llist_is_empty(list) ? "true" : "false");
    printf("  Size: %d\n", llist_size(list));
    
    // Step 3: Add a single node
    printf("\nStep 3: Adding single node...\n");
    simple_data_t *data1 = malloc(sizeof(simple_data_t));
    data1->value = 42;
    
    E_LLIST result = llist_add_node(list, data1, ADD_NODE_REAR);
    printf("  Add result: %d (0=success)\n", result);
    if (result != LLIST_SUCCESS) {
        printf("ERROR: Failed to add node\n");
        return 1;
    }
    printf("✓ Node added successfully\n");
    
    // Step 4: Test list properties after adding
    printf("\nStep 4: Testing list after adding node...\n");
    printf("  Is empty: %s\n", llist_is_empty(list) ? "true" : "false");
    printf("  Size: %d\n", llist_size(list));
    
    // Step 5: Test iterator initialization
    printf("\nStep 5: Testing iterator initialization...\n");
    E_LLIST iter_result = llist_reset_iter(list);
    printf("  Reset iter result: %d (0=success)\n", iter_result);
    if (iter_result != LLIST_SUCCESS) {
        printf("ERROR: Failed to reset iterator\n");
        return 1;
    }
    printf("✓ Iterator reset successfully\n");
    
    // Step 6: Test getting first element (using correct API)
    printf("\nStep 6: Testing get_iter (using correct void** parameter)...\n");
    printf("  About to call llist_get_iter...\n");
    fflush(stdout);  // Ensure output is printed before potential crash
    
    void *current_ptr = NULL;
    E_LLIST get_result = llist_get_iter(list, &current_ptr);
    printf("  Get iter result: %d (0=success)\n", get_result);
    printf("  Retrieved pointer: %p\n", current_ptr);
    
    if (get_result == LLIST_SUCCESS && current_ptr) {
        simple_data_t *data = (simple_data_t *)current_ptr;
        printf("  Retrieved value: %d\n", data->value);
        printf("✓ Successfully retrieved first element\n");
        
        // Step 7: Test iteration
        printf("\nStep 7: Testing iteration...\n");
        E_LLIST next_result = llist_go_next(list);
        printf("  Go next result: %d\n", next_result);
        
        void *next_ptr = NULL;
        E_LLIST next_get_result = llist_get_iter(list, &next_ptr);
        printf("  Next get iter result: %d\n", next_get_result);
        printf("  Next element: %p (should be NULL for single item list)\n", next_ptr);
    } else {
        printf("ERROR: get_iter failed with result %d\n", get_result);
    }
    
    // Step 8: Clean up
    printf("\nStep 8: Cleaning up...\n");
    llist_destroy(list, true, free);
    printf("✓ List destroyed successfully\n");
    
    printf("\n=== All diagnostic tests completed successfully! ===\n");
    return 0;
}
