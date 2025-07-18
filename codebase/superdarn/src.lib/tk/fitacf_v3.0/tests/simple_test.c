#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

// Include the header first
#include "../include/llist.h"

// Simple test data structure
typedef struct {
    int id;
    double value;
    char name[32];
} test_data_t;

// Comparator function for sorting (matches SUPERDARN API)
int compare_test_data(llist_node first, llist_node second) {
    test_data_t *ta = (test_data_t *)first;
    test_data_t *tb = (test_data_t *)second;
    
    if (ta->value < tb->value) return -1;
    if (ta->value > tb->value) return 1;
    return 0;
}

// Equality function (matches SUPERDARN API)
bool equal_test_data(llist_node first, llist_node second) {
    test_data_t *ta = (test_data_t *)first;
    test_data_t *tb = (test_data_t *)second;
    return ta->id == tb->id;
}

// Test basic linked list operations
int test_basic_operations() {
    printf("Testing basic linked list operations...\n");
    
    // Create list
    llist list = llist_create(compare_test_data, equal_test_data, MT_SUPPORT_FALSE);
    if (!list) {
        printf("ERROR: Failed to create list\n");
        return 0;
    }
    
    // Test empty list
    if (!llist_is_empty(list)) {
        printf("ERROR: New list should be empty\n");
        return 0;
    }
    
    if (llist_size(list) != 0) {
        printf("ERROR: New list size should be 0\n");
        return 0;
    }
    
    // Add some test data
    test_data_t *data1 = malloc(sizeof(test_data_t));
    test_data_t *data2 = malloc(sizeof(test_data_t));
    test_data_t *data3 = malloc(sizeof(test_data_t));
    
    data1->id = 1; data1->value = 10.5; strcpy(data1->name, "first");
    data2->id = 2; data2->value = 20.3; strcpy(data2->name, "second");
    data3->id = 3; data3->value = 15.7; strcpy(data3->name, "third");
    
    // Add nodes (data is the node in SUPERDARN API)
    if (llist_add_node(list, data1, ADD_NODE_REAR) != LLIST_SUCCESS) {
        printf("ERROR: Failed to add node1\n");
        return 0;
    }
    
    if (llist_add_node(list, data2, ADD_NODE_REAR) != LLIST_SUCCESS) {
        printf("ERROR: Failed to add node2\n");
        return 0;
    }
    
    if (llist_add_node(list, data3, ADD_NODE_REAR) != LLIST_SUCCESS) {
        printf("ERROR: Failed to add node3\n");
        return 0;
    }
    
    // Test size
    if (llist_size(list) != 3) {
        printf("ERROR: List size should be 3, got %d\n", llist_size(list));
        return 0;
    }
    
    // Test iteration
    llist_reset_iter(list);
    int count = 0;
    void *current_ptr;
    while (llist_get_iter(list, &current_ptr) == LLIST_SUCCESS && current_ptr != NULL) {
        test_data_t *data = (test_data_t *)current_ptr;
        printf("  Node %d: id=%d, value=%.1f, name=%s\n", 
               count, data->id, data->value, data->name);
        count++;
        if (llist_go_next(list) != LLIST_SUCCESS) break;
    }
    
    if (count != 3) {
        printf("ERROR: Iterator should find 3 nodes, found %d\n", count);
        return 0;
    }
    
    // Test find
    test_data_t search_data = {2, 0, ""};
    void *found_ptr = NULL;
    if (llist_find_node(list, &search_data, &found_ptr) != LLIST_SUCCESS || !found_ptr) {
        printf("ERROR: Failed to find node with id=2\n");
        return 0;
    }
    
    test_data_t *found_data = (test_data_t *)found_ptr;
    if (found_data->id != 2) {
        printf("ERROR: Found wrong node, expected id=2, got id=%d\n", found_data->id);
        return 0;
    }
    
    // Test sorting
    if (llist_sort(list, SORT_LIST_ASCENDING) != LLIST_SUCCESS) {
        printf("ERROR: Failed to sort list\n");
        return 0;
    }
    
    // Verify sort order
    llist_reset_iter(list);
    double prev_value = -1.0;
    void *sort_ptr;
    while (llist_get_iter(list, &sort_ptr) == LLIST_SUCCESS && sort_ptr != NULL) {
        test_data_t *data = (test_data_t *)sort_ptr;
        if (data->value < prev_value) {
            printf("ERROR: List not properly sorted\n");
            return 0;
        }
        prev_value = data->value;
        if (llist_go_next(list) != LLIST_SUCCESS) break;
    }
    
    // Clean up
    llist_destroy(list, true, free);
    
    printf("✓ Basic operations test passed\n");
    return 1;
}

// Test performance with larger dataset
int test_performance() {
    printf("Testing performance with larger dataset...\n");
    
    const int num_items = 1000;  // Reduced for faster testing
    clock_t start, end;
    
    // Create list
    llist list = llist_create(compare_test_data, equal_test_data, MT_SUPPORT_FALSE);
    if (!list) {
        printf("ERROR: Failed to create list\n");
        return 0;
    }
    
    // Generate test data
    test_data_t *test_data = malloc(num_items * sizeof(test_data_t));
    
    for (int i = 0; i < num_items; i++) {
        test_data[i].id = i;
        test_data[i].value = (double)rand() / RAND_MAX * 1000.0;
        snprintf(test_data[i].name, sizeof(test_data[i].name), "item_%d", i);
    }
    
    // Time insertion
    start = clock();
    for (int i = 0; i < num_items; i++) {
        if (llist_add_node(list, &test_data[i], ADD_NODE_REAR) != LLIST_SUCCESS) {
            printf("ERROR: Failed to add node %d\n", i);
            return 0;
        }
    }
    end = clock();
    
    double insert_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Inserted %d items in %.3f seconds (%.0f items/sec)\n", 
           num_items, insert_time, num_items / insert_time);
    
    // Verify size
    if (llist_size(list) != num_items) {
        printf("ERROR: Expected %d items, got %d\n", num_items, llist_size(list));
        return 0;
    }
    
    // Time iteration
    start = clock();
    llist_reset_iter(list);
    int count = 0;
    void *current_ptr;
    while (llist_get_iter(list, &current_ptr) == LLIST_SUCCESS && current_ptr != NULL) {
        count++;
        if (llist_go_next(list) != LLIST_SUCCESS) break;
    }
    end = clock();
    
    double iter_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Iterated through %d items in %.3f seconds (%.0f items/sec)\n", 
           count, iter_time, count / iter_time);
    
    if (count != num_items) {
        printf("ERROR: Iterator found %d items, expected %d\n", count, num_items);
        return 0;
    }
    
    // Time sorting
    start = clock();
    if (llist_sort(list, SORT_LIST_ASCENDING) != LLIST_SUCCESS) {
        printf("ERROR: Failed to sort list\n");
        return 0;
    }
    end = clock();
    
    double sort_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Sorted %d items in %.3f seconds\n", num_items, sort_time);
    
    // Verify sort
    llist_reset_iter(list);
    double prev_value = -1.0;
    int sort_errors = 0;
    void *verify_ptr;
    while (llist_get_iter(list, &verify_ptr) == LLIST_SUCCESS && verify_ptr != NULL) {
        test_data_t *data = (test_data_t *)verify_ptr;
        if (data->value < prev_value) {
            sort_errors++;
        }
        prev_value = data->value;
        if (llist_go_next(list) != LLIST_SUCCESS) break;
    }
    
    if (sort_errors > 0) {
        printf("ERROR: Found %d sort order violations\n", sort_errors);
        return 0;
    }
    
    // Clean up - don't let llist_destroy free nodes, we'll handle it ourselves
    llist_destroy(list, false, NULL);
    free(test_data);
    
    printf("✓ Performance test passed\n");
    return 1;
}

int main() {
    printf("========================================\n");
    printf("Simple SUPERDARN Linked List Test\n");
    printf("========================================\n\n");
    
    srand(time(NULL));
    
    int tests_passed = 0;
    int total_tests = 2;
    
    if (test_basic_operations()) {
        tests_passed++;
    }
    
    if (test_performance()) {
        tests_passed++;
    }
    
    printf("\n========================================\n");
    printf("Test Results: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("✓ All tests PASSED!\n");
        return 0;
    } else {
        printf("✗ Some tests FAILED!\n");
        return 1;
    }
}
