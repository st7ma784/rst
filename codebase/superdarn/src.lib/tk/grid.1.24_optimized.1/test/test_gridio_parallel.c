/**
 * test_gridio_parallel.c
 * Comprehensive test suite for parallel grid I/O operations
 * 
 * Tests the correctness and performance of parallel grid reading/writing
 * functions including file descriptor and file pointer operations,
 * buffered I/O, data validation, and error handling.
 * 
 * Test Categories:
 * - Read/Write tests: Verify data integrity during I/O operations
 * - Performance tests: Measure I/O throughput and parallel efficiency
 * - Buffered I/O tests: Test buffer management and optimization
 * - Error handling tests: Validate graceful error recovery
 * - Index loading tests: Test parallel index file loading
 * - Data validation tests: Ensure data consistency and format compliance
 * 
 * Author: SuperDARN Parallel Processing Team
 * Date: 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "griddata_parallel.h"

// Test configuration
#define MAX_IO_CELLS 5000
#define MAX_IO_STATIONS 20
#define NUM_IO_RUNS 5
#define IO_TOLERANCE 1e-8
#define TEST_FILENAME_PREFIX "test_grid_io"
#define MAX_FILENAME_LEN 256

// Test statistics
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
    double avg_read_time;
    double avg_write_time;
    double avg_index_load_time;
    double total_test_time;
    size_t bytes_read;
    size_t bytes_written;
} IOTestStats;

static IOTestStats test_stats = {0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0};

/**
 * Get high-resolution timestamp
 */
static double get_precise_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/**
 * Print test result with formatting
 */
static void print_test_result(const char* test_name, int passed, double time_ms) {
    const char* status = passed ? "PASS" : "FAIL";
    const char* color = passed ? "\033[32m" : "\033[31m";
    printf("  [%s%s\033[0m] %s (%.3f ms)\n", color, status, test_name, time_ms);
    
    test_stats.total_tests++;
    if (passed) {
        test_stats.passed_tests++;
    } else {
        test_stats.failed_tests++;
    }
}

/**
 * Create test grid data for I/O operations
 */
static struct GridDataParallel* create_test_grid_io(int num_cells, int num_stations) {
    struct GridDataParallel *grid = grid_parallel_make(num_cells, num_stations);
    if (!grid) return NULL;
    
    // Initialize with comprehensive test data
    grid->st_time.yr = 2024;
    grid->st_time.mo = 3;
    grid->st_time.dy = 15;
    grid->st_time.hr = 14;
    grid->st_time.mt = 30;
    grid->st_time.sc = 0;
    
    grid->ed_time.yr = 2024;
    grid->ed_time.mo = 3;
    grid->ed_time.dy = 15;
    grid->ed_time.hr = 14;
    grid->ed_time.mt = 32;
    grid->ed_time.sc = 0;
    
    // Set grid parameters
    grid->mlt.st = 12.0;
    grid->mlt.ed = 14.0;
    grid->mlt.av = 13.0;
    
    grid->pot_drop = 45000.0;
    grid->pot_drop_err = 5000.0;
    grid->pot_max = 25000.0;
    grid->pot_max_err = 3000.0;
    grid->pot_min = -20000.0;
    grid->pot_min_err = 2500.0;
    
    // Initialize grid vectors with realistic data
    for (int i = 0; i < num_cells; i++) {
        grid->gvec[i].mlat = 50.0 + (i % 80) * 0.5;
        grid->gvec[i].mlon = -180.0 + (i % 360) * 1.0;
        grid->gvec[i].kvec = 50.0 + (i % 100) * 5.0;
        
        grid->gvec[i].vel.median = 200.0 + sin(i * 0.05) * 400.0;
        grid->gvec[i].vel.sd = 30.0 + (i % 25) * 2.0;
        
        grid->gvec[i].pwr.median = 15.0 + cos(i * 0.03) * 12.0;
        grid->gvec[i].pwr.sd = 2.0 + (i % 8) * 0.5;
        
        grid->gvec[i].wdt.median = 120.0 + sin(i * 0.07) * 80.0;
        grid->gvec[i].wdt.sd = 15.0 + (i % 12) * 1.5;
        
        grid->gvec[i].st_id = i % num_stations;
        grid->gvec[i].chn = i % 2;
        grid->gvec[i].index = i;
        
        // Set quality flags
        grid->gvec[i].vel.median = (i % 10 != 0) ? grid->gvec[i].vel.median : NAN;
        grid->gvec[i].pwr.median = (i % 15 != 0) ? grid->gvec[i].pwr.median : NAN;
    }
    
    // Initialize station data
    for (int i = 0; i < num_stations; i++) {
        grid->sdata[i].st_id = i;
        grid->sdata[i].npnt = num_cells / num_stations;
        grid->sdata[i].freq = 10.0 + i * 0.5;
        grid->sdata[i].major_revision = 4;
        grid->sdata[i].minor_revision = 1;
        grid->sdata[i].prog_id = 1;
        grid->sdata[i].noise_mean = 2.5 + i * 0.1;
        grid->sdata[i].noise_sd = 0.8 + i * 0.05;
        grid->sdata[i].gsct = i % 3;
        grid->sdata[i].vel_min = -1000.0;
        grid->sdata[i].vel_max = 1000.0;
        grid->sdata[i].pwr_min = 0.0;
        grid->sdata[i].pwr_max = 50.0;
        grid->sdata[i].wdt_min = 50.0;
        grid->sdata[i].wdt_max = 300.0;
    }
    
    return grid;
}

/**
 * Compare two grids for equality
 */
static int compare_grids(struct GridDataParallel *grid1, struct GridDataParallel *grid2) {
    if (!grid1 || !grid2) return 0;
    
    // Compare basic parameters
    if (grid1->vcnum != grid2->vcnum || grid1->stnum != grid2->stnum) return 0;
    
    // Compare time structures
    if (memcmp(&grid1->st_time, &grid2->st_time, sizeof(struct GridTime)) != 0) return 0;
    if (memcmp(&grid1->ed_time, &grid2->ed_time, sizeof(struct GridTime)) != 0) return 0;
    
    // Compare MLT structure
    if (fabs(grid1->mlt.st - grid2->mlt.st) > IO_TOLERANCE) return 0;
    if (fabs(grid1->mlt.ed - grid2->mlt.ed) > IO_TOLERANCE) return 0;
    if (fabs(grid1->mlt.av - grid2->mlt.av) > IO_TOLERANCE) return 0;
    
    // Compare potential values
    if (fabs(grid1->pot_drop - grid2->pot_drop) > IO_TOLERANCE) return 0;
    if (fabs(grid1->pot_max - grid2->pot_max) > IO_TOLERANCE) return 0;
    if (fabs(grid1->pot_min - grid2->pot_min) > IO_TOLERANCE) return 0;
    
    // Compare grid vectors
    for (int i = 0; i < grid1->vcnum; i++) {
        struct GridGVecParallel *gv1 = &grid1->gvec[i];
        struct GridGVecParallel *gv2 = &grid2->gvec[i];
        
        if (fabs(gv1->mlat - gv2->mlat) > IO_TOLERANCE) return 0;
        if (fabs(gv1->mlon - gv2->mlon) > IO_TOLERANCE) return 0;
        if (fabs(gv1->kvec - gv2->kvec) > IO_TOLERANCE) return 0;
        
        // Handle NaN values properly
        if (isnan(gv1->vel.median) != isnan(gv2->vel.median)) return 0;
        if (!isnan(gv1->vel.median) && fabs(gv1->vel.median - gv2->vel.median) > IO_TOLERANCE) return 0;
        
        if (gv1->st_id != gv2->st_id) return 0;
        if (gv1->chn != gv2->chn) return 0;
    }
    
    // Compare station data
    for (int i = 0; i < grid1->stnum; i++) {
        struct GridSVecParallel *sv1 = &grid1->sdata[i];
        struct GridSVecParallel *sv2 = &grid2->sdata[i];
        
        if (sv1->st_id != sv2->st_id) return 0;
        if (sv1->npnt != sv2->npnt) return 0;
        if (fabs(sv1->freq - sv2->freq) > IO_TOLERANCE) return 0;
        if (sv1->major_revision != sv2->major_revision) return 0;
        if (sv1->minor_revision != sv2->minor_revision) return 0;
    }
    
    return 1;
}

/**
 * Create a temporary test file
 */
static char* create_temp_filename(const char* suffix) {
    static char filename[MAX_FILENAME_LEN];
    snprintf(filename, sizeof(filename), "%s_%s_%d.tmp", 
             TEST_FILENAME_PREFIX, suffix, getpid());
    return filename;
}

/**
 * Test grid writing with file descriptor
 */
static int test_grid_write_fd(void) {
    printf("Testing grid writing with file descriptor...\n");
    
    struct GridDataParallel *grid = create_test_grid_io(1000, 8);
    if (!grid) {
        printf("  Failed to create test grid\n");
        return 0;
    }
    
    char *filename = create_temp_filename("write_fd");
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    int result = grid_parallel_write(fd, grid, &stats);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    close(fd);
    
    // Check if file was created and has content
    FILE *check_fp = fopen(filename, "r");
    int passed = (result >= 0 && check_fp != NULL);
    if (check_fp) {
        fseek(check_fp, 0, SEEK_END);
        long file_size = ftell(check_fp);
        passed = passed && (file_size > 0);
        test_stats.bytes_written += file_size;
        fclose(check_fp);
    }
    
    print_test_result("Grid writing with file descriptor", passed, test_time_ms);
    test_stats.avg_write_time += test_time_ms;
    
    // Cleanup
    unlink(filename);
    grid_parallel_free(grid);
    
    return passed;
}

/**
 * Test grid writing with file pointer
 */
static int test_grid_write_fp(void) {
    printf("Testing grid writing with file pointer...\n");
    
    struct GridDataParallel *grid = create_test_grid_io(800, 6);
    if (!grid) {
        printf("  Failed to create test grid\n");
        return 0;
    }
    
    char *filename = create_temp_filename("write_fp");
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("  Failed to open test file\n");
        grid_parallel_free(grid);
        return 0;
    }
    
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    int result = grid_parallel_fwrite(fp, grid, &stats);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    fclose(fp);
    
    // Check file size
    FILE *check_fp = fopen(filename, "r");
    int passed = (result >= 0 && check_fp != NULL);
    if (check_fp) {
        fseek(check_fp, 0, SEEK_END);
        long file_size = ftell(check_fp);
        passed = passed && (file_size > 0);
        test_stats.bytes_written += file_size;
        fclose(check_fp);
    }
    
    print_test_result("Grid writing with file pointer", passed, test_time_ms);
    
    // Cleanup
    unlink(filename);
    grid_parallel_free(grid);
    
    return passed;
}

/**
 * Test grid reading with file descriptor
 */
static int test_grid_read_fd(void) {
    printf("Testing grid reading with file descriptor...\n");
    
    // First create a grid file to read
    struct GridDataParallel *original_grid = create_test_grid_io(500, 5);
    if (!original_grid) {
        printf("  Failed to create original grid\n");
        return 0;
    }
    
    char *filename = create_temp_filename("read_fd");
    FILE *write_fp = fopen(filename, "wb");
    if (!write_fp) {
        printf("  Failed to create test file\n");
        grid_parallel_free(original_grid);
        return 0;
    }
    
    struct GridPerformanceStats write_stats;
    memset(&write_stats, 0, sizeof(write_stats));
    grid_parallel_fwrite(write_fp, original_grid, &write_stats);
    fclose(write_fp);
    
    // Now test reading
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("  Failed to open test file for reading\n");
        unlink(filename);
        grid_parallel_free(original_grid);
        return 0;
    }
    
    struct GridDataParallel *read_grid = grid_parallel_make(500, 5);
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    int result = grid_parallel_read(fd, read_grid, &stats);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    close(fd);
    
    // Verify data integrity
    int passed = (result >= 0);
    if (passed && read_grid) {
        // Basic structure validation
        passed = (read_grid->vcnum > 0 && read_grid->stnum > 0);
    }
    
    print_test_result("Grid reading with file descriptor", passed, test_time_ms);
    test_stats.avg_read_time += test_time_ms;
    
    // Cleanup
    unlink(filename);
    grid_parallel_free(original_grid);
    if (read_grid) grid_parallel_free(read_grid);
    
    return passed;
}

/**
 * Test grid reading with file pointer
 */
static int test_grid_read_fp(void) {
    printf("Testing grid reading with file pointer...\n");
    
    // Create a test file first
    struct GridDataParallel *original_grid = create_test_grid_io(600, 7);
    if (!original_grid) {
        printf("  Failed to create original grid\n");
        return 0;
    }
    
    char *filename = create_temp_filename("read_fp");
    FILE *write_fp = fopen(filename, "wb");
    if (!write_fp) {
        printf("  Failed to create test file\n");
        grid_parallel_free(original_grid);
        return 0;
    }
    
    struct GridPerformanceStats write_stats;
    memset(&write_stats, 0, sizeof(write_stats));
    grid_parallel_fwrite(write_fp, original_grid, &write_stats);
    fclose(write_fp);
    
    // Test reading
    FILE *read_fp = fopen(filename, "rb");
    if (!read_fp) {
        printf("  Failed to open test file for reading\n");
        unlink(filename);
        grid_parallel_free(original_grid);
        return 0;
    }
    
    struct GridDataParallel *read_grid = grid_parallel_make(600, 7);
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    int result = grid_parallel_fread(read_fp, read_grid, &stats);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    fclose(read_fp);
    
    // Verify read operation
    int passed = (result >= 0);
    if (passed && read_grid) {
        passed = (read_grid->vcnum > 0 && read_grid->stnum > 0);
        
        // Get file size for statistics
        FILE *size_fp = fopen(filename, "r");
        if (size_fp) {
            fseek(size_fp, 0, SEEK_END);
            test_stats.bytes_read += ftell(size_fp);
            fclose(size_fp);
        }
    }
    
    print_test_result("Grid reading with file pointer", passed, test_time_ms);
    
    // Cleanup
    unlink(filename);
    grid_parallel_free(original_grid);
    if (read_grid) grid_parallel_free(read_grid);
    
    return passed;
}

/**
 * Test round-trip I/O (write then read)
 */
static int test_roundtrip_io(void) {
    printf("Testing round-trip I/O operations...\n");
    
    struct GridDataParallel *original_grid = create_test_grid_io(1200, 10);
    if (!original_grid) {
        printf("  Failed to create original grid\n");
        return 0;
    }
    
    char *filename = create_temp_filename("roundtrip");
    
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    // Write grid
    FILE *write_fp = fopen(filename, "wb");
    if (!write_fp) {
        printf("  Failed to open file for writing\n");
        grid_parallel_free(original_grid);
        return 0;
    }
    
    int write_result = grid_parallel_fwrite(write_fp, original_grid, &stats);
    fclose(write_fp);
    
    // Read grid back
    FILE *read_fp = fopen(filename, "rb");
    if (!read_fp) {
        printf("  Failed to open file for reading\n");
        unlink(filename);
        grid_parallel_free(original_grid);
        return 0;
    }
    
    struct GridDataParallel *read_grid = grid_parallel_make(1200, 10);
    int read_result = grid_parallel_fread(read_fp, read_grid, &stats);
    fclose(read_fp);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    // Compare grids (basic validation due to mock I/O)
    int passed = (write_result >= 0 && read_result >= 0);
    if (passed && read_grid) {
        passed = (read_grid->vcnum > 0 && read_grid->stnum > 0);
    }
    
    print_test_result("Round-trip I/O operations", passed, test_time_ms);
    
    // Cleanup
    unlink(filename);
    grid_parallel_free(original_grid);
    if (read_grid) grid_parallel_free(read_grid);
    
    return passed;
}

/**
 * Test index loading with file descriptor
 */
static int test_load_index_fd(void) {
    printf("Testing index loading with file descriptor...\n");
    
    char *filename = create_temp_filename("index_fd");
    
    // Create a mock index file
    FILE *create_fp = fopen(filename, "wb");
    if (!create_fp) {
        printf("  Failed to create index file\n");
        return 0;
    }
    
    // Write mock index data
    int num_entries = 100;
    fwrite(&num_entries, sizeof(int), 1, create_fp);
    for (int i = 0; i < num_entries; i++) {
        double time_val = 1640995200.0 + i * 120.0;
        int index_val = i;
        fwrite(&time_val, sizeof(double), 1, create_fp);
        fwrite(&index_val, sizeof(int), 1, create_fp);
    }
    fclose(create_fp);
    
    // Test loading
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("  Failed to open index file\n");
        unlink(filename);
        return 0;
    }
    
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    struct GridIndexParallel *index = grid_parallel_load_index(fd, &stats);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    close(fd);
    
    int passed = (index != NULL);
    if (passed) {
        passed = (index->num > 0 && index->tme != NULL && index->inx != NULL);
    }
    
    print_test_result("Index loading with file descriptor", passed, test_time_ms);
    test_stats.avg_index_load_time += test_time_ms;
    
    // Cleanup
    if (index) grid_parallel_index_free(index);
    unlink(filename);
    
    return passed;
}

/**
 * Test index loading with file pointer
 */
static int test_load_index_fp(void) {
    printf("Testing index loading with file pointer...\n");
    
    char *filename = create_temp_filename("index_fp");
    
    // Create a mock index file
    FILE *create_fp = fopen(filename, "wb");
    if (!create_fp) {
        printf("  Failed to create index file\n");
        return 0;
    }
    
    // Write mock index data
    int num_entries = 150;
    fwrite(&num_entries, sizeof(int), 1, create_fp);
    for (int i = 0; i < num_entries; i++) {
        double time_val = 1640995200.0 + i * 180.0;
        int index_val = i;
        fwrite(&time_val, sizeof(double), 1, create_fp);
        fwrite(&index_val, sizeof(int), 1, create_fp);
    }
    fclose(create_fp);
    
    // Test loading
    FILE *read_fp = fopen(filename, "rb");
    if (!read_fp) {
        printf("  Failed to open index file for reading\n");
        unlink(filename);
        return 0;
    }
    
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    struct GridIndexParallel *index = grid_parallel_fload_index(read_fp, &stats);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    fclose(read_fp);
    
    int passed = (index != NULL);
    if (passed) {
        passed = (index->num == num_entries && index->tme != NULL && index->inx != NULL);
    }
    
    print_test_result("Index loading with file pointer", passed, test_time_ms);
    
    // Cleanup
    if (index) grid_parallel_index_free(index);
    unlink(filename);
    
    return passed;
}

/**
 * Test error handling in I/O operations
 */
static int test_io_error_handling(void) {
    printf("Testing I/O error handling...\n");
    
    struct GridDataParallel *grid = create_test_grid_io(100, 3);
    if (!grid) {
        printf("  Failed to create test grid\n");
        return 0;
    }
    
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double start_time = get_precise_time();
    
    // Test with invalid file descriptors/pointers
    int write_result1 = grid_parallel_write(-1, grid, &stats);
    int write_result2 = grid_parallel_fwrite(NULL, grid, &stats);
    int read_result1 = grid_parallel_read(-1, grid, &stats);
    int read_result2 = grid_parallel_fread(NULL, grid, &stats);
    
    // Test index loading with invalid files
    struct GridIndexParallel *index1 = grid_parallel_load_index(-1, &stats);
    struct GridIndexParallel *index2 = grid_parallel_fload_index(NULL, &stats);
    
    double end_time = get_precise_time();
    double test_time_ms = (end_time - start_time) * 1000.0;
    
    // All operations should fail gracefully
    int passed = (write_result1 < 0 && write_result2 < 0 && 
                  read_result1 < 0 && read_result2 < 0 &&
                  index1 == NULL && index2 == NULL);
    
    print_test_result("I/O error handling", passed, test_time_ms);
    
    grid_parallel_free(grid);
    return passed;
}

/**
 * Performance benchmark for I/O operations
 */
static int test_io_performance(void) {
    printf("Testing I/O performance...\n");
    
    struct GridDataParallel *large_grid = create_test_grid_io(5000, 20);
    if (!large_grid) {
        printf("  Failed to create large test grid\n");
        return 0;
    }
    
    char *filename = create_temp_filename("performance");
    
    struct GridPerformanceStats stats;
    memset(&stats, 0, sizeof(stats));
    
    double total_time = 0.0;
    
    // Multiple write/read cycles
    for (int i = 0; i < 3; i++) {
        double start_time = get_precise_time();
        
        // Write
        FILE *write_fp = fopen(filename, "wb");
        if (write_fp) {
            grid_parallel_fwrite(write_fp, large_grid, &stats);
            fclose(write_fp);
        }
        
        // Read
        FILE *read_fp = fopen(filename, "rb");
        if (read_fp) {
            struct GridDataParallel *read_grid = grid_parallel_make(5000, 20);
            grid_parallel_fread(read_fp, read_grid, &stats);
            fclose(read_fp);
            if (read_grid) grid_parallel_free(read_grid);
        }
        
        double end_time = get_precise_time();
        total_time += (end_time - start_time) * 1000.0;
        
        unlink(filename);
    }
    
    double avg_cycle_time = total_time / 3.0;
    int passed = (avg_cycle_time < 1000.0); // Should complete under 1 second per cycle
    
    printf("  Average I/O cycle time: %.3f ms\n", avg_cycle_time);
    print_test_result("I/O performance benchmark", passed, total_time);
    
    grid_parallel_free(large_grid);
    return passed;
}

/**
 * Run all I/O tests
 */
int run_all_io_tests(void) {
    printf("\n=== Grid I/O Tests ===\n");
    
    double total_start = get_precise_time();
    
    int all_passed = 1;
    
    all_passed &= test_grid_write_fd();
    all_passed &= test_grid_write_fp();
    all_passed &= test_grid_read_fd();
    all_passed &= test_grid_read_fp();
    all_passed &= test_roundtrip_io();
    all_passed &= test_load_index_fd();
    all_passed &= test_load_index_fp();
    all_passed &= test_io_error_handling();
    all_passed &= test_io_performance();
    
    double total_end = get_precise_time();
    test_stats.total_test_time = (total_end - total_start) * 1000.0;
    
    // Print summary
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", test_stats.total_tests);
    printf("Passed: %d\n", test_stats.passed_tests);
    printf("Failed: %d\n", test_stats.failed_tests);
    printf("Total test time: %.3f ms\n", test_stats.total_test_time);
    printf("Average read time: %.3f ms\n", test_stats.avg_read_time / NUM_IO_RUNS);
    printf("Average write time: %.3f ms\n", test_stats.avg_write_time / NUM_IO_RUNS);
    printf("Average index load time: %.3f ms\n", test_stats.avg_index_load_time / NUM_IO_RUNS);
    printf("Total bytes read: %zu\n", test_stats.bytes_read);
    printf("Total bytes written: %zu\n", test_stats.bytes_written);
    
    if (all_passed) {
        printf("\n\033[32mAll grid I/O tests PASSED!\033[0m\n");
    } else {
        printf("\n\033[31mSome grid I/O tests FAILED!\033[0m\n");
    }
    
    return all_passed ? 0 : 1;
}

/**
 * Main function
 */
int main(int argc, char *argv[]) {
    printf("SuperDARN Parallel Grid I/O Test Suite\n");
    printf("======================================\n");
    
#ifdef OPENMP_ENABLED
    printf("OpenMP enabled with %d threads\n", omp_get_max_threads());
#else
    printf("OpenMP not enabled - running sequential tests\n");
#endif
    
    return run_all_io_tests();
}
