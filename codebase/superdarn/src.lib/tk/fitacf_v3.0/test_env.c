#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    printf("=================================\n");
    printf("FitACF v3.0 Docker Environment Test\n");
    printf("=================================\n");
    
    // Test basic C compilation
    printf("✓ C compilation working\n");
    
    // Test OpenMP
    #ifdef _OPENMP
    printf("✓ OpenMP enabled\n");
    printf("  Max threads: %d\n", omp_get_max_threads());
    #else
    printf("✗ OpenMP not available\n");
    #endif
    
    // Test timing functions
    clock_t start = clock();
    volatile int sum = 0;
    for(int i = 0; i < 1000000; i++) {
        sum += i;
    }
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("✓ Timing test: %f seconds\n", cpu_time);
    
    printf("=================================\n");
    printf("Environment test completed successfully!\n");
    return 0;
}
