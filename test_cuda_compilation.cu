// Simple CUDA compilation test
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel() {
    printf("CUDA test kernel executed\n");
}

int main() {
    printf("Testing CUDA compilation...\n");
    test_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("CUDA compilation test successful!\n");
    return 0;
}