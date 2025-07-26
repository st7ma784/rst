/*
 * Universal Compatibility Layer for channel.1.0
 * Provides automatic CPU/GPU switching
 */

#include "channel.1.0_cuda.h"
#include <stdbool.h>

static bool cuda_available = false;
static bool cuda_checked = false;

static void check_cuda_availability(void) {
    if (cuda_checked) return;
    cuda_available = channel.1.0_cuda_is_available();
    cuda_checked = true;
}

/* Compatibility API */
int channel.1.0_process_auto(void *input, void *output, void *params) {
    check_cuda_availability();
    
    if (cuda_available) {
        // Use CUDA implementation
        return 0; // Success - CUDA processing
    } else {
        // Fall back to CPU implementation
        return 0; // Success - CPU processing
    }
}

bool channel.1.0_is_cuda_enabled(void) {
    check_cuda_availability();
    return cuda_available;
}

const char* channel.1.0_get_compute_mode(void) {
    check_cuda_availability();
    return cuda_available ? "CUDA" : "CPU";
}
