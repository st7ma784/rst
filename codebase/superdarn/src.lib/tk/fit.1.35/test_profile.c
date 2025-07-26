#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>     // For math functions
#include <unistd.h>   // For getopt
#include <ctype.h>    // For isprint
#include <stdatomic.h> // For memory barriers
#include <sys/time.h>  // For gettimeofday
#include <strings.h>   // For bzero, bcopy, etc.
#include "include/rtypes.h"
#include "include/fitdata.h"
#include "include/fitcfit.h"
#include "include/cfitdata.h"
#include "include/fitread.h"
#include "include/rprm.h"  /* For RadarParm structure */

/* Helper function to dump memory contents */
void dump_memory(uint8_t *addr, size_t size, uintptr_t base_addr) {
    const int bytes_per_line = 16;
    
    for (size_t i = 0; i < size; i += bytes_per_line) {
        // Print address
        printf("%08zX: ", (uintptr_t)(addr + i) - base_addr);
        
        // Print hex values
        for (int j = 0; j < bytes_per_line; j++) {
            if (i + j < size) {
                printf("%02X ", addr[i + j]);
            } else {
                printf("   ");
            }
            
            if (j == 7) printf(" ");  // Extra space after 8 bytes
        }
        
        printf(" |");
        
        // Print ASCII values
        for (int j = 0; j < bytes_per_line && i + j < size; j++) {
            uint8_t c = addr[i + j];
            printf("%c", (c >= 32 && c <= 126) ? c : '.');
        }
        
        printf("|\n");
    }
}

// Function to validate RadarParm structure
void validate_radar_parm(const RadarParm *prm, const char *context) {
    printf("\n=== Validating RadarParm at %s ===\n", context);
    printf("Address: %p, Size: %zu bytes\n", (void *)prm, sizeof(*prm));
    
    // Print detailed structure information
    printf("RadarParm structure layout:\n");
    printf("  revision: major=%d, minor=%d\n", prm->revision.major, prm->revision.minor);
    printf("  origin: code=%d, time='%s', command='%s'\n", 
           prm->origin.code, prm->origin.time, prm->origin.command);
    printf("  cp=%d, stid=%d\n", prm->cp, prm->stid);
    printf("  time: %04d-%02d-%02d %02d:%02d:%02d.%06d\n",
           prm->time.yr, prm->time.mo, prm->time.dy,
           prm->time.hr, prm->time.mt, prm->time.sc, prm->time.us);
    printf("  txpow=%d, nave=%d, bmnum=%d, bmazm=%.1f\n",
           prm->txpow, prm->nave, prm->bmnum, prm->bmazm);
    printf("  nrang=%d, num=%d, rng=%p\n", prm->nrang, prm->num, (void *)prm->rng);
    printf("  frang=%d, rsep=%d, mpinc=%d, mppul=%d, mplgs=%d\n",
           prm->frang, prm->rsep, prm->mpinc, prm->mppul, prm->mplgs);
    
    // Check if nrang and num are within reasonable bounds
    if (prm->nrang < 1 || prm->nrang > 100000) {
        printf("  WARNING: Invalid nrang=%d (expected 1-100000)\n", prm->nrang);
    }
    
    if (prm->num < 1 || prm->num > 100000) {
        printf("  WARNING: Invalid num=%d (expected 1-100000)\n", prm->num);
    }
    
    // Check if range pointer is valid
    if (prm->rng == NULL) {
        printf("  WARNING: NULL range pointer\n");
    } else {
        printf("  Range array at %p (%s)\n", 
               (void *)prm->rng,
               ((uintptr_t)prm->rng >= (uintptr_t)prm && 
                (uintptr_t)prm->rng < (uintptr_t)prm + sizeof(*prm)) ? 
               "within structure" : "external allocation");
        
        // If we have a valid range array, check its contents
        if (prm->nrang > 0 && prm->nrang <= 100000) {
            printf("  First few range values: ");
            int num_to_print = (prm->nrang < 5) ? prm->nrang : 5;
            for (int i = 0; i < num_to_print; i++) {
                printf("%d ", prm->rng[i]);
            }
            printf("\n");
        }
    }
    
    // Check guard words if they exist
    if (prm->rng != NULL && prm->nrang > 0) {
        // Try to find guard words before and after the array
        uint64_t *before = (uint64_t *)((uint8_t *)prm->rng - sizeof(uint64_t));
        uint64_t *after = (uint64_t *)((uint8_t *)prm->rng + prm->nrang * sizeof(int16_t));
        
        if (*before == 0xDEADBEEFDEADBEEF) {
            printf("  Guard word before range array is intact\n");
        } else {
            printf("  WARNING: Guard word before range array is corrupted: 0x%016lX\n", 
                   (unsigned long)*before);
        }
        
        if (*after == 0xDEADBEEFDEADBEEF) {
            printf("  Guard word after range array is intact\n");
        } else {
            printf("  WARNING: Guard word after range array is corrupted: 0x%016lX\n", 
                   (unsigned long)*after);
        }
    }
    
    // Check guard words if range array is allocated
    if (prm->rng != NULL) {
        uint64_t *guard_before = (uint64_t *)((char *)prm->rng - sizeof(uint64_t));
        size_t alloc_size = prm->nrang * sizeof(int16_t);
        uint64_t *guard_after = (uint64_t *)((char *)prm->rng + alloc_size);
        
        printf("\nRange array memory layout (with guard words):\n");
        printf("  Guard before: %p, Range array: %p, Guard after: %p\n", 
               (void*)guard_before, (void*)prm->rng, (void*)guard_after);
        
        // Check guard words
        if (*guard_before != 0xDEADBEEFDEADBEEF) {
            printf("  CORRUPTION: Guard word before range array is corrupted!\n");
            printf("    Expected: 0xDEADBEEFDEADBEEF, Got: 0x%016lX\n", *guard_before);
            
            // Dump memory around the corrupted guard word
            printf("\n  Memory around guard_before (%p):\n", (void*)guard_before);
            dump_memory((uint8_t*)((char*)guard_before - 32), 64, (uintptr_t)guard_before - 32);
        } else {
            printf("  Guard word before range array is valid\n");
        }
        
        if (*guard_after != 0xDEADBEEFDEADBEEF) {
            printf("  CORRUPTION: Guard word after range array is corrupted!\n");
            printf("    Expected: 0xDEADBEEFDEADBEEF, Got: 0x%016lX\n", *guard_after);
            
            // Dump memory around the corrupted guard word
            printf("\n  Memory around guard_after (%p):\n", (void*)guard_after);
            dump_memory((uint8_t*)((char*)guard_after - 32), 64, (uintptr_t)guard_after - 32);
        } else {
            printf("  Guard word after range array is valid\n");
        }
    }
    
    // Check for potential buffer overflows in the RadarParm structure
    if (prm->nrang != prm->num) {
        printf("  WARNING: nrang (%d) != num (%d)\n", prm->nrang, prm->num);
    }
    
    // Check for other suspicious values
    if (prm->tfreq < 8000 || prm->tfreq > 20000) {  // Typical frequency range in kHz
        printf("  WARNING: Suspicious tfreq value: %d kHz\n", prm->tfreq);
    }
    
    if (prm->bmazm < 0.0 || prm->bmazm >= 360.0) {
        printf("  WARNING: Invalid beam azimuth: %.2f degrees\n", prm->bmazm);
    }
    
    printf("=== End of validation ===\n\n");
}

// dump_memory function is already defined at the top of the file

/**
 * Dump memory in hex format with ASCII representation
 * 
 * @param data Pointer to the start of memory to dump
 * @param size Number of bytes to dump
 * @param base_addr Base address for offset calculation (can be NULL)
 */
void dump_memory_hex(const uint8_t *data, size_t size, const uint8_t *base_addr) {
    const size_t bytes_per_line = 16;
    
    for (size_t offset = 0; offset < size; offset += bytes_per_line) {
        // Print address/offset
        if (base_addr) {
            printf("%08lx: ", (unsigned long)(data - base_addr + offset));
        } else {
            printf("%p: ", (void *)(data + offset));
        }
        
        // Print hex bytes
        for (size_t i = 0; i < bytes_per_line; i++) {
            if (offset + i < size) {
                printf("%02x ", data[offset + i]);
            } else {
                printf("   ");
            }
            
            // Add an extra space after 8 bytes for readability
            if (i == 7) printf(" ");
        }
        
        printf(" ");
        
        // Print ASCII representation
        for (size_t i = 0; i < bytes_per_line && offset + i < size; i++) {
            uint8_t c = data[offset + i];
            printf("%c", (c >= 32 && c <= 126) ? c : '.');
        }
        
        printf("\n");
    }
}

// Function to get current time in microseconds
long long current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000000LL + te.tv_usec;
}

// Function to generate test data
void generate_test_data(struct FitData *fit, int num_ranges) {
    printf("Debug: generate_test_data - Start (num_ranges=%d)\n", num_ranges);
    
    // Initialize the FitData structure
    printf("Debug: Setting algorithm...\n");
    FitSetAlgorithm(fit, "test");
    
    printf("Debug: Allocating range array...\n");
    if (FitSetRng(fit, num_ranges) != 0) {
        printf("Error: Failed to allocate range array in generate_test_data\n");
        return;
    }
    
    printf("Debug: Filling with random data...\n");
    // Fill with random data
    for (int i = 0; i < num_ranges; i++) {
        if (fit->rng == NULL) {
            printf("Error: fit->rng is NULL at index %d\n", i);
            return;
        }
        
        fit->rng[i].qflg = 1;  // Mark as valid
        fit->rng[i].p_0 = (double)rand() / RAND_MAX * 100.0;  // Random power 0-100
        fit->rng[i].v = (double)rand() / RAND_MAX * 2000.0 - 1000.0;  // Random velocity -1000 to 1000
        fit->rng[i].v_err = (double)rand() / RAND_MAX * 100.0;  // Random error 0-100
        fit->rng[i].p_l = (double)rand() / RAND_MAX * 100.0;  // Random power 0-100
        fit->rng[i].p_l_err = (double)rand() / RAND_MAX * 10.0;  // Random error 0-10
        fit->rng[i].gsct = rand() % 10;  // Random ground scatter flag
    }
    printf("Debug: generate_test_data - Done\n");
}

// Function to safely initialize and validate RadarParm
static RadarParm *init_radar_parm(int num_ranges) {
    printf("\n=== Initializing RadarParm with guard pages ===\n");
    
    // Calculate total size needed with guard pages
    const size_t guard_size = 4096;  // One page
    const size_t total_size = sizeof(RadarParm) + 2 * guard_size;
    
    // Allocate memory with guard pages
    uint8_t *prm_mem = (uint8_t *)malloc(total_size);
    if (!prm_mem) {
        printf("Failed to allocate memory for RadarParm with guard pages\n");
        return NULL;
    }
    
    // Set up guard pages
    memset(prm_mem, 0xCC, guard_size);  // Before guard (0xCC pattern)
    memset(prm_mem + guard_size, 0, sizeof(RadarParm));  // Actual structure (zeroed)
    memset(prm_mem + guard_size + sizeof(RadarParm), 0xDD, guard_size);  // After guard (0xDD pattern)
    
    // Get pointer to the actual structure
    RadarParm *prm = (RadarParm *)(prm_mem + guard_size);
    
    // Store guard pointers for later verification
    uint64_t *guard_before = (uint64_t *)((uint8_t *)prm - 8);
    uint64_t *guard_after = (uint64_t *)((uint8_t *)prm + sizeof(RadarParm));
    
    printf("RadarParm allocated at %p\n", (void *)prm);
    printf("Guard before at %p, Guard after at %p\n", (void *)guard_before, (void *)guard_after);
    
    // Initialize RadarParm fields
    prm->revision.major = 1;
    prm->revision.minor = 0;
    prm->origin.code = 1;
    
    // Initialize time as a string (assuming time is a char array in RadarParm)
    char time_str[32] = "2025-07-19 22:04:16.000000";
    memcpy(prm->time, time_str, strlen(time_str) + 1);
    
    // Initialize command (store in comment buffer if command field doesn't exist)
    char cmd_str[32] = "test_profile";
    memcpy(prm->combf, cmd_str, strlen(cmd_str) + 1);
    
    prm->cp = 123;
    prm->stid = 1;
    prm->txpow = 1000;
    prm->nave = 50;
    prm->nrang = num_ranges;
    prm->num = num_ranges;
    prm->frang = 180;
    prm->rsep = 45;
    prm->mppul = 1;
    prm->mplgs = 1;
    
    // Verify guard pages are intact
    printf("Verifying guard pages...\n");
    bool guards_ok = true;
    for (size_t i = 0; i < 8; i++) {  // Check first 8 bytes of each guard
        if (((uint8_t *)guard_before)[i] != 0xCC) {
            printf("  ERROR: Before guard corrupted at offset %zu: 0x%02x\n", 
                   i, ((uint8_t *)guard_before)[i]);
            guards_ok = false;
        }
        if (((uint8_t *)guard_after)[i] != 0xDD) {
            printf("  ERROR: After guard corrupted at offset %zu: 0x%02x\n", 
                   i, ((uint8_t *)guard_after)[i]);
            guards_ok = false;
        }
    }
    
    if (!guards_ok) {
        printf("\n=== Memory Dump (around RadarParm) ===\n");
        dump_memory_hex((uint8_t *)prm - 32, 128, (uint8_t *)prm);
        free(prm_mem);
        return NULL;
    }
    
    // Store the original allocation pointer in the first unused field we can find
    // This is a hack to track the original pointer for proper cleanup
    memcpy(&prm->origin.time[16], &prm_mem, sizeof(void *));
    
    return prm;
}

// Function to safely free RadarParm
static void free_radar_parm(RadarParm *prm) {
    if (!prm) return;
    
    // Retrieve the original allocation pointer
    void *prm_mem = NULL;
    memcpy(&prm_mem, &prm->origin.time[16], sizeof(void *));
    
    if (prm_mem) {
        printf("Freeing RadarParm at %p (original allocation: %p)\n", 
               (void *)prm, prm_mem);
        free(prm_mem);
    } else {
        printf("Warning: Could not determine original allocation for RadarParm at %p\n", 
               (void *)prm);
        free(prm);
    }
}

int main(int argc, char *argv[]) {
    // Test with different range sizes
    int test_sizes[] = {100, 1000, 10000, 50000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int num_iterations = 1000;
    
    printf("=== Fit.1.35 Memory Corruption Test ===\n");
    
    for (int i = 0; i < num_tests; i++) {
        int num_ranges = test_sizes[i];
        
        printf("\n--- Test %d: %d ranges, %d iterations ---\n", 
               i + 1, num_ranges, num_iterations);
        
        // Initialize RadarParm with guard pages
        RadarParm *prm = init_radar_parm(num_ranges);
        if (!prm) {
            printf("Failed to initialize RadarParm\n");
            continue;
        }
        
        // Allocate FitData
        struct FitData *fit = FitMake();
        if (!fit) {
            printf("Failed to allocate FitData\n");
            free_radar_parm(prm);
            continue;
        }
        
        // Allocate the range array
        if (FitSetRng(fit, num_ranges) != 0) {
            printf("Failed to allocate range array\n");
            FitFree(fit);
            free_radar_parm(prm);
            continue;
        }
        
        // Generate test data
        generate_test_data(fit, num_ranges);
        
        // Profile FitToCFit
        struct CFitdata *cfit = CFitMake();
        if (!cfit) {
            printf("Failed to allocate CFitdata\n");
            FitFree(fit);
            free_radar_parm(prm);
            continue;
        }
        
        // Declare variables that were being used but not declared
        int result;
        struct timeval start_time, end_time;
        long long total_time = 0;
        
            // Dump memory around the RadarParm structure for analysis
            printf("\n=== Memory Dump (RadarParm at %p) ===\n", (void *)prm);
            dump_memory_hex((uint8_t *)prm - 32, 256, (uint8_t *)prm);
            printf("\nFull RadarParm structure dump (first 256 bytes):\n");
            dump_memory((uint8_t*)prm, 256, (uintptr_t)prm);
            
            RadarParmFree(prm);
            continue;
        }
        
        // Verify range count is valid
        if (prm->nrang <= 0 || prm->nrang > 100000) {
            printf("ERROR: Invalid range count: %d (expected 1-100000)\n", prm->nrang);
            RadarParmFree(prm);
            continue;
        }
        
        // Initialize the time structure
        prm->time.yr = 2023;
        prm->time.mo = 1;
        prm->time.dy = 1;
        prm->time.hr = 0;
        prm->time.mt = 0;
        prm->time.sc = 0;
        prm->time.us = 0;
        
        // Set radar parameters
        prm->revision.major = 1;  // Set version
        prm->revision.minor = 0;
        prm->stid = 1;  // Station ID
        prm->channel = 0;  // Channel number
        prm->bmnum = 0;  // Beam number
        prm->bmazm = 0.0;  // Beam azimuth
        prm->scan = 0;  // Scan flag
        prm->rxrise = 0;  // Receiver rise time
        prm->intt.sc = 1;  // Integration time (seconds)
        prm->intt.us = 0;  // Integration time (microseconds)
        prm->txpl = 300;  // Transmit pulse length (microseconds)
        prm->mpinc = 3000;  // Multi-pulse increment (microseconds)
        prm->mppul = 1;  // Number of pulses in sequence
        prm->mplgs = 1;  // Number of lags in sequence
        prm->frang = 180;  // Distance to first range (km)
        prm->rsep = 45;  // Range separation (km)
        prm->xcf = 0;  // XCF flag
        prm->tfreq = 10500;  // Transmit frequency (kHz)
        prm->offset = 0;  // Offset between channels for stereo radar
        
        printf("Debug: Initialized RadarParm at %p: nrang=%d, num=%d, rng=%p\n", 
               (void*)prm, prm->nrang, prm->num, (void*)prm->rng);
        printf("Debug: First few range gates: %d, %d, %d\n", 
               prm->rng[0], prm->rng[1], prm->rng[2]);
        
        // Profile FitToCFit
        long long start = current_timestamp();
        for (int j = 0; j < num_iterations; j++) {
            // Print RadarParm info before the call
            printf("Debug: Before FitToCFit[%d]: prm=%p, nrang=%d, num=%d, rng=%p\n", 
                   j, (void*)prm, prm->nrang, prm->num, (void*)prm->rng);
            
            // Call FitToCFit with the test data
            int result = FitToCFit(0.0, cfit, prm, fit);
            
            // Print RadarParm info after the call
            printf("Debug: After FitToCFit[%d]: prm=%p, nrang=%d, num=%d, rng=%p\n", 
                   j, (void*)prm, prm->nrang, prm->num, (void*)prm->rng);
            
            if (result != 0) {
                printf("Error: FitToCFit returned %d\n", result);
                break;
            }
        }
        
        // Free the range array with guard word validation
        if (prm->rng != NULL) {
            printf("\n=== Range Array Deallocation ===\n");
            
            // Calculate the start of the metadata block
            // Metadata is stored before the guard word, which is before the range array
            uint8_t *metadata_ptr = (uint8_t *)prm->rng - sizeof(uint64_t) - (sizeof(void*) + sizeof(size_t) + sizeof(size_t) + sizeof(int));
            
            // Get the metadata
            uint8_t *raw_mem = *((void **)metadata_ptr);
            metadata_ptr += sizeof(void*);
            metadata_ptr += sizeof(size_t);  // Skip padding field
            size_t stored_alloc_size = *((size_t *)metadata_ptr);
            metadata_ptr += sizeof(size_t);
            int stored_num_ranges = *((int *)metadata_ptr);
            
            // Get the guard words
            uint8_t *guard_before_ptr = (uint8_t *)metadata_ptr + sizeof(int);
            uint64_t *guard_before = (uint64_t *)guard_before_ptr;
            uint64_t *guard_after = (uint64_t *)((uint8_t *)prm->rng + (stored_alloc_size / sizeof(int16_t) * sizeof(int16_t)));
            
            // Debug information
            printf("Range array:        %p\n", (void*)prm->rng);
            printf("Raw pointer:        %p\n", (void*)raw_mem);
            printf("Metadata:           %p\n", (void*)(prm->rng - sizeof(uint64_t) - (sizeof(void*) + sizeof(size_t) + sizeof(size_t) + sizeof(int))));
            printf("Guard before:       %p (value: 0x%016lX)\n", (void*)guard_before, *guard_before);
            printf("Guard after:        %p (value: 0x%016lX)\n", (void*)guard_after, *guard_after);
            printf("Stored alloc size:  %zu bytes\n", stored_alloc_size);
            printf("Stored num_ranges:  %d\n", stored_num_ranges);
            
            // Validate the guard words
            int corruption_detected = 0;
            
            if (*guard_before != 0xDEADBEEFDEADBEEF) {
                printf("ERROR: Memory corruption detected before range array!\n");
                printf("  Expected: 0x%016lX, Actual: 0x%016lX\n", 
                       0xDEADBEEFDEADBEEF, *guard_before);
                corruption_detected = 1;
                
                // Dump memory around the corrupted guard word
                printf("\nMemory before guard word (first 32 bytes):\n");
                dump_memory(guard_before_ptr - 32, 64, (uintptr_t)(guard_before_ptr - 32));
            }
            
            if (*guard_after != 0xDEADBEEFDEADBEEF) {
                printf("ERROR: Memory corruption detected after range array!\n");
                printf("  Expected: 0x%016lX, Actual: 0x%016lX\n", 
                       0xDEADBEEFDEADBEEF, *guard_after);
                corruption_detected = 1;
                
                // Dump memory around the corrupted guard word
                printf("\nMemory around guard_after (first 32 bytes):\n");
                dump_memory((uint8_t *)guard_after - 32, 64, (uintptr_t)((uint8_t *)guard_after - 32));
            }
            
            // Validate the stored allocation size
            size_t expected_alloc_size = stored_num_ranges * sizeof(int16_t);
            if (stored_alloc_size != expected_alloc_size) {
                printf("WARNING: Stored allocation size mismatch!\n");
                printf("  Expected: %zu, Actual: %zu\n", 
                       expected_alloc_size, stored_alloc_size);
            }
            
            // Free the original allocation
            printf("Freeing range array at %p (raw allocation at %p)\n", 
                   (void*)prm->rng, (void*)raw_mem);
            free(raw_mem);
            prm->rng = NULL;
            
            if (corruption_detected) {
                printf("WARNING: Memory corruption was detected!\n");
            } else {
                printf("Range array freed successfully\n");
            }
        }
        
        // Free the RadarParm structure with guard page checking
        if (prm) {
            // Check guard pages before freeing
            uint8_t *full_allocation = (uint8_t*)prm - 4096;
            uint8_t *before_guard = full_allocation;
            uint8_t *after_guard = (uint8_t*)prm + sizeof(RadarParm);
            
            printf("\n=== Before freeing RadarParm ===\n");
            printf("Checking guard pages for corruption...\n");
            
            // Check guard page before
            for (size_t i = 0; i < 64; i++) {  // Only check first 64 bytes for corruption
                if (before_guard[i] != 0xCC) {
                    printf("ERROR: Guard page before corrupted at offset %zu (0x%02X != 0xCC)\n", 
                           i, before_guard[i]);
                    break;
                }
            }
            
            // Check guard page after
            for (size_t i = 0; i < 64; i++) {  // Only check first 64 bytes for corruption
                if (after_guard[i] != 0xDD) {
                    printf("ERROR: Guard page after corrupted at offset %zu (0x%02X != 0xDD)\n", 
                           i, after_guard[i]);
                    break;
                }
            }
            
            // Free the saved allocation pointer
            if (saved_allocation) {
                free(saved_allocation);
                saved_allocation = NULL;
                printf("RadarParm freed successfully\n");
            }
}

// Declare variables that were being used but not declared
int result;
struct timeval start_time, end_time;
long long total_time = 0;

// Dump memory around the RadarParm structure for analysis
printf("\n=== Memory Dump (RadarParm at %p) ===\n", (void *)prm);
dump_memory_hex((uint8_t *)prm - 32, 256, (uint8_t *)prm);
printf("\nFull RadarParm structure dump (first 256 bytes):\n");
dump_memory((uint8_t*)prm, 256, (uintptr_t)prm);

RadarParmFree(prm);
continue;
}

    // Main test loop
    for (int i = 0; i < num_tests; i++) {
        int num_ranges = test_sizes[i];
        
        printf("\n--- Test %d: %d ranges, %d iterations ---\n", 
               i + 1, num_ranges, num_iterations);
        
        // Initialize RadarParm with guard pages
        RadarParm *prm = init_radar_parm(num_ranges);
        if (!prm) {
            printf("Failed to initialize RadarParm\n");
            continue;
        }
        
        // Allocate FitData
        struct FitData *fit = FitMake();
        if (!fit) {
            printf("Failed to allocate FitData\n");
            free_radar_parm(prm);
            continue;
        }
        
        // Allocate the range array
        if (FitSetRng(fit, num_ranges) != 0) {
            printf("Failed to allocate range array\n");
            FitFree(fit);
            free_radar_parm(prm);
            continue;
        }
        
        // Generate test data
        generate_test_data(fit, num_ranges);
        
        // Profile FitToCFit
        struct CFitdata *cfit = CFitMake();
        if (!cfit) {
            printf("Failed to allocate CFitdata\n");
            FitFree(fit);
            free_radar_parm(prm);
            continue;
        }
        
        // Profile FitToCFit
        printf("\n=== Running FitToCFit benchmark ===\n");
        long long start = current_timestamp();
        
        for (int j = 0; j < num_iterations; j++) {
            int result = FitToCFit(0.0, cfit, prm, fit);
            if (result != 0) {
                printf("Error: FitToCFit returned %d\n", result);
                break;
            }
        }
        
        long long end = current_timestamp();
        double total_time = (end - start) / 1000.0;  // Convert to milliseconds
        double avg_time = total_time / num_iterations;
        double ranges_per_second = (num_ranges * num_iterations) / (total_time / 1000.0);
        
        printf("\n=== Results ===\n");
        printf("Ranges processed: %d\n", num_ranges * num_iterations);
        printf("Total time: %.3f ms\n", total_time);
        printf("Avg time per iteration: %.3f ms\n", avg_time);
        printf("Ranges per second: %.0f\n", ranges_per_second);
        
        // Free resources
        CFitFree(cfit);
        FitFree(fit);
        free_radar_parm(prm);
    }
    
    printf("\n=== Profiling Complete ===\n");
    return 0;
