#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/scandata.h"

/* 
 * Stub implementations of scandata functions for fit.1.35
 * These are minimal implementations to allow the code to compile
 */

struct RadarScan *RadarScanMake(void) {
    struct RadarScan *scan = (struct RadarScan *)calloc(1, sizeof(struct RadarScan));
    if (!scan) return NULL;
    
    // Initialize with default values
    scan->version.major = 1;
    scan->version.minor = 0;
    scan->nrang = 0;
    scan->rng = NULL;
    scan->data = NULL;
    scan->lag = NULL;
    scan->bm = NULL;
    scan->num = 0;
    
    return scan;
}

void RadarScanFree(struct RadarScan *ptr) {
    if (!ptr) return;
    
    // Free allocated arrays
    if (ptr->rng) free(ptr->rng);
    if (ptr->lag) free(ptr->lag);
    if (ptr->data) free(ptr->data);
    
    // Free beam data
    if (ptr->bm) {
        for (int i = 0; i < ptr->num; i++) {
            if (ptr->bm[i]) {
                if (ptr->bm[i]->sct) free(ptr->bm[i]->sct);
                if (ptr->bm[i]->rng) free(ptr->bm[i]->rng);
                free(ptr->bm[i]);
            }
        }
        free(ptr->bm);
    }
    
    // Free the structure itself
    free(ptr);
}

int RadarScanSetRng(struct RadarScan *ptr, int nrang) {
    if (!ptr) return -1;
    
    // If new size is the same as current, nothing to do
    if (nrang == ptr->nrang) return 0;
    
    // If new size is zero, free all data
    if (nrang <= 0) {
        if (ptr->rng) free(ptr->rng);
        if (ptr->data) free(ptr->data);
        ptr->rng = NULL;
        ptr->data = NULL;
        ptr->nrang = 0;
        return 0;
    }
    
    // Allocate new arrays with the new size
    int16_t *new_rng = (int16_t *)realloc(ptr->rng, nrang * sizeof(int16_t));
    if (!new_rng) return -1;
    
    float *new_data = (float *)realloc(ptr->data, nrang * sizeof(float) * 10); // Assuming 10 floats per range
    if (!new_data) {
        // If data allocation fails, free the new_rng and return error
        free(new_rng);
        return -1;
    }
    
    // Initialize new elements if we're growing the arrays
    if (nrang > ptr->nrang) {
        // Initialize new range gates
        for (int i = ptr->nrang; i < nrang; i++) {
            new_rng[i] = ptr->frang + i * ptr->rsep;
            // Initialize data to zero
            memset((char *)new_data + (i * 10 * sizeof(float)), 0, 10 * sizeof(float));
        }
    }
    
    // Update the structure with new arrays and sizes
    ptr->rng = (int *)new_rng;  // Cast to int* to match the structure definition
    ptr->data = new_data;
    ptr->nrang = nrang;
    
    return 0;
}

struct RadarBeam *RadarScanAddBeam(struct RadarScan *ptr, int nrang) {
    if (!ptr) return NULL;
    
    // Allocate memory for a new beam
    struct RadarBeam *beam = (struct RadarBeam *)calloc(1, sizeof(struct RadarBeam));
    if (!beam) return NULL;
    
    // Initialize the beam
    beam->nrang = nrang;
    beam->sct = (int *)calloc(nrang, sizeof(int));
    beam->rng = (RadarRange *)calloc(nrang, sizeof(RadarRange));
    
    if (!beam->sct || !beam->rng) {
        if (beam->sct) free(beam->sct);
        if (beam->rng) free(beam->rng);
        free(beam);
        return NULL;
    }
    
    // Add the beam to the scan
    int new_num = ptr->num + 1;
    struct RadarBeam **new_bm = (struct RadarBeam **)realloc(ptr->bm, new_num * sizeof(struct RadarBeam *));
    if (!new_bm) {
        free(beam->sct);
        free(beam->rng);
        free(beam);
        return NULL;
    }
    
    ptr->bm = new_bm;
    ptr->bm[ptr->num] = beam;
    ptr->num = new_num;
    
    return beam;
}

/**
 * Reset a RadarScan structure to its initial state
 * 
 * @param ptr Pointer to the RadarScan structure to reset
 */
void RadarScanReset(struct RadarScan *ptr) {
    if (!ptr) return;
    
    // Free all beams
    if (ptr->bm) {
        for (int i = 0; i < ptr->num; i++) {
            if (ptr->bm[i]) {
                if (ptr->bm[i]->sct) free(ptr->bm[i]->sct);
                if (ptr->bm[i]->rng) free(ptr->bm[i]->rng);
                free(ptr->bm[i]);
            }
        }
        free(ptr->bm);
        ptr->bm = NULL;
        ptr->num = 0;
    }
    
    // Reset other fields
    ptr->st_time = 0.0;
    ptr->ed_time = 0.0;
    ptr->nrang = 0;
    
    // Reset time structure
    memset(&ptr->time, 0, sizeof(ptr->time));
    
    // Reset version
    ptr->version.major = 1;
    ptr->version.minor = 0;
}
