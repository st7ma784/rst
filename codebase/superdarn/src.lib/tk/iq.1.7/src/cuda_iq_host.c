/* cuda_iq_host.c
   ===============
   Host wrapper functions for CUDA-accelerated IQ processing
   Author: R.J.Barnes (CUDA implementation)
*/

/*
  Copyright (c) 2012 The Johns Hopkins University/Applied Physics Laboratory
 
This file is part of the Radar Software Toolkit (RST).

RST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Modifications:
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "iq.h"
#include "iq.1.7_cuda.h"
#include "datamap.h"

/* Global settings for CUDA acceleration */
static bool cuda_acceleration_enabled = true;
static bool cuda_debug_mode = false;

/* Memory management functions */

cuda_iq_data_t* cuda_iq_data_create(int seqnum) {
    cuda_iq_data_t *iq_data = (cuda_iq_data_t*)malloc(sizeof(cuda_iq_data_t));
    if (!iq_data) return NULL;
    
    memset(iq_data, 0, sizeof(cuda_iq_data_t));
    iq_data->seqnum = seqnum;
    
    return iq_data;
}

void cuda_iq_data_destroy(cuda_iq_data_t *iq_data) {
    if (!iq_data) return;
    
    cuda_iq_data_free_arrays(iq_data);
    free(iq_data);
}

cudaError_t cuda_iq_data_allocate_arrays(cuda_iq_data_t *iq_data) {
    if (!iq_data || iq_data->seqnum <= 0) return cudaErrorInvalidValue;
    
    int seqnum = iq_data->seqnum;
    
    // Allocate unified memory for all arrays
    cudaError_t err;
    
    err = cudaMallocManaged(&iq_data->tv_sec, seqnum * sizeof(long));
    if (err != cudaSuccess) return err;
    
    err = cudaMallocManaged(&iq_data->tv_nsec, seqnum * sizeof(long));
    if (err != cudaSuccess) return err;
    
    err = cudaMallocManaged(&iq_data->atten, seqnum * sizeof(int));
    if (err != cudaSuccess) return err;
    
    err = cudaMallocManaged(&iq_data->noise, seqnum * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMallocManaged(&iq_data->offset, seqnum * sizeof(int));
    if (err != cudaSuccess) return err;
    
    err = cudaMallocManaged(&iq_data->size, seqnum * sizeof(int));
    if (err != cudaSuccess) return err;
    
    err = cudaMallocManaged(&iq_data->badtr, seqnum * sizeof(int));
    if (err != cudaSuccess) return err;
    
    // Initialize arrays to zero
    memset(iq_data->tv_sec, 0, seqnum * sizeof(long));
    memset(iq_data->tv_nsec, 0, seqnum * sizeof(long));
    memset(iq_data->atten, 0, seqnum * sizeof(int));
    memset(iq_data->noise, 0, seqnum * sizeof(float));
    memset(iq_data->offset, 0, seqnum * sizeof(int));
    memset(iq_data->size, 0, seqnum * sizeof(int));
    memset(iq_data->badtr, 0, seqnum * sizeof(int));
    
    return cudaSuccess;
}

void cuda_iq_data_free_arrays(cuda_iq_data_t *iq_data) {
    if (!iq_data) return;
    
    if (iq_data->tv_sec) { cudaFree(iq_data->tv_sec); iq_data->tv_sec = NULL; }
    if (iq_data->tv_nsec) { cudaFree(iq_data->tv_nsec); iq_data->tv_nsec = NULL; }
    if (iq_data->atten) { cudaFree(iq_data->atten); iq_data->atten = NULL; }
    if (iq_data->noise) { cudaFree(iq_data->noise); iq_data->noise = NULL; }
    if (iq_data->offset) { cudaFree(iq_data->offset); iq_data->offset = NULL; }
    if (iq_data->size) { cudaFree(iq_data->size); iq_data->size = NULL; }
    if (iq_data->badtr) { cudaFree(iq_data->badtr); iq_data->badtr = NULL; }
}

/* Conversion functions between original IQ and CUDA IQ formats */

static int iq_to_cuda_iq(const struct IQ *iq, cuda_iq_data_t **cuda_iq) {
    if (!iq || !cuda_iq) return -1;
    
    *cuda_iq = cuda_iq_data_create(iq->seqnum);
    if (!*cuda_iq) return -1;
    
    cuda_iq_data_t *ciq = *cuda_iq;
    ciq->revision.major = iq->revision.major;
    ciq->revision.minor = iq->revision.minor;
    ciq->chnnum = iq->chnnum;
    ciq->smpnum = iq->smpnum;
    ciq->skpnum = iq->skpnum;
    ciq->seqnum = iq->seqnum;
    ciq->tbadtr = iq->tbadtr;
    
    if (cuda_iq_data_allocate_arrays(ciq) != cudaSuccess) {
        cuda_iq_data_destroy(ciq);
        *cuda_iq = NULL;
        return -1;
    }
    
    // Copy array data
    if (iq->tval) {
        for (int i = 0; i < iq->seqnum; i++) {
            ciq->tv_sec[i] = iq->tval[i].tv_sec;
            ciq->tv_nsec[i] = iq->tval[i].tv_nsec;
        }
    }
    
    if (iq->atten) {
        memcpy(ciq->atten, iq->atten, iq->seqnum * sizeof(int));
    }
    
    if (iq->noise) {
        memcpy(ciq->noise, iq->noise, iq->seqnum * sizeof(float));
    }
    
    if (iq->offset) {
        memcpy(ciq->offset, iq->offset, iq->seqnum * sizeof(int));
    }
    
    if (iq->size) {
        memcpy(ciq->size, iq->size, iq->seqnum * sizeof(int));
    }
    
    if (iq->badtr) {
        memcpy(ciq->badtr, iq->badtr, iq->seqnum * sizeof(int));
    }
    
    return 0;
}

static int cuda_iq_to_iq(const cuda_iq_data_t *cuda_iq, struct IQ *iq) {
    if (!cuda_iq || !iq) return -1;
    
    iq->revision.major = cuda_iq->revision.major;
    iq->revision.minor = cuda_iq->revision.minor;
    iq->chnnum = cuda_iq->chnnum;
    iq->smpnum = cuda_iq->smpnum;
    iq->skpnum = cuda_iq->skpnum;
    iq->seqnum = cuda_iq->seqnum;
    iq->tbadtr = cuda_iq->tbadtr;
    
    // Allocate arrays in original IQ structure
    if (cuda_iq->tv_sec && cuda_iq->tv_nsec) {
        iq->tval = (struct timespec*)malloc(iq->seqnum * sizeof(struct timespec));
        if (!iq->tval) return -1;
        
        for (int i = 0; i < iq->seqnum; i++) {
            iq->tval[i].tv_sec = cuda_iq->tv_sec[i];
            iq->tval[i].tv_nsec = cuda_iq->tv_nsec[i];
        }
    }
    
    if (cuda_iq->atten) {
        iq->atten = (int*)malloc(iq->seqnum * sizeof(int));
        if (!iq->atten) return -1;
        memcpy(iq->atten, cuda_iq->atten, iq->seqnum * sizeof(int));
    }
    
    if (cuda_iq->noise) {
        iq->noise = (float*)malloc(iq->seqnum * sizeof(float));
        if (!iq->noise) return -1;
        memcpy(iq->noise, cuda_iq->noise, iq->seqnum * sizeof(float));
    }
    
    if (cuda_iq->offset) {
        iq->offset = (int*)malloc(iq->seqnum * sizeof(int));
        if (!iq->offset) return -1;
        memcpy(iq->offset, cuda_iq->offset, iq->seqnum * sizeof(int));
    }
    
    if (cuda_iq->size) {
        iq->size = (int*)malloc(iq->seqnum * sizeof(int));
        if (!iq->size) return -1;
        memcpy(iq->size, cuda_iq->size, iq->seqnum * sizeof(int));
    }
    
    if (cuda_iq->badtr) {
        iq->badtr = (int*)malloc(iq->seqnum * sizeof(int));
        if (!iq->badtr) return -1;
        memcpy(iq->badtr, cuda_iq->badtr, iq->seqnum * sizeof(int));
    }
    
    return 0;
}

/* High-level processing function */

cudaError_t cuda_iq_process_sequences(cuda_iq_data_t *iq_data,
                                     const int16_t *raw_samples,
                                     bool enable_badtr_detection,
                                     int16_t saturation_threshold) {
    if (!iq_data) return cudaErrorInvalidValue;
    
    cudaError_t err = cudaSuccess;
    
    // Calculate total samples
    int total_samples = 0;
    for (int i = 0; i < iq_data->seqnum; i++) {
        total_samples += iq_data->size[i];
    }
    
    if (enable_badtr_detection && raw_samples) {
        // Allocate memory for bad transmit mask
        bool *badtr_mask;
        err = cudaMalloc(&badtr_mask, total_samples * sizeof(bool));
        if (err != cudaSuccess) return err;
        
        // Detect bad transmit samples
        err = cuda_iq_detect_badtr(raw_samples, badtr_mask, total_samples,
                                  -saturation_threshold, saturation_threshold);
        
        if (err == cudaSuccess) {
            // Count bad transmit samples per sequence
            bool *host_mask = (bool*)malloc(total_samples * sizeof(bool));
            if (host_mask) {
                cudaMemcpy(host_mask, badtr_mask, total_samples * sizeof(bool), 
                          cudaMemcpyDeviceToHost);
                
                int sample_offset = 0;
                iq_data->tbadtr = 0;
                
                for (int seq = 0; seq < iq_data->seqnum; seq++) {
                    int badtr_count = 0;
                    for (int s = 0; s < iq_data->size[seq]; s++) {
                        if (host_mask[sample_offset + s]) {
                            badtr_count++;
                        }
                    }
                    iq_data->badtr[seq] = badtr_count;
                    iq_data->tbadtr += badtr_count;
                    sample_offset += iq_data->size[seq];
                }
                
                free(host_mask);
            }
        }
        
        cudaFree(badtr_mask);
    }
    
    return err;
}

/* Validation functions */

bool cuda_iq_validate_data(const cuda_iq_data_t *iq_data) {
    if (!iq_data) return false;
    
    if (iq_data->seqnum <= 0 || iq_data->seqnum > 1000000) return false;
    if (iq_data->chnnum <= 0 || iq_data->chnnum > 16) return false;
    if (iq_data->smpnum < 0 || iq_data->smpnum > 1000000) return false;
    
    // Check that required arrays are allocated
    if (iq_data->seqnum > 0) {
        if (!iq_data->atten || !iq_data->noise || !iq_data->offset ||
            !iq_data->size || !iq_data->badtr) {
            return false;
        }
    }
    
    // Validate array contents
    for (int i = 0; i < iq_data->seqnum; i++) {
        if (iq_data->size[i] < 0 || iq_data->size[i] > 1000000) return false;
        if (iq_data->offset[i] < 0) return false;
        if (iq_data->badtr[i] < 0 || iq_data->badtr[i] > iq_data->size[i]) return false;
    }
    
    return true;
}

cudaError_t cuda_iq_check_memory_requirements(const cuda_iq_data_t *iq_data,
                                             size_t *required_bytes) {
    if (!iq_data || !required_bytes) return cudaErrorInvalidValue;
    
    *required_bytes = 0;
    
    // Calculate memory for arrays
    int seqnum = iq_data->seqnum;
    *required_bytes += seqnum * sizeof(long);    // tv_sec
    *required_bytes += seqnum * sizeof(long);    // tv_nsec
    *required_bytes += seqnum * sizeof(int);     // atten
    *required_bytes += seqnum * sizeof(float);   // noise
    *required_bytes += seqnum * sizeof(int);     // offset
    *required_bytes += seqnum * sizeof(int);     // size
    *required_bytes += seqnum * sizeof(int);     // badtr
    
    // Calculate memory for sample data
    for (int i = 0; i < seqnum; i++) {
        *required_bytes += iq_data->size[i] * sizeof(int16_t) * 2;  // I+Q samples
    }
    
    // Add overhead for CUDA contexts and temporary buffers
    *required_bytes += 64 * 1024 * 1024;  // 64MB overhead
    
    return cudaSuccess;
}

/* Backward compatibility wrapper functions */

// The original IQ functions are implemented in the existing C files.
// These are CUDA-accelerated versions that can be used as drop-in replacements.

int IQSetTime_CUDA(struct IQ *ptr, int seq, struct timespec *tval) {
    if (!ptr || seq < 0 || seq >= ptr->seqnum || !tval) return -1;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        // Fall back to original implementation
        return IQSetTime(ptr, seq, tval);
    }
    
    if (!ptr->tval) {
        ptr->tval = (struct timespec*)malloc(ptr->seqnum * sizeof(struct timespec));
        if (!ptr->tval) return -1;
    }
    
    ptr->tval[seq] = *tval;
    return 0;
}

int IQSetAtten_CUDA(struct IQ *ptr, int seq, int atten) {
    if (!ptr || seq < 0 || seq >= ptr->seqnum) return -1;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        return IQSetAtten(ptr, seq, atten);
    }
    
    if (!ptr->atten) {
        ptr->atten = (int*)malloc(ptr->seqnum * sizeof(int));
        if (!ptr->atten) return -1;
    }
    
    ptr->atten[seq] = atten;
    return 0;
}

int IQSetNoise_CUDA(struct IQ *ptr, int seq, float noise) {
    if (!ptr || seq < 0 || seq >= ptr->seqnum) return -1;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        return IQSetNoise(ptr, seq, noise);
    }
    
    if (!ptr->noise) {
        ptr->noise = (float*)malloc(ptr->seqnum * sizeof(float));
        if (!ptr->noise) return -1;
    }
    
    ptr->noise[seq] = noise;
    return 0;
}

int IQSetOffset_CUDA(struct IQ *ptr, int seq, int offset) {
    if (!ptr || seq < 0 || seq >= ptr->seqnum) return -1;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        return IQSetOffset(ptr, seq, offset);
    }
    
    if (!ptr->offset) {
        ptr->offset = (int*)malloc(ptr->seqnum * sizeof(int));
        if (!ptr->offset) return -1;
    }
    
    ptr->offset[seq] = offset;
    return 0;
}

int IQSetSize_CUDA(struct IQ *ptr, int seq, int size) {
    if (!ptr || seq < 0 || seq >= ptr->seqnum) return -1;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        return IQSetSize(ptr, seq, size);
    }
    
    if (!ptr->size) {
        ptr->size = (int*)malloc(ptr->seqnum * sizeof(int));
        if (!ptr->size) return -1;
    }
    
    ptr->size[seq] = size;
    return 0;
}

int IQSetBadTR_CUDA(struct IQ *ptr, int seq, int badtr) {
    if (!ptr || seq < 0 || seq >= ptr->seqnum) return -1;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        return IQSetBadTR(ptr, seq, badtr);
    }
    
    if (!ptr->badtr) {
        ptr->badtr = (int*)malloc(ptr->seqnum * sizeof(int));
        if (!ptr->badtr) return -1;
    }
    
    ptr->badtr[seq] = badtr;
    return 0;
}

char *IQFlatten_CUDA(struct IQ *ptr, int num, int *size) {
    if (!ptr || !size) return NULL;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        return IQFlatten(ptr, num, size);
    }
    
    // Convert to CUDA format
    cuda_iq_data_t *cuda_iq;
    if (iq_to_cuda_iq(ptr, &cuda_iq) != 0) {
        return IQFlatten(ptr, num, size);  // Fall back
    }
    
    // Calculate required buffer size
    *size = sizeof(cuda_iq_data_t);
    for (int i = 0; i < cuda_iq->seqnum; i++) {
        *size += cuda_iq->size[i] * sizeof(int16_t) * 2;  // I+Q samples
    }
    
    char *buffer = (char*)malloc(*size);
    if (!buffer) {
        cuda_iq_data_destroy(cuda_iq);
        return NULL;
    }
    
    // Allocate offsets array
    int *offsets;
    cudaMallocManaged(&offsets, cuda_iq->seqnum * sizeof(int));
    
    // Calculate offsets
    int current_offset = sizeof(cuda_iq_data_t);
    for (int i = 0; i < cuda_iq->seqnum; i++) {
        offsets[i] = current_offset;
        current_offset += cuda_iq->size[i] * sizeof(int16_t) * 2;
    }
    
    // Use CUDA kernel for flattening
    cudaError_t err = cuda_iq_flatten(cuda_iq, buffer, offsets, *size);
    
    cudaFree(offsets);
    cuda_iq_data_destroy(cuda_iq);
    
    if (err != cudaSuccess) {
        free(buffer);
        return IQFlatten(ptr, num, size);  // Fall back
    }
    
    return buffer;
}

int IQExpand_CUDA(struct IQ **ptr, int num, void *buffer) {
    if (!ptr || !buffer) return -1;
    
    if (!cuda_acceleration_enabled || !iq_1_7_cuda_is_available()) {
        return IQExpand(ptr, num, buffer);
    }
    
    // Create CUDA IQ structure
    cuda_iq_data_t *cuda_iq = cuda_iq_data_create(num);
    if (!cuda_iq) return IQExpand(ptr, num, buffer);
    
    // Allocate arrays
    if (cuda_iq_data_allocate_arrays(cuda_iq) != cudaSuccess) {
        cuda_iq_data_destroy(cuda_iq);
        return IQExpand(ptr, num, buffer);
    }
    
    // Calculate offsets (simplified)
    int *offsets;
    cudaMallocManaged(&offsets, num * sizeof(int));
    
    int current_offset = sizeof(cuda_iq_data_t);
    for (int i = 0; i < num; i++) {
        offsets[i] = current_offset;
        current_offset += 1000 * sizeof(int16_t) * 2;  // Estimated size
    }
    
    // Use CUDA kernel for expansion
    cudaError_t err = cuda_iq_expand((const char*)buffer, cuda_iq, offsets, num);
    
    cudaFree(offsets);
    
    if (err != cudaSuccess) {
        cuda_iq_data_destroy(cuda_iq);
        return IQExpand(ptr, num, buffer);
    }
    
    // Convert back to original format
    *ptr = IQMake();
    if (!*ptr) {
        cuda_iq_data_destroy(cuda_iq);
        return -1;
    }
    
    int result = cuda_iq_to_iq(cuda_iq, *ptr);
    cuda_iq_data_destroy(cuda_iq);
    
    return result;
}

/* Control functions */

void iq_1_7_cuda_enable_acceleration(bool enable) {
    cuda_acceleration_enabled = enable;
}

bool iq_1_7_cuda_acceleration_enabled(void) {
    return cuda_acceleration_enabled && iq_1_7_cuda_is_available();
}

void iq_1_7_cuda_set_debug_mode(bool enable) {
    cuda_debug_mode = enable;
}