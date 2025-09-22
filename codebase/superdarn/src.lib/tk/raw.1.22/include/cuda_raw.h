/**
 * @file cuda_raw.h
 * @brief CUDA interface for raw.1.22 module
 * 
 * Provides CUDA acceleration for SuperDARN raw data processing
 * operations including data reorganization, filtering, and search.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#ifndef CUDA_RAW_H
#define CUDA_RAW_H

#include <cuda_runtime.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA-compatible raw data structures */
typedef struct {
    float *pwr0;         // Power array (nrang elements)
    float *acfd_real;    // ACF real data
    float *acfd_imag;    // ACF imaginary data  
    float *xcfd_real;    // XCF real data
    float *xcfd_imag;    // XCF imaginary data
    int nrang;           // Number of range gates
    int mplgs;           // Number of lags
    float threshold;     // Power threshold
} cuda_raw_data_t;

typedef struct {
    double *tme;         // Time array
    int *inx;           // Index array
    int num_records;    // Number of records
} cuda_raw_index_t;

typedef struct {
    float min_val, max_val;  // Min/max values
    float mean_val;          // Mean value
    float sum_val;           // Sum of values
    int count;              // Number of elements
} cuda_raw_statistics_t;

/* Core CUDA functions for raw data processing */

/**
 * CUDA-accelerated complex data interleaving
 * Combines real and imaginary components into interleaved format
 */
cudaError_t cuda_raw_interleave_complex(const float *real_data,
                                        const float *imag_data,
                                        float *output,
                                        int nrang, int mplgs);

/**
 * CUDA-accelerated complex data deinterleaving  
 * Separates interleaved complex data back to real/imaginary arrays
 */
cudaError_t cuda_raw_deinterleave_complex(const float *interleaved_data,
                                          float *real_data,
                                          float *imag_data,
                                          int nrang, int mplgs);

/**
 * CUDA-accelerated threshold filtering with sample list generation
 * Generates compact sample list of ranges above power threshold
 */
cudaError_t cuda_raw_threshold_filter(const float *pwr0,
                                      int *slist,
                                      int *snum,
                                      float threshold,
                                      int nrang);

/**
 * CUDA-accelerated sparse data gathering
 * Gathers data from sparse sample list for efficient storage
 */
cudaError_t cuda_raw_sparse_gather(const float *input,
                                   const int *slist,
                                   float *output,
                                   int snum, int mplgs);

/**
 * CUDA-accelerated data reorganization for encoding
 * Reorganizes raw data for efficient storage/transmission
 */
cudaError_t cuda_raw_data_reorganize(const float *acfd_real,
                                     const float *acfd_imag,
                                     const float *xcfd_real,
                                     const float *xcfd_imag,
                                     const int *slist,
                                     float *encoded_acfd,
                                     float *encoded_xcfd,
                                     int snum, int mplgs);

/**
 * CUDA-accelerated power-based range filtering
 * Applies power threshold filtering and updates statistics
 */
cudaError_t cuda_raw_power_filter(const float *pwr0,
                                  float *filtered_pwr,
                                  int *valid_indices,
                                  float threshold,
                                  int nrang);

/**
 * CUDA-accelerated time-based binary search
 * Performs parallel binary search in sorted time arrays
 */
cudaError_t cuda_raw_time_search(const double *time_array,
                                 const double *search_times,
                                 int *result_indices,
                                 int num_records,
                                 int num_searches);

/**
 * CUDA-accelerated statistics calculation
 * Computes min, max, mean, sum for raw data arrays
 */
cudaError_t cuda_raw_calculate_statistics(const float *data,
                                          cuda_raw_statistics_t *stats,
                                          int num_elements);

/* Memory management functions */

/**
 * Allocate CUDA-compatible raw data structure
 */
cuda_raw_data_t* cuda_raw_data_alloc(int nrang, int mplgs);

/**
 * Free CUDA-compatible raw data structure
 */
void cuda_raw_data_free(cuda_raw_data_t *raw_data);

/**
 * Allocate CUDA-compatible raw index structure
 */
cuda_raw_index_t* cuda_raw_index_alloc(int num_records);

/**
 * Free CUDA-compatible raw index structure
 */
void cuda_raw_index_free(cuda_raw_index_t *raw_index);

/**
 * Copy raw data from host to device
 */
cudaError_t cuda_raw_data_copy_to_device(cuda_raw_data_t *raw_data);

/**
 * Copy raw data from device to host
 */
cudaError_t cuda_raw_data_copy_to_host(cuda_raw_data_t *raw_data);

/* Utility functions */

/**
 * Check if CUDA is available for raw processing
 */
bool cuda_raw_is_available(void);

/**
 * Get number of CUDA devices available
 */
int cuda_raw_get_device_count(void);

/**
 * Initialize CUDA context for raw processing
 */
cudaError_t cuda_raw_init(void);

/**
 * Cleanup CUDA context for raw processing
 */
void cuda_raw_cleanup(void);

/* Performance profiling */
typedef struct {
    float cpu_time_ms;      // CPU processing time
    float gpu_time_ms;      // GPU processing time  
    float transfer_time_ms; // Memory transfer time
    float speedup_factor;   // GPU/CPU speedup ratio
    size_t memory_used;     // Peak GPU memory usage
} cuda_raw_profile_t;

/**
 * Enable/disable performance profiling
 */
cudaError_t cuda_raw_enable_profiling(bool enable);

/**
 * Get performance profile from last operation
 */
cudaError_t cuda_raw_get_profile(cuda_raw_profile_t *profile);

/**
 * Reset performance counters
 */
void cuda_raw_reset_profile(void);

/* High-level wrapper functions for backward compatibility */

/**
 * CUDA-accelerated version of RawEncode
 * Encodes raw data with GPU acceleration
 */
cudaError_t cuda_raw_encode(cuda_raw_data_t *raw_data,
                           const int *slist, int snum,
                           float *encoded_data);

/**
 * CUDA-accelerated version of RawDecode  
 * Decodes raw data with GPU acceleration
 */
cudaError_t cuda_raw_decode(const float *encoded_data,
                           cuda_raw_data_t *raw_data,
                           const int *slist, int snum);

/**
 * CUDA-accelerated version of RawSeek
 * Performs time-based seeking with GPU acceleration
 */
cudaError_t cuda_raw_seek(cuda_raw_index_t *index,
                         double target_time,
                         int *result_position);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_RAW_H */