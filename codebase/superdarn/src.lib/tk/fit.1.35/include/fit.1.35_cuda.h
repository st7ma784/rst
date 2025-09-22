#ifndef FIT.1.35_CUDA_H
#define FIT.1.35_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cufft.h>
#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Native CUDA Data Structures */

// CUDA-native array structure with unified memory
typedef struct {
    void *data;
    size_t size;
    size_t element_size;
    cudaDataType_t type;
    int device_id;
    bool is_on_device;
} cuda_array_t;

// CUDA-native matrix structure
typedef struct {
    void *data;
    int rows;
    int cols;
    int ld;
    cudaDataType_t type;
    int device_id;
} cuda_matrix_t;

// CUDA-native complex arrays
typedef struct {
    cuFloatComplex *data;
    size_t size;
    int device_id;
} cuda_complex_array_t;

// Range processing structure (SuperDARN specific)
typedef struct {
    int *ranges;
    float *powers;
    cuFloatComplex *phases;
    float *velocities;
    int num_ranges;
    int device_id;
} cuda_range_data_t;

/* Memory Management Functions */
cuda_array_t* cuda_array_create(size_t size, size_t element_size, cudaDataType_t type);
void cuda_array_destroy(cuda_array_t *array);
cudaError_t cuda_array_copy_to_device(cuda_array_t *array);
cudaError_t cuda_array_copy_to_host(cuda_array_t *array);

cuda_matrix_t* cuda_matrix_create(int rows, int cols, cudaDataType_t type);
void cuda_matrix_destroy(cuda_matrix_t *matrix);

cuda_complex_array_t* cuda_complex_array_create(size_t size);
void cuda_complex_array_destroy(cuda_complex_array_t *array);

cuda_range_data_t* cuda_range_data_create(int num_ranges);
void cuda_range_data_destroy(cuda_range_data_t *data);

/* CUDA-compatible FIT data structures */
typedef struct {
    float v;        // Line-of-sight velocity (m/s)
    float v_e;      // Velocity error (m/s)
    float p_l;      // Lambda power (dB)
    float p_l_e;    // Lambda power error (dB)
    float p_s;      // Sigma power (dB)
    float p_s_e;    // Sigma power error (dB)
    float w_l;      // Lambda spectral width (m/s)
    float w_l_e;    // Lambda spectral width error (m/s)
    float w_s;      // Sigma spectral width (m/s)
    float w_s_e;    // Sigma spectral width error (m/s)
    float sd_l;     // Lambda standard deviation
    float sd_s;     // Sigma standard deviation
    float sd_phi;   // Phase standard deviation
    float phi0;     // Phase of lag zero (degrees)
    float phi0_e;   // Phase error (degrees)
    float elv;      // Elevation angle (degrees)
    float elv_low;  // Lower elevation angle (degrees)
    float elv_high; // Upper elevation angle (degrees)
    int qflg;       // Quality flag
    int gsct;       // Ground scatter flag
    bool valid;     // Data validity flag
} cuda_fit_range_t;

typedef struct {
    int stid;       // Station ID
    int bmnum;      // Beam number
    int scan;       // Scan flag
    int cp;         // Control program ID
    int channel;    // Channel number
    double time;    // Time
    int nrang;      // Number of range gates
    int *slist;     // Sample list
    cuda_fit_range_t *ranges;  // Range data
} cuda_fit_data_t;

typedef struct {
    float v;        // Velocity (m/s)
    float p_l;      // Lambda power (dB)
    float w_l;      // Lambda spectral width (m/s)
    float phi0;     // Phase (degrees)
    float elv;      // Elevation angle (degrees)
    int qflg;       // Quality flag
    int gsct;       // Ground scatter flag
} cuda_cfit_cell_t;

/* Core fit.1.35 CUDA Functions */

/**
 * CUDA-accelerated range validation
 * Validates range gates based on power thresholds and quality criteria
 */
cudaError_t cuda_fit_validate_ranges(const cuda_fit_range_t *ranges,
                                     int *valid_indices,
                                     int *valid_count,
                                     int total_ranges,
                                     float min_power_threshold);

/**
 * CUDA-accelerated FIT to CFIT conversion
 * Converts full FIT data to compact CFIT format
 */
cudaError_t cuda_fit_to_cfit(const cuda_fit_range_t *fit_ranges,
                             cuda_cfit_cell_t *cfit_cells,
                             const int *valid_indices,
                             int num_valid_ranges);

/**
 * CUDA-accelerated range data processing
 * Applies quality control and data conditioning
 */
cudaError_t cuda_fit_process_ranges(cuda_fit_range_t *ranges,
                                    int num_ranges,
                                    float noise_level,
                                    float velocity_limit);

/**
 * CUDA-accelerated statistics calculation
 * Computes statistical parameters for quality assessment
 */
cudaError_t cuda_fit_calculate_statistics(const cuda_fit_range_t *ranges,
                                          int num_ranges,
                                          float *velocity_mean,
                                          float *power_mean,
                                          float *width_mean,
                                          int *valid_count);

/**
 * CUDA-accelerated elevation angle calculation
 * Computes elevation angles based on range and phase measurements
 */
cudaError_t cuda_fit_calculate_elevation(cuda_fit_range_t *ranges,
                                         int num_ranges,
                                         float antenna_separation,
                                         float operating_frequency);

/**
 * High-level FIT processing pipeline
 * Complete processing pipeline with validation, conditioning, and conversion
 */
cudaError_t cuda_fit_process_pipeline(cuda_fit_data_t *fit_data,
                                      cuda_cfit_cell_t *cfit_cells,
                                      int *num_valid_ranges,
                                      float min_power_threshold,
                                      float noise_level,
                                      float velocity_limit,
                                      float antenna_separation,
                                      float operating_frequency);

/* Utility Functions */
bool fit.1.35_cuda_is_available(void);
int fit.1.35_cuda_get_device_count(void);
const char* fit.1.35_cuda_get_error_string(cudaError_t error);

/* Performance Profiling */
typedef struct {
    float cpu_time_ms;
    float gpu_time_ms;
    float speedup_factor;
    size_t memory_used_bytes;
} fit.1.35_cuda_profile_t;

cudaError_t fit.1.35_cuda_enable_profiling(bool enable);
cudaError_t fit.1.35_cuda_get_profile(fit.1.35_cuda_profile_t *profile);

#ifdef __cplusplus
}
#endif

#endif /* FIT.1.35_CUDA_H */
