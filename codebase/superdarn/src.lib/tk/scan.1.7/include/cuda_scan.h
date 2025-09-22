/**
 * @file cuda_scan.h
 * @brief CUDA interface for scan.1.7 module
 * 
 * Provides CUDA acceleration for SuperDARN scan data processing
 * operations including beam management, range gate processing, and scan validation.
 * 
 * @author CUDA Conversion Project
 * @date 2025
 */

#ifndef CUDA_SCAN_H
#define CUDA_SCAN_H

#include <cuda_runtime.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA-compatible scan data structures */

typedef struct {
    int gsct;           // Ground scatter flag
    float p_0;          // Lag-0 power
    float p_0_e;        // Lag-0 power error
    float v;            // Velocity
    float v_e;          // Velocity error  
    float w_l;          // Lambda spectral width
    float w_l_e;        // Lambda spectral width error
    float p_l;          // Lambda power
    float p_l_e;        // Lambda power error
    float phi0;         // Phase
    float elv;          // Elevation angle
} cuda_radar_cell_t;

typedef struct {
    int scan, bm;               // Scan and beam numbers
    float bmazm;                // Beam azimuth
    double time;                // Timestamp
    int cpid, nave, frang, rsep, rxrise, freq, noise, atten, channel, nrang;
    int intt_sc, intt_us;       // Integration time
    unsigned char *sct;         // Scatter flags array
    cuda_radar_cell_t *rng;     // Range cell data array
    bool valid;                 // Validity flag for filtering
} cuda_radar_beam_t;

typedef struct {
    int stid;                   // Station ID
    int version_major, version_minor;  // Version info
    double st_time, ed_time;    // Start and end times
    int num;                    // Number of beams
    cuda_radar_beam_t *bm;      // Beam array
    int capacity;               // Allocated capacity
} cuda_radar_scan_t;

typedef struct {
    float power_mean;           // Mean power
    float velocity_mean;        // Mean velocity
    float width_mean;           // Mean spectral width
    float noise_level;          // Estimated noise level
    int valid_ranges;           // Number of valid range gates
    int ionospheric_count;      // Number of ionospheric scatter gates
    int ground_scatter_count;   // Number of ground scatter gates
} cuda_scan_statistics_t;

/* Core CUDA functions for scan processing */

/**
 * CUDA-accelerated beam processing
 * Processes multiple radar beams in parallel with noise filtering
 */
cudaError_t cuda_scan_process_beams(cuda_radar_beam_t *beams,
                                    int num_beams,
                                    int max_range_gates);

/**
 * CUDA-accelerated beam filtering  
 * Filters beams based on beam number (parallel RadarScanResetBeam)
 */
cudaError_t cuda_scan_filter_beams(cuda_radar_beam_t *beams,
                                   int num_beams,
                                   int target_beam,
                                   bool *valid_mask,
                                   int *new_count);

/**
 * CUDA-accelerated scan validation
 * Validates scan integrity and removes invalid beams (parallel exclude_outofscan)
 */
cudaError_t cuda_scan_validate(cuda_radar_beam_t *beams,
                               int num_beams,
                               bool *valid_mask);

/**
 * CUDA-accelerated range gate statistics
 * Computes statistics across range gates for quality assessment
 */
cudaError_t cuda_scan_range_statistics(cuda_radar_beam_t *beams,
                                       int num_beams,
                                       float *power_mean,
                                       float *velocity_mean,
                                       float *width_mean,
                                       int *valid_count);

/**
 * CUDA-accelerated scatter classification
 * Advanced ground scatter vs ionospheric scatter classification
 */
cudaError_t cuda_scan_scatter_classification(cuda_radar_beam_t *beams,
                                            int num_beams,
                                            int max_range_gates);

/**
 * CUDA-accelerated beam compaction
 * Removes invalid beams from scan array
 */
cudaError_t cuda_scan_compact_beams(cuda_radar_beam_t *input_beams,
                                    cuda_radar_beam_t *output_beams,
                                    bool *valid_mask,
                                    int num_beams,
                                    int *new_count);

/**
 * CUDA-accelerated beam sorting
 * Sorts beams by time and beam number for chronological order
 */
cudaError_t cuda_scan_sort_beams(cuda_radar_beam_t *beams,
                                 int num_beams);

/**
 * CUDA-accelerated noise level estimation
 * Estimates noise levels across all beams for quality control
 */
cudaError_t cuda_scan_estimate_noise(cuda_radar_beam_t *beams,
                                     int num_beams,
                                     float *noise_estimates);

/* Memory management functions */

/**
 * Allocate CUDA-compatible radar scan structure
 */
cuda_radar_scan_t* cuda_scan_alloc(int max_beams, int max_ranges_per_beam);

/**
 * Free CUDA-compatible radar scan structure
 */
void cuda_scan_free(cuda_radar_scan_t *scan);

/**
 * Allocate CUDA-compatible radar beam structure
 */
cuda_radar_beam_t* cuda_beam_alloc(int num_beams, int max_ranges_per_beam);

/**
 * Free CUDA-compatible radar beam structure
 */
void cuda_beam_free(cuda_radar_beam_t *beams, int num_beams);

/**
 * Copy scan data from host to device
 */
cudaError_t cuda_scan_copy_to_device(cuda_radar_scan_t *scan);

/**
 * Copy scan data from device to host
 */
cudaError_t cuda_scan_copy_to_host(cuda_radar_scan_t *scan);

/* Utility functions */

/**
 * Check if CUDA is available for scan processing
 */
bool cuda_scan_is_available(void);

/**
 * Get number of CUDA devices available
 */
int cuda_scan_get_device_count(void);

/**
 * Initialize CUDA context for scan processing
 */
cudaError_t cuda_scan_init(void);

/**
 * Cleanup CUDA context for scan processing
 */
void cuda_scan_cleanup(void);

/* Performance profiling */
typedef struct {
    float cpu_time_ms;          // CPU processing time
    float gpu_time_ms;          // GPU processing time  
    float transfer_time_ms;     // Memory transfer time
    float speedup_factor;       // GPU/CPU speedup ratio
    size_t memory_used;         // Peak GPU memory usage
    int beams_processed;        // Number of beams processed
    int ranges_processed;       // Total number of range gates processed
} cuda_scan_profile_t;

/**
 * Enable/disable performance profiling
 */
cudaError_t cuda_scan_enable_profiling(bool enable);

/**
 * Get performance profile from last operation
 */
cudaError_t cuda_scan_get_profile(cuda_scan_profile_t *profile);

/**
 * Reset performance counters
 */
void cuda_scan_reset_profile(void);

/* High-level wrapper functions for backward compatibility */

/**
 * CUDA-accelerated version of RadarScanReset
 * Resets scan data with GPU acceleration
 */
cudaError_t cuda_radar_scan_reset(cuda_radar_scan_t *scan);

/**
 * CUDA-accelerated version of RadarScanResetBeam
 * Filters beams by beam number with GPU acceleration
 */
cudaError_t cuda_radar_scan_reset_beam(cuda_radar_scan_t *scan, int beam_number);

/**
 * CUDA-accelerated version of RadarScanAddBeam
 * Adds beam to scan with GPU-optimized memory management
 */
cudaError_t cuda_radar_scan_add_beam(cuda_radar_scan_t *scan, 
                                     cuda_radar_beam_t *new_beam);

/**
 * CUDA-accelerated scan processing pipeline
 * Complete processing pipeline with filtering, validation, and statistics
 */
cudaError_t cuda_scan_process_pipeline(cuda_radar_scan_t *scan,
                                       cuda_scan_statistics_t *stats,
                                       int filter_beam,
                                       bool validate_beams,
                                       bool compute_statistics);

/* Data conversion functions */

/**
 * Convert original RadarScan to CUDA-compatible format
 */
cudaError_t cuda_scan_from_host(const void *host_scan, cuda_radar_scan_t *cuda_scan);

/**
 * Convert CUDA-compatible format back to original RadarScan
 */
cudaError_t cuda_scan_to_host(const cuda_radar_scan_t *cuda_scan, void *host_scan);

/**
 * Convert original RadarBeam to CUDA-compatible format
 */
cudaError_t cuda_beam_from_host(const void *host_beam, cuda_radar_beam_t *cuda_beam);

/**
 * Convert CUDA-compatible format back to original RadarBeam
 */
cudaError_t cuda_beam_to_host(const cuda_radar_beam_t *cuda_beam, void *host_beam);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SCAN_H */