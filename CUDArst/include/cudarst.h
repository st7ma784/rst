/**
 * @file cudarst.h
 * @brief CUDArst - CUDA-accelerated RST SuperDARN Library
 * 
 * Unified interface providing backward compatibility with original RST toolkit
 * while leveraging CUDA acceleration for improved performance.
 * 
 * @author CUDA Conversion Project
 * @version 1.0.0
 * @date 2025
 */

#ifndef CUDARST_H
#define CUDARST_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Library version information */
#define CUDARST_VERSION_MAJOR 2
#define CUDARST_VERSION_MINOR 0
#define CUDARST_VERSION_PATCH 0
#define CUDARST_VERSION_STRING "2.0.0"

/* Runtime configuration */
typedef enum {
    CUDARST_MODE_AUTO,      /* Automatic CPU/CUDA selection */
    CUDARST_MODE_CPU_ONLY,  /* Force CPU processing */
    CUDARST_MODE_CUDA_ONLY  /* Force CUDA processing */
} cudarst_mode_t;

typedef enum {
    CUDARST_SUCCESS = 0,
    CUDARST_ERROR_INVALID_ARGS = -1,
    CUDARST_ERROR_CUDA_UNAVAILABLE = -2,
    CUDARST_ERROR_MEMORY_ALLOCATION = -3,
    CUDARST_ERROR_PROCESSING_FAILED = -4
} cudarst_error_t;

/**
 * @brief Initialize CUDArst library
 * @param mode Processing mode selection
 * @return Error code (CUDARST_SUCCESS on success)
 */
cudarst_error_t cudarst_init(cudarst_mode_t mode);

/**
 * @brief Cleanup CUDArst library resources
 */
void cudarst_cleanup(void);

/**
 * @brief Get library version string
 * @return Version string
 */
const char* cudarst_get_version(void);

/**
 * @brief Check if CUDA is available
 * @return true if CUDA is available and functional
 */
bool cudarst_is_cuda_available(void);

/* ====================================================================
 * FITACF v3.0 Compatible Interface
 * ====================================================================*/

/* Original FITACF data structures (preserved for compatibility) */
typedef struct {
    int16_t bmnum;     /* beam number */
    int16_t scan;      /* scan flag */
    int16_t offset;    /* offset from start of second */
    int16_t nave;      /* number of pulse sequences transmitted */
    int16_t nrang;     /* number of range gates */
    int16_t frang;     /* distance to first range gate (km) */
    int16_t rsep;      /* range separation (km) */
    int16_t xcf;       /* XCF flag */
    int16_t tfreq;     /* transmitted frequency */
    int16_t noise;     /* noise level */
    int16_t atten;     /* attenuation level */
    int16_t channel;   /* channel number for stereo radars */
    int16_t cpid;      /* control program identifier */
    int16_t maxpwr;    /* maximum power */
    int16_t maxnoise;  /* maximum noise */
    int16_t maxatten;  /* maximum attenuation */
    /* Additional fields for compatibility */
    int32_t time_sec;  /* time in seconds */
    int32_t time_usec; /* time in microseconds */
} cudarst_fitacf_prm_t;

typedef struct {
    float *acfd;       /* auto-correlation function data (real) */
    float *xcfd;       /* cross-correlation function data (real) */
    float *acfd_imag;  /* auto-correlation function data (imaginary) */
    float *xcfd_imag;  /* cross-correlation function data (imaginary) */
    int nrang;         /* number of range gates */
    int mplgs;         /* number of lags */
} cudarst_fitacf_raw_t;

typedef struct {
    float *pwr0;       /* lag zero power */
    float *slist;      /* list of range gates with scatter */
    float *v;          /* line-of-sight velocity */
    float *v_e;        /* velocity error */
    float *p_l;        /* lambda power */
    float *p_l_e;      /* lambda power error */
    float *p_s;        /* sigma power */
    float *p_s_e;      /* sigma power error */
    float *w_l;        /* lambda spectral width */
    float *w_l_e;      /* lambda spectral width error */
    float *w_s;        /* sigma spectral width */
    float *w_s_e;      /* sigma spectral width error */
    float *phi0;       /* phase of lag zero */
    float *phi0_e;     /* phase error */
    float *elv;        /* elevation angle */
    float *elv_low;    /* lowest elevation angle */
    float *elv_high;   /* highest elevation angle */
    float *x_qflg;     /* XCF quality flag */
    float *x_gflg;     /* XCF ground scatter flag */
    float *x_p_l;      /* XCF lambda power */
    float *x_p_l_e;    /* XCF lambda power error */
    float *x_p_s;      /* XCF sigma power */
    float *x_p_s_e;    /* XCF sigma power error */
    float *x_v;        /* XCF velocity */
    float *x_v_e;      /* XCF velocity error */
    float *x_w_l;      /* XCF lambda spectral width */
    float *x_w_l_e;    /* XCF lambda spectral width error */
    float *x_w_s;      /* XCF sigma spectral width */
    float *x_w_s_e;    /* XCF sigma spectral width error */
    float *x_phi0;     /* XCF phase */
    float *x_phi0_e;   /* XCF phase error */
    int nrang;         /* number of range gates */
} cudarst_fitacf_fit_t;

/**
 * @brief Process FITACF data (compatible with original interface)
 * @param prm Parameter structure
 * @param raw Raw data structure
 * @param fit Output fitted data structure
 * @return Error code
 */
cudarst_error_t cudarst_fitacf_process(const cudarst_fitacf_prm_t *prm,
                                       const cudarst_fitacf_raw_t *raw,
                                       cudarst_fitacf_fit_t *fit);

/**
 * @brief Allocate FITACF structures
 */
cudarst_fitacf_raw_t* cudarst_fitacf_raw_alloc(int nrang, int mplgs);
cudarst_fitacf_fit_t* cudarst_fitacf_fit_alloc(int nrang);

/**
 * @brief Free FITACF structures
 */
void cudarst_fitacf_raw_free(cudarst_fitacf_raw_t *raw);
void cudarst_fitacf_fit_free(cudarst_fitacf_fit_t *fit);

/* ====================================================================
 * LMFIT v2.0 Compatible Interface
 * ====================================================================*/

/* LMFIT data structures (preserved for compatibility) */
typedef struct {
    float *y;          /* data vector */
    float *x;          /* independent variable vector */
    float *sig;        /* standard deviation vector */
    int ndata;         /* number of data points */
    int ma;            /* number of parameters */
    float *a;          /* parameter vector */
    float **covar;     /* covariance matrix */
    float *alpha;      /* alpha matrix (linearized fitting matrix) */
    float chisq;       /* chi-squared value */
    float alamda;      /* lambda parameter */
} cudarst_lmfit_data_t;

typedef struct {
    float (*func)(float x, float *a, int na);                    /* fitting function */
    void (*funcs)(float x, float *a, float *dyda, int na);      /* derivatives */
    int max_iterations;     /* maximum number of iterations */
    float tolerance;        /* convergence tolerance */
    bool use_cuda;         /* enable CUDA acceleration */
} cudarst_lmfit_config_t;

/**
 * @brief Levenberg-Marquardt fitting (compatible with original interface)
 * @param data Data structure containing observations and parameters
 * @param config Configuration structure
 * @return Error code
 */
cudarst_error_t cudarst_lmfit_solve(cudarst_lmfit_data_t *data,
                                    const cudarst_lmfit_config_t *config);

/**
 * @brief Allocate LMFIT data structure
 */
cudarst_lmfit_data_t* cudarst_lmfit_data_alloc(int ndata, int ma);

/**
 * @brief Free LMFIT data structure
 */
void cudarst_lmfit_data_free(cudarst_lmfit_data_t *data);

/* ====================================================================
 * Performance and Diagnostics
 * ====================================================================*/

typedef struct {
    double total_time_ms;      /* Total processing time */
    double cuda_time_ms;       /* CUDA kernel time */
    double memory_transfer_ms; /* Memory transfer time */
    double cpu_fallback_ms;    /* CPU fallback time */
    size_t memory_used_bytes;  /* Peak memory usage */
    int cuda_device_id;        /* CUDA device used */
    bool cuda_used;            /* Whether CUDA was actually used */
} cudarst_performance_t;

/**
 * @brief Get performance statistics for last operation
 * @param perf Performance structure to fill
 * @return Error code
 */
cudarst_error_t cudarst_get_performance(cudarst_performance_t *perf);

/**
 * @brief Reset performance counters
 */
void cudarst_reset_performance(void);

/**
 * @brief Print performance summary to stdout
 */
void cudarst_print_performance(void);

/* ====================================================================
 * ACF v1.16 Compatible Interface
 * ====================================================================*/

/**
 * @brief Process ACF data using CUDA acceleration
 * @param inbuf Raw input buffer (I/Q samples)
 * @param acfbuf Output ACF buffer
 * @param xcfbuf Output XCF buffer (optional)
 * @param lagfr Lag frame table
 * @param smsep Sample separation
 * @param pat Pulse pattern
 * @param nrang Number of range gates
 * @param mplgs Number of lags
 * @param mpinc Number of pulses
 * @param nave Number of averages
 * @param offset Time offset
 * @param xcf_enabled Enable cross-correlation function processing
 * @return Error code
 */
cudarst_error_t cudarst_acf_process(const int16_t *inbuf, float *acfbuf, float *xcfbuf,
                                   const int *lagfr, const int *smsep, const int *pat,
                                   int nrang, int mplgs, int mpinc, int nave,
                                   int offset, bool xcf_enabled);

/* ====================================================================
 * IQ v1.7 Compatible Interface
 * ====================================================================*/

/**
 * @brief Process IQ time series data using CUDA acceleration
 * @param input_time Input time stamps
 * @param iq_data Raw I/Q data
 * @param tv_sec Output time seconds
 * @param tv_nsec Output time nanoseconds
 * @param encoded_iq Output encoded I/Q data
 * @param num_samples Number of samples
 * @param scale_factor Encoding scale factor
 * @return Error code
 */
cudarst_error_t cudarst_iq_process_time_series(const double *input_time, const float *iq_data,
                                              long *tv_sec, long *tv_nsec, int16_t *encoded_iq,
                                              int num_samples, float scale_factor);

/* ====================================================================
 * CNVMAP v1.17 Compatible Interface
 * ====================================================================*/

/**
 * @brief Spherical harmonic fitting for convection mapping
 * @param theta Colatitude coordinates
 * @param phi Longitude coordinates  
 * @param v_los Line-of-sight velocities
 * @param n_points Number of data points
 * @param coefficients Output spherical harmonic coefficients
 * @param lmax Maximum spherical harmonic degree
 * @return Error code
 */
cudarst_error_t cudarst_cnvmap_spherical_harmonic_fit(const double *theta, const double *phi,
                                                     const double *v_los, int n_points,
                                                     double *coefficients, int lmax);

/* ====================================================================
 * GRID v1.24 Compatible Interface
 * ====================================================================*/

/**
 * @brief Grid data interpolation using CUDA acceleration
 * @param x_data Input X coordinates
 * @param y_data Input Y coordinates
 * @param values Input data values
 * @param n_points Number of input points
 * @param grid_x Grid X coordinates
 * @param grid_y Grid Y coordinates
 * @param grid_values Output gridded values
 * @param grid_nx Grid X dimension
 * @param grid_ny Grid Y dimension
 * @param cell_size Grid cell size
 * @return Error code
 */
cudarst_error_t cudarst_grid_interpolate_data(const float *x_data, const float *y_data,
                                             const float *values, int n_points,
                                             const float *grid_x, const float *grid_y,
                                             float *grid_values, int grid_nx, int grid_ny,
                                             float cell_size);

/* ====================================================================
 * Memory Management
 * ====================================================================*/

/**
 * @brief Allocate unified memory (CPU/CUDA accessible)
 * @param size Size in bytes
 * @return Pointer to allocated memory or NULL on failure
 */
void* cudarst_malloc(size_t size);

/**
 * @brief Free unified memory
 * @param ptr Pointer to memory allocated with cudarst_malloc
 */
void cudarst_free(void *ptr);

/**
 * @brief Copy data between host and device
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Size in bytes
 * @param to_device true for host->device, false for device->host
 * @return Error code
 */
cudarst_error_t cudarst_memcpy(void *dst, const void *src, size_t size, bool to_device);

/* ====================================================================
 * Backward Compatibility Macros
 * ====================================================================*/

/* Map original RST function names to CUDArst equivalents */
#define FitACFStart()           cudarst_init(CUDARST_MODE_AUTO)
#define FitACFEnd()             cudarst_cleanup()
#define FitACF(prm, raw, fit)   cudarst_fitacf_process(prm, raw, fit)

#define LMFitStart()            cudarst_init(CUDARST_MODE_AUTO)
#define LMFitEnd()              cudarst_cleanup()
#define LMFit(data, config)     cudarst_lmfit_solve(data, config)

/* ACF v1.16 compatibility macros */
#define ACFStart()              cudarst_init(CUDARST_MODE_AUTO)
#define ACFEnd()                cudarst_cleanup()
#define ACFProcess(...)         cudarst_acf_process(__VA_ARGS__)

/* IQ v1.7 compatibility macros */
#define IQStart()               cudarst_init(CUDARST_MODE_AUTO)
#define IQEnd()                 cudarst_cleanup()
#define IQProcessTimeSeries(...) cudarst_iq_process_time_series(__VA_ARGS__)

/* CNVMAP v1.17 compatibility macros */
#define CNVMAPStart()           cudarst_init(CUDARST_MODE_AUTO)
#define CNVMAPEnd()             cudarst_cleanup()
#define CNVMAPFit(...)          cudarst_cnvmap_spherical_harmonic_fit(__VA_ARGS__)

/* GRID v1.24 compatibility macros */
#define GRIDStart()             cudarst_init(CUDARST_MODE_AUTO)
#define GRIDEnd()               cudarst_cleanup()
#define GRIDInterpolate(...)    cudarst_grid_interpolate_data(__VA_ARGS__)

/* Original structure type aliases */
typedef cudarst_fitacf_prm_t FitPrm;
typedef cudarst_fitacf_raw_t FitRaw;
typedef cudarst_fitacf_fit_t FitData;
typedef cudarst_lmfit_data_t LMFitData;
typedef cudarst_lmfit_config_t LMFitConfig;

#ifdef __cplusplus
}
#endif

#endif /* CUDARST_H */