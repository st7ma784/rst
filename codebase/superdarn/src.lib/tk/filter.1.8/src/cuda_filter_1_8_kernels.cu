/*
 * cuda_filter_1_8_kernels.cu
 * ==========================
 * CUDA implementation of the RST filter.1.8 3-D median filter.
 *
 * Algorithm mirrors FilterRadarScan() in filter.c:
 *   - 3 beams × 3 range gates × depth scans neighbourhood
 *   - Weighted threshold check (mode 0: thresh=12, mode 1: thresh=24)
 *   - Per-parameter (v, p_l, w_l, p_0) mean/sigma → within-2σ median
 *   - Each output (beam, range) cell processed by one GPU thread
 */

#include "../include/filter.1.8_cuda.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return _e; \
        } \
    } while (0)

/* Weight table: centre cell of the 3×3 beam-range kernel gets weight 4,
 * edge cells get 1, corner cells get 1. Temporal middle scan doubles all. */
__device__ static int cell_weight(int db, int dr, int dt, int depth)
{
    int w = 1;
    if (db == 0 && dr == 0) w = 4;
    else if (db == 0 || dr == 0) w = 2;
    /* middle time scan carries double weight */
    if (depth == 3 && dt == 1) w *= 2;
    return w;
}

/* -----------------------------------------------------------------------
 * Kernel 1: gather neighbourhood cells and accumulate weight sum
 * -----------------------------------------------------------------------
 * One thread per output (beam, range) pair.
 * Walks the 3×3×depth cube and stores valid scatter cells into the
 * preallocated neighbourhood buffer d_nbr[out * MAXNBR].
 */
__global__ void cuda_filter_gather_kernel(
    const cuda_filter_cell_t *d_cells,  /* [depth * n_beams * max_nrang] */
    const unsigned char      *d_sct,    /* [depth * n_beams * max_nrang] */
    int depth, int n_beams, int max_nrang, int inx,
    cuda_filter_cell_t *d_nbr,          /* [n_beams * max_nrang * MAXNBR] */
    int                *d_wsum,         /* [n_beams * max_nrang] */
    int                *d_ncnt)         /* [n_beams * max_nrang] */
{
    int ob = blockIdx.x * blockDim.x + threadIdx.x;  /* output beam   */
    int or_ = blockIdx.y * blockDim.y + threadIdx.y; /* output range  */

    if (ob >= n_beams || or_ >= max_nrang) return;

    int out_idx = ob * max_nrang + or_;
    cuda_filter_cell_t *nbr = d_nbr + out_idx * CUDA_FILTER_MAXNBR;
    int cnt  = 0;
    int wsum = 0;

    for (int dt = 0; dt < depth; dt++) {
        /* inx is the index of the centre scan within src[] */
        int scan = (inx - (depth / 2) + dt + depth) % depth;
        int scan_off = scan * n_beams * max_nrang;

        for (int db = -1; db <= 1; db++) {
            int sb = ob + db;
            if (sb < 0 || sb >= n_beams) continue;

            for (int dr = -1; dr <= 1; dr++) {
                int sr = or_ + dr;
                if (sr < 0 || sr >= max_nrang) continue;

                int cidx = scan_off + sb * max_nrang + sr;
                if (!d_sct[cidx]) continue;  /* no scatter at this cell */

                int w = cell_weight(db, dr, dt, depth);
                wsum += w;
                if (cnt < CUDA_FILTER_MAXNBR)
                    nbr[cnt++] = d_cells[cidx];
            }
        }
    }

    d_wsum[out_idx] = wsum;
    d_ncnt[out_idx] = cnt;
}

/* -----------------------------------------------------------------------
 * Kernel 2: threshold check — mark which output cells have enough weight
 * -----------------------------------------------------------------------
 */
__global__ void cuda_filter_threshold_kernel(
    const int    *d_wsum,
    unsigned char *d_valid,
    int n_out, int mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_out) return;
    int thresh = (mode == 0) ? 12 : 24;
    d_valid[idx] = (d_wsum[idx] >= thresh) ? 1u : 0u;
}

/* -----------------------------------------------------------------------
 * Kernel 3: per-parameter mean and sigma over the neighbourhood
 * -----------------------------------------------------------------------
 * param_idx: 0=v, 1=p_l, 2=w_l, 3=p_0
 */
__device__ static float get_param(const cuda_filter_cell_t *c, int param)
{
    switch (param) {
        case CUDA_FILTER_PARAM_V:  return c->v;
        case CUDA_FILTER_PARAM_PL: return c->p_l;
        case CUDA_FILTER_PARAM_WL: return c->w_l;
        default:                   return c->p_0;
    }
}

__global__ void cuda_filter_stats_kernel(
    const cuda_filter_cell_t *d_nbr,     /* [n_out * MAXNBR] */
    const int                *d_ncnt,
    const unsigned char      *d_valid,
    int n_out, int param_idx,
    float *d_mean,                       /* [n_out] */
    float *d_sigma)                      /* [n_out] */
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_out) return;

    if (!d_valid[idx]) { d_mean[idx] = 0; d_sigma[idx] = 0; return; }

    const cuda_filter_cell_t *nbr = d_nbr + idx * CUDA_FILTER_MAXNBR;
    int cnt = d_ncnt[idx];

    float sum = 0.0f, sum2 = 0.0f;
    int   n   = 0;
    for (int i = 0; i < cnt; i++) {
        float v = get_param(&nbr[i], param_idx);
        if (v == 0.0f) continue;
        sum  += v;
        sum2 += v * v;
        n++;
    }

    if (n == 0) { d_mean[idx] = 0; d_sigma[idx] = 0; return; }

    float mean  = sum / n;
    float var   = sum2 / n - mean * mean;
    d_mean[idx]  = mean;
    d_sigma[idx] = (var > 0.0f) ? sqrtf(var) : 0.0f;
}

/* -----------------------------------------------------------------------
 * Device helper: insertion sort on a small float array (≤27 elements)
 */
__device__ static void insertion_sort(float *a, int n)
{
    for (int i = 1; i < n; i++) {
        float key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j+1] = a[j]; j--; }
        a[j+1] = key;
    }
}

/* -----------------------------------------------------------------------
 * Kernel 4: median extraction for one parameter
 * -----------------------------------------------------------------------
 * Filters neighbourhood to within-2σ of mean, sorts, takes centre value.
 * Fills the corresponding field of d_out.
 */
__global__ void cuda_filter_median_kernel(
    const cuda_filter_cell_t *d_nbr,
    const int                *d_ncnt,
    const unsigned char      *d_valid,
    const float              *d_mean,
    const float              *d_sigma,
    cuda_filter_out_t        *d_out,
    int n_out, int param_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_out) return;

    /* Write valid flag once (param 0 does it) */
    if (param_idx == CUDA_FILTER_PARAM_V)
        d_out[idx].valid = d_valid[idx];

    if (!d_valid[idx]) return;

    const cuda_filter_cell_t *nbr = d_nbr + idx * CUDA_FILTER_MAXNBR;
    int   cnt   = d_ncnt[idx];
    float mean  = d_mean[idx];
    float sigma = d_sigma[idx];
    float lo    = mean - 2.0f * sigma;
    float hi    = mean + 2.0f * sigma;

    float buf[CUDA_FILTER_MAXNBR];
    int   n = 0;
    for (int i = 0; i < cnt; i++) {
        float v = get_param(&nbr[i], param_idx);
        if (v == 0.0f) continue;
        if (v >= lo && v <= hi) buf[n++] = v;
    }

    if (n == 0) {
        /* Store zero and return; caller can decide how to handle */
        switch (param_idx) {
            case CUDA_FILTER_PARAM_V:  d_out[idx].v   = 0; d_out[idx].v_e   = 0; break;
            case CUDA_FILTER_PARAM_PL: d_out[idx].p_l = 0; d_out[idx].p_l_e = 0; break;
            case CUDA_FILTER_PARAM_WL: d_out[idx].w_l = 0; d_out[idx].w_l_e = 0; break;
            default:                   d_out[idx].p_0 = 0; d_out[idx].p_0_e = 0; break;
        }
        return;
    }

    insertion_sort(buf, n);
    float median = buf[n / 2];

    /* Error = stddev of the filtered set */
    float sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) { sum += buf[i]; sum2 += buf[i] * buf[i]; }
    float mn  = sum / n;
    float err = (n > 1) ? sqrtf(fmaxf(0.0f, sum2/n - mn*mn)) : 0.0f;

    switch (param_idx) {
        case CUDA_FILTER_PARAM_V:
            d_out[idx].v   = median; d_out[idx].v_e   = err; break;
        case CUDA_FILTER_PARAM_PL:
            d_out[idx].p_l = median; d_out[idx].p_l_e = err; break;
        case CUDA_FILTER_PARAM_WL:
            d_out[idx].w_l = median; d_out[idx].w_l_e = err; break;
        default:
            d_out[idx].p_0 = median; d_out[idx].p_0_e = err; break;
    }

    /* Ground scatter: flag if abs(v)<30 && w_l<30 (written on last param pass) */
    if (param_idx == CUDA_FILTER_PARAM_P0) {
        d_out[idx].gsct =
            (fabsf(d_out[idx].v) < 30.0f && d_out[idx].w_l < 30.0f) ? 1 : 0;
    }
}

/* -----------------------------------------------------------------------
 * Host wrapper — called from C
 */
extern "C" int FilterRadarScanCuda(
    int mode, int depth, int inx,
    const cuda_filter_cell_t **src_cells,
    const unsigned char      **src_sct,
    const cuda_filter_beam_desc_t *src_bm,
    int n_beams, int max_nrang,
    cuda_filter_out_t *dst_out,
    int prm, int isort)
{
    (void)src_bm; (void)prm; (void)isort;

    int n_out   = n_beams * max_nrang;
    size_t flat = (size_t)depth * n_beams * max_nrang;

    /* --- allocate device arrays ---------------------------------------- */
    cuda_filter_cell_t *d_cells = NULL;
    unsigned char      *d_sct   = NULL;
    cuda_filter_cell_t *d_nbr   = NULL;
    int                *d_wsum  = NULL;
    int                *d_ncnt  = NULL;
    unsigned char      *d_valid = NULL;
    float              *d_mean  = NULL;
    float              *d_sigma = NULL;
    cuda_filter_out_t  *d_out   = NULL;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(&d_cells, flat * sizeof(cuda_filter_cell_t)); if (err) goto fail;
    err = cudaMalloc(&d_sct,   flat * sizeof(unsigned char));      if (err) goto fail;
    err = cudaMalloc(&d_nbr,   (size_t)n_out * CUDA_FILTER_MAXNBR
                                             * sizeof(cuda_filter_cell_t)); if (err) goto fail;
    err = cudaMalloc(&d_wsum,  n_out * sizeof(int));           if (err) goto fail;
    err = cudaMalloc(&d_ncnt,  n_out * sizeof(int));           if (err) goto fail;
    err = cudaMalloc(&d_valid, n_out * sizeof(unsigned char)); if (err) goto fail;
    err = cudaMalloc(&d_mean,  n_out * sizeof(float));         if (err) goto fail;
    err = cudaMalloc(&d_sigma, n_out * sizeof(float));         if (err) goto fail;
    err = cudaMalloc(&d_out,   n_out * sizeof(cuda_filter_out_t)); if (err) goto fail;

    /* --- copy input data ------------------------------------------------ */
    for (int d = 0; d < depth; d++) {
        size_t off = (size_t)d * n_beams * max_nrang;
        cudaMemcpy(d_cells + off, src_cells[d],
                   n_beams * max_nrang * sizeof(cuda_filter_cell_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_sct   + off, src_sct[d],
                   n_beams * max_nrang * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
    }

    /* --- kernel launches ----------------------------------------------- */
    {
        dim3 block(16, 16);
        dim3 grid((n_beams  + 15) / 16,
                  (max_nrang + 15) / 16);
        cuda_filter_gather_kernel<<<grid, block>>>(
            d_cells, d_sct, depth, n_beams, max_nrang, inx,
            d_nbr, d_wsum, d_ncnt);
    }

    {
        int threads = 256;
        int blocks  = (n_out + threads - 1) / threads;
        cuda_filter_threshold_kernel<<<blocks, threads>>>(
            d_wsum, d_valid, n_out, mode);

        /* 4 parameter passes */
        for (int p = 0; p < CUDA_FILTER_NPARAMS; p++) {
            cuda_filter_stats_kernel<<<blocks, threads>>>(
                d_nbr, d_ncnt, d_valid, n_out, p, d_mean, d_sigma);
            cuda_filter_median_kernel<<<blocks, threads>>>(
                d_nbr, d_ncnt, d_valid, d_mean, d_sigma, d_out, n_out, p);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    /* --- copy results back --------------------------------------------- */
    cudaMemcpy(dst_out, d_out, n_out * sizeof(cuda_filter_out_t),
               cudaMemcpyDeviceToHost);

fail:
    if (d_cells) cudaFree(d_cells);
    if (d_sct)   cudaFree(d_sct);
    if (d_nbr)   cudaFree(d_nbr);
    if (d_wsum)  cudaFree(d_wsum);
    if (d_ncnt)  cudaFree(d_ncnt);
    if (d_valid) cudaFree(d_valid);
    if (d_mean)  cudaFree(d_mean);
    if (d_sigma) cudaFree(d_sigma);
    if (d_out)   cudaFree(d_out);

    return (err == cudaSuccess) ? 0 : -1;
}
