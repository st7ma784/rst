#ifndef FILTER_1_8_CUDA_H
#define FILTER_1_8_CUDA_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of RadarCell for GPU transfer */
typedef struct {
    int   gsct;
    float p_0, p_0_e;
    float v,   v_e;
    float w_l, w_l_e;
    float p_l, p_l_e;
    float phi0;
} cuda_filter_cell_t;

/* Flattened beam descriptor passed to GPU */
typedef struct {
    int bm;
    int nrang;
    int frang, rsep, rxrise, freq, noise, atten, nave, channel;
} cuda_filter_beam_desc_t;

/* Per-output-cell result written back to RadarCell fields */
typedef struct {
    float v,   v_e;
    float p_l, p_l_e;
    float w_l, w_l_e;
    float p_0, p_0_e;
    int   gsct;
    int   valid;    /* 1 if weight threshold passed */
} cuda_filter_out_t;

/* Maximum neighbourhood depth / width / height as in filter.h */
#define CUDA_FILTER_DEPTH  3
#define CUDA_FILTER_WIDTH  3
#define CUDA_FILTER_HEIGHT 3
#define CUDA_FILTER_MAXNBR (CUDA_FILTER_DEPTH * CUDA_FILTER_WIDTH * CUDA_FILTER_HEIGHT)

/* Parameter indices used by stats / median kernels */
#define CUDA_FILTER_PARAM_V   0
#define CUDA_FILTER_PARAM_PL  1
#define CUDA_FILTER_PARAM_WL  2
#define CUDA_FILTER_PARAM_P0  3
#define CUDA_FILTER_NPARAMS   4

/*
 * GPU-accelerated equivalent of FilterRadarScan().
 * src[depth] are pre-flattened scan cell arrays (caller allocates).
 * Returns 0 on success, -1 if no CUDA device or allocation failure
 * (caller should then fall back to CPU FilterRadarScan).
 */
int FilterRadarScanCuda(
    int mode, int depth, int inx,
    const cuda_filter_cell_t **src_cells,   /* [depth][n_beams * max_nrang] */
    const unsigned char      **src_sct,     /* [depth][n_beams * max_nrang] */
    const cuda_filter_beam_desc_t *src_bm,  /* [depth][n_beams] beam descriptors */
    int n_beams, int max_nrang,
    cuda_filter_out_t *dst_out,             /* [n_beams * max_nrang] output */
    int prm, int isort);

#ifdef __cplusplus
}
#endif

#endif /* FILTER_1_8_CUDA_H */
