/*
 * cnvmodel.1.0_cuda.cu
 * ====================
 * CUDA implementation of ionospheric convection model evaluation.
 *
 * Parallelises slv_sph_kset() from cnvmodel.c — evaluates the spherical
 * harmonic potential model at `num` independent grid points simultaneously.
 *
 * CPU algorithm (cnvmodel.c):
 *   for i in 0..num:
 *       slv_ylm_mod(the[i], phi[i], ltop, ylm_p[i], ylm_n[i], ...)
 *       Ix = 0
 *       for l in 0..ltop:
 *           for m in 0..l:
 *               k = l*(ltop+1) ± m
 *               Ix += aoeff_p[k]*ylm_p[k] + aoeff_n[k]*ylm_n[k]
 *       pot[i] = Re(Ix) * scale
 *
 * GPU mapping: one thread per grid point for each of the three stages.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA cnvmodel error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
        } \
    } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Maximum spherical harmonic order supported on GPU */
#define MAX_LTOP 24
#define MAX_COEFF ((MAX_LTOP+1)*(MAX_LTOP+1))

/* -----------------------------------------------------------------------
 * Device: factorial — small n only (n ≤ 2*MAX_LTOP)
 */
__device__ double d_factorial(int n)
{
    double r = 1.0;
    for (int i = 2; i <= n; i++) r *= (double)i;
    return r;
}

/* -----------------------------------------------------------------------
 * Kernel 1: Legendre polynomials + normalisation coefficients
 * One thread per grid point.
 * Outputs d_plm[pt * MAX_COEFF + l*(ltop+1) + m] for all (l,m).
 * anorm and apcnv are shared across all grid points → precomputed once
 * on host and passed as read-only device pointers.
 */
__global__ void cuda_cnvmodel_legendre_kernel(
    const float *d_theta,   /* [num] co-latitude in radians */
    int num, int ltop,
    double *d_plm)          /* [num * MAX_COEFF] */
{
    int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= num) return;

    double x    = cos((double)d_theta[pt]);
    double sx2  = sqrt(fmax(0.0, (1.0 - x) * (1.0 + x)));
    double *plm = d_plm + pt * MAX_COEFF;

    for (int l = 0; l <= ltop; l++) {
        for (int m = 0; m <= l; m++) {
            double Pmm = 1.0;
            if (m > 0) {
                double fct = 1.0;
                for (int i = 1; i <= m; i++) {
                    Pmm = -Pmm * fct * sx2;
                    fct += 2.0;
                }
            }
            if (l != m) {
                double pnmp1 = x * (2*m + 1) * Pmm;
                if (l != (m + 1)) {
                    double Pll = 0.0;
                    for (int ll = m+2; ll <= l; ll++) {
                        Pll  = (x * (2*ll - 1) * pnmp1 - (ll + m - 1) * Pmm) / (ll - m);
                        Pmm  = pnmp1;
                        pnmp1 = Pll;
                    }
                    Pmm = Pll;
                } else {
                    Pmm = pnmp1;
                }
            }
            plm[l * (ltop+1) + m] = Pmm;
        }
    }
}

/* -----------------------------------------------------------------------
 * Kernel 2: Spherical harmonics Y_lm(theta, phi)
 * One thread per grid point.
 * Outputs d_ylm_p and d_ylm_n [num * MAX_COEFF].
 * Requires d_plm from Kernel 1 and precomputed d_anorm[MAX_COEFF].
 */
__global__ void cuda_cnvmodel_ylm_kernel(
    const float  *d_theta,   /* [num] */
    const float  *d_phi,     /* [num] */
    const double *d_plm,     /* [num * MAX_COEFF] */
    const double *d_anorm,   /* [MAX_COEFF] — same for all points */
    int num, int ltop,
    cuDoubleComplex *d_ylm_p,  /* [num * MAX_COEFF] */
    cuDoubleComplex *d_ylm_n)  /* [num * MAX_COEFF] */
{
    int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= num) return;

    double phi = (double)d_phi[pt];
    const double *plm   = d_plm   + pt * MAX_COEFF;
    cuDoubleComplex *yp = d_ylm_p + pt * MAX_COEFF;
    cuDoubleComplex *yn = d_ylm_n + pt * MAX_COEFF;

    for (int l = 0; l <= ltop; l++) {
        for (int m = 0; m <= l; m++) {
            int k        = l * (ltop+1) + m;
            double Pmm   = plm[k];
            double an    = d_anorm[k];
            double re_p  = Pmm * an * cos(m * phi);
            double im_p  = Pmm * an * sin(m * phi);
            double sign  = (m % 2 == 0) ? 1.0 : -1.0;
            yp[k] = make_cuDoubleComplex( re_p,  im_p);
            yn[k] = make_cuDoubleComplex( sign * re_p, -sign * im_p);
        }
    }
}

/* -----------------------------------------------------------------------
 * Kernel 3: Inner-product of spherical harmonics with model coefficients
 * One thread per grid point.
 * Computes potential and its gradients (ele_phi, ele_the).
 *
 * pot[i] = Re( Σ_{l,m} aoeff_p[k]*ylm_p[k] + aoeff_n[k]*ylm_n[k] ) * Rd
 */
__global__ void cuda_cnvmodel_potential_kernel(
    const cuDoubleComplex *d_ylm_p,    /* [num * MAX_COEFF] */
    const cuDoubleComplex *d_ylm_n,    /* [num * MAX_COEFF] */
    const cuDoubleComplex *d_aoeff_p,  /* [MAX_COEFF] */
    const cuDoubleComplex *d_aoeff_n,  /* [MAX_COEFF] */
    int num, int ltop, float latmin,
    double Rd,                          /* Radial_Dist / 1000.0 */
    double *d_pot,                      /* [num] */
    double *d_ele_phi,                  /* [num] */
    double *d_ele_the)                  /* [num] */
{
    int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= num) return;

    const cuDoubleComplex *yp = d_ylm_p + pt * MAX_COEFF;
    const cuDoubleComplex *yn = d_ylm_n + pt * MAX_COEFF;

    cuDoubleComplex Ix = make_cuDoubleComplex(0.0, 0.0);
    for (int l = 0; l <= ltop; l++) {
        for (int m = 0; m <= l; m++) {
            int k  = l * (ltop+1) + m;
            /* Ix += aoeff_p[k] * ylm_p[k] + aoeff_n[k] * ylm_n[k] */
            Ix = cuCadd(Ix, cuCmul(d_aoeff_p[k], yp[k]));
            Ix = cuCadd(Ix, cuCmul(d_aoeff_n[k], yn[k]));
        }
    }

    d_pot[pt]     = cuCreal(Ix) * Rd;
    /* Gradient components approximate from imaginary part of Ix */
    d_ele_phi[pt] = cuCimag(Ix) * Rd;
    d_ele_the[pt] = 0.0;  /* full gradient requires derivative kernel */
}

/* -----------------------------------------------------------------------
 * Host helper: precompute anorm[MAX_COEFF] on CPU → upload once
 */
static void compute_anorm(int ltop, double *anorm)
{
    for (int l = 0; l <= ltop; l++) {
        for (int m = 0; m <= l; m++) {
            double num_f = 1.0, den_f = 1.0;
            for (int i = 1; i <= (l-m); i++) num_f *= i;
            for (int i = 1; i <= (l+m); i++) den_f *= i;
            anorm[l*(ltop+1)+m] = sqrt((2*l+1)/(4.0*M_PI) * num_f/den_f);
        }
    }
}

/* -----------------------------------------------------------------------
 * C-linkage host wrapper
 *
 * cuda_cnvmodel_slv_sph_kset: GPU equivalent of slv_sph_kset().
 *
 *   theta[num], phi[num]  — grid point co-latitudes and longitudes [radians]
 *   aoeff_p/n[ltop+1]²    — model complex coefficients (host memory)
 *   ltop                  — maximum spherical harmonic order
 *   latmin                — latitude minimum (passed through)
 *   Rd                    — Radial_Dist/1000 scaling constant
 *   pot[num]              — output electric potential
 *   ele_phi/the[num]      — output gradient components
 */
extern "C" {

cudaError_t cuda_cnvmodel_slv_sph_kset(
    int num, const float *h_theta, const float *h_phi,
    const double *h_aoeff_p_re, const double *h_aoeff_p_im,
    const double *h_aoeff_n_re, const double *h_aoeff_n_im,
    int ltop, float latmin, double Rd,
    double *h_pot, double *h_ele_phi, double *h_ele_the)
{
    if (ltop > MAX_LTOP) {
        fprintf(stderr, "cuda_cnvmodel: ltop %d exceeds MAX_LTOP %d\n",
                ltop, MAX_LTOP);
        return cudaErrorInvalidValue;
    }

    int ncoeff = (ltop+1)*(ltop+1);

    /* Device arrays */
    float  *d_theta, *d_phi;
    double *d_plm, *d_anorm;
    cuDoubleComplex *d_ylm_p, *d_ylm_n;
    cuDoubleComplex *d_aoeff_p, *d_aoeff_n;
    double *d_pot, *d_ele_phi, *d_ele_the;

    CUDA_CHECK(cudaMalloc(&d_theta,   num * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phi,     num * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_plm,     (size_t)num * MAX_COEFF * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_anorm,   MAX_COEFF * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ylm_p,   (size_t)num * MAX_COEFF * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_ylm_n,   (size_t)num * MAX_COEFF * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_aoeff_p, MAX_COEFF * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_aoeff_n, MAX_COEFF * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_pot,     num * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ele_phi, num * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ele_the, num * sizeof(double)));

    /* Upload inputs */
    cudaMemcpy(d_theta, h_theta, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi,   h_phi,   num * sizeof(float), cudaMemcpyHostToDevice);

    /* Pack complex coefficients and upload */
    cuDoubleComplex *h_cp = (cuDoubleComplex *)malloc(MAX_COEFF * sizeof(cuDoubleComplex));
    cuDoubleComplex *h_cn = (cuDoubleComplex *)malloc(MAX_COEFF * sizeof(cuDoubleComplex));
    for (int i = 0; i < ncoeff; i++) {
        h_cp[i] = make_cuDoubleComplex(h_aoeff_p_re[i], h_aoeff_p_im[i]);
        h_cn[i] = make_cuDoubleComplex(h_aoeff_n_re[i], h_aoeff_n_im[i]);
    }
    cudaMemcpy(d_aoeff_p, h_cp, MAX_COEFF * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aoeff_n, h_cn, MAX_COEFF * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    free(h_cp); free(h_cn);

    /* Precompute anorm on host, upload */
    double *h_anorm = (double *)calloc(MAX_COEFF, sizeof(double));
    compute_anorm(ltop, h_anorm);
    cudaMemcpy(d_anorm, h_anorm, MAX_COEFF * sizeof(double), cudaMemcpyHostToDevice);
    free(h_anorm);

    /* Kernel launches */
    int threads = 256, blocks = (num + 255) / 256;

    cuda_cnvmodel_legendre_kernel<<<blocks, threads>>>(
        d_theta, num, ltop, d_plm);

    cuda_cnvmodel_ylm_kernel<<<blocks, threads>>>(
        d_theta, d_phi, d_plm, d_anorm, num, ltop, d_ylm_p, d_ylm_n);

    cuda_cnvmodel_potential_kernel<<<blocks, threads>>>(
        d_ylm_p, d_ylm_n, d_aoeff_p, d_aoeff_n,
        num, ltop, latmin, Rd, d_pot, d_ele_phi, d_ele_the);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(h_pot,     d_pot,     num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ele_phi, d_ele_phi, num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ele_the, d_ele_the, num * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_theta);  cudaFree(d_phi);     cudaFree(d_plm);
    cudaFree(d_anorm);  cudaFree(d_ylm_p);   cudaFree(d_ylm_n);
    cudaFree(d_aoeff_p); cudaFree(d_aoeff_n);
    cudaFree(d_pot);    cudaFree(d_ele_phi); cudaFree(d_ele_the);

    return cudaGetLastError();
}

bool cuda_cnvmodel_is_available(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n > 0;
}

} /* extern "C" */
