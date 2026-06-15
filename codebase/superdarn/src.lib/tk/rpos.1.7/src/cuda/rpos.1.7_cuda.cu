/*
 * rpos.1.7_cuda.cu
 * ================
 * GPU-parallel batch coordinate transforms for SuperDARN radar range gates.
 *
 * Implements the geometric (RPosGeo) path from cnvtcoord.c in parallel.
 * AACGM magnetic-coordinate conversion (RPosMag) is CPU-only and is not
 * ported here — this file covers slant_range, geodtgc, fldpnt, fldpnth,
 * RPosGeo and RPosCubic for all range gates of one or more beams.
 *
 * Device helper functions mirror cnvtcoord.c exactly so CPU and CUDA
 * results are numerically equivalent.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA rpos error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
        } \
    } while (0)

/* -----------------------------------------------------------------------
 * Degree-based trig helpers matching rmath.h macros
 */
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232

__device__ __forceinline__ double d_sind(double x)  { return sin(x*DEG2RAD); }
__device__ __forceinline__ double d_cosd(double x)  { return cos(x*DEG2RAD); }
__device__ __forceinline__ double d_tand(double x)  { return tan(x*DEG2RAD); }
__device__ __forceinline__ double d_atand(double x) { return atan(x)*RAD2DEG; }
__device__ __forceinline__ double d_asind(double x) { return asin(x)*RAD2DEG; }
__device__ __forceinline__ double d_acosd(double x) { return acos(x)*RAD2DEG; }
__device__ __forceinline__ double d_atan2d(double y, double x) { return atan2(y,x)*RAD2DEG; }

/* -----------------------------------------------------------------------
 * Device: slant_range — mirrors slant_range() in cnvtcoord.c
 */
__device__ double d_slant_range(int frang, int rsep, double rxris,
                                 double range_edge, int gate)
{
    int lagfr = frang * 20 / 3;
    int smsep = rsep  * 20 / 3;
    return (lagfr - rxris + (gate - 1) * smsep + range_edge) * 0.15;
}

/* -----------------------------------------------------------------------
 * Device: geodtgc — WGS84 geodetic ↔ geocentric
 * iopt > 0 : geodetic→geocentric,   iopt ≤ 0 : geocentric→geodetic
 */
__device__ void d_geodtgc(int iopt,
                           double *gdlat, double *gdlon,
                           double *grho,  double *glat,
                           double *glon,  double *del)
{
    const double a  = 6378.137;
    const double f  = 1.0 / 298.257223563;
    const double b  = a * (1.0 - f);
    const double e2 = (a*a) / (b*b) - 1.0;

    if (iopt > 0) {
        *glat = d_atand((b*b)/(a*a) * d_tand(*gdlat));
        *glon = *gdlon;
        if (*glon > 180.0) *glon -= 360.0;
    } else {
        *gdlat = d_atand((a*a)/(b*b) * d_tand(*glat));
        *gdlon = *glon;
    }
    *grho = a / sqrt(1.0 + e2 * d_sind(*glat) * d_sind(*glat));
    *del  = *gdlat - *glat;
}

/* -----------------------------------------------------------------------
 * Device: fldpnt — geocentric coordinates of a field point
 */
__device__ void d_fldpnt(double rrho, double rlat, double rlon,
                          double ral,  double rel,  double r,
                          double *frho, double *flat, double *flon)
{
    double sinteta = d_sind(90.0 - rlat);
    double rx = rrho * sinteta * d_cosd(rlon);
    double ry = rrho * sinteta * d_sind(rlon);
    double rz = rrho * d_cosd(90.0 - rlat);

    double sx = -r * d_cosd(rel) * d_cosd(ral);
    double sy =  r * d_cosd(rel) * d_sind(ral);
    double sz =  r * d_sind(rel);

    double tx =  d_cosd(90.0 - rlat)*sx + d_sind(90.0 - rlat)*sz;
    double ty =  sy;
    double tz = -d_sind(90.0 - rlat)*sx + d_cosd(90.0 - rlat)*sz;

    sx = d_cosd(rlon)*tx - d_sind(rlon)*ty;
    sy = d_sind(rlon)*tx + d_cosd(rlon)*ty;
    sz = tz;

    tx = rx + sx;  ty = ry + sy;  tz = rz + sz;

    *frho = sqrt(tx*tx + ty*ty + tz*tz);
    *flat = 90.0 - d_acosd(tz / (*frho));
    *flon = (tx == 0.0 && ty == 0.0) ? 0.0 : d_atan2d(ty, tx);
}

/* -----------------------------------------------------------------------
 * Device: geocnvrt — local horizon → geocentric azimuth/elevation
 */
__device__ void d_geocnvrt(double gdlat, double gdlon,
                            double xal, double xel,
                            double *ral, double *rel_out)
{
    double kxg = d_cosd(xel) * d_sind(xal);
    double kyg = d_cosd(xel) * d_cosd(xal);
    double kzg = d_sind(xel);
    double rrad, rlat, rlon, del;
    d_geodtgc(1, &gdlat, &gdlon, &rrad, &rlat, &rlon, &del);
    double kxr =  kxg;
    double kyr =  kyg * d_cosd(del) + kzg * d_sind(del);
    double kzr = -kyg * d_sind(del) + kzg * d_cosd(del);
    *ral     = d_atan2d(kxr, kyr);
    *rel_out = d_atand(kzr / sqrt(kxr*kxr + kyr*kyr));
}

/* -----------------------------------------------------------------------
 * Device: fldpnth — iterative field-point with virtual height model
 * Hard iteration cap of 20 (CPU typically converges in 8–15).
 */
__device__ void d_fldpnth(double gdlat, double gdlon, double psi, double bore,
                           double fh, double r,
                           double *frho, double *flat, double *flon,
                           int chisham)
{
    double rrad, rlat, rlon, del, frad;
    double tan_azi, azi, rel, xel, fhx, xal, ral_out, dum1, dum2, dum3, dum;
    double gmma, beta, xh;

    if (chisham) {
        const double A[3] = {108.974,    384.416,    1098.28};
        const double B[3] = {0.0191271, -0.178640,  -0.354557};
        const double C[3] = {6.68283e-5, 1.81405e-4, 9.39961e-5};
        if      (r < 787.5)   xh = A[0] + B[0]*r + C[0]*r*r;
        else if (r <= 2137.5) xh = A[1] + B[1]*r + C[1]*r*r;
        else                  xh = A[2] + B[2]*r + C[2]*r*r;
        if (r < 115) xh = (r / 115.0) * 112.0;
    } else {
        if (fh <= 150) xh = fh;
        else {
            if      (r <= 600) xh = 115;
            else if (r < 800)  xh = (r - 600) / 200.0 * (fh - 115) + 115;
            else                xh = fh;
        }
        if (r < 150) xh = (r / 150.0) * 115.0;
    }

    if (r == 0) r = 0.1;

    d_geodtgc(1, &gdlat, &gdlon, &rrad, &rlat, &rlon, &del);
    frad = rrad;

    int iter = 0;
    do {
        *frho = frad + xh;
        rel   = d_asind((*frho**frho - rrad*rrad - r*r) / (2.0*rrad*r));

        if (chisham && r > 2137.5) {
            gmma = d_acosd((rrad*rrad + *frho**frho - r*r) / (2.0*rrad**frho));
            beta = d_asind(rrad * d_sind(gmma/3.0) / (r/3.0));
            xel  = 90.0 - beta - gmma/3.0;
        } else {
            xel = rel;
        }

        double cp2 = d_cosd(psi)*d_cosd(psi) - d_sind(xel)*d_sind(xel);
        tan_azi = (cp2 < 0) ? 1e32
                : sqrt((d_sind(psi)*d_sind(psi)) / cp2);
        azi = (psi > 0) ? d_atand(tan_azi) : -d_atand(tan_azi);
        xal = azi + bore;

        d_geocnvrt(gdlat, gdlon, xal, xel, &ral_out, &dum);
        d_fldpnt(rrad, rlat, rlon, ral_out, rel, r, frho, flat, flon);
        d_geodtgc(-1, &dum1, &dum2, &frad, flat, flon, &dum3);
        fhx = *frho - frad;
        iter++;
    } while (fabs(fhx - xh) > 0.5 && iter < 20);
}

/* -----------------------------------------------------------------------
 * Kernel 1: batch slant range — one thread per range gate
 */
__global__ void cuda_rpos_slant_range_kernel(
    int n, int frang, int rsep, double rxrise, double range_edge,
    double *d_ranges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_ranges[idx] = d_slant_range(frang, rsep, rxrise, range_edge, idx + 1);
}

/* -----------------------------------------------------------------------
 * Kernel 2: batch geodtgc (geodetic → geocentric)
 */
__global__ void cuda_rpos_geodtgc_kernel(
    const double *d_gdlat, const double *d_gdlon, int n,
    double *d_grho, double *d_glat, double *d_glon, double *d_del)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double gdlat = d_gdlat[idx], gdlon = d_gdlon[idx];
    double grho, glat, glon, del;
    d_geodtgc(1, &gdlat, &gdlon, &grho, &glat, &glon, &del);
    d_grho[idx] = grho;  d_glat[idx] = glat;
    d_glon[idx] = glon;  d_del[idx]  = del;
}

/* -----------------------------------------------------------------------
 * Kernel 3: batch fldpnt
 */
__global__ void cuda_rpos_fldpnt_kernel(
    const double *d_rrho, const double *d_rlat, const double *d_rlon,
    const double *d_ral,  const double *d_rel,  const double *d_r,
    int n, double *d_frho, double *d_flat, double *d_flon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_fldpnt(d_rrho[idx], d_rlat[idx], d_rlon[idx],
             d_ral[idx],  d_rel[idx],  d_r[idx],
             &d_frho[idx], &d_flat[idx], &d_flon[idx]);
}

/* -----------------------------------------------------------------------
 * Kernel 4: batch RPosGeo — one thread per (beam, gate) pair
 * Radar site parameters are scalars broadcast to all threads.
 */
__global__ void cuda_rpos_geo_batch_kernel(
    int n_gates, int n_beams,
    double geolat, double geolon, double boresite,
    double bmsep, double bmoff, double recrise, double maxbeam,
    int frang, int rsep, int rxrise, double height, int chisham, int center,
    double *d_rho, double *d_lat, double *d_lng)
{
    int beam = blockIdx.x * blockDim.x + threadIdx.x;
    int gate = blockIdx.y * blockDim.y + threadIdx.y;
    if (beam >= n_beams || gate >= n_gates) return;

    double bm_edge    = center ? 0.0 : -bmsep * 0.5;
    double range_edge = center ? 0.0 : -0.5 * rsep * 20.0 / 3.0;
    double rx         = (rxrise == 0) ? recrise : (double)rxrise;
    double offset     = maxbeam / 2.0 - 0.5;
    double psi        = bmsep * (beam - offset) + bm_edge + bmoff;
    double r          = d_slant_range(frang, rsep, rx, range_edge, gate + 1);

    double frho, flat, flon;
    d_fldpnth(geolat, geolon, psi, boresite, height, r,
              &frho, &flat, &flon, chisham);

    int idx = beam * n_gates + gate;
    d_rho[idx] = frho;
    d_lat[idx] = flat;
    d_lng[idx] = flon;
}

/* -----------------------------------------------------------------------
 * Kernel 5: batch RPosCubic — geocentric spherical → Cartesian unit vector
 */
__global__ void cuda_rpos_cubic_batch_kernel(
    const double *d_rho, const double *d_lat, const double *d_lng,
    int n, double *d_x, double *d_y, double *d_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double rho = d_rho[idx], lat = d_lat[idx], lng = d_lng[idx];
    double sinteta = d_sind(90.0 - lat);
    d_x[idx] = sinteta * d_cosd(lng) / rho;
    d_y[idx] = sinteta * d_sind(lng) / rho;
    d_z[idx] = d_cosd(90.0 - lat)   / rho;
}

/* -----------------------------------------------------------------------
 * C-linkage host wrappers
 */
extern "C" {

/*
 * cuda_rpos_geo_scan: RPosGeo for all (beam, gate) pairs in a scan.
 * h_rho/h_lat/h_lng: caller-allocated output arrays [n_beams * n_gates].
 */
cudaError_t cuda_rpos_geo_scan(
    double geolat, double geolon, double boresite,
    double bmsep,  double bmoff,  double recrise, int maxbeam,
    int frang, int rsep, int rxrise, double height, int chisham, int center,
    int n_beams, int n_gates,
    double *h_rho, double *h_lat, double *h_lng)
{
    int n = n_beams * n_gates;
    double *d_rho, *d_lat, *d_lng;

    CUDA_CHECK(cudaMalloc(&d_rho, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lat, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lng, n * sizeof(double)));

    dim3 block(16, 16);
    dim3 grid((n_beams  + 15) / 16, (n_gates + 15) / 16);

    cuda_rpos_geo_batch_kernel<<<grid, block>>>(
        n_gates, n_beams,
        geolat, geolon, boresite, bmsep, bmoff, recrise, (double)maxbeam,
        frang, rsep, rxrise, height, chisham, center,
        d_rho, d_lat, d_lng);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(h_rho, d_rho, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lat, d_lat, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lng, d_lng, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rho);  cudaFree(d_lat);  cudaFree(d_lng);
    return cudaGetLastError();
}

/*
 * cuda_rpos_cubic_scan: convert geocentric spherical to Cartesian unit vectors.
 */
cudaError_t cuda_rpos_cubic_scan(
    const double *h_rho, const double *h_lat, const double *h_lng, int n,
    double *h_x, double *h_y, double *h_z)
{
    double *d_rho, *d_lat, *d_lng, *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_rho, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lat, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lng, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z,   n * sizeof(double)));

    cudaMemcpy(d_rho, h_rho, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lat, h_lat, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lng, h_lng, n*sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + 255) / 256;
    cuda_rpos_cubic_batch_kernel<<<blocks, threads>>>(
        d_rho, d_lat, d_lng, n, d_x, d_y, d_z);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(h_x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rho); cudaFree(d_lat); cudaFree(d_lng);
    cudaFree(d_x);   cudaFree(d_y);   cudaFree(d_z);
    return cudaGetLastError();
}

bool cuda_rpos_is_available(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n > 0;
}

} /* extern "C" */
