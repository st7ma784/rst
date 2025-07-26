#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include "fitblk.h"

/**
 * @brief Matrix-vector multiplication using AVX intrinsics
 * 
 * @param A Input matrix (row-major order)
 * @param x Input vector
 * @param y Output vector (y = A * x)
 * @param nrows Number of rows in matrix A
 * @param ncols Number of columns in matrix A
 */
void matrix_vector_mult_avx(const float *A, const float *x, float *y, int nrows, int ncols) {
    const int vec_size = 8; // AVX can process 8 floats at once
    
    for (int i = 0; i < nrows; i++) {
        __m256 sum = _mm256_setzero_ps();
        int j;
        
        // Process elements in chunks of 8
        for (j = 0; j <= ncols - vec_size; j += vec_size) {
            __m256 a = _mm256_loadu_ps(&A[i * ncols + j]);
            __m256 b = _mm256_loadu_ps(&x[j]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
        }
        
        // Horizontal sum of the vector
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), 
                                  _mm256_extractf128_ps(sum, 1));
        sum128 = _mm_add_ps(sum128, _mm_movehdup_ps(sum128));
        sum128 = _mm_add_ss(sum128, _mm_movehl_ps(sum128, sum128));
        
        float result;
        _mm_store_ss(&result, sum128);
        
        // Handle remaining elements
        for (; j < ncols; j++) {
            result += A[i * ncols + j] * x[j];
        }
        
        y[i] = result;
    }
}

/**
 * @brief Matrix-matrix multiplication using AVX intrinsics
 * 
 * @param A First input matrix (row-major order)
 * @param B Second input matrix (row-major order)
 * @param C Output matrix (C = A * B)
 * @param m Rows of A and C
 * @param n Columns of A / Rows of B
 * @param p Columns of B and C
 */
void matrix_matrix_mult_avx(const float *A, const float *B, float *C, 
                           int m, int n, int p) {
    // Zero out the output matrix
    memset(C, 0, m * p * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            
            // Process 8 columns at a time
            int j;
            for (j = 0; j <= p - 8; j += 8) {
                __m256 b = _mm256_loadu_ps(&B[k * p + j]);
                __m256 c = _mm256_loadu_ps(&C[i * p + j]);
                c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_set1_ps(a), b));
                _mm256_storeu_ps(&C[i * p + j], c);
            }
            
            // Handle remaining columns
            for (; j < p; j++) {
                C[i * p + j] += a * B[k * p + j];
            }
        }
    }
}

/**
 * @brief In-place matrix transpose using AVX intrinsics
 * 
 * @param A Matrix to transpose (square matrix in-place)
 * @param n Size of the matrix (n x n)
 */
void matrix_transpose_avx(float *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // Load elements to swap
            __m256 row = _mm256_loadu_ps(&A[i * n + j]);
            __m256 col = _mm256_loadu_ps(&A[j * n + i]);
            
            // Store swapped elements
            _mm256_storeu_ps(&A[j * n + i], row);
            _mm256_storeu_ps(&A[i * n + j], col);
        }
    }
}

/**
 * @brief Vector addition using AVX intrinsics
 * 
 * @param a First input vector
 * @param b Second input vector
 * @param c Output vector (c = a + b)
 * @param n Length of vectors
 */
void vector_add_avx(const float *a, const float *b, float *c, int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief Vector dot product using AVX intrinsics
 * 
 * @param a First input vector
 * @param b Second input vector
 * @param n Length of vectors
 * @return float Dot product of a and b
 */
float vector_dot_avx(const float *a, const float *b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i;
    
    // Process 8 elements at a time
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    
    // Horizontal sum of the vector
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), 
                              _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_add_ps(sum128, _mm_movehdup_ps(sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehl_ps(sum128, sum128));
    
    float result;
    _mm_store_ss(&result, sum128);
    
    // Handle remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}
