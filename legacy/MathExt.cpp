#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <immintrin.h>
#include <memory>
#include <algorithm>

namespace py = pybind11;

// Updated cache blocking for Intel i7-10750H
// L1D: 32 KB, L2: 256 KB, L3: 12 MB shared
constexpr int L1_BLOCK_SIZE = 32;
constexpr int L2_BLOCK_SIZE = 256;    // Increased from 128 to 256
constexpr int L3_BLOCK_SIZE = 768;    // Increased from 512 to 768
constexpr int VECTOR_SIZE = 4;
constexpr int NUM_THREADS = 12;
constexpr int LARGE_MATRIX_THRESHOLD = 2000;
constexpr int STRASSEN_THRESHOLD = 100000;  // too much overhead

// BLIS parameters for optimized implementation
constexpr int MR = 6;      // Micro-kernel rows
constexpr int NR = 8;      // Micro-kernel cols
constexpr int MC = 256;    // L2 cache blocking for M
constexpr int KC = 256;    // L2 cache blocking for K
constexpr int NC = 768;    // L3 cache blocking for N

void optimize_threads() {
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
}

class Matrix {
private:
    std::vector<double> data;
    int rows;
    int cols;

public:
    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    
    Matrix(const std::vector<std::vector<double>>& mat) {
        rows = mat.size();
        cols = mat[0].size();
        data.resize(rows * cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = mat[i][j];
            }
        }
    }
    
    double& at(int i, int j) {
        return data[i * cols + j];
    }
    
    double at(int i, int j) const {
        return data[i * cols + j];
    }
    
    std::vector<std::vector<double>> to_vector() const {
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i * cols + j];
            }
        }
        return result;
    }
    
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
    
    const double* get_row_ptr(int row) const {
        return &data[row * cols];
    }
    
    double* get_row_ptr(int row) {
        return &data[row * cols];
    }

    const double* get_ptr(int i, int j) const {
        return &data[i * cols + j];
    }
    
    double* get_ptr(int i, int j) {
        return &data[i * cols + j];
    }
};

// Memory allocation helpers
inline double* allocate_aligned(size_t n, size_t alignment = 32) {
#ifdef _WIN32
    return (double*)_aligned_malloc(n * sizeof(double), alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, n * sizeof(double)) != 0) {
        return nullptr;
    }
    return (double*)ptr;
#endif
}

inline void free_aligned(double* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Reduce memory traffic by accumulating in registers
Matrix simple_optimized_multiply(const Matrix& A, const Matrix& B) {
    const int M = A.get_rows();
    const int K = A.get_cols();
    const int N = B.get_cols();

    Matrix C(M, N);

    // Adaptive blocking based on matrix size
    int BLOCK_M, BLOCK_N, BLOCK_K;
    if (M <= 800 && N <= 800) {
        // Small-medium matrices: aggressive blocking
        BLOCK_M = 192;
        BLOCK_N = 384;
        BLOCK_K = 512;
    } else {
        // Large matrices: conservative blocking to reduce cache thrashing
        BLOCK_M = 128;
        BLOCK_N = 256;
        BLOCK_K = 256;
    }

    constexpr int REG_M = 4;       // Register blocking for M
    constexpr int REG_N = 12;      // Register blocking for N (3 vectors)

    #pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
            const int i_max = std::min(i0 + BLOCK_M, M);
            const int j_max = std::min(j0 + BLOCK_N, N);

            for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                const int k_max = std::min(k0 + BLOCK_K, K);

                // Process in register-blocked tiles (4 rows × 12 cols)
                for (int i = i0; i < i_max; i += REG_M) {
                    const int i_end = std::min(i + REG_M, i_max);
                    const int i_count = i_end - i;

                    for (int j = j0; j < j_max; j += REG_N) {
                        const int j_end = std::min(j + REG_N, j_max);

                        // Accumulate in registers (4 rows × 3 vectors = 12 registers)
                        __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd(), c02 = _mm256_setzero_pd();
                        __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd(), c12 = _mm256_setzero_pd();
                        __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd(), c22 = _mm256_setzero_pd();
                        __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd(), c32 = _mm256_setzero_pd();

                        // Accumulate over K dimension
                        for (int k = k0; k < k_max; ++k) {
                            // Load B values (3 vectors = 12 elements)
                            __m256d b0 = (j + 0 < j_end) ? _mm256_loadu_pd(B.get_ptr(k, j + 0)) : _mm256_setzero_pd();
                            __m256d b1 = (j + 4 < j_end) ? _mm256_loadu_pd(B.get_ptr(k, j + 4)) : _mm256_setzero_pd();
                            __m256d b2 = (j + 8 < j_end) ? _mm256_loadu_pd(B.get_ptr(k, j + 8)) : _mm256_setzero_pd();

                            // Process each row
                            if (i_count > 0) {
                                __m256d a0 = _mm256_broadcast_sd(A.get_ptr(i + 0, k));
                                c00 = _mm256_fmadd_pd(a0, b0, c00);
                                c01 = _mm256_fmadd_pd(a0, b1, c01);
                                c02 = _mm256_fmadd_pd(a0, b2, c02);
                            }

                            if (i_count > 1) {
                                __m256d a1 = _mm256_broadcast_sd(A.get_ptr(i + 1, k));
                                c10 = _mm256_fmadd_pd(a1, b0, c10);
                                c11 = _mm256_fmadd_pd(a1, b1, c11);
                                c12 = _mm256_fmadd_pd(a1, b2, c12);
                            }

                            if (i_count > 2) {
                                __m256d a2 = _mm256_broadcast_sd(A.get_ptr(i + 2, k));
                                c20 = _mm256_fmadd_pd(a2, b0, c20);
                                c21 = _mm256_fmadd_pd(a2, b1, c21);
                                c22 = _mm256_fmadd_pd(a2, b2, c22);
                            }

                            if (i_count > 3) {
                                __m256d a3 = _mm256_broadcast_sd(A.get_ptr(i + 3, k));
                                c30 = _mm256_fmadd_pd(a3, b0, c30);
                                c31 = _mm256_fmadd_pd(a3, b1, c31);
                                c32 = _mm256_fmadd_pd(a3, b2, c32);
                            }
                        }

                        // Write accumulated results back to C
                        if (i_count > 0 && j + 0 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 0, j + 0));
                            _mm256_storeu_pd(C.get_ptr(i + 0, j + 0), _mm256_add_pd(c_old, c00));
                        }
                        if (i_count > 0 && j + 4 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 0, j + 4));
                            _mm256_storeu_pd(C.get_ptr(i + 0, j + 4), _mm256_add_pd(c_old, c01));
                        }
                        if (i_count > 0 && j + 8 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 0, j + 8));
                            _mm256_storeu_pd(C.get_ptr(i + 0, j + 8), _mm256_add_pd(c_old, c02));
                        }

                        if (i_count > 1 && j + 0 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 1, j + 0));
                            _mm256_storeu_pd(C.get_ptr(i + 1, j + 0), _mm256_add_pd(c_old, c10));
                        }
                        if (i_count > 1 && j + 4 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 1, j + 4));
                            _mm256_storeu_pd(C.get_ptr(i + 1, j + 4), _mm256_add_pd(c_old, c11));
                        }
                        if (i_count > 1 && j + 8 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 1, j + 8));
                            _mm256_storeu_pd(C.get_ptr(i + 1, j + 8), _mm256_add_pd(c_old, c12));
                        }

                        if (i_count > 2 && j + 0 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 2, j + 0));
                            _mm256_storeu_pd(C.get_ptr(i + 2, j + 0), _mm256_add_pd(c_old, c20));
                        }
                        if (i_count > 2 && j + 4 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 2, j + 4));
                            _mm256_storeu_pd(C.get_ptr(i + 2, j + 4), _mm256_add_pd(c_old, c21));
                        }
                        if (i_count > 2 && j + 8 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 2, j + 8));
                            _mm256_storeu_pd(C.get_ptr(i + 2, j + 8), _mm256_add_pd(c_old, c22));
                        }

                        if (i_count > 3 && j + 0 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 3, j + 0));
                            _mm256_storeu_pd(C.get_ptr(i + 3, j + 0), _mm256_add_pd(c_old, c30));
                        }
                        if (i_count > 3 && j + 4 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 3, j + 4));
                            _mm256_storeu_pd(C.get_ptr(i + 3, j + 4), _mm256_add_pd(c_old, c31));
                        }
                        if (i_count > 3 && j + 8 < j_end) {
                            __m256d c_old = _mm256_loadu_pd(C.get_ptr(i + 3, j + 8));
                            _mm256_storeu_pd(C.get_ptr(i + 3, j + 8), _mm256_add_pd(c_old, c32));
                        }

                        // Handle edge cases for columns not covered by 12-element blocks
                        for (int ii = i; ii < i_end; ++ii) {
                            for (int jj = std::max(j + 12, j0); jj < j_end; ++jj) {
                                for (int kk = k0; kk < k_max; ++kk) {
                                    C.at(ii, jj) += A.at(ii, kk) * B.at(kk, jj);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return C;
}

// Transpose matrix B for cache-friendly access
Matrix transpose(const Matrix& B) {
    const int rows = B.get_rows();
    const int cols = B.get_cols();
    Matrix BT(cols, rows);

    // Blocked transpose for cache efficiency
    constexpr int TRANS_BLOCK = 32;
    for (int i = 0; i < rows; i += TRANS_BLOCK) {
        for (int j = 0; j < cols; j += TRANS_BLOCK) {
            const int i_max = std::min(i + TRANS_BLOCK, rows);
            const int j_max = std::min(j + TRANS_BLOCK, cols);

            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    BT.at(jj, ii) = B.at(ii, jj);
                }
            }
        }
    }
    return BT;
}

// Optimized 8×8 micro-kernel with AVX2+FMA
// Computes C[0:8, 0:8] += A[0:8, 0:k] * BT[0:8, 0:k]^T
inline void micro_kernel_8x8(const Matrix& A, const Matrix& BT, Matrix& C, int i_start, int j_start, int k_start, int k_end) {
    // Accumulate in registers (16 YMM registers: 8 rows × 2 vectors of 4 doubles)
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();
    __m256d c60 = _mm256_setzero_pd(), c61 = _mm256_setzero_pd();
    __m256d c70 = _mm256_setzero_pd(), c71 = _mm256_setzero_pd();

    // Main loop over K dimension
    for (int k = k_start; k < k_end; ++k) {
        // Load BT rows: BT[j,k] for j in [j_start, j_start+8)
        // We need to load them individually and pack into vectors
        __m256d b0 = _mm256_set_pd(
            BT.at(j_start + 3, k),
            BT.at(j_start + 2, k),
            BT.at(j_start + 1, k),
            BT.at(j_start + 0, k)
        );
        __m256d b1 = _mm256_set_pd(
            BT.at(j_start + 7, k),
            BT.at(j_start + 6, k),
            BT.at(j_start + 5, k),
            BT.at(j_start + 4, k)
        );

        // Process 8 rows of A with FMA
        __m256d a;

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 0, k));
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 1, k));
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 2, k));
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 3, k));
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 4, k));
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 5, k));
        c50 = _mm256_fmadd_pd(a, b0, c50);
        c51 = _mm256_fmadd_pd(a, b1, c51);

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 6, k));
        c60 = _mm256_fmadd_pd(a, b0, c60);
        c61 = _mm256_fmadd_pd(a, b1, c61);

        a = _mm256_broadcast_sd(A.get_ptr(i_start + 7, k));
        c70 = _mm256_fmadd_pd(a, b0, c70);
        c71 = _mm256_fmadd_pd(a, b1, c71);
    }

    // Store accumulated results back to C
    __m256d c_old;

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 0, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 0, j_start + 0), _mm256_add_pd(c_old, c00));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 0, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 0, j_start + 4), _mm256_add_pd(c_old, c01));

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 1, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 1, j_start + 0), _mm256_add_pd(c_old, c10));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 1, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 1, j_start + 4), _mm256_add_pd(c_old, c11));

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 2, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 2, j_start + 0), _mm256_add_pd(c_old, c20));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 2, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 2, j_start + 4), _mm256_add_pd(c_old, c21));

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 3, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 3, j_start + 0), _mm256_add_pd(c_old, c30));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 3, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 3, j_start + 4), _mm256_add_pd(c_old, c31));

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 4, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 4, j_start + 0), _mm256_add_pd(c_old, c40));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 4, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 4, j_start + 4), _mm256_add_pd(c_old, c41));

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 5, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 5, j_start + 0), _mm256_add_pd(c_old, c50));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 5, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 5, j_start + 4), _mm256_add_pd(c_old, c51));

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 6, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 6, j_start + 0), _mm256_add_pd(c_old, c60));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 6, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 6, j_start + 4), _mm256_add_pd(c_old, c61));

    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 7, j_start + 0));
    _mm256_storeu_pd(C.get_ptr(i_start + 7, j_start + 0), _mm256_add_pd(c_old, c70));
    c_old = _mm256_loadu_pd(C.get_ptr(i_start + 7, j_start + 4));
    _mm256_storeu_pd(C.get_ptr(i_start + 7, j_start + 4), _mm256_add_pd(c_old, c71));
}

Matrix ultra_optimized_multiply(const Matrix& A, const Matrix& B) {
    const int M = A.get_rows();
    const int K = A.get_cols();
    const int N = B.get_cols();

    Matrix C(M, N);

    // Transpose B for better cache access
    Matrix BT = transpose(B);

    // Micro-kernel size
    constexpr int MICRO_M = 8;
    constexpr int MICRO_N = 8;

    // Block size for cache optimization
    constexpr int BLOCK_SIZE = 256;

    // Blocked algorithm with 8×8 micro-kernel
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        const int i_max = std::min(i0 + BLOCK_SIZE, M);

        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            const int j_max = std::min(j0 + BLOCK_SIZE, N);

            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                const int k_max = std::min(k0 + BLOCK_SIZE, K);

                // Process in 8×8 micro-tiles
                for (int i = i0; i < i_max; i += MICRO_M) {
                    for (int j = j0; j < j_max; j += MICRO_N) {
                        // Check if we can use full 8×8 kernel
                        if (i + MICRO_M <= i_max && j + MICRO_N <= j_max) {
                            micro_kernel_8x8(A, BT, C, i, j, k0, k_max);
                        } else {
                            // Edge case: scalar fallback
                            const int i_end = std::min(i + MICRO_M, i_max);
                            const int j_end = std::min(j + MICRO_N, j_max);

                            for (int ii = i; ii < i_end; ++ii) {
                                for (int kk = k0; kk < k_max; ++kk) {
                                    const double a_val = A.at(ii, kk);

                                    // Scalar code for edge cases (BT access pattern not vectorizable)
                                    for (int jj = j; jj < j_end; ++jj) {
                                        C.at(ii, jj) += a_val * BT.at(jj, kk);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return C;
}

// ====== BLIS-STYLE IMPLEMENTATION (EXPERIMENTAL - HAS BUGS) ======
// Note: BLIS constants (MR, NR, MC, KC, NC) are defined at the top of the file
inline void pack_A_panel(const Matrix& A, double* packed_A, int i_start, int k_start, int m_r, int k_c) {
    const int M = A.get_rows();
    const int K = A.get_cols();
    const int m_actual = std::min(m_r, M - i_start);
    const int k_actual = std::min(k_c, K - k_start);

    for (int k = 0; k < k_actual; ++k) {
        for (int i = 0; i < m_actual; ++i) {
            packed_A[k * m_r + i] = A.at(i_start + i, k_start + k);
        }
        for (int i = m_actual; i < m_r; ++i) {
            packed_A[k * m_r + i] = 0.0;
        }
    }

    for (int k = k_actual; k < k_c; ++k) {
        for (int i = 0; i < m_r; ++i) {
            packed_A[k * m_r + i] = 0.0;
        }
    }
}

// Pack panel of B
inline void pack_B_panel(const Matrix& B, double* packed_B, int k_start, int j_start, int k_c, int n_r) {
    const int K = B.get_rows();
    const int N = B.get_cols();
    const int k_actual = std::min(k_c, K - k_start);
    const int n_actual = std::min(n_r, N - j_start);

    for (int k = 0; k < k_actual; ++k) {
        for (int j = 0; j < n_actual; ++j) {
            packed_B[k * n_r + j] = B.at(k_start + k, j_start + j);
        }
        for (int j = n_actual; j < n_r; ++j) {
            packed_B[k * n_r + j] = 0.0;
        }
    }

    for (int k = k_actual; k < k_c; ++k) {
        for (int j = 0; j < n_r; ++j) {
            packed_B[k * n_r + j] = 0.0;
        }
    }
}

// Simplified micro-kernel
inline void micro_kernel_simple(const double* A_packed, const double* B_packed, double* C, int k, int ldc, int m_r, int n_r) {
    // Simple but efficient implementation
    for (int i = 0; i < m_r; ++i) {
        for (int p = 0; p < k; ++p) {
            double a_val = A_packed[p * MR + i];
            __m256d a_vec = _mm256_set1_pd(a_val);

            int j = 0;
            for (; j + 4 <= n_r; j += 4) {
                __m256d b_vec = _mm256_loadu_pd(B_packed + p * NR + j);
                __m256d c_vec = _mm256_loadu_pd(C + i * ldc + j);
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                _mm256_storeu_pd(C + i * ldc + j, c_vec);
            }

            for (; j < n_r; ++j) {
                C[i * ldc + j] += a_val * B_packed[p * NR + j];
            }
        }
    }
}

// BLIS 5-loop structure
Matrix blis_multiply(const Matrix& A, const Matrix& B) {
    const int M = A.get_rows();
    const int K = A.get_cols();
    const int N = B.get_cols();

    Matrix C(M, N);

    double* packed_B = allocate_aligned(KC * NC);
    if (!packed_B) {
        throw std::runtime_error("Failed to allocate packing buffer for B");
    }

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);

        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);

            for (int jr = 0; jr < nc; jr += NR) {
                const int nr = std::min(NR, nc - jr);
                pack_B_panel(B, packed_B + jr * kc, pc, jc + jr, kc, nr);
            }

            #pragma omp parallel
            {
                double* packed_A = allocate_aligned(MC * KC);
                if (!packed_A) {
                    #pragma omp critical
                    {
                        throw std::runtime_error("Failed to allocate thread-local packing buffer for A");
                    }
                }

                #pragma omp for schedule(dynamic, 1) nowait
                for (int ic = 0; ic < M; ic += MC) {
                    const int mc = std::min(MC, M - ic);

                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = std::min(MR, mc - ir);
                        pack_A_panel(A, packed_A + ir * kc, ic + ir, pc, mr, kc);
                    }

                    for (int jr = 0; jr < nc; jr += NR) {
                        const int nr = std::min(NR, nc - jr);

                        for (int ir = 0; ir < mc; ir += MR) {
                            const int mr = std::min(MR, mc - ir);

                            micro_kernel_simple(
                                packed_A + ir * kc,
                                packed_B + jr * kc,
                                C.get_ptr(ic + ir, jc + jr),
                                kc,
                                N,
                                mr,
                                nr
                            );
                        }
                    }
                }

                free_aligned(packed_A);
            }
        }
    }

    free_aligned(packed_B);
    return C;
}

std::string factorial(long long n) {
    if (n < 0) {
        throw std::invalid_argument("Input must be non-negative");
    }

    if (n <= 1) {
        return "1";
    }

    py::object result = py::cast(1);
    
    for (long long i = 2; i <= n; i++) {
        result = result * py::cast(i);
    }

    return py::str(result);
}

void validate_matrices(const std::vector<std::vector<double>>& A, 
                      const std::vector<std::vector<double>>& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        throw std::invalid_argument("Matrices cannot be empty");
    }
    if (A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }
}

bool should_use_strassen(int M, int N, int K) {
    return (M >= STRASSEN_THRESHOLD && N >= STRASSEN_THRESHOLD && K >= STRASSEN_THRESHOLD && 
            M == N && N == K);
}

Matrix strassen_multiply(const Matrix& A, const Matrix& B, int threshold = 128) {
    int n = A.get_rows();
    
    if (n <= threshold) {
        Matrix C(n, n);
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                __m256d a_val = _mm256_set1_pd(A.at(i, k));
                for (int j = 0; j < n; j += VECTOR_SIZE) {
                    if (j + VECTOR_SIZE <= n) {
                        __m256d b_vec = _mm256_loadu_pd(B.get_ptr(k, j));
                        __m256d c_vec = _mm256_loadu_pd(C.get_ptr(i, j));
                        c_vec = _mm256_fmadd_pd(a_val, b_vec, c_vec);
                        _mm256_storeu_pd(C.get_ptr(i, j), c_vec);
                    }
                    else {
                        for (int jr = j; jr < n; jr++) {
                            C.at(i, jr) += A.at(i, k) * B.at(k, jr);
                        }
                    }
                }
            }
        }
        return C;
    }
    
    int m = n / 2;
    
    Matrix A11(m, m), A12(m, m), A21(m, m), A22(m, m);
    Matrix B11(m, m), B12(m, m), B21(m, m), B22(m, m);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A11.at(i, j) = A.at(i, j);
                    B11.at(i, j) = B.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A12.at(i, j) = A.at(i, j + m);
                    B12.at(i, j) = B.at(i, j + m);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A21.at(i, j) = A.at(i + m, j);
                    B21.at(i, j) = B.at(i + m, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    A22.at(i, j) = A.at(i + m, j + m);
                    B22.at(i, j) = B.at(i + m, j + m);
                }
            }
        }
    }
    
    Matrix P1(m, m), P2(m, m), P3(m, m), P4(m, m), P5(m, m), P6(m, m), P7(m, m);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A11.at(i, j) + A22.at(i, j);
                }
            }
            
            Matrix temp2(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp2.at(i, j) = B11.at(i, j) + B22.at(i, j);
                }
            }
            
            P1 = strassen_multiply(temp1, temp2, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A21.at(i, j) + A22.at(i, j);
                }
            }
            
            P2 = strassen_multiply(temp1, B11, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = B12.at(i, j) - B22.at(i, j);
                }
            }
            
            P3 = strassen_multiply(A11, temp1, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = B21.at(i, j) - B11.at(i, j);
                }
            }
            
            P4 = strassen_multiply(A22, temp1, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A11.at(i, j) + A12.at(i, j);
                }
            }
            
            P5 = strassen_multiply(temp1, B22, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A21.at(i, j) - A11.at(i, j);
                }
            }
            
            Matrix temp2(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp2.at(i, j) = B11.at(i, j) + B12.at(i, j);
                }
            }
            
            P6 = strassen_multiply(temp1, temp2, threshold);
        }
        
        #pragma omp section
        {
            Matrix temp1(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp1.at(i, j) = A12.at(i, j) - A22.at(i, j);
                }
            }
            
            Matrix temp2(m, m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    temp2.at(i, j) = B21.at(i, j) + B22.at(i, j);
                }
            }
            
            P7 = strassen_multiply(temp1, temp2, threshold);
        }
    }
    
    Matrix C(n, n);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i, j) = P1.at(i, j) + P4.at(i, j) - P5.at(i, j) + P7.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i, j + m) = P3.at(i, j) + P5.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i + m, j) = P2.at(i, j) + P4.at(i, j);
                }
            }
        }
        
        #pragma omp section
        {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    C.at(i + m, j + m) = P1.at(i, j) - P2.at(i, j) + P3.at(i, j) + P6.at(i, j);
                }
            }
        }
    }
    
    return C;
}

// REMOVED: Manual prefetching - hardware prefetchers on Comet Lake are superior

Matrix standard_multiply(const Matrix& A, const Matrix& B) {
    int M = A.get_rows();
    int K = A.get_cols();
    int N = B.get_cols();
    
    Matrix C(M, N);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for (int i2 = 0; i2 < M; i2 += L2_BLOCK_SIZE) {
            for (int j2 = 0; j2 < N; j2 += L2_BLOCK_SIZE) {
                for (int k2 = 0; k2 < K; k2 += L2_BLOCK_SIZE) {
                    // REMOVED: Manual prefetching - hardware prefetchers handle this better

                    for (int i1 = i2; i1 < std::min(i2 + L2_BLOCK_SIZE, M); i1 += L1_BLOCK_SIZE) {
                        for (int j1 = j2; j1 < std::min(j2 + L2_BLOCK_SIZE, N); j1 += L1_BLOCK_SIZE) {
                            alignas(32) double local_sum[L1_BLOCK_SIZE][L1_BLOCK_SIZE] = {{0.0}};
                            
                            for (int k1 = k2; k1 < std::min(k2 + L2_BLOCK_SIZE, K); k1 += L1_BLOCK_SIZE) {
                                for (int i = i1; i < std::min(i1 + L1_BLOCK_SIZE, M); ++i) {
                                    for (int k = k1; k < std::min(k1 + L1_BLOCK_SIZE, K); ++k) {
                                        double a_val = A.at(i, k);
                                        __m256d a_vec = _mm256_set1_pd(a_val);
                                        
                                        int j = j1;
                                        for (; j + VECTOR_SIZE <= std::min(j1 + L1_BLOCK_SIZE, N); j += VECTOR_SIZE) {
                                            __m256d b_vec = _mm256_loadu_pd(B.get_ptr(k, j));
                                            __m256d sum_vec = _mm256_loadu_pd(&local_sum[i - i1][j - j1]);
                                            sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
                                            _mm256_storeu_pd(&local_sum[i - i1][j - j1], sum_vec);
                                        }
                                        
                                        for (; j < std::min(j1 + L1_BLOCK_SIZE, N); ++j) {
                                            local_sum[i - i1][j - j1] += a_val * B.at(k, j);
                                        }
                                    }
                                }
                            }
                            
                            for (int i = i1; i < std::min(i1 + L1_BLOCK_SIZE, M); ++i) {
                                for (int j = j1; j < std::min(j1 + L1_BLOCK_SIZE, N); ++j) {
                                    C.at(i, j) += local_sum[i - i1][j - j1];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

Matrix tiled_multiply_large(const Matrix& A, const Matrix& B) {
    int M = A.get_rows();
    int K = A.get_cols();
    int N = B.get_cols();
    
    Matrix C(M, N);

    constexpr int TILE_SIZE = 512;
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < M; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            int imax = std::min(i0 + TILE_SIZE, M);
            int jmax = std::min(j0 + TILE_SIZE, N);
            
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                int kmax = std::min(k0 + TILE_SIZE, K);
                
                for (int i = i0; i < imax; i++) {
                    for (int k = k0; k < kmax; k++) {
                        double a_val = A.at(i, k);
                        __m256d a_vec = _mm256_set1_pd(a_val);
                        
                        int j = j0;
                        for (; j + VECTOR_SIZE <= jmax; j += VECTOR_SIZE) {
                            __m256d b_vec = _mm256_loadu_pd(B.get_ptr(k, j));
                            __m256d c_vec = _mm256_loadu_pd(C.get_ptr(i, j));
                            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                            _mm256_storeu_pd(C.get_ptr(i, j), c_vec);
                        }
                        
                        for (; j < jmax; j++) {
                            C.at(i, j) += a_val * B.at(k, j);
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

Matrix optimized_matrix_multiply(const Matrix& A, const Matrix& B) {
    int M = A.get_rows();
    int K = A.get_cols();
    int N = B.get_cols();

    // Strassen is disabled (threshold = 100000) - too much overhead
    if (should_use_strassen(M, N, K)) {
        return strassen_multiply(A, B);
    }
    
    // Reverting to simple_optimized_multiply while we try other approaches
    return simple_optimized_multiply(A, B);
}

std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    
    validate_matrices(A, B);
    
    try {
        size_t rows_A = A.size();
        size_t cols_A = A[0].size();
        size_t rows_B = B.size();
        size_t cols_B = B[0].size();
        
        Matrix A_flat(A);
        Matrix B_flat(B);
        Matrix C_flat;
        
        C_flat = optimized_matrix_multiply(A_flat, B_flat);
        
        return C_flat.to_vector();
    }
    catch (const std::bad_alloc& e) {
        throw std::runtime_error("Memory allocation failed. Matrix may be too large for available memory.");
    }
}

PYBIND11_MODULE(MathExt, m) {
    optimize_threads();
    
    m.doc() = "Extension module for efficient matrix multiplication";
    m.def("matrix_multiply", &matrix_multiply,
          "Multiply two matrices using optimized SIMD and blocking",
          py::arg("A"), py::arg("B"),
          py::return_value_policy::move);
    
    m.doc() = "Extension module for exact factorial calculations using arbitrary precision";
    m.def("factorial", &factorial, 
          "Calculate exact factorial of n\n"
          "Returns the precise result as a string to handle arbitrary-precision numbers",
          py::arg("n"));
}