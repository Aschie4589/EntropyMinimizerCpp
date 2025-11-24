#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include <cmath>
#include <random>

#include "compute/core/ComputeTypes.h"
#include "compute/linalg/ILinearAlgebra.h"
#include "compute/linalg/ISolver.h"
#include "compute/backends/cuda/CudaLinearAlgebra.h"
#include "compute/backends/cuda/CudaSolver.h"
#include "compute/backends/cpu/CpuLinearAlgebra.h"
#include "compute/backends/cpu/CpuSolver.h"
#include "compute/backends/cuda/CudaMemory.h"
#include "compute/backends/cpu/CpuMemory.h"

using Complex = std::complex<double>;

// ============================================================================
// Test Fixtures
// ============================================================================

class LinearAlgebraTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random seed
        gen.seed(42);
    }
    
    std::mt19937_64 gen;
    
    // Helper: Create random complex vector
    std::vector<Complex> randomVector(int n) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<Complex> v(n);
        for (int i = 0; i < n; ++i) {
            v[i] = Complex(dist(gen), dist(gen));
        }
        return v;
    }
    
    // Helper: Create random complex matrix (column-major)
    std::vector<Complex> randomMatrix(int m, int n) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<Complex> A(m * n);
        for (int i = 0; i < m * n; ++i) {
            A[i] = Complex(dist(gen), dist(gen));
        }
        return A;
    }
    
    // Helper: Check if two vectors are approximately equal
    bool approxEqual(const std::vector<Complex>& a, const std::vector<Complex>& b,
                     double tol = 1e-10) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tol) {
                return false;
            }
        }
        return true;
    }
    
    bool approxEqual(double a, double b, double tol = 1e-10) {
        return std::abs(a - b) < tol;
    }
};

// ============================================================================
// BLAS Level 1 Tests
// ============================================================================

TEST_F(LinearAlgebraTest, CUDA_AXPY) {
    CudaLinearAlgebra linalg;
    int n = 100;
    Complex alpha(2.0, 1.0);
    
    auto x_host = randomVector(n);
    auto y_host = randomVector(n);
    auto y_expected = y_host;
    
    // Compute expected result on CPU
    for (int i = 0; i < n; ++i) {
        y_expected[i] += alpha * x_host[i];
    }
    
    // Transfer to GPU
    CudaMemory x_dev(n * sizeof(Complex));
    CudaMemory y_dev(n * sizeof(Complex));
    x_dev.copyFromHost(x_host.data(), n * sizeof(Complex));
    y_dev.copyFromHost(y_host.data(), n * sizeof(Complex));
    
    // Perform AXPY on GPU
    linalg.axpy(n, &alpha, x_dev.data(), y_dev.data(), PrecisionType::DOUBLE);
    
    // Transfer result back
    std::vector<Complex> y_result(n);
    y_dev.copyToHost(y_result.data(), n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(y_result, y_expected, 1e-12));
}

TEST_F(LinearAlgebraTest, CPU_AXPY) {
    CpuLinearAlgebra linalg;
    int n = 100;
    Complex alpha(2.0, 1.0);
    
    auto x = randomVector(n);
    auto y = randomVector(n);
    auto y_expected = y;
    
    for (int i = 0; i < n; ++i) {
        y_expected[i] += alpha * x[i];
    }
    
    linalg.axpy(n, &alpha, x.data(), y.data(), PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(y, y_expected, 1e-12));
}

TEST_F(LinearAlgebraTest, CPU_DOTC) {
    CpuLinearAlgebra linalg;
    int n = 100;
    
    auto x = randomVector(n);
    auto y = randomVector(n);
    
    Complex expected(0.0, 0.0);
    for (int i = 0; i < n; ++i) {
        expected += std::conj(x[i]) * y[i];
    }
    
    Complex result;
    linalg.dotc(n, x.data(), y.data(), &result, PrecisionType::DOUBLE);
    
    EXPECT_TRUE(std::abs(result - expected) < 1e-10);
}

TEST_F(LinearAlgebraTest, CPU_NORM2) {
    CpuLinearAlgebra linalg;
    int n = 100;
    
    auto x = randomVector(n);
    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        expected += std::norm(x[i]);
    }
    expected = std::sqrt(expected);
    
    double result;
    linalg.norm2(n, x.data(), &result, PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_SCAL) {
    CpuLinearAlgebra linalg;
    int n = 100;
    Complex alpha(2.0, -1.5);
    
    auto x = randomVector(n);
    auto expected = x;
    for (int i = 0; i < n; ++i) {
        expected[i] *= alpha;
    }
    
    linalg.scal(n, &alpha, x.data(), PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(x, expected, 1e-12));
}

TEST_F(LinearAlgebraTest, CPU_GEMV_NoTrans) {
    CpuLinearAlgebra linalg;
    int m = 50, n = 40;
    Complex alpha(1.5, 0.5), beta(0.5, -0.5);
    
    auto A = randomMatrix(m, n);
    auto x = randomVector(n);
    auto y = randomVector(m);
    
    std::vector<Complex> expected(m, Complex(0,0));
    for (int i = 0; i < m; ++i) {
        Complex sum(0, 0);
        for (int j = 0; j < n; ++j) {
            sum += A[j*m + i] * x[j];
        }
        expected[i] = alpha * sum + beta * y[i];
    }
    
    linalg.gemv(Transpose::NO_TRANS, m, n, &alpha, A.data(),
                x.data(), &beta, y.data(), PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(y, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_GEMV_ConjTrans) {
    CpuLinearAlgebra linalg;
    int m_storage = 40, n_storage = 50;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A = randomMatrix(m_storage, n_storage);
    auto x = randomVector(m_storage);
    std::vector<Complex> y(n_storage, Complex(0,0));
    
    std::vector<Complex> expected(n_storage, Complex(0,0));
    for (int i = 0; i < n_storage; ++i) {
        Complex sum(0, 0);
        for (int j = 0; j < m_storage; ++j) {
            sum += std::conj(A[i*m_storage + j]) * x[j];
        }
        expected[i] = sum;
    }
    
    linalg.gemv(Transpose::CONJ_TRANS, m_storage, n_storage, &alpha, A.data(),
                x.data(), &beta, y.data(), PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(y, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_DOTC) {
    CudaLinearAlgebra linalg;
    int n = 100;
    
    auto x_host = randomVector(n);
    auto y_host = randomVector(n);
    
    Complex expected(0.0, 0.0);
    for (int i = 0; i < n; ++i) {
        expected += std::conj(x_host[i]) * y_host[i];
    }
    
    CudaMemory x_dev(n * sizeof(Complex));
    CudaMemory y_dev(n * sizeof(Complex));
    CudaMemory result_dev(sizeof(Complex));
    
    x_dev.copyFromHost(x_host.data(), n * sizeof(Complex));
    y_dev.copyFromHost(y_host.data(), n * sizeof(Complex));
    
    linalg.dotc(n, x_dev.data(), y_dev.data(), result_dev.data(), PrecisionType::DOUBLE);
    
    Complex result;
    result_dev.copyToHost(&result, sizeof(Complex));
    
    EXPECT_TRUE(std::abs(result - expected) < 1e-10);
}

TEST_F(LinearAlgebraTest, CUDA_NORM2) {
    CudaLinearAlgebra linalg;
    int n = 100;
    
    auto x_host = randomVector(n);
    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        expected += std::norm(x_host[i]);
    }
    expected = std::sqrt(expected);
    
    CudaMemory x_dev(n * sizeof(Complex));
    x_dev.copyFromHost(x_host.data(), n * sizeof(Complex));
    
    double result;
    linalg.norm2(n, x_dev.data(), &result, PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_SCAL) {
    CudaLinearAlgebra linalg;
    int n = 100;
    Complex alpha(2.0, -1.5);
    
    auto x_host = randomVector(n);
    auto expected = x_host;
    for (int i = 0; i < n; ++i) {
        expected[i] *= alpha;
    }
    
    CudaMemory x_dev(n * sizeof(Complex));
    x_dev.copyFromHost(x_host.data(), n * sizeof(Complex));
    
    linalg.scal(n, &alpha, x_dev.data(), PrecisionType::DOUBLE);
    
    std::vector<Complex> result(n);
    x_dev.copyToHost(result.data(), n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-12));
}

// ============================================================================
// BLAS Level 2 Tests
// ============================================================================

TEST_F(LinearAlgebraTest, CUDA_GEMV_NoTrans) {
    CudaLinearAlgebra linalg;
    int m = 50, n = 40;
    Complex alpha(1.5, 0.5), beta(0.5, -0.5);
    
    auto A_host = randomMatrix(m, n);
    auto x_host = randomVector(n);
    auto y_host = randomVector(m);
    
    // Expected: y = alpha*A*x + beta*y
    std::vector<Complex> expected(m, Complex(0,0));
    for (int i = 0; i < m; ++i) {
        Complex sum(0, 0);
        for (int j = 0; j < n; ++j) {
            sum += A_host[j*m + i] * x_host[j];
        }
        expected[i] = alpha * sum + beta * y_host[i];
    }
    
    CudaMemory A_dev(m * n * sizeof(Complex));
    CudaMemory x_dev(n * sizeof(Complex));
    CudaMemory y_dev(m * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * n * sizeof(Complex));
    x_dev.copyFromHost(x_host.data(), n * sizeof(Complex));
    y_dev.copyFromHost(y_host.data(), m * sizeof(Complex));
    
    linalg.gemv(Transpose::NO_TRANS, m, n, &alpha, A_dev.data(),
                x_dev.data(), &beta, y_dev.data(), PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m);
    y_dev.copyToHost(result.data(), m * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_GEMV_ConjTrans) {
    CudaLinearAlgebra linalg;
    int m_storage = 40, n_storage = 50;  // A stored as 40×50
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A_host = randomMatrix(m_storage, n_storage);  // 40×50 in column-major
    auto x_host = randomVector(m_storage);  // x has m elements for conjugate transpose
    std::vector<Complex> y_host(n_storage, Complex(0,0));  // y has n elements
    
    // Expected: y = A^H * x (where A is 40×50 in column-major storage)
    // A^H is 50×40, so y(50) = A^H(50×40) * x(40)
    std::vector<Complex> expected(n_storage, Complex(0,0));
    for (int i = 0; i < n_storage; ++i) {
        Complex sum(0, 0);
        for (int j = 0; j < m_storage; ++j) {
            // A is column-major: A[col*m + row] gives element at (row, col)
            // A^H[i,j] = conj(A[j,i]) = conj(A[i*m + j])
            sum += std::conj(A_host[i*m_storage + j]) * x_host[j];
        }
        expected[i] = sum;
    }
    
    CudaMemory A_dev(m_storage * n_storage * sizeof(Complex));
    CudaMemory x_dev(m_storage * sizeof(Complex));
    CudaMemory y_dev(n_storage * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m_storage * n_storage * sizeof(Complex));
    x_dev.copyFromHost(x_host.data(), m_storage * sizeof(Complex));
    y_dev.copyFromHost(y_host.data(), n_storage * sizeof(Complex));
    
    linalg.gemv(Transpose::CONJ_TRANS, m_storage, n_storage, &alpha, A_dev.data(),
                x_dev.data(), &beta, y_dev.data(), PrecisionType::DOUBLE);
    
    std::vector<Complex> result(n_storage);
    y_dev.copyToHost(result.data(), n_storage * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

// ============================================================================
// BLAS Level 3 Tests
// ============================================================================

TEST_F(LinearAlgebraTest, CUDA_GEMM_NoTrans) {
    CudaLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A_host = randomMatrix(m, k);
    auto B_host = randomMatrix(k, n);
    std::vector<Complex> C_host(m * n, Complex(0,0));
    
    // Expected: C = A * B
    std::vector<Complex> expected(m * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += A_host[l*m + i] * B_host[j*k + l];
            }
            expected[j*m + i] = sum;
        }
    }
    
    CudaMemory A_dev(m * k * sizeof(Complex));
    CudaMemory B_dev(k * n * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * k * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), k * n * sizeof(Complex));
    C_dev.copyFromHost(C_host.data(), m * n * sizeof(Complex));
    
    linalg.gemm(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n, k,
                &alpha, A_dev.data(), B_dev.data(), &beta, C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CUDA_GEMM_TransA) {
    CudaLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A_host = randomMatrix(k, m);  // Stored as k×m for transpose
    auto B_host = randomMatrix(k, n);
    std::vector<Complex> C_host(m * n, Complex(0,0));
    
    // Expected: C = A^H * B
    std::vector<Complex> expected(m * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += std::conj(A_host[i*k + l]) * B_host[j*k + l];
            }
            expected[j*m + i] = sum;
        }
    }
    
    CudaMemory A_dev(k * m * sizeof(Complex));
    CudaMemory B_dev(k * n * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), k * m * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), k * n * sizeof(Complex));
    C_dev.copyFromHost(C_host.data(), m * n * sizeof(Complex));
    
    linalg.gemm(Transpose::CONJ_TRANS, Transpose::NO_TRANS, m, n, k,
                &alpha, A_dev.data(), B_dev.data(), &beta, C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CUDA_GEMM_TransB) {
    CudaLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A_host = randomMatrix(m, k);
    auto B_host = randomMatrix(n, k);  // Stored as n×k for transpose
    std::vector<Complex> C_host(m * n, Complex(0,0));
    
    // Expected: C = A * B^H
    std::vector<Complex> expected(m * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += A_host[l*m + i] * std::conj(B_host[l*n + j]);
            }
            expected[j*m + i] = sum;
        }
    }
    
    CudaMemory A_dev(m * k * sizeof(Complex));
    CudaMemory B_dev(n * k * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * k * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), n * k * sizeof(Complex));
    C_dev.copyFromHost(C_host.data(), m * n * sizeof(Complex));
    
    linalg.gemm(Transpose::NO_TRANS, Transpose::CONJ_TRANS, m, n, k,
                &alpha, A_dev.data(), B_dev.data(), &beta, C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CUDA_GEMM_BothTrans) {
    CudaLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.5, 0.5), beta(0.5, -0.5);
    
    auto A_host = randomMatrix(k, m);  // Stored as k×m
    auto B_host = randomMatrix(n, k);  // Stored as n×k
    auto C_host = randomMatrix(m, n);  // Existing values for beta*C
    auto C_original = C_host;
    
    // Expected: C = alpha * A^H * B^H + beta * C
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += std::conj(A_host[i*k + l]) * std::conj(B_host[l*n + j]);
            }
            expected[j*m + i] = alpha * sum + beta * C_original[j*m + i];
        }
    }
    
    CudaMemory A_dev(k * m * sizeof(Complex));
    CudaMemory B_dev(n * k * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), k * m * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), n * k * sizeof(Complex));
    C_dev.copyFromHost(C_host.data(), m * n * sizeof(Complex));
    
    linalg.gemm(Transpose::CONJ_TRANS, Transpose::CONJ_TRANS, m, n, k,
                &alpha, A_dev.data(), B_dev.data(), &beta, C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CUDA_GEAM_NoTrans) {
    CudaLinearAlgebra linalg;
    int m = 30, n = 25;
    Complex alpha(2.0, 0.5), beta(1.5, -0.5);
    
    auto A_host = randomMatrix(m, n);
    auto B_host = randomMatrix(m, n);
    
    // Expected: C = alpha*A + beta*B
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * A_host[j*m + i] + beta * B_host[j*m + i];
        }
    }
    
    CudaMemory A_dev(m * n * sizeof(Complex));
    CudaMemory B_dev(m * n * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * n * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), m * n * sizeof(Complex));
    
    linalg.geam(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A_dev.data(), &beta, B_dev.data(), C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_GEAM_TransA) {
    CudaLinearAlgebra linalg;
    int m = 30, n = 25;
    Complex alpha(1.0, 0.0), beta(1.0, 0.0);
    
    auto A_host = randomMatrix(n, m);  // A stored as n×m for transpose
    auto B_host = randomMatrix(m, n);
    
    // Expected: C = A^H + B
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = std::conj(A_host[i*n + j]) + B_host[j*m + i];
        }
    }
    
    CudaMemory A_dev(n * m * sizeof(Complex));
    CudaMemory B_dev(m * n * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), n * m * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), m * n * sizeof(Complex));
    
    linalg.geam(Transpose::CONJ_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A_dev.data(), &beta, B_dev.data(), C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_GEMV_Trans) {
    CudaLinearAlgebra linalg;
    int m_storage = 40, n_storage = 50;
    Complex alpha(1.0, 0.5), beta(0.5, 0.0);
    
    auto A_host = randomMatrix(m_storage, n_storage);
    auto x_host = randomVector(m_storage);
    auto y_host = randomVector(n_storage);
    
    std::vector<Complex> expected(n_storage);
    for (int i = 0; i < n_storage; ++i) {
        Complex sum(0, 0);
        for (int j = 0; j < m_storage; ++j) {
            // A^T[i,j] = A[j,i] (no conjugate for TRANS)
            sum += A_host[i*m_storage + j] * x_host[j];
        }
        expected[i] = alpha * sum + beta * y_host[i];
    }
    
    CudaMemory A_dev(m_storage * n_storage * sizeof(Complex));
    CudaMemory x_dev(m_storage * sizeof(Complex));
    CudaMemory y_dev(n_storage * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m_storage * n_storage * sizeof(Complex));
    x_dev.copyFromHost(x_host.data(), m_storage * sizeof(Complex));
    y_dev.copyFromHost(y_host.data(), n_storage * sizeof(Complex));
    
    linalg.gemv(Transpose::TRANS, m_storage, n_storage, &alpha, A_dev.data(),
                x_dev.data(), &beta, y_dev.data(), PrecisionType::DOUBLE);
    
    std::vector<Complex> result(n_storage);
    y_dev.copyToHost(result.data(), n_storage * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_GEMM_NoTrans) {
    CpuLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A = randomMatrix(m, k);
    auto B = randomMatrix(k, n);
    std::vector<Complex> C(m * n, Complex(0,0));
    
    std::vector<Complex> expected(m * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += A[l*m + i] * B[j*k + l];
            }
            expected[j*m + i] = sum;
        }
    }
    
    linalg.gemm(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n, k,
                &alpha, A.data(), B.data(), &beta, C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CPU_GEMM_TransA) {
    CpuLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A = randomMatrix(k, m);  // A stored as k×m for transpose
    auto B = randomMatrix(k, n);
    std::vector<Complex> C(m * n, Complex(0,0));
    
    // Expected: C = A^H * B
    std::vector<Complex> expected(m * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += std::conj(A[i*k + l]) * B[j*k + l];
            }
            expected[j*m + i] = sum;
        }
    }
    
    linalg.gemm(Transpose::CONJ_TRANS, Transpose::NO_TRANS, m, n, k,
                &alpha, A.data(), B.data(), &beta, C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CPU_GEMM_TransB) {
    CpuLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A = randomMatrix(m, k);
    auto B = randomMatrix(n, k);  // B stored as n×k for transpose
    std::vector<Complex> C(m * n, Complex(0,0));
    
    // Expected: C = A * B^H
    std::vector<Complex> expected(m * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += A[l*m + i] * std::conj(B[l*n + j]);
            }
            expected[j*m + i] = sum;
        }
    }
    
    linalg.gemm(Transpose::NO_TRANS, Transpose::CONJ_TRANS, m, n, k,
                &alpha, A.data(), B.data(), &beta, C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CPU_GEMM_BothTrans) {
    CpuLinearAlgebra linalg;
    int m = 30, n = 25, k = 20;
    Complex alpha(1.5, 0.5), beta(0.5, -0.5);
    
    auto A = randomMatrix(k, m);  // A stored as k×m
    auto B = randomMatrix(n, k);  // B stored as n×k
    auto C = randomMatrix(m, n);  // Existing values for beta*C
    auto C_original = C;
    
    // Expected: C = alpha * A^H * B^H + beta * C
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Complex sum(0, 0);
            for (int l = 0; l < k; ++l) {
                sum += std::conj(A[i*k + l]) * std::conj(B[l*n + j]);
            }
            expected[j*m + i] = alpha * sum + beta * C_original[j*m + i];
        }
    }
    
    linalg.gemm(Transpose::CONJ_TRANS, Transpose::CONJ_TRANS, m, n, k,
                &alpha, A.data(), B.data(), &beta, C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-9));
}

TEST_F(LinearAlgebraTest, CPU_GEAM_NoTrans) {
    CpuLinearAlgebra linalg;
    int m = 30, n = 25;
    Complex alpha(2.0, 0.5), beta(1.5, -0.5);
    
    auto A = randomMatrix(m, n);
    auto B = randomMatrix(m, n);
    std::vector<Complex> C(m * n);
    
    // Expected: C = alpha*A + beta*B
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * A[j*m + i] + beta * B[j*m + i];
        }
    }
    
    linalg.geam(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A.data(), &beta, B.data(), C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_GEAM_TransA) {
    CpuLinearAlgebra linalg;
    int m = 30, n = 25;
    Complex alpha(1.0, 0.0), beta(1.0, 0.0);
    
    auto A = randomMatrix(n, m);  // A stored as n×m for transpose
    auto B = randomMatrix(m, n);
    std::vector<Complex> C(m * n);
    
    // Expected: C = A^H + B
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = std::conj(A[i*n + j]) + B[j*m + i];
        }
    }
    
    linalg.geam(Transpose::CONJ_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A.data(), &beta, B.data(), C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_GEAM_TransBoth) {
    CpuLinearAlgebra linalg;
    int m = 20, n = 15;
    Complex alpha(1.5, 0.25), beta(0.5, -0.25);
    
    auto A = randomMatrix(n, m);  // A stored as n×m for transpose
    auto B = randomMatrix(n, m);  // B stored as n×m for transpose
    std::vector<Complex> C(m * n);
    
    // Expected: C = alpha*A^H + beta*B^T
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * std::conj(A[i*n + j]) + beta * B[i*n + j];
        }
    }
    
    linalg.geam(Transpose::CONJ_TRANS, Transpose::TRANS, m, n,
                &alpha, A.data(), &beta, B.data(), C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_GEAM_AlphaZero) {
    CpuLinearAlgebra linalg;
    int m = 25, n = 20;
    Complex alpha(0.0, 0.0), beta(2.0, 1.0);
    
    auto A = randomMatrix(m, n);
    auto B = randomMatrix(m, n);
    std::vector<Complex> C(m * n);
    
    // Expected: C = 0*A + beta*B = beta*B
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = beta * B[j*m + i];
        }
    }
    
    linalg.geam(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A.data(), &beta, B.data(), C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_GEAM_BetaZero) {
    CpuLinearAlgebra linalg;
    int m = 25, n = 20;
    Complex alpha(2.0, -1.0), beta(0.0, 0.0);
    
    auto A = randomMatrix(m, n);
    auto B = randomMatrix(m, n);
    std::vector<Complex> C(m * n);
    
    // Expected: C = alpha*A + 0*B = alpha*A
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * A[j*m + i];
        }
    }
    
    linalg.geam(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A.data(), &beta, B.data(), C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_GEAM_TransBoth) {
    CudaLinearAlgebra linalg;
    int m = 20, n = 15;
    Complex alpha(1.5, 0.25), beta(0.5, -0.25);
    
    auto A_host = randomMatrix(n, m);  // A stored as n×m for transpose
    auto B_host = randomMatrix(n, m);  // B stored as n×m for transpose
    
    // Expected: C = alpha*A^H + beta*B^T
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * std::conj(A_host[i*n + j]) + beta * B_host[i*n + j];
        }
    }
    
    CudaMemory A_dev(n * m * sizeof(Complex));
    CudaMemory B_dev(n * m * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), n * m * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), n * m * sizeof(Complex));
    
    linalg.geam(Transpose::CONJ_TRANS, Transpose::TRANS, m, n,
                &alpha, A_dev.data(), &beta, B_dev.data(), C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_GEAM_AlphaZero) {
    CudaLinearAlgebra linalg;
    int m = 25, n = 20;
    Complex alpha(0.0, 0.0), beta(2.0, 1.0);
    
    auto A_host = randomMatrix(m, n);
    auto B_host = randomMatrix(m, n);
    
    // Expected: C = 0*A + beta*B = beta*B
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = beta * B_host[j*m + i];
        }
    }
    
    CudaMemory A_dev(m * n * sizeof(Complex));
    CudaMemory B_dev(m * n * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * n * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), m * n * sizeof(Complex));
    
    linalg.geam(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A_dev.data(), &beta, B_dev.data(), C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_GEAM_BetaZero) {
    CudaLinearAlgebra linalg;
    int m = 25, n = 20;
    Complex alpha(2.0, -1.0), beta(0.0, 0.0);
    
    auto A_host = randomMatrix(m, n);
    auto B_host = randomMatrix(m, n);
    
    // Expected: C = alpha*A + 0*B = alpha*A
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * A_host[j*m + i];
        }
    }
    
    CudaMemory A_dev(m * n * sizeof(Complex));
    CudaMemory B_dev(m * n * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * n * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), m * n * sizeof(Complex));
    
    linalg.geam(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n,
                &alpha, A_dev.data(), &beta, B_dev.data(), C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CUDA_GEAM_TransBOnly) {
    CudaLinearAlgebra linalg;
    int m = 18, n = 22;
    Complex alpha(1.0, 0.5), beta(2.0, -0.5);
    
    auto A_host = randomMatrix(m, n);
    auto B_host = randomMatrix(n, m);  // B stored as n×m for transpose
    
    // Expected: C = alpha*A + beta*B^T
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * A_host[j*m + i] + beta * B_host[i*n + j];
        }
    }
    
    CudaMemory A_dev(m * n * sizeof(Complex));
    CudaMemory B_dev(n * m * sizeof(Complex));
    CudaMemory C_dev(m * n * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * n * sizeof(Complex));
    B_dev.copyFromHost(B_host.data(), n * m * sizeof(Complex));
    
    linalg.geam(Transpose::NO_TRANS, Transpose::TRANS, m, n,
                &alpha, A_dev.data(), &beta, B_dev.data(), C_dev.data(),
                PrecisionType::DOUBLE);
    
    std::vector<Complex> result(m * n);
    C_dev.copyToHost(result.data(), m * n * sizeof(Complex));
    
    EXPECT_TRUE(approxEqual(result, expected, 1e-10));
}

TEST_F(LinearAlgebraTest, CPU_GEAM_TransBOnly) {
    CpuLinearAlgebra linalg;
    int m = 18, n = 22;
    Complex alpha(1.0, 0.5), beta(2.0, -0.5);
    
    auto A = randomMatrix(m, n);
    auto B = randomMatrix(n, m);  // B stored as n×m for transpose
    std::vector<Complex> C(m * n);
    
    // Expected: C = alpha*A + beta*B^T
    std::vector<Complex> expected(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            expected[j*m + i] = alpha * A[j*m + i] + beta * B[i*n + j];
        }
    }
    
    linalg.geam(Transpose::NO_TRANS, Transpose::TRANS, m, n,
                &alpha, A.data(), &beta, B.data(), C.data(),
                PrecisionType::DOUBLE);
    
    EXPECT_TRUE(approxEqual(C, expected, 1e-10));
}

// ============================================================================
// Solver Tests
// ============================================================================

TEST_F(LinearAlgebraTest, CUDA_QR_Factorization) {
    CudaSolver solver;
    int m = 50, n = 30;
    
    auto A_host = randomMatrix(m, n);
    auto A_copy = A_host;  // Keep original for verification
    std::vector<Complex> tau(std::min(m, n));
    
    CudaMemory A_dev(m * n * sizeof(Complex));
    CudaMemory tau_dev(std::min(m,n) * sizeof(Complex));
    
    A_dev.copyFromHost(A_host.data(), m * n * sizeof(Complex));
    
    // Query workspace
    int lwork = solver.geqrf_workspace_size(m, n, PrecisionType::DOUBLE);
    CudaMemory work_dev(lwork * sizeof(Complex));
    
    // Perform QR
    solver.geqrf(m, n, A_dev.data(), tau_dev.data(), work_dev.data(),
                 lwork, PrecisionType::DOUBLE);
    
    // Extract Q
    lwork = solver.orgqr_workspace_size(m, n, std::min(m,n), PrecisionType::DOUBLE);
    work_dev = CudaMemory(lwork * sizeof(Complex));
    solver.orgqr(m, n, std::min(m,n), A_dev.data(), tau_dev.data(),
                 work_dev.data(), lwork, PrecisionType::DOUBLE);
    
    std::vector<Complex> Q(m * n);
    A_dev.copyToHost(Q.data(), m * n * sizeof(Complex));
    
    // Verify Q is orthogonal: Q^H * Q ≈ I
    std::vector<Complex> QtQ(n * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Complex sum(0, 0);
            for (int k = 0; k < m; ++k) {
                sum += std::conj(Q[i*m + k]) * Q[j*m + k];
            }
            QtQ[j*n + i] = sum;
        }
    }
    
    // Check diagonal is close to 1, off-diagonal close to 0
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                EXPECT_NEAR(std::abs(QtQ[j*n + i] - Complex(1,0)), 0.0, 1e-8);
            } else {
                EXPECT_NEAR(std::abs(QtQ[j*n + i]), 0.0, 1e-8);
            }
        }
    }
}

TEST_F(LinearAlgebraTest, CPU_QR_Factorization) {
    CpuSolver solver;
    int m = 50, n = 30;
    
    auto A = randomMatrix(m, n);
    std::vector<Complex> tau(std::min(m, n));
    
    int lwork = solver.geqrf_workspace_size(m, n, PrecisionType::DOUBLE);
    std::vector<Complex> work(lwork);
    
    solver.geqrf(m, n, A.data(), tau.data(), work.data(), lwork, PrecisionType::DOUBLE);
    
    lwork = solver.orgqr_workspace_size(m, n, std::min(m,n), PrecisionType::DOUBLE);
    work.resize(lwork);
    solver.orgqr(m, n, std::min(m,n), A.data(), tau.data(), work.data(), lwork, PrecisionType::DOUBLE);
    
    // Verify Q^H * Q ≈ I
    std::vector<Complex> QtQ(n * n, Complex(0,0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Complex sum(0, 0);
            for (int k = 0; k < m; ++k) {
                sum += std::conj(A[i*m + k]) * A[j*m + k];
            }
            QtQ[j*n + i] = sum;
        }
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                EXPECT_NEAR(std::abs(QtQ[j*n + i] - Complex(1,0)), 0.0, 1e-8);
            } else {
                EXPECT_NEAR(std::abs(QtQ[j*n + i]), 0.0, 1e-8);
            }
        }
    }
}

// ============================================================================
// Cross-Backend Compatibility Test
// ============================================================================

TEST_F(LinearAlgebraTest, CrossBackend_GEMM_Consistency) {
    int m = 20, n = 15, k = 10;
    Complex alpha(1.0, 0.0), beta(0.0, 0.0);
    
    auto A_host = randomMatrix(m, k);
    auto B_host = randomMatrix(k, n);
    
    // CUDA computation
    CudaLinearAlgebra cuda_linalg;
    CudaMemory A_cuda(m * k * sizeof(Complex));
    CudaMemory B_cuda(k * n * sizeof(Complex));
    CudaMemory C_cuda(m * n * sizeof(Complex));
    
    A_cuda.copyFromHost(A_host.data(), m * k * sizeof(Complex));
    B_cuda.copyFromHost(B_host.data(), k * n * sizeof(Complex));
    
    cuda_linalg.gemm(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n, k,
                     &alpha, A_cuda.data(), B_cuda.data(), &beta, C_cuda.data(),
                     PrecisionType::DOUBLE);
    
    std::vector<Complex> C_cuda_result(m * n);
    C_cuda.copyToHost(C_cuda_result.data(), m * n * sizeof(Complex));
    
    // CPU computation
    CpuLinearAlgebra cpu_linalg;
    std::vector<Complex> C_cpu(m * n, Complex(0,0));
    
    cpu_linalg.gemm(Transpose::NO_TRANS, Transpose::NO_TRANS, m, n, k,
                    &alpha, A_host.data(), B_host.data(), &beta, C_cpu.data(),
                    PrecisionType::DOUBLE);
    
    // Results should match
    EXPECT_TRUE(approxEqual(C_cuda_result, C_cpu, 1e-9));
}

TEST_F(LinearAlgebraTest, CrossBackend_GEMV_Consistency) {
    int m = 50, n = 40;
    Complex alpha(1.5, 0.5), beta(0.5, -0.5);
    
    auto A_host = randomMatrix(m, n);
    auto x_host = randomVector(n);
    auto y_host = randomVector(m);
    
    // CUDA computation
    CudaLinearAlgebra cuda_linalg;
    CudaMemory A_cuda(m * n * sizeof(Complex));
    CudaMemory x_cuda(n * sizeof(Complex));
    CudaMemory y_cuda(m * sizeof(Complex));
    
    A_cuda.copyFromHost(A_host.data(), m * n * sizeof(Complex));
    x_cuda.copyFromHost(x_host.data(), n * sizeof(Complex));
    y_cuda.copyFromHost(y_host.data(), m * sizeof(Complex));
    
    cuda_linalg.gemv(Transpose::NO_TRANS, m, n, &alpha, A_cuda.data(),
                     x_cuda.data(), &beta, y_cuda.data(), PrecisionType::DOUBLE);
    
    std::vector<Complex> y_cuda_result(m);
    y_cuda.copyToHost(y_cuda_result.data(), m * sizeof(Complex));
    
    // CPU computation
    CpuLinearAlgebra cpu_linalg;
    auto y_cpu = y_host;
    
    cpu_linalg.gemv(Transpose::NO_TRANS, m, n, &alpha, A_host.data(),
                    x_host.data(), &beta, y_cpu.data(), PrecisionType::DOUBLE);
    
    // Results should match
    EXPECT_TRUE(approxEqual(y_cuda_result, y_cpu, 1e-10));
}

TEST_F(LinearAlgebraTest, CrossBackend_QR_Consistency) {
    int m = 50, n = 30;
    
    auto A_host = randomMatrix(m, n);
    
    // CUDA computation
    CudaSolver cuda_solver;
    auto A_cuda_host = A_host;
    std::vector<Complex> tau_cuda(std::min(m, n));
    
    CudaMemory A_cuda(m * n * sizeof(Complex));
    CudaMemory tau_cuda_dev(std::min(m, n) * sizeof(Complex));
    
    A_cuda.copyFromHost(A_cuda_host.data(), m * n * sizeof(Complex));
    
    int lwork_cuda = cuda_solver.geqrf_workspace_size(m, n, PrecisionType::DOUBLE);
    CudaMemory work_cuda(lwork_cuda * sizeof(Complex));
    
    cuda_solver.geqrf(m, n, A_cuda.data(), tau_cuda_dev.data(), work_cuda.data(),
                      lwork_cuda, PrecisionType::DOUBLE);
    
    lwork_cuda = cuda_solver.orgqr_workspace_size(m, n, std::min(m, n), PrecisionType::DOUBLE);
    work_cuda = CudaMemory(lwork_cuda * sizeof(Complex));
    
    cuda_solver.orgqr(m, n, std::min(m, n), A_cuda.data(), tau_cuda_dev.data(),
                      work_cuda.data(), lwork_cuda, PrecisionType::DOUBLE);
    
    std::vector<Complex> Q_cuda(m * n);
    A_cuda.copyToHost(Q_cuda.data(), m * n * sizeof(Complex));
    
    // CPU computation
    CpuSolver cpu_solver;
    auto A_cpu = A_host;
    std::vector<Complex> tau_cpu(std::min(m, n));
    
    int lwork_cpu = cpu_solver.geqrf_workspace_size(m, n, PrecisionType::DOUBLE);
    std::vector<Complex> work_cpu(lwork_cpu);
    
    cpu_solver.geqrf(m, n, A_cpu.data(), tau_cpu.data(), work_cpu.data(),
                     lwork_cpu, PrecisionType::DOUBLE);
    
    lwork_cpu = cpu_solver.orgqr_workspace_size(m, n, std::min(m, n), PrecisionType::DOUBLE);
    work_cpu.resize(lwork_cpu);
    
    cpu_solver.orgqr(m, n, std::min(m, n), A_cpu.data(), tau_cpu.data(),
                     work_cpu.data(), lwork_cpu, PrecisionType::DOUBLE);
    
    // Both Q matrices should be orthogonal - verify Q^H*Q ≈ I for both
    auto verify_orthogonal = [&](const std::vector<Complex>& Q) {
        std::vector<Complex> QtQ(n * n, Complex(0,0));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Complex sum(0, 0);
                for (int k = 0; k < m; ++k) {
                    sum += std::conj(Q[i*m + k]) * Q[j*m + k];
                }
                QtQ[j*n + i] = sum;
            }
        }
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    EXPECT_NEAR(std::abs(QtQ[j*n + i] - Complex(1,0)), 0.0, 1e-8);
                } else {
                    EXPECT_NEAR(std::abs(QtQ[j*n + i]), 0.0, 1e-8);
                }
            }
        }
    };
    
    verify_orthogonal(Q_cuda);
    verify_orthogonal(A_cpu);
}
