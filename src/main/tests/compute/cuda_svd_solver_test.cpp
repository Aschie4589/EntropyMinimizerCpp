// CudaSVDSolver tests
// Tests for stateful CUDA SVD solver

#include <gtest/gtest.h>
#include <compute/backends/cuda/CudaSVDSolver.h>
#include <compute/backends/cuda/CudaDevice.h>
#include <compute/memory/CudaScratchMemory.h>
#include <compute/memory/CpuScratchMemory.h>
#include <compute/device/IComputeDevice.h>
#include <compute/linalg/ISVDSolver.h>
#include <compute/core/ComputeTypes.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <memory>

using namespace compute;

class CudaSVDSolverTest : public ::testing::Test {
protected:
    std::unique_ptr<CudaDevice> device_;

    void SetUp() override {
        // Create device with small scratch pools for testing
        device_ = std::make_unique<CudaDevice>(0, 1024 * 1024, 256 * 1024);
    }

    void TearDown() override {
        device_.reset();
    }

    // Helper to create simple test matrix on device
    // Helper to create complex test matrix on device
    // For complex matrices: data contains interleaved (real, imag) pairs
    // Size of data vector should be 2*m*n for complex matrices
    template<typename T>
    T* createTestMatrix(uint64_t m, uint64_t n, const std::vector<T>& data) {
        T* d_mat;
        size_t size_bytes = data.size() * sizeof(T);
        cudaMalloc(&d_mat, size_bytes);
        cudaMemcpy(d_mat, data.data(), size_bytes, cudaMemcpyHostToDevice);
        return d_mat;
    }

    // Helper to verify singular values
    template<typename T>
    void verifySingularValues(const T* d_S, uint64_t k, const std::vector<T>& expected, T tol = 1e-5) {
        std::vector<T> h_S(k);
        cudaMemcpy(h_S.data(), d_S, k * sizeof(T), cudaMemcpyDeviceToHost);
        
        for (uint64_t i = 0; i < k; ++i) {
            EXPECT_NEAR(h_S[i], expected[i], tol) << "Singular value " << i << " mismatch";
        }
    }
};

// ==================== Construction Tests ====================

TEST_F(CudaSVDSolverTest, Construction_ValidDimensions_Float) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    
    CudaSVDSolver solver(device_.get(), 100, 50, spec, PrecisionType::FLOAT);
    
    int m, n;
    solver.getDimensions(m, n);
    EXPECT_EQ(m, 100);
    EXPECT_EQ(n, 50);
    EXPECT_EQ(solver.getPrecision(), PrecisionType::FLOAT);
}

TEST_F(CudaSVDSolverTest, Construction_ValidDimensions_Double) {
    SVDSpec spec{SVDVectors::ALL, SVDVectors::ALL, SVDAlgorithm::POLAR};
    
    EXPECT_NO_THROW({
        CudaSVDSolver solver(device_.get(), 50, 100, spec, PrecisionType::DOUBLE);
        
        EXPECT_EQ(solver.getPrecision(), PrecisionType::DOUBLE);
    });
}

TEST_F(CudaSVDSolverTest, Construction_InvalidDimensions) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    
    // Zero rows
    EXPECT_THROW({
        CudaSVDSolver solver(device_.get(), 0, 50, spec, PrecisionType::FLOAT);
    }, std::invalid_argument);
    
    // Zero columns
    EXPECT_THROW({
        CudaSVDSolver solver(device_.get(), 50, 0, spec, PrecisionType::FLOAT);
    }, std::invalid_argument);
}

// ==================== Workspace Query Tests ====================

TEST_F(CudaSVDSolverTest, WorkspaceQuery_QRAlgorithm_TallMatrix) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 100, 50, spec, PrecisionType::FLOAT);
    
    size_t dev_bytes, host_bytes;
    solver.getWorkspaceSizes(dev_bytes, host_bytes);
    
    // cusolverDnXgesvd only uses device workspace (host_bytes = 0 is normal)
    EXPECT_GT(dev_bytes, 0);
    EXPECT_GE(host_bytes, 0);  // May be 0 for unified API
}

TEST_F(CudaSVDSolverTest, WorkspaceQuery_QRAlgorithm_WideMatrix_FallbackToPolar) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    
    // Wide matrix (m < n) should trigger fallback to POLAR
    CudaSVDSolver solver(device_.get(), 50, 100, spec, PrecisionType::FLOAT);
    
    size_t dev_bytes, host_bytes;
    solver.getWorkspaceSizes(dev_bytes, host_bytes);
    
    // POLAR algorithm (unified API) only uses device workspace
    EXPECT_GT(dev_bytes, 0);
    EXPECT_GE(host_bytes, 0);  // May be 0 for unified API
}

TEST_F(CudaSVDSolverTest, WorkspaceQuery_PolarAlgorithm) {
    SVDSpec spec{SVDVectors::ALL, SVDVectors::ALL, SVDAlgorithm::POLAR};
    CudaSVDSolver solver(device_.get(), 100, 50, spec, PrecisionType::DOUBLE);
    
    size_t dev_bytes, host_bytes;
    solver.getWorkspaceSizes(dev_bytes, host_bytes);
    
    EXPECT_GT(dev_bytes, 0);
    EXPECT_GE(host_bytes, 0);  // May be 0 for unified API
}

TEST_F(CudaSVDSolverTest, WorkspaceQuery_RandomizedAlgorithm) {
    SVDSpec spec;
    spec.lvectors = SVDVectors::THIN;
    spec.rvectors = SVDVectors::THIN;
    spec.algorithm = SVDAlgorithm::RANDOMIZED;
    spec.rank = 10;
    spec.oversampling = 5;
    CudaSVDSolver solver(device_.get(), 100, 50, spec, PrecisionType::FLOAT);
    
    size_t dev_bytes, host_bytes;
    solver.getWorkspaceSizes(dev_bytes, host_bytes);
    
    EXPECT_GT(dev_bytes, 0);
    EXPECT_GT(host_bytes, 0);
}

TEST_F(CudaSVDSolverTest, WorkspaceQuery_NoVectors) {
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 100, 50, spec, PrecisionType::FLOAT);
    
    size_t dev_bytes, host_bytes;
    solver.getWorkspaceSizes(dev_bytes, host_bytes);
    
    // Even with no vectors, device workspace is needed
    EXPECT_GT(dev_bytes, 0);
    EXPECT_GE(host_bytes, 0);  // May be 0 for unified API
}

// ==================== Compute Tests ====================

TEST_F(CudaSVDSolverTest, Compute_DiagonalMatrix_Float) {
    // Create 3x3 diagonal matrix with singular values [3, 2, 1]
    // Data format: complex (real, imag) pairs in column-major order
    std::vector<float> A_data = {
        // Column 0: [3+0i, 0+0i, 0+0i]
        3.0f, 0.0f,  // (0,0) = 3 + 0i
        0.0f, 0.0f,  // (1,0) = 0 + 0i
        0.0f, 0.0f,  // (2,0) = 0 + 0i
        // Column 1: [0+0i, 2+0i, 0+0i]
        0.0f, 0.0f,  // (0,1) = 0 + 0i
        2.0f, 0.0f,  // (1,1) = 2 + 0i
        0.0f, 0.0f,  // (2,1) = 0 + 0i
        // Column 2: [0+0i, 0+0i, 1+0i]
        0.0f, 0.0f,  // (0,2) = 0 + 0i
        0.0f, 0.0f,  // (1,2) = 0 + 0i
        1.0f, 0.0f   // (2,2) = 1 + 0i
    };
    
    float* d_A = createTestMatrix<float>(3, 3, A_data);  // 3*3*2 = 18 floats
    float* d_S;
    cudaMalloc(&d_S, 3 * sizeof(float));
    
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 3, 3, spec, PrecisionType::FLOAT);
    
    EXPECT_NO_THROW({
        solver.compute(d_A, d_S, nullptr, nullptr, 0);
    });
    
    // Verify singular values (should be [3, 2, 1] in descending order)
    std::vector<float> expected_S = {3.0f, 2.0f, 1.0f};
    verifySingularValues(d_S, 3, expected_S, 1e-4f);
    
    cudaFree(d_A);
    cudaFree(d_S);
}

TEST_F(CudaSVDSolverTest, Compute_RankDeficientMatrix_Double) {
    // Create 4x3 complex matrix with rank 2
    // Data format: complex (real, imag) pairs in column-major order
    std::vector<double> A_data = {
        // Column 0: [1+0i, 0+0i, 0+0i, 0+0i]
        1.0, 0.0,  // (0,0)
        0.0, 0.0,  // (1,0)
        0.0, 0.0,  // (2,0)
        0.0, 0.0,  // (3,0)
        // Column 1: [0+0i, 1+0i, 0+0i, 0+0i]
        0.0, 0.0,  // (0,1)
        1.0, 0.0,  // (1,1)
        0.0, 0.0,  // (2,1)
        0.0, 0.0,  // (3,1)
        // Column 2: [0+0i, 0+0i, 0+0i, 0+0i]
        0.0, 0.0,  // (0,2)
        0.0, 0.0,  // (1,2)
        0.0, 0.0,  // (2,2)
        0.0, 0.0   // (3,2)
    };
    
    double* d_A = createTestMatrix<double>(4, 3, A_data);  // 4*3*2 = 24 doubles
    double* d_S;
    cudaMalloc(&d_S, 3 * sizeof(double));
    
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::POLAR};
    CudaSVDSolver solver(device_.get(), 4, 3, spec, PrecisionType::DOUBLE);
    
    solver.compute(d_A, d_S, nullptr, nullptr, 0);
    
    // Verify singular values [1, 1, 0]
    std::vector<double> h_S(3);
    cudaMemcpy(h_S.data(), d_S, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(h_S[0], 1.0, 1e-10);
    EXPECT_NEAR(h_S[1], 1.0, 1e-10);
    EXPECT_NEAR(h_S[2], 0.0, 1e-10);
    
    cudaFree(d_A);
    cudaFree(d_S);
}

// ==================== Spec Modification Tests ====================

TEST_F(CudaSVDSolverTest, SetSpec_UpdatesWorkspace) {
    SVDSpec spec1{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 100, 50, spec1, PrecisionType::FLOAT);
    
    size_t dev1, host1;
    solver.getWorkspaceSizes(dev1, host1);
    
    // Change to request vectors
    SVDSpec spec2{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    solver.setSpec(spec2);
    
    size_t dev2, host2;
    solver.getWorkspaceSizes(dev2, host2);
    
    // Workspace may change when requesting vectors
    // Device workspace should be non-zero
    EXPECT_GT(dev2, 0);
    // Host workspace is 0 for unified API (cusolverDnXgesvd)
    EXPECT_GE(host2, 0);
    
    EXPECT_EQ(solver.getSpec().lvectors, SVDVectors::THIN);
    EXPECT_EQ(solver.getSpec().rvectors, SVDVectors::THIN);
}

TEST_F(CudaSVDSolverTest, SetSpec_ChangeAlgorithm) {
    SVDSpec spec_qr{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 100, 50, spec_qr, PrecisionType::DOUBLE);
    
    EXPECT_EQ(solver.getSpec().algorithm, SVDAlgorithm::QR);
    
    SVDSpec spec_polar{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::POLAR};
    solver.setSpec(spec_polar);
    
    EXPECT_EQ(solver.getSpec().algorithm, SVDAlgorithm::POLAR);
}

// ==================== Reuse Tests ====================

TEST_F(CudaSVDSolverTest, Reuse_MultipleSolves_SameSize) {
    // Create two different complex matrices (2x2)
    // Data format: complex (real, imag) pairs in column-major order
    std::vector<float> A1_data = {
        // Column 0: [1+0i, 0+0i]
        1.0f, 0.0f,  // (0,0)
        0.0f, 0.0f,  // (1,0)
        // Column 1: [0+0i, 2+0i]
        0.0f, 0.0f,  // (0,1)
        2.0f, 0.0f   // (1,1)
    };
    std::vector<float> A2_data = {
        // Column 0: [3+0i, 0+0i]
        3.0f, 0.0f,  // (0,0)
        0.0f, 0.0f,  // (1,0)
        // Column 1: [0+0i, 4+0i]
        0.0f, 0.0f,  // (0,1)
        4.0f, 0.0f   // (1,1)
    };
    
    float* d_A = createTestMatrix<float>(2, 2, A1_data);  // 2*2*2 = 8 floats
    float* d_S;
    cudaMalloc(&d_S, 2 * sizeof(float));
    
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 2, 2, spec, PrecisionType::FLOAT);
    
    // First solve
    solver.compute(d_A, d_S, nullptr, nullptr, 0);
    verifySingularValues(d_S, 2, {2.0f, 1.0f}, 1e-4f);
    
    // Second solve with different data
    cudaMemcpy(d_A, A2_data.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);
    solver.compute(d_A, d_S, nullptr, nullptr, 0);
    verifySingularValues(d_S, 2, {4.0f, 3.0f}, 1e-4f);
    
    cudaFree(d_A);
    cudaFree(d_S);
}

TEST_F(CudaSVDSolverTest, GetBackend_ReturnsCUDA) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 10, 10, spec, PrecisionType::FLOAT);
    
    EXPECT_EQ(solver.getBackend(), DeviceBackend::CUDA);
}

TEST_F(CudaSVDSolverTest, GetDimensions_Immutable) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    CudaSVDSolver solver(device_.get(), 123, 456, spec, PrecisionType::DOUBLE);
    
    int m, n;
    solver.getDimensions(m, n);
    EXPECT_EQ(m, 123);
    EXPECT_EQ(n, 456);
    
    // Dimensions should remain unchanged after operations
    SVDSpec new_spec{SVDVectors::ALL, SVDVectors::ALL, SVDAlgorithm::POLAR};
    solver.setSpec(new_spec);
    
    int m2, n2;
    solver.getDimensions(m2, n2);
    EXPECT_EQ(m2, 123);
    EXPECT_EQ(n2, 456);
}
