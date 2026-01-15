#include <gtest/gtest.h>
#include "compute/backends/cpu/CpuDevice.h"
#include "compute/backends/cpu/CpuSVDSolver.h"
#include "compute/backends/cpu/CpuMemory.h"
#include <vector>
#include <complex>
#include <cmath>

using namespace compute;

class CpuSVDSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = std::make_unique<CpuDevice>(0, 1024 * 1024);  // 1 MB scratch
    }

    std::unique_ptr<CpuDevice> device_;

    // Helper to create complex test matrix in CPU memory
    // For complex matrices: data contains interleaved (real, imag) pairs
    template<typename T>
    T* createTestMatrix(int m, int n, const std::vector<T>& data) {
        T* mat = new T[data.size()];
        std::memcpy(mat, data.data(), data.size() * sizeof(T));
        return mat;
    }

    // Helper to verify singular values
    template<typename T>
    void verifySingularValues(const T* S, int k, const std::vector<T>& expected, T tol = 1e-5) {
        for (int i = 0; i < k; ++i) {
            EXPECT_NEAR(S[i], expected[i], tol) << "Singular value " << i << " mismatch";
        }
    }
};

// ==================== Construction Tests ====================

TEST_F(CpuSVDSolverTest, Construction_ValidDimensions_Float) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    
    CpuSVDSolver solver(device_.get(), 100, 50, spec, PrecisionType::FLOAT);
    
    int m, n;
    solver.getDimensions(m, n);
    EXPECT_EQ(m, 100);
    EXPECT_EQ(n, 50);
    EXPECT_EQ(solver.getPrecision(), PrecisionType::FLOAT);
}

TEST_F(CpuSVDSolverTest, Construction_ValidDimensions_Double) {
    SVDSpec spec{SVDVectors::ALL, SVDVectors::ALL, SVDAlgorithm::QR};
    
    EXPECT_NO_THROW({
        CpuSVDSolver solver(device_.get(), 50, 100, spec, PrecisionType::DOUBLE);
        EXPECT_EQ(solver.getPrecision(), PrecisionType::DOUBLE);
    });
}

TEST_F(CpuSVDSolverTest, Construction_InvalidDimensions) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    
    // Zero rows
    EXPECT_THROW({
        CpuSVDSolver solver(device_.get(), 0, 50, spec, PrecisionType::FLOAT);
    }, std::invalid_argument);
    
    // Zero columns
    EXPECT_THROW({
        CpuSVDSolver solver(device_.get(), 50, 0, spec, PrecisionType::FLOAT);
    }, std::invalid_argument);
}

// ==================== Workspace Query Tests ====================

TEST_F(CpuSVDSolverTest, WorkspaceQuery_AllVectors) {
    SVDSpec spec{SVDVectors::ALL, SVDVectors::ALL, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 100, 50, spec, PrecisionType::DOUBLE);
    
    size_t device_bytes, host_bytes;
    solver.getWorkspaceSizes(device_bytes, host_bytes);
    
    EXPECT_GT(device_bytes, 0);
    EXPECT_EQ(host_bytes, 0);  // CPU doesn't use separate host workspace
}

TEST_F(CpuSVDSolverTest, WorkspaceQuery_NoVectors) {
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 80, 60, spec, PrecisionType::FLOAT);
    
    size_t device_bytes, host_bytes;
    solver.getWorkspaceSizes(device_bytes, host_bytes);
    
    EXPECT_GT(device_bytes, 0);
    EXPECT_EQ(host_bytes, 0);
}

// ==================== Compute Tests ====================

TEST_F(CpuSVDSolverTest, Compute_DiagonalMatrix_Float) {
    // Create 3x3 diagonal matrix with singular values [3, 2, 1]
    // Data format: complex (real, imag) pairs in column-major order
    std::vector<std::complex<float>> A_data = {
        // Column 0: [3+0i, 0+0i, 0+0i]
        {3.0f, 0.0f},  // (0,0)
        {0.0f, 0.0f},  // (1,0)
        {0.0f, 0.0f},  // (2,0)
        // Column 1: [0+0i, 2+0i, 0+0i]
        {0.0f, 0.0f},  // (0,1)
        {2.0f, 0.0f},  // (1,1)
        {0.0f, 0.0f},  // (2,1)
        // Column 2: [0+0i, 0+0i, 1+0i]
        {0.0f, 0.0f},  // (0,2)
        {0.0f, 0.0f},  // (1,2)
        {1.0f, 0.0f}   // (2,2)
    };
    
    std::complex<float>* A = createTestMatrix<std::complex<float>>(3, 3, A_data);
    float* S = new float[3];
    
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 3, 3, spec, PrecisionType::FLOAT);
    
    EXPECT_NO_THROW({
        solver.compute(A, S, nullptr, nullptr, nullptr);
    });
    
    // Verify singular values (should be [3, 2, 1] in descending order)
    std::vector<float> expected_S = {3.0f, 2.0f, 1.0f};
    verifySingularValues(S, 3, expected_S, 1e-4f);
    
    delete[] A;
    delete[] S;
}

TEST_F(CpuSVDSolverTest, Compute_RankDeficientMatrix_Double) {
    // Create 4x3 complex matrix with rank 2
    // Data format: complex (real, imag) pairs in column-major order
    std::vector<std::complex<double>> A_data = {
        // Column 0: [1+0i, 0+0i, 0+0i, 0+0i]
        {1.0, 0.0},  // (0,0)
        {0.0, 0.0},  // (1,0)
        {0.0, 0.0},  // (2,0)
        {0.0, 0.0},  // (3,0)
        // Column 1: [0+0i, 1+0i, 0+0i, 0+0i]
        {0.0, 0.0},  // (0,1)
        {1.0, 0.0},  // (1,1)
        {0.0, 0.0},  // (2,1)
        {0.0, 0.0},  // (3,1)
        // Column 2: [0+0i, 0+0i, 0+0i, 0+0i]
        {0.0, 0.0},  // (0,2)
        {0.0, 0.0},  // (1,2)
        {0.0, 0.0},  // (2,2)
        {0.0, 0.0}   // (3,2)
    };
    
    std::complex<double>* A = createTestMatrix<std::complex<double>>(4, 3, A_data);
    double* S = new double[3];
    
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 4, 3, spec, PrecisionType::DOUBLE);
    
    solver.compute(A, S, nullptr, nullptr, nullptr);
    
    // Verify singular values [1, 1, 0]
    EXPECT_NEAR(S[0], 1.0, 1e-10);
    EXPECT_NEAR(S[1], 1.0, 1e-10);
    EXPECT_NEAR(S[2], 0.0, 1e-10);
    
    delete[] A;
    delete[] S;
}

// ==================== Spec Modification Tests ====================

TEST_F(CpuSVDSolverTest, SetSpec_UpdatesWorkspace) {
    SVDSpec spec1{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 100, 50, spec1, PrecisionType::FLOAT);
    
    size_t dev1, host1;
    solver.getWorkspaceSizes(dev1, host1);
    
    // Change to request vectors
    SVDSpec spec2{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    solver.setSpec(spec2);
    
    size_t dev2, host2;
    solver.getWorkspaceSizes(dev2, host2);
    
    // Workspace should be non-zero
    EXPECT_GT(dev2, 0);
    EXPECT_EQ(host2, 0);
    
    EXPECT_EQ(solver.getSpec().lvectors, SVDVectors::THIN);
    EXPECT_EQ(solver.getSpec().rvectors, SVDVectors::THIN);
}

TEST_F(CpuSVDSolverTest, SetSpec_ChangeAlgorithm) {
    SVDSpec spec_qr{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 100, 50, spec_qr, PrecisionType::DOUBLE);
    
    EXPECT_EQ(solver.getSpec().algorithm, SVDAlgorithm::QR);
    
    // POLAR maps to same LAPACK implementation on CPU
    SVDSpec spec_polar{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::POLAR};
    solver.setSpec(spec_polar);
    
    EXPECT_EQ(solver.getSpec().algorithm, SVDAlgorithm::POLAR);
}

// ==================== Reuse Tests ====================

TEST_F(CpuSVDSolverTest, Reuse_MultipleSolves_SameSize) {
    // Create two different complex matrices (2x2)
    std::vector<std::complex<float>> A1_data = {
        {1.0f, 0.0f}, {0.0f, 0.0f},  // Column 0
        {0.0f, 0.0f}, {2.0f, 0.0f}   // Column 1
    };
    std::vector<std::complex<float>> A2_data = {
        {3.0f, 0.0f}, {0.0f, 0.0f},  // Column 0
        {0.0f, 0.0f}, {4.0f, 0.0f}   // Column 1
    };
    
    std::complex<float>* A = createTestMatrix<std::complex<float>>(2, 2, A1_data);
    float* S = new float[2];
    
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 2, 2, spec, PrecisionType::FLOAT);
    
    // First solve
    solver.compute(A, S, nullptr, nullptr, nullptr);
    verifySingularValues(S, 2, {2.0f, 1.0f}, 1e-4f);
    
    // Second solve with different data
    std::memcpy(A, A2_data.data(), 4 * sizeof(std::complex<float>));
    solver.compute(A, S, nullptr, nullptr, nullptr);
    verifySingularValues(S, 2, {4.0f, 3.0f}, 1e-4f);
    
    delete[] A;
    delete[] S;
}

// ==================== Utility Tests ====================

TEST_F(CpuSVDSolverTest, GetBackend_ReturnsCPU) {
    SVDSpec spec{SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 10, 10, spec, PrecisionType::FLOAT);
    
    EXPECT_EQ(solver.getBackend(), DeviceBackend::CPU);
}

TEST_F(CpuSVDSolverTest, GetDimensions_Immutable) {
    SVDSpec spec{SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR};
    CpuSVDSolver solver(device_.get(), 123, 456, spec, PrecisionType::DOUBLE);
    
    int m, n;
    solver.getDimensions(m, n);
    EXPECT_EQ(m, 123);
    EXPECT_EQ(n, 456);
}
