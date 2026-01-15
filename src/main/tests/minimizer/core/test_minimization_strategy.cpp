#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "minimizer/algorithm/minimization_strategy.h"
#include <complex>
#include <vector>
#include <cmath>

using namespace std::complex_literals;

// ============================================================================
// Test Fixtures and Mocks
// ============================================================================

/**
 * @brief Mock implementation for testing interface compliance
 */
class MockMinimizationStrategy : public IMinimizationStrategy {
public:
    MOCK_METHOD(void, initialize, 
                (int, int, int, double, PrecisionType), 
                (override));
    
    MOCK_METHOD(double, stepOnce, (IDeviceMemory*, IDeviceMemory*), (override));
    MOCK_METHOD(PrecisionType, getPrecision, (), (const, override));
    MOCK_METHOD(size_t, getWorkspaceSize, (), (const, override));
};

// Note: TestMinimizationStrategy removed as the new stateless interface
// requires actual device memory management which is better tested via
// integration tests with real implementations (CudaMinimizationStrategy, etc.)
// The interface is now tested via the Mock above.

// ============================================================================
// HostKrausOperators Tests
// ============================================================================

TEST(HostKrausOperatorsTest, DefaultConstruction) {
    HostKrausOperators kraus;
    
    EXPECT_EQ(kraus.kraus_count, 0);
    EXPECT_EQ(kraus.input_dim, 0);
    EXPECT_EQ(kraus.output_dim, 0);
    EXPECT_EQ(kraus.precision, PrecisionType::DOUBLE);
    EXPECT_TRUE(kraus.data.empty());
    EXPECT_EQ(kraus.sizeBytes(), 0);
}

TEST(HostKrausOperatorsTest, ValidConstructionDouble) {
    // Create 2 Kraus operators of dimension 3×2 (output×input)
    int d = 2, N = 2, M = 3;
    std::vector<std::complex<double>> data(d * M * N);
    
    // Fill with test pattern: K_i[row,col] = i + row*0.1 + col*0.01
    for (int i = 0; i < d; ++i) {
        for (int col = 0; col < N; ++col) {
            for (int row = 0; row < M; ++row) {
                data[i*M*N + col*M + row] = 
                    std::complex<double>(i + row*0.1 + col*0.01, 0.0);
            }
        }
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(data, d, N, M);
    
    EXPECT_EQ(kraus.kraus_count, 2);
    EXPECT_EQ(kraus.input_dim, 2);
    EXPECT_EQ(kraus.output_dim, 3);
    EXPECT_EQ(kraus.precision, PrecisionType::DOUBLE);
    EXPECT_EQ(kraus.data.size(), 12 * sizeof(std::complex<double>));
    EXPECT_EQ(kraus.sizeBytes(), 12 * sizeof(std::complex<double>));
    EXPECT_EQ(kraus.complexSize(), sizeof(std::complex<double>));
}

TEST(HostKrausOperatorsTest, ValidConstructionFloat) {
    int d = 2, N = 2, M = 3;
    std::vector<std::complex<float>> data(d * M * N, {1.0f, 0.0f});
    
    HostKrausOperators kraus = HostKrausOperators::fromFloat(data, d, N, M);
    
    EXPECT_EQ(kraus.kraus_count, 2);
    EXPECT_EQ(kraus.input_dim, 2);
    EXPECT_EQ(kraus.output_dim, 3);
    EXPECT_EQ(kraus.precision, PrecisionType::FLOAT);
    EXPECT_EQ(kraus.data.size(), 12 * sizeof(std::complex<float>));
    EXPECT_EQ(kraus.sizeBytes(), 12 * sizeof(std::complex<float>));
    EXPECT_EQ(kraus.complexSize(), sizeof(std::complex<float>));
}

TEST(HostKrausOperatorsTest, DimensionMismatchThrowsDouble) {
    std::vector<std::complex<double>> data(10);  // Wrong size
    
    EXPECT_THROW(
        HostKrausOperators::fromDouble(data, 2, 2, 3),  // Expects 2*2*3=12
        std::invalid_argument
    );
}

TEST(HostKrausOperatorsTest, DimensionMismatchThrowsFloat) {
    std::vector<std::complex<float>> data(10);  // Wrong size
    
    EXPECT_THROW(
        HostKrausOperators::fromFloat(data, 2, 2, 3),  // Expects 2*2*3=12
        std::invalid_argument
    );
}

TEST(HostKrausOperatorsTest, GetKrausPointerDouble) {
    int d = 3, N = 2, M = 2;
    std::vector<std::complex<double>> data(d * M * N);
    
    // Set first element of each Kraus to its index
    for (int i = 0; i < d; ++i) {
        data[i * M * N] = std::complex<double>(static_cast<double>(i), 0.0);
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(data, d, N, M);
    
    // Verify getKrausDouble returns correct pointers
    EXPECT_EQ(kraus.getKrausDouble(0)[0], 0.0 + 0.0i);
    EXPECT_EQ(kraus.getKrausDouble(1)[0], 1.0 + 0.0i);
    EXPECT_EQ(kraus.getKrausDouble(2)[0], 2.0 + 0.0i);
    
    // Verify pointer arithmetic
    const auto* base = reinterpret_cast<const std::complex<double>*>(kraus.getData());
    EXPECT_EQ(kraus.getKrausDouble(1), base + M*N);
    EXPECT_EQ(kraus.getKrausDouble(2), base + 2*M*N);
}

TEST(HostKrausOperatorsTest, GetKrausPointerFloat) {
    int d = 3, N = 2, M = 2;
    std::vector<std::complex<float>> data(d * M * N);
    
    // Set first element of each Kraus to its index
    for (int i = 0; i < d; ++i) {
        data[i * M * N] = std::complex<float>(static_cast<float>(i), 0.0f);
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromFloat(data, d, N, M);
    
    // Verify getKrausFloat returns correct pointers
    EXPECT_FLOAT_EQ(kraus.getKrausFloat(0)[0].real(), 0.0f);
    EXPECT_FLOAT_EQ(kraus.getKrausFloat(1)[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(kraus.getKrausFloat(2)[0].real(), 2.0f);
}

TEST(HostKrausOperatorsTest, GetKrausWrongPrecisionThrows) {
    std::vector<std::complex<double>> data_double(4, {1.0, 0.0});
    std::vector<std::complex<float>> data_float(4, {1.0f, 0.0f});
    
    auto kraus_double = HostKrausOperators::fromDouble(data_double, 1, 2, 2);
    auto kraus_float = HostKrausOperators::fromFloat(data_float, 1, 2, 2);
    
    // Double precision object should throw on getKrausFloat
    EXPECT_THROW(kraus_double.getKrausFloat(0), std::logic_error);
    
    // Float precision object should throw on getKrausDouble
    EXPECT_THROW(kraus_float.getKrausDouble(0), std::logic_error);
}

// ============================================================================
// IMinimizationStrategy Interface Tests
// ============================================================================

TEST(IMinimizationStrategyTest, MockUsage) {
    MockMinimizationStrategy strategy;
    
    // Setup expectations for the stateless interface
    EXPECT_CALL(strategy, getPrecision())
        .WillOnce(::testing::Return(PrecisionType::DOUBLE));
    
    EXPECT_CALL(strategy, getWorkspaceSize())
        .WillOnce(::testing::Return(1024));
    
    // Execute
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
    EXPECT_EQ(strategy.getWorkspaceSize(), 1024);
}

// Note: Integration tests for concrete implementations (CudaMinimizationStrategy)
// Note: Integration tests for concrete implementations (CudaMinimizationStrategy)
// are in separate test files that can properly set up device memory and GPU resources.

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
