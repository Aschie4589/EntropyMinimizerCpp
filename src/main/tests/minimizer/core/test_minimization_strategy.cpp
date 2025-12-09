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
                (const HostKrausOperators&, double, 
                 const std::vector<std::complex<double>>&), 
                (override));
    
    MOCK_METHOD(void, stepOnce, (), (override));
    MOCK_METHOD(double, getEntropy, (), (const, override));
    MOCK_METHOD(void, getCurrentVector, 
                (std::vector<std::complex<double>>&), 
                (const, override));
    MOCK_METHOD(PrecisionType, getPrecision, (), (const, override));
    MOCK_METHOD(size_t, getMemoryRequired, 
                (int, int, int, PrecisionType), 
                (const, override));
};

/**
 * @brief Concrete test implementation for integration testing
 * 
 * Simulates a trivial "algorithm" that just tracks state without
 * actual computation. Used to test the interface contract.
 */
class TestMinimizationStrategy : public IMinimizationStrategy {
private:
    PrecisionType precision_;
    std::vector<std::complex<double>> current_vector_;
    double current_entropy_;
    int kraus_count_;
    int input_dim_;
    int output_dim_;
    double epsilon_;
    bool initialized_;
    int step_count_;
    
public:
    TestMinimizationStrategy() 
        : precision_(PrecisionType::DOUBLE),
          current_entropy_(-1.0),
          kraus_count_(0),
          input_dim_(0),
          output_dim_(0),
          epsilon_(0.0),
          initialized_(false),
          step_count_(0) {}
    
    void initialize(
        const HostKrausOperators& kraus,
        double epsilon,
        const std::vector<std::complex<double>>& initial_vector
    ) override {
        // Validate dimensions
        if (kraus.kraus_count > kraus.output_dim) {
            throw std::runtime_error(
                "Invalid dimensions: kraus_count > output_dim (violates SVD constraint)"
            );
        }
        
        if (initial_vector.size() != static_cast<size_t>(kraus.input_dim)) {
            throw std::runtime_error(
                "Initial vector size mismatch: expected " + 
                std::to_string(kraus.input_dim) + ", got " + 
                std::to_string(initial_vector.size())
            );
        }
        
        if (epsilon <= 0.0 || epsilon >= 1.0) {
            throw std::runtime_error("Epsilon must be in (0, 1)");
        }
        
        // Store configuration
        precision_ = kraus.precision;
        kraus_count_ = kraus.kraus_count;
        input_dim_ = kraus.input_dim;
        output_dim_ = kraus.output_dim;
        epsilon_ = epsilon;
        current_vector_ = initial_vector;
        initialized_ = true;
        step_count_ = 0;
        
        // Initial entropy (arbitrary test value)
        current_entropy_ = 1.5;
    }
    
    void stepOnce() override {
        if (!initialized_) {
            throw std::runtime_error("stepOnce called before initialize");
        }
        
        // Simulate algorithm: decrease entropy and slightly modify vector
        step_count_++;
        current_entropy_ *= 0.95;  // Decreasing entropy
        
        // Rotate phase slightly
        for (auto& val : current_vector_) {
            val *= std::exp(1i * 0.01);
        }
    }
    
    double getEntropy() const override {
        if (!initialized_) {
            throw std::runtime_error("getEntropy called before initialize");
        }
        return current_entropy_;
    }
    
    void getCurrentVector(std::vector<std::complex<double>>& host_buffer) const override {
        if (!initialized_) {
            throw std::runtime_error("getCurrentVector called before initialize");
        }
        
        if (host_buffer.size() != current_vector_.size()) {
            throw std::runtime_error(
                "Buffer size mismatch: expected " + 
                std::to_string(current_vector_.size()) + ", got " + 
                std::to_string(host_buffer.size())
            );
        }
        
        host_buffer = current_vector_;
    }
    
    PrecisionType getPrecision() const override {
        return precision_;
    }
    
    size_t getMemoryRequired(
        int kraus_count,
        int input_dim,
        int output_dim,
        PrecisionType precision
    ) const override {
        size_t complex_size = (precision == PrecisionType::FLOAT) ? 
                              sizeof(std::complex<float>) : 
                              sizeof(std::complex<double>);
        
        // Simplified calculation: Kraus + vector + workspace
        size_t kraus_mem = kraus_count * output_dim * input_dim * complex_size;
        size_t vector_mem = input_dim * complex_size;
        size_t workspace = output_dim * kraus_count * complex_size;  // Intermediate vectors
        
        return kraus_mem + vector_mem + workspace;
    }
    
    // Test accessors
    int getStepCount() const { return step_count_; }
    bool isInitialized() const { return initialized_; }
};

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
    
    // Setup expectations
    EXPECT_CALL(strategy, getPrecision())
        .WillOnce(::testing::Return(PrecisionType::DOUBLE));
    
    EXPECT_CALL(strategy, getEntropy())
        .WillOnce(::testing::Return(1.234));
    
    // Execute
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
    EXPECT_DOUBLE_EQ(strategy.getEntropy(), 1.234);
}

TEST(TestMinimizationStrategyTest, InitializationValidation) {
    TestMinimizationStrategy strategy;
    
    EXPECT_FALSE(strategy.isInitialized());
    
    // Create valid Kraus operators (double precision)
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    // Valid initialization should succeed
    EXPECT_NO_THROW(
        strategy.initialize(kraus, 0.01, initial_vec)
    );
    
    EXPECT_TRUE(strategy.isInitialized());
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
}

TEST(TestMinimizationStrategyTest, InitializationInvalidDimensions) {
    TestMinimizationStrategy strategy;
    
    // d > M violates SVD constraint
    int d = 5, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    EXPECT_THROW(
        strategy.initialize(kraus, 0.01, initial_vec),
        std::runtime_error
    );
}

TEST(TestMinimizationStrategyTest, InitializationInvalidVectorSize) {
    TestMinimizationStrategy strategy;
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> wrong_size_vec(N + 1, 1.0 + 0.0i);
    
    EXPECT_THROW(
        strategy.initialize(kraus, 0.01, wrong_size_vec),
        std::runtime_error
    );
}

TEST(TestMinimizationStrategyTest, InitializationInvalidEpsilon) {
    TestMinimizationStrategy strategy;
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    // Epsilon out of range
    EXPECT_THROW(
        strategy.initialize(kraus, 0.0, initial_vec),
        std::runtime_error
    );
    
    EXPECT_THROW(
        strategy.initialize(kraus, 1.0, initial_vec),
        std::runtime_error
    );
    
    EXPECT_THROW(
        strategy.initialize(kraus, -0.1, initial_vec),
        std::runtime_error
    );
}

TEST(TestMinimizationStrategyTest, StepOnceBeforeInitializeThrows) {
    TestMinimizationStrategy strategy;
    
    EXPECT_THROW(strategy.stepOnce(), std::runtime_error);
}

TEST(TestMinimizationStrategyTest, GetEntropyBeforeInitializeThrows) {
    TestMinimizationStrategy strategy;
    
    EXPECT_THROW(strategy.getEntropy(), std::runtime_error);
}

TEST(TestMinimizationStrategyTest, GetVectorBeforeInitializeThrows) {
    TestMinimizationStrategy strategy;
    std::vector<std::complex<double>> buffer(3);
    
    EXPECT_THROW(strategy.getCurrentVector(buffer), std::runtime_error);
}

TEST(TestMinimizationStrategyTest, StepOnceUpdatesState) {
    TestMinimizationStrategy strategy;
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    double initial_entropy = strategy.getEntropy();
    EXPECT_EQ(strategy.getStepCount(), 0);
    
    // Take one step
    strategy.stepOnce();
    
    EXPECT_EQ(strategy.getStepCount(), 1);
    EXPECT_LT(strategy.getEntropy(), initial_entropy);  // Entropy should decrease
    
    // Take another step
    double entropy_after_one = strategy.getEntropy();
    strategy.stepOnce();
    
    EXPECT_EQ(strategy.getStepCount(), 2);
    EXPECT_LT(strategy.getEntropy(), entropy_after_one);
}

TEST(TestMinimizationStrategyTest, GetCurrentVectorWorks) {
    TestMinimizationStrategy strategy;
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec{
        1.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i
    };
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    std::vector<std::complex<double>> retrieved_vec(N);
    strategy.getCurrentVector(retrieved_vec);
    
    // Should match initial vector
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(std::abs(retrieved_vec[i] - initial_vec[i]), 0.0, 1e-10);
    }
}

TEST(TestMinimizationStrategyTest, GetCurrentVectorWrongSizeThrows) {
    TestMinimizationStrategy strategy;
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    std::vector<std::complex<double>> wrong_size_buffer(N + 1);
    
    EXPECT_THROW(strategy.getCurrentVector(wrong_size_buffer), std::runtime_error);
}

TEST(TestMinimizationStrategyTest, GetMemoryRequiredScalesCorrectly) {
    TestMinimizationStrategy strategy;
    
    // Test DOUBLE precision
    size_t mem_double = strategy.getMemoryRequired(2, 3, 4, PrecisionType::DOUBLE);
    
    // Should include: Kraus (2*4*3) + vector (3) + workspace (4*2) = 24+3+8 = 35 complex<double>
    size_t expected_double = 35 * sizeof(std::complex<double>);
    EXPECT_EQ(mem_double, expected_double);
    
    // Test FLOAT precision (should be half)
    size_t mem_float = strategy.getMemoryRequired(2, 3, 4, PrecisionType::FLOAT);
    size_t expected_float = 35 * sizeof(std::complex<float>);
    EXPECT_EQ(mem_float, expected_float);
    
    // Verify FLOAT uses less memory
    EXPECT_LT(mem_float, mem_double);
}

TEST(TestMinimizationStrategyTest, PrecisionPersistence) {
    TestMinimizationStrategy strategy_float, strategy_double;
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data_double(d * M * N, 0.1 + 0.0i);
    std::vector<std::complex<float>> kraus_data_float(d * M * N, {0.1f, 0.0f});
    
    HostKrausOperators kraus_double = HostKrausOperators::fromDouble(kraus_data_double, d, N, M);
    HostKrausOperators kraus_float = HostKrausOperators::fromFloat(kraus_data_float, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    strategy_float.initialize(kraus_float, 0.01, initial_vec);
    strategy_double.initialize(kraus_double, 0.01, initial_vec);
    
    EXPECT_EQ(strategy_float.getPrecision(), PrecisionType::FLOAT);
    EXPECT_EQ(strategy_double.getPrecision(), PrecisionType::DOUBLE);
}

TEST(TestMinimizationStrategyTest, MultipleStepsMonotoneEntropy) {
    TestMinimizationStrategy strategy;
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    // Take 10 steps, entropy should monotonically decrease
    std::vector<double> entropies;
    for (int i = 0; i < 10; ++i) {
        entropies.push_back(strategy.getEntropy());
        strategy.stepOnce();
    }
    entropies.push_back(strategy.getEntropy());
    
    for (size_t i = 1; i < entropies.size(); ++i) {
        EXPECT_LT(entropies[i], entropies[i-1]) 
            << "Entropy should decrease at step " << i;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
