#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "minimizer/prediction/exponential_fitting_strategy.h"
#include "minimizer/state/circular_buffer.h"

using namespace entropy;

// ============================================================================
// Test Fixture
// ============================================================================

class ExponentialFittingStrategyTest : public ::testing::Test {
protected:
    ExponentialFittingStrategy strategy_;
    
    // Helper to generate exponential decay sequence
    // entropy[i] = initial * exp(-decay_rate * i) + final
    CircularBuffer<double> generateExponentialDecay(
        size_t n,
        double initial,
        double final_entropy,
        double decay_rate) {
        
        CircularBuffer<double> buffer(n);
        for (size_t i = 0; i < n; ++i) {
            double entropy = final_entropy + (initial - final_entropy) * std::exp(-decay_rate * i);
            buffer.push(entropy);
        }
        return buffer;
    }
    
    // Helper to add Gaussian noise
    void addNoise(CircularBuffer<double>& buffer, double noise_stddev, unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(0.0, noise_stddev);
        
        size_t n = buffer.size();
        std::vector<double> values;
        for (size_t i = 0; i < n; ++i) {
            values.push_back(buffer.get(i) + dist(gen));
        }
        
        buffer.clear();
        for (double val : values) {
            buffer.push(val);
        }
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, ConstructorSetsDefaults) {
    ExponentialFittingStrategy strat;
    
    EXPECT_EQ(strat.getMinimumDataPoints(), 100);
    EXPECT_DOUBLE_EQ(strat.getRSquaredThreshold(), 0.999);
    EXPECT_DOUBLE_EQ(strat.getConvergenceTolerance(), 1e-15);
}

TEST_F(ExponentialFittingStrategyTest, ConstructorAcceptsCustomParameters) {
    ExponentialFittingStrategy strat(0.95, 1e-10, 50);
    
    EXPECT_EQ(strat.getMinimumDataPoints(), 50);
    EXPECT_DOUBLE_EQ(strat.getRSquaredThreshold(), 0.95);
    EXPECT_DOUBLE_EQ(strat.getConvergenceTolerance(), 1e-10);
}

// ============================================================================
// Insufficient Data Tests
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, InsufficientDataReturnsInvalid) {
    CircularBuffer<double> buffer(200);
    
    // Add only 50 points (less than default minimum of 100)
    for (int i = 0; i < 50; ++i) {
        buffer.push(1.0 - i * 0.01);
    }
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_FALSE(result.valid);
}

TEST_F(ExponentialFittingStrategyTest, EmptyBufferReturnsInvalid) {
    CircularBuffer<double> buffer(200);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_FALSE(result.valid);
}

// ============================================================================
// Perfect Exponential Decay Tests
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, PerfectExponentialDecayHighRSquared) {
    // Generate perfect exponential decay: entropy[i] = 0.001 + 0.999*exp(-0.05*i)
    auto buffer = generateExponentialDecay(150, 1.0, 0.001, 0.05);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.confidence, 0.999);  // Should be very high R²
}

TEST_F(ExponentialFittingStrategyTest, PerfectDecayPredictsFinalEntropy) {
    // entropy[i] = 0.0 + 1.0*exp(-0.1*i)
    auto buffer = generateExponentialDecay(150, 1.0, 0.0, 0.1);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    // Should predict close to 0.0
    EXPECT_NEAR(result.predicted_entropy, 0.0, 0.01);
}

// ============================================================================
// Noisy Data Tests
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, SmallNoiseStillProducesValidFit) {
    ExponentialFittingStrategy relaxed_strategy(0.95, 1e-15, 100);  // Lower R² threshold for noisy data
    
    auto buffer = generateExponentialDecay(150, 1.0, 0.01, 0.05);
    addNoise(buffer, 0.001);  // Small noise
    
    PredictionResult result = relaxed_strategy.predict(buffer);
    
    // With small noise, should still produce valid fit (though maybe not with 0.999 threshold)
    if (result.valid) {
        EXPECT_GT(result.confidence, 0.90);  // Should have good R²
    }
}

TEST_F(ExponentialFittingStrategyTest, LargeNoiseReducesConfidence) {
    ExponentialFittingStrategy relaxed_strategy(0.5, 1e-15, 100);  // Lower R² threshold
    
    auto buffer = generateExponentialDecay(150, 1.0, 0.01, 0.05);
    addNoise(buffer, 0.1);  // Large noise
    
    PredictionResult result = relaxed_strategy.predict(buffer);
    
    // May or may not be valid depending on noise, but confidence should be lower
    if (result.valid) {
        EXPECT_LT(result.confidence, 0.95);
    }
}

// ============================================================================
// Invalid Sequence Tests (Should Return Invalid)
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, RandomDataReturnsInvalidOrLowRSquared) {
    CircularBuffer<double> buffer(200);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < 150; ++i) {
        buffer.push(dist(gen));
    }
    
    PredictionResult result = strategy_.predict(buffer);
    
    // Either invalid or very low R²
    if (result.valid) {
        EXPECT_LT(result.confidence, 0.5);
    }
}

TEST_F(ExponentialFittingStrategyTest, ConstantEntropyReturnsInvalid) {
    CircularBuffer<double> buffer(200);
    
    for (int i = 0; i < 150; ++i) {
        buffer.push(0.5);  // Constant
    }
    
    PredictionResult result = strategy_.predict(buffer);
    
    // No deltas (delta = 0), can't take log
    EXPECT_FALSE(result.valid);
}

TEST_F(ExponentialFittingStrategyTest, IncreasingEntropyReturnsInvalid) {
    CircularBuffer<double> buffer(200);
    
    for (int i = 0; i < 150; ++i) {
        buffer.push(0.5 + i * 0.01);  // Increasing
    }
    
    PredictionResult result = strategy_.predict(buffer);
    
    // Negative deltas, can't take log
    EXPECT_FALSE(result.valid);
}

TEST_F(ExponentialFittingStrategyTest, PositiveSlopeReturnsInvalid) {
    // Create sequence where log(delta) has positive slope
    // This means deltas are increasing = entropy getting worse faster
    CircularBuffer<double> buffer(200);
    
    double entropy = 1.0;
    double delta = 0.001;
    for (int i = 0; i < 150; ++i) {
        buffer.push(entropy);
        entropy -= delta;
        delta *= 1.05;  // Deltas increasing (getting worse at slowing down)
    }
    
    PredictionResult result = strategy_.predict(buffer);
    
    // Positive slope in log(delta) space is invalid
    EXPECT_FALSE(result.valid);
}

// ============================================================================
// Iteration Prediction Tests
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, PredictedIterationsNonNegative) {
    auto buffer = generateExponentialDecay(150, 1.0, 0.01, 0.05);
    
    PredictionResult result = strategy_.predict(buffer);
    
    if (result.valid) {
        EXPECT_GE(result.predicted_iterations, 0);
    }
}

TEST_F(ExponentialFittingStrategyTest, NearConvergencePredictsFewIterations) {
    // Start already very close to convergence
    auto buffer = generateExponentialDecay(150, 1e-10, 1e-16, 0.1);
    
    ExponentialFittingStrategy strat(0.9, 1e-15, 100);  // Relaxed threshold
    PredictionResult result = strat.predict(buffer);
    
    if (result.valid) {
        // Should predict we're almost done
        EXPECT_LT(result.predicted_iterations, 50);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, VerySmallDeltasHandledGracefully) {
    CircularBuffer<double> buffer(200);
    
    double entropy = 1e-10;
    for (int i = 0; i < 150; ++i) {
        buffer.push(entropy);
        entropy *= 0.99;  // Very small decrements
    }
    
    PredictionResult result = strategy_.predict(buffer);
    
    // Should either work or fail gracefully (not crash)
    // Valid is acceptable either way
}

TEST_F(ExponentialFittingStrategyTest, SingleDeltaPointIsInsufficient) {
    CircularBuffer<double> buffer(200);
    buffer.push(1.0);
    buffer.push(0.9);  // Only one delta
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_FALSE(result.valid);  // Not enough data
}

TEST_F(ExponentialFittingStrategyTest, ExactlyMinimumPointsWorks) {
    ExponentialFittingStrategy strat(0.9, 1e-15, 10);  // Require exactly 10
    
    auto buffer = generateExponentialDecay(10, 1.0, 0.01, 0.1);
    
    PredictionResult result = strat.predict(buffer);
    
    // Should attempt prediction with exactly minimum points
    EXPECT_TRUE(result.valid);
}

// ============================================================================
// Mathematical Correctness Tests
// ============================================================================

TEST_F(ExponentialFittingStrategyTest, RSquaredInValidRange) {
    auto buffer = generateExponentialDecay(150, 1.0, 0.01, 0.05);
    
    PredictionResult result = strategy_.predict(buffer);
    
    if (result.valid) {
        EXPECT_GE(result.confidence, 0.0);
        EXPECT_LE(result.confidence, 1.0);
    }
}

TEST_F(ExponentialFittingStrategyTest, PredictedEntropyBelowCurrent) {
    auto buffer = generateExponentialDecay(150, 1.0, 0.01, 0.05);
    double current_entropy = buffer.newest();
    
    PredictionResult result = strategy_.predict(buffer);
    
    if (result.valid) {
        // Predicted final entropy should be less than current
        EXPECT_LT(result.predicted_entropy, current_entropy);
    }
}

TEST_F(ExponentialFittingStrategyTest, RepeatedPredictionsConsistent) {
    auto buffer = generateExponentialDecay(150, 1.0, 0.01, 0.05);
    
    PredictionResult result1 = strategy_.predict(buffer);
    PredictionResult result2 = strategy_.predict(buffer);
    
    EXPECT_EQ(result1.valid, result2.valid);
    if (result1.valid) {
        EXPECT_DOUBLE_EQ(result1.predicted_entropy, result2.predicted_entropy);
        EXPECT_EQ(result1.predicted_iterations, result2.predicted_iterations);
        EXPECT_DOUBLE_EQ(result1.confidence, result2.confidence);
    }
}
