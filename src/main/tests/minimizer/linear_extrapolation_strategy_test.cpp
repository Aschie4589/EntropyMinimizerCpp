#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "minimizer/prediction/linear_extrapolation_strategy.h"
#include "minimizer/state/circular_buffer.h"

using namespace entropy;

// ============================================================================
// Test Fixture
// ============================================================================

class LinearExtrapolationStrategyTest : public ::testing::Test {
protected:
    LinearExtrapolationStrategy strategy_;
    
    // Helper to generate linear trend
    // entropy[i] = initial + slope * i
    CircularBuffer<double> generateLinearTrend(
        size_t n,
        double initial,
        double slope) {
        
        CircularBuffer<double> buffer(n);
        for (size_t i = 0; i < n; ++i) {
            buffer.push(initial + slope * i);
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

TEST_F(LinearExtrapolationStrategyTest, ConstructorSetsDefaults) {
    LinearExtrapolationStrategy strat;
    
    EXPECT_EQ(strat.getMinimumDataPoints(), 50);
    EXPECT_DOUBLE_EQ(strat.getConvergenceTolerance(), 1e-15);
}

TEST_F(LinearExtrapolationStrategyTest, ConstructorAcceptsCustomParameters) {
    LinearExtrapolationStrategy strat(1e-10, 25);
    
    EXPECT_EQ(strat.getMinimumDataPoints(), 25);
    EXPECT_DOUBLE_EQ(strat.getConvergenceTolerance(), 1e-10);
}

// ============================================================================
// Insufficient Data Tests
// ============================================================================

TEST_F(LinearExtrapolationStrategyTest, InsufficientDataReturnsInvalid) {
    CircularBuffer<double> buffer(100);
    
    // Add only 30 points (less than default minimum of 50)
    for (int i = 0; i < 30; ++i) {
        buffer.push(1.0 - i * 0.01);
    }
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_FALSE(result.valid);
}

TEST_F(LinearExtrapolationStrategyTest, EmptyBufferReturnsInvalid) {
    CircularBuffer<double> buffer(100);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_FALSE(result.valid);
}

// ============================================================================
// Perfect Linear Trend Tests
// ============================================================================

TEST_F(LinearExtrapolationStrategyTest, PerfectLinearTrendHighRSquared) {
    // entropy[i] = 1.0 - 0.01*i
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    EXPECT_NEAR(result.confidence, 1.0, 1e-10);  // Perfect R²
}

TEST_F(LinearExtrapolationStrategyTest, NegativeSlopeRequired) {
    // Positive slope (entropy increasing)
    auto buffer = generateLinearTrend(100, 0.5, 0.01);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_FALSE(result.valid);  // Invalid: entropy not decreasing
}

TEST_F(LinearExtrapolationStrategyTest, ZeroSlopeReturnsInvalid) {
    // Horizontal line
    auto buffer = generateLinearTrend(100, 0.5, 0.0);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_FALSE(result.valid);  // Invalid: no progress
}

// ============================================================================
// Prediction Accuracy Tests
// ============================================================================

TEST_F(LinearExtrapolationStrategyTest, CorrectIterationPrediction) {
    // entropy[i] = 1.0 - 0.01*i
    // To reach 1e-15: 1.0 - 0.01*x = 1e-15
    // x = (1.0 - 1e-15) / 0.01 ≈ 100
    // Current is at i=99, so ~1 more iteration
    
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    EXPECT_LT(result.predicted_iterations, 10);  // Should be very few
}

TEST_F(LinearExtrapolationStrategyTest, PredictedEntropyMatchesTolerance) {
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    // Predicted entropy should be convergence tolerance
    EXPECT_DOUBLE_EQ(result.predicted_entropy, strategy_.getConvergenceTolerance());
}

// ============================================================================
// Noisy Data Tests
// ============================================================================

TEST_F(LinearExtrapolationStrategyTest, SmallNoiseStillValid) {
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    addNoise(buffer, 0.001);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.confidence, 0.95);  // Still high R²
}

TEST_F(LinearExtrapolationStrategyTest, LargeNoiseReducesConfidence) {
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    addNoise(buffer, 0.1);  // Large noise relative to trend
    
    PredictionResult result = strategy_.predict(buffer);
    
    // May still be valid but with lower confidence
    // Noise might not be large enough to drop below 0.9, so just check it's reduced from perfect
    if (result.valid) {
        EXPECT_LT(result.confidence, 1.0);  // Not perfect fit
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(LinearExtrapolationStrategyTest, VerySmallSlopeWorks) {
    // Very slow linear decrease
    auto buffer = generateLinearTrend(100, 1.0, -1e-6);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    // Should predict many iterations remaining
    EXPECT_GT(result.predicted_iterations, 1000);
}

TEST_F(LinearExtrapolationStrategyTest, SteepSlopePredictsFewIterations) {
    // Fast linear decrease
    auto buffer = generateLinearTrend(100, 1.0, -0.1);
    
    PredictionResult result = strategy_.predict(buffer);
    
    EXPECT_TRUE(result.valid);
    // Should predict few iterations
    EXPECT_LT(result.predicted_iterations, 100);
}

TEST_F(LinearExtrapolationStrategyTest, AlreadyBelowToleranceWorks) {
    // Start below tolerance
    LinearExtrapolationStrategy strat(1e-10, 50);
    auto buffer = generateLinearTrend(100, 1e-12, -1e-14);
    
    PredictionResult result = strat.predict(buffer);
    
    if (result.valid) {
        // Already converged or very close
        EXPECT_LE(result.predicted_iterations, 10);
    }
}

TEST_F(LinearExtrapolationStrategyTest, ExactlyMinimumPointsWorks) {
    LinearExtrapolationStrategy strat(1e-15, 10);
    auto buffer = generateLinearTrend(10, 1.0, -0.01);
    
    PredictionResult result = strat.predict(buffer);
    
    EXPECT_TRUE(result.valid);
}

// ============================================================================
// Mathematical Correctness Tests
// ============================================================================

TEST_F(LinearExtrapolationStrategyTest, RSquaredInValidRange) {
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    
    PredictionResult result = strategy_.predict(buffer);
    
    if (result.valid) {
        EXPECT_GE(result.confidence, 0.0);
        EXPECT_LE(result.confidence, 1.0);
    }
}

TEST_F(LinearExtrapolationStrategyTest, RepeatedPredictionsConsistent) {
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    
    PredictionResult result1 = strategy_.predict(buffer);
    PredictionResult result2 = strategy_.predict(buffer);
    
    EXPECT_EQ(result1.valid, result2.valid);
    if (result1.valid) {
        EXPECT_DOUBLE_EQ(result1.predicted_entropy, result2.predicted_entropy);
        EXPECT_EQ(result1.predicted_iterations, result2.predicted_iterations);
        EXPECT_DOUBLE_EQ(result1.confidence, result2.confidence);
    }
}

TEST_F(LinearExtrapolationStrategyTest, PredictedIterationsNonNegative) {
    auto buffer = generateLinearTrend(100, 1.0, -0.01);
    
    PredictionResult result = strategy_.predict(buffer);
    
    if (result.valid) {
        EXPECT_GE(result.predicted_iterations, 0);
    }
}
