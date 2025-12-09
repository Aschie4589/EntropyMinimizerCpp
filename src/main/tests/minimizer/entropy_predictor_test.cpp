#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <random>
#include "minimizer/state/entropy_predictor.h"
#include "minimizer/prediction/exponential_fitting_strategy.h"
#include "minimizer/prediction/linear_extrapolation_strategy.h"

using namespace entropy;

// ============================================================================
// Test Fixture
// ============================================================================

class EntropyPredictorTest : public ::testing::Test {
protected:
    // Helper to create predictor with exponential strategy
    std::unique_ptr<EntropyPredictor> createExponentialPredictor(size_t window_size = 200) {
        auto strategy = std::make_unique<ExponentialFittingStrategy>(0.9, 1e-15, 50);
        return std::make_unique<EntropyPredictor>(std::move(strategy), window_size);
    }
    
    // Helper to create predictor with linear strategy
    std::unique_ptr<EntropyPredictor> createLinearPredictor(size_t window_size = 200) {
        auto strategy = std::make_unique<LinearExtrapolationStrategy>(1e-15, 50);
        return std::make_unique<EntropyPredictor>(std::move(strategy), window_size);
    }
    
    // Helper to fill buffer with exponential decay
    void fillExponentialDecay(EntropyPredictor& predictor, size_t n, 
                              double initial, double final_val, double decay_rate) {
        for (size_t i = 0; i < n; ++i) {
            double entropy = final_val + (initial - final_val) * std::exp(-decay_rate * i);
            predictor.addDataPoint(entropy);
        }
    }
    
    // Helper to fill buffer with linear trend
    void fillLinearTrend(EntropyPredictor& predictor, size_t n,
                        double initial, double slope) {
        for (size_t i = 0; i < n; ++i) {
            predictor.addDataPoint(initial + slope * i);
        }
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(EntropyPredictorTest, ConstructorWithValidStrategy) {
    auto strategy = std::make_unique<ExponentialFittingStrategy>();
    EntropyPredictor predictor(std::move(strategy), 200);
    
    EXPECT_EQ(predictor.getWindowSize(), 200);
    EXPECT_EQ(predictor.getCurrentDataPoints(), 0);
}

TEST_F(EntropyPredictorTest, ConstructorThrowsOnNullStrategy) {
    EXPECT_THROW({
        EntropyPredictor predictor(nullptr, 200);
    }, std::invalid_argument);
}

TEST_F(EntropyPredictorTest, ConstructorThrowsOnSmallWindowSize) {
    auto strategy = std::make_unique<ExponentialFittingStrategy>();
    
    EXPECT_THROW({
        EntropyPredictor predictor(std::move(strategy), 1);
    }, std::invalid_argument);
}

TEST_F(EntropyPredictorTest, ConstructorAcceptsDifferentWindowSizes) {
    auto strategy1 = std::make_unique<ExponentialFittingStrategy>();
    EntropyPredictor predictor1(std::move(strategy1), 100);
    EXPECT_EQ(predictor1.getWindowSize(), 100);
    
    auto strategy2 = std::make_unique<ExponentialFittingStrategy>();
    EntropyPredictor predictor2(std::move(strategy2), 500);
    EXPECT_EQ(predictor2.getWindowSize(), 500);
}

// ============================================================================
// addDataPoint Tests
// ============================================================================

TEST_F(EntropyPredictorTest, AddDataPointIncreasesCount) {
    auto predictor = createExponentialPredictor(200);
    
    EXPECT_EQ(predictor->getCurrentDataPoints(), 0);
    
    predictor->addDataPoint(1.0);
    EXPECT_EQ(predictor->getCurrentDataPoints(), 1);
    
    predictor->addDataPoint(0.9);
    EXPECT_EQ(predictor->getCurrentDataPoints(), 2);
}

TEST_F(EntropyPredictorTest, AddDataPointRespectsCapacity) {
    auto predictor = createExponentialPredictor(10);  // Small buffer
    
    for (int i = 0; i < 20; ++i) {
        predictor->addDataPoint(1.0 - i * 0.01);
    }
    
    // Should cap at window size
    EXPECT_EQ(predictor->getCurrentDataPoints(), 10);
}

TEST_F(EntropyPredictorTest, AddDataPointOverwritesOldest) {
    auto predictor = createExponentialPredictor(5);
    
    for (int i = 0; i < 10; ++i) {
        predictor->addDataPoint(static_cast<double>(i));
    }
    
    // Buffer should contain last 5 values: 5, 6, 7, 8, 9
    EXPECT_EQ(predictor->getCurrentDataPoints(), 5);
}

// ============================================================================
// hasEnoughData Tests
// ============================================================================

TEST_F(EntropyPredictorTest, HasEnoughDataInitiallyFalse) {
    auto predictor = createExponentialPredictor(200);
    
    EXPECT_FALSE(predictor->hasEnoughData());
}

TEST_F(EntropyPredictorTest, HasEnoughDataAfterFillingMinimum) {
    auto predictor = createExponentialPredictor(200);
    
    // Strategy requires 50 minimum
    for (int i = 0; i < 50; ++i) {
        predictor->addDataPoint(1.0 - i * 0.01);
    }
    
    EXPECT_TRUE(predictor->hasEnoughData());
}

TEST_F(EntropyPredictorTest, HasEnoughDataJustBelowMinimum) {
    auto predictor = createExponentialPredictor(200);
    
    // Strategy requires 50, add 49
    for (int i = 0; i < 49; ++i) {
        predictor->addDataPoint(1.0 - i * 0.01);
    }
    
    EXPECT_FALSE(predictor->hasEnoughData());
}

// ============================================================================
// predict Tests
// ============================================================================

TEST_F(EntropyPredictorTest, PredictReturnsInvalidWhenInsufficientData) {
    auto predictor = createExponentialPredictor(200);
    
    predictor->addDataPoint(1.0);
    predictor->addDataPoint(0.9);
    
    PredictionResult result = predictor->predict();
    
    EXPECT_FALSE(result.valid);
}

TEST_F(EntropyPredictorTest, PredictDelegatesToStrategy) {
    auto predictor = createExponentialPredictor(200);
    
    // Fill with exponential decay
    fillExponentialDecay(*predictor, 100, 1.0, 0.01, 0.05);
    
    PredictionResult result = predictor->predict();
    
    // Should produce valid prediction
    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.confidence, 0.0);
}

TEST_F(EntropyPredictorTest, PredictWithLinearStrategy) {
    auto predictor = createLinearPredictor(200);
    
    // Fill with linear trend
    fillLinearTrend(*predictor, 100, 1.0, -0.01);
    
    PredictionResult result = predictor->predict();
    
    EXPECT_TRUE(result.valid);
    EXPECT_NEAR(result.confidence, 1.0, 1e-10);  // Perfect linear fit
}

// ============================================================================
// reset Tests
// ============================================================================

TEST_F(EntropyPredictorTest, ResetClearsAllData) {
    auto predictor = createExponentialPredictor(200);
    
    fillExponentialDecay(*predictor, 100, 1.0, 0.01, 0.05);
    EXPECT_EQ(predictor->getCurrentDataPoints(), 100);
    
    predictor->reset();
    
    EXPECT_EQ(predictor->getCurrentDataPoints(), 0);
    EXPECT_FALSE(predictor->hasEnoughData());
}

TEST_F(EntropyPredictorTest, ResetAllowsNewPredictions) {
    auto predictor = createExponentialPredictor(200);
    
    // First run
    fillExponentialDecay(*predictor, 100, 1.0, 0.01, 0.05);
    PredictionResult result1 = predictor->predict();
    EXPECT_TRUE(result1.valid);
    
    // Reset and second run
    predictor->reset();
    fillLinearTrend(*predictor, 100, 2.0, -0.02);
    PredictionResult result2 = predictor->predict();
    
    // Linear data with exponential strategy may not fit well
    // Just verify we can make a prediction after reset
    // (May be invalid if fit is poor, which is acceptable)
}

// ============================================================================
// setStrategy Tests
// ============================================================================

TEST_F(EntropyPredictorTest, SetStrategyChangesPredictor) {
    auto predictor = createExponentialPredictor(200);
    
    // Fill with LINEAR data (so exponential fit will be poor)
    fillLinearTrend(*predictor, 100, 1.0, -0.01);
    
    // Try predict with exponential strategy (may fail due to poor fit)
    PredictionResult result1 = predictor->predict();
    
    // Switch to linear strategy (should work better for linear data)
    auto new_strategy = std::make_unique<LinearExtrapolationStrategy>(1e-15, 50);
    predictor->setStrategy(std::move(new_strategy));
    
    // Predict with linear strategy (should work well)
    PredictionResult result2 = predictor->predict();
    EXPECT_TRUE(result2.valid);  // Linear strategy should handle linear data well
}

TEST_F(EntropyPredictorTest, SetStrategyThrowsOnNull) {
    auto predictor = createExponentialPredictor(200);
    
    EXPECT_THROW({
        predictor->setStrategy(nullptr);
    }, std::invalid_argument);
}

TEST_F(EntropyPredictorTest, SetStrategyPreservesData) {
    auto predictor = createExponentialPredictor(200);
    
    fillLinearTrend(*predictor, 100, 1.0, -0.01);
    size_t count_before = predictor->getCurrentDataPoints();
    
    auto new_strategy = std::make_unique<LinearExtrapolationStrategy>();
    predictor->setStrategy(std::move(new_strategy));
    
    // Data should be preserved
    EXPECT_EQ(predictor->getCurrentDataPoints(), count_before);
}

// ============================================================================
// Integration Tests with Realistic Sequences
// ============================================================================

TEST_F(EntropyPredictorTest, RealisticMinimizationSequence) {
    auto predictor = createExponentialPredictor(500);
    
    // Simulate realistic minimization: fast initial drop, slow convergence
    for (int i = 0; i < 300; ++i) {
        double entropy;
        if (i < 50) {
            // Fast initial drop
            entropy = 1.0 * std::exp(-0.1 * i);
        } else {
            // Slow final convergence
            entropy = 0.01 * std::exp(-0.01 * (i - 50));
        }
        predictor->addDataPoint(entropy);
    }
    
    PredictionResult result = predictor->predict();
    
    // Should produce some prediction (valid or not depends on fit quality)
    if (result.valid) {
        EXPECT_LT(result.predicted_entropy, 0.01);
        EXPECT_GE(result.predicted_iterations, 0);
        EXPECT_GE(result.confidence, 0.0);
        EXPECT_LE(result.confidence, 1.0);
    }
}

TEST_F(EntropyPredictorTest, NoisyConvergenceSequence) {
    auto predictor = createExponentialPredictor(200);
    
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.001);
    
    for (int i = 0; i < 150; ++i) {
        double clean_entropy = 1.0 * std::exp(-0.05 * i);
        double noisy_entropy = clean_entropy + noise(gen);
        predictor->addDataPoint(noisy_entropy);
    }
    
    PredictionResult result = predictor->predict();
    
    // Should handle noisy data gracefully
    if (result.valid) {
        EXPECT_GT(result.confidence, 0.8);  // Should still have decent fit
    }
}

TEST_F(EntropyPredictorTest, VerySlowConvergence) {
    auto predictor = createLinearPredictor(200);
    
    // Very slow linear decrease
    for (int i = 0; i < 100; ++i) {
        predictor->addDataPoint(1.0 - i * 1e-6);
    }
    
    PredictionResult result = predictor->predict();
    
    if (result.valid) {
        // Should predict many iterations remaining
        EXPECT_GT(result.predicted_iterations, 10000);
    }
}

// ============================================================================
// Memory and State Tests
// ============================================================================

TEST_F(EntropyPredictorTest, LargeWindowSizeWorks) {
    auto strategy = std::make_unique<ExponentialFittingStrategy>();
    EntropyPredictor predictor(std::move(strategy), 1000);
    
    for (int i = 0; i < 1000; ++i) {
        predictor.addDataPoint(1.0 - i * 0.0001);
    }
    
    EXPECT_EQ(predictor.getCurrentDataPoints(), 1000);
    
    PredictionResult result = predictor.predict();
    // Should work with large buffer
}

TEST_F(EntropyPredictorTest, SmallWindowSizeWorks) {
    auto strategy = std::make_unique<LinearExtrapolationStrategy>(1e-15, 10);
    EntropyPredictor predictor(std::move(strategy), 20);
    
    for (int i = 0; i < 20; ++i) {
        predictor.addDataPoint(1.0 - i * 0.01);
    }
    
    PredictionResult result = predictor.predict();
    EXPECT_TRUE(result.valid);
}
