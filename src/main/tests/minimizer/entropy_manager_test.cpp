#include <gtest/gtest.h>
#include "minimizer/state/entropy_manager.h"
#include <cmath>
#include <limits>

using namespace entropy;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(EntropyManagerTest, ConstructorValidatesEpsilon) {
    // Valid epsilon values
    EXPECT_NO_THROW(EntropyManager(0.001, 4));
    EXPECT_NO_THROW(EntropyManager(0.5, 4));
    EXPECT_NO_THROW(EntropyManager(0.999, 4));
    
    // Invalid epsilon: ≤ 0
    EXPECT_THROW(EntropyManager(0.0, 4), std::invalid_argument);
    EXPECT_THROW(EntropyManager(-0.1, 4), std::invalid_argument);
    
    // Invalid epsilon: ≥ 1
    EXPECT_THROW(EntropyManager(1.0, 4), std::invalid_argument);
    EXPECT_THROW(EntropyManager(1.1, 4), std::invalid_argument);
}

TEST(EntropyManagerTest, ConstructorValidatesDimension) {
    // Valid dimensions
    EXPECT_NO_THROW(EntropyManager(0.001, 1));
    EXPECT_NO_THROW(EntropyManager(0.001, 4));
    EXPECT_NO_THROW(EntropyManager(0.001, 100));
    
    // Invalid dimensions
    EXPECT_THROW(EntropyManager(0.001, 0), std::invalid_argument);
    EXPECT_THROW(EntropyManager(0.001, -1), std::invalid_argument);
}

TEST(EntropyManagerTest, ConstructorPrecomputesConstants) {
    double epsilon = 0.001;
    int dim = 4;
    
    EntropyManager manager(epsilon, dim);
    
    // Verify binary entropy is computed
    double expected_bin_entropy = -epsilon * std::log(epsilon) 
                                  - (1.0 - epsilon) * std::log(1.0 - epsilon);
    EXPECT_NEAR(manager.getBinaryEntropy(), expected_bin_entropy, 1e-15);
    
    // Verify epsilon correction
    double expected_correction = epsilon * std::log(static_cast<double>(dim));
    EXPECT_NEAR(manager.getEpsilonCorrection(), expected_correction, 1e-15);
    
    // Verify estimated error
    double expected_error = expected_bin_entropy / (2.0 * (1.0 - epsilon));
    EXPECT_NEAR(manager.getEstimatedError(), expected_error, 1e-15);
}

TEST(EntropyManagerTest, GettersReturnConfigValues) {
    double epsilon = 0.002;
    int dim = 8;
    
    EntropyManager manager(epsilon, dim);
    
    EXPECT_DOUBLE_EQ(manager.getEpsilon(), epsilon);
    EXPECT_EQ(manager.getSystemDimension(), dim);
}

// ============================================================================
// Binary Entropy Tests
// ============================================================================

TEST(EntropyManagerTest, BinaryEntropyAtBoundaries) {
    // H_bin(ε) should be 0 at ε=0 and ε=1
    EntropyManager manager1(1e-16, 4);  // Very small ε
    EXPECT_NEAR(manager1.getBinaryEntropy(), 0.0, 1e-10);
    
    EntropyManager manager2(1.0 - 1e-16, 4);  // ε very close to 1
    EXPECT_NEAR(manager2.getBinaryEntropy(), 0.0, 1e-10);
}

TEST(EntropyManagerTest, BinaryEntropySymmetry) {
    // H_bin(ε) should be symmetric: H_bin(ε) = H_bin(1-ε)
    double eps1 = 0.3;
    double eps2 = 0.7;
    
    EntropyManager manager1(eps1, 4);
    EntropyManager manager2(eps2, 4);
    
    EXPECT_NEAR(manager1.getBinaryEntropy(), manager2.getBinaryEntropy(), 1e-15);
}

TEST(EntropyManagerTest, BinaryEntropyMaximumAtHalf) {
    // H_bin(ε) is maximized at ε = 0.5
    EntropyManager manager_half(0.5, 4);
    EntropyManager manager_quarter(0.25, 4);
    EntropyManager manager_three_quarter(0.75, 4);
    
    double h_half = manager_half.getBinaryEntropy();
    double h_quarter = manager_quarter.getBinaryEntropy();
    double h_three_quarter = manager_three_quarter.getBinaryEntropy();
    
    EXPECT_GT(h_half, h_quarter);
    EXPECT_GT(h_half, h_three_quarter);
    
    // At ε=0.5: H_bin(0.5) = -0.5·log(0.5) - 0.5·log(0.5) = log(2)
    EXPECT_NEAR(h_half, std::log(2.0), 1e-15);
}

// ============================================================================
// Update and Query Tests
// ============================================================================

TEST(EntropyManagerTest, UpdateEntropyTriggersRecomputation) {
    double epsilon = 0.001;
    int dim = 4;
    EntropyManager manager(epsilon, dim);
    
    // Initially, epsilon entropy is 0
    EXPECT_DOUBLE_EQ(manager.getEpsilonEntropy(), 0.0);
    EXPECT_DOUBLE_EQ(manager.getEstimatedEntropy(), 0.0);
    
    // Update with new value
    double eps_entropy = 0.5;
    manager.updateEpsilonEntropy(eps_entropy);
    
    // Verify epsilon entropy is updated
    EXPECT_DOUBLE_EQ(manager.getEpsilonEntropy(), eps_entropy);
    
    // Verify estimated entropy is computed
    EXPECT_NE(manager.getEstimatedEntropy(), 0.0);
}

TEST(EntropyManagerTest, EstimatedEntropyFormula) {
    double epsilon = 0.01;
    int dim = 4;
    EntropyManager manager(epsilon, dim);
    
    double eps_entropy = 1.2;
    manager.updateEpsilonEntropy(eps_entropy);
    
    // Manually compute expected value
    double bin_ent = -epsilon * std::log(epsilon) 
                     - (1.0 - epsilon) * std::log(1.0 - epsilon);
    double eps_corr = epsilon * std::log(static_cast<double>(dim));
    double error = bin_ent / (2.0 * (1.0 - epsilon));
    double expected = (eps_entropy - eps_corr) / (1.0 - epsilon) - error;
    
    EXPECT_NEAR(manager.getEstimatedEntropy(), expected, 1e-12);
}

TEST(EntropyManagerTest, MultipleUpdates) {
    EntropyManager manager(0.001, 4);
    
    // Update multiple times
    manager.updateEpsilonEntropy(1.0);
    double est1 = manager.getEstimatedEntropy();
    
    manager.updateEpsilonEntropy(0.8);
    double est2 = manager.getEstimatedEntropy();
    
    manager.updateEpsilonEntropy(0.6);
    double est3 = manager.getEstimatedEntropy();
    
    // Each update should produce different estimated entropy
    EXPECT_NE(est1, est2);
    EXPECT_NE(est2, est3);
    EXPECT_NE(est1, est3);
    
    // Verify monotonicity: smaller eps_entropy → smaller estimated_entropy
    EXPECT_GT(est1, est2);
    EXPECT_GT(est2, est3);
}

// ============================================================================
// Bounds Tests
// ============================================================================

TEST(EntropyManagerTest, BoundsSatisfyInequality) {
    EntropyManager manager(0.01, 4);
    
    double eps_entropy = 1.0;
    manager.updateEpsilonEntropy(eps_entropy);
    
    auto [lower, upper] = manager.getEntropyBounds();
    
    // Bounds should be symmetric around estimated entropy: estimated ± error
    double estimated = manager.getEstimatedEntropy();
    double error = manager.getEstimatedError();
    
    EXPECT_DOUBLE_EQ(lower, estimated - error);
    EXPECT_DOUBLE_EQ(upper, estimated + error);
    
    // Bounds should satisfy lower ≤ estimated ≤ upper
    EXPECT_LE(lower, estimated);
    EXPECT_LE(estimated, upper);
}

TEST(EntropyManagerTest, BoundsUpdateWithEntropy) {
    EntropyManager manager(0.001, 4);
    
    manager.updateEpsilonEntropy(1.0);
    auto [lower1, upper1] = manager.getEntropyBounds();
    
    manager.updateEpsilonEntropy(0.5);
    auto [lower2, upper2] = manager.getEntropyBounds();
    
    // Bounds should change with different epsilon entropy
    EXPECT_NE(lower1, lower2);
    EXPECT_NE(upper1, upper2);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST(EntropyManagerTest, ResetClearsState) {
    EntropyManager manager(0.001, 4);
    
    // Set some entropy
    manager.updateEpsilonEntropy(1.0);
    EXPECT_NE(manager.getEpsilonEntropy(), 0.0);
    
    // Reset
    manager.reset();
    
    // State should be cleared
    EXPECT_DOUBLE_EQ(manager.getEpsilonEntropy(), 0.0);
    EXPECT_DOUBLE_EQ(manager.getEstimatedEntropy(), 0.0);
    
    auto [lower, upper] = manager.getEntropyBounds();
    EXPECT_DOUBLE_EQ(lower, 0.0);
    EXPECT_DOUBLE_EQ(upper, 0.0);
}

TEST(EntropyManagerTest, ResetPreservesConfiguration) {
    double epsilon = 0.005;
    int dim = 8;
    EntropyManager manager(epsilon, dim);
    
    // Set and reset
    manager.updateEpsilonEntropy(1.0);
    manager.reset();
    
    // Configuration should be preserved
    EXPECT_DOUBLE_EQ(manager.getEpsilon(), epsilon);
    EXPECT_EQ(manager.getSystemDimension(), dim);
    EXPECT_NE(manager.getBinaryEntropy(), 0.0);  // Precomputed constant preserved
    EXPECT_NE(manager.getEpsilonCorrection(), 0.0);  // Precomputed constant preserved
}

TEST(EntropyManagerTest, CanUpdateAfterReset) {
    EntropyManager manager(0.001, 4);
    
    manager.updateEpsilonEntropy(1.0);
    manager.reset();
    manager.updateEpsilonEntropy(0.8);
    
    EXPECT_DOUBLE_EQ(manager.getEpsilonEntropy(), 0.8);
    EXPECT_NE(manager.getEstimatedEntropy(), 0.0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(EntropyManagerTest, VerySmallEpsilon) {
    // When ε → 0, correction terms should be very small
    double epsilon = 1e-10;
    EntropyManager manager(epsilon, 4);
    
    double eps_entropy = 1.0;
    manager.updateEpsilonEntropy(eps_entropy);
    
    // With very small ε, estimated ≈ epsilon_entropy
    EXPECT_NEAR(manager.getEstimatedEntropy(), eps_entropy, 0.01);
    
    // Error should be very small
    EXPECT_NEAR(manager.getEstimatedError(), 0.0, 1e-8);
}

TEST(EntropyManagerTest, LargeEpsilon) {
    // When ε → 1, formulas have edge cases
    double epsilon = 0.99;
    EntropyManager manager(epsilon, 4);
    
    double eps_entropy = 1.0;
    manager.updateEpsilonEntropy(eps_entropy);
    
    // Should not crash or produce NaN
    EXPECT_FALSE(std::isnan(manager.getEstimatedEntropy()));
    EXPECT_FALSE(std::isinf(manager.getEstimatedEntropy()));
}

TEST(EntropyManagerTest, ZeroEpsilonEntropy) {
    EntropyManager manager(0.001, 4);
    
    manager.updateEpsilonEntropy(0.0);
    
    // Should handle zero entropy gracefully
    EXPECT_DOUBLE_EQ(manager.getEpsilonEntropy(), 0.0);
    EXPECT_FALSE(std::isnan(manager.getEstimatedEntropy()));
}

TEST(EntropyManagerTest, NegativeEpsilonEntropy) {
    // While unusual, negative entropy can occur numerically
    EntropyManager manager(0.001, 4);
    
    manager.updateEpsilonEntropy(-0.01);
    
    // Should handle without crashing
    EXPECT_DOUBLE_EQ(manager.getEpsilonEntropy(), -0.01);
    EXPECT_FALSE(std::isnan(manager.getEstimatedEntropy()));
}

// ============================================================================
// Realistic Scenario Tests
// ============================================================================

TEST(EntropyManagerTest, TypicalPhase1Parameters) {
    // Typical parameters from phase1
    double epsilon = 0.001;
    int dim = 4;  // For 2-qubit system
    
    EntropyManager manager(epsilon, dim);
    
    // Typical entropy value
    double eps_entropy = 0.123456;
    manager.updateEpsilonEntropy(eps_entropy);
    
    // Verify all computations work
    EXPECT_GT(manager.getEstimatedEntropy(), 0.0);
    
    // With epsilon=0.001, error should be small but not necessarily < 1e-3
    // The actual error depends on the binary entropy
    EXPECT_GT(manager.getEstimatedError(), 0.0);
    EXPECT_LT(manager.getEstimatedError(), 0.01);  // Should be reasonably small
    
    auto [lower, upper] = manager.getEntropyBounds();
    
    // Bounds should be symmetric: estimated ± error
    double estimated = manager.getEstimatedEntropy();
    double error = manager.getEstimatedError();
    EXPECT_DOUBLE_EQ(lower, estimated - error);
    EXPECT_DOUBLE_EQ(upper, estimated + error);
}

TEST(EntropyManagerTest, DecreasingEntropySequence) {
    // Simulate minimization: entropy decreases over iterations
    EntropyManager manager(0.001, 4);
    
    std::vector<double> entropies = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5};
    std::vector<double> estimates;
    
    for (double ent : entropies) {
        manager.updateEpsilonEntropy(ent);
        estimates.push_back(manager.getEstimatedEntropy());
    }
    
    // Estimated entropy should also decrease monotonically
    for (size_t i = 1; i < estimates.size(); ++i) {
        EXPECT_LT(estimates[i], estimates[i-1]);
    }
}

TEST(EntropyManagerTest, DifferentDimensions) {
    double epsilon = 0.001;
    double eps_entropy = 1.0;
    
    // Test various system dimensions
    for (int dim : {2, 4, 8, 16, 64}) {
        EntropyManager manager(epsilon, dim);
        manager.updateEpsilonEntropy(eps_entropy);
        
        // Larger dimension → larger correction
        double correction = manager.getEpsilonCorrection();
        EXPECT_NEAR(correction, epsilon * std::log(static_cast<double>(dim)), 1e-15);
    }
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST(EntropyManagerTest, ConsistentWithOldImplementation) {
    // Values extracted from tensor_entropy.cu example
    double epsilon = 0.001;
    int d = 4;  // System dimension
    double eps_entropy = 0.123456;  // Example value
    
    EntropyManager manager(epsilon, d);
    manager.updateEpsilonEntropy(eps_entropy);
    
    // Manually compute using old formulas
    double bin_ent = -epsilon * std::log(epsilon) 
                     - (1.0 - epsilon) * std::log(1.0 - epsilon);
    double eps_log_dim = epsilon * std::log(static_cast<double>(d));
    double one_minus_eps = 1.0 - epsilon;
    double corrected = (eps_entropy - eps_log_dim) / one_minus_eps;
    double error = bin_ent / (2.0 * one_minus_eps);
    double expected_est = corrected - error;
    
    EXPECT_NEAR(manager.getEstimatedEntropy(), expected_est, 1e-12);
    EXPECT_NEAR(manager.getEstimatedError(), error, 1e-12);
}
