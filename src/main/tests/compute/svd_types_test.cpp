#include <gtest/gtest.h>
#include <set>
#include "compute/linalg/SVDTypes.h"

// =============================================================================
// SVDVectors Enum Tests
// =============================================================================

TEST(SVDVectorsTest, EnumValues) {
    // Test that enum values are distinct
    EXPECT_NE(SVDVectors::ALL, SVDVectors::THIN);
    EXPECT_NE(SVDVectors::ALL, SVDVectors::NONE);
    EXPECT_NE(SVDVectors::THIN, SVDVectors::NONE);
}

TEST(SVDVectorsTest, Assignment) {
    SVDVectors v1 = SVDVectors::ALL;
    SVDVectors v2 = SVDVectors::THIN;
    SVDVectors v3 = SVDVectors::NONE;
    
    EXPECT_EQ(v1, SVDVectors::ALL);
    EXPECT_EQ(v2, SVDVectors::THIN);
    EXPECT_EQ(v3, SVDVectors::NONE);
}

// =============================================================================
// SVDAlgorithm Enum Tests
// =============================================================================

TEST(SVDAlgorithmTest, EnumValues) {
    // Test that enum values are distinct
    std::set<SVDAlgorithm> algorithms = {
        SVDAlgorithm::AUTO,
        SVDAlgorithm::QR,
        SVDAlgorithm::POLAR,
        SVDAlgorithm::RANDOMIZED
    };
    
    EXPECT_EQ(algorithms.size(), 4);  // All distinct
}

TEST(SVDAlgorithmTest, Assignment) {
    SVDAlgorithm a1 = SVDAlgorithm::AUTO;
    SVDAlgorithm a2 = SVDAlgorithm::RANDOMIZED;
    
    EXPECT_EQ(a1, SVDAlgorithm::AUTO);
    EXPECT_EQ(a2, SVDAlgorithm::RANDOMIZED);
}

// =============================================================================
// SVDSpec Struct Tests
// =============================================================================

TEST(SVDSpecTest, DefaultConstructor) {
    SVDSpec spec;
    
    // Check default values
    EXPECT_EQ(spec.lvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.rvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::AUTO);
    EXPECT_EQ(spec.rank, -1);
    EXPECT_EQ(spec.oversampling, 10);
    EXPECT_EQ(spec.power_iterations, 2);
    EXPECT_DOUBLE_EQ(spec.tolerance, 1e-7);
    EXPECT_EQ(spec.max_sweeps, 100);
}

TEST(SVDSpecTest, ConvenienceConstructor) {
    SVDSpec spec1(SVDVectors::ALL, SVDVectors::ALL);
    EXPECT_EQ(spec1.lvectors, SVDVectors::ALL);
    EXPECT_EQ(spec1.rvectors, SVDVectors::ALL);
    EXPECT_EQ(spec1.algorithm, SVDAlgorithm::AUTO);
    
    SVDSpec spec2(SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::RANDOMIZED);
    EXPECT_EQ(spec2.lvectors, SVDVectors::THIN);
    EXPECT_EQ(spec2.rvectors, SVDVectors::THIN);
    EXPECT_EQ(spec2.algorithm, SVDAlgorithm::RANDOMIZED);
    
    SVDSpec spec3(SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::QR);
    EXPECT_EQ(spec3.lvectors, SVDVectors::NONE);
    EXPECT_EQ(spec3.rvectors, SVDVectors::NONE);
    EXPECT_EQ(spec3.algorithm, SVDAlgorithm::QR);
}

TEST(SVDSpecTest, MemberModification) {
    SVDSpec spec;
    
    // Modify vectors
    spec.lvectors = SVDVectors::ALL;
    spec.rvectors = SVDVectors::ALL;
    EXPECT_EQ(spec.lvectors, SVDVectors::ALL);
    EXPECT_EQ(spec.rvectors, SVDVectors::ALL);
    
    // Modify algorithm
    spec.algorithm = SVDAlgorithm::POLAR;
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::POLAR);
    
    // Modify randomized SVD parameters
    spec.rank = 50;
    spec.oversampling = 20;
    spec.power_iterations = 2;
    EXPECT_EQ(spec.rank, 50);
    EXPECT_EQ(spec.oversampling, 20);
    EXPECT_EQ(spec.power_iterations, 2);
    
    // Modify tolerance parameters (still useful for future algorithms)
    spec.tolerance = 1e-9;
    spec.max_sweeps = 200;
    EXPECT_DOUBLE_EQ(spec.tolerance, 1e-9);
    EXPECT_EQ(spec.max_sweeps, 200);
}

TEST(SVDSpecTest, CopyConstructor) {
    SVDSpec spec1;
    spec1.lvectors = SVDVectors::ALL;
    spec1.rvectors = SVDVectors::ALL;
    spec1.algorithm = SVDAlgorithm::RANDOMIZED;
    spec1.rank = 100;
    spec1.tolerance = 1e-10;
    
    SVDSpec spec2(spec1);
    
    EXPECT_EQ(spec2.lvectors, SVDVectors::ALL);
    EXPECT_EQ(spec2.rvectors, SVDVectors::ALL);
    EXPECT_EQ(spec2.algorithm, SVDAlgorithm::RANDOMIZED);
    EXPECT_EQ(spec2.rank, 100);
    EXPECT_DOUBLE_EQ(spec2.tolerance, 1e-10);
}

TEST(SVDSpecTest, Assignment) {
    SVDSpec spec1;
    spec1.lvectors = SVDVectors::NONE;
    spec1.rvectors = SVDVectors::NONE;
    spec1.algorithm = SVDAlgorithm::QR;
    spec1.max_sweeps = 500;
    
    SVDSpec spec2;
    spec2 = spec1;
    
    EXPECT_EQ(spec2.lvectors, SVDVectors::NONE);
    EXPECT_EQ(spec2.rvectors, SVDVectors::NONE);
    EXPECT_EQ(spec2.algorithm, SVDAlgorithm::QR);
    EXPECT_EQ(spec2.max_sweeps, 500);
}

// =============================================================================
// Typical Usage Patterns
// =============================================================================

TEST(SVDSpecTest, UsagePattern_SingularValuesOnly) {
    // Common case: only need singular values, not vectors
    SVDSpec spec(SVDVectors::NONE, SVDVectors::NONE, SVDAlgorithm::AUTO);
    
    EXPECT_EQ(spec.lvectors, SVDVectors::NONE);
    EXPECT_EQ(spec.rvectors, SVDVectors::NONE);
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::AUTO);
}

TEST(SVDSpecTest, UsagePattern_EconomySVD) {
    // Common case: economy-size SVD (thin U and V)
    SVDSpec spec(SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::QR);
    
    EXPECT_EQ(spec.lvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.rvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::QR);
}

TEST(SVDSpecTest, UsagePattern_FullSVD) {
    // Full SVD with all singular vectors
    SVDSpec spec(SVDVectors::ALL, SVDVectors::ALL, SVDAlgorithm::QR);
    
    EXPECT_EQ(spec.lvectors, SVDVectors::ALL);
    EXPECT_EQ(spec.rvectors, SVDVectors::ALL);
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::QR);
}

TEST(SVDSpecTest, UsagePattern_LowRankApproximation) {
    // Randomized SVD for low-rank approximation
    SVDSpec spec(SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::RANDOMIZED);
    spec.rank = 50;
    spec.oversampling = 10;
    spec.power_iterations = 2;
    
    EXPECT_EQ(spec.lvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.rvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::RANDOMIZED);
    EXPECT_EQ(spec.rank, 50);
    EXPECT_EQ(spec.oversampling, 10);
    EXPECT_EQ(spec.power_iterations, 2);
}

TEST(SVDSpecTest, UsagePattern_HighAccuracy) {
    // Polar SVD for high accuracy
    SVDSpec spec(SVDVectors::THIN, SVDVectors::THIN, SVDAlgorithm::POLAR);
    
    EXPECT_EQ(spec.lvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.rvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::POLAR);
}

TEST(SVDSpecTest, UsagePattern_DefaultAuto) {
    // Let backend choose everything
    SVDSpec spec;
    
    // Sensible defaults
    EXPECT_EQ(spec.lvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.rvectors, SVDVectors::THIN);
    EXPECT_EQ(spec.algorithm, SVDAlgorithm::AUTO);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(SVDSpecTest, EdgeCase_NegativeRank) {
    // rank = -1 means full rank (no truncation)
    SVDSpec spec;
    spec.rank = -1;
    
    EXPECT_EQ(spec.rank, -1);
}

TEST(SVDSpecTest, EdgeCase_ZeroOversampling) {
    // Oversampling = 0 is valid (exact rank, no safety margin)
    SVDSpec spec;
    spec.oversampling = 0;
    
    EXPECT_EQ(spec.oversampling, 0);
}

TEST(SVDSpecTest, EdgeCase_ZeroPowerIterations) {
    // Power iterations = 0 is valid (no power method refinement)
    SVDSpec spec;
    spec.power_iterations = 0;
    
    EXPECT_EQ(spec.power_iterations, 0);
}

TEST(SVDSpecTest, EdgeCase_VerySmallTolerance) {
    // Very tight tolerance for Jacobi
    SVDSpec spec;
    spec.tolerance = 1e-15;
    
    EXPECT_DOUBLE_EQ(spec.tolerance, 1e-15);
}

TEST(SVDSpecTest, EdgeCase_VeryLargeSweeps) {
    // Many sweeps allowed
    SVDSpec spec;
    spec.max_sweeps = 10000;
    
    EXPECT_EQ(spec.max_sweeps, 10000);
}
