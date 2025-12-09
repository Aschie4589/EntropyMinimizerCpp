#include <gtest/gtest.h>
#include "minimizer/state/circular_buffer.h"
#include "minimizer/stopping/stop_reason.h"
#include "minimizer/stopping/condition.h"
#include "minimizer/stopping/max_iterations_condition.h"
#include "minimizer/stopping/convergence_condition.h"
#include "minimizer/stopping/numerical_instability_condition.h"
#include "minimizer/stopping/target_entropy_condition.h"
#include "minimizer/stopping/conditions_checker.h"

using namespace entropy;

// ============================================================================
// CircularBuffer Tests
// ============================================================================

TEST(CircularBufferTest, ConstructorValidatesCapacity) {
    EXPECT_NO_THROW(CircularBuffer<double>(10));
    EXPECT_THROW(CircularBuffer<double>(0), std::invalid_argument);
}

TEST(CircularBufferTest, InitiallyEmpty) {
    CircularBuffer<double> buffer(5);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), 5);
    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.full());
}

TEST(CircularBufferTest, PushIncreasesSize) {
    CircularBuffer<double> buffer(3);
    buffer.push(1.0);
    EXPECT_EQ(buffer.size(), 1);
    EXPECT_FALSE(buffer.empty());
    EXPECT_FALSE(buffer.full());
    
    buffer.push(2.0);
    EXPECT_EQ(buffer.size(), 2);
    
    buffer.push(3.0);
    EXPECT_EQ(buffer.size(), 3);
    EXPECT_TRUE(buffer.full());
}

TEST(CircularBufferTest, PushOverwritesWhenFull) {
    CircularBuffer<double> buffer(3);
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.push(3.0);
    EXPECT_TRUE(buffer.full());
    EXPECT_EQ(buffer.oldest(), 1.0);
    
    // Overwrite oldest (1.0) with 4.0
    buffer.push(4.0);
    EXPECT_EQ(buffer.size(), 3);
    EXPECT_EQ(buffer.oldest(), 2.0);
    EXPECT_EQ(buffer.newest(), 4.0);
}

TEST(CircularBufferTest, GetReturnsCorrectElements) {
    CircularBuffer<double> buffer(5);
    buffer.push(10.0);
    buffer.push(20.0);
    buffer.push(30.0);
    
    EXPECT_EQ(buffer.get(0), 10.0);  // Oldest
    EXPECT_EQ(buffer.get(1), 20.0);
    EXPECT_EQ(buffer.get(2), 30.0);  // Newest
    EXPECT_THROW(buffer.get(3), std::out_of_range);
}

TEST(CircularBufferTest, GetAfterWrapAround) {
    CircularBuffer<double> buffer(3);
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.push(3.0);
    buffer.push(4.0);  // Overwrites 1.0
    buffer.push(5.0);  // Overwrites 2.0
    
    EXPECT_EQ(buffer.get(0), 3.0);  // Oldest
    EXPECT_EQ(buffer.get(1), 4.0);
    EXPECT_EQ(buffer.get(2), 5.0);  // Newest
}

TEST(CircularBufferTest, OldestAndNewest) {
    CircularBuffer<double> buffer(4);
    buffer.push(100.0);
    EXPECT_EQ(buffer.oldest(), 100.0);
    EXPECT_EQ(buffer.newest(), 100.0);
    
    buffer.push(200.0);
    buffer.push(300.0);
    EXPECT_EQ(buffer.oldest(), 100.0);
    EXPECT_EQ(buffer.newest(), 300.0);
}

TEST(CircularBufferTest, OldestAndNewestThrowWhenEmpty) {
    CircularBuffer<double> buffer(5);
    EXPECT_THROW(buffer.oldest(), std::logic_error);
    EXPECT_THROW(buffer.newest(), std::logic_error);
}

TEST(CircularBufferTest, Average) {
    CircularBuffer<double> buffer(5);
    buffer.push(10.0);
    buffer.push(20.0);
    buffer.push(30.0);
    
    EXPECT_DOUBLE_EQ(buffer.average(), 20.0);
    
    buffer.push(40.0);
    EXPECT_DOUBLE_EQ(buffer.average(), 25.0);
}

TEST(CircularBufferTest, AverageAfterWrapAround) {
    CircularBuffer<double> buffer(3);
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.push(3.0);
    EXPECT_DOUBLE_EQ(buffer.average(), 2.0);
    
    buffer.push(4.0);  // Buffer now: [2, 3, 4]
    EXPECT_DOUBLE_EQ(buffer.average(), 3.0);
}

TEST(CircularBufferTest, AverageThrowsWhenEmpty) {
    CircularBuffer<double> buffer(5);
    EXPECT_THROW(buffer.average(), std::logic_error);
}

TEST(CircularBufferTest, Clear) {
    CircularBuffer<double> buffer(5);
    buffer.push(1.0);
    buffer.push(2.0);
    buffer.push(3.0);
    
    buffer.clear();
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.full());
}

// ============================================================================
// StopReason Tests
// ============================================================================

TEST(StopReasonTest, ToStringConversion) {
    EXPECT_EQ(to_string(StopReason::CONTINUE), "CONTINUE");
    EXPECT_EQ(to_string(StopReason::CONVERGED), "CONVERGED");
    EXPECT_EQ(to_string(StopReason::MAX_ITERATIONS), "MAX_ITERATIONS");
    EXPECT_EQ(to_string(StopReason::NUMERICAL_INSTABILITY), "NUMERICAL_INSTABILITY");
    EXPECT_EQ(to_string(StopReason::TARGET_REACHED), "TARGET_REACHED");
}

// ============================================================================
// MaxIterationsCondition Tests
// ============================================================================

TEST(MaxIterationsConditionTest, ContinuesBeforeLimit) {
    MaxIterationsCondition condition(100);
    
    EXPECT_EQ(condition.check(0, 1.0, 2.0), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(50, 1.0, 2.0), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(99, 1.0, 2.0), StopReason::CONTINUE);
}

TEST(MaxIterationsConditionTest, StopsAtLimit) {
    MaxIterationsCondition condition(100);
    
    EXPECT_EQ(condition.check(100, 1.0, 2.0), StopReason::MAX_ITERATIONS);
    EXPECT_EQ(condition.check(101, 1.0, 2.0), StopReason::MAX_ITERATIONS);
}

TEST(MaxIterationsConditionTest, Description) {
    MaxIterationsCondition condition(1000);
    std::string desc = condition.description();
    EXPECT_NE(desc.find("1000"), std::string::npos);
}

// ============================================================================
// ConvergenceCondition Tests
// ============================================================================

TEST(ConvergenceConditionTest, ContinuesBeforeWindowFull) {
    ConvergenceCondition condition(5, 1e-10);
    
    EXPECT_EQ(condition.check(0, 1.0, 1.1), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(1, 0.9, 1.0), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(2, 0.8, 0.9), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(3, 0.7, 0.8), StopReason::CONTINUE);
}

TEST(ConvergenceConditionTest, DetectsConvergence) {
    ConvergenceCondition condition(5, 1e-3);
    
    // Add sequence with small improvements
    // Total improvement over window = 0.001 / 5 = 0.0002 < 1e-3
    condition.check(0, 1.000000, 1.000001);
    condition.check(1, 0.999999, 1.000000);
    condition.check(2, 0.999998, 0.999999);
    condition.check(3, 0.999997, 0.999998);
    
    // Window now full, avg improvement = (1.000000 - 0.999997) / 5 = 0.0006 < 1e-3
    EXPECT_EQ(condition.check(4, 0.999996, 0.999997), StopReason::CONVERGED);
}

TEST(ConvergenceConditionTest, ContinuesWithLargeImprovements) {
    ConvergenceCondition condition(3, 0.01);
    
    condition.check(0, 1.0, 1.1);
    condition.check(1, 0.9, 1.0);
    
    // Large improvement: (1.0 - 0.8) / 3 = 0.0667 > 0.01
    EXPECT_EQ(condition.check(2, 0.8, 0.9), StopReason::CONTINUE);
}

TEST(ConvergenceConditionTest, Reset) {
    ConvergenceCondition condition(3, 1e-10);
    
    condition.check(0, 1.0, 1.1);
    condition.check(1, 0.9, 1.0);
    
    condition.reset();
    
    // After reset, should start fresh
    EXPECT_EQ(condition.check(0, 1.0, 1.1), StopReason::CONTINUE);
}

// ============================================================================
// NumericalInstabilityCondition Tests
// ============================================================================

TEST(NumericalInstabilityConditionTest, ContinuesWithDecreasingEntropy) {
    NumericalInstabilityCondition condition(5);
    
    EXPECT_EQ(condition.check(0, 1.0, 1.1), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(1, 0.9, 1.0), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(2, 0.8, 0.9), StopReason::CONTINUE);
}

TEST(NumericalInstabilityConditionTest, DetectsIncreasingEntropy) {
    NumericalInstabilityCondition condition(5);
    
    condition.check(0, 1.0, 1.1);
    condition.check(1, 0.9, 1.0);
    
    // Entropy increases: 0.95 > 0.9
    EXPECT_EQ(condition.check(2, 0.95, 0.9), StopReason::NUMERICAL_INSTABILITY);
}

TEST(NumericalInstabilityConditionTest, DetectsInstabilityInHistory) {
    NumericalInstabilityCondition condition(5);
    
    condition.check(0, 1.0, 1.1);
    condition.check(1, 0.9, 1.0);
    condition.check(2, 1.1, 0.9);  // Increase here
    
    // Should detect the increase in history
    EXPECT_EQ(condition.check(3, 0.8, 1.1), StopReason::NUMERICAL_INSTABILITY);
}

TEST(NumericalInstabilityConditionTest, Reset) {
    NumericalInstabilityCondition condition(3);
    
    condition.check(0, 1.0, 1.1);
    condition.reset();
    
    // After reset, should accept increasing entropy initially
    EXPECT_EQ(condition.check(0, 1.0, 0.9), StopReason::CONTINUE);
}

// ============================================================================
// TargetEntropyCondition Tests
// ============================================================================

TEST(TargetEntropyConditionTest, ContinuesAboveTarget) {
    TargetEntropyCondition condition(0.5);
    
    EXPECT_EQ(condition.check(0, 1.0, 1.1), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(1, 0.8, 1.0), StopReason::CONTINUE);
    EXPECT_EQ(condition.check(2, 0.6, 0.8), StopReason::CONTINUE);
}

TEST(TargetEntropyConditionTest, StopsAtTarget) {
    TargetEntropyCondition condition(0.5);
    
    EXPECT_EQ(condition.check(0, 0.5, 0.6), StopReason::TARGET_REACHED);
}

TEST(TargetEntropyConditionTest, StopsBelowTarget) {
    TargetEntropyCondition condition(0.5);
    
    EXPECT_EQ(condition.check(0, 0.3, 0.6), StopReason::TARGET_REACHED);
}

TEST(TargetEntropyConditionTest, ToleranceHandling) {
    TargetEntropyCondition condition(0.5, 1e-6);
    
    // Slightly above target but within tolerance
    EXPECT_EQ(condition.check(0, 0.5 + 0.5e-6, 0.6), StopReason::TARGET_REACHED);
    
    // Above tolerance
    EXPECT_EQ(condition.check(1, 0.5 + 2e-6, 0.6), StopReason::CONTINUE);
}

// ============================================================================
// ConditionsChecker Tests
// ============================================================================

TEST(ConditionsCheckerTest, EmptyCheckerContinues) {
    ConditionsChecker checker;
    EXPECT_EQ(checker.size(), 0);
    EXPECT_EQ(checker.checkStoppingConditions(0, 1.0, 1.1), StopReason::CONTINUE);
}

TEST(ConditionsCheckerTest, SingleCondition) {
    ConditionsChecker checker;
    checker.addCondition(std::make_unique<MaxIterationsCondition>(10));
    
    EXPECT_EQ(checker.size(), 1);
    EXPECT_EQ(checker.checkStoppingConditions(5, 1.0, 1.1), StopReason::CONTINUE);
    EXPECT_EQ(checker.checkStoppingConditions(10, 1.0, 1.1), StopReason::MAX_ITERATIONS);
}

TEST(ConditionsCheckerTest, MultipleConditionsPriority) {
    ConditionsChecker checker;
    
    // Add conditions in priority order
    checker.addCondition(std::make_unique<MaxIterationsCondition>(100));
    checker.addCondition(std::make_unique<TargetEntropyCondition>(0.5));
    
    EXPECT_EQ(checker.size(), 2);
    
    // At iteration 100, both conditions would trigger
    // MaxIterations should win (added first = higher priority)
    EXPECT_EQ(checker.checkStoppingConditions(100, 0.3, 0.6), StopReason::MAX_ITERATIONS);
    
    // Below max iterations, target entropy triggers
    EXPECT_EQ(checker.checkStoppingConditions(50, 0.3, 0.6), StopReason::TARGET_REACHED);
}

TEST(ConditionsCheckerTest, Reset) {
    ConditionsChecker checker;
    auto* conv_cond = new ConvergenceCondition(3, 1e-10);
    checker.addCondition(std::unique_ptr<Condition>(conv_cond));
    
    checker.checkStoppingConditions(0, 1.0, 1.1);
    checker.checkStoppingConditions(1, 0.9, 1.0);
    
    checker.reset();
    
    // After reset, convergence buffer should be cleared
    EXPECT_EQ(checker.checkStoppingConditions(0, 1.0, 1.1), StopReason::CONTINUE);
}

TEST(ConditionsCheckerTest, ComplexScenario) {
    ConditionsChecker checker;
    
    // Setup realistic condition set
    checker.addCondition(std::make_unique<NumericalInstabilityCondition>(20));
    checker.addCondition(std::make_unique<MaxIterationsCondition>(1000));
    checker.addCondition(std::make_unique<ConvergenceCondition>(20, 1e-15));
    checker.addCondition(std::make_unique<TargetEntropyCondition>(0.0));
    
    EXPECT_EQ(checker.size(), 4);
    
    // Simulate decreasing entropy sequence
    double entropy = 1.0;
    for (size_t i = 0; i < 30; ++i) {
        double prev_entropy = entropy;
        entropy *= 0.95;  // Decrease by 5% each iteration
        
        StopReason reason = checker.checkStoppingConditions(i, entropy, prev_entropy);
        
        // Should continue for first 30 iterations with 5% improvement
        EXPECT_EQ(reason, StopReason::CONTINUE);
    }
}

TEST(ConditionsCheckerTest, Description) {
    ConditionsChecker checker;
    checker.addCondition(std::make_unique<MaxIterationsCondition>(100));
    checker.addCondition(std::make_unique<ConvergenceCondition>(20, 1e-15));
    
    std::string desc = checker.description();
    EXPECT_NE(desc.find("2 conditions"), std::string::npos);
    EXPECT_NE(desc.find("MaxIterations"), std::string::npos);
    EXPECT_NE(desc.find("Convergence"), std::string::npos);
}
