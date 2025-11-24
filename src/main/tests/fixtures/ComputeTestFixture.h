#ifndef COMPUTE_TEST_FIXTURE_H_
#define COMPUTE_TEST_FIXTURE_H_

#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <random>
#include <cmath>

// Fixture for compute backend tests
class ComputeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator with fixed seed for reproducibility
        rng.seed(42);
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
    
    // Random number generator
    std::mt19937 rng;
    
    // Helper: Generate random double vector
    std::vector<double> generateRandomDoubles(size_t count, double min = -1.0, double max = 1.0) {
        std::uniform_real_distribution<double> dist(min, max);
        std::vector<double> result(count);
        for (auto& val : result) {
            val = dist(rng);
        }
        return result;
    }
    
    // Helper: Generate random complex vector
    std::vector<std::complex<double>> generateRandomComplex(size_t count) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::complex<double>> result(count);
        for (auto& val : result) {
            val = std::complex<double>(dist(rng), dist(rng));
        }
        return result;
    }
    
    // Helper: Compare floating-point values with tolerance
    bool almostEqual(double a, double b, double epsilon = 1e-10) {
        return std::abs(a - b) < epsilon;
    }
    
    // Helper: Compare complex values with tolerance
    bool almostEqual(std::complex<double> a, std::complex<double> b, double epsilon = 1e-10) {
        return std::abs(a - b) < epsilon;
    }
    
    // Helper: Compare vectors element-wise
    template<typename T>
    bool vectorsAlmostEqual(const std::vector<T>& a, const std::vector<T>& b, double epsilon = 1e-10) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!almostEqual(a[i], b[i], epsilon)) {
                return false;
            }
        }
        return true;
    }
};

#endif // COMPUTE_TEST_FIXTURE_H_
