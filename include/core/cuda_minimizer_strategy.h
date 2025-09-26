#pragma once
#include "common_includes.h"

/*

Define a STRATEGY for the CudaMinimizer.
This is a strategy pattern that allows to define different strategies for the minimization algorithm.
The class CudaMinimizer has a pointer to a MinimizationStrategy object.
Whenever CudaMinimizer needs to do something strategy-dependent, it calls the appropriate method of the strategy.
CudaMinimizer delegates the responsibility of the specific minimization algorithm to the strategy object.

*/


template<typename T>
class CudaMinimizer; // Forward declaration

// Minimization Strategy Class
template<typename T>
class MinimizationStrategy {
public:
    virtual ~MinimizationStrategy() = default;
    
    // Pure virtual methods that strategies must implement
    virtual cudaError_t initialize(CudaMinimizer<T>* minimizer) = 0;
    virtual cudaError_t stepAlgorithm(CudaMinimizer<T>* minimizer) = 0;
    virtual cudaError_t step(CudaMinimizer<T>* minimizer) = 0;
    
    virtual cudaError_t calculateEpsilonEntropy(CudaMinimizer<T>* minimizer, bool force = false) = 0;

    virtual cudaError_t cleanup(CudaMinimizer<T>* minimizer) = 0;
    virtual const char* getName() const = 0;
    virtual size_t getMemoryRequired(CudaMinimizer<T>* minimizer) const = 0;
};