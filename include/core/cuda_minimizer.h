// Minimizer.h
#pragma once

#include "config/config.h"
#include "helpers/vector_serializer.h"
#include "core/entropy_estimator.h"

#include "core/cuda_traits.h"
#include "core/cuda_minimizer_strategy.h"

#include "cuComplex.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

// Struct to let the user select the minimization strategy they want
enum class CudaMinimizerStrategy {
    AUTO_DETECT,     // Default: automatically choose based on available memory
    LOW_MEMORY,      // Force low memory strategy (current implementation)
    LOW_MEMORY_PROBABILISTIC,      // Force low memory probabilistic strategy
    HIGH_MEMORY,     // Force high memory strategy (precompute transposes)
    BALANCED         // Future: medium memory strategy
};

// Forward declarations for friend classes
template<typename T> class HighMemoryStrategy;
template<typename T> class LowMemoryStrategy;
template<typename T> class LowMemoryStrategyProb;

class CudaMinimizerBase {
public:
    virtual ~CudaMinimizerBase() {}

    // Initialization
    virtual cudaError_t initializeVectorFromDevice(void* v) = 0; // This initializes the vector to a given one on device.
    virtual cudaError_t initializeVectorFromHost(void* v) = 0; // This initializes the vector to a given one on host.
    virtual cudaError_t initializeRandomVector() = 0; // Initializes the vector for the algorithm to a random one.

    // Algorithm
    virtual cudaError_t step() = 0;

    // Getters
    virtual double getEntropy() = 0;

    //Updaters
    virtual cudaError_t calculateEpsilonEntropy(bool force) = 0;
};


template<typename T>
class CudaMinimizer : public CudaMinimizerBase  {
public:
    CudaMinimizer(typename CudaTraits<T>::Complex* d_kraus, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, typename CudaTraits<T>::Real eps, CudaMinimizerStrategy strategy_preference=CudaMinimizerStrategy::AUTO_DETECT);             // Constructor declaration
    ~CudaMinimizer() override;            // Destructor declaration


    // Initialization
    cudaError_t initializeVectorFromDevice(void* v) override; // This initializes the vector to a given one on device.
    cudaError_t initializeVectorFromHost(void* v) override; // This initializes the vector to a given one on host.
    cudaError_t initializeRandomVector() override; // Initializes the vector for the algorithm to a random one,

    // Updaters
    //cudaError_t updateProjector(); // Calculates the rank one projector from the vector stored in memory

    // Algorithm
    cudaError_t stepAlgorithm(); // Run one step of the algorithm: update the vector with a new, better one. In so doing, scramble both input and output matrix.
    cudaError_t step() override; // Run one step of the algorithm and update the entropy. Makes sure to initialize vectors and matrices correctly.

    // Getters
    typename CudaTraits<T>::Complex* getVector(); // Return pointer to vector state on device
    //cuDoubleComplex* getInputState(); // Return pointer to state on device (projector)
    //cuDoubleComplex* getOutputState(); // Return pointer to output state on device
    double getEntropy() override;

    // Updaters
    cudaError_t calculateEpsilonEntropy(bool force= false) override; // Calculates the entropy of Phi_e(d_inmatrix)



private:
    friend class HighMemoryStrategy<T>;
    friend class LowMemoryStrategy<T>;
    friend class LowMemoryStrategyProb<T>;
    // Strategy selection
    std::unique_ptr<MinimizationStrategy<T>> createStrategy(CudaMinimizerStrategy preference, 
                                                            int d, int N, int M);
    std::unique_ptr<MinimizationStrategy<T>> strategy;

    // Members
    int d, M, N;
    typename CudaTraits<T>::Real epsilon, entropy;

    // Handles and streams for parallel computation
    int num_streams;
    cudaStream_t* streams; // CUDA streams for parallel execution
    cublasHandle_t* blas_handles; // CUBLAS handles, one per stream
    cusolverDnHandle_t cusolver_handle; // CUSOLVER handle, used for SVD. Only one needed.

    // Internal tracker for current stage of the algorithm
    int minimizer_state; // Deprecated I think


    // Device memory pointers
    // Matrices and vectors 
    typename CudaTraits<T>::Complex* d_kraus;         // Pointer to kraus operators
    typename CudaTraits<T>::Complex* d_vec;           // Pointer to the vector state on device

};
