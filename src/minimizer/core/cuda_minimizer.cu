// Minimizer.cpp
#include <iostream>
#include <stdexcept>

#include "minimizer/core/cuda_minimizer.h"
#include "minimizer/config/compile_time_config.h"
#include "utilities/config/compile_time_macros.h"
#include "minimizer/core/cuda_traits.h"
#include "minimizer/core/kernels/rescale_vecs.h"
#include "minimizer/utilities/generate_random_vector.h"

#include "minimizer/core/cuda_minimizer_strategy/low_memory_strategy.h"
#include "minimizer/core/cuda_minimizer_strategy/high_memory_strategy.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>

// Profiling
#include <nvtx3/nvtx3.hpp>




template<typename T>
CudaMinimizer<T>::CudaMinimizer(typename CudaTraits<T>::Complex* d_kraus_p, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, typename CudaTraits<T>::Real eps, CudaMinimizerStrategy strategy_preference) {
    /*
    CudaMinimizer constructor.
    
    Arguments:
    - *d_kraus (device): pointer to an array of complex numbers (cuComplex or cuComplexDouble), which contains the Kraus operators.
        - Promise: the channel is trace preserving (up to a multiplicative factor). Else won't work.
        - Note: Depending on the strategy adopted, d_kraus might be preserved (low_memory_strategy) or scrambled (high_memory_strategy).
    - *kraus_number (host): pointer to number of Kraus operators.
    - *kraus_in_dimension (host): pointer to the input dimension of the Kraus operators.
    - *kraus_out_dimension (host): pointer to the output dimension of the Kraus operators
    - *eps (host): pointer to the precision of the minimization algorithm.

    Notes:
    - The d_kraus array should be an array of d contiguous NxN matrices, each stored in column-major order.
      So entry (i,j) of matrix k corresponds to index k*N*N+j*N+i.
    - CudaMinimizer is NOT responsible for lifetime management of these resources.
    - The prefix d_ is used for device data. If nothing is specified, the data is stored on the host.
    */

    // Step 1: save the pointers that were passed.
    d_kraus = d_kraus_p;
    d = kraus_number;
    N = kraus_in_dimension;
    M = kraus_out_dimension;
    epsilon = eps; 
    entropy = -1;
    // Step 2: allocate space for the vector state for the system
    d_vec = nullptr;
    CUDA_MALLOC_CHECK(d_vec, N * sizeof(typename CudaTraits<T>::Complex), "d_vec");



    // SVD only supports the case cols >= rows.
    /* TODO: Implement a better algorithm for the second SVD, where only the top eigenvector is actually needed!*/
    if (d > M){
        std::cerr << "Error: d > M. This is not supported by the SVD algorithm." << std::endl;
        throw std::runtime_error("Invalid dimensions for SVD");
    }


    // Step 3: create handles for cuBLAS, cuSolver and appropriate streams
    num_streams = MINIMIZER_CUDA_STREAMS; // Number of streams to use for parallel execution of minimizer operations
    streams = new cudaStream_t[num_streams]; // Array of streams for parallel execution
    blas_handles = new cublasHandle_t[num_streams];
    cudaError_t err;
    // Create streams and blas_handles
    for (int i = 0; i < num_streams; i++) {
        // Streams
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Error creating CUDA stream: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to create CUDA stream");
        }
        // cuBLAS handles
        cublasStatus_t cublas_status;
        cublas_status = cublasCreate(&blas_handles[i]);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS initialization failed for stream " << i << "!" << std::endl;
            throw std::runtime_error("CUBLAS initialization failed for stream");
        }
        cublas_status = cublasSetStream(blas_handles[i], streams[i]);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS failed to set stream for stream " << i << "!" << std::endl;
            throw std::runtime_error("CUBLAS failed to set stream");
        }
    }
    // create and bind cusolver_handle to streams[0]
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization failed!" << std::endl;
        throw std::runtime_error("CUSOLVER initialization failed");
    }
    cusolver_status = cusolverDnSetStream(cusolver_handle, streams[0]);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER failed to set stream for stream 0!" << std::endl;
        throw std::runtime_error("CUSOLVER failed to set stream");
    }


    // Step 3: Select the appropriate strategy
    strategy = createStrategy(strategy_preference, d, N, M);
    std::cout << "Selected strategy: " << strategy->getName() << std::endl;
    // Step 4: Let the strategy initialize (allocate memory on device for scratch space etc.)
    strategy->initialize(this);


    // Update internal state
    minimizer_state = CUDA_MINIMIZER_CREATED;

}



// Strategy selection
template<typename T>
std::unique_ptr<MinimizationStrategy<T>> CudaMinimizer<T>::createStrategy(
    CudaMinimizerStrategy preference, int d, int N, int M) {

    switch (preference) {
        case CudaMinimizerStrategy::LOW_MEMORY:
            std::cout << "Force-selecting LowMemory strategy" << std::endl;
            return std::make_unique<LowMemoryStrategy<T>>();
            
        case CudaMinimizerStrategy::HIGH_MEMORY: {
            // Check if high memory strategy is feasible
            size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            
            auto high_strategy = std::make_unique<HighMemoryStrategy<T>>();
            size_t required_memory = high_strategy->getMemoryRequired(this);
            
            if (free_bytes > required_memory * 1.2) {
                std::cout << "Force-selecting HighMemory strategy" << std::endl;
                return high_strategy;
            } else {
                std::cerr << "WARNING: Insufficient memory for HIGH_MEMORY strategy!" << std::endl;
                std::cerr << "Required: " << required_memory / (1024*1024) << " MB, ";
                std::cerr << "Available: " << free_bytes / (1024*1024) << " MB" << std::endl;
                std::cerr << "Falling back to LowMemory strategy" << std::endl;
                return std::make_unique<LowMemoryStrategy<T>>();
            }
        }

        case CudaMinimizerStrategy::AUTO_DETECT: {
            // Automatic selection based on available memory
            size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            
            auto high_strategy = std::make_unique<HighMemoryStrategy<T>>();
            size_t required_memory = high_strategy->getMemoryRequired(this);
            
            std::cout << "Auto-detecting strategy..." << std::endl;
            std::cout << "  Available memory: " << free_bytes / (1024*1024) << " MB" << std::endl;
            std::cout << "  Additional memory required for HighMemory: " << required_memory / (1024*1024) << " MB" << std::endl;
            
            if (free_bytes > required_memory * 1.3) {  // 30% buffer for auto-detect
                std::cout << "  Auto-selected: HighMemory strategy" << std::endl;
                return high_strategy;
            } else {
                std::cout << "  Auto-selected: LowMemory strategy" << std::endl;
                return std::make_unique<LowMemoryStrategy<T>>();
            }
        }

        case CudaMinimizerStrategy::BALANCED:
            // Future implementation
            std::cout << "BALANCED strategy not implemented yet, using LowMemory" << std::endl;
            return std::make_unique<LowMemoryStrategy<T>>();
            
        default:
            std::cerr << "Unknown strategy preference, defaulting to AUTO_DETECT" << std::endl;
            return createStrategy(CudaMinimizerStrategy::AUTO_DETECT, d, N, M);
    }
}

template<typename T>
cudaError_t CudaMinimizer<T>::initializeVectorFromDevice(void* v) {
    /*
    Initializes the vector state to a given vector v.

    Arguments:
        - v (device): pointer to a vector of complex numbers (single or double precision), which is the vector to initialize the state to.
    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The dimension is not checked! If the dimension does not match, undefined behavior.
        - The data in v is copied to d_vec.
        - No memory management is performed - in particular, memory for v is not freed.
        - pointer has to be to void type and is cast internally
    */
    CUDA_MEMCPY_CHECK(d_vec, v, N * sizeof(typename CudaTraits<T>::Complex), cudaMemcpyDeviceToDevice, "d_vec");
    minimizer_state = CUDA_MINIMIZER_STAGE_0; // Reset the state to stage 0
    return cudaSuccess;
}

template<typename T>
cudaError_t CudaMinimizer<T>::initializeVectorFromHost(void* v){
    /*
    Initializes the vector state to a given vector v, which is stored on the host.
    Arguments:
        - v (host): pointer to a vector of complex numbers (single or double precision), which is the vector to initialize the state to.
    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The dimension is not checked! If the dimension does not match, undefined behavior.
        - The data in v is simply copied to the d_vec.
        - No memory management is performed - in particular, memory for v is not freed.
        - pointer has to be to void type and is cast internally
    */
    CUDA_MEMCPY_CHECK(d_vec, v, N * sizeof(typename CudaTraits<T>::Complex), cudaMemcpyHostToDevice, "d_vec");
    minimizer_state = CUDA_MINIMIZER_STAGE_0; // Reset the state to stage 0
    return cudaSuccess; // Returns success if the copy was successful

}

template<typename T>
cudaError_t CudaMinimizer<T>::initializeRandomVector(){
    /*
    Use the generate_random_vector function to initialize the vector state to a random vector.

    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The vector state is initialized to a random vector of dimension N.
        - The data in the d_vec is overwritten.
    */

    // Generate a random vector on the device
    cudaError_t err = generateUniformRandomVectorsCuda<T>(d_vec, N, 1);
    minimizer_state = CUDA_MINIMIZER_STAGE_0; // Reset the state to stage 0

    return err;
}



template<typename T>
cudaError_t CudaMinimizer<T>::stepAlgorithm(){
    /*
        Run through one minimization pass of the algorithm, by delegating to the selected strategy. On success, updates d_vec with a new one which provides better entropy.

        Returns:
            - cudaSuccess if the step was successful.
    */
    cudaError_t err;
    err = strategy->step(this);
    if (err != cudaSuccess) {
        std::cerr << "Error while stepping through the minimization algorithm: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 2 failed
    }
    return cudaSuccess;
}

template<typename T>
cudaError_t CudaMinimizer<T>::step(){
    /*
        Will compute one step of the minimization algorithm.

        1. It will replace the content d_vec variable with the updated vector state
        2. It will compute the epsilon entropy

        Returns:
            - cudaSuccess if the step was successful.
        Note: 
            - d_vecstate must be initialized to a valid vector, no checks are performed

    */

    cudaError_t err = strategy->step(this);
    if (err != cudaSuccess) {
        std::cerr << "Error while stepping through the minimization algorithm: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 2 failed
    }
    return cudaSuccess;


}


/*

        UPDATERS

*/



template<typename T>
cudaError_t CudaMinimizer<T>::calculateEpsilonEntropy(bool force){
/*
    Calculates the von Neumann entropy of the output matrix after applying the epsilon channel. Delegates the task to the strategy selected.
    
    Returns:
        - cudaSuccess if the entropy was calculated successfully.
    Note:
        - The entropy is stored in the member variable `entropy`.

*/
    cudaError_t err = strategy->calculateEpsilonEntropy(this, force);
    if (err != cudaSuccess) {
        std::cerr << "Error while calculating epsilon entropy: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if the entropy calculation failed
    }

    return cudaSuccess; // Return success if the entropy was calculated successfully    
}


/*

                GETTERS

*/
template<typename T>
typename CudaTraits<T>::Complex* CudaMinimizer<T>::getVector() {
    /*
    Returns a pointer to the vector state on the device.

    Returns:
        - Pointer to the vector state on the device.
    */
    return d_vec;
}

template<typename T>
double CudaMinimizer<T>::getEntropy() {
    /*
    Returns the current von Neumann entropy.

    Returns:
        - Pointer to the entropy value on the device.
    Note:
        - The entropy is computed as the von Neumann entropy of the output matrix.
        - This method does not update the entropy.
    */
    return static_cast<double>(entropy);
}

template<typename T>
CudaMinimizer<T>::~CudaMinimizer() {
    /*
    CudaMinimizer destructor.
    
    Notes:
    - The destructor frees the memory allocated for the internally allocated GPU resources.
    - The d_kraus are not freed, as they are not managed by this class.
    */

    // Free the device memory for the matrices and vectors
    if (d_vec != nullptr) cudaFree(d_vec);

    // Let the strategy clean up its resources
    strategy->cleanup(this);

    // free the cusolver handle
    cusolverDnDestroy(cusolver_handle);

    // Free the streams and handles
    for (int i = 0; i < num_streams; i++) {
        cublasDestroy(blas_handles[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete[] blas_handles;
    delete[] streams;
}

// Instantiate the types I need
template class CudaMinimizer<double>;
template class CudaMinimizer<float>;