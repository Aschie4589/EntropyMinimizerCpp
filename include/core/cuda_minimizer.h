// Minimizer.h
#pragma once

#include "config/config.h"
#include "helpers/vector_serializer.h"
#include "core/entropy_estimator.h"

#include "core/cuda_traits.h"

#include "cuComplex.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

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
    virtual cudaError_t calculateEpsilonEntropy() = 0;
};


template<typename T>
class CudaMinimizer : public CudaMinimizerBase  {
public:
    CudaMinimizer(typename CudaTraits<T>::Complex* d_kraus, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, typename CudaTraits<T>::Real eps);             // Constructor declaration
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
    cudaError_t calculateEpsilonEntropy() override; // Calculates the entropy of Phi_e(d_inmatrix)



private:
    // Members
    int d, M, N;
    typename CudaTraits<T>::Real epsilon, entropy;

    // Handles and streams for parallel computation
    int num_streams;
    cudaStream_t* streams; // CUDA streams for parallel execution
    cublasHandle_t* blas_handles; // CUBLAS handles, one per stream
    cusolverDnHandle_t cusolver_handle; // CUSOLVER handle, used for SVD. Only one needed.

    // Internal tracker for current stage of the algorithm
    int minimizer_state;

    // Updaters
    cudaError_t step_1();
    cudaError_t step_2();
    /*OLD
    cudaError_t applyChannel();
    cudaError_t applyDualChannel();
    cudaError_t applyEpsilonChannel();
    cudaError_t applyEpsilonDualChannel();
    */


    // Device memory pointers
    // Matrices and vectors 
    typename CudaTraits<T>::Complex* d_kraus;         // Pointer to kraus operators
    typename CudaTraits<T>::Complex* d_vec;           // Pointer to the vector state on device
    typename CudaTraits<T>::Complex* d_vecs_1;        // Pointer to memory for storing {K_i |d_vec>}_i
    typename CudaTraits<T>::Complex* d_vecs_2;        // Pointer to memory for storing {K_i |phi_j>}_ij where |phi_j> comes from SVD of d_vecs_1
    typename CudaTraits<T>::Complex* d_scratch;       // Pointer to scratch space for doing computations
    typename CudaTraits<T>::Real* d_sv_1;                   // Pointer to memory for storing singular values of {K_i |d_vec>}_i
    typename CudaTraits<T>::Real* d_sv_2;                   // Pointer to memory for storing singular values of {K_i |phi_j>}_ij
    // Other relevant internal quantities
    int work_size_1;                  // For SVD
    int work_size_2;                  // For SVD
    // OLD
    /*
    bool input_matrix_scrambled; // Flag to indicate if the input matrix has been scrambled
    double *d_entropy_scratch; // Pointer to device memory for entropy calculation
    double bin_entropy, entropy_error, estimated_entropy, estimated_entropy_ub, estimated_entropy_lb;

    cuDoubleComplex* d_kraus;
    cuDoubleComplex* d_vecstate;
    cuDoubleComplex* d_inmatrix;
    cuDoubleComplex* d_outmatrix; 
    cuDoubleComplex* d_tmpmat;
    double* d_eigs;
    */
    

};
