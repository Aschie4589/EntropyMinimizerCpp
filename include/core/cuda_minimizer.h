// Minimizer.h
#pragma once

#include "config/config.h"
#include "helpers/vector_serializer.h"
#include "core/entropy_estimator.h"

#include "cuComplex.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

class CudaMinimizer {
public:
    CudaMinimizer(cuDoubleComplex* d_kraus, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, double eps);             // Constructor declaration
    ~CudaMinimizer();            // Destructor declaration

    // Initialization
    cudaError_t initializeVectorFromDevice(cuDoubleComplex* v); // This initializes the vector to a given one on device.
    cudaError_t initializeVectorFromHost(cuDoubleComplex* v); // This initializes the vector to a given one on host.
    cudaError_t initializeRandomVector(); // Initializes the vector for the algorithm to a random one,

    // Updaters
    cudaError_t updateProjector(); // Calculates the rank one projector from the vector stored in memory

    // Algorithm
    cudaError_t stepAlgorithm(); // Run one step of the algorithm: update the vector with a new, better one. In so doing, scramble both input and output matrix.
    cudaError_t step(); // Run stepAlgorithm and update the entropy. Makes sure to initialize vectors and matrices correctly.

    // Getters
    cuDoubleComplex* getVector(); // Return pointer to vector state on device
    cuDoubleComplex* getInputState(); // Return pointer to state on device (projector)
    cuDoubleComplex* getOutputState(); // Return pointer to output state on device
    double getEntropy();

    // Updaters
    cudaError_t calculateEpsilonEntropy(); // Calculates the entropy of Phi_e(d_inmatrix)



private:
    // Members
    int d, M, N;
    double epsilon, bin_entropy, entropy_error, entropy, estimated_entropy, estimated_entropy_ub, estimated_entropy_lb;
    double *d_entropy_scratch; // Pointer to device memory for entropy calculation
    bool input_matrix_scrambled; // Flag to indicate if the input matrix has been scrambled

    cublasHandle_t handle; // CUBLAS handle for matrix operations
    cusolverDnHandle_t cusolver_handle; // CUSOLVER handle for singular value decomposition and other operations

    // Updaters
    cudaError_t applyChannel();
    cudaError_t applyDualChannel();
    cudaError_t applyEpsilonChannel();
    cudaError_t applyEpsilonDualChannel();


    // Device memory pointers
    // Matrices and vectors 
    cuDoubleComplex* d_kraus;
    cuDoubleComplex* d_vecstate;
    cuDoubleComplex* d_inmatrix;
    cuDoubleComplex* d_outmatrix; 
    cuDoubleComplex* d_tmpmat;
    double* d_eigs;
    

};
