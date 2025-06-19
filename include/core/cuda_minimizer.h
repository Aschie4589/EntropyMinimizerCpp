// Minimizer.h
#pragma once

#include "config/config.h"
#include "helpers/vector_serializer.h"
#include "core/entropy_estimator.h"

#include "cuComplex.h"
#include <cublas_v2.h>

class CudaMinimizer {
public:
    CudaMinimizer(cuDoubleComplex* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, double eps);             // Constructor declaration
    ~CudaMinimizer();            // Destructor declaration

    // Initialization
    cudaError_t initializeVector(cuDoubleComplex* v); // This initializes the vector to a given one. If dimensions don't match, it defaults to initializing a random vector
    cudaError_t initializeRandomVector(); // Initializes the vector for the algorithm to a random one

    // Updaters
    cudaError_t updateProjector(); // Calculates the rank one projector from the vector stored in memory

    // Algorithm
    cudaError_t stepAlgorithm(); // Run one step of the algorithm: update the vector with a new, better one. In so doing, scramble both input and output matrix.

    // Getters
    cuDoubleComplex* getVector(); // Return pointer to vector state on device
    cuDoubleComplex* getState(); // Return pointer to state on device (projector)
    cuDoubleComplex* getOutputState(); // Return pointer to output state on device


    /*
    // Updaters
    int calculateEntropy(); // Calculates the entropy of Phi(projector), recalculates the projector for safety

    // Algorithm
    int step(); // This both runs one step of the algorithm, and updates the entropy.

    // Getters
    double* getEntropy();
    int getN();
    int getD();

    // Various helper functions (printing and testing, logging, debugging)
    int printVectorState();
    int printState();
    */
    cudaError_t applyChannel();
    cudaError_t applyDualChannel();
    cudaError_t applyEpsilonChannel();
    cudaError_t applyEpsilonDualChannel();

private:
    // Members
    int d, M, N;
    double epsilon, bin_entropy, entropy_error, entropy, estimated_entropy, estimated_entropy_ub, estimated_entropy_lb;

    cublasHandle_t handle; // CUBLAS handle for matrix operations
    // Matrices and vectors
    cuDoubleComplex* kraus_operators;
    cuDoubleComplex* vector_state;
    cuDoubleComplex* input_matrix;
    cuDoubleComplex* output_matrix; 
    cuDoubleComplex* tmp_matrix;
    // Methods


    /*
    int printMatrix(std::vector<std::complex<double> >* matrix_pointer, int n, int m);
    int applyEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon);
    int applyDualEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon);
    */
};
