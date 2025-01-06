// Minimizer.cpp
#include "Minimizer.h"
#include <iostream>
#include <iomanip>
#include <complex>
#include <random>

Minimizer::Minimizer(std::vector<std::complex<double> > kraus_ops, 
                        int kraus_number, int kraus_in_dimension, int kraus_out_dimension,double epsilon) {
    // PARAMETERS ASSIGNMENT
    // kraus_operators stores the pointer to a vector of double precision complex numbers. I don't want to have two large objects in memory so this is the way to go.
    // It consists of a list of d contiguous NxN matrices, each stored in column-major order.
    // So entry (i,j) of matrix k corresponds to index k*N*N+j*N+i
    kraus_operators = &kraus_ops;
    // assign the dimensions. Since I assign values, those are copied and d, N, M live in memory where the class instance lives
    d = kraus_number;
    N = kraus_in_dimension;
    M = kraus_out_dimension;
    // assign precision
    this->epsilon = epsilon; // "this" is like self for python, but returns a pointer to the instance of the class

    // INITIALIZATION OF MATRICES AND VECTORS
    // start by initializing the vector_state to a zero vector, to avoid seg faults.
    vector_state = new std::vector(N,std::complex<double>(0.0f,0.0f));
    // also initialize the input and output matrices to zero vectors, again for safety.
    input_matrix = new std::vector(N*N,std::complex<double>(0.0f,0.0f));
    output_matrix = new std::vector(M*M,std::complex<double>(0.0f,0.0f));

    // OTHER USEFUL CONSTANTS
    // we need to know the binary entropy of epsilon, which is used in obtaining the error in the approximation of the entropy
    bin_entropy = -epsilon * std::log(epsilon) - (1-epsilon) * std::log(1-epsilon);
    // the corresponding error is quantified by bin_ent(eps)/2*(1-eps)
    entropy_error = bin_entropy / (2*(1-epsilon));
}

int Minimizer::initializeVector(std::vector<std::complex<double> >* pointer) {
    // By construction, vector_state is always a vector that we have created with "new".
    // We want to keep that pointer and just copy the data over, if "pointer" is a valid
    // reference to a new vector of the right size

    // Step 1: Check that the pointer points to a valid, non-empty vector:
    if (pointer == nullptr || pointer->empty()){
        std::cout << "Wait a second, no starting vector was provided! I will generate a random one..." << std::endl;
        return initializeRandomVector();
    } 
    // Step 2: Check that the vector has the right dimension (equal to the input dimension)
    if (pointer->size()==N){
        std::cout << "The input vector has the right dimension" << std::endl;
        // Copy data over, keep the pointer
        *vector_state = *pointer;
        return 0;
    } else {
        // If it doesn't, generate a random vector as a fallback.
        std::cout << "The vector does not match the specified dimension! Fallback: generating random vector..." << std::endl;
        return initializeRandomVector();
    }
}

int Minimizer::initializeRandomVector(){
    // By construction, the pointer points to a valid vector of the right size. Note that N is fixed and cannot change.

    // Step 1: Set up random number generator
    std::random_device rd;                  // Random device to seed the generator
    std::mt19937 gen(rd());                 // Mersenne Twister generator, seeded by rd()
    std::normal_distribution<double> dist(0.0f, 1.0f); // Normal distribution with mean and stddev

    // Step 2: Generate random real and imaginary parts
    for (int i=0; i < N; i++){
        double real_part = dist(gen);  // Generate the real part (normal distribution)
        double imag_part = dist(gen);  // Generate the imaginary part (normal distribution)
        vector_state->at(i) = std::complex<double>(real_part, imag_part);
    }

    // Step 3: Renormalize the vector
    double norm = 0.0f;
    for (std::complex<double> entry : *vector_state){
        norm += std::pow(std::abs(entry),2);
    }
    norm = std::sqrt(norm);
    // std::cout << "Norm of the vector is: "<<norm<<std::endl;
    if (norm != 0) {
        for (int i = 0; i < vector_state->size(); i++){
            vector_state->at(i) /= norm;
        }
    }
    return 0;
}

int Minimizer::updateProjector(){
    // We use column-major order, to comply with LAPACK
    for (int i=0;i<N;i++){ // Go through rows
        for (int j=0; j<N; j++){
            // Entry (i,j) of matrix (row i, col j) is stored at j*N+i (0-based)
            // Entry (i,j) is |i><j|
            input_matrix->at(i+j*N) = vector_state->at(i) * std::conj(vector_state->at(j));
        }
    }
    return 0;
}

int Minimizer::applyChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension){
    // The three pointers are for: - where kraus ops are stored; - where input matrix is stored; where output matrix needs to be stored
    // Then we need to know how many kraus operators there are, and what is their size.
    return 0;
}





/// GETTERS
std::vector<std::complex<double> >* Minimizer::getState(){
    return input_matrix;
}



/// DEBUGGING AND PRINTING AND ETC

int Minimizer::test(){
    std::cout << "The vector state pointer is:" << vector_state << std::endl;
    return 0;
}

int Minimizer::printVectorState(){
    // Print a list. Some variables are hard coded!
    if (vector_state == nullptr){
        std::cout << "No assigned value for vector..." << std::endl;
        return 1;
    } else {
        int size = vector_state->size();
        std::cout << "[";
        for (int i = 0; i < size; i++) {
                double r = (*vector_state)[i].real();
                double j = (*vector_state)[i].imag();
                std::cout <<std::fixed << std::setprecision(4)<< r;
                if (j>-1e-10){
                    std::cout << "+";
                }
                std::cout <<std::fixed << std::setprecision(4)<<j << "j, ";
        }
        std::cout << "]"<< std::endl;
        return 0;
    }

}

int Minimizer::printMatrix(std::vector<std::complex<double> >* matrix_pointer, int n, int m) {
    // "Pretty" prints a nxm matrix in column-major order
    // To begin with, let's just print the entries and separate them by tabs or something
    // All variables are hardcoded here!
    for (int i=0; i<n; i++){
        for (int j=0;j<m; j++){
                double re = (*matrix_pointer)[i*m+j].real();
                double im = (*matrix_pointer)[i*m+j].imag();
                if (re>-1e-10){
                    std::cout << " ";
                }
                std::cout <<std::fixed << std::setprecision(4)<< re;
                if (im>-1e-10){
                    std::cout << "+";
                }
                std::cout <<std::fixed << std::setprecision(4)<<im << "j ";
        }
        std::cout << std::endl;
    }
    return 0;
}

int Minimizer::printState(){
    printMatrix(getState(), N, N);
    return 0;
}

Minimizer::~Minimizer() {
    delete vector_state;
    delete input_matrix;
    delete output_matrix;
}