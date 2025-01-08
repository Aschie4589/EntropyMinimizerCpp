// Minimizer.h
#ifndef MINIMIZER_H  // Include guard to prevent multiple inclusions
#define MINIMIZER_H

#include <complex>

#include "Config.h"
class Minimizer {
public:
    Minimizer(std::vector<std::complex<double> >* kraus_ops,int kraus_number,int kraus_in_dimension,int kraus_out_dimension, double eps = EPS);             // Constructor declaration
    ~Minimizer();            // Destructor declaration
    // Initialization
    int initializeVector(std::vector<std::complex<double> >* vector_pointer); // This initializes the vector to a given one. If dimensions don't match, it defaults to initializing a random vector
    int initializeRandomVector(); // Initializes the vector for the algorithm to a random one

    // Updaters
    int updateProjector(); // Calculates the rank one projector from the vector stored in memory
    int calculateEntropy(); // Calculates the entropy of Phi(projector), recalculates the projector for safety

    // Algorithm
    int stepAlgorithm(); // Run one step of the algorithm: update the vector with a new, better one. In so doing, scramble both input and output matrix.
    int minimizeEntropy(); // Run a full minimization pass, until tolerance is reached
    int step(); // This both runs one step of the algorithm, and updates the entropy.

    // Getters
    std::vector<std::complex<double> >* getState();
    double* getEntropy();

    // Various helper functions (printing and testing, logging, debugging)
    int printVectorState();
    int printState();

private:
    // Members
    int N, M, d;
    double epsilon, bin_entropy, entropy_error, entropy, estimated_entropy, estimated_entropy_ub, estimated_entropy_lb;
    
    std::vector<std::complex<double> >* kraus_operators;
    std::vector<std::complex<double> >* vector_state;
    std::vector<std::complex<double> >* input_matrix;
    std::vector<std::complex<double> >* output_matrix; // is this one necessary?
    // Methods
    int printMatrix(std::vector<std::complex<double> >* matrix_pointer, int n, int m);
    int applyChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension);
    int applyDualChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension);
    int applyEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon);
    int applyDualEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon);

};

#endif