// Minimizer.h
#ifndef MYCLASS_H  // Include guard to prevent multiple inclusions
#define MYCLASS_H

#include <complex>

class Minimizer {
public:
    Minimizer(std::vector<std::complex<double> >* kraus_ops,
                            int kraus_number, 
                            int kraus_in_dimension,
                            int kraus_out_dimension, 
                            double epsilon = 1e-15);             // Constructor declaration
    ~Minimizer();            // Destructor declaration
    int initializeVector(std::vector<std::complex<double> >* vector_pointer);
    int initializeRandomVector();
    int updateProjector();
    int stepAlgorithm();
    int calculateEntropy();

    // Getters
    std::vector<std::complex<double> >* getState();

    // Various helper functions (printing and testing, logging, debugging)
    int printVectorState();
    int printState();
    int test();


    // Should be made private...
    int printMatrix(std::vector<std::complex<double> >* matrix_pointer, int n, int m);
    int applyChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension);
    int applyDualChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension);
    int applyEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon);
    int applyDualEpsilonChannel(std::vector<std::complex<double> >* kraus,std::vector<std::complex<double> >* in_matrix,std::vector<std::complex<double> >* out_matrix, int number_kraus, int in_dimension, int out_dimension, double epsilon);
    std::vector<std::complex<double> >* output_matrix; // is this one necessary?

private:
    int N, M, d;
    double epsilon, bin_entropy, entropy_error, entropy, estimated_entropy, estimated_entropy_ub, estimated_entropy_lb;
    std::vector<std::complex<double> >* kraus_operators;
    std::vector<std::complex<double> >* vector_state;
    std::vector<std::complex<double> >* input_matrix;
};

#endif