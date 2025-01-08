#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
// These two could need to be installed
#include <cblas.h>
#include <lapacke.h>
// My headers
#include "Minimizer.h"
#include "Config.h"



const int d = 20;
const int N = 1000;

std::vector<std::complex<double> >* generateHaarRandomUnitary(int N){
    // Step 0: Initialize output
    std::vector<std::complex<double> >* out = new std::vector<std::complex<double> >(N*N);
    // Step 1: Set up random number generator
    std::random_device rd;                  // Random device to seed the generator
    std::mt19937 gen(rd());                 // Mersenne Twister generator, seeded by rd()
    std::normal_distribution<double> dist(0.0f, 1.0f); // Normal distribution with mean and stddev

    // Step 2: Generate random real and imaginary parts
    for (int i=0; i < N*N; i++){
        double real_part = dist(gen);  // Generate the real part (normal distribution)
        double imag_part = dist(gen);  // Generate the imaginary part (normal distribution)
        out->at(i) = std::complex<double>(real_part, imag_part);
    }

    // Step 3: Perform QR decomposition
    std::vector<std::complex<double> > tau(N);
    std::complex<double> work_size;
    // Query the optimal work size
    LAPACKE_zgeqrfp_work(LAPACK_COL_MAJOR, N, N, reinterpret_cast<lapack_complex_double*>(out->data()),N,reinterpret_cast<lapack_complex_double*>(tau.data()),reinterpret_cast<lapack_complex_double*>(&work_size),-1);
    // Actually perform QR
    int lwork = static_cast<int>(work_size.real());
    std::vector<std::complex<double>> work(lwork);
    LAPACKE_zgeqrfp_work(LAPACK_COL_MAJOR, N, N, reinterpret_cast<lapack_complex_double*>(out->data()),N,reinterpret_cast<lapack_complex_double*>(tau.data()),reinterpret_cast<lapack_complex_double*>(work.data()), lwork);
    // Query optimal work size to reconstruct Q
    LAPACKE_zungqr_work(LAPACK_COL_MAJOR, N,N,N,reinterpret_cast<lapack_complex_double*>(out->data()),N,reinterpret_cast<lapack_complex_double*>(tau.data()),reinterpret_cast<lapack_complex_double*>(&work_size),-1);
    lwork = static_cast<int>(work_size.real());
    work.resize(lwork);
    LAPACKE_zungqr_work(LAPACK_COL_MAJOR, N,N,N,reinterpret_cast<lapack_complex_double*>(out->data()),N,reinterpret_cast<lapack_complex_double*>(tau.data()),reinterpret_cast<lapack_complex_double*>(work.data()), lwork);
    return out;
}

int main() {
    std::vector<std::complex<double> >* kraus_operators = new std::vector<std::complex<double> >(d*N*N); // kraus_operators is the pointer.
    for (int m = 0; m < d; m++){
        std::cout << "Generating Haar random unitary " << m+1 << " of " << d << "... ";
        // append a new unitary at position i of the kraus operator.
        std::vector<std::complex<double> >* new_haar_unitary = generateHaarRandomUnitary(N);
        for (int i = 0; i < N*N ;i++){
            kraus_operators->at(m*N*N+i) = new_haar_unitary->at(i)/std::sqrt(double(d));
        }
        std::cout << "Done!" << std::endl;
        delete new_haar_unitary;
    }


    Minimizer minimizer = Minimizer(kraus_operators, d, N, N); 

    minimizer.initializeRandomVector();

    for (int i=0; i< 100000; i++){
        minimizer.stepAlgorithm();
        std::cout << i << ": ";
        minimizer.calculateEntropy();
    }
    delete kraus_operators;
    return 0;
}

