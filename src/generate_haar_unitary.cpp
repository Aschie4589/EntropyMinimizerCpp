#include "common_includes.h"
#include "matrix_operations.h"
#include "generate_haar_unitary.h"

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
    zgeqrfp_wrapper(N, N, out, N, &tau, &work_size,-1);
    // Actually perform QR
    int lwork = static_cast<int>(work_size.real());
    std::vector<std::complex<double>> work(lwork);
    zgeqrfp_wrapper(N, N, out, N, &tau, &work, lwork);
    // Query optimal work size to reconstruct Q
    zungqr_wrapper(N,N,N,out,N,&tau,&work_size,-1);
    lwork = static_cast<int>(work_size.real());
    work.resize(lwork);
    zungqr_wrapper(N,N,N,out,N,&tau,&work, lwork);
    return out;
}
