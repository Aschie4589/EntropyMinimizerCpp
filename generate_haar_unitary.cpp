#include "common_includes.h"
#include "core/matrix_operations.h"
#include "core/generate_haar_unitary.h"
#include <omp.h>

std::vector<std::complex<double> >* generateHaarRandomUnitaryOld(int N){
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

std::vector<std::complex<double>> generateHaarRandomUnitary(int N) {
    // Step 0: Initialize output matrix (row-major)
    std::vector<std::complex<double>> out(N * N);

    // Step 1: Parallel random number generation using thread-local RNGs
    #pragma omp parallel // Define parallel region
    {
        // Each thread gets its own RNG seeded differently
        std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
        std::normal_distribution<double> dist(0.0, 1.0);

        #pragma omp for // Distribute the work across threads
        for (int i = 0; i < N * N; ++i) {
            double real_part = dist(gen);
            double imag_part = dist(gen);
            out[i] = std::complex<double>(real_part, imag_part);
        }
    }

    // Step 2: QR decomposition (zgeqrfp)
    std::vector<std::complex<double>> tau(N);
    std::complex<double> work_size;

    // Query optimal workspace size
    zgeqrfp_wrapper(N, N, &out, N, &tau, &work_size, -1);
    int lwork = static_cast<int>(work_size.real());
    std::vector<std::complex<double>> work(lwork);

    // Actual QR computation
    zgeqrfp_wrapper(N, N, &out, N, &tau, &work, lwork);

    // Step 3: Generate unitary matrix Q (zungqr)
    zungqr_wrapper(N, N, N, &out, N, &tau, &work_size, -1);
    lwork = static_cast<int>(work_size.real());
    work.resize(lwork);
    zungqr_wrapper(N, N, N, &out, N, &tau, &work, lwork);

    return out;
}
