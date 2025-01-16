#include "common_includes.h"
// My headers
#include "minimizer.h"
#include "entropy_minimizer.h"
#include "message_handler.h"
#include "logger.h"
#include "config.h"
#include "matrix_operations.h"

const int d = 3;
const int N = 512;

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


int main() {

    // Check if Accelerate is running in single-threaded or multi-threaded mode
    #ifdef LAPACK_ACCELERATE
    BLAS_THREADING mode = BLASGetThreading();
    if (mode == BLAS_THREADING_SINGLE_THREADED) {
        std::cout << "BLAS is running in single-threaded mode." << std::endl;
    } else if (mode == BLAS_THREADING_MULTI_THREADED) {
        std::cout << "BLAS is running in multi-threaded mode." << std::endl;
    }
    #endif

    // Step 1: Generate the Kraus operators of the random unitary channel
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

    // Step 2: initialize the minimizer that will take care of finding the MOE
    // Config setup
    EntropyConfig config = EntropyConfig();
    config.setLogging(false);
    config.setLogFile("loglog.log");
    config.setPrinting(true);
    // Actual minimizer
    EntropyMinimizer minimizer = EntropyMinimizer(kraus_operators, d, N, N, &config); 

    // Step 3: get a random initial vector for the minimizer
    minimizer.minimizer->initializeRandomVector();


    // Step 4: run a minimization run
    minimizer.runMinimization();

    // Step 5: free memory
    delete kraus_operators;
    return 0;
}

