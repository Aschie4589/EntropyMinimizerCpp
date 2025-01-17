// My headers
#include "common_includes.h"
#include "config.h"

#include "minimizer.h"
#include "entropy_minimizer.h"

#include "message_handler.h"
#include "logger.h"
#include "matrix_operations.h"

#include "generate_haar_unitary.h"

#include "vector_serializer.h"

const int d = 10;
const int N = 50;


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
    // minimizer.runMinimization();

    // Step 4-b: save kraus operators
    VectorSerializer serializer = VectorSerializer();
    //serializer.serializeVector(kraus_operators, "kraus_operators.dat");

    // Step 4-c: load kraus operators
    std::vector<std::complex<double> >* loaded_kraus_operators = serializer.deserializeVector("kraus_operators.dat");

    // print kraus vectors
    for (int i = 0; i < d; i++){
        std::cout << "Kraus operator " << i << std::endl;
        for (int j = 0; j < N*N; j++){
            std::cout << loaded_kraus_operators->at(i*N*N+j) << " ";
        }
        std::cout << std::endl;
    }

    // Step 5: free memory
    delete kraus_operators;
    return 0;
}

