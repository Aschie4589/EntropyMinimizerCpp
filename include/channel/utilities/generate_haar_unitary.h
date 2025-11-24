#ifndef GENERATE_HAAR_UNITARY_H
#define GENERATE_HAAR_UNITARY_H

#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>

// Forward declaration
class MessageHandler;

cudaError_t generateHaarRandomUnitaries(
    cuDoubleComplex *d_A, 
    int N, 
    int num_matrices, 
    int num_streams,
    MessageHandler* msg_handler = nullptr
);

#endif // GENERATE_HAAR_UNITARY_H