#ifndef GENERATE_HAAR_UNITARY_H
#define GENERATE_HAAR_UNITARY_H

#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>

std::vector<std::complex<double> > generateHaarRandomUnitary(int N);
cudaError_t generateHaarRandomUnitaries(cuDoubleComplex *d_A, int N, int num_matrices, int num_streams);

#endif // GENERATE_HAAR_UNITARY_H