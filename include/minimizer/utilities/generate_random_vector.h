#ifndef GENERATE_RANDOM_VECTOR_H
#define GENERATE_RANDOM_VECTOR_H

#include <vector>
#include <complex>

#include "minimizer/core/cuda_traits.h"

std::vector<std::complex<double> >* generateUniformRandomVector(int N);

template<typename T>
cudaError_t generateUniformRandomVectorsCuda(typename CudaTraits<T>::Complex* v, int N, int num_vectors);

#endif // GENERATE_RANDOM_VECTOR_H