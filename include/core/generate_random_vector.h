#ifndef GENERATE_RANDOM_VECTOR_H
#define GENERATE_RANDOM_VECTOR_H

#include "common_includes.h"
#include "cuComplex.h"

std::vector<std::complex<double> >* generateUniformRandomVector(int N);
cudaError_t generateUniformRandomVectorsCuda(cuDoubleComplex* v, int N, int num_vectors);

#endif // GENERATE_RANDOM_VECTOR_H