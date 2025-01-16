#ifndef COMMON_INCLUDES_H
#define COMMON_INCLUDES_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <vector>
#include <random>
#include <complex>
#include <algorithm>
#include <stdexcept>

#include <cmath>
#include <filesystem> // For accessing, creating folders etc.; also, path manipulation

// These two could need to be installed
// In order to make the code platform agnostic, we check the desired library and include the appropriate header

#ifdef LAPACK_ACCELERATE
#include <Accelerate/Accelerate.h>
typedef __LAPACK_double_complex lapack_complex_t;
#elif defined(LAPACK_MKL)
#include <mkl.h>
typedef lapack_complex_double lapack_complex_t;
#elif defined(LAPACK_OPENBLAS)
#include <cblas.h>
#include <lapacke.h>
typedef lapack_complex_double lapack_complex_t;
#else
#error "No LAPACK backend defined. Define LAPACK_ACCELERATE, LAPACK_MKL, or LAPACK_OPENBLAS."
#endif


#endif // COMMON_INCLUDES_H