#ifndef LAPACK_INTERFACE_H
#define LAPACK_INTERFACE_H

// This file provides a minimal LAPACK interface for the Clebsch-Gordan library.
// It handles platform-specific LAPACK configuration without depending on the project's common_includes.h

// Standard C++ headers needed by sud_cg.cpp
#include <cmath>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <vector>
#include <map>
#include <functional>
#include <numeric>

// LAPACK backend-specific headers
#ifdef LAPACK_ACCELERATE
    #include <Accelerate/Accelerate.h>
#elif defined(LAPACK_MKL)
    #include <mkl.h>
#elif defined(LAPACK_OPENBLAS)
    #include <cblas.h>
    #include <lapacke.h>
#elif defined(LAPACK_AMD)
    #include <cblas.h>
    #include <lapacke.h>
#else
    // If no backend is defined, use standard LAPACK Fortran interface
    // (This allows standalone compilation without explicit backend selection)
    // This is the most portable fallback for systems with system LAPACK
    extern "C" {
        // SVD decomposition
        void dgesvd_(const char *jobu, const char *jobvt, int *m, int *n,
                    double *a, int *lda, double *s, double *u, int *ldu,
                    double *vt, int *ldvt, double *work, int *lwork,
                    int *info, int jobu_len, int jobvt_len);

        // General least squares
        void dgels_(const char *trans, int *m, int *n, int *nrhs,
                   double *a, int *lda, double *b, int *ldb,
                   double *work, int *lwork, int *info, int trans_len);
    }
#endif

#endif // LAPACK_INTERFACE_H
