#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "common_includes.h"
/*
Assumes col major ordering!
*/
void zgeqrfp_wrapper(int M, int N, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::complex<double>* work, int lwork);
void zgeqrfp_wrapper(int M, int N, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::vector<std::complex<double> >* work, int lwork);
void zungqr_wrapper(int M, int N, int K, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::complex<double>* work, int lwork);
void zungqr_wrapper(int M, int N, int K, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::vector<std::complex<double> >* work, int lwork);
void zheev_wrapper(char jobz, char uplo, int N, std::vector<std::complex<double> >* A, int lda, std::vector<double>* w);
#endif // MATRIX_OPS_H