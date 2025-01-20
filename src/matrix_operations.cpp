#include "matrix_operations.h"
#include "common_includes.h"


void zgeqrfp_wrapper(int M, int N, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::complex<double>* work, int lwork){
    #ifdef LAPACK_ACCELERATE
        int info = 0;
        zgeqrf_(&M, &N, reinterpret_cast<lapack_complex_t*>(A->data()),&lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work), &lwork, &info);
    #elif (defined(LAPACK_MKL) || defined(LAPACK_OPENBLAS))
        LAPACKE_zgeqrfp_work(LAPACK_COL_MAJOR, M, N, reinterpret_cast<lapack_complex_t*>(A->data()),lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work), lwork);
    #endif
}

void zgeqrfp_wrapper(int M, int N, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::vector<std::complex<double> >* work, int lwork){
    #ifdef LAPACK_ACCELERATE
        int info = 0;
        zgeqrf_(&M, &N, reinterpret_cast<lapack_complex_t*>(A->data()),&lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work->data()), &lwork, &info);
    #elif defined(LAPACK_MKL) || defined(LAPACK_OPENBLAS)
        LAPACKE_zgeqrfp_work(LAPACK_COL_MAJOR, M, N, reinterpret_cast<lapack_complex_t*>(A->data()),lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work->data()), lwork);
    #endif
}

void zungqr_wrapper(int M, int N, int K, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::complex<double>* work, int lwork){
    #ifdef LAPACK_ACCELERATE
        int info = 0;
        zungqr_(&M, &N, &K, reinterpret_cast<lapack_complex_t*>(A->data()),&lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work), &lwork, &info);
    #elif defined(LAPACK_MKL) || defined(LAPACK_OPENBLAS)
        LAPACKE_zungqr_work(LAPACK_COL_MAJOR, M, N, K, reinterpret_cast<lapack_complex_t*>(A->data()),lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work), lwork);
    #endif
}

void zungqr_wrapper(int M, int N, int K, std::vector<std::complex<double> >* A, int lda, std::vector<std::complex<double> >* tau, std::vector<std::complex<double> >* work, int lwork){
    #ifdef LAPACK_ACCELERATE
        int info = 0;
        zungqr_(&M, &N, &K, reinterpret_cast<lapack_complex_t*>(A->data()),&lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work->data()), &lwork, &info);
    #elif defined(LAPACK_MKL) || defined(LAPACK_OPENBLAS)
        LAPACKE_zungqr_work(LAPACK_COL_MAJOR, M, N, K, reinterpret_cast<lapack_complex_t*>(A->data()),lda,reinterpret_cast<lapack_complex_t*>(tau->data()),reinterpret_cast<lapack_complex_t*>(work->data()), lwork);
    #endif
}

void zheev_wrapper(char jobz, char uplo, int N, std::vector<std::complex<double> >* A, int lda, std::vector<double>* w){
    #ifdef LAPACK_ACCELERATE
    int info = 0;
    int lwork = -1;  // First query to get optimal work size
    int lrwork = 3 * N - 1;  // Allocate real workspace

    // For Accelerate, use work and rwork arrays
    std::vector<lapack_complex_t> work(1);
    std::vector<double> rwork(lrwork);
    
    // Query optimal work size
    // Accelerate uses LAPACK zheev
    zheev_(&jobz, &uplo, &N, reinterpret_cast<lapack_complex_t*>(A->data()), &lda, w->data(),
            reinterpret_cast<lapack_complex_t*>(work.data()), &lwork, rwork.data(), &info);

    if (info != 0) {
        std::cerr << "Error during workspace query: info = " << info << std::endl;
        return;
    }

    // Get the optimal work size and allocate appropriate space
    lwork = static_cast<int>(work[0].real());  // Assuming the optimal size is returned as a real number

    // Allocate the workspace arrays with optimal size
    work.resize(lwork);
    
    // Now run the actual zheev computation
    zheev_(&jobz, &uplo, &N, reinterpret_cast<lapack_complex_t*>(A->data()), &lda, w->data(),
            reinterpret_cast<lapack_complex_t*>(work.data()), &lwork, rwork.data(), &info);
    #elif defined(LAPACK_MKL) || defined(LAPACK_OPENBLAS)
        LAPACKE_zheev(LAPACK_COL_MAJOR, jobz, uplo, N, reinterpret_cast<lapack_complex_t*>(A->data()), lda, w->data());
    #endif
}

void dgesv_wrapper(int N, int NRHS, double* A, int lda, int* ipiv, double* B, int ldb){
    #ifdef LAPACK_ACCELERATE
        int info = 0;
        dgesv_(&N, &NRHS, A, &lda, ipiv, B, &ldb, &info);
    #elif defined(LAPACK_MKL) || defined(LAPACK_OPENBLAS)
        LAPACKE_dgesv(LAPACK_COL_MAJOR, N, NRHS, A, lda, ipiv, B, ldb);
    #endif
}

