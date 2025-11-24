#include "compute/backends/cpu/CpuSVDSolver.h"
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

// LAPACK declarations
extern "C" {
    // Double precision complex SVD
    void zgesvd_(const char* jobu, const char* jobvt,
                 const int* m, const int* n,
                 std::complex<double>* A, const int* lda,
                 double* S,
                 std::complex<double>* U, const int* ldu,
                 std::complex<double>* VT, const int* ldvt,
                 std::complex<double>* work, const int* lwork,
                 double* rwork,
                 int* info);
    
    // Single precision complex SVD
    void cgesvd_(const char* jobu, const char* jobvt,
                 const int* m, const int* n,
                 std::complex<float>* A, const int* lda,
                 float* S,
                 std::complex<float>* U, const int* ldu,
                 std::complex<float>* VT, const int* ldvt,
                 std::complex<float>* work, const int* lwork,
                 float* rwork,
                 int* info);
}

CpuSVDSolver::CpuSVDSolver(
    IComputeDevice* device,
    int m, int n,
    const SVDSpec& spec,
    PrecisionType precision
)
    : m_(m)
    , n_(n)
    , spec_(spec)
    , precision_(precision)
    , workspace_bytes_(0)
    , device_(device)
{
    if (m <= 0 || n <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    if (!device) {
        throw std::invalid_argument("Device pointer cannot be null");
    }
    
    // Query workspace
    queryWorkspace();
}

void CpuSVDSolver::queryWorkspace() {
    // Determine job characters based on spec
    char jobu, jobvt;
    
    if (spec_.vectors == SVDVectors::NONE) {
        jobu = jobvt = 'N';
    } else if (spec_.vectors == SVDVectors::THIN) {
        jobu = jobvt = 'S';
    } else {  // ALL
        jobu = jobvt = 'A';
    }
    
    int lda = m_;
    int ldu = (jobu == 'A') ? m_ : (jobu == 'S') ? m_ : 1;
    int ldvt = (jobvt == 'A') ? n_ : (jobvt == 'S') ? std::min(m_, n_) : 1;
    int info = 0;
    
    // Query workspace with lwork = -1
    int lwork = -1;
    
    if (precision_ == PrecisionType::DOUBLE) {
        std::complex<double> work_query;
        double* dummy_rwork = nullptr;
        
        zgesvd_(&jobu, &jobvt, &m_, &n_,
                nullptr, &lda,
                nullptr,
                nullptr, &ldu,
                nullptr, &ldvt,
                &work_query, &lwork,
                dummy_rwork,
                &info);
        
        if (info != 0) {
            throw std::runtime_error("LAPACK workspace query failed with info = " + std::to_string(info));
        }
        
        int optimal_lwork = static_cast<int>(work_query.real());
        workspace_bytes_ = optimal_lwork * sizeof(std::complex<double>);
    } else {
        std::complex<float> work_query;
        float* dummy_rwork = nullptr;
        
        cgesvd_(&jobu, &jobvt, &m_, &n_,
                nullptr, &lda,
                nullptr,
                nullptr, &ldu,
                nullptr, &ldvt,
                &work_query, &lwork,
                dummy_rwork,
                &info);
        
        if (info != 0) {
            throw std::runtime_error("LAPACK workspace query failed with info = " + std::to_string(info));
        }
        
        int optimal_lwork = static_cast<int>(work_query.real());
        workspace_bytes_ = optimal_lwork * sizeof(std::complex<float>);
    }
}

void CpuSVDSolver::compute(void* A, void* S, void* U, void* VT, IStream* stream) {
    // CPU is synchronous, ignore stream
    (void)stream;
    
    computeLAPACK(A, S, U, VT);
}

void CpuSVDSolver::computeLAPACK(void* A, void* S, void* U, void* VT) {
    auto* scratch = device_->getDeviceScratch();
    if (!scratch) {
        throw std::runtime_error("CPU backend requires scratch memory");
    }
    
    // Request workspace
    void* work = scratch->request(workspace_bytes_);
    
    // Determine job characters
    char jobu, jobvt;
    
    if (spec_.vectors == SVDVectors::NONE) {
        jobu = jobvt = 'N';
    } else if (spec_.vectors == SVDVectors::THIN) {
        jobu = jobvt = 'S';
    } else {  // ALL
        jobu = jobvt = 'A';
    }
    
    int lda = m_;
    int ldu = (jobu == 'A') ? m_ : (jobu == 'S') ? m_ : 1;
    int ldvt = (jobvt == 'A') ? n_ : (jobvt == 'S') ? std::min(m_, n_) : 1;
    int info = 0;
    int min_mn = std::min(m_, n_);
    
    if (precision_ == PrecisionType::DOUBLE) {
        int lwork = workspace_bytes_ / sizeof(std::complex<double>);
        
        // Allocate rwork (real workspace for complex routines)
        std::vector<double> rwork(5 * min_mn);
        
        zgesvd_(&jobu, &jobvt, &m_, &n_,
                static_cast<std::complex<double>*>(A), &lda,
                static_cast<double*>(S),
                static_cast<std::complex<double>*>(U), &ldu,
                static_cast<std::complex<double>*>(VT), &ldvt,
                static_cast<std::complex<double>*>(work), &lwork,
                rwork.data(),
                &info);
    } else {
        int lwork = workspace_bytes_ / sizeof(std::complex<float>);
        
        // Allocate rwork (real workspace for complex routines)
        std::vector<float> rwork(5 * min_mn);
        
        cgesvd_(&jobu, &jobvt, &m_, &n_,
                static_cast<std::complex<float>*>(A), &lda,
                static_cast<float*>(S),
                static_cast<std::complex<float>*>(U), &ldu,
                static_cast<std::complex<float>*>(VT), &ldvt,
                static_cast<std::complex<float>*>(work), &lwork,
                rwork.data(),
                &info);
    }
    
    if (info != 0) {
        if (info < 0) {
            throw std::runtime_error("LAPACK zgesvd: argument " + std::to_string(-info) + " had an illegal value");
        } else {
            throw std::runtime_error("LAPACK zgesvd: failed to converge, info = " + std::to_string(info));
        }
    }
}

void CpuSVDSolver::setSpec(const SVDSpec& spec) {
    spec_ = spec;
    queryWorkspace();
}

void CpuSVDSolver::getWorkspaceSizes(size_t& device_bytes, size_t& host_bytes) const {
    device_bytes = workspace_bytes_;  // CPU uses "device" scratch for workspace
    host_bytes = 0;                   // No separate host workspace needed
}
