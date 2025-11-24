#include "compute/backends/cuda/CudaSVDSolver.h"
#include "compute/backends/cuda/CudaMemory.h"
#include "compute/backends/cuda/CudaStream.h"
#include "utilities/cuda/error_handling.h"
#include <cuComplex.h>
#include <stdexcept>
#include <algorithm>
#include <iostream>

CudaSVDSolver::CudaSVDSolver(
    IComputeDevice* device,
    int m, int n,
    const SVDSpec& spec,
    PrecisionType precision
) : m_(m), n_(n), spec_(spec), precision_(precision),
    workspace_device_bytes_(0), workspace_host_bytes_(0),
    device_(device), cusolver_handle_(nullptr), cusolver_params_(nullptr),
    devinfo_(nullptr), h_err_sigma_(0.0), original_stream_(nullptr)
{
    // Validate dimensions
    if (m_ <= 0 || n_ <= 0) {
        throw std::invalid_argument(
            "CudaSVDSolver: invalid dimensions m=" + std::to_string(m_) +
            ", n=" + std::to_string(n_) + " (must be > 0)"
        );
    }
    
    // Create cuSOLVER handle
    cusolverStatus_t status = cusolverDnCreate(&cusolver_handle_);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            "CudaSVDSolver: Failed to create cuSOLVER handle, status=" + std::to_string(status)
        );
    }
    
    // Create cuSOLVER params
    status = cusolverDnCreateParams(&cusolver_params_);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cusolverDnDestroy(cusolver_handle_);
        throw std::runtime_error(
            "CudaSVDSolver: Failed to create cuSOLVER params, status=" + std::to_string(status)
        );
    }
    
    // Allocate devInfo (small, owned by this instance)
    devinfo_ = std::make_unique<CudaMemory>(sizeof(int));
    
    // Query workspace requirements
    queryWorkspace();
}

CudaSVDSolver::~CudaSVDSolver() {
    if (cusolver_params_) {
        cusolverDnDestroyParams(cusolver_params_);
    }
    if (cusolver_handle_) {
        cusolverDnDestroy(cusolver_handle_);
    }
}

void CudaSVDSolver::queryWorkspace() {
    // Map SVDVectors to cuSOLVER parameters
    cusolverEigMode_t jobz = (spec_.vectors == SVDVectors::NONE) 
        ? CUSOLVER_EIG_MODE_NOVECTOR 
        : CUSOLVER_EIG_MODE_VECTOR;
    
    int econ = (spec_.vectors == SVDVectors::THIN) ? 1 : 0;
    
    cudaDataType dtype = (precision_ == PrecisionType::DOUBLE) ? CUDA_C_64F : CUDA_C_32F;
    cudaDataType dtype_real = (precision_ == PrecisionType::DOUBLE) ? CUDA_R_64F : CUDA_R_32F;
    
    int64_t lda = m_;
    int64_t ldu = m_;
    int64_t ldv = n_;
    
    // Choose algorithm and query workspace
    SVDAlgorithm effective_algorithm = spec_.algorithm;
    
    // Handle QR algorithm limitations: cusolverDnXgesvd only works well for m>=n
    // For wide matrices (m < n), fall back to POLAR which handles both cases
    if (spec_.algorithm == SVDAlgorithm::QR && m_ < n_) {
        std::cout << "Warning: QR algorithm requested for wide matrix (" << m_ << "x" << n_ 
                  << "). Falling back to POLAR algorithm." << std::endl;
        effective_algorithm = SVDAlgorithm::POLAR;
        spec_.algorithm = SVDAlgorithm::POLAR;  // Update spec
    }
    
    // cusolverDnXgesvdp does NOT support CUSOLVER_EIG_MODE_NOVECTOR
    // If POLAR is requested with NONE vectors, fallback to QR
    if (effective_algorithm == SVDAlgorithm::POLAR && spec_.vectors == SVDVectors::NONE) {
        std::cout << "Warning: POLAR algorithm does not support computing singular values only. "
                  << "Falling back to QR algorithm." << std::endl;
        effective_algorithm = SVDAlgorithm::QR;
    }
    
    if (effective_algorithm == SVDAlgorithm::AUTO || effective_algorithm == SVDAlgorithm::QR) {
        // Use cusolverDnXgesvd (QR-based)
        int64_t ldvt = (spec_.vectors == SVDVectors::ALL) ? n_ : 
                       (spec_.vectors == SVDVectors::THIN) ? std::min(m_, n_) : 1;
        
        CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
            cusolver_handle_, cusolver_params_,
            (spec_.vectors == SVDVectors::NONE) ? 'N' : 
            (spec_.vectors == SVDVectors::ALL) ? 'A' : 'S',  // jobu
            (spec_.vectors == SVDVectors::NONE) ? 'N' : 
            (spec_.vectors == SVDVectors::ALL) ? 'A' : 'S',  // jobvt
            m_, n_,
            dtype, nullptr, lda,
            dtype_real, nullptr,
            dtype, nullptr, ldu,
            dtype, nullptr, ldvt,
            dtype,
            &workspace_device_bytes_,
            &workspace_host_bytes_
        ));
    }
    else if (effective_algorithm == SVDAlgorithm::POLAR) {
        // Use cusolverDnXgesvdp (polar decomposition)
        CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
            cusolver_handle_, cusolver_params_,
            jobz, econ,
            m_, n_,
            dtype, nullptr, lda,
            dtype_real, nullptr,
            dtype, nullptr, ldu,
            dtype, nullptr, ldv,
            dtype,
            &workspace_device_bytes_,
            &workspace_host_bytes_
        ));
    }
    else if (effective_algorithm == SVDAlgorithm::RANDOMIZED) {
        // Use cusolverDnXgesvdr (randomized)
        int rank = (spec_.rank <= 0) ? std::min(m_, n_) : spec_.rank;
        int oversampling = (spec_.oversampling < 0) ? 
            std::min(2 * rank, std::min(m_, n_) - rank) : spec_.oversampling;
        
        CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
            cusolver_handle_, cusolver_params_,
            (spec_.vectors == SVDVectors::NONE) ? 'N' : 'S',  // jobu
            (spec_.vectors == SVDVectors::NONE) ? 'N' : 'S',  // jobv
            m_, n_,
            rank, oversampling, spec_.power_iterations,
            dtype, nullptr, lda,
            dtype_real, nullptr,
            dtype, nullptr, ldu,
            dtype, nullptr, ldv,
            dtype,
            &workspace_device_bytes_,
            &workspace_host_bytes_
        ));
    }
}

void CudaSVDSolver::compute(void* A, void* S, void* U, void* VT, IStream* stream) {
    // Save original stream
    CUSOLVER_CHECK(cusolverDnGetStream(cusolver_handle_, &original_stream_));
    
    // Set stream if provided
    if (stream) {
        auto* cudaStream = dynamic_cast<CudaStream*>(stream);
        if (cudaStream) {
            CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_, cudaStream->getCudaStream()));
        }
    }
    
    // Determine effective algorithm (handle fallbacks)
    SVDAlgorithm effective_algorithm = spec_.algorithm;
    if (spec_.algorithm == SVDAlgorithm::QR && m_ < n_) {
        effective_algorithm = SVDAlgorithm::POLAR;  // Already warned in queryWorkspace
    }
    if (effective_algorithm == SVDAlgorithm::POLAR && spec_.vectors == SVDVectors::NONE) {
        effective_algorithm = SVDAlgorithm::QR;  // Already warned in queryWorkspace
    }
    
    // Dispatch to appropriate algorithm
    if (effective_algorithm == SVDAlgorithm::AUTO || effective_algorithm == SVDAlgorithm::QR) {
        computeQR(A, S, U, VT);
    }
    else if (effective_algorithm == SVDAlgorithm::POLAR) {
        computePolar(A, S, U, VT);
    }
    else if (effective_algorithm == SVDAlgorithm::RANDOMIZED) {
        computeRandomized(A, S, U, VT);
    }
    
    // Synchronize and check error
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_devinfo;
    devinfo_->copyToHost(&h_devinfo, sizeof(int));
    if (h_devinfo != 0) {
        throw std::runtime_error(
            "cuSOLVER SVD failed with info = " + std::to_string(h_devinfo)
        );
    }
    
    // Restore original stream
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_, original_stream_));
}

void CudaSVDSolver::computeQR(void* A, void* S, void* U, void* VT) {
    // Get scratch pools
    auto* dev_scratch = device_->getDeviceScratch();
    auto* host_scratch = device_->getHostScratch();
    
    if (!dev_scratch) {
        throw std::runtime_error("CUDA backend requires device scratch memory");
    }
    
    // Request workspace
    void* d_work = dev_scratch->request(workspace_device_bytes_);
    void* h_work = (workspace_host_bytes_ > 0) ? host_scratch->request(workspace_host_bytes_) : nullptr;
    
    // Map SVDVectors to jobu/jobvt characters
    signed char jobu, jobvt;
    if (spec_.vectors == SVDVectors::NONE) {
        jobu = jobvt = 'N';
    } else if (spec_.vectors == SVDVectors::ALL) {
        jobu = jobvt = 'A';
    } else {  // THIN
        jobu = jobvt = 'S';
    }
    
    // Calculate leading dimensions
    int64_t lda = m_;
    int64_t ldu = (jobu == 'A' || jobu == 'S') ? m_ : 1;
    int64_t ldvt = (jobvt == 'A') ? n_ : 
                   (jobvt == 'S') ? std::min(m_, n_) : 1;
    
    cudaDataType dtype = (precision_ == PrecisionType::DOUBLE) ? CUDA_C_64F : CUDA_C_32F;
    cudaDataType dtype_real = (precision_ == PrecisionType::DOUBLE) ? CUDA_R_64F : CUDA_R_32F;
    
    // Initialize devInfo
    int zero = 0;
    devinfo_->copyFromHost(&zero, sizeof(int));
    
    // Call cusolverDnXgesvd
    CUSOLVER_CHECK(cusolverDnXgesvd(
        cusolver_handle_, cusolver_params_,
        jobu, jobvt,
        m_, n_,
        dtype, A, lda,
        dtype_real, S,
        dtype, U, ldu,
        dtype, VT, ldvt,
        dtype,
        d_work, workspace_device_bytes_,
        h_work, workspace_host_bytes_,
        static_cast<int*>(devinfo_->data())
    ));
}

void CudaSVDSolver::computePolar(void* A, void* S, void* U, void* VT) {
    // Get scratch pools
    auto* dev_scratch = device_->getDeviceScratch();
    auto* host_scratch = device_->getHostScratch();
    
    if (!dev_scratch) {
        throw std::runtime_error("CUDA backend requires device scratch memory");
    }
    
    // Request workspace
    void* d_work = dev_scratch->request(workspace_device_bytes_);
    void* h_work = (workspace_host_bytes_ > 0) ? host_scratch->request(workspace_host_bytes_) : nullptr;
    
    // Map SVDVectors to cuSOLVER parameters
    cusolverEigMode_t jobz = (spec_.vectors == SVDVectors::NONE) 
        ? CUSOLVER_EIG_MODE_NOVECTOR 
        : CUSOLVER_EIG_MODE_VECTOR;
    int econ = (spec_.vectors == SVDVectors::THIN) ? 1 : 0;
    
    // Calculate leading dimensions
    int64_t lda = m_;
    int64_t ldu = m_;
    int64_t ldv = n_;
    
    cudaDataType dtype = (precision_ == PrecisionType::DOUBLE) ? CUDA_C_64F : CUDA_C_32F;
    cudaDataType dtype_real = (precision_ == PrecisionType::DOUBLE) ? CUDA_R_64F : CUDA_R_32F;
    
    // Initialize devInfo and h_err_sigma
    int zero = 0;
    devinfo_->copyFromHost(&zero, sizeof(int));
    h_err_sigma_ = 0.0;
    
    // Call cusolverDnXgesvdp
    // NOTE: gesvdp returns V, not V^H (unlike gesvd which returns V^H)
    CUSOLVER_CHECK(cusolverDnXgesvdp(
        cusolver_handle_, cusolver_params_,
        jobz, econ,
        m_, n_,
        dtype, A, lda,
        dtype_real, S,
        dtype, U, ldu,
        dtype, VT, ldv,  // This is actually V, not V^H!
        dtype,
        d_work, workspace_device_bytes_,
        h_work, workspace_host_bytes_,
        static_cast<int*>(devinfo_->data()),
        &h_err_sigma_
    ));
}

void CudaSVDSolver::computeRandomized(void* A, void* S, void* U, void* VT) {
    // Get scratch pools
    auto* dev_scratch = device_->getDeviceScratch();
    auto* host_scratch = device_->getHostScratch();
    
    if (!dev_scratch) {
        throw std::runtime_error("CUDA backend requires device scratch memory");
    }
    
    // Request workspace
    void* d_work = dev_scratch->request(workspace_device_bytes_);
    void* h_work = (workspace_host_bytes_ > 0) ? host_scratch->request(workspace_host_bytes_) : nullptr;
    
    // Extract randomized SVD parameters
    int rank = (spec_.rank <= 0) ? std::min(m_, n_) : spec_.rank;
    int oversampling = (spec_.oversampling < 0) ? 
        std::min(2 * rank, std::min(m_, n_) - rank) : spec_.oversampling;
    
    // Map SVDVectors to jobu/jobv characters
    signed char jobu, jobv;
    if (spec_.vectors == SVDVectors::NONE) {
        jobu = jobv = 'N';
    } else {
        jobu = jobv = 'S';  // Randomized only supports 'S' or 'N'
    }
    
    // Calculate leading dimensions
    int64_t lda = m_;
    int64_t ldu = (jobu == 'S') ? m_ : 1;
    int64_t ldv = (jobv == 'S') ? n_ : 1;
    
    cudaDataType dtype = (precision_ == PrecisionType::DOUBLE) ? CUDA_C_64F : CUDA_C_32F;
    cudaDataType dtype_real = (precision_ == PrecisionType::DOUBLE) ? CUDA_R_64F : CUDA_R_32F;
    
    // Initialize devInfo
    int zero = 0;
    devinfo_->copyFromHost(&zero, sizeof(int));
    
    // Call cusolverDnXgesvdr
    // NOTE: gesvdr also returns V, not V^H
    CUSOLVER_CHECK(cusolverDnXgesvdr(
        cusolver_handle_, cusolver_params_,
        jobu, jobv,
        m_, n_,
        rank, oversampling, spec_.power_iterations,
        dtype, A, lda,
        dtype_real, S,
        dtype, U, ldu,
        dtype, VT, ldv,  // This is actually V, not V^H!
        dtype,
        d_work, workspace_device_bytes_,
        h_work, workspace_host_bytes_,
        static_cast<int*>(devinfo_->data())
    ));
}

void CudaSVDSolver::setSpec(const SVDSpec& spec) {
    spec_ = spec;
    // Re-query workspace as algorithm or vector mode may have changed
    queryWorkspace();
}

void CudaSVDSolver::getWorkspaceSizes(size_t& device_bytes, size_t& host_bytes) const {
    device_bytes = workspace_device_bytes_;
    host_bytes = workspace_host_bytes_;
}
