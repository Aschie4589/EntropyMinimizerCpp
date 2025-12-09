#include "minimizer/algorithm/generic_minimization_strategy.h"
#include "compute/linalg/ILinearAlgebra.h"
#include "compute/linalg/ISolver.h"
#include "compute/linalg/SVDTypes.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// debug
#include <iostream>
// ============================================================================
// GenericMinimizationStrategy Implementation
// ============================================================================

GenericMinimizationStrategy::GenericMinimizationStrategy(
    IComputeDevice& device,
    int num_streams
)
    : device_(device),
      precision_(PrecisionType::DOUBLE),
      num_streams_(num_streams),
      kraus_count_(0),
      input_dim_(0),
      output_dim_(0),
      epsilon_(0.0),
      initialized_(false)
{
    if (num_streams_ < 1) {
        throw std::invalid_argument(
            "num_streams must be >= 1, got " + std::to_string(num_streams_)
        );
    }
}

void GenericMinimizationStrategy::initialize(
    int kraus_count,
    int input_dim,
    int output_dim,
    double epsilon,
    PrecisionType precision
) {
    // Validate dimensions
    if (kraus_count > output_dim) {
        throw std::runtime_error(
            "Invalid dimensions: kraus_count (" + std::to_string(kraus_count) +
            ") > output_dim (" + std::to_string(output_dim) +
            "). SVD requires d <= M."
        );
    }

    // Validate dimensions
    if (kraus_count*kraus_count > input_dim) {
        throw std::runtime_error(
            "Invalid dimensions: kraus_count^2 (" + std::to_string(kraus_count*kraus_count) +
            ") > input_dim (" + std::to_string(input_dim) +
            "). SVD requires d^2 <= N."
        );
    }

    // Validate epsilon    
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::runtime_error(
            "Epsilon must be in (0, 1), got " + std::to_string(epsilon)
        );
    }
    
    // Store configuration
    precision_ = precision;
    kraus_count_ = kraus_count;
    input_dim_ = input_dim;
    output_dim_ = output_dim;
    epsilon_ = epsilon;
    
    // Dispatch to templated implementation
    if (precision_ == PrecisionType::FLOAT) {
        initializeImpl<float>(kraus_count, input_dim, output_dim, epsilon);
    } else {
        initializeImpl<double>(kraus_count, input_dim, output_dim, epsilon);
    }
    
    initialized_ = true;
}

double GenericMinimizationStrategy::stepOnce(
    IDeviceMemory* d_kraus,
    IDeviceMemory* d_current_vector
) {
    if (!initialized_) {
        throw std::runtime_error("stepOnce called before initialize");
    }
    if (precision_ == PrecisionType::FLOAT) {
        return stepOnceImpl<float>(d_kraus, d_current_vector);
    } else {
        return stepOnceImpl<double>(d_kraus, d_current_vector);
    }
}

PrecisionType GenericMinimizationStrategy::getPrecision() const {
    return precision_;
}

size_t GenericMinimizationStrategy::getWorkspaceSize() const {
    if (!initialized_) {
        return 0;
    }
    
    size_t complex_size = (precision_ == PrecisionType::FLOAT) ?
                          sizeof(std::complex<float>) :
                          sizeof(std::complex<double>);
    
    size_t real_size = (precision_ == PrecisionType::FLOAT) ?
                       sizeof(float) : sizeof(double);
    
    const int d = kraus_count_;
    const int N = input_dim_;
    const int M = output_dim_;
    
    // Intermediate vectors from step 1: d × M
    size_t vecs1_mem = d * M * complex_size;
    size_t singvecs1_mem = d * M * complex_size;
    
    // Intermediate vectors from step 2: d² × N
    size_t vecs2_mem = d * d * N * complex_size;
    size_t singvecs2_mem = N * complex_size;

    // Singular values: d + d²
    size_t sv_mem = (d + d * d) * real_size;

    // SVD workspace
    size_t svd_1_host, svd_1_device;
    if (svd_solver_1_) {
        svd_solver_1_->getWorkspaceSizes(svd_1_host, svd_1_device);
    } else {
        svd_1_device = 0;
    }
    
    size_t svd_2_host, svd_2_device;
    if (svd_solver_2_) {
        svd_solver_2_->getWorkspaceSizes(svd_2_host, svd_2_device);
    } else {
        svd_2_device = 0;
    }
    
    return vecs1_mem + singvecs1_mem + vecs2_mem + singvecs2_mem + sv_mem + svd_1_device + svd_2_device;
}

// ============================================================================
// Templated Implementation Methods
// ============================================================================

template<typename T>
void GenericMinimizationStrategy::initializeImpl(
    int kraus_count,
    int input_dim,
    int output_dim,
    double epsilon
) {
    using Complex = std::complex<T>;
    using Real = T;
    
    const int d = kraus_count;
    const int N = input_dim;
    const int M = output_dim;
    
    // Allocate workspace memory only (no Kraus or current vector)
    d_vecs_1_ = device_.allocate(d * M * sizeof(Complex));
    d_sing_1_ = device_.allocate(d * M * sizeof(Complex));  // U matrix from SVD1: M × min(M,d)
    d_vecs_2_ = device_.allocate(d * d * N * sizeof(Complex));
    
    // SVD2 produces U matrix of size N × min(N, d²)
    // Since we use THIN vectors: min(N, d*d) 
    int svd2_u_cols = std::min(N, d * d);
    d_sing_2_ = device_.allocate(N * svd2_u_cols * sizeof(Complex));
    
    d_sv_1_ = device_.allocate(d * sizeof(Real));
    d_sv_2_ = device_.allocate(d * d * sizeof(Real));
    
    // Create streams for parallel operations
    streams_.reserve(num_streams_);
    for (int i = 0; i < num_streams_; ++i) {
        streams_.push_back(device_.createStream());
    }
    
    // Create SVD solvers for fixed-size problems
    SVDSpec spec1{SVDVectors::THIN, SVDVectors::NONE, SVDAlgorithm::QR};
    svd_solver_1_ = device_.createSVDSolver(M, d, spec1, precision_);
    
    SVDSpec spec2{SVDVectors::THIN, SVDVectors::NONE, SVDAlgorithm::RANDOMIZED};
    spec2.rank = 1; // TODO: This is hardcoded, bad? (rank of output matrix)
    svd_solver_2_ = device_.createSVDSolver(N, d * d, spec2, precision_);
    
    device_.synchronize();
}

template<typename T>
double GenericMinimizationStrategy::stepOnceImpl(
    IDeviceMemory* d_kraus,
    IDeviceMemory* d_current_vector
) {
    // Execute step1 and step2
    step1Impl<T>(d_kraus, d_current_vector);
    step2Impl<T>(d_kraus, d_current_vector);
    
    // Compute and return entropy
    return computeEntropyImpl<T>();
}

template<typename T>
void GenericMinimizationStrategy::step1Impl(
    IDeviceMemory* d_kraus,
    IDeviceMemory* d_current_vector
) {
    using Complex = std::complex<T>;
    using Real = T;
    
    const int d = kraus_count_;
    const int N = input_dim_;
    const int M = output_dim_;
    
    ILinearAlgebra* linalg = device_.getLinearAlgebra();
    
    // Step 1a: Compute {K_i|ψ⟩}_i in parallel across streams
    Complex one{1.0, 0.0};
    Complex zero{0.0, 0.0};
    
    for (int stream_idx = 0; stream_idx < num_streams_; ++stream_idx) {
        for (int i = stream_idx; i < d; i += num_streams_) {
            // K_i is M×N, |ψ⟩ is N×1, result is M×1
            // K_i|ψ⟩ -> d_vecs_1[i*M : (i+1)*M]
            const Complex* kraus_i = reinterpret_cast<const Complex*>(d_kraus->data()) + (i * M * N);
            Complex* result_i = reinterpret_cast<Complex*>(d_vecs_1_->data()) + (i * M);
            
            linalg->gemv(
                Transpose::NO_TRANS,            // op(K_i) = K_i
                M, N,                           // Matrix dimensions M×N
                &one,                           // alpha = 1
                kraus_i,                        // Matrix K_i
                d_current_vector->data(),       // Vector |ψ⟩
                &zero,                          // beta = 0
                result_i,                       // Output vector
                precision_,
                streams_[stream_idx].get()
            );
        }
    }
    
    device_.synchronize();
    
    // Step 1b: SVD of {K_i|ψ⟩} matrix (M × d)
    // Result: d_vecs_1 = U (overwritten), d_sv_1 = Σ
    svd_solver_1_->compute(
        d_vecs_1_->data(),  // Input M×d matrix, overwritten with U
        d_sv_1_->data(),    // Output: d singular values
        d_sing_1_->data(),  // U 
        nullptr             // V^H not needed
    );
    
    // Step 1c: Logarithmic rescaling
    // |φ_i⟩ = sqrt(log((1-ε)λ_i² + ε/M) - log(ε/M)) · |u_i⟩
    
    // Copy singular values and vectors to host for rescaling
    std::vector<Real> sv_host(d);
    std::vector<Complex> vecs_host(d * M);
    
    d_sv_1_->copyToHost(sv_host.data(), d * sizeof(Real));
    d_sing_1_->copyToHost(vecs_host.data(), d * M * sizeof(Complex));
    
    // Apply rescaling on host
    Real eps_M = static_cast<Real>(epsilon_ / static_cast<double>(M));
    Real log_eps_M = std::log(eps_M);
    
    for (int i = 0; i < d; ++i) {
        Real lambda = sv_host[i];
        Real log_arg = (Real(1.0) - eps_M) * lambda * lambda + eps_M;
        Real rescale = std::sqrt(std::log(log_arg) - log_eps_M);
        
        for (int j = 0; j < M; ++j) {
            vecs_host[i * M + j] *= rescale;
        }
    }
    
    // Copy back to device
    d_sing_1_->copyFromHost(vecs_host.data(), d * M * sizeof(Complex));
    
    device_.synchronize();
}

template<typename T>
void GenericMinimizationStrategy::step2Impl(
    IDeviceMemory* d_kraus,
    IDeviceMemory* d_current_vector
) {
    using Complex = std::complex<T>;
    
    const int d = kraus_count_;
    const int N = input_dim_;
    const int M = output_dim_;
    
    ILinearAlgebra* linalg = device_.getLinearAlgebra();
    
    // Step 2a: Compute {K_j^H|φ_i⟩}_{ij} in parallel
    // For each j, multiply K_j^H by all d vectors from step 1
    // Result is d×d blocks of size N
    Complex one{1.0, 0.0};
    Complex zero{0.0, 0.0};
    
    for (int stream_idx = 0; stream_idx < num_streams_; ++stream_idx) {
        for (int j = stream_idx; j < d; j += num_streams_) {
            // K_j^H is N×M, d_vecs_1 is M×d, result is N×d
            const Complex* kraus_j = reinterpret_cast<const Complex*>(d_kraus->data()) + (j * M * N);
            Complex* result_j = reinterpret_cast<Complex*>(d_vecs_2_->data()) + (j * N * d);
            
            linalg->gemm(
                Transpose::CONJ_TRANS,        // op(K_j) = K_j^H
                Transpose::NO_TRANS,          // op(d_vecs_1) = d_vecs_1
                N, d, M,                      // C is N×d = (N×M) × (M×d)
                &one,                         // alpha = 1
                kraus_j,                      // K_j (M×N in storage)
                d_sing_1_->data(),            // d_vecs_1 (M×d)
                &zero,                        // beta = 0
                result_j,                     // Output (N×d)
                precision_,
                streams_[stream_idx].get()
            );
        }
    }
    
    device_.synchronize();
    
    // Step 2b: SVD of {K_j^H|φ_i⟩} matrix (N × d²)
    // Extract top left singular vector as new state
    
    svd_solver_2_->compute(
        d_vecs_2_->data(),  // Input N×(d²) matrix
        d_sv_2_->data(),    // Output: d² singular values
        d_sing_2_->data(),  // U: N×1 (we only need first column)
        nullptr             // V^H not needed
    );

    // Copy first column of U to current_vector (top eigenvector)
    // U is stored column-major, so first column is first N elements
    d_current_vector->copyFrom(d_sing_2_.get(), N * sizeof(Complex));
    
    device_.synchronize();
}

template<typename T>
double GenericMinimizationStrategy::computeEntropyImpl() {
    using Real = T;
    
    const int d = kraus_count_;
    
    // Compute von Neumann entropy from singular values
    // S(Φ_ε(ρ)) = -Σ λ_i² log(λ_i²) where {λ_i} are from step 1
    
    // Copy singular values to host
    std::vector<Real> sv_host(d);
    d_sv_1_->copyToHost(sv_host.data(), d * sizeof(Real));
    
    // Compute entropy: S = -Σ p_i log(p_i) where p_i = λ_i²
    Real entropy = 0.0;
    Real normalization = 0.0;
    
    // First normalize (singular values should already be normalized, but be safe)
    for (int i = 0; i < d; ++i) {
        normalization += sv_host[i] * sv_host[i];
    }
    
    for (int i = 0; i < d; ++i) {
        Real p = (sv_host[i] * sv_host[i]) / normalization;
        if (p > 1e-12) {  // Avoid log(0)
            entropy -= p * std::log(p);
        }
    }
    
    return static_cast<double>(entropy);
}

template<typename T>
std::vector<std::complex<T>> GenericMinimizationStrategy::convertVector(
    const std::vector<std::complex<double>>& vec
) const {
    std::vector<std::complex<T>> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = std::complex<T>(
            static_cast<T>(vec[i].real()),
            static_cast<T>(vec[i].imag())
        );
    }
    return result;
}

// Explicit template instantiations
template void GenericMinimizationStrategy::initializeImpl<float>(
    int, int, int, double);
template void GenericMinimizationStrategy::initializeImpl<double>(
    int, int, int, double);

template double GenericMinimizationStrategy::stepOnceImpl<float>(
    IDeviceMemory*, IDeviceMemory*);
template double GenericMinimizationStrategy::stepOnceImpl<double>(
    IDeviceMemory*, IDeviceMemory*);

template void GenericMinimizationStrategy::step1Impl<float>(
    IDeviceMemory*, IDeviceMemory*);
template void GenericMinimizationStrategy::step1Impl<double>(
    IDeviceMemory*, IDeviceMemory*);

template void GenericMinimizationStrategy::step2Impl<float>(
    IDeviceMemory*, IDeviceMemory*);
template void GenericMinimizationStrategy::step2Impl<double>(
    IDeviceMemory*, IDeviceMemory*);

template double GenericMinimizationStrategy::computeEntropyImpl<float>();
template double GenericMinimizationStrategy::computeEntropyImpl<double>();

template std::vector<std::complex<float>> GenericMinimizationStrategy::convertVector<float>(
    const std::vector<std::complex<double>>&) const;
template std::vector<std::complex<double>> GenericMinimizationStrategy::convertVector<double>(
    const std::vector<std::complex<double>>&) const;
