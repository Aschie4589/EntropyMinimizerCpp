#ifndef SVD_TYPES_H_
#define SVD_TYPES_H_

/**
 * @file SVDTypes.h
 * @brief Type definitions for Singular Value Decomposition operations
 * 
 * Platform-agnostic types for specifying and configuring SVD computations.
 */

/**
 * @brief Specifies which singular vectors to compute
 * 
 * Platform-agnostic enum for SVD vector computation mode.
 * Backend implementations map this to their native parameters.
 */
enum class SVDVectors {
    ALL,   ///< Compute all columns/rows (full U: m×m, full V^H: n×n)
    THIN,  ///< Compute min(m,n) columns/rows (thin U: m×min(m,n), thin V^H: min(m,n)×n)
    NONE   ///< Don't compute vectors, only singular values
};

/**
 * @brief Specifies SVD algorithm preference
 * 
 * Platform-agnostic algorithm selection. Backend chooses appropriate
 * implementation. Not all backends support all algorithms.
 * 
 * Algorithm mapping on CUDA:
 * - AUTO: cusolverDnXgesvd (QR-based, works for tall/wide matrices)
 * - QR: cusolverDnXgesvd (standard QR-based, limitations: only m>=n or n>=m)
 * - POLAR: cusolverDnXgesvdp (polar decomposition, more accurate)
 * - RANDOMIZED: cusolverDnXgesvdr (randomized, fast low-rank)
 * 
 * Algorithm mapping on CPU:
 * - All algorithms fall back to LAPACK zgesvd/cgesvd
 */
enum class SVDAlgorithm {
    AUTO,       ///< Let backend choose optimal algorithm (CUDA: gesvd, CPU: zgesvd)
    QR,         ///< QR-based (CUDA: cusolverDnXgesvd, CPU: zgesvd) - Note: CUDA has m>=n or n>=m limitations
    POLAR,      ///< Polar decomposition (CUDA: cusolverDnXgesvdp, CPU: falls back to zgesvd)
    RANDOMIZED  ///< Randomized (CUDA: cusolverDnXgesvdr, CPU: falls back to zgesvd)
};

/**
 * @brief SVD problem specification
 * 
 * Complete specification for a singular value decomposition problem.
 * Instances are passed to ISVDSolver to configure computation.
 * 
 * Usage:
 * @code
 * SVDSpec spec;
 * spec.vectors = SVDVectors::THIN;
 * spec.algorithm = SVDAlgorithm::RANDOMIZED;
 * spec.rank = 50;  // Compute top 50 singular values/vectors
 * @endcode
 */
struct SVDSpec {
    /// Which singular vectors to compute
    SVDVectors vectors = SVDVectors::THIN;
    
    /// Algorithm preference (backend may override if unsupported)
    SVDAlgorithm algorithm = SVDAlgorithm::AUTO;
    
    // Algorithm-specific parameters
    // (ignored by algorithms that don't use them)
    
    /// For randomized SVD: target rank (-1 = full rank)
    int rank = -1;
    
    /// For randomized SVD: oversampling parameter (adds extra vectors for accuracy)
    int oversampling = 10;
    
    /// For randomized SVD: number of power iterations (improves accuracy)
    int power_iterations = 0;
    
    /// For Jacobi SVD: convergence tolerance
    double tolerance = 1e-7;
    
    /// For Jacobi SVD: maximum number of sweeps
    int max_sweeps = 100;
    
    /// Default constructor
    SVDSpec() = default;
    
    /// Convenience constructor for common cases
    SVDSpec(SVDVectors v, SVDAlgorithm a = SVDAlgorithm::AUTO)
        : vectors(v), algorithm(a) {}
};

#endif // SVD_TYPES_H_
