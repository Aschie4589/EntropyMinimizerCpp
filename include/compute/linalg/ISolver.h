#ifndef ISOLVER_H_
#define ISOLVER_H_

#include "compute/core/ComputeTypes.h"
#include "compute/stream/IComputeStream.h"

/**
 * @brief Platform-agnostic solver interface (LAPACK-style operations)
 * 
 * Simplified interface - no need to specify leading dimensions!
 * 
 * Conventions:
 * - All matrices stored in COLUMN-MAJOR order (Fortran/LAPACK convention)
 * - All matrices assumed to be stored contiguously (lda = num_rows)
 * - Leading dimensions calculated automatically: lda = m for m×n matrix
 * - Workspace queries return size in number of elements (not bytes)
 */
class ISolver {
public:
    virtual ~ISolver() = default;
    
    // ========================================================================
    // QR FACTORIZATION
    // ========================================================================
    
    /**
     * @brief QR factorization: A = Q*R
     * 
     * Computes QR factorization of m×n matrix A. On exit:
     * - A is overwritten with R in upper triangle and Householder reflectors below
     * - tau contains scalar factors of elementary reflectors
     * - Q can be generated using orgqr/ungqr
     * 
     * Matrix A stored as m×n in column-major (lda = m, computed automatically)
     * 
     * @param m Number of rows in A
     * @param n Number of columns in A
     * @param A Input/output matrix m×n (stored contiguously, modified in place). Complex matrix!!
     * @param tau Output array of length min(m,n) for Householder scalars
     * @param workspace Work array
     * @param workspace_size Size of workspace (query with geqrf_workspace_size)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void geqrf(
        int m, int n,
        void* A,
        void* tau,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief Generate orthogonal/unitary matrix Q from QR factorization
     * 
     * Generates the m×n matrix Q with orthonormal columns from geqrf output.
     * For complex matrices, generates unitary matrix (orgqr for real, ungqr for complex).
     * 
     * Matrix A stored as m×n in column-major (lda = m, computed automatically)
     * 
     * @param m Number of rows in Q
     * @param n Number of columns in Q (n <= m)
     * @param k Number of elementary reflectors (k <= n), typically k = min(m,n)
     * @param A Input from geqrf, output Q (m×n, stored contiguously, modified in place)
     * @param tau Householder scalars from geqrf (length k)
     * @param workspace Work array
     * @param workspace_size Size of workspace (query with orgqr_workspace_size)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void orgqr(
        int m, int n, int k,
        void* A,
        const void* tau,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    // ========================================================================
    // SINGULAR VALUE DECOMPOSITION
    // ========================================================================
    
    /**
     * @brief Singular Value Decomposition: A = U * Σ * V^H
     * 
     * Computes the SVD of m×n matrix A. Supports multiple computation modes.
     * All matrices stored contiguously in column-major order.
     * 
     * @param jobu Controls computation of U:
     *   - 'A': All m columns of U computed (U is m×m)
     *   - 'S': First min(m,n) columns of U computed (U is m×min(m,n))
     *   - 'O': Overwrite A with columns of U
     *   - 'N': No columns of U computed
     * @param jobvt Controls computation of V^H:
     *   - 'A': All n rows of V^H computed (VT is n×n)
     *   - 'S': First min(m,n) rows of V^H computed (VT is min(m,n)×n)
     *   - 'O': Overwrite A with rows of V^H
     *   - 'N': No rows of V^H computed
     * @param m Number of rows in A
     * @param n Number of columns in A
     * @param A Input matrix m×n (stored contiguously, may be overwritten depending on jobu/jobvt)
     * @param S Output array of length min(m,n) containing singular values (real, descending order)
     * @param U Output matrix for left singular vectors (if jobu != 'N', 'O'), stored contiguously
     * @param VT Output matrix for right singular vectors V^H (if jobvt != 'N', 'O'), stored contiguously
     * @param workspace Work array
     * @param workspace_size Size of workspace (query with gesvd_workspace_size)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void gesvd(
        char jobu, char jobvt,
        int m, int n,
        void* A,
        void* S,
        void* U,
        void* VT,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief Polar-based SVD: A = U * Σ * V^H (CUDA-optimized, CPU falls back to gesvd)
     * 
     * Uses polar decomposition for improved accuracy. CUDA uses cusolverDnXgesvdp.
     * CPU backend falls back to standard gesvd.
     * 
     * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
     * @param econ 1 for economy size (min(m,n)), 0 for full size
     * @param m Number of rows in A
     * @param n Number of columns in A
     * @param A Input matrix m×n (stored contiguously, overwritten)
     * @param S Output array of length min(m,n) containing singular values
     * @param U Output matrix for left singular vectors (m×min(m,n) if econ=1, m×m if econ=0)
     * @param V Output matrix for right singular vectors (n×min(m,n) if econ=1, n×n if econ=0)
     * @param workspace Work array
     * @param workspace_size Size of workspace (query with gesvdp_workspace_size)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void gesvdp(
        int jobz,
        int econ,
        int m, int n,
        void* A,
        void* S,
        void* U,
        void* V,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief Randomized SVD for low-rank approximation (CUDA-optimized, CPU falls back to gesvd)
     * 
     * Computes rank-k approximation using randomized algorithms. Much faster for low-rank.
     * CUDA uses cusolverDnXgesvdr. CPU backend falls back to standard gesvd.
     * 
     * @param jobu 'S' to compute first k left singular vectors, 'N' for none
     * @param jobv 'S' to compute first k right singular vectors, 'N' for none
     * @param m Number of rows in A
     * @param n Number of columns in A
     * @param k Rank of approximation (k <= min(m,n))
     * @param A Input matrix m×n (stored contiguously, overwritten)
     * @param S Output array of length k containing singular values
     * @param U Output matrix for left singular vectors (m×k if jobu='S')
     * @param V Output matrix for right singular vectors (n×k if jobv='S')
     * @param workspace Work array
     * @param workspace_size Size of workspace (query with gesvdr_workspace_size)
     * @param precision FLOAT or DOUBLE
     * @param oversampling Oversampling parameter p (recommended: 2*k, min: k)
     * @param niters Number of iterations (recommended: 2)
     * @param stream Optional stream for async execution
     */
    virtual void gesvdr(
        char jobu, char jobv,
        int m, int n, int k,
        void* A,
        void* S,
        void* U,
        void* V,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        int oversampling = -1,  // -1 means use default (2*k)
        int niters = 2,
        IStream* stream = nullptr
    ) = 0;
    
    // ========================================================================
    // WORKSPACE QUERIES
    // ========================================================================
    
    /**
     * @brief Query optimal workspace size for geqrf
     * @param m Number of rows
     * @param n Number of columns
     * @param precision FLOAT or DOUBLE
     * @return Workspace size in number of elements (complex<T> or T)
     */
    virtual int geqrf_workspace_size(
        int m, int n,
        PrecisionType precision
    ) = 0;
    
    /**
     * @brief Query optimal workspace size for orgqr/ungqr
     * @param m Number of rows
     * @param n Number of columns
     * @param k Number of reflectors
     * @param precision FLOAT or DOUBLE
     * @return Workspace size in number of elements (complex<T> or T)
     */
    virtual int orgqr_workspace_size(
        int m, int n, int k,
        PrecisionType precision
    ) = 0;
    
    /**
     * @brief Query optimal workspace size for gesvd
     * @param m Number of rows
     * @param n Number of columns
     * @param precision FLOAT or DOUBLE
     * @return Workspace size in number of elements (complex<T> or T)
     */
    virtual int gesvd_workspace_size(
        int m, int n,
        PrecisionType precision
    ) = 0;
    
    /**
     * @brief Query optimal workspace size for gesvdp
     * @param m Number of rows
     * @param n Number of columns
     * @param econ Economy size flag
     * @param precision FLOAT or DOUBLE
     * @return Workspace size in number of elements (complex<T> or T)
     */
    virtual int gesvdp_workspace_size(
        int m, int n,
        int econ,
        PrecisionType precision
    ) = 0;
    
    /**
     * @brief Query optimal workspace size for gesvdr
     * @param m Number of rows
     * @param n Number of columns
     * @param k Rank of approximation
     * @param oversampling Oversampling parameter
     * @param niters Number of iterations
     * @param precision FLOAT or DOUBLE
     * @return Workspace size in number of elements (complex<T> or T)
     */
    virtual int gesvdr_workspace_size(
        int m, int n, int k,
        int oversampling,
        int niters,
        PrecisionType precision
    ) = 0;
};

#endif // ISOLVER_H_