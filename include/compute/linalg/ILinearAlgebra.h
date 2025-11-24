#ifndef ILINEARALGEBRA_H_
#define ILINEARALGEBRA_H_

#include "compute/core/ComputeTypes.h"
#include "compute/stream/IComputeStream.h"

/**
 * @brief Transpose operation types for BLAS routines
 */
enum class Transpose {
    NO_TRANS,      ///< No transpose: op(A) = A
    TRANS,         ///< Transpose: op(A) = A^T
    CONJ_TRANS     ///< Conjugate transpose: op(A) = A^H
};

/**
 * @brief Platform-agnostic linear algebra interface (BLAS operations)
 * 
 * Conventions:
 * - All matrices stored in COLUMN-MAJOR order (Fortran/BLAS convention)
 * - All matrices assumed to be stored contiguously (lda = num_rows_in_storage, stride=1)
 * - All vectors assumed contiguous (stride = 1)
 * - Leading dimensions calculated automatically based on transpose operations
 * 
 * How it works:
 * - User specifies logical operation dimensions (e.g., "C is m×n")
 * - Implementation calculates storage layout from transpose flags
 * - Example: gemm with transA=CONJ_TRANS, m=5, k=3 means A stored as 3×5
 */
class ILinearAlgebra {
public:
    virtual ~ILinearAlgebra() = default;
    
    // ========================================================================
    // LEVEL 1 BLAS: Vector-Vector Operations
    // ========================================================================
    
    /**
     * @brief Vector addition: y := alpha*x + y
     * @param n Number of elements in vectors
     * @param alpha Scalar multiplier (complex<T>* for complex, T* for real)
     * @param x Input vector (length n, complex)
     * @param y Input/output vector (length n, modified in place, complex)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void axpy(
        int n,
        const void* alpha,
        const void* x,
        void* y,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief Complex conjugate dot product: result := x^H * y
     * @param n Number of elements
     * @param x Input vector (length n, complex)
     * @param y Input vector (length n, complex)
     * @param result Output scalar (complex<T>*)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void dotc(
        int n,
        const void* x,
        const void* y,
        void* result,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief Euclidean norm: result := ||x||_2
     * @param n Number of elements
     * @param x Input vector (length n, complex)
     * @param result Output scalar (void*, always real, float or double depending on precision of input)
     * @param precision FLOAT or DOUBLE (input precision)
     * @param stream Optional stream for async execution
     */
    virtual void norm2(
        int n,
        const void* x,
        void* result,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief Vector scaling: x := alpha * x
     * @param n Number of elements
     * @param alpha Scalar multiplier (complex<T>* for complex, T* for real)
     * @param x Input/output vector (length n, modified in place)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void scal(
        int n,
        const void* alpha,
        void* x,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    // ========================================================================
    // LEVEL 2 BLAS: Matrix-Vector Operations
    // ========================================================================
    
    /**
     * @brief General matrix-vector multiply: y := alpha*op(A)*x + beta*y
     * 
     * The operation performed is based on the transpose flag:
     * - NO_TRANS:    y(m) := alpha * A(m×n) * x(n) + beta * y(m)
     * - TRANS:       y(n) := alpha * A^T(n×m) * x(m) + beta * y(n)
     * - CONJ_TRANS:  y(n) := alpha * A^H(n×m) * x(m) + beta * y(n)
     * 
     * IMPORTANT: m and n refer to STORAGE dimensions of A, not operation dimensions!
     * - m = number of rows in A as stored (leading dimension)
     * - n = number of cols in A as stored
     * - Leading dimension (lda) is computed automatically as m
     * 
     * @param trans Transpose operation on A
     * @param m Number of rows in A (as stored, before any transpose)
     * @param n Number of columns in A (as stored, before any transpose)
     * @param alpha Scalar multiplier (complex<T>* or T*)
     * @param A Input matrix (column-major storage, m×n as stored)
     * @param x Input vector (size must match op(A) dimensions)
     * @param beta Scalar multiplier for y (complex<T>* or T*)
     * @param y Input/output vector (length m if NO_TRANS, length n if TRANS/CONJ_TRANS)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void gemv(
        Transpose trans,
        int m, int n,
        const void* alpha,
        const void* A,
        const void* x,
        const void* beta,
        void* y,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    // ========================================================================
    // LEVEL 3 BLAS: Matrix-Matrix Operations
    // ========================================================================
    
    /**
     * @brief General matrix-matrix multiply: C := alpha*op(A)*op(B) + beta*C
     * 
     * Performs: C := alpha * op(A) * op(B) + beta * C
     * 
     * [mxn] = scal * [mxk] * [kxn] + beta * [mxn]
     * 
     * Storage layouts deduced automatically:
     * - If transA = NO_TRANS:         A stored as m×k (lda = m)
     * - If transA = TRANS/CONJ_TRANS: A stored as k×m (lda = k)
     * - If transB = NO_TRANS:         B stored as k×n (ldb = k)
     * - If transB = TRANS/CONJ_TRANS: B stored as n×k (ldb = n)
     * - C always stored as m×n (ldc = m)
     * 
     * @param transA Transpose operation on A
     * @param transB Transpose operation on B
     * @param m Number of rows in result C and op(A)
     * @param n Number of columns in result C and op(B)
     * @param k Contraction dimension (columns of op(A), rows of op(B))
     * @param alpha Scalar multiplier (complex<T>* or T*)
     * @param A Input matrix (column major)
     * @param B Input matrix (column major)
     * @param beta Scalar multiplier for C (complex<T>* or T*)
     * @param C Input/output matrix m×n (stored as m×n, column major)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void gemm(
        Transpose transA, Transpose transB,
        int m, int n, int k,
        const void* alpha,
        const void* A,
        const void* B,
        const void* beta,
        void* C,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief General matrix addition/transpose: C := alpha*op(A) + beta*op(B)
     * 
     * Performs: C(m×n) := alpha * op(A)(m×n) + beta * op(B)(m×n)
     * 
     * Storage layouts deduced automatically:
     * - If transA = NO_TRANS:         A stored as m×n (lda = m)
     * - If transA = TRANS/CONJ_TRANS: A stored as n×m (lda = n)
     * - If transB = NO_TRANS:         B stored as m×n (ldb = m)
     * - If transB = TRANS/CONJ_TRANS: B stored as n×m (ldb = n)
     * - C always stored as m×n (ldc = m)
     * 
     * @param transA Transpose operation on A
     * @param transB Transpose operation on B
     * @param m Number of rows in result C
     * @param n Number of columns in result C
     * @param alpha Scalar multiplier for A (complex<T>* or T*)
     * @param A Input matrix (storage layout deduced from transA, m, n)
     * @param beta Scalar multiplier for B (complex<T>* or T*)
     * @param B Input matrix (storage layout deduced from transB, m, n), can be nullptr if beta=0
     * @param C Output matrix m×n (stored as m×n, column major)
     * @param precision FLOAT or DOUBLE
     * @param stream Optional stream for async execution
     */
    virtual void geam(
        Transpose transA, Transpose transB,
        int m, int n,
        const void* alpha,
        const void* A,
        const void* beta,
        const void* B,
        void* C,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
};

#endif // ILINEARALGEBRA_H_