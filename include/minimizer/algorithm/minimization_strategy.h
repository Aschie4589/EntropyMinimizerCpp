#ifndef MINIMIZATION_STRATEGY_H_
#define MINIMIZATION_STRATEGY_H_

#include <complex>
#include <vector>
#include <memory>
#include <cstring>
#include "compute/core/ComputeTypes.h"

// Forward declaration
class IDeviceMemory;

/**
 * @brief Host-side representation of Kraus operators with type-erased storage
 * 
 * Contains d Kraus operators K_i, each of dimension M×N (output × input).
 * Stored as a contiguous byte array in column-major order to support both
 * float and double precision without conversion overhead.
 * 
 * Layout: [K_0][K_1]...[K_{d-1}] where each K_i is M×N complex numbers
 * - Entry (row, col) of K_i is at byte offset: 
 *   (i*M*N + col*M + row) * sizeof(complex<T>)
 * 
 * Invariant: data.size() == kraus_count * output_dim * input_dim * complex_size
 */
struct HostKrausOperators {
    std::vector<char> data;  ///< Type-erased byte storage for all Kraus operators
    PrecisionType precision; ///< FLOAT or DOUBLE precision
    int kraus_count;         ///< Number of Kraus operators (d)
    int input_dim;           ///< Input dimension (N)
    int output_dim;          ///< Output dimension (M)
    
    /**
     * @brief Default constructor - creates empty operator set
     */
    HostKrausOperators() 
        : precision(PrecisionType::DOUBLE),
          kraus_count(0), 
          input_dim(0), 
          output_dim(0) {}
    
    /**
     * @brief Create from float precision data
     * 
     * @param kraus_data Source data in float precision
     * @param d Number of Kraus operators
     * @param N Input dimension
     * @param M Output dimension
     * @return HostKrausOperators with FLOAT precision
     */
    static HostKrausOperators fromFloat(
        const std::vector<std::complex<float>>& kraus_data,
        int d, int N, int M
    ) {
        HostKrausOperators result;
        result.precision = PrecisionType::FLOAT;
        result.kraus_count = d;
        result.input_dim = N;
        result.output_dim = M;
        
        // Validate input size
        if (kraus_data.size() != static_cast<size_t>(d * N * M)) {
            throw std::invalid_argument(
                "HostKrausOperators::fromFloat: data size mismatch. Expected " + 
                std::to_string(d * N * M) + ", got " + std::to_string(kraus_data.size())
            );
        }
        
        // Copy data as bytes
        size_t byte_size = kraus_data.size() * sizeof(std::complex<float>);
        result.data.resize(byte_size);
        std::memcpy(result.data.data(), kraus_data.data(), byte_size);
        
        return result;
    }
    
    /**
     * @brief Create from double precision data
     * 
     * @param kraus_data Source data in double precision
     * @param d Number of Kraus operators
     * @param N Input dimension
     * @param M Output dimension
     * @return HostKrausOperators with DOUBLE precision
     */
    static HostKrausOperators fromDouble(
        const std::vector<std::complex<double>>& kraus_data,
        int d, int N, int M
    ) {
        HostKrausOperators result;
        result.precision = PrecisionType::DOUBLE;
        result.kraus_count = d;
        result.input_dim = N;
        result.output_dim = M;
        
        // Validate input size
        if (kraus_data.size() != static_cast<size_t>(d * N * M)) {
            throw std::invalid_argument(
                "HostKrausOperators::fromDouble: data size mismatch. Expected " + 
                std::to_string(d * N * M) + ", got " + std::to_string(kraus_data.size())
            );
        }
        
        // Copy data as bytes
        size_t byte_size = kraus_data.size() * sizeof(std::complex<double>);
        result.data.resize(byte_size);
        std::memcpy(result.data.data(), kraus_data.data(), byte_size);
        
        return result;
    }
    
    /**
     * @brief Get type-erased pointer to all Kraus data
     * 
     * Strategy implementations can cast this to the appropriate type
     * based on the precision field.
     * 
     * @return Pointer to raw byte data
     */
    const void* getData() const {
        return data.data();
    }
    
    /**
     * @brief Get typed pointer to i-th Kraus operator (float precision)
     * 
     * @param i Index of Kraus operator (0 <= i < kraus_count)
     * @return Pointer to start of K_i data
     * @throws std::logic_error if precision is not FLOAT
     */
    const std::complex<float>* getKrausFloat(int i) const {
        if (precision != PrecisionType::FLOAT) {
            throw std::logic_error("getKrausFloat called but precision is not FLOAT");
        }
        const auto* typed_data = reinterpret_cast<const std::complex<float>*>(data.data());
        return typed_data + i * output_dim * input_dim;
    }
    
    /**
     * @brief Get typed pointer to i-th Kraus operator (double precision)
     * 
     * @param i Index of Kraus operator (0 <= i < kraus_count)
     * @return Pointer to start of K_i data
     * @throws std::logic_error if precision is not DOUBLE
     */
    const std::complex<double>* getKrausDouble(int i) const {
        if (precision != PrecisionType::DOUBLE) {
            throw std::logic_error("getKrausDouble called but precision is not DOUBLE");
        }
        const auto* typed_data = reinterpret_cast<const std::complex<double>*>(data.data());
        return typed_data + i * output_dim * input_dim;
    }
    
    /**
     * @brief Get total size in bytes of all Kraus operators
     */
    size_t sizeBytes() const {
        return data.size();
    }
    
    /**
     * @brief Get size of a single complex number in current precision
     */
    size_t complexSize() const {
        return (precision == PrecisionType::FLOAT) ? 
               sizeof(std::complex<float>) : 
               sizeof(std::complex<double>);
    }
};

/**
 * @brief Abstract interface for quantum channel minimization algorithms
 * 
 * Represents the Strategy pattern for different device-specific implementations
 * of the minimization algorithm:
 * 
 * Algorithm: Find minimum of S(Φ_ε(|ψ⟩⟨ψ|)) where
 * - Φ is a quantum channel with Kraus operators {K_i}
 * - Φ_ε is the epsilon-perturbed channel
 * - S is the von Neumann entropy
 * 
 * Each iteration performs:
 * 1. Apply channel: compute {K_i|ψ⟩}_i
 * 2. SVD and logarithmic scaling
 * 3. Second SVD to extract new |ψ⟩
 * 
 * This is a stateless strategy - Kraus operators and current vector are managed
 * by AlgorithmManager and passed into each step. The strategy owns only its
 * workspace buffers via DeviceMemory<T> RAII wrappers.
 */
class IMinimizationStrategy {
public:
    virtual ~IMinimizationStrategy() = default;
    
    /**
     * @brief Initialize strategy workspace without Kraus operators or state
     * 
     * Allocates workspace memory for SVD operations and intermediate results.
     * Does NOT allocate or transfer Kraus operators or current vector - those
     * are managed by AlgorithmManager and passed into stepOnce().
     * 
     * @param kraus_count Number of Kraus operators (d)
     * @param input_dim Input dimension (N)
     * @param output_dim Output dimension (M)
     * @param epsilon Perturbation parameter (0 < ε < 1)
     * @param precision FLOAT or DOUBLE precision
     * 
     * @throws std::runtime_error if device doesn't support precision
     * @throws std::runtime_error if dimensions invalid (d > M constraint)
     * @throws std::runtime_error if device memory allocation fails
     */
    virtual void initialize(
        int kraus_count,
        int input_dim,
        int output_dim,
        double epsilon,
        PrecisionType precision
    ) = 0;
    
    /**
     * @brief Execute one iteration of the minimization algorithm
     * 
     * Performs the full algorithm step:
     * 1. Compute {K_i|ψ⟩}_i via matrix-vector multiplications
     * 2. SVD to get {λ_i, |φ_i⟩}
     * 3. Apply logarithmic transformation: log((1-ε)λ_i² + ε) - log(ε)
     * 4. Second SVD to extract new |ψ⟩ (top eigenvector)
     * 5. Compute entropy from singular values
     * 
     * Updates d_current_vector in-place with new state.
     * Does NOT cache entropy - returns it directly.
     * 
     * @param d_kraus Device memory containing Kraus operators (d×M×N, managed by caller)
     * @param d_current_vector Device memory containing current state (N, updated in-place)
     * @return Computed entropy S(Φ_ε(|ψ⟩⟨ψ|))
     * 
     * @throws std::runtime_error if called before initialize()
     * @throws std::runtime_error if device computation fails
     */
    virtual double stepOnce(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    ) = 0;
    
    /**
     * @brief Get the precision type used for computation
     * 
     * @return FLOAT or DOUBLE
     */
    virtual PrecisionType getPrecision() const = 0;
    
    /**
     * @brief Get workspace memory required by this strategy in bytes
     * 
     * Returns device memory needed for algorithm-specific workspace:
     * - Intermediate vectors for SVD operations
     * - Singular value storage
     * - SVD scratch space
     * 
     * Does NOT include Kraus operators or current vector - those are
     * managed by AlgorithmManager.
     * 
     * @return Total workspace bytes required on device
     */
    virtual size_t getWorkspaceSize() const = 0;
};

#endif // MINIMIZATION_STRATEGY_H_
