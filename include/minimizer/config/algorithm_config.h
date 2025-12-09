#ifndef ALGORITHM_CONFIG_H_
#define ALGORITHM_CONFIG_H_

#include "compute/core/ComputeTypes.h"
#include <stdexcept>
#include <string>
#include <sstream>

namespace entropy {

/**
 * @brief Configuration for algorithm execution parameters
 * 
 * Controls the core algorithm behavior including numerical precision,
 * depolarization parameters, and device selection.
 */
struct AlgorithmConfig {
    // Depolarization parameter (must be in (0,1))
    double epsilon = 1e-3;
    
    // Computation precision
    PrecisionType precision = PrecisionType::DOUBLE;
    
    // Device ID for multi-GPU systems (must be >= 0)
    int device_id = 0;
    
    /**
     * @brief Construct with default values
     */
    AlgorithmConfig() = default;
    
    /**
     * @brief Construct with custom values
     */
    AlgorithmConfig(double eps, PrecisionType prec = PrecisionType::DOUBLE, int dev_id = 0)
        : epsilon(eps), precision(prec), device_id(dev_id) {
        validate();
    }
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (epsilon <= 0.0 || epsilon >= 1.0) {
            throw std::invalid_argument(
                "AlgorithmConfig: epsilon must be in (0,1), got " + std::to_string(epsilon)
            );
        }
        if (device_id < 0) {
            throw std::invalid_argument(
                "AlgorithmConfig: device_id must be >= 0, got " + std::to_string(device_id)
            );
        }
    }
    
    /**
     * @brief Convert to string for debugging/logging
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "AlgorithmConfig{epsilon=" << epsilon 
            << ", precision=" << (precision == PrecisionType::DOUBLE ? "DOUBLE" : "FLOAT")
            << ", device_id=" << device_id << "}";
        return oss.str();
    }
    
    // Equality comparison for testing
    bool operator==(const AlgorithmConfig& other) const {
        return epsilon == other.epsilon && 
               precision == other.precision && 
               device_id == other.device_id;
    }
    
    bool operator!=(const AlgorithmConfig& other) const {
        return !(*this == other);
    }
};

} // namespace entropy

#endif // ALGORITHM_CONFIG_H_
