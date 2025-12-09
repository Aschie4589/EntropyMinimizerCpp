#ifndef MINIMIZER_CONDITIONS_PREDICTION_STRATEGY_H
#define MINIMIZER_CONDITIONS_PREDICTION_STRATEGY_H

#include <cstddef>

namespace entropy {

// Forward declaration
template<typename T>
class CircularBuffer;

/**
 * @brief Result of entropy prediction
 */
struct PredictionResult {
    double predicted_entropy;   ///< Predicted final entropy value
    int predicted_iterations;   ///< Predicted iterations to reach target
    double confidence;          ///< Confidence metric (e.g., R² value)
    bool valid;                 ///< Whether prediction is valid
    
    PredictionResult() 
        : predicted_entropy(0.0), 
          predicted_iterations(0), 
          confidence(0.0), 
          valid(false) {}
};

/**
 * @brief Abstract base class for entropy prediction strategies
 * 
 * Prediction strategies analyze historical entropy data to forecast
 * convergence behavior and estimate final entropy values.
 */
class PredictionStrategy {
public:
    virtual ~PredictionStrategy() = default;
    
    /**
     * @brief Predict future entropy based on historical data
     * 
     * @param history Circular buffer containing entropy history
     * @return PredictionResult with prediction details and validity
     * 
     * The strategy analyzes the entropy history and returns:
     * - predicted_entropy: estimated final entropy value
     * - predicted_iterations: estimated iterations to reach convergence
     * - confidence: quality metric (e.g., R² for regression)
     * - valid: false if insufficient data or poor fit quality
     */
    virtual PredictionResult predict(const CircularBuffer<double>& history) = 0;
    
    /**
     * @brief Get minimum number of data points required for prediction
     * 
     * @return Minimum buffer size needed for valid prediction
     */
    virtual size_t getMinimumDataPoints() const = 0;
};

} // namespace entropy

#endif // MINIMIZER_CONDITIONS_PREDICTION_STRATEGY_H
