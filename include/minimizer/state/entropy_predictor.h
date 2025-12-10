#ifndef MINIMIZER_CORE_ENTROPY_PREDICTOR_H
#define MINIMIZER_CORE_ENTROPY_PREDICTOR_H

#include <cstddef>
#include <memory>
#include "minimizer/state/circular_buffer.h"
#include "minimizer/prediction/prediction_strategy.h"

namespace entropy {

/**
 * @brief Manages entropy prediction using historical data
 * 
 * The EntropyPredictor maintains a circular buffer of entropy values
 * and delegates prediction to a configurable PredictionStrategy.
 * 
 * Typical Usage:
 * ```cpp
 * auto strategy = std::make_unique<ExponentialFittingStrategy>();
 * EntropyPredictor predictor(std::move(strategy), 200);
 * 
 * // During minimization loop
 * for (int i = 0; i < max_iterations; ++i) {
 *     double entropy = computeEntropy();
 *     predictor.addDataPoint(entropy);
 *     
 *     if (predictor.hasEnoughData()) {
 *         PredictionResult pred = predictor.predict();
 *         if (pred.valid && pred.confidence > 0.999) {
 *             // Prediction is reliable
 *             if (pred.predicted_entropy < target + tolerance) {
 *                 break;  // Will converge to target
 *             }
 *         }
 *     }
 * }
 * ```
 * 
 * Memory Efficiency:
 * - Window size 200: ~1.6 KB per instance
 * - Window size 500: ~4 KB per instance
 * - Much more efficient than storing full iteration history (MB-GB)
 * 
 * @note This replaces old EntropyEstimator class with cleaner separation
 *       between data management (EntropyPredictor) and fitting logic (strategies)
 */
class EntropyPredictor {
public:
    /**
     * @brief Construct entropy predictor
     * 
     * @param strategy Prediction strategy (ownership transferred)
     * @param window_size Size of circular buffer for history (default: 200)
     * 
     * @throws std::invalid_argument if strategy is nullptr or window_size < 2
     */
    EntropyPredictor(
        std::unique_ptr<PredictionStrategy> strategy,
        size_t window_size = 200
    );
    
    /**
     * @brief Add new entropy data point to history
     * 
     * @param entropy Current entropy value
     * 
     * Pushes entropy to circular buffer (overwrites oldest if full).
     */
    void addDataPoint(double entropy);
    
    /**
     * @brief Reset predictor state (clears all history)
     */
    void reset();
    
    /**
     * @brief Perform prediction based on current history
     * 
     * @return PredictionResult with prediction and validity flag
     * 
     * Delegates to strategy->predict(). Returns invalid result if
     * insufficient data or strategy returns invalid.
     */
    PredictionResult predict() const;
    
    /**
     * @brief Check if enough data is available for prediction
     * 
     * @return true if buffer has at least minimum required points
     */
    bool hasEnoughData() const;
    
    /**
     * @brief Get current window size
     */
    size_t getWindowSize() const { return entropy_history_.capacity(); }
    
    /**
     * @brief Get number of data points currently stored
     */
    size_t getCurrentDataPoints() const { return entropy_history_.size(); }
    
    /**
     * @brief Set a new prediction strategy
     * 
     * @param strategy New strategy (ownership transferred)
     * 
     * @throws std::invalid_argument if strategy is nullptr
     */
    void setStrategy(std::unique_ptr<PredictionStrategy> strategy);
    
    /**
     * @brief Get reference to current strategy (read-only)
     */
    const PredictionStrategy& getStrategy() const { return *strategy_; }
    
private:
    std::unique_ptr<PredictionStrategy> strategy_;  ///< Prediction algorithm
    CircularBuffer<double> entropy_history_;        ///< Entropy data buffer
};

} // namespace entropy

#endif // MINIMIZER_CORE_ENTROPY_PREDICTOR_H
