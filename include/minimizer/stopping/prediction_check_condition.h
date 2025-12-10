#ifndef PREDICTION_CHECK_CONDITION_H
#define PREDICTION_CHECK_CONDITION_H

#include "minimizer/stopping/condition.h"
#include <memory>
#include <optional>

namespace entropy {

// Forward declarations
class EntropyPredictor;

/**
 * @brief Stops when predictor forecasts entropy will diverge above target
 * 
 * Uses entropy prediction to detect when the minimization trajectory is heading
 * toward entropy values significantly above the target. This allows early termination
 * to save computation and initiate a new run instead of wasting iterations on a
 * diverging trajectory.
 * 
 * Detection Strategy:
 * 1. Waits for predictor to collect enough data (delay period)
 * 2. Only trusts confident predictions (configurable R² threshold)
 * 3. Checks if forecasted entropy exceeds target by margin
 * 4. Detects upward drift when current entropy > target
 * 
 * Early termination prevents:
 * - Wasting iterations on trajectories that won't reach target
 * - Numerical instability from unbounded entropy growth
 * - Inefficient use of computational resources
 * 
 * @note This is a "grace period" stopping condition - provides a last chance to
 *       detect problems before they become severe
 * @note Requires EntropyPredictor to be enabled and initialized
 */
class PredictionCheckCondition : public Condition {
public:
    /**
     * @brief Construct prediction check condition
     * 
     * @param predictor Entropy predictor instance (non-owning reference)
     * @param target_entropy Target entropy threshold (if none, no checks performed)
     * @param divergence_multiplier How much above target triggers stop (e.g., 1.5 = 1.5x)
     * @param prediction_delay Iterations to wait before starting predictions (default: 50)
     * @param min_confidence Minimum confidence (R²) to trust prediction (default: 0.95)
     * 
     * @throws std::invalid_argument if predictor is nullptr
     */
    PredictionCheckCondition(
        const EntropyPredictor* predictor,
        std::optional<double> target_entropy,
        double divergence_multiplier = 1.05,
        size_t prediction_delay = 200,
        double min_confidence = 0.999
    );

    StopReason check(size_t iteration, 
                    double current_entropy, 
                    double previous_entropy) override;

    void reset() override;

    std::string description() const override;

private:
    /**
     * @brief Check if prediction indicates unrecoverable divergence
     * 
     * Returns true if:
     * - Predicted entropy significantly exceeds target (> target × multiplier)
     * - OR: Currently above target and moving higher
     * 
     * @param predicted_entropy Entropy value forecasted by predictor
     * @return true if divergence detected
     */
    bool isPredictedDivergence(double predicted_entropy) const;
    
    const EntropyPredictor* predictor_;          ///< Non-owning reference to predictor
    std::optional<double> target_entropy_;       ///< Target entropy threshold
    double divergence_multiplier_;               ///< Margin multiplier for target
    size_t prediction_delay_;                    ///< Iterations before checking predictions
    double min_confidence_;                      ///< Minimum R² to trust prediction
    double current_min_entropy_;                 ///< Min entropy seen so far (for drift detection)
};

} // namespace entropy

#endif // PREDICTION_CHECK_CONDITION_H
