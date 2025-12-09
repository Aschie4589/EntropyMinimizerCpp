#ifndef MINIMIZER_CONDITIONS_EXPONENTIAL_FITTING_STRATEGY_H
#define MINIMIZER_CONDITIONS_EXPONENTIAL_FITTING_STRATEGY_H

#include <cstddef>
#include "minimizer/prediction/prediction_strategy.h"

namespace entropy {

/**
 * @brief Prediction strategy using exponential decay model
 * 
 * This strategy fits an exponential decay model to the entropy deltas:
 * 
 * Mathematical Model:
 * - Define delta[i] = entropy[i-1] - entropy[i] (entropy improvement at step i)
 * - Exponential model: delta[i] ≈ a * exp(b * i)
 * - Log-linearization: log(delta[i]) ≈ log(a) + b * i
 * - Linear regression on (i, log(delta[i])) yields slope b and intercept log(a)
 * 
 * Prediction Formula (from old implementation):
 * - Given current entropy S and last delta log(δ_n)
 * - Total remaining improvement: exp(δ_n) * exp(b) / (1 - exp(b))
 * - Predicted final entropy: S - improvement
 * 
 * Quality Assessment:
 * - Computes R² (coefficient of determination) for linear fit
 * - Returns valid=false if:
 *   * Insufficient data (< minimum points)
 *   * Poor fit quality (R² < threshold)
 *   * Positive slope (entropy increasing)
 *   * Numerical issues (log of negative/zero)
 * 
 * @note Based on old EntropyEstimator::exponentialFit() implementation
 */
class ExponentialFittingStrategy : public PredictionStrategy {
public:
    /**
     * @brief Construct exponential fitting strategy
     * 
     * @param rsquared_threshold Minimum R² value for valid prediction (default: 0.999)
     * @param convergence_tolerance Threshold for "converged" entropy (default: 1e-15)
     * @param min_data_points Minimum buffer size for fitting (default: 100)
     */
    ExponentialFittingStrategy(
        double rsquared_threshold = 0.999,
        double convergence_tolerance = 1e-15,
        size_t min_data_points = 100
    );
    
    PredictionResult predict(const CircularBuffer<double>& history) override;
    
    size_t getMinimumDataPoints() const override;
    
    /**
     * @brief Get R² threshold for valid predictions
     */
    double getRSquaredThreshold() const { return rsquared_threshold_; }
    
    /**
     * @brief Get convergence tolerance used in predictions
     */
    double getConvergenceTolerance() const { return convergence_tolerance_; }
    
private:
    double rsquared_threshold_;      ///< Minimum R² for valid fit
    double convergence_tolerance_;   ///< Target entropy for prediction
    size_t min_data_points_;         ///< Minimum required data points
    
    /**
     * @brief Perform linear regression on (X, Y) data
     * 
     * @param X Independent variable (iteration indices)
     * @param Y Dependent variable (log of deltas)
     * @param n Number of data points
     * @param[out] intercept Fitted intercept (log(a))
     * @param[out] slope Fitted slope (b)
     * @return R² value of the fit
     */
    double linearRegression(
        const double* X, 
        const double* Y, 
        size_t n,
        double& intercept,
        double& slope
    ) const;
    
    /**
     * @brief Compute R² (coefficient of determination)
     * 
     * @param Y Observed values
     * @param Y_pred Predicted values
     * @param n Number of data points
     * @return R² value in [0, 1] (1 = perfect fit)
     */
    double computeRSquared(const double* Y, const double* Y_pred, size_t n) const;
};

} // namespace entropy

#endif // MINIMIZER_CONDITIONS_EXPONENTIAL_FITTING_STRATEGY_H
