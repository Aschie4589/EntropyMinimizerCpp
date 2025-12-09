#ifndef MINIMIZER_CONDITIONS_LINEAR_EXTRAPOLATION_STRATEGY_H
#define MINIMIZER_CONDITIONS_LINEAR_EXTRAPOLATION_STRATEGY_H

#include <cstddef>
#include "minimizer/prediction/prediction_strategy.h"

namespace entropy {

/**
 * @brief Simple linear extrapolation prediction strategy
 * 
 * This strategy fits a linear model directly to entropy values:
 * - Model: entropy[i] ≈ a + b * i
 * - Uses least squares regression
 * - Extrapolates to predict when entropy reaches threshold
 * 
 * This is a simpler, more robust fallback when exponential fitting fails.
 * Useful when:
 * - Exponential fit has poor R²
 * - Late-stage convergence (nearly linear descent)
 * - Unstable exponential fits
 */
class LinearExtrapolationStrategy : public PredictionStrategy {
public:
    /**
     * @brief Construct linear extrapolation strategy
     * 
     * @param convergence_tolerance Threshold for "converged" entropy (default: 1e-15)
     * @param min_data_points Minimum buffer size for fitting (default: 50)
     */
    LinearExtrapolationStrategy(
        double convergence_tolerance = 1e-15,
        size_t min_data_points = 50
    );
    
    PredictionResult predict(const CircularBuffer<double>& history) override;
    
    size_t getMinimumDataPoints() const override;
    
    double getConvergenceTolerance() const { return convergence_tolerance_; }
    
private:
    double convergence_tolerance_;
    size_t min_data_points_;
    
    /**
     * @brief Perform linear regression on (X, Y) data
     * 
     * @param X Independent variable (iteration indices)
     * @param Y Dependent variable (entropy values)
     * @param n Number of data points
     * @param[out] intercept Fitted intercept
     * @param[out] slope Fitted slope
     * @return R² value of the fit
     */
    double linearRegression(
        const double* X,
        const double* Y,
        size_t n,
        double& intercept,
        double& slope
    ) const;
};

} // namespace entropy

#endif // MINIMIZER_CONDITIONS_LINEAR_EXTRAPOLATION_STRATEGY_H
