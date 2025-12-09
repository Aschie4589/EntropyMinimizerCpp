#include "minimizer/prediction/linear_extrapolation_strategy.h"
#include "minimizer/state/circular_buffer.h"
#include <cmath>
#include <vector>

namespace entropy {

LinearExtrapolationStrategy::LinearExtrapolationStrategy(
    double convergence_tolerance,
    size_t min_data_points)
    : convergence_tolerance_(convergence_tolerance),
      min_data_points_(min_data_points) {
}

size_t LinearExtrapolationStrategy::getMinimumDataPoints() const {
    return min_data_points_;
}

PredictionResult LinearExtrapolationStrategy::predict(const CircularBuffer<double>& history) {
    PredictionResult result;
    
    // Check if we have enough data
    size_t n = history.size();
    if (n < min_data_points_) {
        return result;  // Invalid
    }
    
    // Create X values (iteration indices)
    std::vector<double> X(n);
    std::vector<double> Y(n);
    
    for (size_t i = 0; i < n; ++i) {
        X[i] = static_cast<double>(i);
        Y[i] = history.get(i);
    }
    
    // Perform linear regression
    double intercept, slope;
    double rsquared = linearRegression(X.data(), Y.data(), n, intercept, slope);
    
    // Check if slope is negative (entropy decreasing)
    if (slope >= 0.0) {
        return result;  // Invalid: entropy not decreasing
    }
    
    // Predict final entropy by extrapolating to where entropy reaches convergence_tolerance_
    // Model: entropy(x) = intercept + slope * x
    // Solve: convergence_tolerance_ = intercept + slope * x_final
    // x_final = (convergence_tolerance_ - intercept) / slope
    
    double x_final = (convergence_tolerance_ - intercept) / slope;
    
    // Current position is at x = n-1
    int iterations_remaining = static_cast<int>(std::round(x_final - (n - 1)));
    
    // Ensure non-negative
    result.predicted_iterations = std::max(0, iterations_remaining);
    
    // Predicted final entropy is simply the convergence tolerance
    // (or we could extrapolate to a large x value)
    result.predicted_entropy = convergence_tolerance_;
    
    result.confidence = rsquared;
    result.valid = true;
    
    return result;
}

double LinearExtrapolationStrategy::linearRegression(
    const double* X,
    const double* Y,
    size_t n,
    double& intercept,
    double& slope) const {
    
    // Compute means
    double mean_x = 0.0, mean_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean_x += X[i];
        mean_y += Y[i];
    }
    mean_x /= n;
    mean_y /= n;
    
    // Compute slope and intercept
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double dx = X[i] - mean_x;
        double dy = Y[i] - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }
    
    if (denominator == 0.0) {
        slope = 0.0;
        intercept = mean_y;
        return 0.0;
    }
    
    slope = numerator / denominator;
    intercept = mean_y - slope * mean_x;
    
    // Compute R²
    double tss = 0.0;  // Total sum of squares
    double rss = 0.0;  // Residual sum of squares
    
    for (size_t i = 0; i < n; ++i) {
        double y_pred = intercept + slope * X[i];
        double diff_total = Y[i] - mean_y;
        double diff_residual = Y[i] - y_pred;
        
        tss += diff_total * diff_total;
        rss += diff_residual * diff_residual;
    }
    
    if (tss == 0.0) {
        return 1.0;  // Perfect fit
    }
    
    return 1.0 - rss / tss;
}

} // namespace entropy
