#include "minimizer/prediction/exponential_fitting_strategy.h"
#include "minimizer/state/circular_buffer.h"
#include <cmath>
#include <vector>
#include <limits>

namespace entropy {

ExponentialFittingStrategy::ExponentialFittingStrategy(
    double rsquared_threshold,
    double convergence_tolerance,
    size_t min_data_points)
    : rsquared_threshold_(rsquared_threshold),
      convergence_tolerance_(convergence_tolerance),
      min_data_points_(min_data_points) {
}

size_t ExponentialFittingStrategy::getMinimumDataPoints() const {
    return min_data_points_;
}

PredictionResult ExponentialFittingStrategy::predict(const CircularBuffer<double>& history) {
    PredictionResult result;
    
    // Step 1: Check if we have enough data
    size_t n = history.size();
    if (n < min_data_points_) {
        return result;  // Invalid result (default)
    }
    
    // Step 2: Compute deltas: delta[i] = log(entropy[i-1] - entropy[i])
    // We have n entropy values, so we get n-1 deltas
    std::vector<double> log_deltas;
    log_deltas.reserve(n - 1);
    
    bool has_invalid_delta = false;
    for (size_t i = 1; i < n; ++i) {
        double prev_entropy = history.get(i - 1);
        double curr_entropy = history.get(i);
        double delta = prev_entropy - curr_entropy;  // Entropy improvement
        
        if (delta <= 0.0) {
            // Entropy increased or stayed same - can't take log
            has_invalid_delta = true;
            break;
        }
        
        log_deltas.push_back(std::log(delta));
    }
    
    if (has_invalid_delta || log_deltas.empty()) {
        return result;  // Invalid: non-positive deltas
    }
    
    // Step 3: Create X values (iteration indices)
    // In old code: iteration starts at (current_window_index - window_size + 1)
    // For us: just use indices 0, 1, 2, ..., n-2 (we have n-1 deltas)
    std::vector<double> X;
    X.reserve(log_deltas.size());
    for (size_t i = 0; i < log_deltas.size(); ++i) {
        X.push_back(static_cast<double>(i));
    }
    
    // Step 4: Perform linear regression on (X, log_deltas)
    double intercept, slope;
    double rsquared = linearRegression(X.data(), log_deltas.data(), log_deltas.size(), 
                                       intercept, slope);
    
    // Step 5: Check validity conditions
    if (rsquared < rsquared_threshold_) {
        return result;  // Poor fit quality
    }
    
    if (slope >= 0.0) {
        return result;  // Model invalid: deltas are increasing (entropy getting worse)
    }
    
    // Step 6: Predict final entropy
    // From old code: improvement = exp(last_log_delta) * exp(slope) / (1 - exp(slope))
    // predicted_entropy = current_entropy - improvement
    
    double current_entropy = history.newest();
    double last_log_delta = log_deltas.back();
    
    double exp_slope = std::exp(slope);
    if (exp_slope >= 1.0) {
        return result;  // Would cause division by zero or negative
    }
    
    double improvement = std::exp(last_log_delta) * exp_slope / (1.0 - exp_slope);
    
    // Guard against numerical overflow
    if (!std::isfinite(improvement) || improvement < 0.0) {
        return result;
    }
    
    result.predicted_entropy = current_entropy - improvement;
    
    // Step 7: Predict iterations to reach convergence_tolerance_
    // From old code: predicted_steps = (log(tolerance) - intercept) / slope
    // This gives the iteration where log(delta) = log(tolerance)
    // Since our X starts at 0, we need to adjust
    
    double target_log_delta = std::log(convergence_tolerance_);
    double predicted_x = (target_log_delta - intercept) / slope;
    
    // predicted_x is relative to start of our window
    // Current iteration is at index (n-1)
    // So iterations remaining = predicted_x - (n-1)
    int iterations_remaining = static_cast<int>(std::round(predicted_x - (n - 1)));
    
    // Ensure non-negative
    result.predicted_iterations = std::max(0, iterations_remaining);
    
    // Step 8: Set confidence and validity
    result.confidence = rsquared;
    result.valid = true;
    
    return result;
}

double ExponentialFittingStrategy::linearRegression(
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
    
    // Compute slope and intercept using least squares
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
        return 0.0;  // No variation in X
    }
    
    slope = numerator / denominator;
    intercept = mean_y - slope * mean_x;
    
    // Compute R²
    std::vector<double> Y_pred(n);
    for (size_t i = 0; i < n; ++i) {
        Y_pred[i] = intercept + slope * X[i];
    }
    
    return computeRSquared(Y, Y_pred.data(), n);
}

double ExponentialFittingStrategy::computeRSquared(
    const double* Y,
    const double* Y_pred,
    size_t n) const {
    
    // Compute mean of Y
    double mean_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean_y += Y[i];
    }
    mean_y /= n;
    
    // Total sum of squares (TSS)
    double tss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = Y[i] - mean_y;
        tss += diff * diff;
    }
    
    // Residual sum of squares (RSS)
    double rss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = Y[i] - Y_pred[i];
        rss += diff * diff;
    }
    
    // R² = 1 - RSS/TSS
    if (tss == 0.0) {
        return 1.0;  // Perfect fit (all Y values are identical)
    }
    
    return 1.0 - rss / tss;
}

} // namespace entropy
