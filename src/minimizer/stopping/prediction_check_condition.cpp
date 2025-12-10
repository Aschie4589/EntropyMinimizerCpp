#include "minimizer/stopping/prediction_check_condition.h"
#include "minimizer/state/entropy_predictor.h"
#include "minimizer/prediction/prediction_strategy.h"
#include <limits>
#include <sstream>
#include <stdexcept>

namespace entropy {

PredictionCheckCondition::PredictionCheckCondition(
    const EntropyPredictor* predictor,
    std::optional<double> target_entropy,
    double divergence_multiplier,
    size_t prediction_delay,
    double min_confidence
)
    : predictor_(predictor),
      target_entropy_(target_entropy),
      divergence_multiplier_(divergence_multiplier),
      prediction_delay_(prediction_delay),
      min_confidence_(min_confidence),
      current_min_entropy_(std::numeric_limits<double>::infinity())
{
    if (predictor == nullptr) {
        throw std::invalid_argument(
            "PredictionCheckCondition: predictor cannot be nullptr"
        );
    }
    
    if (divergence_multiplier < 1.0) {
        throw std::invalid_argument(
            "PredictionCheckCondition: divergence_multiplier must be >= 1.0"
        );
    }
    
    if (min_confidence < 0.0 || min_confidence > 1.0) {
        throw std::invalid_argument(
            "PredictionCheckCondition: min_confidence must be in [0, 1]"
        );
    }
}

StopReason PredictionCheckCondition::check(
    size_t iteration,
    double current_entropy,
    double previous_entropy
)
{
    // Skip if no target entropy specified (nothing to diverge from)
    if (!target_entropy_.has_value()) {
        return StopReason::CONTINUE;
    }
    
    // Wait for prediction delay period to gather data
    if (iteration < prediction_delay_) {
        return StopReason::CONTINUE;
    }
    
    // Update minimum entropy tracking
    if (current_entropy < current_min_entropy_) {
        current_min_entropy_ = current_entropy;
    }
    
    // Check if predictor has enough data for reliable predictions
    if (!predictor_->hasEnoughData()) {
        return StopReason::CONTINUE;
    }
    
    // Get prediction from predictor
    PredictionResult prediction = predictor_->predict();
    
    // Skip if prediction is invalid
    if (!prediction.valid) {
        return StopReason::CONTINUE;
    }
    
    // Only trust confident predictions
    if (prediction.confidence < min_confidence_) {
        return StopReason::CONTINUE;
    }
    
    // Check if forecasted divergence detected
    if (isPredictedDivergence(prediction.predicted_entropy)) {
        return StopReason::FORECASTED_DIVERGENCE;
    }
    
    return StopReason::CONTINUE;
}

void PredictionCheckCondition::reset()
{
    current_min_entropy_ = std::numeric_limits<double>::infinity();
}

std::string PredictionCheckCondition::description() const
{
    std::ostringstream oss;
    oss << "PredictionCheckCondition(";
    
    if (target_entropy_.has_value()) {
        oss << "target=" << target_entropy_.value()
            << ", multiplier=" << divergence_multiplier_
            << ", delay=" << prediction_delay_
            << ", confidence=" << min_confidence_;
    } else {
        oss << "target=none";
    }
    
    oss << ")";
    return oss.str();
}

bool PredictionCheckCondition::isPredictedDivergence(double predicted_entropy) const
{
    // Divergence check 1: Predicted entropy significantly exceeds target
    // If predictor forecasts entropy will be > target × multiplier, divergence detected
    double target = target_entropy_.value();
    double threshold = target * divergence_multiplier_;
    
    if (predicted_entropy > threshold) {
        return true;
    }
    
    // Divergence check 2: Currently above target and heading higher
    // If we've never gone below target, and predictor says we're going even higher,
    // the trajectory is irrecoverably divergent
    if (current_min_entropy_ > target && predicted_entropy > current_min_entropy_) {
        return true;
    }
    
    return false;
}

} // namespace entropy
