#include "minimizer/state/entropy_predictor.h"
#include <stdexcept>

namespace entropy {

EntropyPredictor::EntropyPredictor(
    std::unique_ptr<PredictionStrategy> strategy,
    size_t window_size)
    : strategy_(std::move(strategy)),
      entropy_history_(window_size) {
    
    if (!strategy_) {
        throw std::invalid_argument("PredictionStrategy cannot be null");
    }
    
    if (window_size < 2) {
        throw std::invalid_argument("Window size must be at least 2");
    }
}

void EntropyPredictor::addDataPoint(double entropy) {
    entropy_history_.push(entropy);
}

void EntropyPredictor::reset() {
    entropy_history_.clear();
}

PredictionResult EntropyPredictor::predict() {
    if (!hasEnoughData()) {
        return PredictionResult();  // Invalid result
    }
    
    return strategy_->predict(entropy_history_);
}

bool EntropyPredictor::hasEnoughData() const {
    return entropy_history_.size() >= strategy_->getMinimumDataPoints();
}

void EntropyPredictor::setStrategy(std::unique_ptr<PredictionStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("PredictionStrategy cannot be null");
    }
    
    strategy_ = std::move(strategy);
}

} // namespace entropy
