#include "minimizer/stopping/conditions_checker.h"

namespace entropy {

ConditionsChecker::ConditionsChecker() = default;

void ConditionsChecker::addCondition(std::unique_ptr<Condition> condition) {
    conditions_.push_back(std::move(condition));
}

StopReason ConditionsChecker::checkStoppingConditions(size_t iteration,
                                                      double current_entropy,
                                                      double previous_entropy) {
    for (auto& condition : conditions_) {
        StopReason reason = condition->check(iteration, current_entropy, previous_entropy);
        if (reason != StopReason::CONTINUE) {
            return reason;
        }
    }
    return StopReason::CONTINUE;
}

void ConditionsChecker::reset() {
    for (auto& condition : conditions_) {
        condition->reset();
    }
}

size_t ConditionsChecker::size() const {
    return conditions_.size();
}

std::string ConditionsChecker::description() const {
    std::string desc = "ConditionsChecker with " + std::to_string(conditions_.size()) + " conditions:\n";
    for (size_t i = 0; i < conditions_.size(); ++i) {
        desc += "  " + std::to_string(i) + ". " + conditions_[i]->description() + "\n";
    }
    return desc;
}

} // namespace entropy
