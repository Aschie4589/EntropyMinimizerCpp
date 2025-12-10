#include "minimizer/orchestration/run_orchestrator.h"
#include "minimizer/management/checkpoint_manager.h"
#include "minimizer/stopping/max_iterations_condition.h"
#include "minimizer/stopping/convergence_condition.h"
#include "minimizer/stopping/numerical_instability_condition.h"
#include "minimizer/stopping/target_entropy_condition.h"
#include "minimizer/prediction/exponential_fitting_strategy.h"
#include "minimizer/prediction/linear_extrapolation_strategy.h"
#include "minimizer/algorithm/strategy_factory.h"
#include "minimizer/state/entropy_manager.h"
#include "utilities/serializer/serializer.h"
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <memory>

namespace entropy {

RunOrchestrator::~RunOrchestrator() = default;

RunOrchestrator::RunOrchestrator(
    const MinimizerConfig& config,
    const HostKrausOperators& kraus_ops,
    int input_dim,
    IComputeDevice& device,
    ProgressCallback progress_cb
)
    : config_(config)
    , device_(device)
    , input_dim_(input_dim)
    , progress_cb_(progress_cb)
    , current_iteration_(0)
    , min_entropy_seen_(std::numeric_limits<double>::infinity())
{
    // Validate config
    if (config.algorithm.epsilon <= 0.0 || config.algorithm.epsilon >= 1.0) {
        throw std::invalid_argument(
            "Invalid epsilon: " + std::to_string(config.algorithm.epsilon) +
            " (must be in (0,1))"
        );
    }
    
    // Create EntropyManager (must be created before AlgorithmManager)
    // EntropyManager just needs epsilon and system_dimension (output_dim from Kraus ops)
    entropy_manager_ = std::make_unique<EntropyManager>(
        config.algorithm.epsilon,
        kraus_ops.output_dim  // system dimension
    );
    
    // Create AlgorithmManager (holds reference to entropy_manager_)
    algorithm_mgr_ = std::make_unique<AlgorithmManager>(
        device_,
        *entropy_manager_,
        kraus_ops,
        config.algorithm.epsilon,
        config.algorithm.precision
    );
    
    // Set minimization strategy using factory with AUTO selection
    auto strategy = StrategyFactory::create(
        device_,
        kraus_ops.kraus_count,
        kraus_ops.input_dim,
        kraus_ops.output_dim,
        config.algorithm.precision
    );
    algorithm_mgr_->setStrategy(std::move(strategy));
    
    // Create ConditionsChecker and add conditions based on config
    conditions_ = std::make_unique<ConditionsChecker>();
    
    // Add stopping conditions in priority order
    // 1. Numerical instability (highest priority - must catch errors immediately)
    conditions_->addCondition(
        std::make_unique<NumericalInstabilityCondition>()
    );
    
    // 2. Target entropy (second priority - goal reached)
    if (config.stopping.target_entropy.has_value()) {
        conditions_->addCondition(
            std::make_unique<TargetEntropyCondition>(
                config.stopping.target_entropy.value(),
                1e-6  // tolerance
            )
        );
    }
    
    // 3. Convergence (third priority - plateau detected)
    conditions_->addCondition(
        std::make_unique<ConvergenceCondition>(
            config.stopping.convergence_window,
            config.stopping.convergence_tolerance
        )
    );
    
    // 4. Max iterations (lowest priority - fallback)
    conditions_->addCondition(
        std::make_unique<MaxIterationsCondition>(config.stopping.max_iterations)
    );
    
    // Create EntropyPredictor if enabled
    if (config.prediction.enabled) {
        std::unique_ptr<PredictionStrategy> strategy;
        
        // Select prediction strategy based on config
        if (config.prediction.strategy_type == PredictionStrategyType::EXPONENTIAL) {
            strategy = std::make_unique<ExponentialFittingStrategy>(
                config.prediction.min_data_points
            );
        } else if (config.prediction.strategy_type == PredictionStrategyType::LINEAR) {
            strategy = std::make_unique<LinearExtrapolationStrategy>(
                config.prediction.min_data_points
            );
        } else {
            // Default to exponential
            strategy = std::make_unique<ExponentialFittingStrategy>(
                config.prediction.min_data_points
            );
        }
        
        predictor_ = std::make_unique<EntropyPredictor>(
            std::move(strategy),
            config.prediction.window_size
        );
    }
    
    // Create CheckpointManager if enabled
    if (config.checkpoint.enabled) {
        auto serializer = std::make_unique<VectorSerializer>();
        checkpoint_mgr_ = std::make_unique<CheckpointManager>(
            config.checkpoint,
            std::move(serializer)
        );
    }
}

RunResult RunOrchestrator::execute(int run_id, const HostVector& initial_vector) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate initial vector
        if (initial_vector.data.size() != static_cast<size_t>(input_dim_)) {
            throw std::invalid_argument(
                "Initial vector dimension mismatch: got " +
                std::to_string(initial_vector.data.size()) +
                ", expected " + std::to_string(input_dim_)
            );
        }
        
        // Reset state for new run
        current_iteration_ = 0;
        min_entropy_seen_ = std::numeric_limits<double>::infinity();
        conditions_->reset();
        if (predictor_) {
            predictor_->reset();
        }
        
        // Initialize algorithm with starting vector
        algorithm_mgr_->initialize(initial_vector.data);
        
        // Execute main minimization loop
        double final_entropy = mainLoop(run_id);
        
        // Calculate runtime
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        // Build and return result
        return buildResult(run_id, final_entropy, current_iteration_, elapsed.count());
        
    } catch (const std::invalid_argument& e) {
        // Input validation error
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        RunResult error_result;
        error_result.run_id = run_id;
        error_result.final_entropy = min_entropy_seen_;
        error_result.iterations_taken = current_iteration_;
        error_result.runtime_seconds = elapsed.count();
        error_result.error_type = RunErrorType::INVALID_INPUT;
        error_result.error_message = std::string("Input validation failed: ") + e.what();
        
        return error_result;
        
    } catch (const std::runtime_error& e) {
        // Runtime error (could be device error, algorithm failure, etc.)
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        RunResult error_result;
        error_result.run_id = run_id;
        error_result.final_entropy = min_entropy_seen_;
        error_result.iterations_taken = current_iteration_;
        error_result.runtime_seconds = elapsed.count();
        
        // Try to categorize runtime errors
        std::string msg = e.what();
        if (msg.find("device") != std::string::npos || 
            msg.find("CUDA") != std::string::npos ||
            msg.find("GPU") != std::string::npos) {
            error_result.error_type = RunErrorType::DEVICE_ERROR;
        } else if (msg.find("checkpoint") != std::string::npos) {
            error_result.error_type = RunErrorType::CHECKPOINT_FAILURE;
        } else {
            error_result.error_type = RunErrorType::ALGORITHM_FAILURE;
        }
        error_result.error_message = std::string("Runtime error: ") + e.what();
        
        return error_result;
        
    } catch (const std::exception& e) {
        // Generic exception
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        RunResult error_result;
        error_result.run_id = run_id;
        error_result.final_entropy = min_entropy_seen_;
        error_result.iterations_taken = current_iteration_;
        error_result.runtime_seconds = elapsed.count();
        error_result.error_type = RunErrorType::UNKNOWN;
        error_result.error_message = std::string("Exception during execution: ") + e.what();
        
        return error_result;
        
    } catch (...) {
        // Non-standard exception
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        RunResult error_result;
        error_result.run_id = run_id;
        error_result.final_entropy = min_entropy_seen_;
        error_result.iterations_taken = current_iteration_;
        error_result.runtime_seconds = elapsed.count();
        error_result.error_type = RunErrorType::UNKNOWN;
        error_result.error_message = "Unknown exception during execution";
        
        return error_result;
    }
}

double RunOrchestrator::mainLoop(int run_id) {
    double current_entropy = std::numeric_limits<double>::infinity();
    double previous_entropy = std::numeric_limits<double>::infinity();
    
    // Main iteration loop
    while (true) {
        // Perform one minimization iteration
        algorithm_mgr_->stepOnce();
        
        // Get current entropy (use estimated entropy, not epsilon-perturbed)
        previous_entropy = current_entropy;
        current_entropy = algorithm_mgr_->getEstimatedEntropy();
        
        // Update minimum
        if (current_entropy < min_entropy_seen_) {
            min_entropy_seen_ = current_entropy;
        }
        
        // Add to predictor if enabled
        if (predictor_) {
            predictor_->addDataPoint(current_entropy);
        }
        
        // Check stopping conditions
        StopReason reason = checkStoppingConditions(
            current_iteration_,
            current_entropy,
            previous_entropy
        );
        
        // Handle checkpoints
        if (checkpoint_mgr_ && checkpoint_mgr_->shouldCheckpoint(current_iteration_)) {
            handleCheckpoint(run_id, current_iteration_, current_entropy, min_entropy_seen_);
        }
        
        // Invoke progress callback (with exception guard)
        if (progress_cb_) {
            try {
                progress_cb_(run_id, current_iteration_, current_entropy);
            } catch (const std::exception& e) {
                // Log error but don't crash the run
                // In production, would use proper logging framework
                // For now, silently continue (callback errors shouldn't stop minimization)
                (void)e;
            } catch (...) {
                // Catch all other exceptions (non-std types)
                // Silently continue
            }
        }
        
        // Increment iteration counter
        current_iteration_++;
        
        // Check if we should stop
        if (reason != StopReason::CONTINUE) {
            break;
        }
    }
    
    return current_entropy;
}

StopReason RunOrchestrator::checkStoppingConditions(
    size_t iteration,
    double current_entropy,
    double previous_entropy
) {
    return conditions_->checkStoppingConditions(
        iteration,
        current_entropy,
        previous_entropy
    );
}

void RunOrchestrator::handleCheckpoint(
    int run_id,
    int iteration,
    double current_entropy,
    double min_entropy
) {
    if (!checkpoint_mgr_) {
        return;  // Not configured
    }
    
    try {
        // Build checkpoint metadata
        CheckpointMetadata metadata;
        metadata.run_id = "run_" + std::to_string(run_id);
        metadata.iteration = iteration;
        metadata.current_entropy = current_entropy;
        metadata.run_minimum_entropy = min_entropy;
        metadata.timestamp = std::chrono::system_clock::now();
        
        // Get current vector from algorithm manager and wrap in HostVector
        std::vector<std::complex<double>> vec_data = algorithm_mgr_->getCurrentVector();
        HostVector current_vector(vec_data);
        
        // Save checkpoint
        checkpoint_mgr_->saveCheckpoint(
            metadata.run_id,
            iteration,
            current_vector,
            metadata
        );
        
    } catch (const std::exception& e) {
        // Log error but don't stop run
        // In production, use proper logging framework
        // For now, swallow the exception to prevent run failure
        (void)e;
    }
}

RunResult RunOrchestrator::buildResult(
    int run_id,
    double final_entropy,
    int iterations,
    double runtime_seconds
) const {
    RunResult result;
    result.run_id = run_id;
    result.final_entropy = final_entropy;
    result.iterations_taken = iterations;
    result.runtime_seconds = runtime_seconds;
    result.error_type = RunErrorType::NONE;  // Success
    result.error_message = "";  // No error
    
    // Get final vector from algorithm manager
    result.final_vector = HostVector(algorithm_mgr_->getCurrentVector());
    
    return result;
}

} // namespace entropy
