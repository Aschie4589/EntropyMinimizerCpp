#include "minimizer/orchestration/run_orchestrator.h"
#include "minimizer/management/checkpoint_manager.h"
#include "minimizer/stopping/max_iterations_condition.h"
#include "minimizer/stopping/convergence_condition.h"
#include "minimizer/stopping/numerical_instability_condition.h"
#include "minimizer/stopping/prediction_check_condition.h"
#include "minimizer/prediction/exponential_fitting_strategy.h"
#include "minimizer/prediction/linear_extrapolation_strategy.h"
#include "minimizer/algorithm/strategy_factory.h"
#include "minimizer/state/entropy_manager.h"
#include "utilities/serializer/serializer.h"
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <memory>

#include <iostream>

//DEBUG LOGGING
#include "utilities/messaging/DEBUG_LOGGER.h"

namespace entropy {

RunOrchestrator::~RunOrchestrator() = default;


// Construct RunOrchestrator
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
    
    
    // 2. Convergence (second priority - plateau detected)
    conditions_->addCondition(
        std::make_unique<ConvergenceCondition>(
            config.stopping.convergence_window,
            config.stopping.convergence_tolerance
        )
    );
    
    // 3. Max iterations (lowest priority - fallback)
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
        
        // Add prediction check condition if target entropy is set
        // This allows early termination when predictor detects unrecoverable divergence
        if (config.stopping.target_entropy.has_value()) {
            conditions_->addCondition(
                std::make_unique<PredictionCheckCondition>(
                    predictor_.get(),
                    config.stopping.target_entropy,
                    config.prediction.prediction_multiplier,  // Stop if predicted > target × multiplier
                    config.prediction.min_data_points,   // Wait before checking predictions
                    config.prediction.rsquared_threshold  // Only trust predictions with R² >= threshold
                )
            );
        }
    }
    
    // Create CheckpointManager if enabled
    if (config.checkpoint.enabled) {
        auto serializer = std::make_unique<utils::VectorSerializer>();
        checkpoint_mgr_ = std::make_unique<CheckpointManager>(
            config.checkpoint,
            std::move(serializer)
        );
    }
}

RunResult RunOrchestrator::execute(int run_id, const HostVector& initial_vector) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    DEBUG_LOG("RunOrchestrator: Starting execution of run_id=" + std::to_string(run_id), "run_orchestrator_log.txt");
    
    // Open log file for this run
    DEBUG_LOG("=== RUN START ===", "run_orchestrator_log.txt");
    DEBUG_LOG("Run ID: " + std::to_string(run_id), "run_orchestrator_log.txt");
    DEBUG_LOG("Initial vector dimension: " + std::to_string(initial_vector.dimension), "run_orchestrator_log.txt");
    DEBUG_LOG("Config: epsilon=" + std::to_string(config_.algorithm.epsilon) + 
              ", max_iterations=" + std::to_string(config_.stopping.max_iterations) +
              ", precision=" + std::string(config_.algorithm.precision == PrecisionType::DOUBLE ? "DOUBLE" : "FLOAT"), "run_orchestrator_log.txt");
    
    try {
        // Validate initial vector
        DEBUG_LOG("Validating initial vector...", "run_orchestrator_log.txt");
        if (initial_vector.data.size() != static_cast<size_t>(input_dim_)) {
            std::string error_msg = "Initial vector dimension mismatch: got " +
                std::to_string(initial_vector.data.size()) +
                ", expected " + std::to_string(input_dim_);
            DEBUG_LOG("ERROR: " + error_msg, "run_orchestrator_log.txt");
            throw std::invalid_argument(error_msg);
        }
        DEBUG_LOG("Initial vector validation passed", "run_orchestrator_log.txt");
        
        // Reset state for new run
        DEBUG_LOG("Resetting run state...", "run_orchestrator_log.txt");
        current_iteration_ = 0;
        min_entropy_seen_ = std::numeric_limits<double>::infinity();
        conditions_->reset();
        if (predictor_) {
            predictor_->reset();
        }
        DEBUG_LOG("Run state reset complete", "run_orchestrator_log.txt");
        
        // Initialize algorithm with starting vector
        DEBUG_LOG("Initializing algorithm manager...", "run_orchestrator_log.txt");
        algorithm_mgr_->initialize(initial_vector.data);
        DEBUG_LOG("Algorithm manager initialized", "run_orchestrator_log.txt");
   

        // Execute main minimization loop
        DEBUG_LOG("Starting main minimization loop...", "run_orchestrator_log.txt");
        double final_entropy = mainLoop(run_id);
        DEBUG_LOG("Main loop completed with final_entropy=" + std::to_string(final_entropy), "run_orchestrator_log.txt");
        
        // Calculate runtime
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        DEBUG_LOG("Total runtime: " + std::to_string(elapsed.count()) + " seconds", "run_orchestrator_log.txt");
        DEBUG_LOG("Total iterations: " + std::to_string(current_iteration_), "run_orchestrator_log.txt");
        DEBUG_LOG("Minimum entropy achieved: " + std::to_string(min_entropy_seen_), "run_orchestrator_log.txt");
        
        // Build and return result
        DEBUG_LOG("=== RUN SUCCESS ===", "run_orchestrator_log.txt");
        return buildResult(run_id, final_entropy, current_iteration_, elapsed.count());
        
    } catch (const std::invalid_argument& e) {
        // Input validation error
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::string error_msg = "Input validation failed: " + std::string(e.what());
        DEBUG_LOG("=== RUN FAILED ===", "run_orchestrator_log.txt");
        DEBUG_LOG("Error Type: INVALID_INPUT", "run_orchestrator_log.txt");
        DEBUG_LOG("Error: " + error_msg, "run_orchestrator_log.txt");
        DEBUG_LOG("Iterations completed: " + std::to_string(current_iteration_), "run_orchestrator_log.txt");
        DEBUG_LOG("Runtime: " + std::to_string(elapsed.count()) + " seconds", "run_orchestrator_log.txt");
        
        RunResult error_result;
        error_result.run_id = run_id;
        error_result.final_entropy = min_entropy_seen_;
        error_result.iterations_taken = current_iteration_;
        error_result.runtime_seconds = elapsed.count();
        error_result.error_type = RunErrorType::INVALID_INPUT;
        error_result.error_message = error_msg;
        
        return error_result;
        
    } catch (const std::runtime_error& e) {
        // Runtime error (could be device error, algorithm failure, etc.)
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::string msg = e.what();
        RunErrorType error_type = RunErrorType::ALGORITHM_FAILURE;
        
        // Try to categorize runtime errors
        if (msg.find("device") != std::string::npos || 
            msg.find("CUDA") != std::string::npos ||
            msg.find("GPU") != std::string::npos) {
            error_type = RunErrorType::DEVICE_ERROR;
        } else if (msg.find("checkpoint") != std::string::npos) {
            error_type = RunErrorType::CHECKPOINT_FAILURE;
        }
        
        std::string error_msg = "Runtime error: " + msg;
        DEBUG_LOG("=== RUN FAILED ===", "run_orchestrator_log.txt");
        DEBUG_LOG("Error Type: " + std::to_string(static_cast<int>(error_type)), "run_orchestrator_log.txt");
        DEBUG_LOG("Error: " + error_msg, "run_orchestrator_log.txt");
        DEBUG_LOG("Iterations completed: " + std::to_string(current_iteration_), "run_orchestrator_log.txt");
        DEBUG_LOG("Runtime: " + std::to_string(elapsed.count()) + " seconds", "run_orchestrator_log.txt");
        
        RunResult error_result;
        error_result.run_id = run_id;
        error_result.final_entropy = min_entropy_seen_;
        error_result.iterations_taken = current_iteration_;
        error_result.runtime_seconds = elapsed.count();
        error_result.error_type = error_type;
        error_result.error_message = error_msg;
        
        return error_result;
        
    } catch (const std::exception& e) {
        // Generic exception
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::string error_msg = "Exception during execution: " + std::string(e.what());
        DEBUG_LOG("=== RUN FAILED ===", "run_orchestrator_log.txt");
        DEBUG_LOG("Error Type: EXCEPTION", "run_orchestrator_log.txt");
        DEBUG_LOG("Error: " + error_msg, "run_orchestrator_log.txt");
        DEBUG_LOG("Iterations completed: " + std::to_string(current_iteration_), "run_orchestrator_log.txt");
        DEBUG_LOG("Runtime: " + std::to_string(elapsed.count()) + " seconds", "run_orchestrator_log.txt");
        
        RunResult error_result;
        error_result.run_id = run_id;
        error_result.final_entropy = min_entropy_seen_;
        error_result.iterations_taken = current_iteration_;
        error_result.runtime_seconds = elapsed.count();
        error_result.error_type = RunErrorType::UNKNOWN;
        error_result.error_message = error_msg;
        
        return error_result;
        
    } catch (...) {
        // Non-standard exception
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        DEBUG_LOG("=== RUN FAILED ===", "run_orchestrator_log.txt");
        DEBUG_LOG("Error Type: UNKNOWN_EXCEPTION", "run_orchestrator_log.txt");
        DEBUG_LOG("Error: Unknown exception during execution", "run_orchestrator_log.txt");
        DEBUG_LOG("Iterations completed: " + std::to_string(current_iteration_), "run_orchestrator_log.txt");
        DEBUG_LOG("Runtime: " + std::to_string(elapsed.count()) + " seconds", "run_orchestrator_log.txt");
        
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
    
    DEBUG_LOG("Main loop initialized: current_entropy=inf, previous_entropy=inf", "run_orchestrator_log.txt");
    
    // Main iteration loop
    while (true) {
        DEBUG_LOG("--- Iteration " + std::to_string(current_iteration_) + " START ---", "run_orchestrator_log.txt");
        
        try {
            // Perform one minimization iteration
            DEBUG_LOG("  Calling algorithm_mgr_->stepOnce()...", "run_orchestrator_log.txt");
            algorithm_mgr_->stepOnce();
            DEBUG_LOG("  stepOnce() completed", "run_orchestrator_log.txt");
            
            // Get current entropy (use estimated entropy, not epsilon-perturbed)
            DEBUG_LOG("  Fetching estimated entropy...", "run_orchestrator_log.txt");
            previous_entropy = current_entropy;
            current_entropy = algorithm_mgr_->getEstimatedEntropy();
            DEBUG_LOG("  Previous entropy: " + std::to_string(previous_entropy) +
                     ", Current entropy: " + std::to_string(current_entropy), "run_orchestrator_log.txt");
            
            // Update minimum
            if (current_entropy < min_entropy_seen_) {
                DEBUG_LOG("  New minimum entropy: " + std::to_string(current_entropy), "run_orchestrator_log.txt");
                min_entropy_seen_ = current_entropy;
            }
            
            // Add to predictor if enabled
            if (predictor_) {
                DEBUG_LOG("  Adding data point to predictor...", "run_orchestrator_log.txt");
                predictor_->addDataPoint(current_entropy);
                DEBUG_LOG("  Data point added to predictor", "run_orchestrator_log.txt");
            }
            
            // Check stopping conditions
            DEBUG_LOG("  Checking stopping conditions...", "run_orchestrator_log.txt");
            StopReason reason = checkStoppingConditions(
                current_iteration_,
                current_entropy,
                previous_entropy
            );
            DEBUG_LOG("  Stopping condition result: " + std::to_string(static_cast<int>(reason)), "run_orchestrator_log.txt");
            
            // Handle checkpoints
            if (checkpoint_mgr_) {
                if (checkpoint_mgr_->shouldCheckpoint(current_iteration_)) {
                    DEBUG_LOG("  Checkpoint condition met, saving checkpoint...", "run_orchestrator_log.txt");
                    handleCheckpoint(run_id, current_iteration_, current_entropy, min_entropy_seen_);
                    DEBUG_LOG("  Checkpoint saved", "run_orchestrator_log.txt");
                } else {
                    DEBUG_LOG("  No checkpoint needed (interval not met)", "run_orchestrator_log.txt");
                }
            }
            
            // Invoke progress callback (with exception guard)
            if (progress_cb_) {
                DEBUG_LOG("  Invoking progress callback...", "run_orchestrator_log.txt");
                try {
                    progress_cb_(run_id, current_iteration_, current_entropy);
                    DEBUG_LOG("  Progress callback completed", "run_orchestrator_log.txt");
                } catch (const std::exception& e) {
                    // Log error but don't crash the run
                    DEBUG_LOG("  WARNING: Progress callback threw exception: " + std::string(e.what()), "run_orchestrator_log.txt");
                    (void)e;
                } catch (...) {
                    // Catch all other exceptions (non-std types)
                    // Silently continue
                    DEBUG_LOG("  WARNING: Progress callback threw unknown exception", "run_orchestrator_log.txt");
                }
            }
            
            // Increment iteration counter
            current_iteration_++;
            
            // Check if we should stop
            if (reason != StopReason::CONTINUE) {
                DEBUG_LOG("--- Iteration " + std::to_string(current_iteration_ - 1) + " END (STOP REASON: " + 
                         std::to_string(static_cast<int>(reason)) + ") ---", "run_orchestrator_log.txt");
                std::cout << "Stopping minimization at iteration " 
                          << current_iteration_ 
                          << " due to reason: " 
                          << static_cast<int>(reason) 
                          << std::endl;
                break;
            }
            
            DEBUG_LOG("--- Iteration " + std::to_string(current_iteration_ - 1) + " END (CONTINUE) ---", "run_orchestrator_log.txt");
            
        } catch (const std::exception& e) {
            DEBUG_LOG("ERROR during iteration " + std::to_string(current_iteration_) + ": " + 
                     std::string(e.what()), "run_orchestrator_log.txt");
            DEBUG_LOG("--- Iteration " + std::to_string(current_iteration_) + " END (EXCEPTION) ---", "run_orchestrator_log.txt");
            throw;  // Re-throw to be caught by execute()
        } catch (...) {
            DEBUG_LOG("ERROR during iteration " + std::to_string(current_iteration_) + ": Unknown exception", "run_orchestrator_log.txt");
            DEBUG_LOG("--- Iteration " + std::to_string(current_iteration_) + " END (UNKNOWN EXCEPTION) ---", "run_orchestrator_log.txt");
            throw;  // Re-throw to be caught by execute()
        }
    }
    
    DEBUG_LOG("Main loop exited normally", "run_orchestrator_log.txt");
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
