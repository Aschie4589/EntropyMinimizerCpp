#include <iostream>
#include <vector>

#include "minimizer/config/config_builder.h"
#include "minimizer/orchestration/run_orchestrator.h"
#include "minimizer/orchestration/minimizer_orchestrator.h"
#include "minimizer/algorithm/minimization_strategy.h"
#include "compute/core/ComputeTypes.h"
#include "compute/device/DeviceFactory.h"
#include "channel/generator/random_generator.h"
#include "utilities/messaging/message_handler.h"
#include "utilities/messaging/printer.h"
#include "minimizer/orchestration/types.h"

int main(int argc, char const *argv[])
{

    utils::MessageHandler msg_handler;
    msg_handler.addSink(std::make_unique<utils::Printer>(-1, true));

    // Setup generation of random kraus operators
    channel::RandomGeneratorConfig rg_config;
    rg_config.kraus_number = 15;
    rg_config.kraus_in_dimension = 512;
    rg_config.kraus_out_dimension = 512;

    // Preallocate kraus operators vector
    std::vector<std::complex<double>> kraus_ops(
        rg_config.kraus_number * 
        rg_config.kraus_in_dimension * 
        rg_config.kraus_out_dimension
    );

    channel::RandomGenerator random_generator(rg_config, msg_handler);

    int ret = random_generator.generate(&kraus_ops);
    if (ret != 0) {
        std::cerr << "RandomGenerator failed with error code: " << ret << std::endl;
        return ret;
    }

    // Setup config
    entropy::MinimizerConfig config = entropy::ConfigBuilder()
        // Algorithm
        .setEpsilon(1e-4)
        .setPrecision(PrecisionType::DOUBLE)
        .setMaxIterations(100000) // For testing
        .setConvergenceTolerance(1e-14)
        .setConvergenceWindow(20)
        .setNumAttempts(20)

        // Prediction
        .enablePrediction(false)
        .setPredictionStrategy(entropy::PredictionStrategyType::AUTO)
        .setPredictionWindowSize(200)
        .setRSquaredThreshold(0.999)
        .setPredictionConvergenceTolerance(1e-15)
        .setPredictionMultiplier(1.05)
        .setMinDataPoints(100)

        // Checkpointing
        .enableCheckpoints(true)
        .setCheckpointInterval(5000)
        .setCheckpointDirectory("checkpoints/")
        .setCheckpointFilenamePattern("{run_id}_iter{iteration}.dat")
        .setKeepLastN(3)

        // Logging
        .enableLogging(true)
        .setLogInterval(100)
        .setPrintToConsole(true)
        .setColorEnabled(true)

        // Resource Config
        .setDesiredGpus(2)
        .setDesiredCpus(0)
        .setGpuSelectionPolicy(entropy::ResourceConfig::GPUSelectionPolicy::MOST_FREE_MEMORY)
        .setGpuConfigFile("/home/tommasoa/EntropyMinimizerCpp/configs/gpu_config.yaml")
        .setAllowDynamicScaling(true)

        .build();

    // Convert kraus operators to HostKrausOperators
    entropy::HostKrausOperators host_kraus;
    host_kraus = entropy::HostKrausOperators::fromDouble(
        kraus_ops,
        rg_config.kraus_number,
        rg_config.kraus_in_dimension,
        rg_config.kraus_out_dimension
    );

    // Now for something more complicated. Let us try out the minimizer orchestrator class.
    entropy::MinimizerOrchestrator minimizer_orchestrator;

    auto moe_result = minimizer_orchestrator.findMOE(
        config,
        host_kraus,
        rg_config.kraus_in_dimension
    );

    if (moe_result.error_type != entropy::RunErrorType::NONE) {
        msg_handler.error(
            "MOE minimization failed with error: " + moe_result.error_message
        );
        return -1;
    }

    // Now print all the results from all the runs
    msg_handler.info("MOE minimization completed successfully.");
    msg_handler.info("Final entropy: " + std::to_string(moe_result.final_entropy));
    msg_handler.info("Iterations taken: " + std::to_string(moe_result.iterations_taken));
    msg_handler.info("Runtime (seconds): " + std::to_string(moe_result.runtime_seconds));


    return 0;
}