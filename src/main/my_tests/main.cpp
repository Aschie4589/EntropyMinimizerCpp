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


int main(int argc, char const *argv[])
{

    utils::MessageHandler msg_handler;
    msg_handler.addSink(std::make_unique<utils::Printer>(-1, true));

    // Setup generation of random kraus operators
    channel::RandomGeneratorConfig rg_config;
    rg_config.kraus_number = 30;
    rg_config.kraus_in_dimension = 2048;
    rg_config.kraus_out_dimension = 2048;

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

    // Print the first few entries
    for (size_t i = 0; i < std::min(kraus_ops.size(), size_t(10)); ++i) {
        std::cout << "Kraus[" << i << "] = " << kraus_ops[i] << std::endl;
    }

    // So far so good


    // Setup config
    entropy::MinimizerConfig config = entropy::ConfigBuilder()
        // Algorithm
        .setEpsilon(1e-4)
        .setPrecision(PrecisionType::DOUBLE)
        .setMaxIterations(100) // For testing
        .setConvergenceTolerance(1e-12)
        .setConvergenceWindow(20)
        .setNumAttempts(5)

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

        // Physical device
        .setDeviceId(0)  // Use GPU 0 consistently

        .build();

    // Convert kraus operators to HostKrausOperators
    HostKrausOperators host_kraus;
    host_kraus = HostKrausOperators::fromDouble(
        kraus_ops,
        rg_config.kraus_number,
        rg_config.kraus_in_dimension,
        rg_config.kraus_out_dimension
    );

    // Use GPU -> obtain a GPU device from DeviceFactory
    std::unique_ptr<IComputeDevice> device = DeviceFactory::create(
        DeviceFactory::DeviceType::CUDA,
        config.algorithm.device_id
    );

    // Progress callback
    auto progress_callback = [&msg_handler](int run_id, int iteration, double entropy) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(15) << entropy;
        msg_handler.info(
            "Run " + std::to_string(run_id) +
            " Iteration " + std::to_string(iteration) +
            " Entropy (15 decimals) " + oss.str()
        );
    };

    // Create RunOrchestrator
    {
        entropy::RunOrchestrator orchestrator(
            config,
            host_kraus,
            rg_config.kraus_in_dimension,
            *device,
            progress_callback
        );

        // Create random host vector
        std::vector<std::complex<double>> initial_vector(rg_config.kraus_in_dimension, {0.0, 0.0});
        for (size_t i = 0; i < initial_vector.size(); ++i) {
            initial_vector[i] = {static_cast<double>(rand()) / RAND_MAX, 0.0};
        }
        // normalize
        double norm = 0.0;
        for (const auto& val : initial_vector) {
            norm += std::norm(val);
        }
        norm = std::sqrt(norm);
        for (auto& val : initial_vector) {
            val /= norm;
        }

        // Turn into host vector
        entropy::HostVector initial_vector_h;
        initial_vector_h.data = initial_vector;
        initial_vector_h.dimension = static_cast<int>(initial_vector.size());
        // run id
        int run_id = 1;

        // Execute run
        //auto result = orchestrator.execute(run_id, initial_vector_h);
        // Here is where debugging is needed (above)

        //if (result.error_type != entropy::RunErrorType::NONE) {
        //    msg_handler.error(
        //        "Minimization run failed with error: " + result.error_message
        //    );
        //    return -1;
        //}
    }

    // Now for something more complicated. Let us try out the minimizer orchestrator class.
    entropy::MinimizerOrchestrator minimizer_orchestrator;
    entropy::ResourceConfig resource_config = entropy::ResourceConfig::createGPUOnly(3);
    resource_config.gpu_config_file = "/home/tommasoa/EntropyMinimizerCpp/configs/gpu_config.yaml";
    auto moe_result = minimizer_orchestrator.findMOE(
        config,
        host_kraus,
        rg_config.kraus_in_dimension,
        resource_config
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