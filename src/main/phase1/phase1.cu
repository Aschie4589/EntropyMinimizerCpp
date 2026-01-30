

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <complex>
#include <random>
#include "yaml-cpp/yaml.h"
#include "tensor_entropy.h"

#include "minimizer/config/minimizer_config.h"
#include "minimizer/config/config_loader.h"
#include "minimizer/orchestration/gpu_registry.h"
#include "minimizer/orchestration/device_pool.h"
#include "minimizer/orchestration/concurrent_queue.h"
#include "minimizer/orchestration/types.h"
#include "minimizer/orchestration/result_collector.h"
#include "minimizer/orchestration/worker_thread_pool.h"
#include "minimizer/orchestration/progress_tracker.h"


#include "channel/generator/random_generator.h"
#include "utilities/messaging/message_handler.h"
#include "utilities/messaging/printer.h"
#include "utilities/serializer/serializer.h"



/*
    Main application for Phase 1 of the Entropy Minimizer project.
    This application generates multiple channels and performs low and medium accuracy runs
    to identify the most promising channels for further high-accuracy processing.

    - The channels are generated
    - Low-accuracy runs are performed to quickly evaluate the channels
    - A fraction of the least promising vectors from each channel are discarded
    - Medium-accuracy runs are performed on the remaining vectors
    - A fraction of the least promising channels are discarded based on the medium-accuracy results

    At the end of this phase, the most promising channels and vectors are retained for high-accuracy processing in subsequent phases.

*/

#define PHASE1_CONFIG_FILE "/home/tommasoa/EntropyMinimizerCpp/configs/phase1.yml"


/**
 * @brief Reusable callback for GPU configuration changes
 * 
 * Adapts resources dynamically when GPU availability changes during runtime.
 * This function mirrors the logic from MinimizerOrchestrator::adjustResources
 * but works with standalone DevicePool and WorkerThreadPool instances.
 * 
 * @param enabled_gpus List of currently enabled GPU device IDs
 * @param device_pool Reference to the device pool to adjust
 * @param worker_pool Reference to the worker pool to adjust
 * @param pass_name Human-readable name for logging (e.g., "first-pass", "second-pass")
 */
void handleGPUConfigChange(
    const std::vector<int>& enabled_gpus,
    entropy::DevicePool& device_pool,
    entropy::WorkerThreadPool& worker_pool,
    const std::string& pass_name
) {
    std::cout << "\n[" << pass_name << "] ===== GPU Configuration Change Detected =====" << std::endl;
    
    // ========================================================================
    // STEP 1: Identify devices to remove and add
    // ========================================================================
    
    std::vector<int> current_gpus;
    for (const auto& status : device_pool.getDeviceStatus()) {
        if (status.type == entropy::DeviceType::GPU && status.available) {
            current_gpus.push_back(status.physical_device_id);
        }
    }
    
    std::set<int> enabled_set(enabled_gpus.begin(), enabled_gpus.end());
    std::set<int> current_set(current_gpus.begin(), current_gpus.end());
    
    // Devices to remove: in current but not in enabled
    std::vector<int> devices_to_remove;
    std::set_difference(current_set.begin(), current_set.end(),
                       enabled_set.begin(), enabled_set.end(),
                       std::back_inserter(devices_to_remove));
    
    // Devices to add: in enabled but not in current
    std::vector<int> devices_to_add;
    std::set_difference(enabled_set.begin(), enabled_set.end(),
                       current_set.begin(), current_set.end(),
                       std::back_inserter(devices_to_add));
    
    std::cout << "[" << pass_name << "] Current GPUs: ";
    for (int id : current_gpus) std::cout << id << " ";
    std::cout << std::endl;
    
    std::cout << "[" << pass_name << "] Enabled GPUs: ";
    for (int id : enabled_gpus) std::cout << id << " ";
    std::cout << std::endl;
    
    std::cout << "[" << pass_name << "] Devices to remove: " << devices_to_remove.size();
    if (!devices_to_remove.empty()) {
        std::cout << " (";
        for (size_t i = 0; i < devices_to_remove.size(); ++i) {
            std::cout << devices_to_remove[i];
            if (i < devices_to_remove.size() - 1) std::cout << ", ";
        }
        std::cout << ")";
    }
    std::cout << std::endl;
    
    std::cout << "[" << pass_name << "] Devices to add: " << devices_to_add.size();
    if (!devices_to_add.empty()) {
        std::cout << " (";
        for (size_t i = 0; i < devices_to_add.size(); ++i) {
            std::cout << devices_to_add[i];
            if (i < devices_to_add.size() - 1) std::cout << ", ";
        }
        std::cout << ")";
    }
    std::cout << std::endl;
    
    // ========================================================================
    // STEP 2: Mark devices for removal (prevents new acquisitions)
    // ========================================================================
    
    for (int device_id : devices_to_remove) {
        std::cout << "[" << pass_name << "] Marking device " << device_id << " for removal" << std::endl;
        device_pool.markDeviceForRemoval(device_id);
    }
    
    // ========================================================================
    // STEP 3: Gracefully remove workers using marked devices
    // ========================================================================
    
    int total_workers_removed = 0;
    for (int device_id : devices_to_remove) {
        auto worker_ids = worker_pool.getWorkerIDsUsingDevice(device_id);
        std::cout << "[" << pass_name << "] Device " << device_id 
                  << " has " << worker_ids.size() << " worker(s) assigned" << std::endl;
        
        std::cout << "[" << pass_name << "] Removing workers using device " << device_id << std::endl;
        int removed = worker_pool.removeWorkersUsingDevice(device_id);
        total_workers_removed += removed;
        std::cout << "[" << pass_name << "] Removed " << removed << " worker(s) from device " << device_id << std::endl;
    }
    
    std::cout << "[" << pass_name << "] Total workers removed: " << total_workers_removed << std::endl;
    
    // ========================================================================
    // STEP 4: Clean up devices (now safe, ref_count == 0)
    // ========================================================================
    
    std::cout << "[" << pass_name << "] Cleaning up marked devices..." << std::endl;
    device_pool.cleanupMarkedDevices();
    
    // ========================================================================
    // STEP 5: Add new devices
    // ========================================================================
    
    for (int device_id : devices_to_add) {
        std::cout << "[" << pass_name << "] Adding device " << device_id << std::endl;
        try {
            device_pool.addGPU(device_id);
            std::cout << "[" << pass_name << "] Successfully added device " << device_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[" << pass_name << "] Failed to add device " << device_id 
                      << ": " << e.what() << std::endl;
        }
    }
    
    // ========================================================================
    // STEP 6: Add workers for new devices
    // ========================================================================
    
    int workers_to_add = devices_to_add.size();
    if (workers_to_add > 0) {
        std::cout << "[" << pass_name << "] Adding " << workers_to_add << " worker(s) for new devices" << std::endl;
        try {
            worker_pool.addWorkers(workers_to_add);
            std::cout << "[" << pass_name << "] Successfully added " << workers_to_add << " worker(s)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[" << pass_name << "] Failed to add workers: " << e.what() << std::endl;
        }
    }
    
    // ========================================================================
    // Final Status
    // ========================================================================
    
    std::cout << "[" << pass_name << "] ===== Resource Adjustment Complete =====" << std::endl;
    std::cout << "[" << pass_name << "] Total devices: " << device_pool.numDevices() << std::endl;
    std::cout << "[" << pass_name << "] Total workers: " << worker_pool.getWorkerCount() << std::endl;
    std::cout << "[" << pass_name << "] Active workers: " << worker_pool.getActiveWorkerCount() << std::endl;
}


int main(int argc, char** argv){


    // LOAD PARAMETERS FROM YAML CONFIG
    YAML::Node config = YAML::LoadFile(PHASE1_CONFIG_FILE);

    std::vector<int> channel_ns = config["config"]["params"]["N"].as<std::vector<int>>();
    std::vector<int> channel_ds = config["config"]["params"]["d"].as<std::vector<int>>();
    std::string out_folder = config["config"]["output"]["output-directory"].as<std::string>();
    std::string index_file = config["config"]["output"]["index-yaml"].as<std::string>();
    std::string info_file = config["config"]["output"]["info-yaml"].as<std::string>();
    std::string vectors_dirname = config["config"]["output"]["vectors-dirname"].as<std::string>();
    std::string first_pass_dirname = config["config"]["output"]["first-pass-dirname"].as<std::string>();
    std::string second_pass_dirname = config["config"]["output"]["second-pass-dirname"].as<std::string>();
    std::string third_pass_dirname = config["config"]["output"]["third-pass-dirname"].as<std::string>();


    int num_channels = config["config"]["params"]["channel-gen"]["num-channels"].as<int>();
    double eps = config["config"]["params"]["eps"].as<double>();
    double tensor_eps = config["config"]["params"]["moe-tensor-estimation"]["eps"].as<double>();


    // Get the valid (N,d) pairs (d^2 <= N)
    std::vector<std::pair<int,int>> nd_pairs;
    for (int n : channel_ns){
        for (int d : channel_ds){
            if (d*d <= n){
                nd_pairs.push_back(std::make_pair(n,d));
            }
        }
    }

    // Create a message handler object
    auto msg_handler = std::make_unique<utils::MessageHandler>();
    msg_handler->addSink(std::make_unique<utils::Printer>(0, true));



    /*
    ========================================
    INITIAL SETUP: CHANNEL GENERATION
    ========================================
    For each (N,d) pair:
        For each channel to generate:
            Generate random channel with N input/output dimension and d Kraus operators
            Save the Kraus operators to a unique folder
            Compute and save the tensor entropy estimate using TensorEntropyEstimator

    */
    // Create TensorEntropyEstimator to get the tensor entropy of the channels
    TensorEntropyEstimator<double>* tensor_estimator = new TensorEntropyEstimator<double>();
    // Will also need a serializer to save the vectors
    utils::VectorSerializer vec_serializer;

    std::vector<std::complex<double>> h_kraus;

    // YAML for indexing
    YAML::Node index_node;


    // For each (N,d) pair
    for (const auto& nd : nd_pairs){
        int N = nd.first;
        int d = nd.second;
        std::cout << "Preparing to generate channels for N=" << N << ", d=" << d << std::endl;
        channel::RandomGeneratorConfig rg_config;
        rg_config.kraus_number = d;
        rg_config.kraus_in_dimension = N;
        rg_config.kraus_out_dimension = N;
        h_kraus.resize(rg_config.kraus_number * rg_config.kraus_in_dimension * rg_config.kraus_out_dimension);

        // For each channel
        for (int ch=0; ch<num_channels; ch++){
            std::cout << "Generating channel " << ch+1 << " of " << num_channels << std::endl;
            channel::RandomGenerator rg(rg_config, *msg_handler);
            
            // Generate unique string: kraus_N_{N}_d_{d}_ch_{ch}_{YY_MM_DD-hh_mm_ss}
            std::ostringstream oss;
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            oss << "kraus_N_" << N << "_d_" << d << "_ch_" << ch << "_"
                << std::put_time(&tm, "%Y_%m_%d-%H_%M_%S");
            std::string unique_string = oss.str();
            std::cout << "Generated unique string: " << unique_string << std::endl;

            // Create the save folder
            std::string save_folder = out_folder + "/" + unique_string;
            std::cout << "Creating save folder: " << save_folder << std::endl;
            std::filesystem::create_directories(save_folder);

            // Generate kraus (copies to host)
            rg.generate(&h_kraus);

            // Save kraus to file (filename=kraus.dat)
            std::string kraus_filename = save_folder + "/kraus.dat";
            std::cout << "Saving kraus operators to file: " << kraus_filename << std::endl;
            // Use utils::VectorSerializer
            vec_serializer.serialize("kraus", kraus_filename, h_kraus, "Randomly generated Kraus operators", rg_config.kraus_number, rg_config.kraus_in_dimension, rg_config.kraus_out_dimension);
            

            // Compute the tensor product MOE estimate
            tensor_estimator->computeEntropy(reinterpret_cast<cuDoubleComplex*>(h_kraus.data()), rg_config.kraus_number, rg_config.kraus_in_dimension, rg_config.kraus_out_dimension, tensor_eps);


            msg_handler->info("Entropy computed to: " + std::to_string(tensor_estimator->getEstimatedEntropy()) + " +/- " + std::to_string(tensor_estimator->getEstimatedError()));
            // Generate YAML file with information about channel and run
            std::string yaml_filename = save_folder + "/info.yml";
            std::cout << "Saving channel info to YAML file: " << yaml_filename << std::endl;
            YAML::Node info_node;
            info_node["channel"] = YAML::Node();
            info_node["channel"]["N"] = N;
            info_node["channel"]["d"] = d;
            info_node["channel"]["kraus_number"] = rg_config.kraus_number;
            info_node["channel"]["kraus_in_dimension"] = rg_config.kraus_in_dimension;
            info_node["channel"]["kraus_out_dimension"] = rg_config.kraus_out_dimension;
            info_node["run"] = YAML::Node();
            info_node["run"]["channel_index"] = ch;
            info_node["run"]["unique_string"] = unique_string;  
            info_node["analysis"] = YAML::Node();
            info_node["analysis"]["tensor_entropy_estimation"] = YAML::Node();
            info_node["analysis"]["tensor_entropy_estimation"]["eps"] = tensor_eps;
            info_node["analysis"]["tensor_entropy_estimation"]["estimated_entropy"] = tensor_estimator->getEstimatedEntropy();
            info_node["analysis"]["tensor_entropy_estimation"]["estimated_error"] = tensor_estimator->getEstimatedError();
            std::ofstream yaml_file(yaml_filename);
            if (yaml_file.is_open()){
                yaml_file << info_node;
                yaml_file.close();
            } else {
                std::cerr << "Error: Unable to open file for writing YAML info: " << yaml_filename << std::endl;
            }
            
            // Compare to the theoretical entropy of log(d)(2-1/d), d being number of kraus operators
            double theoretical_entropy = std::log(static_cast<double>(d)) * (2.0 - 1.0 / static_cast<double>(d));
            msg_handler->info("Theoretical entropy (tensor channel): " + std::to_string(theoretical_entropy));

            // Add the channel to the index YAML
            YAML::Node channel_entry;
            channel_entry["unique_string"] = unique_string;
            channel_entry["kraus_path"] = kraus_filename;
            channel_entry["folder"] = save_folder;
            channel_entry["discarded"] = false;
            index_node["channels"].push_back(channel_entry);
        }
    }

    // Save the index YAML file
    std::string index_path = out_folder + "/" + index_file;
    std::cout << "Saving index YAML file to: " << index_path << std::endl;
    std::ofstream index_yaml_file(index_path);
    if (index_yaml_file.is_open()){
        index_yaml_file << index_node;
        index_yaml_file.close();
    } else {
        std::cerr << "Error: Unable to open file for writing index YAML: " << index_path << std::endl;
    }

    delete tensor_estimator;
    
    /*  
    =================================================
    INITIAL SETUP: VECTOR GENERATION
    =================================================

    For each channel:
        Generate enough random initial vectors
        Save the vectors to the appropriate folder    
    */

    std::cout << "Initialization: vector generation. Will go through all channels and generate initial vectors." << std::endl;

    // Get the appropriate parameters from config
    int num_initial_vectors = config["config"]["params"]["first-pass"]["num-vectors"].as<int>();
    int first_pass_iters = config["config"]["params"]["first-pass"]["iters"].as<int>();
    double first_pass_vec_survival = config["config"]["params"]["first-pass"]["vec-survival"].as<double>();

    // Load the index YAML file to get the list of channels
    YAML::Node loaded_index = YAML::LoadFile(index_path);

    // Create a random number generator for initial vectors
    std::mt19937 rng_{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, 1.0);

    // Iterate with index so we can modify the YAML node
    for (size_t ch_idx = 0; ch_idx < loaded_index["channels"].size(); ++ch_idx) {
        YAML::Node channel_entry = loaded_index["channels"][ch_idx];
        std::string unique_string = channel_entry["unique_string"].as<std::string>();
        std::string kraus_path = channel_entry["kraus_path"].as<std::string>();
        std::string folder = channel_entry["folder"].as<std::string>();

        std::cout << "Processing channel: " << unique_string << std::endl;
        std::cout << "Kraus path: " << kraus_path << std::endl;
        std::cout << "Folder: " << folder << std::endl;

        // Parse N and d by opening the kraus yaml file
        int N, d;
        YAML::Node kraus_info = YAML::LoadFile(folder + "/info.yml");
        N = kraus_info["channel"]["N"].as<int>();
        d = kraus_info["channel"]["d"].as<int>();

        // Generate the vectors.
        // Make sure the vecs subdir exists
        std::string vecs_folder = folder + "/" + vectors_dirname;
        std::cout << "Creating vectors folder: " << vecs_folder << std::endl;
        std::filesystem::create_directories(vecs_folder);

        // Generate the first pass subdir
        std::string first_pass_folder = vecs_folder + "/" + first_pass_dirname;
        std::cout << "Creating first pass folder: " << first_pass_folder << std::endl;
        std::filesystem::create_directories(first_pass_folder);

        // Now generate the appropriate number of initial vectors
        for (int v=0; v<num_initial_vectors; v++){
            std::cout << "Generating initial vector " << v+1 << " of " << num_initial_vectors << std::endl;

            // Create random name for the vector. Add randomness using rand
            std::ostringstream voss;
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            int rand_suffix = rand() % 10000;
            voss << "vec_" << v << "_" << std::put_time(&tm, "%Y_%m_%d-%H_%M_%S") << "_" << rand_suffix;
            std::string vec_unique_string = voss.str();
            std::cout << "Generated unique vector string: " << vec_unique_string << std::endl;

            // Generate random initial vector
            std::vector<std::complex<double>> vec(N);
            for (int i = 0; i < N; ++i) {
                double real_part = dist(rng_);
                double imag_part = dist(rng_);
                vec[i] = std::complex<double>(real_part, imag_part);
            }
            
            // Normalize to unit norm (required for quantum states)
            double norm = 0.0;
            for (int i = 0; i < N; ++i) {
                norm += std::norm(vec[i]);  // |z|^2
            }
            norm = std::sqrt(norm);
            
            if (norm < 1e-15) {
                // Extremely unlikely but handle gracefully
                throw std::runtime_error(
                    "Error: Generated zero-norm vector during initialization."
                );
            }
            
            for (int i = 0; i < N; ++i) {
                vec[i] /= norm;
            }

            // Now save to file and append to kraus yaml
            std::string vec_filename = first_pass_folder + "/initial/" + vec_unique_string + ".dat";
            std::filesystem::create_directories(first_pass_folder + "/initial/");
            std::cout << "Saving initial vector to file: " << vec_filename << std::endl;
            vec_serializer.serialize("vector", vec_filename, vec, "Initial vector for low-accuracy run", d, N, 1);

            // Append to the channel's initial_vectors list
            if (!channel_entry["initial_vectors"]) {
                channel_entry["initial_vectors"] = YAML::Node(YAML::NodeType::Sequence);
            }
            YAML::Node vector_entry;
            vector_entry["unique_string"] = vec_unique_string;
            kraus_info["initial_vectors"].push_back(vector_entry);

            std::cout << "Initial vector generation and saving complete." << std::endl;
        }

        // Save the kraus info YAML after processing this channel
        std::ofstream kraus_info_save(folder + "/info.yml");
        if (kraus_info_save.is_open()) {
            kraus_info_save << kraus_info;
            kraus_info_save.close();
            std::cout << "Updated kraus info YAML saved for channel: " << unique_string << std::endl;
        } else {
            std::cerr << "Error: Unable to save kraus info YAML: " << folder + "/info.yml" << std::endl;
        }

    }
    std::cout << "All channels processed for initial vector generation." << std::endl;

    /*
    =================================================
    INITIAL SETUP: REGISTRIES AND POOLS
    =================================================
    
    Create all the necessary, reusable components which are used for 
    minimization orchestration:
        - Load MinimizerConfig from YAML
        - Initialize GPURegistry from config file
        - Create DevicePool from GPURegistry
        - Create work queue and result collector
    */

    // CONFIGURATION LOADING
    entropy::ConfigLoader config_loader;
    entropy::MinimizerConfig min_config = config_loader.loadFromFile(PHASE1_CONFIG_FILE);
    try {
        min_config.validate();
    } catch (const std::invalid_argument& e) {
        std::cerr << "MinimizerConfig validation failed: " << e.what() << std::endl;
        return -1;
    }

    // GPU SETUP
    // Initialize GPURegistry. This tells the minimizer what devices are available.
    entropy::GPURegistry& registry = entropy::GPURegistry::instance();
    try {
        registry.loadConfig(min_config.resource.gpu_config_file);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not load GPU config, using default: " << e.what() << std::endl;
    }

    std::cout << "Initialized GPURegistry with "
        << registry.getAllGPUInfo().size() << " available devices. Of them, "
        << registry.getEnabledGPUs().size() << " are enabled."
        << std::endl;

    std::cout << "Creating DevicePool and initializing from registry..." << std::endl;

    // ORCHESTRATION COMPONENTS
    // Create orchestration components. DevicePool is a pool of compute devices for work distribution, 
    // which are created dynamically based on the available hardware.
    entropy::DevicePool device_pool(min_config.resource);
    device_pool.initializeFromRegistry(registry);

    std::cout << "Initialized DevicePool with "
            << device_pool.numDevices() << " devices ("
            << device_pool.numGPUs() << " GPUs, "
            << device_pool.numCPUs() << " CPUs)."
            << std::endl;

    // PROGRESS TRACKER
    std::cout << "Creating progress tracker..." << std::endl;
    entropy::ProgressTracker progress_tracker;

    // RESULT COLLECTOR
    std::cout << "Creating result collector..." << std::endl;
    entropy::ResultCollector result_collector;

    // AT THIS POINT
    // in order to run some minimization tasks, we need to fill the queue, add the workers, and start processing.





    /*  
    =================================================
    FIRST PASS: LOW ACCURACY RUNS
    =================================================

    For each channel:
        For each initial vector:
            Perform low-accuracy run to minimize output entropy
            Save the obtained vector and output entropy
        Discard a fraction of the least promising vectors based on output entropy
    
    */

    std::cout << "Starting low-accuracy runs." << std::endl;
    std::cout << "Each low-accuracy run will perform " << first_pass_iters << " iterations." << std::endl;
    // Set the number of iterations
    min_config.stopping.max_iterations = first_pass_iters;
    std::vector<entropy::MinimizerConfig> configs = {min_config};

    // Run through every channel
    for (size_t ch_idx = 0; ch_idx < loaded_index["channels"].size(); ++ch_idx) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "[CHANNEL LOOP] Starting iteration " << ch_idx << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        YAML::Node channel_entry = loaded_index["channels"][ch_idx];
        std::string unique_string = channel_entry["unique_string"].as<std::string>();
        std::string kraus_path = channel_entry["kraus_path"].as<std::string>();
        std::string folder = channel_entry["folder"].as<std::string>();

        std::cout << "Processing channel for low-accuracy runs: " << unique_string << std::endl;
        std::cout << "Kraus path: " << kraus_path << std::endl;
        std::cout << "Folder: " << folder << std::endl;

        // Load the kraus operator using vec_serializer
        std::vector<std::complex<double>> kraus_ops;
        int d, N_in, N_out;
        utils::DeserializedData deserialized_data = vec_serializer.deserialize(kraus_path);
        kraus_ops = deserialized_data.vectorData;
        d = deserialized_data.d;
        N_in = deserialized_data.N;
        N_out = deserialized_data.M; // This is not actually needed right now as we are using square kraus operators!
        std::cout << "Kraus operators loaded. d=" << d << ", N_in=" << N_in << ", N_out=" << N_out << std::endl;
        // Create work queue and result collector.
        entropy::ConcurrentQueue<entropy::RunTask> work_queue;
        entropy::ResultCollector result_collector;
        
        std::cout << "Created work queue and result collector." << std::endl;



        // Now fill the work queue with tasks for each initial vector. This is in "info.yml" under "initial_vectors"
        // Load the info YAML
        YAML::Node kraus_info = YAML::LoadFile(folder + "/info.yml");
        YAML::Node initial_vectors = kraus_info["initial_vectors"];
        for (size_t v_idx = 0; v_idx < initial_vectors.size(); ++v_idx) {
            YAML::Node vector_entry = initial_vectors[v_idx];
            std::string vec_unique_string = vector_entry["unique_string"].as<std::string>();

            // Use deserializer to load the initial vector
            std::string vec_filename = folder + "/" + vectors_dirname + "/" + first_pass_dirname + "/initial/" + vec_unique_string + ".dat";
            utils::DeserializedData vec_data = vec_serializer.deserialize(vec_filename);
            std::vector<std::complex<double>> initial_vector = vec_data.vectorData;
            std::cout << "Loaded initial vector: " << vec_unique_string << std::endl;

            // Create a RunTask and push to the work queue
            entropy::RunTask task = entropy::RunTask(
                static_cast<int>(v_idx), // run_id
                0,                      // config_id (single config mode)
                entropy::HostVector(initial_vector) // initial_vector
            );

            work_queue.push(std::move(task));
            std::cout << "Pushed RunTask for vector " << vec_unique_string << " to work queue." << std::endl;
            std::cout << "Work queue size: " << work_queue.size() << " tasks." << std::endl;
        }

        std::cout << "All RunTasks for channel " << unique_string << " pushed to work queue." << std::endl;
        std::cout << "Starting worker processing for low-accuracy runs..." << std::endl;
        // Print the number of tasks in the queue
        std::cout << "Work queue size: " << work_queue.size() << " tasks." << std::endl;
        // Convert kraus_ops to HostKrausOperators

        std::cout << "DEBUG: Size of kraus ops is: " << kraus_ops.size() << std::endl;
        entropy::HostKrausOperators host_kraus = entropy::HostKrausOperators::fromDouble(
            kraus_ops, d, N_in, N_out
        );

        entropy::WorkerThreadPool worker_pool(
            device_pool,
            work_queue,
            result_collector,
            configs,
            host_kraus,
            deserialized_data.N
        );

        worker_pool.setProgressTracker(&progress_tracker);

        // Subscribe to configuration changes (if enabled)
        if (min_config.resource.allow_dynamic_scaling) {
            registry.subscribeToChanges(
                [&device_pool, &worker_pool](const std::vector<int>& enabled_gpus) {
                    try {
                        handleGPUConfigChange(enabled_gpus, device_pool, worker_pool, "first-pass");
                    } catch (const std::exception& e) {
                        std::cerr << "[first-pass] Error handling GPU config change: " << e.what() << std::endl;
                    }
                }
            );
            registry.startWatching(min_config.resource.poll_interval);
        }
        worker_pool.start();
        std::cout << "Worker threads started." << std::endl;

        std::vector<int> vecs_saved_idx;
        std::mutex vecs_saved_mutex;  // Protects vecs_saved_idx
        std::mutex yaml_file_mutex;   // Protects kraus_info and file writes
        // Wait for all tasks to complete
        while (result_collector.numCompleted() + result_collector.numErrors() < 
            static_cast<size_t>(min_config.multi_run.num_attempts)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            std::cout << "Waiting for tasks to complete: "
                    << result_collector.numCompleted() << " completed, "
                    << result_collector.numErrors() << " errors, "
                    << work_queue.size() << " pending."
                    << std::endl;

            // Check progress and print summary
            progress_tracker.printSummary();
            
            // If new results are available, save the vectors
            size_t num_completed = result_collector.numCompleted();
            size_t num_saved;
            {
                std::lock_guard<std::mutex> lock(vecs_saved_mutex);
                num_saved = vecs_saved_idx.size();
            }
            
            if (num_completed > num_saved) {

                std::cout << "I noticed a new run finished!" << std::endl;
                for (const auto& res : result_collector.getAllResults()) {
                    std::cout << "I am looking through all the results..." << std::endl;

                    if (res.error_type != entropy::RunErrorType::NONE) {
                        std::cout << "Run ID " << res.run_id << " encountered an error: "
                                  << static_cast<int>(res.error_type) << std::endl;
                        continue; // Skip saving for errored runs
                    }
                    
                    // Check if this result has already been saved and mark it as being saved (thread-safe)
                    {
                        std::lock_guard<std::mutex> lock(vecs_saved_mutex);
                        if (std::find(vecs_saved_idx.begin(), vecs_saved_idx.end(), res.run_id) != vecs_saved_idx.end()) {
                            continue; // Already saved or being saved
                        }
                        // Mark as being saved immediately to prevent duplicate save threads
                        vecs_saved_idx.push_back(res.run_id);
                    }


                    // Save the resulting vector -> use a new thread!
                    // CRITICAL: Capture by VALUE to prevent use-after-free when loop moves to next channel
                    std::string folder_copy = folder;
                    std::string vectors_dirname_copy = vectors_dirname;
                    std::string first_pass_dirname_copy = first_pass_dirname;
                    std::string initial_vector_str = initial_vectors[res.run_id]["unique_string"].as<std::string>();
                    
                    std::thread save_thread([&vec_serializer, &yaml_file_mutex, folder_copy, vectors_dirname_copy, first_pass_dirname_copy, initial_vector_str, res, d, N_in]() {
                        std::cout << "[SAVE THREAD START] Saving result vector for run ID " << res.run_id << std::endl;
                        std::cout << "[SAVE THREAD] folder = " << folder_copy << std::endl;
                        
                        std::string result_vec_filename = folder_copy + "/" + vectors_dirname_copy + "/" + first_pass_dirname_copy + "/results/" + initial_vector_str + ".dat";
                        std::cout << "[SAVE THREAD] result_vec_filename = " << result_vec_filename << std::endl;
                        
                        std::cout << "[SAVE THREAD] Creating directories..." << std::endl;
                        std::filesystem::create_directories(folder_copy + "/" + vectors_dirname_copy + "/" + first_pass_dirname_copy + "/results/");
                        
                        std::cout << "[SAVE THREAD] Serializing vector..." << std::endl;
                        vec_serializer.serialize(
                            "vector",
                            result_vec_filename,
                            res.final_vector.data,
                            "Resulting vector from low-accuracy run",
                            d,
                            N_in,
                            1
                        );
                        std::cout << "Saved result vector for run ID " << res.run_id << " to file: " << result_vec_filename << std::endl;
                        
                        // Thread-safe: Update YAML
                        {
                            std::cout << "[SAVE THREAD] Acquiring YAML lock..." << std::endl;
                            std::lock_guard<std::mutex> yaml_lock(yaml_file_mutex);
                            std::cout << "[SAVE THREAD] YAML lock acquired" << std::endl;
                            
                            try {
                                // Load the YAML fresh to avoid stale references
                                std::cout << "[SAVE THREAD] Loading YAML from: " << folder_copy + "/info.yml" << std::endl;
                                YAML::Node kraus_info_local = YAML::LoadFile(folder_copy + "/info.yml");
                                
                                // Add the result info to the kraus_info YAML
                                std::cout << "[SAVE THREAD] Creating result_entry..." << std::endl;
                                YAML::Node result_entry;
                                result_entry["initial_vector"] = initial_vector_str;
                                result_entry["result_vector"] = result_vec_filename;
                                result_entry["final_entropy"] = res.final_entropy;
                                result_entry["iterations_taken"] = res.iterations_taken;
                                result_entry["runtime_seconds"] = res.runtime_seconds;
                                std::cout << "[SAVE THREAD] Pushing to kraus_info..." << std::endl;
                                kraus_info_local["first_pass_results"].push_back(result_entry);
                                
                                // save the updated kraus_info YAML
                                std::cout << "[SAVE THREAD] Saving YAML to " << folder_copy + "/info.yml" << std::endl;
                                std::ofstream kraus_info_save(folder_copy + "/info.yml");
                                if (kraus_info_save.is_open()) {
                                    kraus_info_save << kraus_info_local;
                                    kraus_info_save.close();
                                    std::cout << "Updated kraus info YAML saved with first pass results." << std::endl;
                                } else {
                                    std::cerr << "Error: Unable to save kraus info YAML: " << folder_copy + "/info.yml" << std::endl;
                                }
                            } catch (const std::exception& e) {
                                std::cerr << "[SAVE THREAD ERROR] Exception in YAML update: " << e.what() << std::endl;
                            }
                            std::cout << "[SAVE THREAD] YAML lock released" << std::endl;
                        }
                        std::cout << "[SAVE THREAD END] Finished saving run ID " << res.run_id << std::endl;
                    });
                    save_thread.detach();
                }
            }

            // Clean up completed runs from tracker
            for (const auto& completed_run : result_collector.getAllResults()) {
                progress_tracker.removeRun(completed_run.run_id);
            }
            
            // Check if queue is empty and workers are idle
            if (work_queue.empty() && worker_pool.getActiveWorkerCount() == 0) {
                // All tasks done
                break;
            }
        }
        
        // Stop watching for changes
        if (min_config.resource.allow_dynamic_scaling) {
            registry.stopWatching();
        }
        
        // Signal completion and stop workers
        work_queue.signalDone();
        worker_pool.stop();
        worker_pool.waitAll();

        // We are done, process results, save the vectors and entropies
        try {
            entropy::RunResult best = result_collector.getMinimum();
            std::cout << "Best result for channel " << unique_string << ": " 
                << "Final Entropy = " << best.final_entropy << ", "
                << "Iterations = " << best.iterations_taken << ", "
                << "Runtime = " << best.runtime_seconds << " seconds."
                << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << "Error retrieving best result: " << e.what() << std::endl;
        }


        // At this point, select the surviving vectors based on output entropy.
        // Do so by reading the YAML info and sorting the results.
        // IMPORTANT: Reload YAML to get all updates from save threads
        std::cout << "[MAIN] Reloading YAML to read final results..." << std::endl;
        YAML::Node kraus_info_fresh = YAML::LoadFile(folder + "/info.yml");
        
        std::vector<std::pair<size_t, double>> vec_entropy_pairs; // pair of (index, entropy)
        for (size_t r_idx = 0; r_idx < kraus_info_fresh["first_pass_results"].size(); ++r_idx) {
            YAML::Node result_entry = kraus_info_fresh["first_pass_results"][r_idx];
            if (!result_entry["final_entropy"]) {
                std::cerr << "[ERROR] Missing final_entropy for first_pass_results[" << r_idx << "]" << std::endl;
                continue;
            }
            double final_entropy = result_entry["final_entropy"].as<double>();
            vec_entropy_pairs.push_back(std::make_pair(r_idx, final_entropy));
        }

        // Sort by entropy
        std::sort(vec_entropy_pairs.begin(), vec_entropy_pairs.end(),
            [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                return a.second < b.second; // ascending order
            }
        );
        // Determine number of survivors
        size_t num_survivors = static_cast<size_t>(std::ceil(first_pass_vec_survival * vec_entropy_pairs.size()));
        std::cout << "Selecting " << num_survivors << " surviving vectors out of " << vec_entropy_pairs.size() << std::endl;

        std::vector<size_t> surviving_indices;
        for (size_t i = 0; i < num_survivors; ++i) {
            surviving_indices.push_back(vec_entropy_pairs[i].first);
        }

        // Now mark the non-survivors in the kraus_info YAML
        // IMPORTANT: Reload from disk to get latest updates from save threads, and use mutex for thread safety
        {
            std::lock_guard<std::mutex> lock(yaml_file_mutex);
            std::cout << "[MAIN] Reloading YAML to mark discarded vectors..." << std::endl;
            YAML::Node kraus_info_fresh = YAML::LoadFile(folder + "/info.yml");
            
            for (size_t r_idx = 0; r_idx < kraus_info_fresh["first_pass_results"].size(); ++r_idx) {
                if (std::find(surviving_indices.begin(), surviving_indices.end(), r_idx) == surviving_indices.end()) {
                    // Not a survivor
                    kraus_info_fresh["first_pass_results"][r_idx]["discarded"] = true;
                } else {
                    kraus_info_fresh["first_pass_results"][r_idx]["discarded"] = false;
                }
            }

            // Save the updated kraus_info YAML
            std::ofstream kraus_info_save_final(folder + "/info.yml");
            if (kraus_info_save_final.is_open()) {
                kraus_info_save_final << kraus_info_fresh;
                kraus_info_save_final.close();
                std::cout << "Final kraus info YAML saved with discarded vector markings." << std::endl;
            } else {
                std::cerr << "Error: Unable to save kraus info YAML: " << folder + "/info.yml" << std::endl;
            }
        }
        std::cout << "Marked discarded vectors in kraus_info YAML." << std::endl;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "[CHANNEL LOOP] Finished iteration " << ch_idx << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    // Reset result collector for next pass
    result_collector.clear();

    /*  
    =================================================
    SECOND PASS: MEDIUM ACCURACY RUNS
    =================================================

    For each channel:
        For each surviving vector:
            Perform medium-accuracy run to minimize output entropy
            Save the obtained vector and output entropy
        Discard a fraction of the least promising vectors based on output entropy
    Discard a fraction of the least promising channels based on best output entropy (delta)
    
    */

    int second_pass_iters = config["config"]["params"]["second-pass"]["iters"].as<int>();
    double second_pass_vec_survival = config["config"]["params"]["second-pass"]["vec-survival"].as<double>();
    std::cout << "Starting medium-accuracy runs." << std::endl;
    std::cout << "Each medium-accuracy run will perform " << second_pass_iters << " iterations." << std::endl;
    // Set the number of iterations
    min_config.stopping.max_iterations = second_pass_iters;
    configs = {min_config};

    // Run through every channel
    for (size_t ch_idx = 0; ch_idx < loaded_index["channels"].size(); ++ch_idx) {
        YAML::Node channel_entry = loaded_index["channels"][ch_idx];
        std::string unique_string = channel_entry["unique_string"].as<std::string>();
        std::string kraus_path = channel_entry["kraus_path"].as<std::string>();
        std::string folder = channel_entry["folder"].as<std::string>();

        std::cout << "Processing channel for medium-accuracy runs: " << unique_string << std::endl;
        std::cout << "Kraus path: " << kraus_path << std::endl;
        std::cout << "Folder: " << folder << std::endl;

        // Load the kraus operator using vec_serializer
        std::vector<std::complex<double>> kraus_ops;
        int d, N_in, N_out;
        utils::DeserializedData deserialized_data = vec_serializer.deserialize(kraus_path);
        kraus_ops = deserialized_data.vectorData;
        d = deserialized_data.d;
        N_in = deserialized_data.N;
        N_out = deserialized_data.M; // This is not actually needed right now as we are using square kraus operators!
        std::cout << "Kraus operators loaded. d=" << d << ", N_in=" << N_in << ", N_out=" << N_out << std::endl;
        // Create work queue
        entropy::ConcurrentQueue<entropy::RunTask> work_queue;
        
        std::cout << "Created work queue and result collector." << std::endl;



        // Now fill the work queue with tasks for each initial vector. This is in "info.yml" under "initial_vectors"
        // Load the info YAML
        YAML::Node kraus_info = YAML::LoadFile(folder + "/info.yml");
        YAML::Node first_pass_vectors = kraus_info["first_pass_results"];
        for (size_t v_idx = 0; v_idx < first_pass_vectors.size(); ++v_idx) {
            YAML::Node vector_entry = first_pass_vectors[v_idx];
            if (vector_entry["discarded"] && vector_entry["discarded"].as<bool>() == true) {
                continue;
            }
            // Ensure the initial and result folders exist for second pass
            std::filesystem::create_directories(folder + "/" + vectors_dirname + "/" + second_pass_dirname + "/initial/");
            std::filesystem::create_directories(folder + "/" + vectors_dirname + "/" + second_pass_dirname + "/results/");

            // Copy the vectors which survived the first pass to the second_pass folder
            std::filesystem::copy(
                folder + "/" + vectors_dirname + "/" + first_pass_dirname + "/results/" + vector_entry["initial_vector"].as<std::string>() + ".dat",
                folder + "/" + vectors_dirname + "/" + second_pass_dirname + "/initial/" + vector_entry["initial_vector"].as<std::string>() + ".dat",
                std::filesystem::copy_options::update_existing
            );
            // Add the corresponsing entry to the second pass results
            if (!kraus_info["second_pass_results"]) {
                kraus_info["second_pass_results"] = YAML::Node(YAML::NodeType::Sequence);
            }
            YAML::Node second_pass_entry;
            second_pass_entry["initial_vector"] = vector_entry["initial_vector"].as<std::string>();
            kraus_info["second_pass_results"].push_back(second_pass_entry);
            // Save the updated kraus_info YAML
            std::ofstream kraus_info_save(folder + "/info.yml");
            if (kraus_info_save.is_open()) {
                kraus_info_save << kraus_info;
                kraus_info_save.close();
                std::cout << "Updated kraus info YAML saved for channel: " << unique_string << std::endl;
            } else {
                std::cerr << "Error: Unable to save kraus info YAML: " << folder + "/info.yml" << std::endl;
            }
            
        }
        // Now go through the second run results to get the vectors
        for (size_t v_idx = 0; v_idx < kraus_info["second_pass_results"].size(); ++v_idx) {
            YAML::Node vector_entry = kraus_info["second_pass_results"][v_idx];
            // Proceed with surviving vectors
            std::string vec_unique_string = vector_entry["initial_vector"].as<std::string>();

            // Use deserializer to load the initial vector
            std::string vec_filename = folder + "/" + vectors_dirname + "/" + second_pass_dirname + "/initial/" + vec_unique_string + ".dat";
            utils::DeserializedData vec_data = vec_serializer.deserialize(vec_filename);
            std::vector<std::complex<double>> initial_vector = vec_data.vectorData;
            std::cout << "Loaded vector from first pass: " << vec_unique_string << std::endl;

            // Create a RunTask and push to the work queue
            entropy::RunTask task = entropy::RunTask(
                static_cast<int>(v_idx), // run_id
                0,                      // config_id (single config mode)
                entropy::HostVector(initial_vector) // initial_vector
            );

            work_queue.push(std::move(task));
            std::cout << "Pushed RunTask for vector " << vec_unique_string << " to work queue." << std::endl;
            std::cout << "Work queue size: " << work_queue.size() << " tasks." << std::endl;
        }

        std::cout << "All RunTasks for channel " << unique_string << " pushed to work queue." << std::endl;
        std::cout << "Starting worker processing for medium-accuracy runs..." << std::endl;
        // Print the number of tasks in the queue
        std::cout << "Work queue size: " << work_queue.size() << " tasks." << std::endl;
        // Convert kraus_ops to HostKrausOperators
        entropy::HostKrausOperators host_kraus = entropy::HostKrausOperators::fromDouble(
            kraus_ops, d, N_in, N_out
        );

        entropy::WorkerThreadPool worker_pool(
            device_pool,
            work_queue,
            result_collector,
            configs,
            host_kraus,
            deserialized_data.N
        );

        worker_pool.setProgressTracker(&progress_tracker);

        // Subscribe to configuration changes (if enabled)
        if (min_config.resource.allow_dynamic_scaling) {
            registry.subscribeToChanges(
                [&device_pool, &worker_pool](const std::vector<int>& enabled_gpus) {
                    try {
                        handleGPUConfigChange(enabled_gpus, device_pool, worker_pool, "first-pass");
                    } catch (const std::exception& e) {
                        std::cerr << "[first-pass] Error handling GPU config change: " << e.what() << std::endl;
                    }
                }
            );
            registry.startWatching(min_config.resource.poll_interval);
        }
        worker_pool.start();
        std::cout << "Worker threads started." << std::endl;

        std::vector<int> vecs_saved_idx;
        std::mutex vecs_saved_mutex;  // Protects vecs_saved_idx
        std::mutex yaml_file_mutex;   // Protects kraus_info and file writes
        // Wait for all tasks to complete
        while (result_collector.numCompleted() + result_collector.numErrors() < 
            static_cast<size_t>(min_config.multi_run.num_attempts)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            std::cout << "Waiting for tasks to complete: "
                    << result_collector.numCompleted() << " completed, "
                    << result_collector.numErrors() << " errors, "
                    << work_queue.size() << " pending."
                    << std::endl;

            // Check progress and print summary
            progress_tracker.printSummary();
            
            // If new results are available, save the vectors
            size_t num_completed = result_collector.numCompleted();
            size_t num_saved;
            {
                std::lock_guard<std::mutex> lock(vecs_saved_mutex);
                num_saved = vecs_saved_idx.size();
            }
            
            if (num_completed > num_saved) {

                std::cout << "I noticed a new run finished!" << std::endl;
                for (const auto& res : result_collector.getAllResults()) {
                    std::cout << "I am looking through all the results..." << std::endl;

                    if (res.error_type != entropy::RunErrorType::NONE) {
                        std::cout << "Run ID " << res.run_id << " encountered an error: "
                                  << static_cast<int>(res.error_type) << std::endl;
                        continue; // Skip saving for errored runs
                    }
                    
                    // Check if this result has already been saved and mark it as being saved (thread-safe)
                    {
                        std::lock_guard<std::mutex> lock(vecs_saved_mutex);
                        if (std::find(vecs_saved_idx.begin(), vecs_saved_idx.end(), res.run_id) != vecs_saved_idx.end()) {
                            continue; // Already saved or being saved
                        }
                        // Mark as being saved immediately to prevent duplicate save threads
                        vecs_saved_idx.push_back(res.run_id);
                    }


                    // Save the resulting vector -> use a new thread!
                    // Capture by value to prevent use-after-free when loop moves to next channel
                    std::string folder_copy = folder;
                    std::string vectors_dirname_copy = vectors_dirname;
                    std::string second_pass_dirname_copy = second_pass_dirname;
                    std::string initial_vector_str = kraus_info["second_pass_results"][res.run_id]["initial_vector"].as<std::string>();
                    
                    std::cout << "[DEBUG] Spawning save thread for run_id=" << res.run_id << ", folder=" << folder_copy << std::endl;
                    
                    std::thread save_thread([&vec_serializer, &yaml_file_mutex, folder_copy, vectors_dirname_copy, second_pass_dirname_copy, initial_vector_str, res, d, N_in]() {
                        std::cout << "[DEBUG THREAD] Starting save for run ID " << res.run_id << ", folder=" << folder_copy << std::endl;
                        std::string result_vec_filename = folder_copy + "/" + vectors_dirname_copy + "/" + second_pass_dirname_copy + "/results/" + initial_vector_str + ".dat";
                        std::cout << "[DEBUG THREAD] Creating directory: " << folder_copy + "/" + vectors_dirname_copy + "/" + second_pass_dirname_copy + "/results/" << std::endl;
                        std::filesystem::create_directories(folder_copy + "/" + vectors_dirname_copy + "/" + second_pass_dirname_copy + "/results/");
                        std::cout << "[DEBUG THREAD] Serializing vector for run ID " << res.run_id << std::endl;
                        vec_serializer.serialize(
                            "vector",
                            result_vec_filename,
                            res.final_vector.data,
                            "Resulting vector from medium-accuracy run",
                            d,
                            N_in,
                            1
                        );
                        std::cout << "Saved result vector for run ID " << res.run_id << " to file: " << result_vec_filename << std::endl;
                        
                        // Thread-safe: Update YAML
                        {
                            std::cout << "[DEBUG THREAD] Acquiring YAML lock for run ID " << res.run_id << std::endl;
                            std::lock_guard<std::mutex> yaml_lock(yaml_file_mutex);
                            
                            // Load the YAML fresh to avoid stale references
                            std::cout << "[DEBUG THREAD] Loading YAML from: " << folder_copy + "/info.yml" << std::endl;
                            YAML::Node kraus_info_local = YAML::LoadFile(folder_copy + "/info.yml");
                            
                            // Edit the result info to the kraus_info YAML
                            YAML::Node result_entry = kraus_info_local["second_pass_results"][res.run_id];
                            result_entry["result_vector"] = result_vec_filename;
                            result_entry["final_entropy"] = res.final_entropy;
                            result_entry["iterations_taken"] = res.iterations_taken;
                            result_entry["runtime_seconds"] = res.runtime_seconds;

                            // save the updated kraus_info YAML
                            std::cout << "[DEBUG THREAD] Writing YAML to: " << folder_copy + "/info.yml" << std::endl;
                            std::ofstream kraus_info_save(folder_copy + "/info.yml");
                            if (kraus_info_save.is_open()) {
                                kraus_info_save << kraus_info_local;
                                kraus_info_save.close();
                                std::cout << "Updated kraus info YAML saved with second pass results." << std::endl;
                            } else {
                                std::cerr << "Error: Unable to save kraus info YAML: " << folder_copy + "/info.yml" << std::endl;
                            }
                            std::cout << "[DEBUG THREAD] Finished save thread for run ID " << res.run_id << std::endl;
                        }
                    });
                    save_thread.detach();
                }
            }

            // Clean up completed runs from tracker
            for (const auto& completed_run : result_collector.getAllResults()) {
                progress_tracker.removeRun(completed_run.run_id);
            }
            
            // Check if queue is empty and workers are idle
            if (work_queue.empty() && worker_pool.getActiveWorkerCount() == 0) {
                // All tasks done
                break;
            }
        }
        
        // Stop watching for changes
        if (min_config.resource.allow_dynamic_scaling) {
            registry.stopWatching();
        }
        
        // Signal completion and stop workers
        work_queue.signalDone();
        worker_pool.stop();
        worker_pool.waitAll();

        // We are done, process results, save the vectors and entropies
        try {
            entropy::RunResult best = result_collector.getMinimum();
            std::cout << "Best result for channel " << unique_string << ": " 
                << "Final Entropy = " << best.final_entropy << ", "
                << "Iterations = " << best.iterations_taken << ", "
                << "Runtime = " << best.runtime_seconds << " seconds."
                << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << "Error retrieving best result: " << e.what() << std::endl;
        }


        // At this point, select the surviving vectors based on output entropy.
        // Do so by reading the YAML info and sorting the results.
        // IMPORTANT: Reload YAML to get all updates from save threads
        std::cout << "[MAIN] Reloading YAML to read final results..." << std::endl;
        YAML::Node kraus_info_fresh = YAML::LoadFile(folder + "/info.yml");
        
        std::vector<std::pair<size_t, double>> vec_entropy_pairs; // pair of (index, entropy)
        for (size_t r_idx = 0; r_idx < kraus_info_fresh["second_pass_results"].size(); ++r_idx) {
            YAML::Node result_entry = kraus_info_fresh["second_pass_results"][r_idx];
            if (!result_entry["final_entropy"]) {
                std::cerr << "[ERROR] Missing final_entropy for second_pass_results[" << r_idx << "]" << std::endl;
                continue;
            }
            double final_entropy = result_entry["final_entropy"].as<double>();
            vec_entropy_pairs.push_back(std::make_pair(r_idx, final_entropy));
        }

        // Sort by entropy
        std::sort(vec_entropy_pairs.begin(), vec_entropy_pairs.end(),
            [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                return a.second < b.second; // ascending order
            }
        );
        // Determine number of survivors
        size_t num_survivors = static_cast<size_t>(std::ceil(second_pass_vec_survival * vec_entropy_pairs.size()));
        std::cout << "Selecting " << num_survivors << " surviving vectors out of " << vec_entropy_pairs.size() << std::endl;

        std::vector<size_t> surviving_indices;
        for (size_t i = 0; i < num_survivors; ++i) {
            surviving_indices.push_back(vec_entropy_pairs[i].first);
        }

        // Now mark the non-survivors in the kraus_info YAML
        // IMPORTANT: Reload from disk to get latest updates from save threads, and use mutex for thread safety
        {
            std::lock_guard<std::mutex> lock(yaml_file_mutex);
            std::cout << "[MAIN] Reloading YAML to mark discarded vectors..." << std::endl;
            YAML::Node kraus_info_fresh = YAML::LoadFile(folder + "/info.yml");
            
            for (size_t r_idx = 0; r_idx < kraus_info_fresh["second_pass_results"].size(); ++r_idx) {
                if (std::find(surviving_indices.begin(), surviving_indices.end(), r_idx) == surviving_indices.end()) {
                    // Not a survivor
                    kraus_info_fresh["second_pass_results"][r_idx]["discarded"] = true;
                } else {
                    kraus_info_fresh["second_pass_results"][r_idx]["discarded"] = false;
                }
            }

            // Save the updated kraus_info YAML
            std::ofstream kraus_info_save_final(folder + "/info.yml");
            if (kraus_info_save_final.is_open()) {
                kraus_info_save_final << kraus_info_fresh;
                kraus_info_save_final.close();
                std::cout << "Final kraus info YAML saved with discarded vector markings." << std::endl;
            } else {
                std::cerr << "Error: Unable to save kraus info YAML: " << folder + "/info.yml" << std::endl;
            }
        }
        std::cout << "Marked discarded vectors in kraus_info YAML." << std::endl;
    }

    // Now find the channels to keep and the ones to discard.
    // Keep the ones with the smallest delta between (calculated tensor entropy) and 2 *(lowest output entropy so far).
    // (remember that MOE is not additive if we find a tensor vector with lower entropy than the sum of individual ones)
    // (i.e. if the delta is negative, it means we found violation)
    std::cout << "Evaluating channels for survival based on second pass results." << std::endl;
    std::vector<std::pair<size_t, double>> channel_delta_pairs; // pair of (index, delta)
    for (size_t ch_idx = 0; ch_idx < loaded_index["channels"].size(); ++ch_idx) {
        YAML::Node channel_entry = loaded_index["channels"][ch_idx];
        std::string folder = channel_entry["folder"].as<std::string>();

        // Load the info YAML
        YAML::Node kraus_info = YAML::LoadFile(folder + "/info.yml");
        YAML::Node second_pass_results = kraus_info["second_pass_results"];

        double min_entropy = std::numeric_limits<double>::max();
        for (size_t r_idx = 0; r_idx < second_pass_results.size(); ++r_idx) {
            YAML::Node result_entry = second_pass_results[r_idx];
            if (result_entry["discarded"] && result_entry["discarded"].as<bool>() == true) {
                continue;
            }
            double final_entropy = result_entry["final_entropy"].as<double>();
            if (final_entropy < min_entropy) {
                min_entropy = final_entropy;
            }
        }
        double delta = kraus_info["analysis"]["tensor_entropy_estimation"]["estimated_entropy"].as<double>() - 2.0 * min_entropy; 
        channel_delta_pairs.push_back(std::make_pair(ch_idx, delta));
    }
    // Sort by delta
    std::sort(channel_delta_pairs.begin(), channel_delta_pairs.end(),
        [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
            return a.second < b.second; // ascending order
        }
    );
    // Determine number of surviving channels
    double channel_survival_fraction = config["config"]["params"]["second-pass"]["channel-survival"].as<double>();
    size_t num_channel_survivors = static_cast<size_t>(std::ceil(channel_survival_fraction * channel_delta_pairs.size()));
    std::cout << "Selecting " << num_channel_survivors << " surviving channels out of " << channel_delta_pairs.size() << std::endl;

    std::vector<size_t> surviving_channel_indices;
    for (size_t i = 0; i < num_channel_survivors; ++i) {
        surviving_channel_indices.push_back(channel_delta_pairs[i].first);
    }

    // Now update the index yml to mark discarded channels
    for (size_t ch_idx = 0; ch_idx < loaded_index["channels"].size(); ++ch_idx) {
        if (std::find(surviving_channel_indices.begin(), surviving_channel_indices.end(), ch_idx) == surviving_channel_indices.end()) {
            // Not a survivor
            loaded_index["channels"][ch_idx]["second_pass_results"]["discarded"] = true;
            loaded_index["channels"][ch_idx]["second_pass_results"]["delta"] = channel_delta_pairs[ch_idx].second;
        } else {
            loaded_index["channels"][ch_idx]["second_pass_results"]["discarded"] = false;
            loaded_index["channels"][ch_idx]["second_pass_results"]["delta"] = channel_delta_pairs[ch_idx].second;
        }
    }
    // Save the updated index YAML
    std::ofstream index_save(index_path);
    if (index_save.is_open()) {
        index_save << loaded_index;
        index_save.close();
        std::cout << "Final index YAML saved with discarded channel markings." << std::endl;
    } else {
        std::cerr << "Error: Unable to save index YAML: " << index_path << std::endl;
    }
    return 0;
}