

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <filesystem>

#include "channel/generator/random_generator.h"
#include "utilities/messaging/message_handler.h"
#include "yaml-cpp/yaml.h"

#include "minimizer/analysis/tensor_entropy.h"

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

int main(int argc, char** argv){


    // LOAD PARAMETERS FROM YAML CONFIG
    YAML::Node config = YAML::LoadFile(PHASE1_CONFIG_FILE);

    std::vector<int> channel_ns = config["config"]["params"]["N"].as<std::vector<int>>();
    std::vector<int> channel_ds = config["config"]["params"]["d"].as<std::vector<int>>();
    std::string out_folder = config["config"]["output"]["output-directory"].as<std::string>();

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
    MessageHandler* msg_handler = new MessageHandler();
    msg_handler->createPrinter();

    // Create TensorEntropyEstimator to get the tensor entropy of the channels
    TensorEntropyEstimator<double>* tensor_estimator = new TensorEntropyEstimator<double>();

    // Find the memory needed on device for the kraus operators, allocate
    int device_kraus_size = 0;
    for (const auto& nd: nd_pairs){
        if (nd.second * nd.first * nd.first > device_kraus_size){
            device_kraus_size = nd.second * nd.first * nd.first;
        }
    }
    std::cout << "Allocating space (on host) for kraus operators. Allocating max size: " << device_kraus_size << std::endl;
    std::vector<std::complex<double>>* h_kraus = new std::vector<std::complex<double> >(device_kraus_size);


    // For each (N,d) pair
    for (const auto& nd : nd_pairs){
        int N = nd.first;
        int d = nd.second;
        std::cout << "Preparing to generate channels for N=" << N << ", d=" << d << std::endl;
        RandomGeneratorConfig rg_config;
        rg_config.kraus_number = d;
        rg_config.kraus_in_dimension = N;
        rg_config.kraus_out_dimension = N;
        rg_config.message_handler = msg_handler;

        // For each channel
        for (int ch=0; ch<num_channels; ch++){
            std::cout << "Generating channel " << ch+1 << " of " << num_channels << std::endl;
            RandomGenerator rg(rg_config);
            
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
            rg.generate(h_kraus);

            // Save kraus to file (filename=kraus.dat)
            std::string kraus_filename = save_folder + "/kraus.dat";
            std::cout << "Saving kraus operators to file: " << kraus_filename << std::endl;
            std::ofstream kraus_file(kraus_filename,  std::ios::binary);
            if (kraus_file.is_open()){
                int kraus_size = rg_config.kraus_number * rg_config.kraus_in_dimension * rg_config.kraus_out_dimension;
                kraus_file.write(reinterpret_cast<const char*>(h_kraus->data()), sizeof(std::complex<double>) * kraus_size);
                kraus_file.close();
            } else {
                std::cerr << "Error: Unable to open file for writing kraus operators: " << kraus_filename << std::endl;
            }

            // Compute the tensor product MOE estimate
            tensor_estimator->computeEntropy(reinterpret_cast<cuDoubleComplex*>(h_kraus->data()), rg_config.kraus_number, rg_config.kraus_in_dimension, rg_config.kraus_out_dimension, tensor_eps);


            msg_handler->message("Entropy computed to: " + std::to_string(tensor_estimator->getEstimatedEntropy()) + " +/- " + std::to_string(tensor_estimator->getEstimatedError()));
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
            msg_handler->message("Theoretical entropy (tensor channel): " + std::to_string(theoretical_entropy));
        }
    }

    delete msg_handler;
    delete tensor_estimator;
    delete h_kraus;
    return 0;
}