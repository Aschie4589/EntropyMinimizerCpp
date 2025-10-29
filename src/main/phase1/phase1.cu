

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

#define OUT_FOLDER "runs/phase1"
#define NUM_CHANNELS 1             // Number of channels to generate and test for each (N,d) pair
#define ITER_LOW 1000               // Number of iterations for low-accuracy phase 1 runs
#define NUM_VECS 50                 // Number of vectors to generate and test
#define VEC_DISCARD 0.95            // Fraction of each vector to discard after low accuracy phase 1 run
#define ITER_MEDIUM 10000           // Number of iterations for medium-accuracy phase 1 runs
#define CHANNEL_DISCARD 0.8         // Fraction of channels to discard after medium accuracy phase 1 run

#define CHANNEL_N {512} //{256, 512, 1024, 2048}
#define CHANNEL_D {20, 24, 28, 32}

int main(int argc, char** argv){

    int channel_ns[] = CHANNEL_N;
    int channel_ds[] = CHANNEL_D;



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

    // Find the memory needed on device for the kraus operators, allocate
    int device_kraus_size = 0;
    for (const auto& nd: nd_pairs){
        if (nd.second * nd.first * nd.first > device_kraus_size){
            device_kraus_size = nd.second * nd.first * nd.first;
        }
    }
    std::cout << "Allocating space for kraus operators of size: " << device_kraus_size << std::endl;
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
        for (int ch=0; ch<NUM_CHANNELS; ch++){
            std::cout << "Generating channel " << ch+1 << " of " << NUM_CHANNELS << std::endl;
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
            std::string save_folder = std::string(OUT_FOLDER) + "/" + unique_string;
            std::cout << "Creating save folder: " << save_folder << std::endl;
            std::filesystem::create_directories(save_folder);

            // Generate kraus
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

        }
    }

    delete msg_handler;
    return 0;
}