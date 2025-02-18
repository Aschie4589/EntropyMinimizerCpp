#include "common_includes.h"
#include "generate_random_vector.h"

std::vector<std::complex<double> >* generateUniformRandomVector(int N){
    // Step 0: Initialize output
    std::vector<std::complex<double> >* out = new std::vector<std::complex<double> >(N);
    // Step 1: Set up random number generator
    std::random_device rd;                  // Random device to seed the generator
    std::mt19937 gen(rd());                 // Mersenne Twister generator, seeded by rd()
    std::normal_distribution<double> dist(0.0f, 1.0f); // Normal distribution with mean and stddev

    // Step 2: Generate random real and imaginary parts
    for (int i=0; i < N; i++){
        double real_part = dist(gen);  // Generate the real part (normal distribution)
        double imag_part = dist(gen);  // Generate the imaginary part (normal distribution)
        out->at(i) = std::complex<double>(real_part, imag_part);
    }

    // Step 3: Renormalize the vector
    double norm = 0;
    for (int i=0; i < N; i++){
        norm += std::pow(std::abs(out->at(i)),2);
    }
    norm = std::sqrt(norm);
    for (int i=0; i < N; i++){
        out->at(i) /= norm;
    }
    // Return the pointer    
    return out;
}
