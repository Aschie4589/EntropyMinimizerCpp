#ifndef VECTOR_SERIALIZER_H
#define VECTOR_SERIALIZER_H

#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <fstream>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace entropy {

// Data structure for deserialized data. Update if metadata changes.
struct DeserializedData {
    std::string type;          // "vector" (quantum state) or "kraus" (channel operators)
    std::vector<std::complex<double>> vectorData;  // Flattened complex data
    
    // Metadata interpretation depends on type:
    // For type="vector" (quantum state checkpoint):
    //   d = Hilbert space dimension (number of basis states)
    //   N = 1 (not used for single vectors)
    //   M = 0 (not used for single vectors)
    //
    // For type="kraus" (quantum channel):
    //   d = number of Kraus operators
    //   N = input dimension (columns of each K_i)
    //   M = output dimension (rows of each K_i)
    //   vectorData contains all operators: [K_0, K_1, ..., K_{d-1}]
    //   Each K_i stored column-major, total size = d * M * N
    int d;                     
    int N;                     
    int M;                     
    
    std::string description;   // Human-readable description
    json metadata;             // Full metadata as JSON (for extensibility)
};

class VectorSerializer
{
    /*
    FILE FORMAT SPECIFICATION:
    
    Binary file structure for quantum states and Kraus operators:
    +----------------------+
    | Magic (5 bytes)      | "VECTR" or "KRAUS"
    +----------------------+
    | Version Size (u32)   | Length of version string
    +----------------------+
    | Version String       | e.g., "1.0"
    +----------------------+
    | Metadata Size (u32)  | Length of JSON metadata
    +----------------------+
    | Metadata (JSON)      | {description, d, N, M, ...}
    +----------------------+
    | Vector Size (u32)    | Number of complex elements
    +----------------------+
    | Complex Data         | Binary doubles (real, imag pairs)
    +----------------------+
    | CRC32 Checksum (u32) | CRC32 of metadata + data
    +----------------------+

    PERFORMANCE NOTES:
    - CRC32 computed incrementally (single pass, no buffer rebuild)
    - For large Kraus operators, this avoids O(n²) behavior
    - Uses standard CRC32 polynomial (0xEDB88320)
    
    USAGE EXAMPLES:
    
    1. Save quantum state vector:
       VectorSerializer::serialize("vector", "state.dat", psi_vector,
                                   "Ground state", hilbert_dim, 1, 0);
    
    2. Save Kraus operators for channel:
       VectorSerializer::serialize("kraus", "channel.dat", flattened_kraus,
                                   "Depolarizing channel", num_kraus, input_dim, output_dim);
    */
public:
    VectorSerializer(/* args */);

    ~VectorSerializer();

    // Serialize the vector to a file
    // For type="vector": d=hilbert_dim, N=1, M=0
    // For type="kraus": d=num_operators, N=input_dim, M=output_dim
    static void serialize(const std::string& type, const std::string& fileName, const std::vector<std::complex<double>>& vec, 
                          const std::string& description, int d, int N, int M = 0);
    
    // Deserialize the vector from a file
    // Returns DeserializedData with type-specific metadata interpretation
    DeserializedData deserialize(const std::string& fileName);
    
private:
    // Calculate CRC32 checksum (standard polynomial 0xEDB88320)
    static uint32_t calculateChecksum(const std::vector<uint8_t>& buffer);


};

} // namespace entropy

#endif