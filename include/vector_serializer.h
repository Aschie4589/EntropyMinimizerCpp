#ifndef VECTOR_SERIALIZER_H
#define VECTOR_SERIALIZER_H

#include "common_includes.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// Data structure for deserialized data. Update if metadata changes.
struct DeserializedData {
    std::string type;          // Metadata: type ("vector" or "kraus")
    std::vector<std::complex<double>> vectorData;
    int d;                     // Metadata: d
    int N;                     // Metadata: N
    std::string description;   // Metadata: description
    json metadata;   // Full metadata as JSON
};

class VectorSerializer
{
    /*
    FILE FORMAT for serialized vector or Kraus operator

    +------------------+
    | Magic identifier |  (always "VECTR" or "KRAUS") 5 characters
    +------------------+
    | Format Version   |  (e.g., 1.0 - Variable length string)
    +------------------+
    | Metadata Size    |  (size in bytes of metadata)
    +------------------+
    | Metadata         |  (JSON string with metadata)
    +------------------+
    | Vector Size      |  (number of elements in vector)
    +------------------+
    | Vector Data      |  (binary data of vector)
    +------------------+
    | Footer (Optional)|  (checksum)
    +------------------+

    */
public:
    VectorSerializer(/* args */);

    ~VectorSerializer();

    // Serialize the vector to a file
    static void serialize(const std::string& type, const std::string& fileName, const std::vector<std::complex<double>>& vec, 
                          const std::string& description, int d, int N);
    // Deserialize the vector from a file
    DeserializedData deserialize(const std::string& fileName);
private:
    // Calculate checksum for a byte buffer
    static uint32_t calculateChecksum(const std::vector<uint8_t>& buffer);


};



#endif