#include "utilities/serializer/serializer.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace utils {

VectorSerializer::VectorSerializer(/* args */)
{
}


VectorSerializer::~VectorSerializer()
{
}


void VectorSerializer::serialize(const std::string& type, const std::string& fileName, const std::vector<std::complex<double>>& vec, 
                                  const std::string& description, int d, int N, int M) {
    std::ofstream outFile(fileName, std::ios::binary);
    /*
    Serialize a vector of std::complex<double> to a file.
    
    FILE FORMAT:
    +------------------+
    | Magic identifier |  (always "VECTR" or "KRAUS") 5 characters
    +------------------+
    | Format Version   |  (e.g., 1.0 - Variable length string)
    +------------------+
    | Metadata Size    |  (size in bytes of metadata JSON)
    +------------------+
    | Metadata (JSON)  |  Contains: description, d, N, M
    +------------------+
    | Vector Size      |  (number of complex elements)
    +------------------+
    | Vector Data      |  (binary: real, imag pairs as doubles)
    +------------------+
    | Checksum (CRC32) |  (uint32_t CRC32 of metadata + data)
    +------------------+

    METADATA FIELDS (stored as JSON):
    For quantum state vectors (type=\"vector\"):
      - d: Dimension of the Hilbert space (number of basis states)
      - N: Not used for single vectors (set to 1)
      - M: Not used for single vectors (set to 0)
    
    For Kraus operators (type=\"kraus\"):
      - d: Number of Kraus operators in the channel
      - N: Input dimension of each operator (columns)
      - M: Output dimension of each operator (rows)
      - vec contains all operators flattened: [K_0, K_1, ..., K_{d-1}]
        Each K_i is stored in column-major order
        Total size: d * M * N complex numbers

    PERFORMANCE:
      - Checksum uses CRC32 (standard polynomial 0xEDB88320)
      - Computed incrementally during write (single pass)
      - No buffer rebuilding required

    Arguments:
    - type: \"vector\" (quantum state) or \"kraus\" (quantum channel operators)
    - fileName: Output file path
    - vec: Flattened complex data
    - description: Human-readable description
    - d: See metadata fields above
    - N: See metadata fields above  
    - M: See metadata fields above
    */
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    // Magic identifier
    if (type == "vector") {
        const std::string magic = "VECTR";
        outFile.write(magic.c_str(), magic.size());
    } else if (type == "kraus") {
        const std::string magic = "KRAUS";
        outFile.write(magic.c_str(), magic.size());
    } else {
        throw std::runtime_error("Invalid type for serialization.");
    }

    // Format version
    const std::string version = "1.0";
    uint32_t versionSize = version.size();
    outFile.write(reinterpret_cast<const char*>(&versionSize), sizeof(versionSize)); // Write the size of the version string
    outFile.write(version.c_str(), versionSize); // Write the version string

    // Metadata
    json metadata = {
        {"metadata_version", "1.0"},
        {"description", description},
        {"d", d},
        {"N", N},
        {"M", M}
    };
    std::string metadataStr = metadata.dump();
    uint32_t metadataSize = metadataStr.size();
    outFile.write(reinterpret_cast<const char*>(&metadataSize), sizeof(metadataSize)); // Write the size of the metadata string
    outFile.write(metadataStr.c_str(), metadataSize); // Write the metadata string

    // Vector size
    uint32_t vectorSize = vec.size(); 
    outFile.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize)); // Write the size of the vector

    // Vector data - compute checksum incrementally during write
    std::vector<uint8_t> checksumBuffer;
    
    // Add metadata to checksum buffer
    checksumBuffer.insert(checksumBuffer.end(), 
                         metadataStr.begin(), metadataStr.end());
    
    // Write vector data and build checksum buffer
    for (const auto& elem : vec) {
        double real = elem.real();
        double imag = elem.imag();
        outFile.write(reinterpret_cast<const char*>(&real), sizeof(real));
        outFile.write(reinterpret_cast<const char*>(&imag), sizeof(imag));
        
        // Add to checksum buffer (more efficient than stringstream)
        const uint8_t* realBytes = reinterpret_cast<const uint8_t*>(&real);
        const uint8_t* imagBytes = reinterpret_cast<const uint8_t*>(&imag);
        checksumBuffer.insert(checksumBuffer.end(), realBytes, realBytes + sizeof(real));
        checksumBuffer.insert(checksumBuffer.end(), imagBytes, imagBytes + sizeof(imag));
    }

    // Footer: Calculate checksum (now only one pass through data)
    uint32_t checksum = calculateChecksum(checksumBuffer);
    outFile.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum)); // Write the checksum

    outFile.close();
}

DeserializedData VectorSerializer::deserialize(const std::string& fileName) {
    std::ifstream inFile(fileName, std::ios::binary);
    if (!inFile.is_open()) {
        throw std::runtime_error("Failed to open file for reading.");
    }

    // Read magic identifier
    char magic[5];
    inFile.read(magic, 5);
    // Check magic identifier
    if (std::strncmp(magic, "VECTR", 5) != 0 && std::strncmp(magic, "KRAUS", 5) != 0) {
        throw std::runtime_error("Invalid magic identifier.");
    }

    // Read format version
    uint32_t versionSize;
    inFile.read(reinterpret_cast<char*>(&versionSize), sizeof(versionSize));
    std::string version(versionSize, '\0');
    inFile.read(&version[0], versionSize);

    // Read metadata
    uint32_t metadataSize;
    inFile.read(reinterpret_cast<char*>(&metadataSize), sizeof(metadataSize));
    std::string metadataStr(metadataSize, '\0');
    inFile.read(&metadataStr[0], metadataSize);
    json metadata = json::parse(metadataStr);

    // Extract metadata to DeserializedData struct
    std::string description;
    int d, N, M;


    try {
        // Check and extract "description"
        if (metadata.contains("description") && metadata["description"].is_string()) {
            description = metadata["description"].get<std::string>();
        } else {
            throw std::runtime_error("Missing or invalid 'description' in metadata.");
        }

        // Check and extract "d"
        if (metadata.contains("d") && metadata["d"].is_number_integer()) {
            d = metadata["d"].get<int>();
        } else {
            throw std::runtime_error("Missing or invalid 'd' in metadata.");
        }

        // Check and extract "N"
        if (metadata.contains("N") && metadata["N"].is_number_integer()) {
            N = metadata["N"].get<int>();
        } else {
            throw std::runtime_error("Missing or invalid 'N' in metadata.");
        }
        // Check and extract "M"
        if (metadata.contains("M") && metadata["M"].is_number_integer()) {
            M = metadata["M"].get<int>();
        } else {
            throw std::runtime_error("Missing or invalid 'M' in metadata.");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error extracting metadata: ") + e.what());
    }

    // Read vector size
    uint32_t vectorSize;
    inFile.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));

    // Read vector data and build checksum buffer incrementally
    std::vector<std::complex<double>> vec(vectorSize);
    std::vector<uint8_t> checksumBuffer;
    
    // Add metadata to checksum buffer
    checksumBuffer.insert(checksumBuffer.end(), 
                         metadataStr.begin(), metadataStr.end());
    
    for (size_t i = 0; i < vectorSize; ++i) {
        double real, imag;
        inFile.read(reinterpret_cast<char*>(&real), sizeof(real));
        inFile.read(reinterpret_cast<char*>(&imag), sizeof(imag));
        vec[i] = std::complex<double>(real, imag);
        
        // Add to checksum buffer (more efficient than stringstream)
        const uint8_t* realBytes = reinterpret_cast<const uint8_t*>(&real);
        const uint8_t* imagBytes = reinterpret_cast<const uint8_t*>(&imag);
        checksumBuffer.insert(checksumBuffer.end(), realBytes, realBytes + sizeof(real));
        checksumBuffer.insert(checksumBuffer.end(), imagBytes, imagBytes + sizeof(imag));
    }

    // Read footer: checksum
    uint32_t checksum;
    inFile.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));

    // Validate checksum (now only one pass through data)
    uint32_t calculatedChecksum = calculateChecksum(checksumBuffer);

    if (calculatedChecksum != checksum) {
        throw std::runtime_error("Checksum validation failed.");
    }

    inFile.close();

    // Return the deserialized data as DeserializedData struct
    DeserializedData deserializedData;
    // choose type
    if (std::strncmp(magic, "VECTR", 5) == 0) {
        deserializedData.type = "vector";
    } else if (std::strncmp(magic, "KRAUS", 5) == 0) {
        deserializedData.type = "kraus";
    } else {
        deserializedData.type = "unknown";
    }
    deserializedData.vectorData = vec;
    deserializedData.d = d;
    deserializedData.N = N;
    deserializedData.M = M;
    deserializedData.description = description;
    deserializedData.metadata = metadata;

    return deserializedData;
}

uint32_t VectorSerializer::calculateChecksum(const std::vector<uint8_t>& buffer) {
    // CRC32 implementation for better error detection
    // Uses standard CRC32 polynomial: 0xEDB88320
    uint32_t crc = 0xFFFFFFFF;
    
    for (uint8_t byte : buffer) {
        crc ^= byte;
        for (int i = 0; i < 8; ++i) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    
    return ~crc;
}

} // namespace utils

