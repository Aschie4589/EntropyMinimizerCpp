#include "common_includes.h"
#include "vector_serializer.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

VectorSerializer::VectorSerializer(/* args */)
{
}


VectorSerializer::~VectorSerializer()
{
}


void VectorSerializer::serialize(const std::string& type, const std::string& fileName, const std::vector<std::complex<double>>& vec, 
                                  const std::string& description, int d, int N) {
    std::ofstream outFile(fileName, std::ios::binary);
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
        {"N", N}
    };
    std::string metadataStr = metadata.dump();
    uint32_t metadataSize = metadataStr.size();
    outFile.write(reinterpret_cast<const char*>(&metadataSize), sizeof(metadataSize)); // Write the size of the metadata string
    outFile.write(metadataStr.c_str(), metadataSize); // Write the metadata string

    // Vector size
    uint32_t vectorSize = vec.size(); 
    outFile.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize)); // Write the size of the vector

    // Vector data
    for (const auto& elem : vec) {
        double real = elem.real();
        double imag = elem.imag();
        outFile.write(reinterpret_cast<const char*>(&real), sizeof(real));
        outFile.write(reinterpret_cast<const char*>(&imag), sizeof(imag));
    }

    // Footer: Calculate checksum
    std::vector<uint8_t> footerBuffer;
    std::stringstream footerStream;
    footerStream.write(metadataStr.c_str(), metadataSize);
    for (const auto& elem : vec) {
        footerStream.write(reinterpret_cast<const char*>(&elem), sizeof(elem));
    }

    footerBuffer.assign(std::istreambuf_iterator<char>(footerStream), std::istreambuf_iterator<char>()); 
    uint32_t checksum = calculateChecksum(footerBuffer);
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

    // Extract metadata to DeserualizedData struct
    std::string description;
    int d, N;


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
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error extracting metadata: ") + e.what());
    }

    // Read vector size
    uint32_t vectorSize;
    inFile.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));

    // Read vector data
    std::vector<std::complex<double>> vec(vectorSize);
    for (size_t i = 0; i < vectorSize; ++i) {
        double real, imag;
        inFile.read(reinterpret_cast<char*>(&real), sizeof(real));
        inFile.read(reinterpret_cast<char*>(&imag), sizeof(imag));
        vec[i] = std::complex<double>(real, imag);
    }

    // Read footer: checksum
    uint32_t checksum;
    inFile.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));

    // Validate checksum
    std::vector<uint8_t> footerBuffer;
    std::stringstream footerStream;
    footerStream.write(metadataStr.c_str(), metadataSize);
    for (const auto& elem : vec) {
        footerStream.write(reinterpret_cast<const char*>(&elem), sizeof(elem));
    }

    footerBuffer.assign(std::istreambuf_iterator<char>(footerStream), std::istreambuf_iterator<char>());
    uint32_t calculatedChecksum = calculateChecksum(footerBuffer);

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
    deserializedData.description = description;
    deserializedData.metadata = metadata;

    return deserializedData;
}

uint32_t VectorSerializer::calculateChecksum(const std::vector<uint8_t>& buffer) {
    uint32_t checksum = 0;
    for (uint8_t byte : buffer) {
        checksum += byte;
    }
    return checksum;
}

