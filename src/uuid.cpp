#include "uuid.h"

#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstdint>

std::string generate_uuid_v4() {
    // Random device and generator
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, std::numeric_limits<uint64_t>::max());

    // Generate 128 bits of random data
    uint64_t random1 = dis(gen);
    uint64_t random2 = dis(gen);

    // Adjust for version and variant bits
    random1 = (random1 & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL; // Version 4
    random2 = (random2 & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL; // Variant 1 (8, 9, a, or b)

    // Convert to string
    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(8) << (random1 >> 32)
        << "-"
        << std::setw(4) << ((random1 >> 16) & 0xFFFF)
        << "-"
        << std::setw(4) << (random1 & 0xFFFF)
        << "-"
        << std::setw(4) << (random2 >> 48)
        << "-"
        << std::setw(12) << (random2 & 0xFFFFFFFFFFFFULL);

    return oss.str();
}
