#ifndef GENERATOR_H
#define GENERATOR_H

/* 
Define abstract class "Generator" which is responsible for generating Kraus operators.
Derived classes will implement different strategies for generating Kraus operators, e.g., randomly, or using some specific structure.
They will also either do that on CPU or GPU.
*/

#include <vector>
#include <memory>
#include <complex>

namespace channel {

// Template config for generator
template<typename ConfigType>
class Generator {
public:
    virtual ~Generator() = default;
    virtual int generate(std::vector<std::complex<double>>* kraus) = 0; // Purely virtual, MUST be implemented by derived classes
    
    ConfigType* getConfig() const { return &config; }
    int setConfig(const ConfigType& cfg) { config = cfg; return 0; }
    
protected:
    ConfigType config;
    
public:
    Generator(const ConfigType& cfg) : config(cfg) {}
};
} // namespace channel







#endif