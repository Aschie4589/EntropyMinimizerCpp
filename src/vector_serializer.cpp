#include "common_includes.h"
#include "vector_serializer.h"

VectorSerializer::VectorSerializer(/* args */)
{
}


VectorSerializer::~VectorSerializer()
{
}


int VectorSerializer::serializeVector(std::vector<std::complex<double> >* vector, std::string filename){
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()){
        throw std::runtime_error("Could not open file for writing.");
    }
    // Write the size of the vector
    size_t size = vector->size();
    file.write((char*)&size, sizeof(size_t));
    // Write the vector
    file.write((char*)vector->data(), size*sizeof(std::complex<double>));
    file.close();
    return 0;
}

std::vector<std::complex<double> >* VectorSerializer::deserializeVector(std::string filename){
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()){
        throw std::runtime_error("Could not open file for reading.");
    }
    // Read the size of the vector
    size_t size;
    file.read((char*)&size, sizeof(size_t));
    // Read the vector
    std::vector<std::complex<double> >* vector = new std::vector<std::complex<double> >(size);
    file.read((char*)vector->data(), size*sizeof(std::complex<double>));
    file.close();
    return vector;
}