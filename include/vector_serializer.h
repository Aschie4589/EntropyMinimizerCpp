#ifndef VECTOR_SERIALIZER_H
#define VECTOR_SERIALIZER_H

#include "common_includes.h"

class VectorSerializer
{
private:
    /* data */
public:
    VectorSerializer(/* args */);

    ~VectorSerializer();

    int serializeVector(std::vector<std::complex<double> >* vector, std::string filename);
    std::vector<std::complex<double> >* deserializeVector(std::string filename);
};



#endif