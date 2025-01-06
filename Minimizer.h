// Minimizer.h
#ifndef MYCLASS_H  // Include guard to prevent multiple inclusions
#define MYCLASS_H

#include <complex>

class Minimizer {
public:
    Minimizer();             // Constructor declaration
    void sayHello();       // Method declaration
    ~Minimizer();            // Destructor declaration
private:
    int N, M, d;
    double epsilon;
    std::vector<std::complex<double> >* kraus_operators;
    std::vector<std::complex<double> >* vector_state;
    
};

#endif