// Minimizer.cpp
#include "Minimizer.h"
#include <iostream>
#include <complex>


Minimizer::Minimizer(std::vector<std::complex<double> >* kraus_pointer, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, double epsilon=1e-15) {
    // kraus_pointer is a pointer to a vector of double precision complex numbers
    kraus_operators = kraus_pointer;
    // assign the dimensions. Since I assign values, those are copied and d, N, M live in memory where the class instance lives
    d = kraus_number;
    N = kraus_in_dimension;
    M = kraus_out_dimension;
    // assign precision
    this->epsilon = epsilon; // "this" is like self for python, but returns a pointer to the instance of the class
    // initialize the vector state? (remember: also a pointer!)
    this->vector_state = 
}

void Minimizer::sayHello() {
    std::cout << "Hello, I am MyClass, and my data is " << data << std::endl;
    for (size_t i = 0; i < 10; i++) {
            std::cout << "kraus[" << i << "] = " << (*kraus_operators)[i] << std::endl;
    }
}

Minimizer::~Minimizer() {
    // Destructor logic (if needed)
}