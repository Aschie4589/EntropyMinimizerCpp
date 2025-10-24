#ifndef SUD_CG_H
#define SUD_CG_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

/*
Source: https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/ClebschGordan.cpp
https://arxiv.org/pdf/1009.0437


*/
// LAPACK functions are included through common_includes.h

namespace clebsch {
    const double EPS = 1e-12;

    // binomial coefficients
    class binomial_t {
        std::vector<int> cache;
        int N;

    public:
        int operator()(int n, int k);
    };

    extern binomial_t binomial;

    // Eq. (19) and (25)
    class weight {
        std::vector<int> elem;

    public:
        // the N in "SU(N)"
        const int N;

        // create a non-initialized weight
        weight(int N); // Note: indices of weight start with 1

        // create irrep weight of given index
        // Eq. (C2)
        weight(int N, int index);

        // assign from another instance
        clebsch::weight &operator=(const clebsch::weight &w);

        // access elements of this weight (k = 1, ..., N)
        int &operator()(int k);
        const int &operator()(int k) const;

        // compare weights
        // Eq. (C1)
        bool operator<(const weight &w) const;
        bool operator==(const weight &w) const;

        // element-wise sum of weights
        clebsch::weight operator+(const weight &w) const;

        // returns the index of this irrep weight (index = 0, 1, ...)
        // Eq. (C2)
        int index() const;

        // returns the dimension of this irrep weight
        // Eq. (22)
        long long dimension() const;
    };

    // Eq. (20)
    class pattern {
        std::vector<int> elem;

    public:
        // the N in "SU(N)"
        const int N;

        // copy constructor
        pattern(const pattern &pat);

        // create pattern of given index from irrep weight
        // Eq. (C7)
        pattern(const weight &irrep, int index = 0); // 0 based index again!

        // access elements of this pattern (l = 1, ..., N; k = 1, ..., l)
        int &operator()(int k, int l);
        const int &operator()(int k, int l) const;

        // find succeeding/preceding pattern, return false if not possible
        // Eq. (C9)
        bool operator++();
        bool operator--();

        // returns the pattern index (index = 0, ..., dimension - 1)
        // Eq. (C7)
        int index() const;

        // returns the pattern weight
        // Eq. (25)
        clebsch::weight get_weight() const;

        // returns matrix element of lowering operator J^(l)_-
        // between this pattern minus M^(k,l) and this pattern
        // (l = 1, ..., N; k = 1, ..., l)
        // Eq. (28)
        double lowering_coeff(int k, int l) const;

        // returns matrix element of raising operator J^(l)_+
        // between this pattern plus M^(k,l) and this pattern
        // (l = 1, ..., N; k = 1, ..., l)
        // Eq. (29)
        double raising_coeff(int k, int l) const;
    };

    class decomposition {
        std::vector<clebsch::weight> weights;
        std::vector<int> multiplicities;

    public:
        // the N in "SU(N)"
        const int N;

        // save given irreps for later use
        const weight factor1, factor2;

        // construct the decomposition of factor1 times factor2 into irreps
        // Eq. (31)
        decomposition(const weight &factor1, const weight &factor2);

        // return the number of occurring irreps
        int size() const;

        // access the occurring irreps
        // j = 0, ..., size() - 1
        const clebsch::weight &operator()(int j) const;

        // return the outer multiplicity of irrep in this decomposition
        int multiplicity(const weight &irrep) const;
    };

    class index_adapter {
        std::vector<int> indices;
        std::vector<int> multiplicities;

    public:
        // the N in "SU(N)"
        const int N;

        // save given irreps for later use
        const int factor1, factor2;

        // construct this index_adapter from a given decomposition
        index_adapter(const clebsch::decomposition &decomp);

        // return the number of occurring irreps
        int size() const;

        // access the occurring irreps
        int operator()(int j) const;

        // return the outer multiplicity of irrep in this decomposition
        int multiplicity(int irrep) const;
    };

    class coefficients {
        std::map<std::vector<int>, double> clzx;

        // access Clebsch-Gordan coefficients in convenient manner
        void set(int factor1_state,
                 int factor2_state,
                 int multiplicity_index,
                 int irrep_state,
                 double value);

        // internal functions, doing most of the work
        void highest_weight_normal_form(); // Eq. (37)
        void compute_highest_weight_coeffs(); // Eq. (36)
        void compute_lower_weight_coeffs(int multip_index, int state, std::vector<char> &done); // Eq. (40)

    public:
        // the N in "SU(N)"
        const int N;

        // save irreps and their dimensions for later use
        const weight factor1, factor2, irrep;
        const int factor1_dimension, factor2_dimension, irrep_dimension;

        // outer multiplicity of irrep in this decomposition
        const int multiplicity;

        // construct all Clebsch-Gordan coefficients of this decomposition
        coefficients(const weight &irrep, const weight &factor1, const weight &factor2);

        // access Clebsch-Gordan coefficients (read-only)
        // multiplicity_index = 0, ..., multiplicity - 1
        // factor1_state = 0, ..., factor1_dimension - 1
        // factor2_state = 0, ..., factor2_dimension - 1
        // irrep_state = 0, ..., irrep_dimension - 1
        double operator()(int factor1_state,
                          int factor2_state,
                          int multiplicity_index,
                          int irrep_state) const;
    };
};


#endif