#include "Minimizer.h"
#include <iostream>
#include <vector>
#include <lapacke.h>

using namespace std;


const int d = 10;
const int N = 200;


int main() {
    kraus_operators = new std::vector<std::complex<double> >(10,std::complex<double>(10.1f,10.2f)); // kraus_operators is the pointer.
    // A is a 3x3 matrix and b is a 3x1 vector (Ax = b)
    const int N = 3;
    const int d = 5;


    Minimizer obj = Minimizer(kraus_operators);  // Dynamically allocate object on the heap
    obj.sayHello();
//    delete obj;  // Automatically calls the destructor and deallocates memory



    double A[9] = {2.0, 1.0, -1.0, 
                   2.0, -1.0, 1.0, 
                   1.0, 2.0, 3.0};
    double b[3] = {3.0, 3.0, 3.0}; // Right-hand side vector

    // LDA is the leading dimension of A, which is N for a square matrix
    int lda = N;

    // IPIV is the pivot array (stores the pivot indices)
    int ipiv[N];

    // Calling LAPACK's dgesv (Solve A * X = B)
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, A, lda, ipiv, b, 1);

    if (info == 0) {
        cout << "Solution vector x: \n";
        for (int i = 0; i < N; ++i) {
            cout << b[i] << " ";
        }
        cout << endl;
    } else {
        cout << "The function failed. Info: " << info << endl;
    }

    return 0;
}
