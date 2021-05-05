/*
# Separating the computation from the construction
  * In the above examples, the decomposition was computed at the same time that the decomposition object was constructed. There are however situations where you might want to separate these two things, for example if you don't know, at the time of the construction, the matrix that you will want to decompose; or if you want to reuse an existing decomposition object.

# What makes this possible is that:

  * all decompositions have a default constructor,
  * all decompositions have a compute(matrix) method that does the computation, and that may be called again on an already-computed decomposition, reinitializing it.
*/

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    Matrix2f A, b;
    LLT<Matrix2f> llt;
    A << 2, -1, -1, 3;
    b << 1, 2, 3, 1;
    cout << "Here is the matrix A:\n"
         << A << endl;
    cout << "Here is the right hand side b:\n"
         << b << endl;
    cout << "Computing LLT decomposition..." << endl;
    llt.compute(A);
    cout << "The solution is:\n"
         << llt.solve(b) << endl;
    A(1, 1)++;
    
    cout << "The matrix A is now:\n"
         << A << endl;
    cout << "Computing LLT decomposition..." << endl;
    llt.compute(A);
    cout << "The solution is now:\n"
         << llt.solve(b) << endl;
}