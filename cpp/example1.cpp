/*
 * simple example to show how to use eigen to solve 
 * a system of linear equations.
*/
#include <iostream>
#include <Eigen/Dense>
#include "mylibrary.hpp"

using namespace std;
using namespace Eigen;


int main()
{
    // Matrix3f A;
    MatrixXd A(3, 3);
    Vector3d b;
    A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
    b << 3, 3, 4;
    cout << "Here is the matrix A:\n"
         << A << endl;
    cout << "Here is the vector b:\n"
         << b << endl;
    Vector3d x = A.colPivHouseholderQr().solve(b);
    cout << "The solution is:\n"
         << x << endl;

    cout << "condition number is :" << cond_number(A) << endl;
    cout << "A x - b :" << A * x - b << endl;

    return 0;
}