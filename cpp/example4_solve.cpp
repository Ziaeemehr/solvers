/**
 * test different method to solve an ill-conditioned system of equations
 */
#include <iostream>
#include <Eigen/Dense>
#include "mylibrary.hpp"

using namespace std;
using namespace Eigen;

void show_results(MatrixXd &A,
                  VectorXd &b,
                  VectorXd &x,
                  const char *method)
{
    // cout << "The solution is:\n"
    //      << x << endl;
    double relative_error = (A * x - b).norm() / b.norm();
    printf("The relative error for %40s  is %25.16f:\n",
           method, relative_error);
}

int main()
{

    MatrixXd A = readMatrix("data/A_large.txt");
    VectorXd b = readVector("data/B_large.txt");
    cout << "condition number is :" << cond_number(A) << endl;

    {
        VectorXd x = A.partialPivLu().solve(b);
        show_results(A, b, x, "partialPivLu");
    }

    {
        VectorXd x = A.fullPivLu().solve(b);
        show_results(A, b, x, "fullPivLu");
    }

    {
        VectorXd x = A.householderQr().solve(b);
        show_results(A, b, x, "householderQr");
    }

    {
        VectorXd x = A.colPivHouseholderQr().solve(b);
        show_results(A, b, x, "colPivHouseholderQr");
    }

    {
        VectorXd x = A.fullPivHouseholderQr().solve(b);
        show_results(A, b, x, "fullPivHouseholderQr");
    }
    {
        VectorXd x = A.completeOrthogonalDecomposition().solve(b);
        show_results(A, b, x, "completeOrthogonalDecomposition");
    }

    {
        VectorXd x = A.llt().solve(b);
        show_results(A, b, x, "llt");
    }

    {
        VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        show_results(A, b, x, "ldlt");
        VectorXd x0 = A.ldlt().solve(b);
        show_results(A, b, x0, "ldlt");
    }

    {

        VectorXd x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
        show_results(A, b, x, "bdcSvd");
    }
    {
        VectorXd x = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
        show_results(A, b, x, "jacobiSvd");
    }

    return 0;
}