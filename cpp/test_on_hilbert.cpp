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
                  VectorXd &xx,
                  const char *method)
{
    int n = A.rows();
    double cond = cond_number(A);
    double err = (x - xx).norm() / x.norm();
    printf("n = %3d, cond = %25e, error %25.16f: %s\n",
           n, cond, err, method);
}

int main()
{
    printf("===================================================================\n");
    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.partialPivLu().solve(b);
        show_results(A, b, x, xx, "partialPivLu");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.fullPivLu().solve(b);
        show_results(A, b, x, xx, "fullPivLu");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.householderQr().solve(b);
        show_results(A, b, x, xx, "householderQr");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.colPivHouseholderQr().solve(b);
        show_results(A, b, x, xx, "colPivHouseholderQr");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.fullPivHouseholderQr().solve(b);
        show_results(A, b, x, xx, "fullPivHouseholderQr");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.completeOrthogonalDecomposition().solve(b);
        show_results(A, b, x, xx, "completeOrthogonalDecomposition");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.llt().solve(b);
        show_results(A, b, x, xx, "llt");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.ldlt().solve(b);
        show_results(A, b, x, xx, "ldlt");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
        show_results(A, b, x, xx, "bdcSvd");
    }
    printf("===================================================================\n");

    for (int n = 4; n < 15; ++n)
    {
        MatrixXd A = hilbert_matrix(n);
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXd b = A * x;
        VectorXd xx = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
        show_results(A, b, x, xx, "jacobiSvd");
    }
    printf("===================================================================\n");


    return 0;
}