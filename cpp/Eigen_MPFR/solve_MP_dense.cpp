#include <Eigen/LU>
#include <iostream>
#include <Eigen/Dense>
// #include "mylibrary.hpp"
#include <unsupported/Eigen/MPRealSupport>

using namespace mpfr;
using namespace Eigen;

using MatrixXmp = Matrix<mpreal, Dynamic, Dynamic>;
using VectorXmp = Matrix<mpreal, Dynamic, 1>;

MatrixXmp hilbert_matrix(const int size);
mpreal cond_number(const MatrixXmp &A);
void show_results(MatrixXmp &A,
                  VectorXmp &b,
                  VectorXmp &x,
                  VectorXmp &xx,
                  const char *method);

int main(int argc, char *argv[])
{
    const int nn = atoi(argv[1]);
    assert(nn > 4);

    mpreal::set_default_prec(256);

    for (int n = 4; n < nn; ++n)
    {
        MatrixXmp A = hilbert_matrix(n);
        VectorXmp x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.;

        VectorXmp b = A * x;
        VectorXmp xx = A.partialPivLu().solve(b);
        show_results(A, b, x, xx, "partialPivLu");
        // VectorXmp xx = A.jacobiSvd((ComputeThinU | ComputeThinV)).solve(b);        
        // show_results(A, b, x, xx, "jacobiSvd");

    }

    return EXIT_SUCCESS;
}

MatrixXmp hilbert_matrix(const int size)
{
    MatrixXmp A = MatrixXmp::Zero(size, size);
    for (int i = 1; i < size + 1; ++i)
        for (int j = 1; j < size + 1; ++j)
            A(i - 1, j - 1) = 1. / (i + j - 1.);

    return A;
}

void show_results(MatrixXmp &A,
                  VectorXmp &b,
                  VectorXmp &x,
                  VectorXmp &xx,
                  const char *method)
{
    int n = A.rows();
    mpreal cond = cond_number(A);
    mpreal err = (x - xx).norm() / x.norm();
    printf("n = %3d, cond = %25e, error %25e: %s\n",
           n, cond.toDouble(),
           err.toDouble(), method);
}
mpreal cond_number(const MatrixXmp &A)
{
    JacobiSVD<MatrixXmp> svd(A);
    mpreal cond = svd.singularValues()(0) /
                  svd.singularValues()(svd.singularValues().size() - 1);

    return cond;
}
