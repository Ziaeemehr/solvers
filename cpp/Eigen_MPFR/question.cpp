#include <omp.h>
#include <vector>
#include <iostream>
#include <assert.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <unsupported/Eigen/MPRealSupport>

using namespace std;
using namespace mpfr;
using namespace Eigen;

using MatrixXmp = Matrix<mpreal, Dynamic, Dynamic>;
using VectorXmp = Matrix<mpreal, Dynamic, 1>;

void show_results(SparseMatrix<mpreal, Eigen::RowMajor> &A,
                  VectorXmp &b,
                  VectorXmp &x,
                  VectorXmp &xx,
                  const char *method);

mpreal cond_number(const MatrixXmp &A);
MatrixXmp hilbert_matrix(const int size);

// ------------------------------------------------------------------

int main(int argc, char *argv[])
{

    const int nn = 20;
    constexpr int maxiter = 2000;
    constexpr int num_decimal = 70;
    constexpr double tol = 1e-30;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(num_decimal));

    for (int n = 4; n < nn; ++n)
    {
        SparseMatrix<mpreal, Eigen::RowMajor> A = hilbert_matrix(n).sparseView();
        VectorXmp x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXmp b = A * x;
        BiCGSTAB<SparseMatrix<mpreal, Eigen::RowMajor>> solver(A);
        solver.setTolerance(tol);
        solver.setMaxIterations(maxiter);
        solver.compute(A);
        VectorXmp xx = solver.solve(b);
        printf("# it: %5ld, est err: %45.16e ", solver.iterations(), solver.error().toDouble());
        show_results(A, b, x, xx, "BiCGSTAB");
    }

    return 0;
}


MatrixXmp hilbert_matrix(const int size)
{
    MatrixXmp A = MatrixXmp::Zero(size, size);
    for (int i = 1; i < size + 1; ++i)
        for (int j = 1; j < size + 1; ++j)
            A(i - 1, j - 1) = 1. / (i + j - 1.);

    return A;
}


void show_results(SparseMatrix<mpreal, Eigen::RowMajor> &A,
                  VectorXmp &b,
                  VectorXmp &x,
                  VectorXmp &xx,
                  const char *method)
{
    int n = A.rows();
    mpreal cond = cond_number(MatrixXmp(A));
    // mpreal cond = 0.;
    mpreal err = (x - xx).norm() / x.norm();
    printf("n = %3d, cond = %25e, error %25.16f: %s\n",
           n, cond.toDouble(), err.toDouble(), method);
}

mpreal cond_number(const MatrixXmp &A)
{
    JacobiSVD<MatrixXmp> svd(A);
    mpreal cond = svd.singularValues()(0) /
                  svd.singularValues()(svd.singularValues().size() - 1);

    return cond;
}