#include <iostream>
#include <vector>
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

MatrixXmp hilbert_matrix(const int size);
void show_results(SparseMatrix<mpreal> &A,
                  VectorXmp &b,
                  VectorXmp &x,
                  VectorXmp &xx,
                  const char *method);

mpreal cond_number(const MatrixXmp &A);

// ------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const int nn = atoi(argv[1]);
    assert(nn > 4);

    mpreal::set_default_prec(256);

    // Built-in direct solvers --------------------------------------
    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<mpreal> A = hilbert_matrix(n).sparseView();
        VectorXmp x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXmp b = A * x;
        SparseLU<SparseMatrix<mpreal>, COLAMDOrdering<int>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        VectorXmp xx = solver.solve(b);
        show_results(A, b, x, xx, "SparseLU");
    }

    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<mpreal> A = hilbert_matrix(n).sparseView();
        VectorXmp x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXmp b = A * x;
        SparseQR<SparseMatrix<mpreal>, COLAMDOrdering<int>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        VectorXmp xx = solver.solve(b);
        show_results(A, b, x, xx, "SparseQR");
    }

    // Built-in iterative solvers -----------------------------------
    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<mpreal> A = hilbert_matrix(n).sparseView();
        VectorXmp x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXmp b = A * x;
        LeastSquaresConjugateGradient<SparseMatrix<mpreal>> lscg;
        lscg.compute(A);
        VectorXmp xx = lscg.solve(b);
        printf("# it: %5ld, est err: %e ", lscg.iterations(), lscg.error().toDouble());
        show_results(A, b, x, xx, "lscg");
    }

    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<mpreal> A = hilbert_matrix(n).sparseView();
        VectorXmp x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXmp b = A * x;
        BiCGSTAB<SparseMatrix<mpreal>> solver;
        solver.compute(A);
        VectorXmp xx = solver.solve(b);
        printf("# it: %5ld, est err: %18e ", solver.iterations(), solver.error().toDouble());
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

void show_results(SparseMatrix<mpreal> &A,
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