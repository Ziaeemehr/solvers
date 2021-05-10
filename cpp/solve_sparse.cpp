#include <iostream>
#include <vector>
#include <assert.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

using namespace std;
using namespace Eigen;

double cond_number(const MatrixXd &A)
{
    JacobiSVD<MatrixXd> svd(A);
    double cond = svd.singularValues()(0) /
                  svd.singularValues()(svd.singularValues().size() - 1);

    return cond;
}

void show_results(SparseMatrix<double> &A,
                  VectorXd &b,
                  VectorXd &x,
                  VectorXd &xx,
                  const char *method)
{
    int n = A.rows();
    double cond = cond_number(MatrixXd(A));
    // double cond = 0.;
    double err = (x - xx).norm() / x.norm();
    printf("n = %3d, cond = %25e, error %25.16f: %s\n",
           n, cond, err, method);
}

MatrixXd hilbert_matrix(const int size)
{
    MatrixXd A = MatrixXd::Zero(size, size);
    for (int i = 1; i < size + 1; ++i)
        for (int j = 1; j < size + 1; ++j)
            A(i - 1, j - 1) = 1. / (i + j - 1.);

    return A;
}
// ------------------------------------------------------------------

int main(int argc, char *argv[])
{

    const int nn = atoi(argv[1]);
    assert(nn > 4);

    // Built-in direct solvers --------------------------------------
    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<double> A = hilbert_matrix(n).sparseView();
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXd b = A * x;
        SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        VectorXd xx = solver.solve(b);
        show_results(A, b, x, xx, "SparseLU");
    }

    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<double> A = hilbert_matrix(n).sparseView();
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXd b = A * x;
        SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        VectorXd xx = solver.solve(b);
        show_results(A, b, x, xx, "SparseQR");
    }

    // Built-in iterative solvers -----------------------------------
    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<double> A = hilbert_matrix(n).sparseView();
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXd b = A * x;
        LeastSquaresConjugateGradient<SparseMatrix<double>> lscg;
        lscg.compute(A);
        VectorXd xx = lscg.solve(b);
        printf("# it: %5ld, est err: %e ", lscg.iterations(), lscg.error());
        show_results(A, b, x, xx, "lscg");
    }

    for (int n = 4; n < nn; ++n)
    {

        SparseMatrix<double> A = hilbert_matrix(n).sparseView();
        VectorXd x(n);
        for (int i = 0; i < n; ++i)
            x(i) = 1.0;

        VectorXd b = A * x;
        BiCGSTAB<SparseMatrix<double> > solver;
        solver.compute(A);
        VectorXd xx = solver.solve(b);
        printf("# it: %5ld, est err: %e ", solver.iterations(), solver.error());
        show_results(A, b, x, xx, "BiCGSTAB");
    }

    return 0;
}