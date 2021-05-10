#include <iostream>
#include <unsupported/Eigen/MPRealSupport>
#include <Eigen/LU>
using namespace mpfr;
using namespace Eigen;
int main()
{
    // set precision to 256 bits (double has only 53 bits)
    mpreal::set_default_prec(256);
    // Declare matrix and vector types with multi-precision scalar type
    typedef Matrix<mpreal, Dynamic, Dynamic> MatrixXmp;
    typedef Matrix<mpreal, Dynamic, 1> VectorXmp;

    MatrixXmp A = MatrixXmp::Random(100, 100);
    VectorXmp b = VectorXmp::Random(100);

    // Solve Ax=b using LU
    VectorXmp x = A.lu().solve(b);
    std::cout << "relative error: " << (A * x - b).norm() / b.norm() << std::endl;
    return 0;
}
