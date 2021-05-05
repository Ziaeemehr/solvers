// Checking if a solution really exists
// Only you know what error margin you want to allow for a solution to be considered valid. So Eigen lets you do this computation for yourself, if you want to, as in this example:
// https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    // MatrixXd A = MatrixXd::Random(100, 100);
    // MatrixXd b = MatrixXd::Random(100, 50);
    // MatrixXd x = A.fullPivLu().solve(b);
    // cout << "rows " << x.rows() << ", cols "<< x.cols() << endl;
    // double relative_error = (A * x - b).norm() / b.norm(); // norm() is L2 norm
    // cout << "The relative error is:\n"
    //      << relative_error << endl;

    MatrixXd A = MatrixXd::Random(100, 100);
    VectorXd b = VectorXd::Random(100);
    MatrixXd x = A.fullPivLu().solve(b);
    double relative_error = (A * x - b).norm() / b.norm(); // norm() is L2 norm
    cout << "The relative error is:\n"
         << relative_error << endl;

}