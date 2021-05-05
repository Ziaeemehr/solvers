#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#define MAXBUFSIZE ((int)1e6)

VectorXd readVector(const char *filename)
{
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];
    ifstream infile;
    infile.open(filename);
    while (!infile.eof())
    {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while (!stream.eof())
            stream >> buff[cols * rows + temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    VectorXd result(rows);
    for (int j = 0; j < rows; ++j)
        result(j) = buff[j];
    return result;
}

MatrixXd readMatrix(const char *filename)
{
    // https://stackoverflow.com/a/22988866/784433

    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (!infile.eof())
    {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while (!stream.eof())
            stream >> buff[cols * rows + temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    MatrixXd result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i, j) = buff[cols * i + j];

    return result;
};

double cond_number(const MatrixXd &A)
{
    JacobiSVD<MatrixXd> svd(A);
    double cond = svd.singularValues()(0) /
                  svd.singularValues()(svd.singularValues().size() - 1);

    return cond;
}
