#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <random>
#include <stdexcept>

class Matrix {
public:
    Matrix(unsigned int rows, unsigned int col);
    
    unsigned int get_num_rows() const;
    unsigned int get_num_col() const;
    double get_val(unsigned int row, unsigned int col) const;
    
    Matrix operator*(const Matrix& other) const;


    Matrix transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const
    Matrix elementwise_multiply(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix apply_function(double (*func)(double)) const;

private:
    unsigned int num_rows;
    unsigned int num_col;
    std::vector<std::vector<double>> arr2D;
    std::mt19937 engine;
    std::uniform_real_distribution<double> dist;
    
    double gen_rand();
    void randomize_matrix();
};

#endif