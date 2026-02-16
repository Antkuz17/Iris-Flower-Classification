// Matrix class implementation used for basic matrix operations within neural networks

#include "matrix.hpp"


Matrix::Matrix(unsigned int rows, unsigned int col)
    : num_rows(rows), num_col(col), engine(42), dist(0.0, 0.1) {
    
    arr2D = std::vector<std::vector<double>>(num_rows, std::vector<double>(num_col));
    randomize_matrix();
}

Matrix::Matrix(){
    num_rows = 0;
    num_col = 0;
    engine = std::mt19937(42);
    dist = std::uniform_real_distribution<double>(0.0, 0.1);
    arr2D = std::vector<std::vector<double>>(num_rows, std::vector<double>(num_col));
}

// Function to get the number of rows
unsigned int Matrix::get_num_rows() const {
    return num_rows;
}

// Function to get the number of columns
unsigned int Matrix::get_num_col() const {
    return num_col;
}

// Function to get the value at a specific position
double Matrix::get_val(unsigned int row, unsigned int col) const {
    return arr2D[row][col];
}

// Given a row and col, set that position to the provided value
void Matrix::set_val(unsigned int row, unsigned int col, double value){
    arr2D[row][col] = value; 
}


// Matrix multiplication through operator overloading
Matrix Matrix::operator*(const Matrix& other) const {
    if(num_col != other.num_rows){
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    // Initialize the result matrix with proper dimensions
    Matrix result(num_rows, other.num_col);

    for(unsigned int i = 0; i < num_rows; i++){
        for(unsigned int j = 0; j < other.num_col; j++){
            result.arr2D[i][j] = 0.0;
            for(unsigned int k = 0; k < num_col; k++){
                result.arr2D[i][j] += arr2D[i][k] * other.arr2D[k][j];
            }
        }
    }
    return result;
}

// Transpose the matrix by swapping rows and columns and return the transposed matrix
Matrix Matrix::transpose() const {
Matrix newMatrix(num_col, num_rows); 

for (unsigned int i = 0; i < num_rows; i++) {
    for (unsigned int j = 0; j < num_col; j++) {
        newMatrix.set_val(j, i, arr2D[i][j]); 
    }
}

return newMatrix;
}

// Returns the result of adding up two matrices
Matrix Matrix::operator+(const Matrix& other) const
{
// Matrix addition only applies if the dimensions of the two matrices are the same, if not throw an error
if(num_rows != other.num_rows || num_col != other.num_col){
    throw std::invalid_argument("Matrix dimensions do not match for addition");
}

// Initialize the result matrix with proper dimensions
Matrix result(num_rows, num_col);

// Main loop where each of the elements from each matrix are added together and stored in the result matrix
for(unsigned int i{}; i < num_rows; i++){
    for(unsigned int j{}; j < num_col; j++){
        result.arr2D[i][j] = arr2D[i][j] + other.arr2D[i][j];
    }
}

return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
// Matrix subtraction only applies if the dimensions of the two matrices are the same, if not throw an error
if(num_rows != other.num_rows || num_col != other.num_col){
    throw std::invalid_argument("Matrix dimensions do not match for subtraction");
}

// Initialize the result matrix with proper dimensions
Matrix result(num_rows, num_col);

// Main loop where each of the elements from each matrix are subtracted and stored in the result matrix
for(unsigned int i{}; i < num_rows; i++){
    for(unsigned int j{}; j < num_col; j++){
        result.arr2D[i][j] = arr2D[i][j] - other.arr2D[i][j];
    }
}

return result;
}

// Each entry (i, j) of matrix one is multipied by the equivalent entry in matrix 
Matrix Matrix::elementwise_multiply(const Matrix& other) const
{
if(num_rows != other.num_rows || num_col != other.num_col){
    throw std::invalid_argument("Matrix dimesions do not match for multiplication");
}

Matrix result(num_rows, num_col);

for(unsigned int i{}; i < num_rows; i++){
    for (unsigned int j{}; j < num_col; j++){
        result.arr2D[i][j] = arr2D[i][j] * other.arr2D[i][j];
    }
}

return result;
}

// Multiply each entry in the matrix by the same scalar value
Matrix Matrix::operator*(double scalar) const
{
Matrix result(num_rows, num_col);

for(unsigned int i{}; i < num_rows; i++){
    for(unsigned int j{}; j < num_col; j++){
        result.arr2D[i][j] = arr2D[i][j] * scalar;
    }
}

return result;
}

// Apply a function to each entry in the matrix and return the resulting matrix
Matrix Matrix::apply_function(double (*func)(double)) const{

// Initialize the result matrix with proper dimensions
Matrix result(num_rows, num_col);

// Main loop where each of the elements from the matrix have the function applied to them and stored in the result matrix   
for(unsigned int i{}; i < num_rows; i++){
    for(unsigned int j{}; j < num_col; j++){
        result.arr2D[i][j] = func(arr2D[i][j]);
    }
}

return result;
}

// Geenerates a random number between 0 and 0.1, this is what fills the matrix at the beginning
double Matrix::gen_rand(){
    return dist(engine);
}   


// Take the matrix and fill it with default values before the training starts
void Matrix::randomize_matrix(){
    // outer loop iterate through col first
    for(unsigned int i{}; i < num_rows; i++){

        // Iterate over each row and fill with random nums
        for(unsigned int j{}; j < num_col; j++){
            arr2D[i][j] = gen_rand();
        }
    }
}




