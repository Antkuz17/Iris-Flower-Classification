// Matrix class implementation used for basic matrix operations within neural networks
#include <vector>
#include <random>
#include <stdexcept>

class Matrix {
    public:

        Matrix(unsigned int rows, unsigned int col){
            num_rows = rows;
            num_col = col;

            engine = engine(42);
            dist = dist(0.0, 0.1);

            arr2D = std::vector<std::vector<double>>(num_rows, std::vector<double>(num_col));
            randomize_matrix();
        }

        // Function to get the number of rows
        unsigned int get_num_rows() const {
            return num_rows;
        }

        // Function to get the number of columns
        unsigned int get_num_col() const {
            return num_col;
        }

        // Function to get the value at a specific position
        double get_val(unsigned int row, unsigned int col) const {
            return arr2D[row][col];
        }


        // Matrix multiplication through operator overloading
        Matrix operator*(const Matrix& other) const {
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




    private:

        // The dimensions of the matrix
        unsigned int num_rows;
        unsigned int num_col;

        // 2d array of doubles that represents the matrix
        std::vector<std::vector<double>> arr2D{};

        // Used for the random number generation, no point in creating a new engine every time
        std::mt19937 engine;
        std::uniform_real_distribution<double> dist;

        // Geenerates a random number between 0 and 0.1, this is what fills the matrix at the beginning
        double gen_rand(){
            return dist(engine);
        }   



        // Take the matrix and fill it with default values before the training starts
        void randomize_matrix(){
            // outer loop iterate through col first
            for(int i{}; i < num_rows; i++){

                // Iterate over each row and fill with random nums
                for(int j{}; j < num_col; j++){
                    arr2D[i][j] = gen_rand();
                }
            }
        }
};

