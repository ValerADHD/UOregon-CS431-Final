#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <cmath>
#include <vector>
#include "glm/mat4x4.hpp"

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Mat = std::vector<std::vector<T>>;

/*
 *  transpose
 */
Mat<double> transpose(Mat<double> matrix) {
    int n = matrix.size(); 
    Mat<double> transposed_matrix(n, Vec<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            transposed_matrix[j][i] = matrix[i][j];
        }   
    }   
    return transposed_matrix;
}

/*
 *  Matrix Inv
 */
Mat<double> MatInv(Mat<double> matrix) {
    Mat<double> mat_inv(4, Vec<double>(4));

    glm::mat4 glm_matrix;
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
                glm_matrix[i][j] = matrix[i][j];
        }
    }

    glm::mat4 inversed_matrix = glm::inverse(glm_matrix);
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
                mat_inv[i][j] = inversed_matrix[i][j];
        }
    }
    return mat_inv;
}


#endif
