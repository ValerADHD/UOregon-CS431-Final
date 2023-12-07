#ifndef CUDA_OPERATIONS_H
#define CUDA_OPERATIONS_H

#include <cmath>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include "glm/mat4x4.hpp"

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Mat = std::vector<std::vector<T>>;

Mat<double> transpose(Mat<double> matrix);

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
