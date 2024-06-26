#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <cmath>
#include <vector>

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Mat = std::vector<std::vector<T>>;


/*
 *  focal2fov
 */
double focal2fov(double focal, double pixels) {
        return 2 * atan(pixels / (2 * focal));
}

/*
 *  NxN transpose
 */
Mat<double> transpose(Mat<double>& matrix) {
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
 *  QVec to RotMat
 */
Mat<double> qvec2rotmat(Vec<double> qvec) {
    Mat<double> rotmat(3, Vec<double>(3));
    rotmat[0][0] = 1 - 2 * (qvec[2] * qvec[2]) - 2 * (qvec[3] * qvec[3]);
    rotmat[0][1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3];
    rotmat[0][2] = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2];
    rotmat[1][0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3];
    rotmat[1][1] = 1 - 2 * (qvec[1] * qvec[1]) - 2 * (qvec[3] * qvec[3]);
    rotmat[1][2] = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1];
    rotmat[2][0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2];
    rotmat[2][1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1];
    rotmat[2][2] = 1 - 2 * (qvec[1] * qvec[1]) - 2 * (qvec[2] * qvec[2]);
    return rotmat;
}

/*
 *  World to View 
 */
Mat<double> world2view(Mat<double> R, Vec<double> T) {
    Mat<double> Rt(4, Vec<double>(4));

/*    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << R[i][j] << " ";
        }
        std::cout << std::endl;
    }
*/
    // TODO
    return Rt;

}


#endif
