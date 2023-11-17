#include <iostream>
#include <vector>
#include <string>

#include <cstdint>
#include <cmath>

#include "camera.h"
#include "image.h"
#include "scene.h"

double focal2fov(double focal, double pixels) {
        return 2 * atan(pixels / (2 * focal));
}

std::vector<std::vector<double>> transpose(
        std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> transposed_matrix(3, std::vector<double>(3));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }

    return transposed_matrix;
}


void Scene::generate_camera_infos(std::vector<Image>* images,
        std::vector<Camera>* cameras) {

    for (int i = 0; i < images->size(); i++) {
        Image extr = (*images)[i];
        Camera intr = (*cameras)[extr.getCameraId() - 1];

        long long width = intr.getWidth();
        long long height = intr.getHeight();
        uint32_t uid = intr.getCameraId();

        std::vector<double> qvec = extr.getQVec();

        std::vector<std::vector<double>> rotmat(3, std::vector<double>(3));
        rotmat[0][0] = 1 - 2 * (qvec[2] * qvec[2]) - 2 * (qvec[3] * qvec[3]);
        rotmat[0][1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3];
        rotmat[0][2] = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2];
        rotmat[1][0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3];
        rotmat[1][1] = 1 - 2 * (qvec[1] * qvec[1]) - 2 * (qvec[3] * qvec[3]);
        rotmat[1][2] = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1];
        rotmat[2][0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2];
        rotmat[2][1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1];
        rotmat[2][2] = 1 - 2 * (qvec[1] * qvec[1]) - 2 * (qvec[2] * qvec[2]);

        std::vector<std::vector<double>> R = transpose(rotmat);
        std::vector<double> T = extr.getTVec();

        std::vector<double> params = intr.getParams();
        double focal_length_x = params[0];
        double focal_length_y = params[1];
        double FovX = focal2fov(focal_length_x, width);
        double FovY = focal2fov(focal_length_y, height);

        std::string image_name = extr.getName();
        std::string image_path = extr.getPath();

        // add Image
        CameraInfo cam_info(uid, R, T, FovX, FovY, image_name, image_path, width, height);
        camera_infos.push_back(cam_info);
    }
}
