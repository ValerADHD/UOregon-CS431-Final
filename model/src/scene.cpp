#include <iostream>
#include <vector>
#include <string>

#include <cstdint>

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "image.h"
#include "scene.h"
#include "operations.h"

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Mat = std::vector<std::vector<T>>;

void Scene::generate_camera_infos(Vec<Image>* images, Vec<Camera>* cameras) {
    for (int i = 0; i < images->size(); i++) {
        Image extr = (*images)[i];
        Camera intr = (*cameras)[extr.getCameraId() - 1];

        long long width = intr.getWidth();
        long long height = intr.getHeight();
        uint32_t uid = intr.getCameraId();

        Vec<double> qvec = extr.getQVec();
        Mat<double> rotmat = qvec2rotmat(qvec);

        Mat<double> R = transpose(rotmat);
        Vec<double> T = extr.getTVec();

        Vec<double> params = intr.getParams();
        double focal_length_x = params[0];
        double focal_length_y = params[1];
        double FovX = focal2fov(focal_length_x, width);
        double FovY = focal2fov(focal_length_y, height);

        std::string image_name = extr.getName();
        std::string image_path = extr.getPath();
        cv::Mat img = cv::imread(image_path.c_str());

        CameraInfo cam_info(uid, R, T, 
                FovX, FovY, img, 
                image_name, image_path, 
                width, height);

        train_cam_infos.push_back(cam_info);
    }
}

void Scene::generate_nerf_ppnorm() {

    CameraInfo cam = train_cam_infos[0];
    Mat<double> R = cam.getR();
    Vec<double> T = cam.getT();

    Mat<double> W2C = world2view(R, T);
    MatInv(W2C);
}
