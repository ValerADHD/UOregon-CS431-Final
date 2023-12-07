
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include <cstdint>

#include <opencv2/opencv.hpp>

#ifdef RUN_PARALLEL
    #include "cuda_operations.h"
#else
    #include "operations.h"
#endif

#include "camera.h"
#include "image.h"
#include "scene.h"

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

    Mat<double> R_transpose = transpose(R); 
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Rt[i][j] = R_transpose[i][j];
        } 
    }

    for (int i = 0; i < 3; i++) {
        Rt[i][3] = T[i];
    }
    Rt[3][3] = 1.0;
    return Rt;
}


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

    int num_cams = train_cam_infos.size();
    Mat<double> cam_centers(num_cams, Vec<double>(3));

    Vec<double> avg_centers(3, 0);
    Vec<double> dist(3, 0);

    for (int i = 0; i < num_cams; i++) {
        CameraInfo cam = train_cam_infos[i];
        Mat<double> R = cam.getR();
        Vec<double> T = cam.getT();

        Mat<double> W2C = world2view(R, T);
        Mat<double> C2W = MatInv(W2C);

        cam_centers[i][0] = C2W[0][3];
        avg_centers[0] += C2W[0][3];

        cam_centers[i][1] = C2W[1][3];
        avg_centers[1] += C2W[1][3];

        cam_centers[i][2] = C2W[2][3];
        avg_centers[2] += C2W[2][3];
    }

    for (int i = 0; i < 3; i++) {
        avg_centers[i] /= num_cams;
    }

    for (int i = 0; i < num_cams; i++) {
        Vec<double> center = cam_centers[i];
        Vec<double> adjusted_center(3);

        for (int j = 0; j < 3; j++) {
            adjusted_center[j] = center[j] - avg_centers[j];
            dist[j] += pow(adjusted_center[j], 2);
        }
    }

    dist[0] = sqrt(dist[0]); 
    dist[1] = sqrt(dist[1]); 
    dist[2] = sqrt(dist[2]); 

    double max_dist = 0;
    for (int i = 0; i < 3; i++) {
        if (dist[i] > max_dist) {
            max_dist = dist[i];           
        }
    }

    for (int i = 0; i < 3; i++) {
        avg_centers[i] *= -1;
    }
    
    radius = max_dist * 1.1;
    translate = avg_centers;
}
