#ifndef CAMERA_H
#define CAMERA_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cstdint>

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Mat = std::vector<std::vector<T>>;

class Camera {
  public:

    // Constructor
    Camera(uint32_t id, 
           std::string name, 
           unsigned long long width,
           unsigned long long height,
           Vec<double> params) :
             camera_id(id),
             model_name(name),
             width(width),
             height(height),
             params(params) {}

    // Methods
    uint32_t getCameraId() const { return camera_id; }
    std::string getName() const { return model_name; }
    unsigned long long getWidth() const { return width; }
    unsigned long long getHeight() const { return height; }
    Vec<double> getParams() const { return params; }

  private:
    uint32_t camera_id;
    std::string model_name;
    unsigned long long width;
    unsigned long long height;
    Vec<double> params;
};


class CameraInfo {
  public:

    CameraInfo(
            uint32_t uid, 
            Mat<double> R, 
            Vec<double> T, 
            double FovX, 
            double FovY, 
            cv::Mat image,
            std::string image_name, 
            std::string image_path,
            long long width, 
            long long height) :
              uid(uid),
              R(R),
              T(T),
              image(image),
              FovX(FovX),
              FovY(FovY),
              image_name(image_name),
              image_path(image_path),
              width(width),
              height(height) {}


    Mat<double> getR() { return R; }
    Vec<double> getT() { return T; }

  private:
    uint32_t uid;
    Mat<double> R;
    Vec<double> T;
    double FovX;
    double FovY;
    cv::Mat image;
    std::string image_name;
    std::string image_path;
    long long width;
    long long height;
};

#endif
