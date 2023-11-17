#ifndef CAMERA_H
#define CAMERA_H

#include <string>
#include <vector>

#include <cstdint>

class Camera {
  public:

    // Constructor
    Camera(uint32_t id, 
           std::string name, 
           unsigned long long width,
           unsigned long long height,
           std::vector<double> params) :
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
    std::vector<double> getParams() const { return params; }

  private:
    uint32_t camera_id;
    std::string model_name;
    unsigned long long width;
    unsigned long long height;
    std::vector<double> params;
};

#endif
