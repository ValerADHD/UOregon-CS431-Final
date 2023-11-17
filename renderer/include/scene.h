#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

#include <cstdint>

#include "image.h"
#include "camera.h"

template <typename T>
using Mat = std::vector<std::vector<T>>;

class Scene {
  public:

    // Methods
    void generate_camera_infos(std::vector<Image>* images,
            std::vector<Camera>* cameras);
    void generate_nerf_ppnorm();

  private:
    std::vector<CameraInfo> train_cam_infos;
    
};

#endif
