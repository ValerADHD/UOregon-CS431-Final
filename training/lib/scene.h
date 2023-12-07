#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

#include <cstdint>

#include "image.h"
#include "camera.h"

template <typename T>
using Vec = std::vector<T>;

class Scene {
  public:

    // Methods
    void generate_camera_infos(Vec<Image>* images,
            Vec<Camera>* cameras);
    void generate_nerf_ppnorm();

  private:
    Vec<CameraInfo> train_cam_infos;
    
};

#endif
