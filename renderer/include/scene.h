#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

#include <cstdint>

#include "image.h"
#include "camera.h"

class Scene {
  public:

    // Constructor

    // Methods
    void generate_camera_infos(std::vector<Image>* images,
            std::vector<Camera>* cameras);

  private:
    std::vector<CameraInfo> camera_infos;
    
};

#endif
