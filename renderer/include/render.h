#ifndef RENDER_H
#define RENDER_H

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include "glm/vec3.hpp"
#include "glm/vec2.hpp"

#include "model.h"

typedef struct {
    glm::vec3 origin;
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;

    float vertical_fov;
} PerspectiveCamera;

void camera_from_axis_angles(PerspectiveCamera *d_pc, glm::vec3 origin, glm::vec3 axis_angles, float FOV);
void camera_from_lookat(PerspectiveCamera *d_pc, glm::vec3 origin, glm::vec3 lookat, float FOV);
void destroy_perspective_camera(PerspectiveCamera *cam);

void render_call_handler(
    float *img_buffer, unsigned int width, unsigned int height, PerspectiveCamera *cam,
    GPUModel *gm
);

__global__ void binGaussians(
    PerspectiveCamera *cam, 
    Gaussian *gaussians, uint32_t n,
    float numTilesX, float numTilesY, //number of tiles for the whole screen. Can be non-integer if tiles are partially off-screen
    float *horizontalFrustumPlanes, float *verticalFrustumPlanes,
    uint32_t *bins, //actual bins, uint32_t[numBinsX][numBinsY][MAX_BIN_SIZE]
    uint32_t *binIndexes //index to the next open element in each bin, uint32_t[numBinsX][numBinsY]
);

__global__ void sortBins(
    PerspectiveCamera *cam,
    Gaussian *gaussians, uint32_t n,
    float numTilesX, float numTilesY,
    uint32_t *bins, uint32_t *binSizes
);

__global__ void render(
    float *img_buffer, unsigned int width, unsigned int height, 
    PerspectiveCamera *cam, 
    Gaussian *gaussians, uint32_t n,
    uint32_t numBinsX, uint32_t numBinsY,
    uint32_t *bins, uint32_t *bin_sizes
);

#endif