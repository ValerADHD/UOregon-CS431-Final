#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"

//accessible CPU-side gaussian struct
typedef struct {
    //central "mean" of the gaussian, essentially the position
    glm::vec3 mean;
    //scale of the gaussian, in XYZ local axes (applied BEFORE ROTATION)
    glm::vec3 scale;
    //Quaternion rotation of the gaussian
    glm::quat rot;

    glm::vec3 color;
    float alpha;
} Gaussian;

//packed GPU-side gaussian struct
typedef struct {
    float mean[3];
    float mat[3][3];
    float color[4];
} GPUGaussian;

//holds references to all of the GPU allocated memory for a model
typedef struct {
    GPUGaussian *data;
} GPUModel;

typedef struct {
    std::vector<Gaussian> gaussians;
} Model;

Model *load_model(char *path);
void destroy_model(Model *mdl);

GPUModel *upload_model(Model *mdl);

#endif