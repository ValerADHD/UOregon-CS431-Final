#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "miniply.h"
#include "happly.h"

#include <cstdio>
#include <cstring>
#include <string>

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "model.h"

#define NUM_PROPERTIES 14
#define PROPERTIES "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"

Model *load_model(char *path) {
    miniply::PLYReader *reader = new miniply::PLYReader(path);
    if(!reader->valid()) {
        fprintf(stderr, "Unable to open %s as .ply file!\n", path);
        delete reader;
        exit(EXIT_FAILURE);
    }
    printf("Successfully opened %s for reading\n", path);
    while(!reader->element_is("vertex")) {
        reader->next_element();
    }
    reader->load_element();
    uint32_t vertex_element_count = reader->num_rows();
    
    printf("Found %d elements to be extracted\n", vertex_element_count);
    uint32_t property_indexes[NUM_PROPERTIES];
    if(!reader->find_properties(property_indexes, NUM_PROPERTIES, PROPERTIES)) {
        fprintf(stderr, "Unable to find all properties!\n");
        delete reader;
        exit(EXIT_FAILURE);
    }    
    
    float *property_buffer = (float *)malloc(sizeof(float) * NUM_PROPERTIES * vertex_element_count);
    if(property_buffer == NULL) {
        fprintf(stderr, "Unable to allocate property buffer!\n");
        delete reader;
        exit(EXIT_FAILURE);
    }
    if(!reader->extract_properties(property_indexes, NUM_PROPERTIES, miniply::PLYPropertyType::Float, property_buffer)) {
        fprintf(stderr, "Unable to extract properties from file!\n");
        free(property_buffer);
        delete reader;
        exit(EXIT_FAILURE);
    }

    float modifier = 1.0;

    std::vector<Gaussian> *gaussians = new std::vector<Gaussian>();
    gaussians->reserve(vertex_element_count);
    for(int i = 0; i < vertex_element_count; i++) {
        int property_buffer_idx = i * NUM_PROPERTIES;
        Gaussian g;
        g.mean.x = property_buffer[property_buffer_idx + 0];
        g.mean.y = -property_buffer[property_buffer_idx + 1];
        g.mean.z = property_buffer[property_buffer_idx + 2];
        g.color.x = property_buffer[property_buffer_idx + 3];
        g.color.y = property_buffer[property_buffer_idx + 4];
        g.color.z = property_buffer[property_buffer_idx + 5];
        g.alpha = property_buffer[property_buffer_idx + 6];
        g.scale.x = modifier * powf(2.0, property_buffer[property_buffer_idx + 7]);
        g.scale.y = modifier * -powf(2.0, property_buffer[property_buffer_idx + 8]);
        g.scale.z = modifier * powf(2.0, property_buffer[property_buffer_idx + 9]);
        g.rot.w = property_buffer[property_buffer_idx + 10];
        g.rot.x = property_buffer[property_buffer_idx + 11];
        g.rot.y = property_buffer[property_buffer_idx + 12];
        g.rot.z = property_buffer[property_buffer_idx + 13];
        g.rot = glm::normalize(g.rot);
        gaussians->push_back(g);
    }
    free(property_buffer);
    delete reader;
    Model *mdl = new Model();
    mdl->gaussians = gaussians;
    return mdl;
}

/*Model *load_model(char *path) {
    happly::PLYData file(path);

    std::vector<float> x = file.getElement("vertex").getProperty<float>("x");
    std::vector<float> y = file.getElement("vertex").getProperty<float>("y");
    std::vector<float> z = file.getElement("vertex").getProperty<float>("z");
    std::vector<float> f_dc_0 = file.getElement("vertex").getProperty<float>("f_dc_0");
    std::vector<float> f_dc_1 = file.getElement("vertex").getProperty<float>("f_dc_1");
    std::vector<float> f_dc_2 = file.getElement("vertex").getProperty<float>("f_dc_2");
    std::vector<float> opacity = file.getElement("vertex").getProperty<float>("opacity");
    std::vector<float> scale_0 = file.getElement("vertex").getProperty<float>("scale_0");
    std::vector<float> scale_1 = file.getElement("vertex").getProperty<float>("scale_1");
    std::vector<float> scale_2 = file.getElement("vertex").getProperty<float>("scale_2");
    std::vector<float> rot_0 = file.getElement("vertex").getProperty<float>("rot_0");
    std::vector<float> rot_1 = file.getElement("vertex").getProperty<float>("rot_1");
    std::vector<float> rot_2 = file.getElement("vertex").getProperty<float>("rot_2");
    std::vector<float> rot_3 = file.getElement("vertex").getProperty<float>("rot_3");

    std::vector<Gaussian> *gaussians = new std::vector<Gaussian>();
    gaussians->reserve(x.size());

    for(int i = 0; i < x.size(); i++) {
        Gaussian g;
        g.mean.x = x[i];
        g.mean.y = -y[i];
        g.mean.z = z[i];
        g.color.r = f_dc_0[i];
        g.color.g = f_dc_1[i];
        g.color.b = f_dc_2[i];
        g.alpha = opacity[i];
        g.scale.x = powf(2.0, scale_0[i]);
        g.scale.y = -powf(2.0, scale_1[i]);
        g.scale.z = powf(2.0, scale_2[i]);
        g.rot.w = rot_0[i];
        g.rot.x = rot_1[i];
        g.rot.y = rot_2[i];
        g.rot.z = rot_3[i];
        g.rot = glm::normalize(g.rot);
        
        gaussians->push_back(g);
    }

    Model *mdl = new Model();
    mdl->gaussians = gaussians;

    return mdl;
}*/

void print_gaussian(Gaussian *g) {
    printf("\tMean: %f %f %f\n", g->mean.x, g->mean.y, g->mean.z);
    printf("\tColor: %f %f %f %f\n", g->color.r, g->color.g, g->color.b, g->alpha);
    printf("\tQuat: %f %f %f %f\n", g->rot.x, g->rot.y, g->rot.z, g->rot.w);
    printf("\tScale: %f %f %f\n", g->scale.x, g->scale.y, g->scale.z);
}

void destroy_model(Model *mdl) {
    delete mdl->gaussians;
    delete mdl;
}

GPUModel *upload_model(Model *mdl) {
    GPUModel *gm = new GPUModel();
    gm->data_len = mdl->gaussians->size();
    // gm->data_len = 3;
    printf("Initiating upload of %d gaussians\n", gm->data_len);
    
    Gaussian *host_data = mdl->gaussians->data();

    Gaussian *temp_data = (Gaussian *)malloc(sizeof(Gaussian) * gm->data_len);

    // temp_data[0].mean = glm::vec3(3.0, 1.5, 0.0);
    // temp_data[0].scale = glm::vec3(0.5 * 2.68, 0.5 * 1.89, 0.5 * 1.0);
    // temp_data[0].rot = glm::quat(0.063, 0.802, -0.362, 0.471);
    // temp_data[0].color = glm::vec3(1.0, 0.0, 0.0);
    // temp_data[0].alpha = 1.0;
    // temp_data[1].mean = glm::vec3(-3.0, 1.5, 0.0);
    // temp_data[1].scale = glm::vec3(0.5 * 2.68, 0.5 * 1.89, 0.5 * 1.0);
    // temp_data[1].rot = glm::quat(0.063, 0.802, -0.362, 0.471);
    // temp_data[1].color = glm::vec3(0.0, 1.0, 0.0);
    // temp_data[1].alpha = 1.0;
    // temp_data[2].mean = glm::vec3(0.0, -1.0, 0.0);
    // temp_data[2].scale = glm::vec3(0.2 * 2.68, 0.2 * 1.89, 0.2 * 1.0);
    // temp_data[2].rot = glm::quat(-0.054, 0.993, -0.102, -0.038);
    // temp_data[2].color = glm::vec3(0.0, 0.0, 1.0);
    // temp_data[2].alpha = 1.0;

    for(int i = 0; i < gm->data_len; i++) {
       temp_data[i] = host_data[i];
    }


    cudaError_t stat = cudaMalloc(&gm->data, gm->data_len * sizeof(Gaussian));
    if(stat != cudaSuccess) {
        fprintf(stderr, "Error! Unable to allocate memory on the GPU: %s\n", cudaGetErrorString(stat));
    }
    stat = cudaMemcpy(gm->data, temp_data, sizeof(Gaussian) * gm->data_len, cudaMemcpyHostToDevice);
    if(stat != cudaSuccess) {
        fprintf(stderr, "Error! Unable to properly transfer memory to GPU: %s\n", cudaGetErrorString(stat));
    }

    free(temp_data);
    
    return gm;
}

void destroy_GPU_model(GPUModel *gm) {
    cudaFree(gm->data);
    delete gm;
}
