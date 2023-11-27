#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "miniply.h"

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

    std::vector<Gaussian> *gaussians = new std::vector<Gaussian>();
    gaussians->reserve(vertex_element_count);

    for(int i = 0; i < vertex_element_count; i++) {
        int property_buffer_idx = i * NUM_PROPERTIES;
        Gaussian g;
        g.mean.x = property_buffer[property_buffer_idx + 0];
        g.mean.y = property_buffer[property_buffer_idx + 1];
        g.mean.z = property_buffer[property_buffer_idx + 2];
        g.color.x = property_buffer[property_buffer_idx + 3];
        g.color.y = property_buffer[property_buffer_idx + 4];
        g.color.z = property_buffer[property_buffer_idx + 5];
        g.alpha = property_buffer[property_buffer_idx + 6];
        g.scale.x = expf(property_buffer[property_buffer_idx + 7]);
        g.scale.y = expf(property_buffer[property_buffer_idx + 8]);
        g.scale.z = expf(property_buffer[property_buffer_idx + 9]);
        g.rot.x = property_buffer[property_buffer_idx + 10];
        g.rot.y = property_buffer[property_buffer_idx + 11];
        g.rot.z = property_buffer[property_buffer_idx + 12];
        g.rot.w = property_buffer[property_buffer_idx + 13];

        gaussians->push_back(g);
    }

    free(property_buffer);
    delete reader;

    Model *mdl = new Model();
    mdl->gaussians = gaussians;

    return mdl;
}

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
    printf("Initiating upload of %d gaussians\n", gm->data_len);
    
    GPUGaussian *temp_data = (GPUGaussian *)malloc(gm->data_len * sizeof(GPUGaussian));
    Gaussian *host_data = mdl->gaussians->data();

    for(int i = 0; i < gm->data_len; i++) {
        GPUGaussian *device = &(temp_data[i]);
        Gaussian host = host_data[i];
        //printf("Gaussian %d\n", i);
        if(i == 0) print_gaussian(&host);

        glm::mat4 imat = glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        imat = glm::scale(imat, host.scale);
        imat = imat * glm::toMat4(host.rot);
        imat = glm::translate(imat, host.mean);

        if(i == 0) printf("Inverse Matrix: \n\t");

        //we only *really* need the inverse matrix for our actual rendering calculation, so we will precompute it
        imat = glm::inverse(imat);
        for(int j = 0; j < 4; j++) {
            for(int k = 0; k < 4; k++) {
                device->imat[j][k] = imat[j][k];
                if(i == 0) printf("%f ", device->imat[j][k]);
            }
            if(i == 0) printf("\n\t");
        }
        
        device->color[0] = host.color.r;
        device->color[1] = host.color.g;
        device->color[2] = host.color.b;
        device->color[3] = host.alpha;
    }

    cudaError_t stat = cudaMalloc(&gm->data, gm->data_len * sizeof(GPUGaussian));
    if(stat != CUDA_SUCCESS) {
        fprintf(stderr, "Error! Unable to allocate memory on the GPU: %s\n", cudaGetErrorString(stat));
    }
    stat = cudaMemcpy(gm->data, temp_data, sizeof(GPUGaussian) * gm->data_len, cudaMemcpyHostToDevice);
    if(stat != CUDA_SUCCESS) {
        fprintf(stderr, "Error! Unable to properly transfer memory to GPU: %s\n", cudaGetErrorString(stat));
    }

    free(temp_data);

    return gm;
}

void destroy_GPU_model(GPUModel *gm) {
    cudaFree(gm->data);
    delete gm;
}
