#ifndef RENDER_H
#define RENDER_H

#include <cuda.h>

void render_call_handler(float *img_buffer, unsigned int width, unsigned int height);

__global__ void render(float *img_buffer, unsigned int width, unsigned int height);

#endif