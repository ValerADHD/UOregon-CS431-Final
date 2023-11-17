#include "render.h"

void render_call_handler(float *img_buffer, unsigned int width, unsigned int height) {
    dim3 blockDim;
    blockDim.x = 32; blockDim.y = 32; blockDim.z = 1;

    dim3 numBlocks;
    numBlocks.x = (width + blockDim.x - 1) / blockDim.x;
    numBlocks.y = (width + blockDim.y - 1) / blockDim.y;

    render<<<numBlocks, blockDim>>>(img_buffer, width, height);
}

__global__ void render(float *img_buffer, unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    int idx = x * width * 4 + y * 4;

    img_buffer[idx + 0] = x / (float)width / blockDim.x + blockIdx.x / (float)blockDim.x;
    img_buffer[idx + 1] = y / (float)height / blockDim.y + blockIdx.y / (float)blockDim.y;
    img_buffer[idx + 2] = 0.5;
    img_buffer[idx + 3] = 1.0;
}