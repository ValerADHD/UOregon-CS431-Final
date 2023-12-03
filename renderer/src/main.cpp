#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "glm/vec3.hpp"

#include "image.h"
#include "render.h"
#include "model.h"

int main(int argc, char **argv) {
    printf("Hello world!\n");

    int width = 800, height = 512;

    Model *mdl = load_model("./resources/bicycle_7000.ply");
    GPUModel *gm = upload_model(mdl);

    Image *img = create_image(width, height);

    PerspectiveCamera *cam;
    cudaMalloc(&cam, sizeof(PerspectiveCamera));

    float* device_img_buf;
    cudaMalloc(&device_img_buf, sizeof(float) * width * height * 4);

    //for(int i = 0; i < 100; i++) {
        camera_from_axis_angles(cam, 
            glm::vec3(0.0, 0.0, -1.0),
            glm::vec3(0.0),
            90
        );

        //gm->data_len = 100000;
        render_call_handler(device_img_buf, width, height, cam, gm);
        cudaDeviceSynchronize();

        cudaMemcpy(img->data, device_img_buf, sizeof(float) * width * height * 4, cudaMemcpyDeviceToHost);
        char path[128];
        int l = sprintf(path, "./test_%d.pnm", 0);
        path[l] = 0;
        export_pnm(img, path);
    //}
    
    
    cudaFree(device_img_buf);
    destroy_GPU_model(gm);
    cudaFree(cam);


    destroy_image(img);
    destroy_model(mdl);

    return 0;
}