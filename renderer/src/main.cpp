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

    int width = 1920, height = 1080;

    Model *mdl = load_model("./resources/point_cloud.ply");
    GPUModel *gm = upload_model(mdl);

    Image *img = create_image(width, height);

    PerspectiveCamera *cam;
    cudaMalloc(&cam, sizeof(PerspectiveCamera));

    float* device_img_buf;
    cudaMalloc(&device_img_buf, sizeof(float) * width * height * 4);

    //for(int i = 0; i < 100; i++) {
    camera_from_axis_angles(cam, 
        //glm::vec3(-3.003, 1.401, -2.228), glm::vec3(0.160570291183, 0.884881930761, 0.0820304748437), 90
        glm::vec3(0.36922905046275895, -1.098805384288247, -3.4042641920559777), glm::vec3(-0.118682389136, 0.171042266695, -0.0401425727959), 90
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