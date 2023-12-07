#include <stdio.h>
#include "lib/model.h"
#include "../renderer/include/image.h"
#include "../renderer/include/render.h"
#include "lib/scene.h"
#include "lib/camera.h"
#include <math.h>

Camera random_camera_choice(Scene scene);
Model get_sfm_points(Scene scene);
PerspectiveCamera cam_to_perspective(Mat<double> R, Mat<double> T);
float calc_loss(Image ref_img, Image gen_img, float lambda);

void training(unsigned int width, unsigned int height, Scene scene) {
    const unsigned int MAX_ITERS = 7000;
    const unsigned int DENSIFY_UNTIL = 100;
    const unsigned int DENSIFY_FROM = 0;
    const float DENSIFY_GRAD_THRESHOLD = 0.1;
    const float SIZE_THRESHOLD = 1000;
    const unsigned int DENSIFY_INTERVAL = 500;
    const unsigned int TRAINING_ITERS;
    const unsigned int OPACITY_RESET_INTERVAL = 1000;
    const float LAMBDA = 0.2;

    Model *mdl = get_sfm_points(scene);
    GPUModel *gm = upload_model(mdl);

    Image *gen_image_cpu = create_image(width, height);
    int i = 0;
    for(i; i < MAX_ITERS; i++) {
        CameraInfo rand_cam = random_camera_choice(scene);
        Image gen_image_gpu;
        gen_image_gpu.width = width;
        gen_image_gpu.height = height;

        cudaMalloc(&gen_image.data, sizeof(float) * width * height * 4);

        PerspectiveCamera rand_cam_pers = cam_to_perspective(rand_cam.getR(), rand_cam.getT());
        render_call_handler(gen_image_gpu.data, width, height, rand_cam_pers, gm);

        cudaMemcpy(gen_image_cpu->data, gen_image_gpu->data, sizeof(float) * width * height * 4, cudaMemcpyDeviceToHost);
        float loss = calc_loss(rand_cam.get_image(), gen_image, LAMBDA);
        
        // Densification
        if(i < DENSIFY_UNTIL) {
            if(i > DENSIFY_FROM && (i % DENSIFY_INTERVAL) == 0) {
                gm->densify_and_prune(DENSIFY_GRAD_THRESHOLD, 0.005, scene.generate_nerf_ppnorm(), 20.0);
            }
            if(i % OPACITY_RESET_INTERVAL == 0) {
                gm->reset_opacity();
            }
        }

        // Optimizer
        if(true) {
            optimizer_call_handler(gm)
        }
    }
    destroy_image(gen_image_cpu);
}

float calc_loss(Image ref_img, Image gen_img, float lambda) {
    float loss = 0;
    #pragma omp parallel for reduction(+:loss)
    for(int i = 0; i < ref_img.height * ref_img.width; i++) {
        loss += std::abs(ref_img.data[i] - gen_img.data[i]);
    }
    return loss;
}