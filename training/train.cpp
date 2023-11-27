#include <stdio.h>
#include "../renderer/include/model.h"
#include "../renderer/include/image.h"

struct View;

void training(int width, int height, View *views) {
    const int MAX_ITERS = 100_000;
    const int TRAINING_ITERS;
    Gaussian *M = get_sfm_points();
    int i = 0;
    for(i; i < MAX_ITERS; i++){
        View V = random_view_choice(views);
        Image gen_image = rasterize(V.location);
        float loss = calc_loss(V.image, gen_image);
        update
    }
}

float loss