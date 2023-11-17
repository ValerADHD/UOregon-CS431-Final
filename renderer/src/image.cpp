#include <stdlib.h>
#include <stdio.h>

#include "image.h"

Image *create_image(int width, int height) {
    Image *img = (Image *)malloc(sizeof(Image));

    img->width = width;
    img->height = height;

    img->data = (float *)malloc(sizeof(float) * width * height * 4);
    
    return img;
}

void destroy_image(Image *img) {
    free(img->data);
    free(img);
}

//writes RGBA pixel to (x, y) in img. Returns 1 if successful, 0 if failed.
int write_pixel(Image *img, unsigned int x, unsigned int y, float r, float g, float b, float a) {
    if(x >= img->width || y >= img->height) return 0;
    int base_idx = (y * img->width * 4) + x * 4;

    img->data[base_idx + 0] = r;
    img->data[base_idx + 1] = g;
    img->data[base_idx + 2] = b;
    img->data[base_idx + 3] = a;

    return 1;
}

int export_pnm(Image *img, char *path) {
    FILE *f = fopen(path, "w");
    if(f == NULL) {
        fprintf(stderr, "ERROR: Unable to open path for writing: %s", path);
        return -1;
    }

    //write buffer for easy bulk write to file
    unsigned char *write_data = (unsigned char *)malloc(sizeof(unsigned char) * img->width * img->height * 3);

    //convert float4 RGBA to 24bit RGB
    for(int i = 0; i < img->width * img->height; i++) {
        int read_base_idx = i * 4;
        int write_base_idx = i * 3;
        float r = img->data[read_base_idx + 0], g = img->data[read_base_idx + 1], b = img->data[read_base_idx + 2];
        
        //clamping range to (0->1) for visible values
        if(r < 0) r = 0; if(r > 1) r = 1;
        if(g < 0) g = 0; if(g > 1) g = 1;
        if(b < 0) b = 0; if(b > 1) b = 1;
        
        write_data[write_base_idx + 0] = (unsigned char)(r * 255);
        write_data[write_base_idx + 1] = (unsigned char)(g * 255);
        write_data[write_base_idx + 2] = (unsigned char)(b * 255);
    }

    //write header
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    //write data
    fwrite(write_data, sizeof(unsigned char), img->width * img->height * 3, f);

    fclose(f);
    free(write_data);
}