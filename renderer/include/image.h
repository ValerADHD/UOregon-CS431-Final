#ifndef IMAGE_H
#define IMAGE_H

typedef struct {
    unsigned int width, height;
    float *data;
} Image;

Image *create_image(int width, int height);
void destroy_image(Image *img);

int write_pixel(Image *img, unsigned int x, unsigned int y, float r, float g, float b, float a);

int export_pnm(Image *img, char *path);

#endif