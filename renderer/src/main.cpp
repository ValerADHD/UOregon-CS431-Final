#include <iostream>
#include <string>
#include <vector>

#include "FileReader.h"
#include "image.h"
#include "camera.h"

int main(int argc, char **argv) {
    if(argc < 2) {
        std::cerr << "Missing project name. Choose from {drjohnson, playroom, train, truck}" << std::endl;
        return 1;  // Return with error code 1
    }

    std::string file_path = std::string("data/") + argv[1] + std::string("/sparse/0/");

    std::vector<Image> images;
    std::vector<Camera> cameras;

    FileReader file(file_path, &images, &cameras);
    file.read_data();

    std::cout << "Number of images: " << images.size() << std::endl;
    std::cout << "Number of camera: " << cameras.size() << std::endl;
}
