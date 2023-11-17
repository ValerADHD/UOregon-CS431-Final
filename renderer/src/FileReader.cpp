#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cstdint>

#include "FileReader.h"
#include "image.h"
#include "camera.h"

/*
 *  Constructor
 */
FileReader::FileReader(const std::string& directory,
                        std::vector<Image>* images,
                        std::vector<Camera>* cameras)
{
    data_directory = directory;
    image_list = images;
    camera_list = cameras;
}

/*
 *  Read data from data directory
 */
void FileReader::read_data(const std::string& data_dir)
{
    std::cout << "Using data directory: "
              << data_dir
              << std::endl; 

    std::string extrinsic_path = data_directory + std::string("images.bin"); 
    std::string intrinsic_path = data_directory + std::string("cameras.bin"); 

    std::cout << "Loading extrinsic file... ";
    read_extrinsic_file(extrinsic_path, data_dir);
    std::cout << "Done!" << std::endl;

    std::cout << "Loading intrinsic file... ";
    read_intrinsic_file(intrinsic_path);
    std::cout << "Done!" << std::endl;
}

/*
 *  Read extrinsic file
 */
void FileReader::read_extrinsic_file(const std::string& file_path, const std::string& data_dir) 
{
    std::ifstream fid(file_path.c_str(), std::ios::binary);
    unsigned long long num_reg_images;
    if (fid) {
        num_reg_images = readNextBytes<unsigned long long>(fid);
    } else {
        std::cerr << "Unable to open file: " << file_path << std::endl;
        exit(1);
    }

    // std::cout << "Num images: " << num_reg_images << std::endl;

    for (int i = 0; i < num_reg_images; i++) {

        uint32_t image_id = readNextBytes<uint32_t>(fid);
        std::vector<double> qvec = readBinaryFile<double>(fid, 4);
        std::vector<double> tvec = readBinaryFile<double>(fid, 3);
        uint32_t camera_id = readNextBytes<uint32_t>(fid);

        std::string image_name = "";
        char current_char = readNextBytes<char>(fid);
        while (current_char != '\0') {
            image_name += current_char;
            current_char = readNextBytes<char>(fid);
        }
        
        uint32_t num_2d_points = readNextBytes<uint32_t>(fid);
        readNextBytes<int>(fid);

        std::vector<double> xs;
        std::vector<double> ys;
        std::vector<long long> p3ds;

        for (int j = 0; j < num_2d_points; j++) {
            xs.push_back(readNextBytes<double>(fid));
            ys.push_back(readNextBytes<double>(fid));
            p3ds.push_back(readNextBytes<long long>(fid));
        }

    /*   
        std::cout << "Image ID: " << image_id << std::endl;
        std::cout << "q vec: [";
        for (int i = 0; i< 4; i++) {
            std::cout << qvec[i] << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "t vec: [";
        for (int i = 0; i< 3; i++) {
            std::cout << tvec[i] << " ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Camera ID: " << camera_id << std::endl;

        std::cout << "Image Name: " << imageName << std::endl;
        std::cout << "Num 2D Points: " << num_2d_points << std::endl;
    */

        std::string image_path = data_dir + std::string("/images/") + image_name;
        Image img(image_id, qvec, tvec, camera_id, image_name, image_path, xs, ys, p3ds);
        image_list->push_back(img);
    }
}

/*
 *  Read intrinsic file
 */
void FileReader::read_intrinsic_file(const std::string& file_path) 
{
    std::ifstream fid(file_path.c_str(), std::ios::binary);
    unsigned long long num_cameras;
    if (fid) {
        num_cameras = readNextBytes<unsigned long long>(fid);
    } else {
        std::cerr << "Unable to open file: " << file_path << std::endl;
        exit(1);
    }

    // std::cout << "Num cameras: " << num_cameras << std::endl;

    for (int i = 0; i < num_cameras; i++) {
        uint32_t camera_id = readNextBytes<uint32_t>(fid);
        uint32_t model_id = readNextBytes<uint32_t>(fid);

        unsigned long long width = readNextBytes<unsigned long long>(fid);
        unsigned long long height = readNextBytes<unsigned long long>(fid);

        std::string model_name;
        uint32_t num_params;
        switch(camera_id) {
            case 0:
                model_name = std::string("SIMPLE_PINHOLE");
                num_params = 3;
                break;
            case 1:
                model_name = std::string("PINHOLE");
                num_params = 4;
                break;
            case 2:
                model_name = std::string("SIMPLE_RADIAL");
                num_params = 4;
                break;
            case 3:
                model_name = std::string("RADIAL");
                num_params = 5;
                break;
            case 4:
                model_name = std::string("OPENCV");
                num_params = 8;
                break;
            case 5:
                model_name = std::string("OPENCV_FISHEYE");
                num_params = 8;
                break;
            case 6:
                model_name = std::string("FULL_OPENCV");
                num_params = 12;
                break;
            case 7:
                model_name = std::string("FOV");
                num_params = 5;
                break;
            case 8:
                model_name = std::string("SIMPLE_RADIAL_FISHEYE");
                num_params = 4;
                break;
            case 9:
                model_name = std::string("RADIAL_FISHEYE");
                num_params = 5;
                break;
            case 10:
                model_name = std::string("THIN_PRISM_FISHEYE");
                num_params = 12;
                break;
            default:
                model_name = std::string("UNDEFINED");
                num_params = 0;
                break;
        }
       
        std::vector<double> params;
        for (int j = 0; j < num_params; j++) {
            params.push_back(readNextBytes<double>(fid));
        }

    /*
        std::cout << "Camera ID: " << camera_id << std::endl;
        std::cout << "Model ID: " << model_id << std::endl;
        std::cout << "Model name: " << model_name << std::endl;
        std::cout << "Num params: " << num_params << std::endl;
        std::cout << "Width: " << width << std::endl;
        std::cout << "Height: " << height << std::endl;
    */
     
        Camera camera(camera_id, model_name, width, height, params);
        camera_list->push_back(camera);
    }
}


template<typename T>
T FileReader::readNextBytes(std::ifstream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

template<typename T>
std::vector<T> FileReader::readBinaryFile(std::ifstream& fid, int numBytes) {
    std::vector<T> buffer(numBytes);
    fid.read(reinterpret_cast<char*>(&(buffer[0])), numBytes * sizeof(T));
    return buffer;
}
