#ifndef FILEREADER_H
#define FILEREADER_H

#include <string>
#include <vector>

#include "image.h"
#include "camera.h"

class FileReader {
  public:

    // Constructer
    FileReader(const std::string& directory, 
            std::vector<Image>* images,
            std::vector<Camera>* cameras);

    // Methods
    void read_data(const std::string& data_dir); 
    void read_extrinsic_file(const std::string& file_path, const std::string& data_dir); 
    void read_intrinsic_file(const std::string& file_path);

    template<typename T>
    T readNextBytes(std::ifstream& fid); 
    template<typename T>
    std::vector<T> readBinaryFile(std::ifstream& fid, int numBytes);

  private:
    std::string data_directory;
    std::vector<Image>* image_list;
    std::vector<Camera>* camera_list;
};

#endif
