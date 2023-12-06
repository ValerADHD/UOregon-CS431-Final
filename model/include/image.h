#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include <cstdint>

class Image {
  public:

    // Constructor
    Image(uint32_t id, 
          std::vector<double> qv, 
          std::vector<double> tv, 
          uint32_t camera_id, 
          std::string name, 
          std::string path, 
          std::vector<double> xs, 
          std::vector<double> ys, 
          std::vector<long long> p3ds) : 
            image_id(id), 
            qvec(qv), 
            tvec(tv), 
            camera_id(camera_id), 
            image_name(name), 
            image_path(path),
            x_vals(xs),
            y_vals(ys),
            p3d_ids(p3ds) {}

    // Methods
    uint32_t getId() const { return image_id; }
    std::vector<double> getQVec() const { return qvec; }
    std::vector<double> getTVec() const { return tvec; }
    uint32_t getCameraId() const { return camera_id; }
    std::string getName() const { return image_name; }
    std::string getPath() const { return image_path; }
    std::vector<double> getXVals() const { return x_vals; }
    std::vector<double> getYVals() const { return y_vals; }
    std::vector<long long> getP3dIds() const { return p3d_ids; }

  private:
    uint32_t image_id;
    std::vector<double> qvec;
    std::vector<double> tvec;
    uint32_t camera_id;
    std::string image_name;
    std::string image_path;
    std::vector<double> x_vals; 
    std::vector<double> y_vals; 
    std::vector<long long> p3d_ids;
};

#endif
