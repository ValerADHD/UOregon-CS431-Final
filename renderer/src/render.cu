#include <stdio.h>
#include <stdlib.h>

#include "render.h"
#include "model.h"

#define GLM_FORCE_CUDA
#include "glm/mat3x3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"
#include "glm/vec3.hpp"
#include "glm/vec2.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"

void camera_from_axis_angles(PerspectiveCamera *d_pc, glm::vec3 origin, glm::vec3 axis_angles, float fov) {
    PerspectiveCamera *pc = new PerspectiveCamera();
    pc->origin = origin;

    glm::mat3 mat = glm::toMat3(glm::quat(glm::vec3(axis_angles.x, axis_angles.y, axis_angles.z)));
    pc->right = mat[0];
    pc->up = mat[1];
    pc->forward = mat[2];

    pc->vertical_fov = fov;
    cudaMemcpy(d_pc, pc, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
    delete pc;
}

void camera_from_lookat(PerspectiveCamera *d_pc, glm::vec3 origin, glm::vec3 lookat, float fov) {
    PerspectiveCamera *pc = new PerspectiveCamera();
    pc->origin = origin;

    glm::mat4 mat = glm::lookAt(
        origin, lookat,
        glm::vec3(0, 1, 0)
    );
    
    pc->right = glm::vec3(mat[0].x, mat[0].y, mat[0].z);
    pc->up = glm::vec3(mat[1].x, mat[1].y, mat[1].z);
    pc->forward = glm::vec3(mat[2].x, mat[2].y, mat[3].z);

    pc->vertical_fov = fov;
    cudaMemcpy(d_pc, pc, sizeof(PerspectiveCamera), cudaMemcpyHostToDevice);
    delete pc;
}

#define TEST_FRUSTUM_PLANE_GENERATION

void generate_frustum_planes(PerspectiveCamera *cam, float numTilesX, float numTilesY, float **hPlanes, float **vPlanes) {
    PerspectiveCamera *c = new PerspectiveCamera(); //'cam' is on the GPU, we need a copy for the CPU
    cudaError_t stat = cudaMemcpy(c, cam, sizeof(PerspectiveCamera), cudaMemcpyDeviceToHost);
    if(stat != cudaSuccess) {
        fprintf(stderr, "Unable to copy camera off the GPU: %s\n", cudaGetErrorName(stat));
        delete c;
        exit(EXIT_FAILURE);
    }

    uint32_t numBinsX = ceil(numTilesX), numBinsY = ceil(numTilesY);
    float *horizontalFrustumPlanes = (float *)malloc(sizeof(float) * 3 * (numBinsY + 1));
    float *verticalFrustumPlanes = (float *)malloc(sizeof(float) * 3 * (numBinsX + 1));
    //generate horizontal frustum planes, bottom to top
    for(int i = 0; i < numBinsY + 1; i++) {
        float uvY = min(1.0, (float)i / numTilesY);
        uvY = uvY * 2.0 - 1.0;
        
        glm::vec3 ray1 = glm::normalize(cam->forward + cam->up * uvY + cam->right);
        glm::vec3 ray2 = glm::normalize(cam->forward + cam->up * uvY - cam->right);
        glm::vec3 planeNorm = glm::cross(ray1, ray2);

        horizontalFrustumPlanes[i * 3 + 0] = planeNorm.x;
        horizontalFrustumPlanes[i * 3 + 1] = planeNorm.y;
        horizontalFrustumPlanes[i * 3 + 2] = planeNorm.z;
    }
    //generate vertical frustum planes, left to right
    for(int i = 0; i < numBinsY + 1; i++) {
        float uvX = min(1.0, (float)i / numTilesX);
        uvX = uvX * 2.0 - 1.0;
        
        glm::vec3 ray1 = glm::normalize(cam->forward + cam->up + cam->right * uvX);
        glm::vec3 ray2 = glm::normalize(cam->forward - cam->up + cam->right * uvX);
        glm::vec3 planeNorm = glm::cross(ray1, ray2);

        verticalFrustumPlanes[i * 3 + 0] = planeNorm.x;
        verticalFrustumPlanes[i * 3 + 1] = planeNorm.y;
        verticalFrustumPlanes[i * 3 + 2] = planeNorm.z;
    }

#ifdef TEST_FRUSTUM_PLANE_GENERATION
    int num_tests_passed = 0;
    for(int x = 0; x < numBinsX; x++) {
        for(int y = 0; y < numBinsY; y++) {
            //generate a point guaranteed to be inside the frustum for tile[x][y]
            float uvXMin = min(1.0, (float)x / numTilesX),
                uvXMax = min(1.0, (float)(x + 1) / numTilesX),
                uvYMin = min(1.0, (float)y / numTilesY),
                uvYMax = min(1.0, (float)(y + 1) / numTilesY);
            float uvX = (uvXMin + uvXMax) / 2.0; uvX = uvX * 2.0 - 1.0;
            float uvY = (uvYMin + uvYMax) / 2.0; uvY = uvY * 2.0 - 1.0;
            
            glm::vec3 point = cam->forward + cam->right * uvX + cam->up * uvY;

            glm::vec3 leftPlaneNormal = glm::vec3(verticalFrustumPlanes[x * 3 + 0], verticalFrustumPlanes[x * 3 + 1], verticalFrustumPlanes[x * 3 + 2]);
            glm::vec3 rightPlaneNormal = -glm::vec3(verticalFrustumPlanes[x * 3 + 3], verticalFrustumPlanes[x * 3 + 4], verticalFrustumPlanes[x * 3 + 5]);
            glm::vec3 bottomPlaneNormal = glm::vec3(horizontalFrustumPlanes[y * 3 + 0], horizontalFrustumPlanes[y * 3 + 1], horizontalFrustumPlanes[y * 3 + 2]);
            glm::vec3 topPlaneNormal = -glm::vec3(horizontalFrustumPlanes[y * 3 + 3], horizontalFrustumPlanes[y * 3 + 4], horizontalFrustumPlanes[y * 3 + 5]);
        
            float leftPlaneDist = glm::dot(-leftPlaneNormal, cam->origin - point);
            float rightPlaneDist = glm::dot(-rightPlaneNormal, cam->origin - point);
            float bottomPlaneDist = glm::dot(-bottomPlaneNormal, cam->origin - point);
            float topPlaneDist = glm::dot(-topPlaneNormal, cam->origin - point);

            if(leftPlaneDist > 0 && rightPlaneDist > 0 && bottomPlaneDist > 0 && topPlaneDist > 0) {
                num_tests_passed++;
            }
        }
    }
    if(num_tests_passed == numBinsX * numBinsY) {
        fprintf(stderr, "Success! The generated frustum planes worked for all %d tests\n", num_tests_passed);
    } else {
        fprintf(stderr, "Failure! The generated frustum planes failed on %d tests\n", numBinsX * numBinsY - num_tests_passed);
    }
#endif

    stat = cudaMalloc(hPlanes, sizeof(float) * 3 * (numBinsY + 1));
    if(stat != cudaSuccess) fprintf(stderr, "Unable to allocate memory for frustum planes: %s\n", cudaGetErrorString(stat));
    stat = cudaMalloc(vPlanes, sizeof(float) * 3 * (numBinsY + 1));
    if(stat != cudaSuccess) fprintf(stderr, "Unable to allocate memory for frustum planes: %s\n", cudaGetErrorString(stat));
    stat = cudaMemcpy(*hPlanes, horizontalFrustumPlanes, sizeof(float) * 3 * (numBinsY + 1), cudaMemcpyHostToDevice);
    if(stat != cudaSuccess) fprintf(stderr, "Unable to upload frustum planes to GPU: %s\n", cudaGetErrorString(stat));
    stat = cudaMemcpy(*vPlanes, verticalFrustumPlanes, sizeof(float) * 3 * (numBinsX + 1), cudaMemcpyHostToDevice);
    if(stat != cudaSuccess) fprintf(stderr, "Unable to upload frustum planes to GPU: %s\n", cudaGetErrorString(stat));
    
    free(horizontalFrustumPlanes);
    free(verticalFrustumPlanes);
}

void render_call_handler(float *img_buffer, unsigned int width, unsigned int height, PerspectiveCamera *cam, GPUModel *gm) {
    dim3 blockDim;
    blockDim.x = 32; blockDim.y = 32; blockDim.z = 1;

    dim3 numBlocks;
    numBlocks.x = (width + blockDim.x - 1) / blockDim.x;
    numBlocks.y = (height + blockDim.y - 1) / blockDim.y;
    numBlocks.z = 1;

    float *horizontalFrustumPlanes, *verticalFrustumPlanes;
    generate_frustum_planes(cam, width / 32.0, height / 32.0, &horizontalFrustumPlanes, &verticalFrustumPlanes);
    cudaFree(horizontalFrustumPlanes); cudaFree(verticalFrustumPlanes);

    render<<<numBlocks, blockDim>>>(img_buffer, width, height, cam, gm->data, gm->data_len);
}

__forceinline__ __device__ glm::mat4 make_mat4(float mat[4][4]) {
    return glm::mat4(
        mat[0][0], mat[0][1], mat[0][2], mat[0][3],
        mat[1][0], mat[1][1], mat[1][2], mat[1][3],
        mat[2][0], mat[2][1], mat[2][2], mat[2][3],
        mat[3][0], mat[3][1], mat[3][2], mat[3][3]
    );
}

__forceinline__ __device__ glm::vec4 make_vec4(float vec[4]) {
    return glm::vec4(
        vec[0], vec[1], vec[2], vec[3]
    );
}

__forceinline__ __device__ float originDistanceToPlane(glm::vec3 planeOrigin, glm::vec3 planeNormal) {
    return glm::dot(-planeNormal, planeOrigin);
}

__global__ void binGaussians(
    PerspectiveCamera *cam, 
    GPUGaussian *gaussians, uint32_t n,
    float numTilesX, float numTilesY, //number of tiles for the whole screen. Can be non-integer if tiles are partially off-screen
    float *horizontalFrustumPlanes, float *verticalFrustumPlanes,
    uint32_t ***bins, //actual bins, uint32_t[numBinsX][numBinsY][MAX_BIN_SIZE]
    uint32_t **binIndexes //index to the next open element in each bin, uint32_t[numBinsX][numBinsY]
) {
    uint32_t numBinsX = ceil(numTilesX), numBinsY = ceil(numTilesY);

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        //all planes are gauranteed to intersect the camera origin, so we will use that as their origins
        glm::vec3 planeOrigin = cam->origin;
        
        glm::mat4 imat = make_mat4(gaussians[i].imat);
        
        //transform the origin into the gaussian's model space
        planeOrigin = imat * glm::vec4(planeOrigin, 1.0);

        //loop through each bin, determine if the gaussian is inside that frustum
        for(int x = 0; x < numBinsX; x++) {
            for(int y = 0; y < numBinsY; y++) {
                //get all of the planes for the edges of this tile's frustum
                glm::vec3 leftPlaneNormal = glm::vec3(verticalFrustumPlanes[x * 3 + 0], verticalFrustumPlanes[x * 3 + 1], verticalFrustumPlanes[x * 3 + 2]);
                glm::vec3 rightPlaneNormal = -glm::vec3(verticalFrustumPlanes[x * 3 + 3], verticalFrustumPlanes[x * 3 + 4], verticalFrustumPlanes[x * 3 + 5]);
                glm::vec3 bottomPlaneNormal = glm::vec3(horizontalFrustumPlanes[y * 3 + 0], horizontalFrustumPlanes[y * 3 + 1], horizontalFrustumPlanes[y * 3 + 2]);
                glm::vec3 topPlaneNormal = -glm::vec3(horizontalFrustumPlanes[y * 3 + 3], horizontalFrustumPlanes[y * 3 + 4], horizontalFrustumPlanes[y * 3 + 5]);
                
                //transform them into the gaussian's model space, so we can simply test against the unit sphere
                leftPlaneNormal = glm::vec3(glm::normalize(imat * glm::vec4(leftPlaneNormal, 0.0)));
                rightPlaneNormal = glm::vec3(glm::normalize(imat * glm::vec4(rightPlaneNormal, 0.0)));
                bottomPlaneNormal = glm::vec3(glm::normalize(imat * glm::vec4(bottomPlaneNormal, 0.0)));
                topPlaneNormal = glm::vec3(glm::normalize(imat * glm::vec4(topPlaneNormal, 0.0)));

                //get the signed distance from the origin to the plane
                float leftPlaneDist = originDistanceToPlane(planeOrigin, leftPlaneNormal);
                float rightPlaneDist = originDistanceToPlane(planeOrigin, rightPlaneNormal);
                float bottomPlaneDist = originDistanceToPlane(planeOrigin, bottomPlaneNormal);
                float topPlaneDist = originDistanceToPlane(planeOrigin, topPlaneNormal);

                //if the unit sphere is < 1 (absolute) distance unit away, it intersects the plane
                //if the signed distance is greater than 0, it's on the forward (inside frustum) side of the plane
                //so if the signed distance > -1 (for all frustum planes), at least part of the unit sphere intersects with
                //the frustum
                if(leftPlaneDist > -1 && rightPlaneDist > -1 && topPlaneDist > -1 && bottomPlaneDist > -1) {
                    //inside frustum
                    uint32_t binIdx = atomicAdd(&(binIndexes[x][y]), 1);
                    bins[x][y][binIdx] = i;
                }
            }
        }
    }
}

__device__ void ray_unit_sphere_intersection(glm::vec3 ray_origin, glm::vec3 ray_direction, glm::vec3 *front_intersection, glm::vec3 *back_intersection) {
    *front_intersection = glm::vec3(0.0, 0.0, 0.0); *back_intersection = glm::vec3(0.0, 0.0, 0.0);

    float ray_parallel_dist = glm::dot(-ray_origin, ray_direction);
    float ray_perpendicular_dist = glm::length(-ray_origin - ray_direction * ray_parallel_dist);

    if(ray_parallel_dist < 0.0 || ray_perpendicular_dist > 1.0) return;

    float intersection_dist = sqrt(1.0 - ray_perpendicular_dist * ray_perpendicular_dist);

    *front_intersection = ray_origin + ray_direction * (ray_parallel_dist - intersection_dist);
    *back_intersection = ray_origin + ray_direction * (ray_parallel_dist + intersection_dist);
}

#define NUM_RAY_STEPS 3
__device__ glm::vec4 raymarch_gaussian(glm::vec3 ray_origin, glm::vec3 ray_dir, glm::mat4 imat, glm::vec4 base_col) {
    glm::vec4 o = imat * glm::vec4(ray_origin, 1.0);
    glm::vec4 rd = imat * glm::vec4(ray_dir, 0.0);
    rd = glm::normalize(rd);

    glm::vec3 front_intersection, back_intersection;
    ray_unit_sphere_intersection(o, rd, &front_intersection, &back_intersection);
    
    glm::vec4 color = glm::vec4(0.0, 0.0, 0.0, 0.0);

    for(int i = 0; i < NUM_RAY_STEPS; i++) {
        glm::vec3 p = front_intersection + (back_intersection - front_intersection) * (i / (float)NUM_RAY_STEPS);

        float len = glm::length(p) * 2.0;
        float gaussian = exp(-len * len);
        color += base_col * gaussian;
    }

    color /= (float)NUM_RAY_STEPS;
    color *= 2.0;

    if(glm::length(front_intersection - back_intersection) < 0.01) return glm::vec4(0.0, 0.0, 0.0, 0.0);
    return color;
}

__forceinline__ __device__ float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

__global__ void render(float *img_buffer, unsigned int width, unsigned int height, PerspectiveCamera *cam, GPUGaussian *gaussians, uint32_t n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    //assign each pixel a UV value -1 -> 1, left to right / bottom to top
    glm::vec2 uv = glm::vec2((float)x / width, (float)y / height);
    uv = uv * (float)2.0 - glm::vec2(1.0, 1.0);
    //aspect ratio correction
    uv.x *= (float)width / height;
    uv.y *= -1.0;

    glm::vec3 ray = cam->forward + cam->up * uv.y + cam->right * uv.x;
    glm::vec3 o = cam->origin;

    float4 color = make_float4(0.0, 0.0, 0.0, 0.0);

    glm::mat4 imat = glm::mat4(
        243.580994,26.130470,-50.983204,-0.000000, 
        55.845165,-435.518982,108.501625,0.000000,
        -14.838394,-29.281708,-108.986649,0.000000, 
        0.019985,-0.316502,-0.649367,1.000000
    );
    for(int i = 0; i < n; i++) {
        glm::vec4 gaussian_color = make_vec4(gaussians[i].color);
        gaussian_color.a = sigmoid(gaussian_color.a);
        gaussian_color.r *= -0.28209479177387814; gaussian_color.g *= -0.28209479177387814; gaussian_color.b *= -0.28209479177387814;
        glm::vec4 c = raymarch_gaussian(o, ray, make_mat4(gaussians[i].imat), gaussian_color);
        color.x += c.x * (1.0 - color.w);
        color.y += c.y * (1.0 - color.w);
        color.z += c.z * (1.0 - color.w);
        color.w += c.w * (1.0 - color.w);
    }
    
    int idx = y * width * 4 + x * 4;

    img_buffer[idx + 0] = color.x;
    img_buffer[idx + 1] = color.y;
    img_buffer[idx + 2] = color.z;
    img_buffer[idx + 3] = color.w; 
}