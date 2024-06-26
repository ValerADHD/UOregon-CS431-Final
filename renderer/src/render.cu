#include <stdio.h>
#include <stdlib.h>

#include "render.h"
#include "model.h"

#define GLM_FORCE_CUDA
#include "glm/mat3x3.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"
#include "glm/vec3.hpp"
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

//#define TEST_FRUSTUM_PLANE_GENERATION
//#define TEST_FRUSTUM_PLANE_GENERATION_VERBOSE

void generate_frustum_planes(PerspectiveCamera *cam, float numTilesX, float numTilesY, float aspectRatio, float **hPlanes, float **vPlanes) {
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
    for(int i = 0; i <= numBinsY; i++) {
        float uvY = min(1.0, (float)i / numTilesY);
        uvY = uvY * 2.0 - 1.0;
        uvY *= -1;
        
        glm::vec3 ray1 = glm::normalize(c->forward + c->up * uvY + c->right);
        glm::vec3 ray2 = glm::normalize(c->forward + c->up * uvY - c->right);
        glm::vec3 planeNorm = glm::cross(ray1, ray2);

        horizontalFrustumPlanes[i * 3 + 0] = planeNorm.x;
        horizontalFrustumPlanes[i * 3 + 1] = planeNorm.y;
        horizontalFrustumPlanes[i * 3 + 2] = planeNorm.z;
    }
    //generate vertical frustum planes, left to right
    for(int i = 0; i <= numBinsX; i++) {
        float uvX = min(1.0, (float)i / numTilesX);
        uvX = uvX * 2.0 - 1.0;
        uvX *= aspectRatio;
        
        glm::vec3 ray1 = glm::normalize(c->forward + c->up + c->right * uvX);
        glm::vec3 ray2 = glm::normalize(c->forward - c->up + c->right * uvX);
        glm::vec3 planeNorm = glm::cross(ray1, ray2);

        verticalFrustumPlanes[i * 3 + 0] = planeNorm.x;
        verticalFrustumPlanes[i * 3 + 1] = planeNorm.y;
        verticalFrustumPlanes[i * 3 + 2] = planeNorm.z;
    }

#ifdef TEST_FRUSTUM_PLANE_GENERATION
    int horizontal_tests_passed = 0;
    int vertical_tests_passed = 0;
    for(int x = 0; x < numBinsX; x++) {
        for(int y = 0; y < numBinsY; y++) {
            //generate a point guaranteed to be inside the frustum for tile[x][y]
            float uvXMin = min(1.0, (float)x / numTilesX),
                uvXMax = min(1.0, (float)(x + 1) / numTilesX),
                uvYMin = min(1.0, (float)y / numTilesY),
                uvYMax = min(1.0, (float)(y + 1) / numTilesY);
            float uvX = (uvXMin + uvXMax) / 2.0; uvX = uvX * 2.0 - 1.0;
            float uvY = (uvYMin + uvYMax) / 2.0; uvY = uvY * 2.0 - 1.0;
            glm::vec3 point = c->origin + c->forward + c->right * uvX + c->up * uvY;

            glm::vec3 leftPlaneNormal = glm::vec3(verticalFrustumPlanes[x * 3 + 0], verticalFrustumPlanes[x * 3 + 1], verticalFrustumPlanes[x * 3 + 2]);
            glm::vec3 rightPlaneNormal = -glm::vec3(verticalFrustumPlanes[x * 3 + 3], verticalFrustumPlanes[x * 3 + 4], verticalFrustumPlanes[x * 3 + 5]);
            glm::vec3 bottomPlaneNormal = glm::vec3(horizontalFrustumPlanes[y * 3 + 0], horizontalFrustumPlanes[y * 3 + 1], horizontalFrustumPlanes[y * 3 + 2]);
            glm::vec3 topPlaneNormal = -glm::vec3(horizontalFrustumPlanes[y * 3 + 3], horizontalFrustumPlanes[y * 3 + 4], horizontalFrustumPlanes[y * 3 + 5]);
        

            float leftPlaneDist = glm::dot(-leftPlaneNormal, c->origin - point);
            float rightPlaneDist = glm::dot(-rightPlaneNormal, c->origin - point);
            float bottomPlaneDist = glm::dot(-bottomPlaneNormal, c->origin - point);
            float topPlaneDist = glm::dot(-topPlaneNormal, c->origin - point);

            if(leftPlaneDist > 0 && rightPlaneDist > 0) {
                vertical_tests_passed++;
            }
            if(topPlaneDist > 0 && bottomPlaneDist > 0) {
                horizontal_tests_passed++;
            }
#ifdef TEST_FRUSTUM_PLANE_GENERATION_VERBOSE
            printf("Testing frustums for tile x: %d y: %d\n", x, y);
            printf("\tU: %f V: %f\n", uvX, uvY);
            printf("\tPoint ended up at x: %f y: %f z: %f\n", point.x, point.y, point.z);
            printf("\tFrustum Planes: (all with origin x: %f y: %f z: %f)\n", c->origin.x, c->origin.y, c->origin.z);
            printf("\t\tLeft: nx: %f ny: %f nz: %f\n", leftPlaneNormal.x, leftPlaneNormal.y, leftPlaneNormal.z);
            printf("\t\t\tDistance from left plane: %f\n", leftPlaneDist);
            printf("\t\tRight: nx: %f ny: %f nz: %f\n", rightPlaneNormal.x, rightPlaneNormal.y, rightPlaneNormal.z);
            printf("\t\t\tDistance from right plane: %f\n", rightPlaneDist);
            printf("\t\tBottom: nx: %f ny: %f nz: %f\n", bottomPlaneNormal.x, bottomPlaneNormal.y, bottomPlaneNormal.z);
            printf("\t\t\tDistance from bottom plane: %f\n", bottomPlaneDist);
            printf("\t\tTop: nx: %f ny: %f nz: %f\n", topPlaneNormal.x, topPlaneNormal.y, topPlaneNormal.z);
            printf("\t\t\tDistance from top plane: %f\n", topPlaneDist);
#endif
        }
    }
    if(horizontal_tests_passed == numBinsX * numBinsY && vertical_tests_passed == numBinsX * numBinsY) {
        fprintf(stderr, "Success! The generated frustum planes worked for all %d tests\n", numBinsX * numBinsY);
    } else {
        fprintf(stderr, "Failure! The generated frustum planes failed on %d horizontal tests and %d vertical tests\n", numBinsX * numBinsY - horizontal_tests_passed, numBinsX * numBinsY - vertical_tests_passed);
    }
#endif

    stat = cudaMalloc(hPlanes, sizeof(float) * 3 * (numBinsY + 1));
    if(stat != cudaSuccess) fprintf(stderr, "Unable to allocate memory for frustum planes: %s\n", cudaGetErrorString(stat));
    stat = cudaMalloc(vPlanes, sizeof(float) * 3 * (numBinsX + 1));
    if(stat != cudaSuccess) fprintf(stderr, "Unable to allocate memory for frustum planes: %s\n", cudaGetErrorString(stat));
    stat = cudaMemcpy(*hPlanes, horizontalFrustumPlanes, sizeof(float) * 3 * (numBinsY + 1), cudaMemcpyHostToDevice);
    if(stat != cudaSuccess) fprintf(stderr, "Unable to upload horizontal frustum planes to GPU: %s\n", cudaGetErrorString(stat));
    stat = cudaMemcpy(*vPlanes, verticalFrustumPlanes, sizeof(float) * 3 * (numBinsX + 1), cudaMemcpyHostToDevice);
    if(stat != cudaSuccess) fprintf(stderr, "Unable to upload vertical frustum planes to GPU: %s\n", cudaGetErrorString(stat));
    
    free(horizontalFrustumPlanes);
    free(verticalFrustumPlanes);
}


#define MAX_BIN_SIZE 12288

//#define PRINT_BIN_SIZES

void render_call_handler(float *img_buffer, unsigned int width, unsigned int height, PerspectiveCamera *cam, GPUModel *gm) {
    printf("Beginning render call!\n");
    cudaEvent_t begin[3], end[3];
    float timings[3];
    for(int i = 0; i < 3; i++) { cudaEventCreate(&(begin[i])); cudaEventCreate(&(end[i])); }
    
    float numTilesX = width / 16.0, numTilesY = height / 16.0;
    float aspectRatio = (float)width / height;
    uint32_t numBinsX = ceil(numTilesX), numBinsY = ceil(numTilesY);

    float *horizontalFrustumPlanes, *verticalFrustumPlanes;
    generate_frustum_planes(cam, numTilesX, numTilesY, aspectRatio, &horizontalFrustumPlanes, &verticalFrustumPlanes);

    uint32_t *bins, *binIdxs;
    cudaError_t stat = cudaMalloc(&bins, sizeof(uint32_t) * numBinsX * numBinsY * MAX_BIN_SIZE);
    if(stat != cudaSuccess) fprintf(stderr, "Unable to create GPU bin array: %s\n", cudaGetErrorString(stat));
    stat = cudaMalloc(&binIdxs, sizeof(uint32_t) * numBinsX * numBinsY);
    if(stat != cudaSuccess) fprintf(stderr, "Unable to create GPU bin index array: %s\n", cudaGetErrorString(stat));

    cudaEventRecord(begin[0]);
    binGaussians<<<(gm->data_len + 1024 - 1) / 1024, 1024>>>(cam, gm->data, gm->data_len, numTilesX, numTilesY, horizontalFrustumPlanes, verticalFrustumPlanes, bins, binIdxs);
    cudaEventRecord(end[0]);
    
#ifdef PRINT_BIN_SIZES
    // uint32_t *debugBinSizes = (uint32_t *)malloc(sizeof(uint32_t) * numBinsX * numBinsY);
    // stat = cudaMemcpy(debugBinSizes, binIdxs, sizeof(uint32_t) * numBinsX * numBinsY, cudaMemcpyDeviceToHost);
    // if(stat != cudaSuccess) fprintf(stderr, "Unable to download bin size array: %s\n", cudaGetErrorString(stat));
    // for(int y = 0; y < numBinsY; y++) for(int x = 0; x < numBinsX; x++) {
    //     printf("Bin X: %d Y: %d has %d elements\n", x, y, debugBinSizes[y * numBinsX + x]);
    // }
    // free(debugBinSizes);
#endif

    dim3 blockDim;
    blockDim.x = 16; blockDim.y = 16; blockDim.z = 1;

    dim3 numBlocks;
    numBlocks.x = (width + blockDim.x - 1) / blockDim.x;
    numBlocks.y = (height + blockDim.y - 1) / blockDim.y;
    numBlocks.z = 1;

    cudaEventRecord(begin[1]);
    sortBins<<<numBlocks, 1024>>>(cam, gm->data, gm->data_len, numTilesX, numTilesY, bins, binIdxs);
    cudaEventRecord(end[1]);

    cudaEventRecord(begin[2]);
    render<<<numBlocks, blockDim>>>(
        img_buffer, width, height, 
        cam, 
        gm->data, gm->data_len,
        numBinsX, numBinsY,
        bins, binIdxs
    );
    cudaEventRecord(end[2]);

    cudaDeviceSynchronize();

    for(int i = 0; i < 3; i++)
        cudaEventElapsedTime(&(timings[i]), begin[i], end[i]);

    printf("Render call done! Timings are:\n");
    printf("\tTile binning of %d Gaussians: %fms\n", gm->data_len, timings[0]);
    printf("\tTile sorting (bin size %d): %fms\n", MAX_BIN_SIZE, timings[1]);
    printf("\tRasterization at %dx%d: %fms\n", width, height, timings[2]);
    printf("\tTotal time: %fms\n", timings[0] + timings[1] + timings[2]);
}

__forceinline__ __device__ glm::vec3 inverse_transform_vec(glm::vec3 v, Gaussian *g) {
    glm::vec3 r = v;
    r -= g->mean;
    r = g->rot * r * glm::inverse(g->rot);
    r = glm::vec3(r.x / g->scale.x, r.y / g->scale.y, r.z / g->scale.z);
    return r;
}

__forceinline__ __device__ glm::vec3 inverse_transform_normal(glm::vec3 v, Gaussian *g) {
    glm::vec3 r = v;
    r = g->rot * r * glm::inverse(g->rot);
    r = glm::vec3(r.x * g->scale.x, r.y * g->scale.y, r.z * g->scale.z);
    return glm::normalize(r);
}

__forceinline__ __device__ float originDistanceToPlane(glm::vec3 planeOrigin, glm::vec3 planeNormal) {
    return glm::dot(-planeNormal, planeOrigin);
}

__global__ void binGaussians(
    PerspectiveCamera *cam, 
    Gaussian *gaussians, uint32_t n,
    float numTilesX, float numTilesY, //number of tiles for the whole screen. Can be non-integer if tiles are partially off-screen
    float *horizontalFrustumPlanes, float *verticalFrustumPlanes,
    uint32_t *bins, //actual bins, uint32_t[numBinsX][numBinsY][MAX_BIN_SIZE]
    uint32_t *binIndexes //index to the next open element in each bin, uint32_t[numBinsX][numBinsY]
) {
    uint32_t numBinsX = ceil(numTilesX), numBinsY = ceil(numTilesY);
    
    float far = 100.0;
    float near = 0.01;

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        Gaussian *g = &(gaussians[i]);

        //transform the origin into the gaussian's model space
        glm::vec3 frustumPlaneOrigin = inverse_transform_vec(cam->origin, g);
        glm::vec3 farPlaneOrigin = inverse_transform_vec(cam->origin + cam->forward * far, g);
        glm::vec3 farPlaneNormal = inverse_transform_normal(-cam->forward, g);
        glm::vec3 nearPlaneOrigin = inverse_transform_vec(cam->origin + cam->forward * near, g);
        glm::vec3 nearPlaneNormal = inverse_transform_normal(cam->forward, g);

        float farPlaneDist = originDistanceToPlane(farPlaneOrigin, farPlaneNormal);
        float nearPlaneDist = originDistanceToPlane(nearPlaneOrigin, nearPlaneNormal);

        if(farPlaneDist < 0 || nearPlaneDist < 0) continue;

        //loop through each bin, determine if the gaussian is inside that frustum
        for(int y0 = 0; y0 < numBinsY; y0++) {
            for(int x0 = 0; x0 < numBinsX; x0++) {
                //avoid all of the threads working on the same frustum at a time
                int x = x0 + tid; x %= numBinsX;
                int y = y0 + tid; y %= numBinsY;
                //get all of the planes for the edges of this tile's frustum
                glm::vec3 leftPlaneNormal = glm::vec3(verticalFrustumPlanes[x * 3 + 0], verticalFrustumPlanes[x * 3 + 1], verticalFrustumPlanes[x * 3 + 2]);
                glm::vec3 rightPlaneNormal = -glm::vec3(verticalFrustumPlanes[x * 3 + 3], verticalFrustumPlanes[x * 3 + 4], verticalFrustumPlanes[x * 3 + 5]);
                glm::vec3 bottomPlaneNormal = glm::vec3(horizontalFrustumPlanes[y * 3 + 0], horizontalFrustumPlanes[y * 3 + 1], horizontalFrustumPlanes[y * 3 + 2]);
                glm::vec3 topPlaneNormal = -glm::vec3(horizontalFrustumPlanes[y * 3 + 3], horizontalFrustumPlanes[y * 3 + 4], horizontalFrustumPlanes[y * 3 + 5]);
                
                //transform them into the gaussian's model space, so we can simply test against the unit sphere
                leftPlaneNormal = inverse_transform_normal(leftPlaneNormal, g);
                rightPlaneNormal = inverse_transform_normal(rightPlaneNormal, g);
                bottomPlaneNormal = inverse_transform_normal(bottomPlaneNormal, g);
                topPlaneNormal = inverse_transform_normal(topPlaneNormal, g);
                //get the signed distance from the origin to the plane
                float leftPlaneDist = originDistanceToPlane(frustumPlaneOrigin, leftPlaneNormal);
                float rightPlaneDist = originDistanceToPlane(frustumPlaneOrigin, rightPlaneNormal);
                float bottomPlaneDist = originDistanceToPlane(frustumPlaneOrigin, bottomPlaneNormal);
                float topPlaneDist = originDistanceToPlane(frustumPlaneOrigin, topPlaneNormal);

                //if the unit sphere is < 1 (absolute) distance unit away, it intersects the plane
                //if the signed distance is greater than 0, it's on the forward (inside frustum) side of the plane
                //so if the signed distance > -1 (for all frustum planes), at least part of the unit sphere intersects with
                //the frustum
                if(leftPlaneDist > -1 && rightPlaneDist > -1 && topPlaneDist > -1 && bottomPlaneDist > -1) {
                    //inside frustum
                    //if(binIndexes[y * numBinsX + x] < MAX_BIN_SIZE - 1) { 
                        uint32_t binIdx = atomicAdd(&(binIndexes[y * numBinsX + x]), 1);
                        if(binIdx < MAX_BIN_SIZE) bins[(y * numBinsX + x) * MAX_BIN_SIZE + binIdx] = i;
                    //}
                }
            }
        }
    }
}

__global__ void sortBins(
    PerspectiveCamera *cam,
    Gaussian *gaussians, uint32_t n,
    float numTilesX, float numTilesY,
    uint32_t *bins, uint32_t *binSizes
) {
    uint32_t numBinsX = ceil(numTilesX), numBinsY = ceil(numTilesY);

    //each thread block gets their own bin
    uint32_t binIdxX = blockIdx.x, binIdxY = numBinsY - blockIdx.y;
    uint32_t *block_bin = &(bins[(binIdxY * numBinsX + binIdxX) * MAX_BIN_SIZE]);
    uint32_t block_bin_size = min(binSizes[binIdxY * numBinsX + binIdxX], MAX_BIN_SIZE);

    __shared__ float keys[MAX_BIN_SIZE];

    uint32_t tid = threadIdx.x;
    for(int i = tid; i < block_bin_size; i += blockDim.x) {
        keys[i] = glm::dot(cam->forward, gaussians[i].mean - cam->origin);
    }
    
    //even odd sort
    for(int i = 0; i < block_bin_size; i++) {
        int parity = i & 1;
        for(int j = tid * 2 + parity; j < block_bin_size - 1; j += blockDim.x * 2) {
            if(keys[j] > keys[j + 1]) {
                float tmp = keys[j];
                keys[j] = keys[j + 1];
                keys[j + 1] = tmp;
                uint32_t t = block_bin[j];
                block_bin[j] = block_bin[j + 1];
                block_bin[j + 1] = t;
            }
        }
        __syncthreads();
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

__forceinline__ __device__ float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

#define NUM_RAY_STEPS 3
__device__ glm::vec4 raymarch_gaussian(glm::vec3 ray_origin, glm::vec3 ray_dir, Gaussian *g) {
    glm::vec3 o = inverse_transform_vec(ray_origin, g);
    glm::vec3 rd = inverse_transform_vec(ray_origin + ray_dir, g) - o;
    rd = glm::normalize(rd);

    glm::vec3 front_intersection, back_intersection;
    ray_unit_sphere_intersection(o, rd, &front_intersection, &back_intersection);
    
    glm::vec4 color = glm::vec4(0.0, 0.0, 0.0, 0.0);

    glm::vec4 base_col = glm::vec4(g->color, g->alpha);
    base_col.a = sigmoid(base_col.a);
    base_col.r *= 0.28209479177387814; base_col.g *= 0.28209479177387814; base_col.b *= 0.28209479177387814;

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

__global__ void render(
    float *img_buffer, unsigned int width, unsigned int height, 
    PerspectiveCamera *cam, 
    Gaussian *gaussians, uint32_t n,
    uint32_t numBinsX, uint32_t numBinsY,
    uint32_t *bins, uint32_t *bin_sizes
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    //each thread block gets their own bin
    uint32_t binIdxX = blockIdx.x, binIdxY = blockIdx.y;
    uint32_t *block_bin = &(bins[(binIdxY * numBinsX + binIdxX) * MAX_BIN_SIZE]);
    uint32_t block_bin_size = min(bin_sizes[binIdxY * numBinsX + binIdxX], MAX_BIN_SIZE);

    //assign each pixel a UV value -1 -> 1, left to right / bottom to top
    glm::vec2 uv = glm::vec2((float)x / width, (float)y / height);
    uv = uv * (float)2.0 - glm::vec2(1.0, 1.0);
    //aspect ratio correction
    uv.x *= (float)width / height;
    uv.y *= -1.0;

    glm::vec3 ray = cam->forward + cam->up * uv.y + cam->right * uv.x;
    glm::vec3 o = cam->origin;

    float4 color = make_float4(0.0, 0.0, 0.0, 1.0);

    for(int i = 0; i < block_bin_size; i++) {
        uint32_t idx = block_bin[i];
        Gaussian g = gaussians[idx];
        glm::vec4 c = raymarch_gaussian(o, ray, &g);
        color.x += c.x * c.w * color.w;
        color.y += c.y * c.w * color.w;
        color.z += c.z * c.w * color.w;
        color.w *= (1.0 - c.w);
    }
    
    color.x += 0.5; color.y += 0.5; color.z += 0.5;

    int idx = y * width * 4 + x * 4;

    img_buffer[idx + 0] = color.x;
    img_buffer[idx + 1] = color.y;
    img_buffer[idx + 2] = color.z;
    img_buffer[idx + 3] = color.w; 
}