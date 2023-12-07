#ifndef PREFIX_H
#define PREFIX_H

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#define ITEMS_PER_THREAD 1
#define BLOCK_SIZE 1024
#define ITEMS_PROCESSED (ITEMS_PER_THREAD * BLOCK_SIZE)
#define PREFIX_ITERS 12

__global__ void gpu_add_aux(uint32_t *initial_arr, uint32_t *aux_arr, uint32_t init_arr_size, uint32_t aux_arr_size);

__global__ void gpu_prefix_part(uint32_t *initial_arr, uint32_t *aux_arr, uint32_t n);

void in_place_gpu_prefix(uint32_t *dev_arr, uint32_t n);

void checkCudaErrors();

#endif