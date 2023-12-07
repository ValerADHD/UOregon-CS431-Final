#include <stdio.h>
#include <stdlib.h>

#include "render.h"
#include "model.h"
#include "prefix.h"


void checkCudaErrors() {
    cudaError_t stat = cudaGetLastError();
    fprintf(stderr, "Cuda Error: %s\n", cudaGetErrorString(stat));
}

__global__ void gpu_add_aux(uint32_t *initial_arr, uint32_t *aux_arr, uint32_t init_arr_size, uint32_t aux_arr_size) 
{
    // Load data into shared memory:
    int blockidx = blockIdx.x;
    int blocksize = blockDim.x;
    int tid = threadIdx.x;

    // extern __shared__ uint32_t sdata[];
    int global_idx;
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        global_idx = i * blocksize + ITEMS_PROCESSED * (blockidx + 1) + tid;
        if(global_idx < init_arr_size) {
            initial_arr[global_idx] += aux_arr[blockidx];
        }
    }
}

__global__ void gpu_prefix_part(uint32_t *initial_arr, uint32_t *aux_arr, uint32_t n)
{
    // Load data into shared memory:
    int blockidx = blockIdx.x;
    int blocksize = blockDim.x;
    int tid = threadIdx.x;

    extern __shared__ uint32_t sdata[];
    int global_idx;
    int local_idx;
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        local_idx = i * blocksize + tid;
        global_idx = blocksize * blockidx + local_idx;
        if(global_idx < n) {
            sdata[local_idx] = initial_arr[global_idx];
        }
        else {
            sdata[local_idx] = 0;
        }
    }
    __syncthreads();


    // Indexing for up-sweep and down-sweep based on wikipedia graphics for the algorithm
    // https://en.m.wikipedia.org/wiki/Prefix_sum

    // Upsweep
    int shift;
    for(int i = 0; i < PREFIX_ITERS; i++) {
        // shift is memory stride length
        shift = (1 << i);
        int blockshift = 0;
        local_idx = (tid + 1) * shift * 2 - 1;
        if(shift*2 < ITEMS_PER_THREAD) {
            for(int j = 0; j < ITEMS_PER_THREAD; j++) {
                if(local_idx < blocksize) {
                    sdata[local_idx + blockshift] += sdata[local_idx + blockshift - shift];
                }
                blockshift += blocksize;
            }
        }
        else {
            if(local_idx < ITEMS_PROCESSED) {
                sdata[local_idx] += sdata[local_idx - shift];
            }
        }
        __syncthreads();
    }
    // Down-sweep
    for(int i = PREFIX_ITERS - 2; i >= 0; i--) {
        // shift is memory stride length
        shift = (1 << i);
        local_idx = (tid + 1) * shift * 2 + shift - 1;
        int blockshift = 0;
        if(shift*2 < ITEMS_PER_THREAD) {
            for(int j = 0; j < ITEMS_PER_THREAD; j++) {
                if(local_idx < blocksize) {
                    sdata[local_idx + blockshift] += sdata[local_idx + blockshift - shift];
                }
                blockshift += blocksize;
            }
        }
        else {
            if(local_idx < ITEMS_PROCESSED) {
                sdata[local_idx] += sdata[local_idx - shift];
            }
        }
        __syncthreads();
    }
    
    // Copy data back to GPU memory from shared (cache)
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        local_idx = i * blocksize + tid;
        global_idx = ITEMS_PROCESSED * blockidx + local_idx;
        if(global_idx < n) {
            initial_arr[global_idx] = sdata[local_idx];
        }
    }

    // Copy last piece of data to auxiliary array
    if(tid == (blocksize - 1)) {
        if(ITEMS_PROCESSED * (blockidx + 1) < n) {
            aux_arr[blockidx] = sdata[ITEMS_PROCESSED - 1];
        }
    }
}

void in_place_gpu_prefix(uint32_t *dev_arr, uint32_t n)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    uint32_t *aux_arr = nullptr;
    unsigned int blocks = (n - 1) / (ITEMS_PROCESSED) + 1; 
    unsigned int threads = BLOCK_SIZE; 
    unsigned int shared = sizeof(uint32_t) * ITEMS_PROCESSED;

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    cudaEventRecord(start, 0);
    checkCudaErrors();
    uint32_t aux_arr_size = blocks - 1;
    if(aux_arr_size > 0) {
        cudaMalloc((void **) &aux_arr, sizeof(uint32_t) * n / ITEMS_PROCESSED);
    }
    // printf("running gpu prefix part with %d blocks\n", blocks);
    fflush(stdout);
    gpu_prefix_part<<<dimGrid, dimBlock, shared>>>(dev_arr, aux_arr, n);
    cudaDeviceSynchronize();
    if(aux_arr_size) {
        // printf("running nested gpu prefix\n");
        fflush(stdout);
        in_place_gpu_prefix(aux_arr, aux_arr_size);
        cudaDeviceSynchronize();
        
        dim3 dimAuxGrid(aux_arr_size, 1, 1);
        dim3 dimAuxBlock(BLOCK_SIZE, 1, 1);
        gpu_add_aux<<<dimAuxGrid, dimAuxBlock>>>(dev_arr, aux_arr, n, aux_arr_size);
        cudaDeviceSynchronize();

        cudaFree(aux_arr);
    }
    cudaEventRecord(stop, 0);
    checkCudaErrors();
    cudaEventSynchronize(stop);
    checkCudaErrors();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    checkCudaErrors();
    // printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3));


}
