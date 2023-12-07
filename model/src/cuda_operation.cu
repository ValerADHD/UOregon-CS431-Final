#ifdef RUN_PARALLEL

#include <iostream>
#include <vector>
#include <cuda.h>

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Mat = std::vector<std::vector<T>>;

// CUDA Kernel
__global__ void CudaTransposeKernel(double* d_matrix, double* d_inv_matrix, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        d_inv_matrix[j * n + i] = d_matrix[i * n + j];
    }
}

// Transpose function
Mat<double> transpose(Mat<double> matrix) {
    int n = matrix.size();

    double* d_matrix;
    double* d_tran_matrix;
    cudaMalloc(&d_matrix, n * n * sizeof(double));
    cudaMalloc(&d_tran_matrix, n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        cudaMemcpy(d_matrix + i*n, matrix[i].data(), n*sizeof(double), cudaMemcpyHostToDevice);
    }

    // Launch Kernel
    dim3 block(16, 16); 
    dim3 grid(1, 1);
    CudaTransposeKernel<<<grid, block>>>(d_matrix, d_tran_matrix, n);

    Mat<double> mat_tran(n, Vec<double>(n));
    // Copy memory back to host
    for (int i = 0; i < n; i++) {
        cudaMemcpy(mat_tran[i].data(), d_tran_matrix + i*n, n*sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_matrix);
    cudaFree(d_tran_matrix);

    return mat_tran;
}

#endif
