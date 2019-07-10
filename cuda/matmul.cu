/*
   Copyright 2019 Cleuton Sampaio

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
/* 
    Matrix multiplication sample using CUDA 
    this sample works for squared matrices!
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

// CUDA Kernel function: 

__global__ void matmul(float *A, float* B, float *C, int size)
{

    // Row and Column indexes: 
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // Are they bellow the maximum?
    if (col < size && row < size) {
       float result = 0;
       for(int ix=0;ix<size;ix++) {
          result += A[row*size+ix]*B[ix*size+col];
       }
       C[row*size+col] = result;
    }

}

int main()
{
    // Matrices and constants
    int size = 3;
    int total = size*size;
    float cpu_A[] = {-1,2,4,0,5,3,6,2,1};
    float cpu_B[] = {3,0,2,3,4,5,4,7,2};
    float cpu_C[total];

    // Allocate device memory:
    float* gpu_A;
    float* gpu_B;
    float* gpu_C;
    int msize = total * sizeof(float);
    cudaMalloc((void**)&gpu_A, msize);
    cudaMemcpy(gpu_A,cpu_A,msize,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_B, msize);
    cudaMemcpy(gpu_B,cpu_B,msize,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_C,msize);

    // Blocks & grids:
    dim3 blocks(size,size);
    dim3 grid(1,1);

    // Call the kernel:
    matmul<<<grid,blocks>>>(gpu_A,gpu_B,gpu_C,size);

    // Get the result Matrix:
    cudaMemcpy(cpu_C,gpu_C,msize,cudaMemcpyDeviceToHost);
    std::cout << cpu_C[0] << '\t' << cpu_C[1] << '\t' << cpu_C[2] << std::endl
              << cpu_C[3] << '\t' << cpu_C[4] << '\t' << cpu_C[5] << std::endl
              << cpu_C[6] << '\t' << cpu_C[7] << '\t' << cpu_C[8] << std::endl;

    //Free device matrices
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    
}
