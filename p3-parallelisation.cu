#include <stdio.h>

#define N       5000000



/*
CUDA Thread and Block Organization:

    threadIdx.x: Unique thread index within a block (0 to blockDim.x - 1)
    blockIdx.x : Unique block index within the grid (0 to gridDim.x - 1)
    blockDim.x : Number of threads in a block
    gridDim.x  : Number of blocks in the grid

Example:
    Grid Structure (with 256 threads per block):
    
    Block 0 (blockIdx.x = 0):
    [ threadIdx.x = 0  1  2 ... 255 ]

    Block 1 (blockIdx.x = 1):
    [ threadIdx.x = 0  1  2 ... 255 ]

    Block 2 (blockIdx.x = 2):
    [ threadIdx.x = 0  1  2 ... 255 ]

    ...

    Block N/256 - 1 (last block):
    [ threadIdx.x = 0  1  2 ... 255 ]

Global thread index calculation:
    index = threadIdx.x + blockIdx.x * blockDim.x
*/


__global__ void vector_add_gpu(float *res, float *a, float *b, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;                      // index + ( block * threads in a block )
    int stride = blockDim.x * gridDim.x;                                    // The stride is the number of elements a thread must skip to reach the next set of elements it should process. 

    for(int i = index; i < n; i += stride){
        res[i] = a[i] + b[i];
    }
}



int main(void){
    float *h_a, *h_b, *h_res;
    float *d_a, *d_b, *d_res;


    // host CPU malloc
    h_a = (float *)malloc(  sizeof(float) * N);
    h_b = (float *)malloc(  sizeof(float) * N);
    h_res = (float *)malloc(sizeof(float) * N);

    for (int i=0;i<N;i++){
        h_a[i] = (float)rand() / (float)RAND_MAX;;
        h_b[i] = (float)rand() / (float)RAND_MAX;;
    }


    // device GPU malloc
    cudaMalloc((void **)&d_a, (sizeof(float) * N) );
    cudaMalloc((void **)&d_b, (sizeof(float) * N) );
    cudaMalloc((void **)&d_res, (sizeof(float) * N) );


    // copy to device
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, sizeof(float) * N, cudaMemcpyHostToDevice);

    // copy back to device
    cudaMemcpy(h_res, d_res, sizeof(float) * N, cudaMemcpyDeviceToHost);


    //The syntax of kernel execution configuration is as follows            <<< M , T >>>
    // Which indicate that a kernel launches with a grid of M thread blocks. Each thread block has T parallel threads. 

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add_gpu<<<blocks_per_grid, threads_per_block>>>(d_res, d_a, d_b, N);

    /*
        threadIdx.x contains the index of the thread within the block 
        blockDim.x contains the size of thread block (number of threads in the thread block).
    */

    cudaDeviceSynchronize();


    free(h_a);
    free(h_b);
    free(h_res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;
}