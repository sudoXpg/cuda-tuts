#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>

#define     N      50000000     

__global__ void prime_check(uint64_t *arr, uint64_t n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(uint64_t i = index; i<n; i+=stride){
        if( (arr[i]%2 ==0)    ||  (arr[i]%3 ==0)    ||  (arr[i]%5 ==0)    ||  (arr[i]%7 ==0)){
            arr[i]=0;
            continue;
        }
        for(uint64_t j = 2; j < sqrtf((float) i);j++){
            if(arr[i]%j==0){
                arr[i]=0;
                break;
            }
        }
    }
}


int main(void){

    uint64_t *h_arr;
    uint64_t *d_arr;

    h_arr = (uint64_t *) malloc(sizeof(uint64_t) * N);

    for(uint64_t i = 0;i<N;i++){
        h_arr[i] = i;
    }

    cudaMalloc((void **)&d_arr, sizeof(uint64_t) * N);
    cudaMemcpy(d_arr, h_arr, sizeof(uint64_t) * N, cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    prime_check<<<blocks_per_grid, threads_per_block>>>(d_arr, N);

    cudaMemcpy(h_arr, d_arr, sizeof(uint64_t) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // for(uint64_t i = 0;i<N;i++){
    //     if(h_arr[i]!=0){
    //         printf("%" PRIu64 ", ", h_arr[i]);
    //     }
    // }

    free(h_arr);
    cudaFree(d_arr);

    return 0;
}