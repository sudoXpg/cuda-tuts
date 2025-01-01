/*
    Compile instructions:
    nvcc -g  p2-vectoraddition.cu -o p2-vectoraddition

    To measure performance:
    nvprof ./p2-vectoraddition
*/


#include <stdio.h>

#define N 1000000

void vector_add_cpu(float *res, float *a, float *b, int n){
    for(int i=0;i<n;i++){
        res[i] = a[i] + b[i];
    }
}



__global__ void vector_add_gpu(float *res, float *a, float *b, int n){
    for(int i=0;i<n;i++){
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
    cudaMemcpy(d_a, h_a, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, N, cudaMemcpyHostToDevice);



    vector_add_gpu<<<1,1>>>(d_res, d_a, d_b, N);
    cudaDeviceSynchronize();

    //vector_add_cpu(h_res, h_a, h_b, N);


    free(h_a);
    free(h_b);
    free(h_res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;
}