/*
    Compile instructions:
    nvcc p1-helloworld.cu -o p1-helloworld
*/


#include <stdio.h>

// The __global__ specifier indicates a function that runs on device (GPU)
// function on GPU are also known as "kernels"

__global__ void cuda_hello(void){
    printf("hello from the gpu\n");
}

int main(void){

    // When a kernel is called, its execution configuration is provided through <<<...>>>
    // this is called "kernel launch"

    cuda_hello<<<1,1>>>();

    // Ensures kernel execution completes before the program terminates.
    cudaDeviceSynchronize();

    return 0;
}