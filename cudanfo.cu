#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA-enabled devices found.\n");
        return 1;
    }
    
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        
        printf("Device %d: %s\n", device, deviceProp.name);
        printf("  CUDA Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Maximum Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Warp Size: %d\n", deviceProp.warpSize);
        printf("  Max Grid Dimensions: [%d, %d, %d]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Max Block Dimensions: [%d, %d, %d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }
    
    return 0;
}
