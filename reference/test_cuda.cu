#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices detected.\n");
    } else {
        printf("Number of CUDA-capable devices: %d\n", deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);

            printf("Device %d: %s\n", i, deviceProp.name);
            printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
            printf("  Total global memory: %lu mega bytes\n", deviceProp.totalGlobalMem / 1024 / 1024);
            printf("  Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
            printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
            printf("  Max threads dimensions: %d x %d x %d\n", 
                   deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
            printf("  Max grid size: %d x %d x %d\n", 
                   deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        }
    }

    return 0;
}
