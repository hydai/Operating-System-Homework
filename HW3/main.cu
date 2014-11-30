#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "declaration.h"

extern __device__ __managed__ int PAGE_ENTRIES = 0;
extern __device__ __managed__ int PAGEFAULT = 0;
extern __device__ __managed__ uchar storage[STORAGE_SIZE];
extern __device__ __managed__ uchar results[STORAGE_SIZE];
extern __device__ __managed__ uchar input[STORAGE_SIZE];

extern __device__ void initPageTable(int entries);
extern int loadBinaryFile(char *fileName, uchar *input, int storageSize);
extern void writeBinaryFile(char *fileName, uchar *input, int storageSize);

extern __shared__ u32 pageTable[];

__global__ void mykernel(int inputSize) {
    __shared__ uchar data[PHYSICAL_MEM_SIZE];
    int ptEntries = PHYSICAL_MEM_SIZE/PAGE_SIZE;
    initPageTable(ptEntries);
    //####Gwrite/Gread code section start####

    //####Gwrite/Gread code section end####
}

int main() {
    int inputSize = loadBinaryFile(DATAFILE, input, STORAGE_SIZE);
    cudaSetDevice(2);
    mykernel<<<1, 1, 16384>>>(inputSize);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    writeBinaryFile(OUTPUTFILE, results, inputSize);
    printf("pagefault times = %d\n", PAGEFAULT);
    return 0;
}
