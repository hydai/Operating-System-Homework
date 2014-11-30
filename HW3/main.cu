#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

// Define constants and data types
#define PAGE_SIZE           32
#define PHYSICAL_MEM_SIZE   32768
#define STORAGE_SIZE        131072
#define DATAFILE            "./data.bin"
#define OUTPUTFILE          "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;
const uint32_t VALID    = 0 | 1;
const uint32_t INVALID  = 0;

// Declare variables
__device__ __managed__ int PAGE_ENTRIES = 0;
__device__ __managed__ int PAGEFAULT = 0;
__device__ __managed__ uchar storage[STORAGE_SIZE];
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];
extern __shared__ u32 pageTable[];
__device__ void initPageTable(int entries) {
    for (int i = 0; i < entries; i++) {
        pageTable[i] = INVALID;
    }
}

int loadBinaryFile(char *fileName, uchar *input, int storageSize) {
    FILE *fptr = fopen(fileName, "rb");
    // Get size
    fseek(fptr, 0, SEEK_END);
    int size = ftell(fptr);
    rewind(fptr);
    // Read data from input file
    fread(input, sizeof(unsigned char), size, fptr);
    if (storageSize < size) {
        printf("ERROR: Storage size is too small to store input data!\n");
    }
	fclose(fptr);
    return size;
}

void writeBinaryFile(char *fileName, uchar *input, int storageSize) {
    FILE *fptr = fopen(fileName, "wb");
    // Read data from input file
    fwrite(input, sizeof(unsigned char), storageSize, fptr);
	fclose(fptr);
}

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
