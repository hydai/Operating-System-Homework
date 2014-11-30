#include <inttypes.h>
#include "declaration.h"
extern __device__ __managed__ int PAGE_ENTRIES = 0;
extern __device__ __managed__ int PAGEFAULT = 0;
extern __device__ __managed__ uchar storage[STORAGE_SIZE];
extern __device__ __managed__ uchar results[STORAGE_SIZE];
extern __device__ __managed__ uchar input[STORAGE_SIZE];

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
