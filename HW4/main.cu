#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

// Define constants and data types
#define MAX_FILE_SIZE       1048576
#define STORAGE_SIZE        1085440
#define DATAFILE            "./data.bin"
#define OUTPUTFILE          "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;
const int G_WRITE   = 1;
const int G_READ    = 2;
const int LS_S      = 3;
const int LS_D      = 4;
const int RM        = 5;
const int RM_RF     = 6;

// Declare variables
__device__ /*__managed__*/ uchar *volume;

// Function
// ******************************************************************
// Initialize
void init_volume() {
    memset(volume, 0, STORAGE_SIZE*sizeof(uchar));
}

// ******************************************************************

// ******************************************************************
// File I/O
int loadBinaryFile(char *fileName, uchar *input, int fileSize) {
    FILE *fptr = fopen(fileName, "rb");
    // Get size
    fseek(fptr, 0, SEEK_END);
    int size = ftell(fptr);
    rewind(fptr);
    // Read data from input file
    fread(input, sizeof(unsigned char), size, fptr);
    if (fileSize < size) {
        printf("ERROR: Input size is too small to store input data!\n");
    }
	fclose(fptr);
    return size;
}

void writeBinaryFile(char *fileName, uchar *input, int fileSize) {
    FILE *fptr = fopen(fileName, "wb");
    // Read data from input file
    fwrite(input, sizeof(unsigned char), fileSize, fptr);
	fclose(fptr);
}
// ******************************************************************

// ******************************************************************
// FS Operation
__device__ u32 open(char *name, int type) {
    u32 fp = 0;
    return fp;
}

__device__ void write(uchar *src, int len, u32 fp) {
    // Not implement
}

__device__ void read(uchar *dst, int len, u32 fp) {
    // Not implement
}

__device__ void gsys(int type) {
    // Not implement
}

__device__ void gsys(int type, char *name) {
    // Not implement
}

// ******************************************************************

// ******************************************************************
// Kernel function
__global__ void mykernel(uchar *input, uchar *output) {
    //####kernel start####
    u32 fp = open("t.txt\0", G_WRITE);
    write(input, 64, fp);
    fp = open("b.txt\0", G_WRITE);
    write(input+32, 32, fp);
    fp = open("t.txt\0", G_WRITE);
    write(input+32, 32, fp);
    read(output, 32, fp);
    gsys(LS_D);
    gsys(LS_S);
    fp = open("b.txt\0", G_WRITE);
    write(input+64, 12, fp);
    gsys(LS_S);
    gsys(LS_D);
    gsys(RM, "t.txt\0");
    gsys(LS_S);
    //####kernel end####
}
// ******************************************************************

int main() {
    cudaMallocManaged(&volume, STORAGE_SIZE);
    init_volume();

    uchar *input, *output;
    cudaMallocManaged(&input, MAX_FILE_SIZE);
    cudaMallocManaged(&output, MAX_FILE_SIZE);
    loadBinaryFile(DATAFILE, input, MAX_FILE_SIZE);

    cudaSetDevice(2);
    mykernel<<<1, 1>>>(input, output);
    cudaDeviceSynchronize();
    writeBinaryFile(OUTPUTFILE, output, MAX_FILE_SIZE);
    cudaDeviceReset();

    return 0;
}
