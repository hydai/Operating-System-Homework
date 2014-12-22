#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

// Define constants and data types
#define MAX_FILE_SIZE       1048576
#define STORAGE_SIZE        1085440
#define DATAFILE            "./data.bin"
#define OUTPUTFILE          "./snapshot.bin"
#define FILE_ENTRIES		1024
#define DNE					0xffffffff
typedef unsigned char uchar;
typedef uint32_t u32;
const int G_WRITE   = 1;
const int G_READ    = 2;
const int LS_S      = 3;
const int LS_D      = 4;
const int RM        = 5;
const int RM_RF     = 6;

// Declare variables
__device__ __managed__ uchar *volume;

// Function
// ******************************************************************
// Initialize
void init_volume() {
	for (int i = 0; i < STORAGE_SIZE; i++) {
		volume[i] = 0;
	}
}
// ******************************************************************

// ******************************************************************
// File I/O
int loadBinaryFile(const char *fileName, uchar *input, int fileSize) {
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

void writeBinaryFile(const char *fileName, uchar *input, int fileSize) {
    FILE *fptr = fopen(fileName, "wb");
    // Read data from input file
    fwrite(input, sizeof(unsigned char), fileSize, fptr);
	fclose(fptr);
}
// ******************************************************************

// ******************************************************************
// FS Helper function
__device__ u32 getFid(const char *name) {
	u32 fid = DNE;
	for (u32 i = 0; i < FILE_ENTRIES; i++) {
		if (isNameMatched(i, name)) {
			fid = i;
			break;
		}
	}
	if (fid == DNE) {
		fid = createNewFile(name);
	}
	return fid;
}
// ******************************************************************

// ******************************************************************
// FS Operation
__device__ u32 open(const char *name, int type) {
    printf("Open %s %d\n", name, type);
	u32 fid = getFid(name);
    return fid;
}

__device__ void write(uchar *src, int len, u32 fid) {
	updateFileSize(fid, len);
	updateFileModifyTime(fid);
	u32 entry = getFileDataAddress(fid);
	for (u32 i = 0; i < len; i++) {
		updateData(entry+i,src[i]);
	}
    printf("Write %s %d %d\n", src, len, fid);
}

__device__ void read(uchar *dst, int len, u32 fid) {
	u32 entry = getFileDataAddress(fid);
	for (u32 i = 0; i < len; i++) {
		dst[i] = getData(entry+i);
	}
    printf("Read %s %d %d\n", dst, len, fid);
}

__device__ void gsys(int type) {
	switch (type) {
		case LS_D:
			sortD();
			printD();
			break;
		case LS_S:
			sortS();
			printS();
			break;
		default:
			sortS();
			printS();
			break;
	}
    printf("Gsys %d\n", type);
}

__device__ void gsys(int type, const char *name) {
	switch (type) {
		case RM:
			deleteFileByName(name);
			break;
		default:
			deleteFileByName(name);
			break;
	}
    printf("Gsys %d %s\n", type, name);
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
	printf("B init\n");
    init_volume();

    uchar *input, *output;
    cudaMallocManaged(&input, MAX_FILE_SIZE);
    cudaMallocManaged(&output, MAX_FILE_SIZE);
    for (int i = 0; i < MAX_FILE_SIZE; i++) {
        output[i] = 0;
    }
    loadBinaryFile(DATAFILE, input, MAX_FILE_SIZE);
	printf("F init\n");

    cudaSetDevice(2);
    mykernel<<<1, 1>>>(input, output);
    cudaDeviceSynchronize();
    writeBinaryFile(OUTPUTFILE, output, MAX_FILE_SIZE);
    cudaDeviceReset();

    return 0;
}
