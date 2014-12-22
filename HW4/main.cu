#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

// Define constants and data types
#define MAX_FILE_SIZE       1048576
#define STORAGE_SIZE        1085440
#define DATAFILE            "./data.bin"
#define OUTPUTFILE          "./snapshot.bin"
#define FILE_ENTRIES		1024
#define META_SIZE			31
#define DATA_SIZE			1024
#define LS_S_OFFSET			1024
#define LS_D_OFFSET			3072
#define META_OFFSET			5120
#define SIZE_OFFSET			21
#define CREATET_OFFSET		SIZE_OFFSET+2
#define MODIFYT_OFFSET		CREATET_OFFSET+4
#define DATA_OFFSET			36864
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
__device__ __managed__ u32 currentTime = 0;
__device__ __managed__ u32 fileCt = 0;
__device__ __managed__ uchar *volume;
// ==================================================================
// How I use volume:
// V--0          V--3072           V--36864
// ------------------------------------------------------------------
// | Temp | LS_S | LS_D | Metadata |              Data              |
// ------------------------------------------------------------------
//        ^--1024       ^--5120
// Temp => For pending
// LS_S => Array for sorting by LS_S
// LS_D => Array for sorting by LS_D
// Metadata =>
//      Entry structure:
//      V--0   V--21  V--23     V--27
//      -----------------------------------
//      | Name | Size | CreateT | ModifyT |
//      -----------------------------------
//      Name    := 21 bytes, for 20 char and '\0'
//      Size    :=  2 bytes
//      CreateT :=  4 bytes
//      ModifyT :=  4 bytes
//      Total size = 31 bytes
// ==================================================================

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
__device__ inline int dstrlen(const char *str) {
	int len = 0;
	for (; str[len] != '\0'; len++);
	return len;
}
__device__ bool isNameMatched(int fid, const char *name) {
	bool isMatched = true;
	int len = dstrlen(name);
	for (int j = 0; j < len; j++) {
		if (volume[META_OFFSET+fid*META_SIZE+j] != name[j]) {
			isMatched = false;
			break;
		}
	}
	if (volume[META_OFFSET+fid*META_SIZE+len] != '\0')
		isMatched = false;
	return isMatched;
}
__device__ u32 createNewFile(const char *name) {
	u32 fid = fileCt++;
	int len = 0;
	for (; name[len] != '\0'; len++) {
		volume[META_OFFSET+fid*META_SIZE+len] = name[len];
	}
	volume[META_OFFSET+fid*META_SIZE+len] = '\0';
	volume[META_OFFSET+fid*META_SIZE+SIZE_OFFSET]   = 0;
	volume[META_OFFSET+fid*META_SIZE+SIZE_OFFSET+1] = 0;
	volume[META_OFFSET+fid*META_SIZE+CREATET_OFFSET]   = (currentTime & 0xFF000000) >> 24;
	volume[META_OFFSET+fid*META_SIZE+CREATET_OFFSET+1] = (currentTime & 0x00FF0000) >> 16;
	volume[META_OFFSET+fid*META_SIZE+CREATET_OFFSET+2] = (currentTime & 0x0000FF00) >> 8;
	volume[META_OFFSET+fid*META_SIZE+CREATET_OFFSET+3] = (currentTime & 0x000000FF);
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET]   = (currentTime & 0xFF000000) >> 24;
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET+1] = (currentTime & 0x00FF0000) >> 16;
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET+2] = (currentTime & 0x0000FF00) >> 8;
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET+3] = (currentTime & 0x000000FF);
	currentTime++;
	return fid;
}
__device__ inline void updateFileSize(int fid, int len) {
	volume[META_OFFSET+fid*META_SIZE+SIZE_OFFSET]   = (len & 0x0000FF00) >> 8;
	volume[META_OFFSET+fid*META_SIZE+SIZE_OFFSET+1] = (len & 0x000000FF);
}
__device__ inline void updateFileModifyTime(int fid) {
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET]   = (currentTime & 0xFF000000) >> 24;
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET+1] = (currentTime & 0x00FF0000) >> 16;
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET+2] = (currentTime & 0x0000FF00) >> 8;
	volume[META_OFFSET+fid*META_SIZE+MODIFYT_OFFSET+3] = (currentTime & 0x000000FF);
	currentTime++;
}
__device__ inline void updateData(int addr, uchar data) {
	volume[addr] = data;
}
__device__ inline u32 getData(int addr) {
	return volume[addr];
}
__device__ inline u32 getFileDataAddress(int fid) {
	return DATA_OFFSET+fid*DATA_SIZE;
}
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
__device__ void sortD() {
	for (int i = 0; i < fileCt; i++) {
		volume[LS_D_OFFSET+i*2] 	= (i & 0x0000FF00) >> 8;
		volume[LS_D_OFFSET+i*2+1] 	= i & 0x000000FF;
	}
}
__device__ void printD() {
	for (int i = 0; i < fileCt; i++) {
		int fid = ((int)(volume[LS_D_OFFSET+i*2]) << 8) | ((int)(volume[LS_D_OFFSET+i*2+1]));
		printf("%s\n", volume[META_OFFSET+fid*META_SIZE]);
	}
}
__device__ void sortS() {
	for (int i = 0; i < fileCt; i++) {
		volume[LS_S_OFFSET+i*2] 	= i & 0xFFFF0000;
		volume[LS_S_OFFSET+i*2+1] 	= i & 0x0000FFFF;
	}
}
__device__ void printS() {
	for (int i = 0; i < fileCt; i++) {
		int fid = ((int)(volume[LS_S_OFFSET+i*2]) << 8) | ((int)(volume[LS_S_OFFSET+i*2+1]));
		printf("%s\n", volume[META_OFFSET+fid*META_SIZE]);
	}
}
__device__ void deleteFileByName(const char *name) {
	int target = DNE;
	for (int i = 0; i < fileCt; i++) {
		if(isNameMatched(i, name)) {
			target = i;
			break;
		}
	}
	fileCt--;
	if (target != fileCt) {
		for (int i = 0; i < META_SIZE; i++) {
			volume[META_OFFSET+target*META_SIZE+i] = volume[META_OFFSET+fileCt*META_SIZE+i];
		}
		for (int i = 0; i < volume[META_OFFSET+fileCt*META_OFFSET+SIZE_OFFSET]; i++) {
			volume[DATA_OFFSET+target*DATA_SIZE+i] = volume[DATA_OFFSET+fileCt*DATA_SIZE+i];
		}
	}
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
