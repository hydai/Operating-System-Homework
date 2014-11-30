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
const uint32_t VALID    		= 0 | 1;
const uint32_t INVALID			= 0;
const uint32_t PAGENUMBERMASK	= 0x00003FFE;
const uint32_t LASTTIMEMASK		= 0xFFFFC000;
const uint32_t DNE				= 0xFFFFFFFF;

// Declare variables
__device__ __managed__ int PAGE_ENTRIES = 0;
__device__ __managed__ int PAGEFAULT = 0;
__device__ __managed__ int CURRENTTIME = 0;
__device__ __managed__ u32 LRU[PHYSICAL_MEM_SIZE/PAGE_SIZE];
__device__ __managed__ uchar storage[STORAGE_SIZE];
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];
extern __shared__ u32 pageTable[];

// Function
// ******************************************************************
// Initialize
__device__ void initPageTable(int entries) {
    for (int i = 0; i < entries; i++) {
        pageTable[i] = DNE;
		LRU[i] = 0;
    }
}
// ******************************************************************

// ******************************************************************
// File I/O
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
// ******************************************************************

// ******************************************************************
// Read/Write
__device__ u32 isValid(u32 PTE) {
	return PTE & VALID;
}
__device__ u32 getPageNumber(u32 PTE) {
	return (PTE & PAGENUMBERMASK) >> 1;
}
__device__ u32 getLastUsedTime(u32 PTE) {
	return (PTE & LASTTIMEMASK) >> 14;
}
__device__ u32 makePTE(u32 time, u32 pageNumber, u32 validbit) {
	return (time << 14) | (pageNumber << 1) | validbit;
}
__device__ u32 paging(uchar *memory, u32 pageNumber, u32 pageOffset) {
	CURRENTTIME++;

	// Find if the target page exists
	for (u32 i = 0; i < PAGE_ENTRIES; i++) {
		if (pageTable[i] == pageNumber) {
			// Update time
			LRU[i] = CURRENTTIME;
			return i * PAGE_SIZE + pageOffset;
		}
	}

	// Find if there is a empty entry to place
	for (u32 i = 0; i < PAGE_ENTRIES; i++) {
		if (pageTable[i] == DNE) {
			// Because of a empty hole, it must be a pagefault
			PAGEFAULT++;
			// Update PTE
			pageTable[i] = pageNumber;
			LRU[i] = CURRENTTIME;
			return i * PAGE_SIZE + pageOffset;
		}
	}

	// Find a place for swaping in by the RULE of LRU
	u32 leastEntry = DNE;
	u32 leastTime  = DNE;
	for (u32 i = 0; i < PAGE_ENTRIES; i++) {
		if (leastTime > LRU[i]) {
			leastTime = LRU[i];
			leastEntry = i;
		}
	}
	// Replace & update infos
	PAGEFAULT++;
	pageTable[leastEntry] = pageNumber;
	LRU[leastEntry] = CURRENTTIME;
	return leastEntry * PAGE_SIZE + pageOffset;
}

__device__ uchar Gread(uchar *memory, u32 address) {
	u32 pageNumber = address/PAGE_SIZE;
	u32 pageOffset = address%PAGE_SIZE;

	u32 reMappingAddress = paging(memory, pageNumber, pageOffset);
	return memory[reMappingAddress];
}

__device__ void Gwrite(uchar *memory, u32 address, uchar writeValue) {
	u32 pageNumber = address/PAGE_SIZE;
	u32 pageOffset = address%PAGE_SIZE;

	u32 reMappingAddress = paging(memory, pageNumber, pageOffset);
	memory[reMappingAddress] = writeValue;
}

__device__ void snapshot(uchar *result, uchar *memory, int offset, int input_size) {
	for (int i = 0; i < input_size; i++) {
		result[i] = Gread(memory, i+offset);
	}
}
// ******************************************************************

// ******************************************************************
// Kernel function
__global__ void mykernel(int input_size) {
    __shared__ uchar data[PHYSICAL_MEM_SIZE];
    PAGE_ENTRIES = PHYSICAL_MEM_SIZE/PAGE_SIZE;
    initPageTable(PAGE_ENTRIES);
	//##Gwrite / Gread code section start###
	for(int i = 0; i < input_size; i++) {
		Gwrite(data, i, input[i]);
	}
	for(int i = input_size - 1; i >= input_size - 10; i--) {
		int value = Gread(data, i);
	}
	//the last line of Gwrite/Gread code section should be snapshot ()
	snapshot(results, data, 0, input_size);
	//###Gwrite/Gread code section end### 
    printf("pagefault times = %d\n", PAGEFAULT);
}
// ******************************************************************

int main() {
    int input_size = loadBinaryFile(DATAFILE, input, STORAGE_SIZE);
	printf("Loading...\n");
    //cudaSetDevice(2);
    mykernel<<<1, 1, 16384>>>(input_size);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    writeBinaryFile(OUTPUTFILE, results, input_size);
    return 0;
}
