#ifndef DECLARATION_H
#define DECLARATION_H

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

// Function prototype
__device__ void initPageTable(int entries);
int loadBinaryFile(char *fileName, uchar *input, int storageSize);
void writeBinaryFile(char *fileName, uchar *input, int storageSize);

#endif
