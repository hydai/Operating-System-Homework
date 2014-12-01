#include <stdio.h>
#include <stdlib.h>
int main() {
	FILE *fptr = fopen("ts.bin", "wb");
	int ct = 131072;
	int i, j;
	unsigned char key;
	while(ct > 0) {
		for (i = 0; i < 32; i++) {
			for (j = 0; j < 131072/32; j++) {
				key = (i+j)%128;
				fwrite(&key, sizeof(unsigned char), 1, fptr);
				ct--;
			}
		}
	}
	fclose(fptr);
	return 0;
}
