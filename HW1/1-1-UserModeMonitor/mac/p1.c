#include <stdio.h>
int main(int argc, char *argv[])
{
    printf("----------------PROCESS START----------------\n");
    printf("Prob 1: segmentFault\n");
    int *a = 0; *a = 0;
    printf("----------------PROCESS   END----------------\n");
    return 0;
}
