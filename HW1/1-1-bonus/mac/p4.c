#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    printf("----------------PROCESS START----------------\n");
    printf("Prob 4: alarm\n");
    alarm(1);
    while(1);
    printf("----------------PROCESS   END----------------\n");
    return 0;
}
