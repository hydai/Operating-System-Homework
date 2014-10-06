#include <signal.h>
#include <unistd.h>
#include <stdio.h>
void sigroutine(int dunno) { /* 信號處理例程，其中dunno將會得到信號的值 */
   switch (dunno) {
   case 1:
      printf("Get a signal -- SIGHUP \n");
      break;
   case 2:
      printf("Get a signal -- SIGINT \n");
      break;
   case 3:
      printf("Get a signal -- SIGQUIT \n");
      break;
   }
   return;
}

int main() {
   printf("process id is %d ",getpid());
   signal(SIGHUP, sigroutine); //* 下面設置三個信號的處理方法
   signal(SIGINT, sigroutine);
   signal(SIGQUIT, sigroutine);
   for (;;) ;
}
