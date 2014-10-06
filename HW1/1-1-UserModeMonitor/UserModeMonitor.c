#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/signal.h>
#include <sys/wait.h>

#define DEBUG

void print_help();
void genSignalMessage(int);
void genPassMessage(int);
int main(int argc, char *argv[])
{
    int status;
    /* if user does not input program file, 
     * print help message and terminate.
     * */
    if (argc < 2) {
        print_help();
#ifdef DEGUB
        printf("ABRT = %d\n", SIGABRT);
        printf("ALAR = %d\n", SIGALRM);
        printf("SEGV = %d\n", SIGSEGV);
        printf("CHLD = %d\n", SIGCHLD);
#endif
        exit(1);
    }
    int signalvalues[] = {SIGHUP, SIGINT, SIGQUIT, SIGILL, SIGABRT, SIGFPE, SIGKILL, SIGSEGV, SIGPIPE, SIGALRM, SIGTERM, SIGUSR1, SIGUSR2, SIGCONT, SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU, SIGBUS, SIGPROF, SIGSYS, SIGTRAP, SIGURG, SIGVTALRM, SIGXCPU, SIGXFSZ, SIGIO, SIGWINCH};
    int i;
    for (i = 0; i < sizeof(signalvalues)/sizeof(int); i++) {
        signal(signalvalues[i], genSignalMessage);
    }
    signal(SIGCHLD, genPassMessage);
    pid_t pid = fork();
    if (pid == 0) {
        /* Child process. */
        execl(argv[1], argv[1], (char *)0);
    } else if (pid < 0) {
        /* fork failed. */
        status = -1;
    } else {
        /* Parent process */
        waitpid(pid, &status, 0);
    }
    return 0;
}

void print_help() {
    printf("Usage:\n");
    printf("\t./UserModeMonitor ./ProgWantToTest\n");
}

void genSignalMessage(int status) {
    printf("@Receiving signal from child.\n");
    printf("OAO - %d\n", status);
    signal(status, genSignalMessage);
    switch(status) {
    case SIGHUP:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGHUP\n");
       printf("\tHangup detected on controlling terminal\n");
       printf("\tor death of controlling process\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGINT:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGINT\n");
       printf("\tInterrupt from keyboard\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGQUIT:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGQUIT\n");
       printf("\tQuit from keyboard\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGILL:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGILL\n");
       printf("\tQuit from keyboard\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGABRT:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGABRT\n");
       printf("\tAbort signal from abort(3)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGFPE:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGFPE\n");
       printf("\tFloating point exception\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGKILL:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGKILL\n");
       printf("\tKill signal\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGSEGV:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGSEGV\n");
       printf("\tInvalid memory reference\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGPIPE:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGPIPE\n");
       printf("\tBroken pipe: write to pipe with no readers\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGALRM:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGALRM\n");
       printf("\tTimer signal from alarm(2)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGTERM:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGTERM\n");
       printf("\tTermination signal\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGUSR1:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGUSR1\n");
       printf("\tUser-defined signal 1\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGUSR2:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGUSR2\n");
       printf("\tUser-defined signal 2\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGCONT:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGCONT\n");
       printf("\tContinue if stopped\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGSTOP:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGSTOP\n");
       printf("\tStop process\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGTSTP:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGTSTP\n");
       printf("\tStop typed at terminal\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGTTIN:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGTTIN\n");
       printf("\tTerminal input for background process\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGTTOU:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGTTOU\n");
       printf("\tTerminal output for background process\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGBUS:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGBUS\n");
       printf("\tBus error (bad memory access)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGPROF:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGPROF\n");
       printf("\tProfiling timer expired\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGSYS:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGSYS\n");
       printf("\tBad argument to routine (SVr4)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGTRAP:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGTRAP\n");
       printf("\tTrace/breakpoint trap\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGURG:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGURG\n");
       printf("\tUrgent condition on socket (4.2BSD)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGVTALRM:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGVTALRM\n");
       printf("\tVirtual alarm clock (4.2BSD)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGXCPU:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGXCPU\n");
       printf("\tCPU time limit exceeded (4.2BSD)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGXFSZ:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGXFSZ\n");
       printf("\tFile size limit exceeded (4.2BSD)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGIO:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGIO\n");
       printf("\tI/O now possible (4.2BSD)\n");
       printf("Child process execution FAILED\n");
       break;
    case SIGWINCH:
       printf("=======Error Detected=======\n");
       printf("CHILD signal: SIGWINCH\n");
       printf("\tWindow resize signal (4.3BSD, Sun)\n");
       printf("Child process execution FAILED\n");
       break;
    }
}

void genPassMessage(int status) {
   signal(status, genPassMessage);
   switch(status) {
   case SIGCHLD:
      printf("Child process normal terminated with exit status = %d\n", status);
      break;
   }
}
