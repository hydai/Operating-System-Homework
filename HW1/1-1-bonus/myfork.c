#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/signal.h>
#include <sys/wait.h>

int childID = 1;
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
        exit(1);
    }
    pid_t *pids = (pid_t *)malloc((argc+2)*sizeof(pid_t));
    pids[0] = getpid();
    for (; childID < argc; ++childID) {
        pids[childID] = fork();
        if (pids[childID] == 0) {
            /* Child process. */
            printf("###### Process %d - PARENT %d - CHILDID %d ######\n", getpid(), getppid(), childID);
            execl(argv[childID], argv[childID], (char *)0);
        } else if (pids[childID] < 0) {
            /* fork failed. */
            status = -1;
        } else {
            /* Parent process */
            waitpid(pids[childID], &status, 0);
            printf("======= SIGNAL START =======\n");
            printf("Receiving signal from child.\n");
            if (WIFSIGNALED(status)) {
                genSignalMessage(WTERMSIG(status));
            } else if (WIFSTOPPED(status)) {
                genSignalMessage(WSTOPSIG(status));
            } else if (WIFEXITED(status)) {
                genPassMessage(WEXITSTATUS(status));
            }
            printf("======= SIGNAL   END =======\n");
            printf("\n\n");
        }
    }
    printf("\n\n======= watproc part =======\n");
    int i, j;
    for (i = 0; i < argc; i++) {
        if (i == 0) printf("Root: %d\n", pids[i]);
        else {
            for (j = 0; j < i; j++) {
                printf("-");
            }
            printf("> %d\n", pids[i]);
        }
    }
    return 0;
}

void print_help() {
    printf("Usage:\n");
    printf("\t./myfork ./ProgWantToTest\n");
}

void genSignalMessage(int status) {
    printf("======= Error Detected =======\n");
    switch(WTERMSIG(status)) {
    case SIGHUP:
        printf("CHILD SIGNAL: SIGHUP\n");
        printf("\tReason:\tHangup detected on controlling terminal or death of controlling process\n");
        break;
    case SIGINT:
        printf("CHILD SIGNAL: SIGINT\n");
        printf("\tReason:\tInterrupt from keyboard\n");
        break;
    case SIGQUIT:
        printf("CHILD SIGNAL: SIGQUIT\n");
        printf("\tReason:\tQuit from keyboard\n");
        break;
    case SIGILL:
        printf("CHILD SIGNAL: SIGILL\n");
        printf("\tReason:\tQuit from keyboard\n");
        break;
    case SIGABRT:
        printf("CHILD SIGNAL: SIGABRT\n");
        printf("\tReason:\tAbort SIGNAL from abort(3)\n");
        break;
    case SIGFPE:
        printf("CHILD SIGNAL: SIGFPE\n");
        printf("\tReason:\tFloating point exception\n");
        break;
    case SIGSEGV:
        printf("CHILD SIGNAL: SIGSEGV\n");
        printf("\tReason:\tInvalid memory reference\n");
        break;
    case SIGPIPE:
        printf("CHILD SIGNAL: SIGPIPE\n");
        printf("\tReason:\tBroken pipe: write to pipe with no readers\n");
        break;
    case SIGALRM:
        printf("CHILD SIGNAL: SIGALRM\n");
        printf("\tReason:\tTimer SIGNAL from alarm(2)\n");
        break;
    case SIGTERM:
        printf("CHILD SIGNAL: SIGTERM\n");
        printf("\tReason:\tTermination SIGNAL\n");
        break;
    case SIGUSR1:
        printf("CHILD SIGNAL: SIGUSR1\n");
        printf("\tReason:\tUser-defined SIGNAL 1\n");
        break;
    case SIGUSR2:
        printf("CHILD SIGNAL: SIGUSR2\n");
        printf("\tReason:\tUser-defined SIGNAL 2\n");
        break;
    case SIGCONT:
        printf("CHILD SIGNAL: SIGCONT\n");
        printf("\tReason:\tContinue if stopped\n");
        break;
    case SIGTSTP:
        printf("CHILD SIGNAL: SIGTSTP\n");
        printf("\tReason:\tStop typed at terminal\n");
        break;
    case SIGTTIN:
        printf("CHILD SIGNAL: SIGTTIN\n");
        printf("\tReason:\tTerminal input for background process\n");
        break;
    case SIGTTOU:
        printf("CHILD SIGNAL: SIGTTOU\n");
        printf("\tReason:\tTerminal output for background process\n");
        break;
    case SIGBUS:
        printf("CHILD SIGNAL: SIGBUS\n");
        printf("\tReason:\tBus error (bad memory access)\n");
        break;
    case SIGPROF:
        printf("CHILD SIGNAL: SIGPROF\n");
        printf("\tReason:\tProfiling timer expired\n");
        break;
    case SIGSYS:
        printf("CHILD SIGNAL: SIGSYS\n");
        printf("\tReason:\tBad argument to routine (SVr4)\n");
        break;
    case SIGTRAP:
        printf("CHILD SIGNAL: SIGTRAP\n");
        printf("\tReason:\tTrace/breakpoint trap\n");
        break;
    case SIGURG:
        printf("CHILD SIGNAL: SIGURG\n");
        printf("\tReason:\tUrgent condition on socket (4.2BSD)\n");
        break;
    case SIGVTALRM:
        printf("CHILD SIGNAL: SIGVTALRM\n");
        printf("\tReason:\tVirtual alarm clock (4.2BSD)\n");
        break;
    case SIGXCPU:
        printf("CHILD SIGNAL: SIGXCPU\n");
        printf("\tReason:\tCPU time limit exceeded (4.2BSD)\n");
        break;
    case SIGXFSZ:
        printf("CHILD SIGNAL: SIGXFSZ\n");
        printf("\tReason:\tFile size limit exceeded (4.2BSD)\n");
        break;
    case SIGIO:
        printf("CHILD SIGNAL: SIGIO\n");
        printf("\tReason:\tI/O now possible (4.2BSD)\n");
        break;
    case SIGWINCH:
        printf("CHILD SIGNAL: SIGWINCH\n");
        printf("\tReason:\tWindow resize SIGNAL (4.3BSD, Sun)\n");
        break;
    default:
        printf("UNKNOWN SIGNAL ERROR\n");
        break;
    }
    printf("Child process execution FAILED\n");
}

void genPassMessage(int status) {
    printf("CHILD PROCESS normally terminated with exit status = %d\n", status);
}
