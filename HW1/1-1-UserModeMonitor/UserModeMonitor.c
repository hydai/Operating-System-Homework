#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/signal.h>
#include <sys/wait.h>

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
    pid_t pid = fork();
    if (pid == 0) {
        /* Child process. */
        printf("=======CHILD START=======\n");
        execl(argv[1], argv[1], (char *)0);
    } else if (pid < 0) {
        /* fork failed. */
        status = -1;
    } else {
        /* Parent process */
        waitpid(pid, &status, 0);
        printf("=======SIGNAL START=======\n");
        printf("Receiving signal from child.\n");
        if (WIFSIGNALED(status)) {
            genSignalMessage(WTERMSIG(status));
        } else if (WIFSTOPPED(status)) {
            genSignalMessage(WSTOPSIG(status));
        } else if (WIFEXITED(status)) {
            genPassMessage(WEXITSTATUS(status));
        }
        printf("=======SIGNAL   END=======\n");
    }
    return 0;
}

void print_help() {
    printf("Usage:\n");
    printf("\t./UserModeMonitor ./ProgWantToTest\n");
}

void genSignalMessage(int status) {
    printf("=======Error Detected=======\n");
    switch(WTERMSIG(status)) {
    case SIGHUP:
        printf("CHILD SIGNAL: SIGHUP\n");
        printf("\tReason:\tHangup detected on controlling terminal or death of controlling process\n");
        break;
    case SIGINT:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGINT\n");
        printf("\tReason:\tInterrupt from keyboard\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGQUIT:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGQUIT\n");
        printf("\tReason:\tQuit from keyboard\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGILL:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGILL\n");
        printf("\tReason:\tQuit from keyboard\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGABRT:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGABRT\n");
        printf("\tReason:\tAbort SIGNAL from abort(3)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGFPE:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGFPE\n");
        printf("\tReason:\tFloating point exception\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGSEGV:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGSEGV\n");
        printf("\tReason:\tInvalid memory reference\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGPIPE:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGPIPE\n");
        printf("\tReason:\tBroken pipe: write to pipe with no readers\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGALRM:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGALRM\n");
        printf("\tReason:\tTimer SIGNAL from alarm(2)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGTERM:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGTERM\n");
        printf("\tReason:\tTermination SIGNAL\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGUSR1:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGUSR1\n");
        printf("\tReason:\tUser-defined SIGNAL 1\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGUSR2:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGUSR2\n");
        printf("\tReason:\tUser-defined SIGNAL 2\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGCONT:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGCONT\n");
        printf("\tReason:\tContinue if stopped\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGTSTP:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGTSTP\n");
        printf("\tReason:\tStop typed at terminal\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGTTIN:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGTTIN\n");
        printf("\tReason:\tTerminal input for background process\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGTTOU:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGTTOU\n");
        printf("\tReason:\tTerminal output for background process\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGBUS:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGBUS\n");
        printf("\tReason:\tBus error (bad memory access)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGPROF:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGPROF\n");
        printf("\tReason:\tProfiling timer expired\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGSYS:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGSYS\n");
        printf("\tReason:\tBad argument to routine (SVr4)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGTRAP:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGTRAP\n");
        printf("\tReason:\tTrace/breakpoint trap\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGURG:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGURG\n");
        printf("\tReason:\tUrgent condition on socket (4.2BSD)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGVTALRM:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGVTALRM\n");
        printf("\tReason:\tVirtual alarm clock (4.2BSD)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGXCPU:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGXCPU\n");
        printf("\tReason:\tCPU time limit exceeded (4.2BSD)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGXFSZ:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGXFSZ\n");
        printf("\tReason:\tFile size limit exceeded (4.2BSD)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGIO:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGIO\n");
        printf("\tReason:\tI/O now possible (4.2BSD)\n");
        printf("Child process execution FAILED\n");
        break;
    case SIGWINCH:
        printf("=======Error Detected=======\n");
        printf("CHILD SIGNAL: SIGWINCH\n");
        printf("\tReason:\tWindow resize SIGNAL (4.3BSD, Sun)\n");
        printf("Child process execution FAILED\n");
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
