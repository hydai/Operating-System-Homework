There are three parts for the project 1.

First, project 1-1, user mode monitor:
    The source files are in "1-1-UserModeMonitor".
    1. cd 1-1-UserModeMonitor
    2. make
        For generating the binary file `UserModeMonitor`
    3. ./UserModeMonitor [programFilePath]+
        For example, if you want to execute the executable binary file named `Normal` in the same folder with `UserModeMonitor`, using the command:
        ./UserModeMonitor ./Normal
    4. I/O format:
        Input:
            Get input from argv.
        Output:
            For every program, print the parent pid first and then following with runtime messages.

Second, project bonus:
    The source files are in "1-1-bonus".
    1. cd 1-1-bonus
    2. make
        For generating the binary file `myfork`
    3. ./myfork [programFilePath]+
        For example, if you want to execute the executable binary files named `Normal`, `abort` in the same folder with `myfork`, using the command:
        ./myfork ./Normal ./abort
    4. I/O format:
        Input:
            Get input list from argv.
        Output:
            Print the list of fork history first.
            And it will dump the runtime message from the last one back to the first one.

Third, project 1-2, kernel object:
    The source files are in `1-2-KernelObject`.
    No implement for this part.
    I give up.
