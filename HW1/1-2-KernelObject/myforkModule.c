#include <linux/module.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/syscalls.h>


MODULE_LICENSE("GPL");
static char *inputFile;
module_param(inputFile, charp, S_IRUGO|S_IWUSR);

struct task_struct *task;
static int executeProgram(void *inputFile) {
    char *argv[] = {
        inputFile,
        NULL };
    static char *envp[] = {
        "HOME=/",
        "TERM=linux",
        "PATH=/sbin:/bin:/usr/sbin:/usr/bin",
        NULL };
    printk("%s\n", argv[0]);
    return call_usermodehelper( argv[0], argv, envp, UMH_WAIT_PROC );
}
int my_fork(void *argv) {
    /* do_fork(clone_flags, stack_start, stack_size, parent_tidptr, child_tidptr);*/
    /*
    pid_t pid = do_fork(SIGCHLD, 0, 0, NULL, NULL);
    if (pid == 0) {
        printk("I'm child\n");
    } else {
        //do_wait(pid);
        printk("I'm parent\n");
    }
    */
    return executeProgram(argv);
}
int my_monitor(void *argv)
{
    task = kthread_create(my_fork, argv, "my_fork");
    wake_up_process(task);
    return 0;
}

static int __init kernel_object_test_init(void)
{
    int result, result1;
    struct pid *kpid;
    struct sched_param param;

    printk("parameter: %s\n", inputFile);
    my_monitor(inputFile);
    return 0;
}

static void __exit kernel_object_test_exit(void)
{
    kthread_stop(task);
    printk("<0>Remove the module\n");
}

module_init(kernel_object_test_init);
module_exit(kernel_object_test_exit);
