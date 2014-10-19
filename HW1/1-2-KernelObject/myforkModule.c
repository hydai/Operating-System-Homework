#include <linux/module.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/syscalls.h>


MODULE_LICENSE("GPL");
static char *inputFile;
module_param(inputFile, charp, S_IRUGO|S_IWUSR);

struct task_struct *task;
int my_fork(void *argv) {

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
