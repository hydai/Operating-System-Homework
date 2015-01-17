#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <asm/uaccess.h>
#include <linux/interrupt.h>
#include <linux/sched.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Hung-Ying Tai");

// CONSTS
#define PREFIX_TITLE "OS_HW5_BONUS"

// Vars
static int counter = 0;
static int IRQ_NUM = 1;
void *irq_dev_id = (void *)&IRQ_NUM; 

// Function
static irqreturn_t watchdog_irq(int irq, void* dev_id) {
	if (irq == IRQ_NUM) {
		counter++;
	}
	return IRQ_NONE;
}
static int __init init_modules(void) {
    printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);
	if (request_irq(IRQ_NUM, watchdog_irq, IRQF_SHARED, "watchdog", irq_dev_id) < 0) {
		printk(KERN_ALERT"%s:GG\n", PREFIX_TITLE);
		return -1;
	}
    return 0;
}

static void __exit exit_modules(void) {
	free_irq(IRQ_NUM, irq_dev_id);
	printk("%s:Total count = %d\n", PREFIX_TITLE, counter);
    printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
