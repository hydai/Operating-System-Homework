#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <asm/uaccess.h>        /* for put_user */
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Hung-Ying Tai");

// CONSTS
#define PREFIX_TITLE "OS_HW5"
#define FALSE 0
#define TRUE 1
#define INITIANL_VALUE 0
#define TIMEPERIOD 1024

// DEVICE
#define DEV_NAME "mydev"        // name for alloc_chrdev_region
#define DEV_BASEMINOR 0         // baseminor for alloc_chrdev_region
#define DEV_COUNT 1             // count for alloc_chrdev_region
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;

// Function follow TA specs
static int drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static int drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static int drv_ioctl(struct file *, unsigned int , unsigned long );

static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// I/O function
void myoutb(unsigned char data,unsigned short int port);
void myoutw(unsigned short data,unsigned short int port);
void myoutl(unsigned int data,unsigned short int port);
unsigned char myinb(unsigned short int port);
unsigned short myinw(unsigned short int port);
unsigned int myinl(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;

// Computating prime
int prime(int base, short nth);

// Arithmetic funciton
static void drv_arithmetic_routine(work_struct* ws);

int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }
    return num;
}

void myoutb(unsigned char data,unsigned short int port) {
    *(unsigned char*)(dma_buf+port) = data;
}
void myoutw(unsigned short data,unsigned short int port) {
    *(unsigned short*)(dma_buf+port) = data;
}
void myoutl(unsigned int data,unsigned short int port) {
    *(unsigned int*)(dma_buf+port) = data;
}
unsigned char myinb(unsigned short int port) {
    return *(unsigned char*)(dma_buf+port);
}
unsigned short myinw(unsigned short int port) {
    return *(unsigned short*)(dma_buf+port);
}
unsigned int myinl(unsigned short int port) {
    return *(unsigned int*)(dma_buf+port);
}

static int drv_open(struct inode* ii, struct file* ff) {
    try_module_get(THIS_MODULE);
    printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
    return 0;
}
static int drv_release(struct inode* ii, struct file* ff) {
    module_put(THIS_MODULE);
    printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
    return 0;
}
static int drv_read(struct file *filp, char __user *buffer, size_t, loff_t*) {
    printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, myinl(DMAANSADDR));
    // Clean answer & set readable to false
    myoutl(INITIANL_VALUE, DMAANSADDR);
    myoutl(FALSE, DMAREADABLEADDR);
    return 0;
}
static int drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*) {
    // Get IO mode
    int IOMode = myinl(DMABLOCKADDR);
    printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);
    INIT_WORK(work_routine, arithmetic_routine);

    // Get data
    dataIn = (struct DataIn *)buffer;
    myoutb(dataIn->a, DMAOPCODEADDR);
    myoutl(dataIn->b, DMAOPERANDBADDR);
    myoutw(dataIn->c, DMAOPERANDCADDR);

    // Decide io mode
    schedule_work(work_routine);
    if(IOMode) {
        // Blocking IO
        printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
        flush_scheduled_work();
    }
    return 0;
}
static int drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
    int value = 1;
    unsigned int studentID = 101062124;
    switch(cmd){
    case HW5_IOCSETSTUID:
        // Copy data from user
        get_user(value, (int *)arg);
        myoutl(studentID, DMASTUIDADDR);
        printk("%s:%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, myinl(DMASTUIDADDR));
        break;
    case HW5_IOCSETRWOK:
        // Copy data from user
        get_user(value, (int *)arg);
        myoutl(value, DMARWOKADDR);
        printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
        break;
    case HW5_IOCSETIOCOK:
        // Copy data from user
        get_user(value, (int *)arg);
        myoutl(value, DMAIOCOKADDR);
        printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
        break;
    case HW5_IOCSETIRQOK:
        // Copy data from user
        get_user(value, (int *)arg);
        myoutl(value, DMAIRQOKADDR);
        printk("%s:%s(): IRC OK\n", PREFIX_TITLE, __func__);
        break;
    case HW5_IOCSETBLOCK:
        // Copy data from user
        get_user(value, (int *)arg);
        myoutl(value, DMABLOCKADDR);
        if(value) {
            printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
        } else {
            printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
        }
        break;
    case HW5_IOCWAITREADABLE:
        while(myinl(DMAREADABLEADDR) != TRUE)
            msleep(TIMEPERIOD);
        // Copy data to user
        put_user(TRUE, (int *)arg);
        printk("%s:%s(): wait readable 1\n", PREFIX_TITLE, __func__);
        break;
    default:
        return -1;
    }
    return 0;
}

static void drv_arithmetic_routine(work_struct* ws) {
    DataIn opData;
    opData.a = myinb(DMAOPCODEADDR);
    opData.b = myinl(DMAOPERANDBADDR);
    opData.c = myinw(DMAOPERANDCADDR);
    int ans = 0;
    if (opData.a == '+') {
        ans = opData.b + opData.c;
    } else if (opData.a == '-') {
        ans = opData.b - opData.c;
    } else if (opData.a == '*') {
        ans = opData.b * opData.c;
    } else if (opData.a == '/') {
        ans = opData.b / opData.c;
    } else if (opData.a == 'p') {
        ans = prime(opData.b, opData.c);
    } else {
        ans = 0;
    }
    printk("%s:%s():%d %c %d = %d\n", PREFIX_TITLE, __func__, opData.a, opData.b, opData.c, ans);
    myoutl(ans, DMAANSADDR);
    if(myinl(DMABLOCKADDR == FALSE)) {
        myoutl(TRUE, DMAREADABLEADDR);
    }
}

static int __init init_modules(void) {
    dev_t dev;
    printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);
    dev_cdev = cdev_alloc();

    // Register chrdev
    if(alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME) < 0) {
        printk(KERN_ALERT"Register chrdev failed!\n");
        return -1;
    } else {
        printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);
    }
    dev_major = MAJOR(dev);
    dev_minor = MINOR(dev);

    // Init cdev
    dev_cdev->ops = &fops;
    dev_cdev->owner = THIS_MODULE;
    if(cdev_add(dev_cdev, dev, 1) < 0) {
        printk(KERN_ALERT"Add cdev failed!\n");
        return -1;
    }

    // Alloc DMA buffer
    dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
    if(dma_buf) {
        printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __func__);
    } else {
        printk(KERN_ALERT"Alloc DMA buffer failed!\n");
        return -1;
    }
    
    // Alloc work routine
    work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);
    return 0;
}

static void __exit exit_modules(void) {
    // Free DMA buffer
    kfree(dma_buf);
    printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

    // Delete char device
    unregister_chrdev_region(MKDEV(dev_major,dev_minor), DEV_COUNT);
    cdev_del(dev_cdev);

    // Free work routine
    kfree(work_routine);
    printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);
    printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
