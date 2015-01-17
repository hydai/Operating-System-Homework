mname := watchdog
$(mname)-objs := main.o
obj-m := $(mname).o

KERNELDIR := /lib/modules/`uname -r`/build

all:
	$(MAKE) -C $(KERNELDIR) M=`pwd` modules
	sudo insmod watchdog.ko

clean:
	$(MAKE) -C $(KERNELDIR) M=`pwd` clean
	sudo rmmod watchdog
	dmesg | grep OS_HW5_BONUS

