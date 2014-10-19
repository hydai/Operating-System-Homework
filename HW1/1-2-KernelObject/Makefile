obj-m       += myforkModule.o
KVERSION := $(shell uname -r)
QAQPATH=/usr/src/linux-source-3.13.0

all:
	    #$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules
	    $(MAKE) -C $(QAQPATH) M=$(PWD) modules

clean:
	    #$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) clean
	    $(MAKE) -C $(QAQPATH) M=$(PWD) clean


