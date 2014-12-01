all:
	nvcc -arch=sm_30 main.cu -o vm

debug:
	nvcc -arch=sm_30 main.cu -g -G -o vm-debug
clean:
	rm vm vm-debug
