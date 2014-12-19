all:
	nvcc -arch=sm_30 main.cu -o fs

bonus:
	nvcc -arch=sm_30 bonus.cu -o ffs

debug:
	nvcc -arch=sm_30 main.cu -g -G -o fs-debug
clean:
	rm fs ffs fs-debug
