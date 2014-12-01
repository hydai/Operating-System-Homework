all:
	nvcc -arch=sm_30 main.cu -o vm

bonus:
	nvcc -arch=sm_30 bonus.cu -o vm-bonus

debug:
	nvcc -arch=sm_30 main.cu -g -G -o vm-debug
clean:
	rm vm vm-debug vm-bonus
