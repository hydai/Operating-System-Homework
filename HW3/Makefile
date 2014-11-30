all:	main.o helpFunction.o
	nvcc -arch=sm_30 main.o helpFunction.o -rdc=true -o vm

main.o:	main.cu
	nvcc -arch=sm_30 main.cu -rdc=true -c

helpFunction.o:	helpFunction.cu
	nvcc -arch=sm_30 helpFunction.cu -rdc=true -c

clean:
	rm vm *.o snapshot.bin
