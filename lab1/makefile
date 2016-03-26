DIR=~/cuda-6.5/bin/
NVCC_FLAG=-std=c++11 -arch=sm_30 -O2

all: lab1_exe

lab1_exe: counting.o main.cu
	$(DIR)/nvcc $(NVCC_FLAG) main.cu counting.o -o lab1_exe

counting.o: counting.cu
	$(DIR)/nvcc $(NVCC_FLAG) -c counting.cu -o counting.o

clean:
	rm -f counting.o lab1_exe
