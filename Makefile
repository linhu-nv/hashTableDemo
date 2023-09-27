
all: hashTableTest 

hashTableTest: hashTable.cu
	nvcc  -O3 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86  -rdc=true hashTable.cu -o hashTableTest #  -g -G #-lineinfo

clean:
	rm -f hashTableTest *.o


