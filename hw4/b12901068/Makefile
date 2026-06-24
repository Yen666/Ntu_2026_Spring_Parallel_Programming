NVFLAGS := -std=c++11 -O3 -Xptxas=-v -use_fast_math  --maxrregcount=64
DEBUGFLAGS := -std=c++11 -G -g -lineinfo -Xptxas=-v
DEBUGTARGET := hw4_debug
TARGET := hw4

.PHONY: all
all: $(TARGET)

.PHONY: debug
debug: $(DEBUGTARGET)

$(TARGET): sha256.o hw4.cu
	nvcc $(NVFLAGS) -rdc=true -o hw4 hw4.cu sha256.o

sha256.o: sha256.cu
	nvcc $(NVFLAGS) -rdc=true -dc sha256.cu

sha256_debug.o: sha256.cu
	nvcc $(DEBUGFLAGS) -rdc=true -dc sha256.cu -o sha256_debug.o

$(DEBUGTARGET): sha256_debug.o hw4.cu
	nvcc $(DEBUGFLAGS) -rdc=true -o hw4_debug hw4.cu sha256_debug.o

clean:
	rm -rf hw4 *.o




