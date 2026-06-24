CXX := g++
CXXFLAGS := -std=c++17 -O3
HIPFLAGS := $(CXXFLAGS) --offload-arch=gfx908 -ffast-math -munsafe-fp-atomics -funroll-loops -fno-math-errno -ffp-contract=fast
TARGET := hw5
SEQUENTIAL := nbody


.PHONY: all
all: $(TARGET)

.PHONY: hw5
hw5: hw5.cpp
	hipcc $(HIPFLAGS) -o hw5 hw5.cpp
.PHONY: seq
seq: nbody.cc
	$(CXX) $(CXXFLAGS) -o nbody nbody.cc

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)


