# Makefile for hw2_sift project
# Compiler and flags
MPICXX = mpicxx
MPIFLAGS = -std=c++17 -O3 -fopenmp -march=native
# Profiling flags for gprof (use `make profile`) and perf (use `make perf`)
PROFILE_FLAGS = -std=c++17 -O2 -g -pg -fopenmp
PERF_FLAGS = -std=c++17 -O2 -g -fopenmp -fno-omit-frame-pointer -march=native
INCLUDES = -I.

# Source files
SOURCES = hw2.cpp sift.cpp image.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = hw2

# STB image library files
STB_HEADERS = stb_image.h stb_image_write.h

# Default target
all: $(TARGET)

# Build targets
$(TARGET): $(OBJECTS)
	$(MPICXX) $(MPIFLAGS) -o $(TARGET) $(OBJECTS)

.PHONY: profile perf
# Build with gprof instrumentation
profile: CXXFLAGS := $(PROFILE_FLAGS)
profile: clean $(SOURCES)
	$(MPICXX) $(PROFILE_FLAGS) -o $(TARGET) $(SOURCES)

# Build with perf-friendly flags (frame pointers)
perf: CXXFLAGS := $(PERF_FLAGS)
perf: clean $(SOURCES)
	$(MPICXX) $(PERF_FLAGS) -o $(TARGET) $(SOURCES)

# Compile source files to object files
hw2.o: hw2.cpp sift.hpp image.hpp
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c hw2.cpp

sift.o: sift.cpp sift.hpp image.hpp $(STB_HEADERS)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c sift.cpp

image.o: image.cpp image.hpp $(STB_HEADERS)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c image.cpp

# Clean up build artifacts
clean:
	rm -f $(OBJECTS) $(MPI_OBJECTS) $(TARGET) result.jpg

.PHONY: all clean
