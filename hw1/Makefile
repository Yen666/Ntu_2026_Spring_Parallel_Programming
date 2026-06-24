CXX = g++
CXXFLAGS = -std=c++17 -O3 -ltbb
TBB ?= 1
ifeq ($(TBB),1)
	CXXFLAGS += -DCOMPILE_TBB
	LDLIBS += -ltbb
endif
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

# Explicit link rule to ensure C++ linker (g++) is used instead of cc
hw1: hw1.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

.PHONY: asan
asan: CXXFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer -g
asan: clean hw1

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o
