# Makefile for WGMMA minimal example
NVCC = nvcc
CFLAGS = -std=c++17 -O2 -arch=sm_90a -lcuda
CUTLASS_PATH=/sgl-workspace/sglang-dev/sglang/moe_kernels/cutlass
TARGET = wgmma_min_example_v2
SOURCE = wgmma_min_example_v2.cu
TARGET2 = wgmma_simple
SOURCE2 = wgmma_simple.cu
TARGET3 = wgmma_min_example_cutlass
SOURCE3 = wgmma_min_example_cutlass.cu
TARGET4 = wgmma_min_example_viewing
SOURCE4 = wgmma_min_example_viewing.cu


all: $(TARGET) $(TARGET2) $(TARGET3) $(TARGET4)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

$(TARGET2): $(SOURCE2)
	$(NVCC) $(CFLAGS) -o $(TARGET2) $(SOURCE2)

$(TARGET3): $(SOURCE3)
	$(NVCC) $(CFLAGS) -o $(TARGET3) $(SOURCE3) -I$(CUTLASS_PATH)/include 

$(TARGET4): $(SOURCE4)
	$(NVCC) $(CFLAGS) -o $(TARGET4) $(SOURCE4)

clean:
	rm -f $(TARGET) $(TARGET2) $(TARGET3) $(TARGET4)

.PHONY: clean all
