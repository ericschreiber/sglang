# Makefile for WGMMA minimal example
NVCC = nvcc
CFLAGS = -std=c++17 -O2 -arch=sm_90a -lcuda
TARGET = wgmma_min_example
SOURCE = wgmma_min_example.cu
TARGET2 = wgmma_simple
SOURCE2 = wgmma_simple.cu

all: $(TARGET) $(TARGET2)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

$(TARGET2): $(SOURCE2)
	$(NVCC) $(CFLAGS) -o $(TARGET2) $(SOURCE2)

clean:
	rm -f $(TARGET) $(TARGET2)

.PHONY: clean all
