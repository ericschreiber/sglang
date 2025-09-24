#!/bin/bash

# Script to compile tma_test.cu standalone
# This requires CUDA Toolkit and CUTLASS headers

echo "Compiling TMA test..."

# Set paths (adjust these based on your system)
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
CUTLASS_PATH=${CUTLASS_PATH:-../sgl-kernel/build/_deps/repo-cutlass-src}

# Check if CUTLASS path exists
if [ ! -d "$CUTLASS_PATH" ]; then
    echo "Warning: CUTLASS path not found at $CUTLASS_PATH"
    echo "You may need to build the sgl-kernel first or set CUTLASS_PATH environment variable"
    echo "To build sgl-kernel: cd ../sgl-kernel && make build"
fi

# Compilation command
nvcc -o tma_test tma_test.cu \
    -std=c++17 \
    -O3 \
    -gencode=arch=compute_90a,code=sm_90a \
    -I${CUTLASS_PATH}/include \
    -I${CUTLASS_PATH}/tools/util/include \
    -I${CUDA_PATH}/include \
    -L${CUDA_PATH}/lib64 \
    -lcuda \
    -lcudart \
    -DCUTE_USE_PACKED_TUPLE=1 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -Xcompiler=-fPIC

if [ $? -eq 0 ]; then
    echo "Compilation successful! Run with: ./tma_test"
else
    echo "Compilation failed!"
    exit 1
fi
