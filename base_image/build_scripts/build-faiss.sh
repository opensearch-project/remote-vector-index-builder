#!/bin/bash

# Exit on any error
set -xe

cd /tmp/faiss

printenv

cmake --version

# Ref: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#step-1-invoking-cmake
# Step 1: Invoke CMake
echo "Running cmake build"
pwd
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_OPT_LEVEL=generic \
    -DFAISS_ENABLE_C_API=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$CONDA/bin/python \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DFAISS_ENABLE_CUVS=ON \
    -DCUDAToolkit_ROOT="/usr/local/cuda/lib64" \
    .

# Step 2 : Invoke Make
# This builds the C++ library (libfaiss.a by default, and libfaiss.so if -DBUILD_SHARED_LIBS=ON was passed to CMake).
# Also Builds the Python bindings(swigfaiss)
echo "Running make command"

# The -j option enables parallel compilation of multiple units, leading to a faster build,
# but increasing the chances of running out of memory,
# in which case it is recommended to set the -j option to a fixed value (such as -j6).
make -C build -j6 faiss swigfaiss

# Step 3: Generate and install python packages
cd build/faiss/python && python3 setup.py build

# make faiss python bindings available for use
export PYTHONPATH="$(ls -d `pwd`/tmp/faiss/build/faiss/python/build/lib*/):`pwd`/"