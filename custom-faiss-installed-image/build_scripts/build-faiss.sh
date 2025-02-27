#!/bin/bash

# Exit on any error
set -xe

cd /tmp/faiss

printenv

cmake --version

echo "Running cmake build"
pwd
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_OPT_LEVEL=generic \
    -DFAISS_ENABLE_C_API=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$CONDA/bin/python \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DFAISS_ENABLE_CUVS=ON \
    -DCUDAToolkit_ROOT="/usr/local/cuda/lib64" \
    .

echo "Running make command"

make -C build -j6 faiss swigfaiss

# Create Python FAISS bindings
cd build/faiss/python && python3 setup.py build

# make faiss python bindings available for use
export PYTHONPATH="$(ls -d `pwd`/tmp/faiss/build/faiss/python/build/lib*/):`pwd`/"