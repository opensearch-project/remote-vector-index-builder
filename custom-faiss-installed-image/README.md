## Creating the image
Below are the steps to create the image that can have a custom faiss python code build.
1. The basic docker file is present in this folder named Dockerfile.
2. Ensure that faiss is checked out before you trigger the build. The commit of faiss I have tested with is: 3c8dc4194907e9b911551d5a009468106f8b9c7f
3. Build the docker image using this command
```
rm -rf faiss ; git restore faiss ; ./build-docker-image.sh
```
4. Once image is build use the below command to run the container.
```
docker run --gpus all -d -v ./:/tmp/ custom-faiss:latest
```
Make sure to run the command from the same folder where this readme is present otherwise wrong files get copied.
5. Now ssh into the container
```
docker container exec -it <container-id>  /bin/bash
```
6. Once you ssh in the container make sure that you are sshed as `appuser`.
7. Now go to `tmp/faiss` using `cd /tmp/faiss`
8. Now run below command to trigger the cmake build.
```
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
```
9. Now run the below command to make faiss. Here I am using 4 cores to build faiss, you can use more if you have more cores. Just make sure that you are not using all cores otherwise the machine will get struck. This command takes a lot of time to run and build faiss
```
make -C build -j6 faiss swigfaiss
```
10. Once the build is complete now run the below command to create faiss bindings.
```
cd build/faiss/python && python3 setup.py build
```
11. One possible way to use faiss which you have built is like this, that can be added in the code.
```
import sys
sys.path.append('/tmp/faiss/build/faiss/python')
import faiss
```
Another approach can be setting the PYTHONPATH variable like this
```
export PYTHONPATH="$(ls -d `pwd`/tmp/faiss/build/faiss/python/build/lib*/):`pwd`/"
```
12. Now to test if everthing is build correctly or not run the below command
```
python faiss-test.py
```
and see an output like this
```
Creating GPU Index.. with IVF_PQ
Indexing done
```
