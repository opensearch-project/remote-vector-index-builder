[Developer Guide](#developer-guide)
- [Getting Started](#getting-started)
    - [Fork OpenSearch Remote Vector Index Builder Repo](#fork-remote-vector-index-builder-repo)
    - [Install Prerequisites](#install-prerequisites)
        - [Python Dependencies](#python-dependencies)
- [Python Guide](#python-guide)
    - [Language Formatting Guide](#language-formatting-guide)
    - [Testing Guide](#testing-guide)
- [Building Docker Images](#building-docker-images)
    - [Faiss Base Image](#faiss-base-image)
    - [Core Image](#core-image)
- [Provisioning an Instance for Development](#provisioning-an-instance-for-development)
- [Memory Profiling](#memory-profiling)
    - [GPU Memory Profiling with NVIDIA SMI](#gpu-memory-profiling-with-nvidia-smi)
    - [CPU Memory Profiling with memory_profiler](#cpu-memory-profiling-with-memory_profiler)

# Developer Guide

So you want to contribute code to OpenSearch Remote Vector Index Builder? Excellent! We're glad you're here. Here's what you need to do.

## Getting Started

### Fork Remote Vector Index Builder Repo

Fork [opensearch-project/OpenSearch Remote Vector Index Builder](https://github.com/opensearch-project/remote-vector-index-builder) and clone locally.

Example:
```
git clone https://github.com/[username]/remote-vector-index-builder.git
```

### Install Prerequisites

#### Python Dependencies

The following are commands to install dependencies during local development and testing.
The required dependencies are installed onto the Docker image during creation.

Core Dependencies:
```
pip install -r remote_vector_index_builder/core/requirements.txt
```

Test Dependencies:
```
pip install -r test_remote_vector_index_builder/requirements.txt
```

## Python Guide
### Language Formatting Guide
Run the following commands from the root folder. Configuration of below tools can be found in [`setup.cfg`](setup.cfg).

The code lint check can be run with:
```
flake8 remote_vector_index_builder/ test_remote_vector_index_builder/
```

The formatting check can be run with:
```
black --check remote_vector_index_builder/ test_remote_vector_index_builder/
```

The code can be formatted with:
```
black remote_vector_index_builder/ test_remote_vector_index_builder/
```

### Testing Guide
The static type checking can be done with:
```
mypy remote_vector_index_builder/ test_remote_vector_index_builder/
```
The Python tests can be run with:
```
pytest test_remote_vector_index_builder/
```

## Building Docker Images
The Github CIs automatically publish snapshot images to Dockerhub at [opensearchstaging/remote-vector-index-builder](https://hub.docker.com/r/opensearchstaging/remote-vector-index-builder).

The following are the commands to build the images locally:

### Faiss Base Image
The [Faiss repository](https://github.com/facebookresearch/faiss/) is added as a submodule in this repository. Run the below command to initialize the submodule first.
```
git submodule update --init
```
The Faiss base image can only be created on an NVIDIA GPU powered machine with CUDA Toolkit installed.

Please see the section [Provisioning an instance for development](#provisioning-an-instance-for-development) to provision an instance for development.

Run the below command to create the Faiss base image:
```
docker build  -f ./base_image/build_scripts/Dockerfile . -t opensearchstaging/remote-vector-index-builder:faiss-base-latest
```

### Core Image
The path [`/remote-vector-index-builder/core`](/remote_vector_index_builder/core/) contains the code for core index build functionalities:
1. Building an Index
2. Object Store I/O

Build an image with the above core functionalities:
```
docker build  -f ./remote_vector_index_builder/core/Dockerfile . -t opensearchstaging/remote-vector-index-builder:core-latest
```

## Provisioning an instance for development

A NVIDIA GPU powered machine with CUDA Toolkit installed is required to build a Faiss base image and to run the Docker images to build an index.

Typically an [EC2 G5](https://aws.amazon.com/ec2/instance-types/g5/) 2xlarge instance running a Deep Learning OSS Nvidia Driver AMI with Docker CLI installed is recommended for development use.

[Setup an EC2 Instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)

## Memory Profiling

Simple memory profiling can be done during development to get memory usage statistics during the Index Build process.

### GPU Memory Profiling with NVIDIA SMI

1. Install [py3nvml](https://pypi.org/project/py3nvml/): In [`/remote_vector_index_builder/core/requirements.txt`](/remote_vector_index_builder/core/requirements.txt) add `py3nvml` on a newline.

2. Add import statement and initialize method in the file containing the driver code.
```
from py3nvml import nvidia_smi
nvidia_smi.nvmlInit()
```

3. Define and call the below method wherever necessary. e.g. before and after calling the GPU index cleanup method.
```
from py3nvml import nvidia_smi

def get_gpu_memory():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # GPU Device ID
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total: {info.total/1024**2:.2f}MB")
    print(f"Free: {info.free/1024**2:.2f}MB")
    print(f"Used: {info.used/1024**2:.2f}MB")

```

### CPU Memory Profiling with memory_profiler

1. Add the below command in [`/remote_vector_index_builder/core/Dockerfile`](/remote_vector_index_builder/core/Dockerfile) to install [memory_profiler](https://pypi.org/project/memory-profiler/).
```
RUN conda install -c conda-forge memory_profiler -y
```

2. In the file that contains the function that needs to be profiled, add the import and an `@profile` annotation on the function.
```
from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a
```
