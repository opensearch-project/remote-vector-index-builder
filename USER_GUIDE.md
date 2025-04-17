# User Guide: OpenSearch Remote Vector Index Builder

## Getting Started

### 1. Provision an Instance
System requirements:
- NVIDIA GPU powered machine with CUDA Toolkit and Docker installed.
  - ex. AWS EC2 `g5.2xlarge` instance running Deep Learning OSS Nvidia Driver AMI.

### 2. Clone the Repository
```bash
git clone https://github.com/opensearch-project/remote-vector-index-builder.git
cd remote-vector-index-builder
```

## Building Docker Images

There are 3 images that need to be built:

### 1. Build Faiss Base Image
First, initialize the Faiss submodule:
```bash
git submodule update --init
```

Then build the Faiss base image:
```bash
docker build -f ./base_image/build_scripts/Dockerfile . -t opensearchstaging/remote-vector-index-builder:faiss-base-latest
```

### 2. Build Core Image
```bash
docker build -f ./remote_vector_index_builder/core/Dockerfile . -t opensearchstaging/remote-vector-index-builder:core-latest
```

### 3. Build API Image
```bash
docker build -f ./remote_vector_index_builder/app/Dockerfile . -t opensearchstaging/remote-vector-index-builder:api-latest
```

## Running the Service

### Starting the Docker Container
```bash
docker run --gpus all -p 80:1025 opensearchstaging/remote-vector-index-builder:api-latest
```

The service will be available at the instance's public IP address on port 80.
For information on configuring OpenSearch to use this service, please refer to the [OpenSearch k-NN documentation.](https://docs.opensearch.org/docs/latest/vector-search/)
