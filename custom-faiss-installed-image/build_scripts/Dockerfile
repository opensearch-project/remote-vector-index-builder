FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install base utilities and also the dependencies for faiss
RUN apt-get update \
    && apt-get install -y build-essential gfortran libblas-dev libopenblas-base zlib1g zlib1g-dev git tmux libssl-dev \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser 

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN chown -R appuser:appuser /opt
COPY faiss-test.py /tmp
RUN mkdir /tmp/faiss
COPY faiss /tmp/faiss
COPY build-faiss.sh /tmp/build-faiss.sh

RUN chown -R appuser:appuser /tmp
#RUN chmod -R 755 /opt

USER appuser
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
#install some necessary dependencies
RUN conda install -y -q python=3.11 cmake=3.26 make=4.2 swig=4.0 "numpy<2" scipy=1.14 pytest=7.4 gflags=2.2
RUN cmake --version

# special for 12.1.1
ENV CUDA_ARCHS "70-real;72-real;75-real;80;86-real"

# install base packages for X86_64
RUN conda install -y -q -c conda-forge gxx_linux-64=14.2 sysroot_linux-64=2.17

# install all the dependencies that we think we will need for installing faiss
# https://github.com/facebookresearch/faiss/blob/3c8dc4194907e9b911551d5a009468106f8b9c7f/.github/actions/build_cmake/action.yml#L57C31-L57C44
RUN conda install -y -q libcuvs=24.12 cuda-nvcc gxx_linux-64=12.4 -c rapidsai -c conda-forge

RUN /tmp/build-faiss.sh

CMD ["sh", "-c", "nvidia-smi && sleep 36000"]
