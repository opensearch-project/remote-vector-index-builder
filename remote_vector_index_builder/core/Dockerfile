FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

USER root

WORKDIR /remote_vector_index_builder

COPY ./remote_vector_index_builder/core/requirements.txt /remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/core/requirements.txt

COPY ./remote_vector_index_builder/core /remote_vector_index_builder/core

RUN pip install -e ./core

RUN ["python", "core/test_imports.py"]