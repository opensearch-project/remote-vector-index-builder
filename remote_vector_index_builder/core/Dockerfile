FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

WORKDIR /remote_vector_index_builder

COPY ./remote_vector_index_builder/core/requirements.txt /remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/core/requirements.txt

COPY ./remote_vector_index_builder/core /remote_vector_index_builder/core

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder'
RUN ["python", "core/test_imports.py"]