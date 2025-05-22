FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

WORKDIR /e2e

COPY ./remote_vector_index_builder/core /remote_vector_index_builder/core

COPY ./remote_vector_index_builder/app /remote_vector_index_builder/app

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/app/requirements.txt

COPY ./e2e/api /e2e/api

RUN pip install --no-cache-dir --upgrade -r /e2e/api/requirements.txt

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder:/'

RUN ["python", "../remote_vector_index_builder/core/test_imports.py"]
RUN ["python", "../remote_vector_index_builder/app/test_imports.py"]

RUN ["python", "api/test_imports.py"]