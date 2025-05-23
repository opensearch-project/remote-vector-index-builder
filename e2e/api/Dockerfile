FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest
USER root
WORKDIR /

COPY ./remote_vector_index_builder/core /remote_vector_index_builder/core

COPY ./remote_vector_index_builder/app /remote_vector_index_builder/app

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/app/requirements.txt

COPY ./e2e/api /e2e/api

RUN pip install --no-cache-dir --upgrade -r /e2e/api/requirements.txt

COPY ./e2e/setup.py /e2e/setup.py

RUN pip install -e ./remote_vector_index_builder/core
RUN pip install -e ./remote_vector_index_builder/app
RUN pip install -e ./e2e

WORKDIR /e2e
RUN ["python", "../remote_vector_index_builder/core/test_imports.py"]
RUN ["python", "../remote_vector_index_builder/app/test_imports.py"]

RUN ["python", "api/test_imports.py"]