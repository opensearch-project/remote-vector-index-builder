FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

WORKDIR /e2e

COPY ./remote_vector_index_builder/core /remote_vector_index_builder/core

COPY ./remote_vector_index_builder/app /remote_vector_index_builder/app

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/app/requirements.txt

COPY ./e2e/test_core /e2e/test_core

RUN pip install --no-cache-dir --upgrade -r /e2e/test_core/requirements.txt

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder:/'

RUN ["python", "../remote_vector_index_builder/core/test_imports.py"]
RUN ["python", "../remote_vector_index_builder/app/test_imports.py"]

RUN ["python", "test_core/test_imports.py"]