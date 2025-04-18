FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

WORKDIR /e2e

COPY ./remote_vector_index_builder/core /remote_vector_index_builder/core

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/core/requirements.txt

COPY ./e2e/test_core /e2e/test_core

RUN pip install --no-cache-dir --upgrade -r /e2e/test_core/requirements.txt

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder:/'
RUN ["python", "test_core/test_imports.py"]
CMD ["python", "test_core/run_e2e.py"]