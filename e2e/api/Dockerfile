ARG BASE_IMAGE_TAG=faiss-base-snapshot
FROM opensearchstaging/remote-vector-index-builder:${BASE_IMAGE_TAG}
USER root

WORKDIR /app

COPY ./remote_vector_index_builder/core /app/remote_vector_index_builder/core

COPY ./remote_vector_index_builder/app /app/remote_vector_index_builder/app

RUN pip install --no-cache-dir --upgrade -r /app/remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/remote_vector_index_builder/app/requirements.txt

COPY ./e2e/api /app/e2e/api

RUN pip install --no-cache-dir --upgrade -r /app/e2e/api/requirements.txt

COPY ./e2e/setup.py /app/e2e/setup.py
COPY ./remote_vector_index_builder/__init__.py /app/remote_vector_index_builder/__init__.py

RUN pip install -e ./remote_vector_index_builder/core
RUN pip install -e ./remote_vector_index_builder/app
RUN pip install -e ./e2e

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN chown -R appuser:appuser /app
USER appuser

RUN ["python", "./remote_vector_index_builder/core/test_imports.py"]
RUN ["python", "./remote_vector_index_builder/app/test_imports.py"]

RUN ["python", "./e2e/api/test_imports.py"]