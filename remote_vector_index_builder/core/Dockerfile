ARG BASE_IMAGE_TAG=faiss-base-snapshot
FROM opensearchstaging/remote-vector-index-builder:${BASE_IMAGE_TAG}
USER root

WORKDIR /app

COPY ./remote_vector_index_builder/core/requirements.txt /app/remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/remote_vector_index_builder/core/requirements.txt

COPY ./remote_vector_index_builder/core /app/remote_vector_index_builder/core

RUN pip install -e ./remote_vector_index_builder/core

COPY ./remote_vector_index_builder/__init__.py /app/remote_vector_index_builder/__init__.py

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN chown -R appuser:appuser /app
USER appuser

RUN ["python", "./remote_vector_index_builder/core/test_imports.py"]