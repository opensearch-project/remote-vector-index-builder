# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

ARG BASE_IMAGE_TAG=faiss-base-snapshot
FROM opensearchstaging/remote-vector-index-builder:${BASE_IMAGE_TAG}

USER root

WORKDIR /app

COPY ./remote_vector_index_builder/core /app/remote_vector_index_builder/core

COPY ./remote_vector_index_builder/app /app/remote_vector_index_builder/app

RUN pip install --no-cache-dir --upgrade -r /app/remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/remote_vector_index_builder/app/requirements.txt

COPY ./remote_vector_index_builder/__init__.py /app/remote_vector_index_builder/__init__.py

RUN pip install -e ./remote_vector_index_builder/core
RUN pip install -e ./remote_vector_index_builder/app

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN chown -R appuser:appuser /app
USER appuser

RUN ["python", "./remote_vector_index_builder/app/test_imports.py"]

CMD ["fastapi", "run", "./remote_vector_index_builder/app/main.py", "--port", "1025"]