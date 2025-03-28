# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

FROM opensearchstaging/remote-vector-index-builder:core-latest

WORKDIR /remote_vector_index_builder

COPY ./remote_vector_index_builder/app/requirements.txt /remote_vector_index_builder/app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/app/requirements.txt

COPY ./remote_vector_index_builder/app /remote_vector_index_builder/app

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder'
RUN ["python", "app/test_imports.py"]
CMD ["fastapi", "run", "app/main.py", "--port", "80"]