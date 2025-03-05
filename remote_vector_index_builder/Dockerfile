FROM rchitale7/remote-index-build-service:faiss-base

WORKDIR /remote_vector_index_builder

COPY core/requirements.txt /remote_vector_index_builder/core/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/core/requirements.txt

COPY . /remote_vector_index_builder

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder'

RUN ["python", "test_imports.py"]
CMD ["python", "core/main.py"]