FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

WORKDIR /remote_vector_index_builder

COPY requirements.txt /remote_vector_index_builder/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /remote_vector_index_builder/requirements.txt

COPY . /remote_vector_index_builder

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder'
RUN ["python", "test_imports.py"]