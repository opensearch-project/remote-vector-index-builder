FROM opensearchstaging/remote-vector-index-builder:core-latest

WORKDIR /benchmarking

COPY requirements.txt /benchmarking/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /benchmarking/requirements.txt

COPY . /benchmarking

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder:/'
RUN ["python", "test_imports.py"]
RUN ["mkdir", "/benchmarking/files"]
RUN ["chown", "-R", "appuser:appuser", "/benchmarking/files"]
RUN ["chmod", "-R", "777", "/benchmarking/files"]
CMD ["python", "main.py"]