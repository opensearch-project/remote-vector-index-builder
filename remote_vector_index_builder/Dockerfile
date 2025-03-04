FROM rchitale7/remote-index-build-service:faiss-base

WORKDIR /app

COPY core/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

CMD ["python", "test_imports.py"]