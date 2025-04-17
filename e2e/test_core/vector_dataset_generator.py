import os
import numpy as np
import yaml
import time
from botocore.exceptions import ClientError
from core.common.models import IndexBuildParameters
from core.object_store.object_store_factory import ObjectStoreFactory
from core.object_store.types import ObjectStoreType
from core.tasks import create_vectors_dataset, build_index, upload_index, run_tasks

# Run full workflow w/o error, assert does not raise

# Test failure of create_vectors_dataset when data uploaded with checksum failure,
# read attempt from s3 fails with checksum error

# Test create_vectors, build_index, upload_index pass successfully


class VectorDatasetGenerator:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.object_store = self.initialize_object_store()

    def initialize_object_store(self):
        s3_config = self.config["storage"]["s3"]

        index_build_params = IndexBuildParameters(
            vector_path="vectos.knnvec",  # Will be set per dataset
            doc_id_path="ids.knndid",  # Will be set per dataset
            repository_type=ObjectStoreType.S3,
            container_name=s3_config["bucket"],
            dimension=128,  # Default dimension, will be overridden per dataset
            doc_count=5,  # Will be set per dataset
        )
        object_store_config = {
            "retries": s3_config["retries"],
            "region": s3_config["region"],
            "S3_ENDPOINT_URL": os.environ.get(
                "S3_ENDPOINT_URL", "http://localhost:4566"
            ),
        }
        return ObjectStoreFactory.create_object_store(
            index_build_params, object_store_config
        )

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def generate_vectors(self, dataset_name):
        """Generate test vector and doc ID data of specified size"""

        start_time = time.time()

        dataset_config = self.config["datasets"][dataset_name]
        gen_config = self.config["generation"]

        n_vectors = dataset_config["num_vectors"]
        dimension = dataset_config["dimension"]
        batch_size = gen_config["batch_size"]

        # Generate vectors in batches

        vectors_list = []
        doc_ids_list = []
        for i in range(0, n_vectors, batch_size):
            batch_size_current = min(batch_size, n_vectors - i)

            # Generate batch
            dist_params = dataset_config["distribution"]
            data_type = dataset_config["data_type"]

            batch = np.random.normal(
                dist_params["mean"], dist_params["std"], (batch_size_current, dimension)
            )

            if dist_params["normalize"]:
                batch = batch / np.linalg.norm(batch, axis=1)[:, np.newaxis]
            batch = batch.astype(data_type)
            vectors_list.append(batch)

            doc_ids = np.arange(i, i + batch_size_current, dtype=np.int32)
            doc_ids_list.append(doc_ids)

        # Combine batches
        vectors = np.concatenate(vectors_list)

        # Generate doc IDs (this is very fast)
        doc_ids = np.concatenate(doc_ids_list)
        # doc_ids = np.arange(n_vectors, dtype=np.int32)

        total_time = time.time() - start_time
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Vectors Memory usage: {vectors.nbytes / (1024**3):.2f}GB")
        print(f"Doc Ids Memory usage: {doc_ids.nbytes / (1024**2):.2f}MB")

        return vectors, doc_ids

    def upload_dataset(self, dataset_name, vectors, doc_ids):
        """Upload vectors and doc_ids directly to S3 using the S3 client"""
        s3_config = self.config["storage"]["s3"]

        # Get paths
        vector_path = s3_config["paths"]["vectors"].format(dataset_name=dataset_name)
        doc_id_path = s3_config["paths"]["doc_ids"].format(dataset_name=dataset_name)

        # Convert numpy arrays to bytes
        vectors_bytes = vectors.tobytes()
        doc_ids_bytes = doc_ids.tobytes()

        # Get the S3 client from the object store
        s3_client = self.object_store.s3_client

        # Upload to S3
        try:
            s3_client.put_object(
                Bucket=s3_config["bucket"], Key=vector_path, Body=vectors_bytes
            )
            print(f"Uploaded vectors to {vector_path}")

            s3_client.put_object(
                Bucket=s3_config["bucket"], Key=doc_id_path, Body=doc_ids_bytes
            )
            print(f"Uploaded doc_ids to {doc_id_path}")

        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            raise

    def generate_and_upload_dataset(self, dataset_name):
        """Generate and upload a single dataset"""
        print(f"\nProcessing dataset: {dataset_name}")

        try:
            # Generate vectors
            vectors, doc_ids = self.generate_vectors(dataset_name)

            self.upload_dataset(dataset_name, vectors, doc_ids)

            del vectors, doc_ids

            print(f"Successfully generated and uploaded {dataset_name}")
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            raise
