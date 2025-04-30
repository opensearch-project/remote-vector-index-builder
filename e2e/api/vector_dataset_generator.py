# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import os
import numpy as np
import yaml
import time
from botocore.exceptions import ClientError
from core.common.models import IndexBuildParameters
from core.object_store.object_store_factory import ObjectStoreFactory
from core.object_store.types import ObjectStoreType
import logging
from tqdm import tqdm

class VectorDatasetGenerator:
    """
    Class to generate dummy vectors and injest in the object store, required for running e2e tests
    """

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

        start_time = time.time()

        dataset_config = self.config["datasets"][dataset_name]
        gen_config = self.config["generation"]

        n_vectors = dataset_config["num_vectors"]
        dimension = dataset_config["dimension"]
        batch_size = gen_config["batch_size"]

        # Generate vectors in batches

        vectors_list = []
        doc_ids_list = []
        for i in tqdm(range(0, n_vectors, batch_size)):
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

        vectors = np.concatenate(vectors_list)
        doc_ids = np.concatenate(doc_ids_list)

        total_time = time.time() - start_time
        metrics = {
            "total_time": total_time,
            "vectors_memory": f"{vectors.nbytes / (1024**3):.2f}GB",
            "doc_ids_memory": f"{doc_ids.nbytes / (1024**2):.2f}MB",
        }

        return vectors, doc_ids, metrics

    def upload_dataset(self, dataset_name, vectors, doc_ids):
        logger = logging.getLogger(__name__)

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
            start_time = time.time()
            s3_client.put_object(
                Bucket=s3_config["bucket"], Key=vector_path, Body=vectors_bytes
            )

            s3_client.put_object(
                Bucket=s3_config["bucket"], Key=doc_id_path, Body=doc_ids_bytes
            )
            metrics = {"total_time": time.time() - start_time}
            return metrics

        except ClientError as e:
            logger.exception(f"Error uploading dataset {dataset_name} to S3: {e}")
            raise

    def generate_and_upload_dataset(self, dataset_name):
        logger = logging.getLogger(__name__)
        """Generate and upload a single dataset"""

        try:
            # Generate vectors
            vectors, doc_ids, gen_metrics = self.generate_vectors(dataset_name)

            upload_metrics = self.upload_dataset(dataset_name, vectors, doc_ids)

            del vectors, doc_ids
            logger.info(f"Successfully generated and uploaded {dataset_name}")

            return {"generation": gen_metrics, "upload": upload_metrics}
        except Exception as e:
            logger.exception(f"Error processing {dataset_name}: {str(e)}")
            raise
