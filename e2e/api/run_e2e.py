# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import logging
import time
import sys
import os
from botocore.exceptions import ClientError
from core.common.models.index_build_parameters import DataType, Engine
from core.object_store.types import ObjectStoreType
from e2e.api.remote_vector_api_client import RemoteVectorAPIClient
from e2e.api.utils.logging_config import configure_logger
from e2e.api.vector_dataset_generator import VectorDatasetGenerator

from app.models.job import JobStatus


def run_e2e_index_builder(config_path: str = "e2e/api/test-datasets.yml"):

    logger = logging.getLogger(__name__)
    dataset_generator = VectorDatasetGenerator(config_path)

    def process_dataset(dataset_name):
        """Process a single dataset in parallel."""
        logger.info(f"\n=== Processing dataset: {dataset_name} ===")

        try:
            gen_and_upload_metrics = dataset_generator.generate_and_upload_dataset(
                dataset_name
            )

            dataset_config = dataset_generator.config["datasets"][dataset_name]
            s3_config = dataset_generator.config["storage"]["s3"]

            index_build_params = {
                "vector_path": s3_config["paths"]["vectors"].format(
                    dataset_name=dataset_name
                ),
                "doc_id_path": s3_config["paths"]["doc_ids"].format(
                    dataset_name=dataset_name
                ),
                "container_name": bucket,
                "dimension": dataset_config["dimension"],
                "doc_count": dataset_config["num_vectors"],
                "data_type": DataType.FLOAT,
                "repository_type": ObjectStoreType.S3,
                "engine": Engine.FAISS,
            }

            logger.info(f"Running vector index builder workflow for {dataset_name}...")

            client.heart_beat()

            start_time = time.time()
            # Submit job
            job_id = client.build_index(index_build_params)
            logger.info(f"Created job: {job_id} for dataset: {dataset_name}")

            jobs = json.loads(client.get_jobs())
            logger.info(f"Jobs: {jobs}")
            if job_id not in jobs:
                logger.error(f"Error in workflow: Job {job_id} not found")
                raise RuntimeError("Job not found")

            # Wait for completion (20 minute timeout)
            result = client.wait_for_job_completion(
                job_id,
                status_request_timeout=1200,  # 20 minutes
                interval=10,  # Check every 10 seconds
            )
            run_tasks_total_time = time.time() - start_time

            if result.task_status != JobStatus.COMPLETED:
                logger.error(
                    f"Error in workflow for {dataset_name}: {result.error_message}"
                )
                raise RuntimeError(
                    f"Job failed for {dataset_name}: {result.error_message}"
                )

            vector_dataset_name = ".".join(
                os.path.basename(index_build_params["vector_path"]).split(".")[0:-1]
            )
            index_file_name = vector_dataset_name + "." + index_build_params["engine"]
            if result.file_name != index_file_name:
                error_msg = (
                    f"Error in workflow for {dataset_name}: "
                    f"Vector file upload path mismatch, "
                    f"expected:{index_file_name}, got: {result.file_name}"
                )
                logger.error(error_msg)
                raise RuntimeError(f"Job Failed for {dataset_name}: {error_msg}")

            logger.info(f"Successfully processed dataset: {dataset_name}")
            metrics = {
                "dataset": gen_and_upload_metrics,
                "run_tasks_total_time": run_tasks_total_time,
            }
            logger.info(f"Metrics for {dataset_name}: {metrics}")
            return dataset_name, True, metrics
        except Exception as e:
            logger.exception(f"Error processing dataset {dataset_name}: {str(e)}")
            return dataset_name, False, str(e)

    bucket = None
    client = RemoteVectorAPIClient(http_request_timeout=30)
    try:
        # Create test bucket if it doesn't exist
        s3_client = dataset_generator.object_store.s3_client
        bucket = dataset_generator.config["storage"]["s3"]["bucket"]
        try:
            s3_client.create_bucket(Bucket=bucket)
            logger.info(f"Created bucket: {bucket}")
        except s3_client.exceptions.BucketAlreadyExists:
            logger.info(f"Using existing bucket: {bucket}")

        # Process datasets in parallel
        all_metrics = {}
        total_start_time = time.time()

        with ThreadPoolExecutor() as executor:
            # Submit all dataset processing tasks to the executor
            future_to_dataset = {
                executor.submit(process_dataset, dataset_name): dataset_name
                for dataset_name in dataset_generator.config["datasets"]
            }

            # Process results as they complete
            all_succeeded = True
            for future in as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                try:
                    ds_name, success, result = future.result()
                    if success:
                        all_metrics[ds_name] = result
                    else:
                        all_succeeded = False
                        logger.error(f"Dataset {ds_name} failed: {result}")
                except Exception as e:
                    all_succeeded = False
                    logger.exception(
                        f"Exception processing dataset {dataset_name}: {str(e)}"
                    )

        total_execution_time = time.time() - total_start_time
        logger.info(f"Total parallel execution time: {total_execution_time:.2f}s")

        if not all_succeeded:
            raise RuntimeError("One or more datasets failed to process.")
        jobs = json.loads(client.get_jobs())
        if len(jobs) != len(dataset_generator.config["datasets"]):
            logger.error("Error in workflow: Not all jobs found")
            raise RuntimeError("Not all jobs found")

        logger.info("All datasets processed successfully.")

    except Exception as e:
        logger.error(f"E2E test failed: {str(e)}")
        sys.exit(1)  # Exit with non-zero status
    finally:
        logger.info("\n=== Cleaning up ===")
        try:
            # Delete all objects in bucket
            response = s3_client.list_objects_v2(Bucket=bucket)
            if "Contents" in response:
                for obj in response["Contents"]:
                    s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
                    logger.info(f"Deleted: {obj['Key']}")

            # Delete bucket
            s3_client.delete_bucket(Bucket=bucket)
            logger.info(f"Deleted bucket: {bucket}")

        except ClientError as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    configure_logger()
    run_e2e_index_builder("e2e/api/test-datasets.yml")
