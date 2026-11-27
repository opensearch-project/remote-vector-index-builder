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
import tempfile
import time
import sys
import os
import faiss
from botocore.exceptions import ClientError
from core.common.models.index_build_parameters import DataType, Engine
from core.object_store.types import ObjectStoreType
from e2e.api.remote_vector_api_client import RemoteVectorAPIClient
from e2e.api.utils.logging_config import configure_logger
from e2e.api.vector_dataset_generator import VectorDatasetGenerator

from app.models.job import JobStatus


def verify_built_index(
    s3_client, bucket: str, index_key: str, binary: bool, expect_base_level_only: bool
) -> dict:
    """
    Downloads a built index and verifies what the service wrote.

    It currently checks the graph levels: the base_level_only flag stored in the index
    and, for an index with upper levels, the entry point and top level that faiss sets
    while it builds them.

    Args:
        s3_client: boto3 S3 client
        bucket: Bucket the index was uploaded to
        index_key: Key of the index blob
        binary: True for an index of binary vectors
        expect_base_level_only: The value of base_level_only the index must have

    Returns:
        dict: the properties read from the index (base_level_only, entry_point, max_level)

    Raises:
        RuntimeError: If the index does not match what is expected
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, os.path.basename(index_key))
        s3_client.download_file(bucket, index_key, local_path)
        # Downcast faiss index to reach the CAGRA-specific fields.
        if binary:
            index_id_map = faiss.read_index_binary(local_path)
            cpu_index = faiss.downcast_IndexBinary(index_id_map.index)
        else:
            index_id_map = faiss.read_index(local_path)
            cpu_index = faiss.downcast_index(index_id_map.index)

    index_properties = {
        "base_level_only": bool(cpu_index.base_level_only),
        "entry_point": int(cpu_index.hnsw.entry_point),
        "max_level": int(cpu_index.hnsw.max_level),
    }
    del cpu_index, index_id_map
    if index_properties["base_level_only"] != expect_base_level_only:
        raise RuntimeError(
            f"Expected base_level_only={expect_base_level_only} in {index_key}, got {index_properties}"
        )
    if not expect_base_level_only and (
        index_properties["entry_point"] < 0 or index_properties["max_level"] < 1
    ):
        raise RuntimeError(
            f"Expected an entry point and upper levels in {index_key}, got {index_properties}"
        )
    return index_properties


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
                # request data type defaults to float; half_float datasets set it explicitly
                "data_type": DataType(
                    dataset_config.get("request_data_type", DataType.FLOAT.value)
                ),
                "repository_type": ObjectStoreType.S3,
                "engine": Engine.FAISS,
            }
            # Pass the dataset's optional algorithm_parameters to the request
            algorithm_parameters = dataset_config.get("algorithm_parameters")
            if algorithm_parameters:
                index_build_params["index_parameters"] = {
                    "algorithm_parameters": algorithm_parameters
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

            metrics = {
                "dataset": gen_and_upload_metrics,
                "run_tasks_total_time": run_tasks_total_time,
            }

            # Check that the built index has the expected shape for datasets that
            # set expect_base_level_only
            expect_base_level_only = dataset_config.get("expect_base_level_only")
            if expect_base_level_only is not None:
                index_key = os.path.join(
                    os.path.dirname(index_build_params["vector_path"]),
                    index_file_name,
                )
                metrics["built_index"] = verify_built_index(
                    s3_client,
                    bucket,
                    index_key,
                    index_build_params["data_type"] == DataType.BINARY,
                    expect_base_level_only,
                )

            logger.info(f"Successfully processed dataset: {dataset_name}")
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

        # Datasets within a group run in parallel; groups run one after another, so the peak
        # load on the runner is bounded by the largest group rather than by every dataset at once
        all_metrics = {}
        all_succeeded = True
        total_start_time = time.time()

        datasets = dataset_generator.config["datasets"]
        groups = list(
            dict.fromkeys(cfg.get("group", "default") for cfg in datasets.values())
        )
        for group in groups:
            logger.info(f"\n=== Processing dataset group: {group} ===")
            with ThreadPoolExecutor() as executor:
                # Submit this group's dataset processing tasks to the executor
                future_to_dataset = {
                    executor.submit(process_dataset, dataset_name): dataset_name
                    for dataset_name, cfg in datasets.items()
                    if cfg.get("group", "default") == group
                }

                # Process results as they complete
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
