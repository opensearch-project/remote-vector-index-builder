# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.


from core.common.models import IndexBuildParameters
from core.common.models.index_build_parameters import DataType
from core.object_store.types import ObjectStoreType
from e2e.test_core.utils.logging_config import configure_logger
from e2e.test_core.vector_dataset_generator import VectorDatasetGenerator
from botocore.exceptions import ClientError
from core.tasks import run_tasks
import os
import logging
import time
import sys


def run_e2e_index_builder(config_path: str = "test_core/test-datasets.yml"):

    logger = logging.getLogger(__name__)
    dataset_generator = VectorDatasetGenerator(config_path)

    try:
        # Create test bucket if it doesn't exist
        s3_client = dataset_generator.object_store.s3_client
        bucket = dataset_generator.config["storage"]["s3"]["bucket"]
        try:
            s3_client.create_bucket(Bucket=bucket)
            logger.info(f"Created bucket: {bucket}")
        except s3_client.exceptions.BucketAlreadyExists:
            logger.info(f"Using existing bucket: {bucket}")

        # Process each dataset
        for dataset_name in dataset_generator.config["datasets"]:
            logger.info(f"\n=== Processing dataset: {dataset_name} ===")

            try:
                gen_and_upload_metrics = dataset_generator.generate_and_upload_dataset(
                    dataset_name
                )

                dataset_config = dataset_generator.config["datasets"][dataset_name]
                s3_config = dataset_generator.config["storage"]["s3"]

                index_build_params = IndexBuildParameters(
                    vector_path=s3_config["paths"]["vectors"].format(
                        dataset_name=dataset_name
                    ),
                    doc_id_path=s3_config["paths"]["doc_ids"].format(
                        dataset_name=dataset_name
                    ),
                    container_name=bucket,
                    dimension=dataset_config["dimension"],
                    doc_count=dataset_config["num_vectors"],
                    data_type=DataType.FLOAT,
                    repository_type=ObjectStoreType.S3,
                )
                logger.info("\nRunning vector index builder workflow...")
                object_store_config = {
                    "retries": s3_config["retries"],
                    "region": s3_config["region"],
                    "S3_ENDPOINT_URL": os.environ.get(
                        "S3_ENDPOINT_URL", "http://localhost:4566"
                    ),
                }
                start_time = time.time()
                result = run_tasks(
                    index_build_params=index_build_params,
                    object_store_config=object_store_config,
                )
                run_tasks_total_time = time.time() - start_time

                if result.error:
                    logger.error(f"Error in workflow: {result.error}")
                    raise RuntimeError(f"Test failed for dataset {dataset_name}: {result.error}")

                logger.info(f"Successfully processed dataset: {dataset_name}")
                metrics = {
                    "dataset": gen_and_upload_metrics,
                    "run_tasks_total_time": run_tasks_total_time,
                }
                logger.info(metrics)

            except Exception as e:
                logger.exception(f"Error processing dataset {dataset_name}: {str(e)}")
                raise
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
    run_e2e_index_builder("test_core/test-datasets.yml")
