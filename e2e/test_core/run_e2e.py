# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.


from core.common.models import IndexBuildParameters
from core.common.models.index_build_parameters import DataType
from core.object_store.types import ObjectStoreType
from e2e.test_core.vector_dataset_generator import VectorDatasetGenerator
from botocore.exceptions import ClientError
from core.tasks import run_tasks
import os


def run_e2e_index_builder(config_path: str):

    # run for each dataset in yml e2e test vector index builder
    # Teardown dataset and sucess artfacts from memory and s3

    # Ideally run this on a container so that memory gets freed up at end of tests
    # Assert docker container quit with exit 0 -> all tests suite exitted normally
    dataset_generator = VectorDatasetGenerator(config_path)

    try:
        # Creat etest bucket if it doesn't exist
        s3_client = dataset_generator.object_store.s3_client
        bucket = dataset_generator.config["storage"]["s3"]["bucket"]
        try:
            s3_client.create_bucket(Bucket=bucket)
            print(f"Created bucket: {bucket}")
        except s3_client.exceptions.BucketAlreadyExists:
            print(f"Using existing bucket: {bucket}")

        # Process each dataset
        for dataset_name in dataset_generator.config["datasets"]:
            print(f"\n=== Processing dataset: {dataset_name} ===")

            try:
                vectors, doc_ids = dataset_generator.generate_vectors(dataset_name)
                dataset_generator.upload_dataset(dataset_name, vectors, doc_ids)
                del vectors
                del doc_ids

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
                print("\nRunning vector index builder workflow...")
                object_store_config = {
                    "retries": s3_config["retries"],
                    "region": s3_config["region"],
                    "S3_ENDPOINT_URL": os.environ.get(
                        "S3_ENDPOINT_URL", "http://localhost:4566"
                    ),
                }
                result = run_tasks(
                    index_build_params=index_build_params,
                    object_store_config=object_store_config,
                )

                if result.error:
                    print(f"Error in workflow: {result.error}")
                    continue

                print(f"Successfully processed dataset: {dataset_name}")
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue
    finally:
        print("\n=== Cleaning up ===")
        try:
            # Delete all objects in bucket
            response = s3_client.list_objects_v2(Bucket=bucket)
            if "Contents" in response:
                for obj in response["Contents"]:
                    s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
                    print(f"Deleted: {obj['Key']}")

            # Delete bucket
            s3_client.delete_bucket(Bucket=bucket)
            print(f"Deleted bucket: {bucket}")

        except ClientError as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    run_e2e_index_builder("test_core/test-datasets.yml")
