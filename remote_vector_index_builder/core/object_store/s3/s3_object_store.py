# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import logging
import os
import threading
from functools import cache
import math
from io import BytesIO
import sys
from typing import Any, Dict, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from core.common.exceptions import BlobError
from core.common.models.index_build_parameters import IndexBuildParameters
from core.object_store.object_store import ObjectStore

logger = logging.getLogger(__name__)


def get_cpus(factor: float) -> int:
    """Get the number of CPUs to use for s3 upload or download operation

    Args:
        factor (float): The factor to multiply total cpu count by

    Returns:
        int: The number of CPUs that will be used for s3 upload or download

    The cpu count is rounded down to the nearest integer
    """

    # according to mypy, os.cpu_count can be None
    # if it is none, then default to 1 thread
    cpu_count = os.cpu_count()
    if cpu_count:
        return max(1, math.floor(cpu_count * factor))
    return 1


@cache
def get_boto3_client(
    region: str, retries: int, endpoint_url: Optional[str] = None
) -> boto3.client:
    """Create or retrieve a cached boto3 S3 client.

    Args:
        region (str): AWS region name for the S3 client
        retries (int): Maximum number of retry attempts for failed requests
        endpoint_url (str): s3 endpoint URL. Defaults to None, in which case boto3
            automatically constructs the appropriate URL to use when communicating
            with a service. During integration testing, this can be set to the endpoint URL
            for LocalStack S3 service.

    Returns:
        boto3.client: Configured S3 client instance
    """
    config = Config(retries={"max_attempts": retries})
    return boto3.client(
        "s3",
        config=config,
        region_name=region,
        endpoint_url=endpoint_url,
    )


class S3ObjectStore(ObjectStore):
    """S3 implementation of the ObjectStore interface for managing vector data files.

    This class handles interactions with AWS S3, including file uploads and downloads,
    with configurable retry logic and transfer settings for optimal performance.

    Attributes:
        DEFAULT_DOWNLOAD_TRANSFER_CONFIG (dict): Default configuration for S3 file downloads,
            including chunk sizes and concurrency
        DEFAULT_UPLOAD_TRANSFER_CONFIG (dict): Default configuration for S3 file uploads,
            including chunk sizes and concurrency
        DEFAULT_DOWNLOAD_ARGS (dict): Default boto3 ALLOWED_DOWNLOAD_ARGS values.
            Includes encryption and checksum settings.
        DEFAULT_UPLOAD_ARGS (dict): Default boto3 ALLOWED_UPLOAD_ARGS values.
            Includes encryption and checksum settings

    Args:
        index_build_params (IndexBuildParameters): Parameters for the index building process
        object_store_config (Dict[str, Any]): Configuration options for S3 interactions
    """

    def __init__(
        self,
        index_build_params: IndexBuildParameters,
        object_store_config: Dict[str, Any],
    ):
        """Initialize the S3ObjectStore with the given parameters and configuration.

        Args:
            index_build_params (IndexBuildParameters): Contains bucket name and other
                index building parameters
            object_store_config (Dict[str, Any]): Configuration dictionary containing:
                - retries (int): Maximum number of retry attempts (default: 3)
                - region (str): AWS region name (default: 'us-west-2')
                - transfer_config (Dict[str, Any]): s3 TransferConfig parameters
                - debug: Turns on debug mode (default: False)
        """

        self.DEFAULT_DOWNLOAD_TRANSFER_CONFIG = {
            "multipart_chunksize": 50 * 1024 * 1024,  # 50MB
            "max_concurrency": get_cpus(factor=0.625),
            "multipart_threshold": 50 * 1024 * 1024,  # 50MB
            "io_chunksize": sys.maxsize,
        }

        self.DEFAULT_UPLOAD_TRANSFER_CONFIG = {
            "multipart_chunksize": 50 * 1024 * 1024,  # 50MB
            "max_concurrency": get_cpus(factor=0.25),
            "multipart_threshold": 50 * 1024 * 1024,  # 50MB
        }

        self.DEFAULT_DOWNLOAD_ARGS = {
            "ChecksumMode": "ENABLED",
        }

        self.DEFAULT_UPLOAD_ARGS = {
            "ChecksumAlgorithm": "CRC32",
        }
        self.bucket = index_build_params.container_name
        self.max_retries = object_store_config.get("retries", 3)
        self.region = object_store_config.get("region", "us-west-2")

        self.s3_client = get_boto3_client(
            region=self.region,
            retries=self.max_retries,
            endpoint_url=object_store_config.get("S3_ENDPOINT_URL"),
        )

        download_transfer_config = object_store_config.get(
            "download_transfer_config", {}
        )
        # Create download transfer config
        # This is passed as the 'Config' parameter to the boto3 download API call
        self.download_transfer_config = S3ObjectStore._create_custom_config(
            download_transfer_config, self.DEFAULT_DOWNLOAD_TRANSFER_CONFIG
        )

        download_args = object_store_config.get("download_args", {})
        # Create download args
        # This is passed as the 'ExtraArgs' parameter to the boto3 download API call
        self.download_args = S3ObjectStore._create_custom_config(
            download_args, self.DEFAULT_DOWNLOAD_ARGS
        )

        upload_transfer_config = object_store_config.get("upload_transfer_config", {})
        # Create upload transfer config
        # This is passed as the 'Config' parameter to the boto3 upload API call
        self.upload_transfer_config = S3ObjectStore._create_custom_config(
            upload_transfer_config, self.DEFAULT_UPLOAD_TRANSFER_CONFIG
        )
        upload_args = object_store_config.get("upload_args", {})
        # Create upload args
        # This is passed as the 'ExtraArgs' parameter to the boto3 upload API call
        self.upload_args = S3ObjectStore._create_custom_config(
            upload_args, self.DEFAULT_UPLOAD_ARGS
        )

        self.debug = object_store_config.get("debug", False)

        # Debug mode provides progress tracking on downloads and uploads
        if self.debug:
            self._read_progress = 0
            self._read_progress_lock = threading.Lock()
            self._write_progress = 0
            self._write_progress_lock = threading.Lock()

    @staticmethod
    def _create_custom_config(
        custom_config: Dict[str, Any], default_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merges custom boto3 configuration parameters with default values, ensuring any
        unspecified parameters retain their default settings.

        Args:
            custom_config (dict): User-provided configuration parameters to override defaults
            default_config (dict): Base configuration parameters that will be used if not
                specified in custom_config

        Returns:
            dict: A merged configuration dictionary suitable for boto3 TransferConfig
                or ExtraArgs parameters, where custom values take precedence over defaults
        """

        # Start with default values
        config_params = default_config.copy()
        # merge with custom_config values
        config_params.update(custom_config)

        return config_params

    def read_blob(self, remote_store_path: str, bytes_buffer: BytesIO) -> None:
        """
        Downloads a blob from S3 to the provided bytes buffer, with retry logic.

        Args:
            remote_store_path (str): The S3 key (path) of the object to download
            bytes_buffer (BytesIO): A bytes buffer to store the downloaded data

        Returns:
            None

        Note:
            - boto3 automatically handles retries for the exceptions given here:
                - https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html
            - Resets buffer position to 0 after successful download
            - Uses configured TransferConfig for download parameters
                - boto3 may perform the download in parallel multipart chunks,
                based on the TransferConfig setting

        Raises:
            BlobError: If download fails after all retry attempts or encounters non-retryable error
        """

        callback_func = None

        # Set up progress callback, if debug mode is on
        if self.debug:
            with self._read_progress_lock:
                self._read_progress = 0

            def callback(bytes_transferred):
                with self._read_progress_lock:
                    self._read_progress += bytes_transferred
                    logger.info(f"Downloaded: {self._read_progress:,} bytes")

            callback_func = callback

        try:
            # Create transfer config object
            s3_transfer_config = TransferConfig(**self.download_transfer_config)

            self.s3_client.download_fileobj(
                self.bucket,
                remote_store_path,
                bytes_buffer,
                Config=s3_transfer_config,
                Callback=callback_func,
                ExtraArgs=self.download_args,
            )
            return
        except TypeError as e:
            raise BlobError(f"Error calling boto3.download_fileobj: {e}") from e
        except ClientError as e:
            raise BlobError(f"Error downloading file: {e}") from e

    def write_blob(self, local_file_path: str, remote_store_path: str) -> None:
        """
        Uploads a local file to S3, with retry logic.

        Args:
            local_file_path (str): Path to the local file to be uploaded
            remote_store_path (str): The S3 key (path) where the file will be stored

        Returns:
            None

        Note:
            - boto3 automatically handles retries for the exceptions given here:
                - https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html
            - Uses configured TransferConfig for upload parameters
                - boto3 may perform the upload in parallel multipart chunks, based on the TransferConfig setting

        Raises:
            BlobError: If upload fails after all retry attempts or encounters a non-retryable error
        """

        callback_func = None
        if self.debug:
            # Set up progress callback, if debug mode is on
            with self._write_progress_lock:
                self._write_progress = 0

            def callback(bytes_amount):
                with self._write_progress_lock:
                    self._write_progress += bytes_amount
                    logger.info(f"Uploaded: {self._write_progress:,} bytes")

            callback_func = callback

        try:
            # Create transfer config object
            s3_transfer_config = TransferConfig(**self.upload_transfer_config)

            self.s3_client.upload_file(
                local_file_path,
                self.bucket,
                remote_store_path,
                Config=s3_transfer_config,
                Callback=callback_func,
                ExtraArgs=self.upload_args,
            )
            return
        except TypeError as e:
            raise BlobError(f"Error calling boto3.upload_file: {e}") from e
        except ClientError as e:
            raise BlobError(f"Error uploading file: {e}") from e
