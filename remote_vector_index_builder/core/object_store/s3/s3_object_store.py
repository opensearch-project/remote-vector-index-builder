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
from io import BytesIO
from typing import Any, Dict

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from core.common.exceptions import BlobError
from core.common.models.index_build_parameters import IndexBuildParameters
from core.object_store.object_store import ObjectStore

logger = logging.getLogger(__name__)


@cache
def get_boto3_client(region: str, retries: int) -> boto3.client:
    """Create or retrieve a cached boto3 S3 client.

    Args:
        region (str): AWS region name for the S3 client
        retries (int): Maximum number of retry attempts for failed requests

    Returns:
        boto3.client: Configured S3 client instance
    """
    config = Config(retries={"max_attempts": retries})
    return boto3.client("s3", config=config, region_name=region)


class S3ObjectStore(ObjectStore):
    """S3 implementation of the ObjectStore interface for managing vector data files.

    This class handles interactions with AWS S3, including file uploads and downloads,
    with configurable retry logic and transfer settings for optimal performance.

    Attributes:
        DEFAULT_TRANSFER_CONFIG (dict): Default configuration for S3 file transfers,
            including chunk sizes, concurrency, and retry settings

    Args:
        index_build_params (IndexBuildParameters): Parameters for the index building process
        object_store_config (Dict[str, Any]): Configuration options for S3 interactions
    """

    DEFAULT_TRANSFER_CONFIG = {
        "multipart_chunksize": 10 * 1024 * 1024,  # 10MB
        "max_concurrency": (os.cpu_count() or 2)
        // 2,  # os.cpu_count can None, according to mypy. If it is none, then default to 1 thread
        "multipart_threshold": 10 * 1024 * 1024,  # 10MB
        "use_threads": True,
        "max_bandwidth": None,
        "io_chunksize": 256 * 1024,  # 256KB
        "num_download_attempts": 5,
        "max_io_queue": 100,
        "preferred_transfer_client": "auto",
    }

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
        self.bucket = index_build_params.container_name
        self.max_retries = object_store_config.get("retries", 3)
        self.region = object_store_config.get("region", "us-west-2")

        self.s3_client = get_boto3_client(region=self.region, retries=self.max_retries)

        transfer_config = object_store_config.get("transfer_config", {})
        # Create transfer config with validated parameters
        self.transfer_config = self._create_transfer_config(transfer_config)

        self.debug = object_store_config.get("debug", False)

        # Debug mode provides progress tracking on downloads and uploads
        if self.debug:
            self._read_progress = 0
            self._read_progress_lock = threading.Lock()
            self._write_progress = 0
            self._write_progress_lock = threading.Lock()

    def _create_transfer_config(self, custom_config: Dict[str, Any]) -> TransferConfig:
        """
        Creates a TransferConfig with custom parameters while maintaining defaults
        for unspecified values.

        Args:
            custom_config: Dictionary of custom transfer configuration parameters

        Returns:
            TransferConfig: Configured transfer configuration object
        """
        # Start with default values
        config_params = self.DEFAULT_TRANSFER_CONFIG.copy()

        # Update with custom values, only if they are valid parameters
        for key, value in custom_config.items():
            if key in self.DEFAULT_TRANSFER_CONFIG:
                config_params[key] = value
            else:
                logger.info(
                    f"Warning: Ignoring invalid transfer config parameter: {key}"
                )

        # Remove None values to let boto3 use its internal defaults
        config_params = {k: v for k, v in config_params.items() if v is not None}

        return TransferConfig(**config_params)

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
            self.s3_client.download_fileobj(
                self.bucket,
                remote_store_path,
                bytes_buffer,
                Config=self.transfer_config,
                Callback=callback_func,
            )
            return
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
            self.s3_client.upload_file(
                local_file_path,
                self.bucket,
                remote_store_path,
                Config=self.transfer_config,
                Callback=callback_func,
            )
            return
        except ClientError as e:
            raise BlobError(f"Error uploading file: {e}") from e
