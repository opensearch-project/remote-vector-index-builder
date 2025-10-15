# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod
from typing import Union
from io import BytesIO


class ObjectStore(ABC):
    """
    Abstract base class defining the interface for object storage operations.

    This class provides a common interface for reading and writing blobs to various
    object storage implementations (e.g., S3, local filesystem, etc.).

    All concrete implementations of object storage must inherit from this class
    and implement the abstract methods.
    """

    @abstractmethod
    def read_blob(self, remote_store_path: str, bytes_buffer: BytesIO) -> None:
        """
        Downloads the blob from the remote_store_path, to a buffer in memory

        Args:
            remote_store_path (str): The path/key to the remote object to be downloaded
            bytes_buffer (BytesIO): A bytes buffer where the downloaded data will be stored

        Returns:
            None

        Note:
            - The bytes_buffer should be properly initialized before passing to this method
            - Caller is also responsible for cleaning up the bytes buffer
            - Implementations should handle any necessary authentication and error handling
        """
        pass

    @abstractmethod
    def write_blob(self, data: Union[str, BytesIO], remote_store_path: str) -> None:
        """
        Uploads the blob at local_file_path to the remote_store_path

        Args:
            data (Union[str, BytesIO]): Either a string representing a local file path
                or a BytesIO object representing a buffer
            remote_store_path (str): The path/key where the file should be stored in remote storage

        Returns:
            None

        Note:
            - Implementations should handle any necessary authentication and error handling
            - The local file must exist and be readable
            - The remote path should be valid for the specific storage implementation
        """
        pass
