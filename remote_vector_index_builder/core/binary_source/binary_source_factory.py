# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.binary_source.buffer_source import BufferSource
from core.binary_source.file_source import FileSource
from enum import Enum


class StorageMode(str, Enum):
    MEMORY = "memory"
    DISK = "disk"


class BinarySourceFactory:
    @staticmethod
    def create_binary_source(storage_mode: StorageMode):
        if storage_mode == StorageMode.MEMORY:
            return BufferSource()
        elif storage_mode == StorageMode.DISK:
            return FileSource()
        else:
            raise ValueError(f"Unsupported storage mode: {storage_mode}")
