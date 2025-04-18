# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import pytest
from core.binary_source.buffer_source import BufferSource
from core.binary_source.file_source import FileSource
from core.binary_source.binary_source_factory import BinarySourceFactory, StorageMode


def test_create_binary_source_memory():
    """Test that StorageMode.MEMORY returns a BufferSource instance"""
    source = BinarySourceFactory.create_binary_source(StorageMode.MEMORY)
    assert isinstance(source, BufferSource)


def test_create_binary_source_disk():
    """Test that StorageMode.DISK returns a FileSource instance"""
    source = BinarySourceFactory.create_binary_source(StorageMode.DISK)
    assert isinstance(source, FileSource)


def test_create_binary_source_unsupported():
    """Test that an unsupported storage mode raises a ValueError"""
    with pytest.raises(ValueError, match="Unsupported storage mode: unsupported_mode"):
        BinarySourceFactory.create_binary_source("unsupported_mode")
