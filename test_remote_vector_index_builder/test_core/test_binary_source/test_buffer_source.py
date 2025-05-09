# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import numpy as np
from unittest.mock import Mock
from io import BytesIO
from core.binary_source.buffer_source import BufferSource
from core.object_store.object_store import ObjectStore


def test_init():
    """Test that initialization creates a BytesIO object"""
    buffer = BufferSource()
    assert isinstance(buffer._source, BytesIO)


def test_context_manager():
    """Test context manager functionality (enter/exit)"""
    # Use a context manager
    with BufferSource() as buffer:
        assert isinstance(buffer, BufferSource)
        # Replace the _source with a mock to check if close is called
        mock_source = Mock(wraps=buffer._source)
        buffer._source = mock_source

    # Verify close was called during exit
    mock_source.close.assert_called_once()


def test_read_from_object_store():
    """Test reading data from an object store"""
    mock_object_store = Mock(spec=ObjectStore)
    buffer = BufferSource()

    buffer.read_from_object_store(mock_object_store, "test/path")

    # Verify object store read_blob was called with correct arguments
    mock_object_store.read_blob.assert_called_once_with("test/path", buffer._source)


def test_parse(sample_doc_ids):
    """Test parsing buffer content into a NumPy array using sample_doc_ids fixture"""
    buffer = BufferSource()

    # Use fixture data
    buffer._source = BytesIO(sample_doc_ids.tobytes())

    # Parse the data
    result = buffer.parse(dtype="int32")

    # Verify result
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype("int32")
    np.testing.assert_array_equal(result, sample_doc_ids)
