# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from unittest.mock import Mock, patch
from core.binary_source.file_source import FileSource
from core.object_store.object_store import ObjectStore


def test_init():
    """Test initialization sets the right defaults"""
    file_source = FileSource()
    assert file_source._temp_dir is None
    assert file_source._source is None


def test_context_manager():
    """Test context manager functionality"""
    # Mock the internal file and temp dir
    with patch("core.binary_source.file_source.TemporaryDirectory") as mock_temp_dir:
        mock_file = Mock()

        # Create and enter context with file_source
        with FileSource() as file_source:
            file_source._source = mock_file
            file_source._temp_dir = mock_temp_dir.return_value
            assert isinstance(file_source, FileSource)

        # Verify cleanup was done
        mock_file.close.assert_called_once()
        mock_temp_dir.return_value.cleanup.assert_called_once()


def test_context_manager_with_none():
    """Test context manager cleanup when source or temp_dir are None"""
    # This test ensures the exit method handles None values safely
    with FileSource():
        # Intentionally leave _source and _temp_dir as None
        pass

    # No assertions needed; test passes if no exceptions are raised


def test_read_from_object_store():
    """Test reading data from an object store"""
    mock_object_store = Mock(spec=ObjectStore)
    file_source = FileSource()

    # Use a patch to avoid actual file operations
    with patch("builtins.open") as mock_open, patch(
        "core.binary_source.file_source.TemporaryDirectory"
    ) as mock_temp_dir:

        mock_temp_dir.return_value.name = "/tmp/fake_dir"
        mock_file = Mock()
        mock_open.return_value = mock_file

        # Call the method
        file_source.read_from_object_store(mock_object_store, "test/path")

        # Verify temp dir was created
        mock_temp_dir.assert_called_once()
        # Verify file was opened with correct path
        mock_open.assert_called_with("/tmp/fake_dir/test_path", "wb+")
        # Verify object store read_blob was called
        mock_object_store.read_blob.assert_called_once_with("test/path", mock_file)


def test_read_from_object_store_path_sanitization():
    """Test path sanitization in read_from_object_store"""
    mock_object_store = Mock(spec=ObjectStore)
    file_source = FileSource()

    with patch("builtins.open") as mock_open, patch(
        "core.binary_source.file_source.TemporaryDirectory"
    ) as mock_temp_dir:

        mock_temp_dir.return_value.name = "/tmp/fake_dir"

        # Test with path containing both forward and backward slashes
        file_source.read_from_object_store(
            mock_object_store, "test/path\\with/mixed\\slashes"
        )

        # Verify file was opened with sanitized path
        mock_open.assert_called_with(
            "/tmp/fake_dir/test_path_with_mixed_slashes", "wb+"
        )


def test_parse(sample_doc_ids):
    """Test parsing file content into a NumPy array"""
    file_source = FileSource()

    with patch("numpy.memmap") as mock_memmap:
        mock_memmap.return_value = sample_doc_ids
        mock_file = Mock()
        file_source._source = mock_file

        # Parse the data
        result = file_source.parse(dtype="int32")

        # Verify numpy.memmap was called with correct params
        mock_memmap.assert_called_once_with(mock_file, dtype="int32", mode="r+")

        # Verify result
        assert result is sample_doc_ids
