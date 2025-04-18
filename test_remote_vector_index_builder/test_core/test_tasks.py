# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from core.tasks import (
    TaskResult,
    build_index,
    create_vectors_dataset,
    run_tasks,
    upload_index,
    _get_index_path_from_vector_path,
    _get_numpy_dtype,
    _check_dimensions,
)
from core.common.exceptions import BlobError
from core.common.models.vectors_dataset import VectorsDataset
from core.object_store.object_store import ObjectStore
from core.binary_source.binary_source import BinarySource
from core.common.models.index_build_parameters import DataType
from core.binary_source.binary_source_factory import StorageMode


@pytest.fixture
def mock_object_store():
    return Mock(spec=ObjectStore)


@pytest.fixture
def mock_object_store_factory():
    with patch("core.tasks.ObjectStoreFactory.create_object_store") as mock:
        yield mock


@pytest.fixture
def mock_binary_source():
    binary_source = Mock(spec=BinarySource)
    binary_source.transform_to_numpy_array = MagicMock()
    return binary_source


@pytest.fixture
def mock_vectors_dataset(sample_vectors, sample_doc_ids):
    return Mock(spec=VectorsDataset, vectors=sample_vectors, doc_ids=sample_doc_ids)


def test_create_vectors_dataset(
    mock_object_store_factory,
    mock_object_store,
    mock_binary_source,
    index_build_parameters,
    object_store_config,
    sample_doc_ids,
    sample_vectors,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store

    # Configure mocks for transform_to_numpy_array calls
    mock_binary_source.transform_to_numpy_array.side_effect = [
        sample_doc_ids,  # First call for doc_ids
        sample_vectors.flatten(),  # Second call for vectors (flattened)
    ]

    # Execute
    result = create_vectors_dataset(
        index_build_parameters,
        object_store_config,
        mock_binary_source,
        mock_binary_source,
    )

    # Verify
    mock_object_store_factory.assert_called_once_with(
        index_build_parameters, object_store_config
    )
    # Check correct paths are used from index_build_parameters
    mock_binary_source.transform_to_numpy_array.assert_any_call(
        mock_object_store,
        index_build_parameters.doc_id_path,
        "<i4",
        index_build_parameters.doc_count,
    )
    mock_binary_source.transform_to_numpy_array.assert_any_call(
        mock_object_store,
        index_build_parameters.vector_path,
        "<f4",
        index_build_parameters.doc_count * index_build_parameters.dimension,
    )
    assert isinstance(result, VectorsDataset)
    assert result.doc_ids is sample_doc_ids
    np.testing.assert_array_equal(result.vectors, sample_vectors)


def test_create_vectors_dataset_dimension_error(
    mock_object_store_factory,
    mock_object_store,
    mock_binary_source,
    index_build_parameters,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    wrong_size_doc_ids = np.array([1, 2, 3], dtype=np.int32)  # Wrong size

    mock_binary_source.transform_to_numpy_array.return_value = wrong_size_doc_ids

    # Execute and verify - should match the doc_count from index_build_parameters
    with pytest.raises(
        ValueError,
        match=f"Expected {index_build_parameters.doc_count} elements, but got 3",
    ):
        create_vectors_dataset(
            index_build_parameters,
            object_store_config,
            mock_binary_source,
            mock_binary_source,
        )


def test_build_index(index_build_parameters, mock_vectors_dataset):
    with patch("core.tasks.FaissIndexBuildService.build_index") as mock_build:
        local_path = "/tmp/index"
        # Execute
        build_index(index_build_parameters, mock_vectors_dataset, local_path)

        # Verify
        mock_build.assert_called_once_with(
            index_build_parameters, mock_vectors_dataset, local_path
        )


def test_upload_index(
    mock_object_store_factory,
    mock_object_store,
    index_build_parameters,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    local_path = "/tmp/index"

    # Execute
    result = upload_index(index_build_parameters, object_store_config, local_path)

    # Verify
    mock_object_store_factory.assert_called_once_with(
        index_build_parameters, object_store_config
    )

    # Use the vector_path from index_build_parameters
    expected_remote_path = "vec.faiss"  # Derived from vec.knnvec in the fixture
    mock_object_store.write_blob.assert_called_once_with(
        local_path, expected_remote_path
    )
    assert result == expected_remote_path


def test_upload_blob_error_handling(
    mock_object_store_factory,
    mock_object_store,
    index_build_parameters,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    mock_object_store.write_blob.side_effect = BlobError("Failed to upload")
    local_path = "/tmp/index"

    # Execute and verify
    with pytest.raises(BlobError):
        upload_index(index_build_parameters, object_store_config, local_path)


def test_run_tasks_successful(index_build_parameters, mock_vectors_dataset):
    # Create proper context manager mocks
    mock_binary_source = MagicMock(spec=BinarySource)

    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ) as mock_build_index, patch("core.tasks.upload_index") as mock_upload_index, patch(
        "os.remove"
    ) as mock_os_remove, patch(
        "os.makedirs"
    ) as mock_os_makedirs, patch(
        "core.binary_source.binary_source_factory.BinarySourceFactory.create_binary_source"
    ) as mock_factory:

        # Set up the factory to return our mock
        mock_factory.return_value = mock_binary_source

        # Setup additional mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_upload_index.return_value = "vec.faiss"

        # Execute function
        result = run_tasks(index_build_parameters)

        # Verify success
        assert isinstance(result, TaskResult)
        assert result.file_name == "vec.faiss"
        assert result.error is None

        # Verify mock calls
        mock_create_dataset.assert_called_once()
        mock_build_index.assert_called_once()
        mock_upload_index.assert_called_once()
        assert mock_vectors_dataset.free_vectors_space.call_count == 2
        mock_os_remove.assert_called_once()
        mock_os_makedirs.assert_called_once()
        assert mock_factory.call_count == 2


def test_run_tasks_with_object_store_config(
    index_build_parameters, mock_vectors_dataset, object_store_config
):
    # Create proper context manager mocks
    mock_binary_source = MagicMock(spec=BinarySource)

    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ), patch("core.tasks.upload_index") as mock_upload_index, patch("os.remove"), patch(
        "os.makedirs"
    ), patch(
        "core.binary_source.binary_source_factory.BinarySourceFactory.create_binary_source"
    ) as mock_factory:

        # Set up the factory to return our mock
        mock_factory.return_value = mock_binary_source

        # Setup additional mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_upload_index.return_value = "vec.faiss"

        # Execute function with object_store_config
        result = run_tasks(index_build_parameters, object_store_config)

        # Verify success
        assert isinstance(result, TaskResult)
        assert result.file_name == "vec.faiss"
        assert result.error is None

        mock_create_dataset.assert_called_once()
        assert mock_create_dataset.call_args[0][0] == index_build_parameters
        assert mock_create_dataset.call_args[0][1] == object_store_config


def test_run_tasks_with_memory_storage_mode(
    index_build_parameters, mock_vectors_dataset
):
    # Create proper context manager mocks
    mock_binary_source = MagicMock(spec=BinarySource)

    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ), patch("core.tasks.upload_index") as mock_upload_index, patch("os.remove"), patch(
        "os.makedirs"
    ), patch(
        "core.binary_source.binary_source_factory.BinarySourceFactory.create_binary_source"
    ) as mock_factory:

        # Set up the factory to return our mock
        mock_factory.return_value = mock_binary_source

        # Setup additional mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_upload_index.return_value = "vec.faiss"

        # Execute function with memory storage mode
        run_tasks(index_build_parameters, storage_mode=StorageMode.MEMORY)

        # Verify correct storage mode was passed
        mock_factory.assert_called_with(StorageMode.MEMORY)


def test_run_tasks_create_vectors_dataset_failure(index_build_parameters):
    # Create proper context manager mocks
    mock_binary_source = MagicMock(spec=BinarySource)

    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.binary_source.binary_source_factory.BinarySourceFactory.create_binary_source"
    ) as mock_factory:

        # Set up the factory to return our mock
        mock_factory.return_value = mock_binary_source

        # Setup failure case
        mock_create_dataset.side_effect = Exception("Dataset creation failed")

        # Execute function
        result = run_tasks(index_build_parameters)

        # Verify failure
        assert isinstance(result, TaskResult)
        assert result.file_name is None
        assert result.error == "Dataset creation failed"


def test_run_tasks_build_index_failure(index_build_parameters, mock_vectors_dataset):
    # Create proper context manager mocks
    mock_binary_source = MagicMock(spec=BinarySource)

    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ) as mock_build_index, patch("os.makedirs"), patch(
        "core.binary_source.binary_source_factory.BinarySourceFactory.create_binary_source"
    ) as mock_factory:

        # Set up the factory to return our mock
        mock_factory.return_value = mock_binary_source

        # Setup mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_build_index.side_effect = Exception("Index building failed")

        # Execute function
        result = run_tasks(index_build_parameters)

        # Verify failure
        assert isinstance(result, TaskResult)
        assert result.file_name is None
        assert result.error == "Index building failed"
        mock_vectors_dataset.free_vectors_space.assert_called_once()


def test_run_tasks_upload_index_failure(index_build_parameters, mock_vectors_dataset):
    # Create proper context manager mocks
    mock_binary_source = MagicMock(spec=BinarySource)

    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ), patch("core.tasks.upload_index") as mock_upload_index, patch("os.remove"), patch(
        "os.makedirs"
    ), patch(
        "core.binary_source.binary_source_factory.BinarySourceFactory.create_binary_source"
    ) as mock_factory:

        # Set up the factory to return our mock
        mock_factory.return_value = mock_binary_source

        # Setup mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_upload_index.side_effect = Exception("Upload failed")

        # Execute function
        result = run_tasks(index_build_parameters)

        # Verify failure
        assert isinstance(result, TaskResult)
        assert result.file_name is None
        assert result.error == "Upload failed"
        assert mock_vectors_dataset.free_vectors_space.call_count == 2


def test_get_index_path_from_vector_path():
    # Use the exact vector path format from the fixture
    vector_path = "vec.knnvec"
    engine = "faiss"
    result = _get_index_path_from_vector_path(vector_path, engine)
    assert result == "vec.faiss"

    # Additional test with dots in filename
    vector_path = "path/to/vectors.v1.knnvec"
    result = _get_index_path_from_vector_path(vector_path, engine)
    assert result == "path/to/vectors.v1.faiss"


def test_get_numpy_dtype():
    # Test valid type
    assert _get_numpy_dtype(DataType.FLOAT) == "<f4"

    # Test invalid type
    with pytest.raises(ValueError, match="Unsupported data type:"):
        _get_numpy_dtype("INVALID_TYPE")


def test_check_dimensions(sample_doc_ids):
    # Test with the exact doc_count from index_build_parameters fixture
    _check_dimensions(sample_doc_ids, 5)  # Should not raise

    # Test invalid dimensions
    with pytest.raises(ValueError, match="Expected 6 elements, but got 5"):
        _check_dimensions(sample_doc_ids, 6)
