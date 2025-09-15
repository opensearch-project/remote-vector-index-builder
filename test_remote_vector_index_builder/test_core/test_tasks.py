# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
import pytest
from core.tasks import (
    TaskResult,
    build_index,
    create_vectors_dataset,
    run_tasks,
    upload_index,
)
from core.common.exceptions import BlobError
from core.common.models.vectors_dataset import VectorsDataset
from core.object_store.object_store import ObjectStore
from core.common.models.index_build_parameters import DataType


@pytest.fixture
def mock_object_store():
    return Mock(spec=ObjectStore)


@pytest.fixture
def mock_object_store_factory():
    with patch("core.tasks.ObjectStoreFactory.create_object_store") as mock:
        yield mock


@pytest.fixture
def mock_vectors_dataset_parse():
    with patch("core.tasks.VectorsDataset.parse") as mock:
        yield mock


@pytest.fixture
def mock_vectors_dataset():
    return Mock(
        spec=VectorsDataset,
        vectors=np.array([]),
        doc_ids=np.array([]),
        dtype=DataType.FLOAT,
    )


def test_successful_creation(
    mock_object_store_factory,
    mock_vectors_dataset_parse,
    mock_object_store,
    index_build_parameters,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    mock_vectors_dataset_parse.return_value = Mock(spec=VectorsDataset)

    vectors = BytesIO()
    doc_ids = BytesIO()
    # Execute
    run_tasks(index_build_parameters, object_store_config)

    vectors.close()
    doc_ids.close()

    # Verify only 1 object store instance is created for the entire task execution
    mock_object_store_factory.assert_called_once_with(
        index_build_parameters, object_store_config
    )
    assert mock_object_store.read_blob.call_count == 2
    mock_vectors_dataset_parse.assert_called_once()


def test_download_blob_error_handling(
    mock_object_store_factory,
    mock_object_store,
    index_build_parameters,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    mock_object_store.read_blob.side_effect = BlobError("Failed to read blob")

    vectors = BytesIO()
    doc_ids = BytesIO()

    # Execute and verify
    with pytest.raises(BlobError):
        create_vectors_dataset(
            index_build_parameters, mock_object_store, vectors, doc_ids
        )

    vectors.close()
    doc_ids.close()


def test_successful_build(index_build_parameters, mock_vectors_dataset):

    local_path = "/tmp/index"
    # Execute
    build_index(index_build_parameters, mock_vectors_dataset, local_path)


def test_successful_upload(
    mock_object_store_factory,
    mock_object_store,
    index_build_parameters,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    local_path = "/tmp/index"

    # Execute
    upload_index(index_build_parameters, mock_object_store, local_path)

    # Verify
    vector_name = index_build_parameters.vector_path.split(".knnvec")[0]
    remote_path = vector_name + "." + index_build_parameters.engine
    mock_object_store.write_blob.assert_called_once_with(local_path, remote_path)


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
        upload_index(index_build_parameters, mock_object_store, local_path)


def test_successful_task_execution(
    index_build_parameters, mock_vectors_dataset, object_store_config
):
    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ) as mock_build_index, patch("core.tasks.upload_index") as mock_upload_index, patch(
        "os.remove"
    ) as mock_os_remove, patch(
        "os.makedirs"
    ) as mock_os_makedirs:

        # Setup mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_upload_index.return_value = "remote/path/to/index.bin"

        # Execute function
        result = run_tasks(index_build_parameters, object_store_config)

        # Verify success
        assert isinstance(result, TaskResult)
        assert result.file_name == "index.bin"
        assert result.error is None

        # Verify mock calls
        mock_create_dataset.assert_called_once()
        mock_build_index.assert_called_once()
        mock_upload_index.assert_called_once()
        assert mock_vectors_dataset.free_vectors_space.call_count == 2
        mock_os_remove.assert_called_once()
        mock_os_makedirs.assert_called_once()


def test_successful_task_execution_with_object_store_config(
    index_build_parameters,
    mock_vectors_dataset,
    object_store_config,
    mock_object_store,
    mock_object_store_factory,
):
    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ) as mock_build_index, patch("core.tasks.upload_index") as mock_upload_index, patch(
        "os.remove"
    ) as mock_os_remove, patch(
        "os.makedirs"
    ) as mock_os_makedirs:

        # Setup mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_upload_index.return_value = "remote/path/to/index.bin"

        # Execute function
        result = run_tasks(index_build_parameters, object_store_config)

        # Verify success
        assert isinstance(result, TaskResult)
        assert result.file_name == "index.bin"
        assert result.error is None

        # Verify mock calls
        mock_create_dataset.assert_called_once()
        call_args = mock_create_dataset.call_args[1]
        assert "object_store" in call_args

        mock_build_index.assert_called_once()
        mock_upload_index.assert_called_once()
        assert mock_vectors_dataset.free_vectors_space.call_count == 2
        mock_os_remove.assert_called_once()
        mock_os_makedirs.assert_called_once()


def test_create_vectors_dataset_failure(index_build_parameters, object_store_config):

    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset:
        mock_create_dataset.side_effect = Exception("Dataset creation failed")

        result = run_tasks(index_build_parameters, object_store_config)

        assert isinstance(result, TaskResult)
        assert result.file_name is None
        assert result.error == "Dataset creation failed"


def test_build_index_failure(
    index_build_parameters, mock_vectors_dataset, object_store_config
):
    with patch("core.tasks.create_vectors_dataset") as mock_create_dataset, patch(
        "core.tasks.build_index"
    ) as mock_build_index, patch("os.makedirs") as mock_os_makedirs:

        # Setup mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_build_index.side_effect = Exception("Index building failed")

        # Execute function
        result = run_tasks(index_build_parameters, object_store_config)

        # Verify success
        assert isinstance(result, TaskResult)
        assert result.file_name is None
        assert result.error == "Index building failed"

        # Verify mock calls
        mock_create_dataset.assert_called_once()
        mock_build_index.assert_called_once()
        mock_vectors_dataset.free_vectors_space.assert_called_once()
        mock_os_makedirs.assert_called_once()
