# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from core.common.exceptions import BlobError
from core.common.models.index_build_parameters import IndexBuildParameters
from core.common.models.vectors_dataset import VectorsDataset
from core.object_store.object_store import ObjectStore
from core.tasks import create_vectors_dataset, upload_index


@pytest.fixture
def mock_object_store():
    return Mock(spec=ObjectStore)


@pytest.fixture
def mock_object_store_factory():
    with patch("core.tasks.ObjectStoreFactory.create_object_store") as mock:
        yield mock


@pytest.fixture
def mock_vectors_dataset():
    return Mock(spec=VectorsDataset)


@pytest.fixture
def mock_vectors_dataset_parse():
    with patch("core.tasks.VectorsDataset.parse") as mock:
        yield mock


@pytest.fixture
def index_build_params():
    return IndexBuildParameters(
        vector_path="vec.knnvec",
        doc_id_path="doc.knndid",
        dimension=128,
        doc_count=1000,
        data_type="fp32",
        repository_type="s3",
        container_name="test-bucket",
    )


@pytest.fixture
def object_store_config():
    return {"region": "us-west-2", "retries": 3}


def test_successful_creation(
    mock_object_store_factory,
    mock_vectors_dataset_parse,
    mock_object_store,
    index_build_params,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    mock_vectors_dataset_parse.return_value = Mock(spec=VectorsDataset)

    vectors = BytesIO()
    doc_ids = BytesIO()
    # Execute
    result = create_vectors_dataset(
        index_build_params, object_store_config, vectors, doc_ids
    )

    vectors.close()
    doc_ids.close()

    # Verify
    mock_object_store_factory.assert_called_once_with(
        index_build_params, object_store_config
    )
    assert mock_object_store.read_blob.call_count == 2
    mock_vectors_dataset_parse.assert_called_once()
    assert isinstance(result, VectorsDataset)

    result.free_vectors_space()


def test_download_blob_error_handling(
    mock_object_store_factory,
    mock_object_store,
    index_build_params,
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
            index_build_params, object_store_config, vectors, doc_ids
        )

    vectors.close()
    doc_ids.close()


def test_successful_upload(
    mock_object_store_factory,
    mock_object_store,
    index_build_params,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    local_path = "/tmp/index"

    # Execute
    upload_index(index_build_params, object_store_config, local_path)

    # Verify
    mock_object_store_factory.assert_called_once_with(
        index_build_params, object_store_config
    )
    mock_object_store.write_blob.assert_called_once_with(
        local_path, index_build_params.vector_path + local_path
    )


def test_upload_blob_error_handling(
    mock_object_store_factory,
    mock_object_store,
    index_build_params,
    object_store_config,
):
    # Setup
    mock_object_store_factory.return_value = mock_object_store
    mock_object_store.write_blob.side_effect = BlobError("Failed to upload")
    local_path = "/tmp/index"

    # Execute and verify
    with pytest.raises(BlobError):
        upload_index(index_build_params, object_store_config, local_path)
