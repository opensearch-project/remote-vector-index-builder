# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from io import BytesIO
from unittest.mock import Mock, patch
import tempfile
import os

import numpy as np
import pytest
from core.tasks import (
    TaskResult,
    _convert_half_float_blob_if_fp32,
    _determine_streaming_buffer,
    build_index,
    create_vectors_dataset,
    run_tasks,
    upload_index,
    index_storage_context,
)

# Imported via the same package path core.tasks uses, so isinstance / pytest.raises see the same class objects
from remote_vector_index_builder.core.common.exceptions import VectorsDatasetError
from remote_vector_index_builder.core.fp32_to_fp16_converting_bytes_io import (
    FP32ToFP16ConvertingBytesIO,
)
from core.common.exceptions import BlobError
from core.common.models.vectors_dataset import VectorsDataset
from core.object_store.object_store import ObjectStore
from core.common.models.index_build_parameters import DataType
from core.common.models.index_build_parameters import IndexSerializationMode


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


def _half_float_params(doc_count=5, dimension=4):
    return Mock(
        data_type=DataType.FLOAT16,
        doc_count=doc_count,
        dimension=dimension,
        vector_path="test/vectors.knnvec",
        doc_id_path="test/vectors.knndid",
    )


def test_streaming_buffer_half_float_fp16_blob_is_read_directly():
    params = _half_float_params()
    vectors = BytesIO()
    fp16_blob_size = params.doc_count * params.dimension * DataType.FLOAT16.get_size()

    assert _determine_streaming_buffer(params, vectors, fp16_blob_size) is vectors


def test_streaming_buffer_half_float_fp32_blob_is_converted():
    params = _half_float_params()
    vectors = BytesIO()
    fp32_blob_size = params.doc_count * params.dimension * DataType.FLOAT.get_size()

    buffer = _determine_streaming_buffer(params, vectors, fp32_blob_size)

    assert isinstance(buffer, FP32ToFP16ConvertingBytesIO)
    assert buffer is not vectors


def test_streaming_buffer_half_float_unexpected_blob_size_raises():
    params = _half_float_params()
    vectors = BytesIO()
    fp16_size = params.doc_count * params.dimension * DataType.FLOAT16.get_size()

    with pytest.raises(VectorsDatasetError, match="matches neither fp16"):
        _determine_streaming_buffer(params, vectors, fp16_size - 1)


def test_streaming_buffer_half_float_unknown_size_defers_to_download():
    params = _half_float_params()
    vectors = BytesIO()
    assert _determine_streaming_buffer(params, vectors, None) is vectors


def _fp16_and_fp32_blobs(params):
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((params.doc_count, params.dimension)).astype(np.float32)
    return truth.astype(np.float16), truth.astype(np.float16).tobytes(), truth.tobytes()


def test_convert_half_float_blob_when_fp32_converts_in_place():
    params = _half_float_params()
    expected, _, fp32_blob = _fp16_and_fp32_blobs(params)
    vectors = BytesIO(fp32_blob)

    _convert_half_float_blob_if_fp32(params, vectors)

    got = np.frombuffer(vectors.getvalue(), dtype="<f2").reshape(
        params.doc_count, params.dimension
    )
    assert np.array_equal(got, expected)


def test_convert_half_float_blob_when_fp16_is_untouched():
    params = _half_float_params()
    _, fp16_blob, _ = _fp16_and_fp32_blobs(params)
    vectors = BytesIO(fp16_blob)

    _convert_half_float_blob_if_fp32(params, vectors)

    assert vectors.getvalue() == fp16_blob


def test_convert_half_float_blob_unexpected_size_raises():
    params = _half_float_params()
    _, fp16_blob, _ = _fp16_and_fp32_blobs(params)
    vectors = BytesIO(fp16_blob[:-1])

    with pytest.raises(VectorsDatasetError, match="matches neither fp16"):
        _convert_half_float_blob_if_fp32(params, vectors)


def test_create_vectors_dataset_half_float_unknown_size_fp32_blob_still_builds(
    mock_object_store,
):
    # HEAD not permitted: size lookup returns None, blob is fp32 -> converted after download
    params = _half_float_params()
    expected, _, fp32_blob = _fp16_and_fp32_blobs(params)
    doc_ids = np.arange(params.doc_count, dtype=np.int32)
    blobs = {params.vector_path: fp32_blob, params.doc_id_path: doc_ids.tobytes()}
    mock_object_store.get_blob_size.return_value = None
    mock_object_store.read_blob.side_effect = lambda path, buf: buf.write(blobs[path])

    dataset = create_vectors_dataset(params, mock_object_store, BytesIO(), BytesIO())

    assert dataset.dtype == DataType.FLOAT16
    assert np.array_equal(dataset.vectors, expected)
    assert np.array_equal(dataset.doc_ids, doc_ids)


def test_streaming_buffer_non_half_float_is_unchanged(index_build_parameters):
    vectors = BytesIO()
    assert index_build_parameters.data_type != DataType.FLOAT16
    assert (
        _determine_streaming_buffer(index_build_parameters, vectors, 12345) is vectors
    )


def test_create_vectors_dataset_uses_blob_size_for_half_float(
    mock_object_store, mock_vectors_dataset_parse
):
    params = _half_float_params()
    mock_object_store.get_blob_size.return_value = (
        params.doc_count * params.dimension * DataType.FLOAT16.get_size()
    )
    mock_vectors_dataset_parse.return_value = Mock(spec=VectorsDataset)
    vectors = BytesIO()
    doc_ids = BytesIO()

    create_vectors_dataset(params, mock_object_store, vectors, doc_ids)

    mock_object_store.get_blob_size.assert_called_once_with(params.vector_path)
    # fp16 blob: downloaded straight into the caller's buffer, no converting wrapper
    vector_read_call = mock_object_store.read_blob.call_args_list[0]
    assert vector_read_call.args == (params.vector_path, vectors)

    vectors.close()
    doc_ids.close()


def test_successful_object_store_creation(
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


def test_successful_build(index_build_parameters, mock_vectors_dataset):

    local_path = "/tmp/index"

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
    with (
        patch("core.tasks.create_vectors_dataset") as mock_create_dataset,
        patch("core.tasks.upload_index") as mock_upload_index,
        patch("os.makedirs") as mock_os_makedirs,
    ):

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
        mock_upload_index.assert_called_once()
        assert mock_vectors_dataset.free_vectors_space.call_count == 2
        mock_os_makedirs.assert_called_once()


def test_successful_task_execution_with_memory_storage_mode(
    index_build_parameters,
    mock_vectors_dataset,
    object_store_config,
):
    with (
        patch("core.tasks.create_vectors_dataset") as mock_create_dataset,
        patch("core.tasks.upload_index") as mock_upload_index,
    ):

        # Setup mocks
        mock_create_dataset.return_value = mock_vectors_dataset
        mock_upload_index.return_value = "remote/path/to/index.bin"

        # Execute function
        result = run_tasks(
            index_build_parameters, object_store_config, IndexSerializationMode.MEMORY
        )

        # Verify success
        assert isinstance(result, TaskResult)
        assert result.file_name == "index.bin"
        assert result.error is None

        # Verify mock calls
        mock_create_dataset.assert_called_once()
        call_args = mock_create_dataset.call_args[1]
        assert "object_store" in call_args

        mock_upload_index.assert_called_once()
        assert mock_vectors_dataset.free_vectors_space.call_count == 2


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
    with (
        patch("core.tasks.create_vectors_dataset") as mock_create_dataset,
        patch("core.tasks.FaissIndexBuildService.build_index") as mock_build_index,
        patch("os.makedirs") as mock_os_makedirs,
    ):

        mock_create_dataset.return_value = mock_vectors_dataset
        mock_build_index.side_effect = Exception("Index building failed")

        result = run_tasks(index_build_parameters, object_store_config)

        assert isinstance(result, TaskResult)
        assert result.file_name is None
        assert result.error == "Index building failed"

        mock_create_dataset.assert_called_once()
        mock_build_index.assert_called_once()
        mock_vectors_dataset.free_vectors_space.assert_called_once()
        mock_os_makedirs.assert_called_once()


def test_memory_mode():
    with tempfile.TemporaryDirectory() as temp_dir:
        with index_storage_context(
            IndexSerializationMode.MEMORY, temp_dir, "test.knnvec"
        ) as storage:
            assert isinstance(storage, BytesIO)
            # simulate writing to buffer
            storage.write(b"test")
            assert storage.tell() > 0
        # check that storage is closed
        assert storage.closed


def test_disk_mode():
    with tempfile.TemporaryDirectory() as temp_dir:
        with index_storage_context(
            IndexSerializationMode.DISK, temp_dir, "sub/test.knnvec"
        ) as storage:
            assert isinstance(storage, str)
            # simulate writing to file
            with open(storage, "wb") as f:
                f.write(b"test")
            assert os.path.getsize(storage) > 0
        # check that file got cleaned up
        assert not os.path.exists(storage)
