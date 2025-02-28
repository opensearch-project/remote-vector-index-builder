# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError
from core.common.exceptions import BlobError
from core.common.models.index_build_parameters import IndexBuildParameters
from core.object_store.s3.s3_object_store import S3ObjectStore, get_boto3_client


# Mock the logger to prevent actual logging during tests
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("core.object_store.s3.s3_object_store.logger"):
        yield


@pytest.fixture
def index_build_params():
    return IndexBuildParameters(
        container_name="test-bucket",
        vector_path="test-vector-path.knnvec",
        doc_id_path="test-doc-id-path",
        dimension=100,
        doc_count=10,
    )


@pytest.fixture
def object_store_config():
    return {
        "retries": 3,
        "region": "us-west-2",
        "debug": False,
        "transfer_config": {
            "multipart_chunksize": 10 * 1024 * 1024,
            "max_concurrency": 4,
        },
    }


@pytest.fixture
def s3_object_store(index_build_params, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        yield store


@pytest.fixture
def bytes_buffer():
    bytes_buffer = BytesIO()
    yield bytes_buffer
    bytes_buffer.close()


def test_get_boto3_client():
    with patch("boto3.client") as mock_client:
        # Test caching behavior
        client1 = get_boto3_client("us-west-2", 3)
        client2 = get_boto3_client("us-west-2", 3)
        assert client1 == client2
        mock_client.assert_called_once()

        # Test different parameters create new client
        get_boto3_client("us-east-1", 3)
        assert mock_client.call_count == 2


def test_s3_object_store_initialization(index_build_params, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        assert store.bucket == "test-bucket"
        assert store.max_retries == 3
        assert store.region == "us-west-2"
        assert store.transfer_config.multipart_chunksize == 10 * 1024 * 1024
        assert store.transfer_config.max_concurrency == 4
        assert not store.debug


def test_s3_object_store_initialization_defaults(index_build_params):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, {})
        assert store.max_retries == 3
        assert store.region == "us-west-2"
        assert not store.debug


# also test if os.cpu_count is none
def test_s3_object_store_initialization_debug_config(index_build_params):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        with patch("os.cpu_count", return_value=None):
            store = S3ObjectStore(index_build_params, {"debug": True})
            assert store.max_retries == 3
            assert store.region == "us-west-2"
            assert store.debug


def test_create_transfer_config(s3_object_store):
    custom_config = {
        "multipart_chunksize": 20 * 1024 * 1024,
        "max_concurrency": 8,
        "invalid_param": "value",  # Should be ignored
    }

    config = s3_object_store._create_transfer_config(custom_config)
    assert config.multipart_chunksize == 20 * 1024 * 1024
    assert config.max_concurrency == 8
    assert "invalid_param" not in config.__dict__


def test_read_blob_success(index_build_params, object_store_config, bytes_buffer):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        store.s3_client.download_fileobj = Mock()

        store.read_blob("test/path", bytes_buffer)
        store.s3_client.download_fileobj.assert_called_once_with(
            store.bucket,
            "test/path",
            bytes_buffer,
            Config=store.transfer_config,
            Callback=None,
        )


def test_read_blob_with_debug(index_build_params, object_store_config, bytes_buffer):
    object_store_config["debug"] = True
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        store.s3_client.download_fileobj = Mock()

        store.read_blob("test/path", bytes_buffer)

        # Verify callback was passed
        callback = store.s3_client.download_fileobj.call_args.kwargs["Callback"]
        assert callback is not None
        # Test the callback directly
        assert store._read_progress == 0
        callback(100)  # Simulate 100 bytes transferred
        assert store._read_progress == 100
        callback(50)  # Simulate 50 more bytes
        assert store._read_progress == 150


def test_read_blob_failure(index_build_params, object_store_config, bytes_buffer):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        error = ClientError(
            {"Error": {"Code": "LimitExceededException", "Message": "Limit Exceeded"}},
            "DownloadFileObj",
        )
        store.s3_client.download_fileobj.side_effect = error
        with pytest.raises(BlobError):
            store.read_blob("test/path", bytes_buffer)


def test_write_blob_success(index_build_params, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        store.s3_client.upload_file = Mock()
        store.write_blob("local/path", "remote/path")

        store.s3_client.upload_file.assert_called_once_with(
            "local/path",
            store.bucket,
            "remote/path",
            Config=store.transfer_config,
            Callback=None,
        )


def test_write_blob_with_debug(index_build_params, object_store_config):
    object_store_config["debug"] = True
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        store.s3_client.upload_file = Mock()

        store.write_blob("local/path", "remote/path")

        # Verify callback was passed
        callback = store.s3_client.upload_file.call_args.kwargs["Callback"]
        assert callback is not None
        # Test the callback directly
        assert store._write_progress == 0
        callback(100)  # Simulate 100 bytes transferred
        assert store._write_progress == 100
        callback(50)  # Simulate 50 more bytes
        assert store._write_progress == 150


def test_write_blob_failure(index_build_params, object_store_config):
    with patch("core.object_store.s3.s3_object_store.get_boto3_client"):
        store = S3ObjectStore(index_build_params, object_store_config)
        error = ClientError(
            {"Error": {"Code": "LimitExceededException", "Message": "Limit Exceeded"}},
            "UploadFile",
        )
        store.s3_client.upload_file.side_effect = error
        with pytest.raises(BlobError):
            store.write_blob("local/path", "remote/path")
